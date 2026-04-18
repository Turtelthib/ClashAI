# clashai/combat/environment_v4.py
# Environnement V4 pour ClashAI.
#
# Hérite de ClashEnvV3 pour réutiliser toute la navigation ADB
# et override les méthodes clés pour le nouvel action space.
#
# Changements vs V3 :
#   - 37 actions (rôle × secteur + sorts auto-ciblés)
#   - Observation compactée (54 dims au lieu de 76)
#   - TroopManager pour le deploy par rôle
#   - Reward shaping centralisé
#   - L'agent choisit librement l'ordre des sorts/abilities

import time
import numpy as np

from clashai.combat.environment import ClashEnvV3
from clashai.combat.action_space import (
    TOTAL_ACTIONS, NUM_ROLES, NUM_SECTORS, NUM_HEROES,
    DEPLOY_ROLES, HERO_NAMES, SPELL_NAMES,
    MAX_STEPS_SAFETY, NUM_POSITIONS,
    decode_action, compute_action_mask,
    build_role_inventory, build_spell_inventory,
)
from clashai.combat.agent_v4 import (
    VECTOR_SIZE,
    ROLE_FEATURES, SPELL_FEATURES, COMBAT_FEATURES_SIZE,
)
from clashai.combat.troop_manager import TroopManager
from clashai.combat.reward_shaping import (
    compute_deploy_reward, compute_combat_reward,
    compute_leftover_penalty, compute_spell_leftover_penalty,
    compute_hero_survival_bonus,
)

# Imports V3 pour accéder aux constantes
from clashai.combat.agent import (
    TROOP_TYPES, TROOP_NAME_TO_IDX,
)

# Delays
DELAY_SWITCH_TROOP = 0.15
DELAY_DEPLOY = 0.08
DELAY_WAIT_SHORT = 0.5
DELAY_WAIT_LONG = 2.0
DELAY_OBSERVE = 2.5
DELAY_ABILITY = 0.3
RESCAN_EVERY_N_STEPS = 8
NO_TROOPS_CHECKS_THRESHOLD = 3


class ClashEnvV4(ClashEnvV3):
    """
    Environnement V4 — action space simplifié (37 actions).

    Hérite de V3 pour la navigation ADB, l'override porte sur :
    - Observation (54 dims)
    - Action mask (37 actions)
    - Exécution d'actions (rôle × secteur)
    - Reward shaping (module séparé)
    """

    def __init__(self, models, verbose=True):
        super().__init__(models, verbose)

        # TroopManager V4 (remplace la sélection directe V3)
        self._troop_mgr = TroopManager(
            troop_finder=self._troop_finder,
            troop_types=TROOP_TYPES,
            troop_name_to_idx=TROOP_NAME_TO_IDX,
            adb_screenshot_fn=self._adb_screenshot,
            adb_tap_fn=self._adb_tap,
            verbose=verbose,
        )

        # V4 state
        self._sector_map = np.zeros(NUM_SECTORS, dtype=np.float32)
        self._last_sector = None
        self._center_pos = NUM_POSITIONS // 2  # mis à jour dans reset
        # Initialisé ici pour que _get_obs() (appelé par super().reset()) ne crash pas
        self._episode_start_time = time.time()
        self._buildings_destroyed_total = 0
        self._prev_building_count = 0
        self._last_rewarded_destroyed = 0
        self._exhaustion_rescanned = False

        if verbose:
            print("\n🎮 ClashEnv V4 initialisé")
            print(f"   Actions     : {TOTAL_ACTIONS} "
                  f"({NUM_ROLES}×{NUM_SECTORS} deploy + "
                  f"{len(SPELL_NAMES)} sorts + {NUM_HEROES} abilities)")
            print(f"   Vector      : {VECTOR_SIZE} dims")
            print("   Phases      : fusionnees (V4.2)")
            print(f"   Safety cap  : {MAX_STEPS_SAFETY} steps")

    # -----------------------------------------------------------------
    #  Observation V4 (54 dims)
    # -----------------------------------------------------------------

    def _get_obs(self):
        """Construit l'observation V4 : grid + vector 54 dims."""
        # Temps écoulé normalisé sur 180s (timer COC 3min) — plus stable que step/MAX
        elapsed = time.time() - self._episode_start_time
        time_norm = np.array([min(elapsed / 180.0, 1.0)], dtype=np.float32)

        combat_feats = (self._combat_features
                        if self._combat_features is not None
                        else np.zeros(COMBAT_FEATURES_SIZE, dtype=np.float32))

        hero_status = self._hero_manager.get_status_vector()

        # V4 : role counts au lieu de troop counts individuels
        role_counts = np.zeros(ROLE_FEATURES, dtype=np.float32)
        for i, role in enumerate(DEPLOY_ROLES):
            for troop in TROOP_TYPES:
                if troop['role'] == role and troop['name'] in TROOP_NAME_TO_IDX:
                    idx = TROOP_NAME_TO_IDX[troop['name']]
                    role_counts[i] += self._remaining_troops[idx]
        role_counts = role_counts / 10.0  # normaliser

        # V4 : spell counts
        spell_counts = np.zeros(SPELL_FEATURES, dtype=np.float32)
        for i, spell_name in enumerate(SPELL_NAMES):
            if spell_name in TROOP_NAME_TO_IDX:
                spell_counts[i] = self._remaining_troops[TROOP_NAME_TO_IDX[spell_name]]
        spell_counts = spell_counts / 3.0  # normaliser

        vector = np.concatenate([
            self._features,           # (20,) village features
            role_counts,              # (5,)  troupes par rôle
            spell_counts,             # (3,)  sorts restants
            self._sector_map,         # (5,)  densité deploy par secteur
            time_norm,                # (1,)  temps écoulé / 180s
            combat_feats,             # (15,)
            hero_status,              # (5,)
        ])

        return self._grid, vector

    # -----------------------------------------------------------------
    #  Action mask V4 (37 actions)
    # -----------------------------------------------------------------

    def _get_mask(self):
        """Masque d'actions V4."""
        hero_mask = self._hero_manager.get_ability_mask()
        return compute_action_mask(
            self._remaining_troops,
            TROOP_TYPES,
            hero_ability_mask=hero_mask,
        )

    # -----------------------------------------------------------------
    #  Execute action V4
    # -----------------------------------------------------------------

    def _execute_action(self, action_idx):
        """Exécute une action V4."""
        action_type, idx1, idx2 = decode_action(action_idx)

        if action_type == 'deploy':
            return self._execute_deploy(idx1, idx2)

        elif action_type == 'spell':
            return self._execute_spell(idx1)

        elif action_type == 'ability':
            return self._execute_ability(idx1)

        elif action_type == 'observe':
            time.sleep(DELAY_OBSERVE)
            self._update_combat_observation()
            return f"👁️ observe ({DELAY_OBSERVE}s)"

        elif action_type == 'wait_short':
            time.sleep(DELAY_WAIT_SHORT)
            self._troop_mgr._last_troop_name = None
            return "attendre 0.5s"

        elif action_type == 'wait_long':
            time.sleep(DELAY_WAIT_LONG)
            self._troop_mgr._last_troop_name = None
            return "attendre 2.0s"

        elif action_type == 'done':
            return "DONE (fin episode)"

        return "???"

    def _execute_deploy(self, role_idx, sector_idx):
        """Deploy une troupe du rôle donné au secteur donné."""
        role_name = DEPLOY_ROLES[role_idx]

        # TroopManager choisit la prochaine troupe du rôle
        troop_idx, troop_name = self._troop_mgr.select_next_for_role(
            role_name, self._remaining_troops
        )

        if troop_idx is None:
            return f"⚠️ {role_name} épuisé"

        time.sleep(DELAY_SWITCH_TROOP)

        # Convertir secteur → position absolue
        abs_pos = TroopManager.sector_to_position(sector_idx, self._center_pos)

        # V4.2 : wraparound sur le nombre RÉEL de positions trouvées
        # (get_perimeter_from_buildings peut en retourner < NUM_POSITIONS).
        # Ne JAMAIS fallback sur _village_center qui est en plein HDV.
        if self._deploy_positions and len(self._deploy_positions) > 0:
            abs_pos = abs_pos % len(self._deploy_positions)
            x, y = self._deploy_positions[abs_pos]
        else:
            # Seul cas de fallback possible : aucune position trouvée du tout.
            # On tape juste à côté du bord gauche (safe), pas au centre.
            x, y = (100, 400)

        self._adb_tap(x, y)
        time.sleep(DELAY_DEPLOY)

        # Tracker héros
        troop = TROOP_TYPES[troop_idx]
        if troop['role'] == 'hero':
            self._hero_manager.mark_deployed(troop_name)

        # Mettre à jour les compteurs
        self._remaining_troops[troop_idx] = max(
            0, self._remaining_troops[troop_idx] - 1
        )
        self._sector_map[sector_idx] += 0.2
        self._last_sector = sector_idx

        return f"{troop_name} → {DEPLOY_ROLES[role_idx]}@{sector_idx}"

    def _execute_spell(self, spell_name):
        """Lance un sort avec ciblage automatique."""
        if spell_name not in TROOP_NAME_TO_IDX:
            return f"⚠️ sort {spell_name} inconnu"

        spell_idx = TROOP_NAME_TO_IDX[spell_name]
        if self._remaining_troops[spell_idx] <= 0:
            return f"⚠️ {spell_name} épuisé"

        # Sélectionner le sort dans la barre
        if not self._troop_mgr.select_troop(spell_name):
            return f"⚠️ {spell_name} non trouvé"

        time.sleep(DELAY_SWITCH_TROOP)

        # Ciblage auto via SpellCaster
        combat_img = self._adb_screenshot()
        if combat_img is not None:
            if self._combat_observer.has_yolo:
                _, raw = self._combat_observer.observe(
                    combat_img, self._village_center, phase='combat')
                targets = self._spell_caster.analyze_from_yolo(
                    raw, self._village_center)
            else:
                targets = self._spell_caster.analyze_battlefield(
                    combat_img, self._village_center)

            target_map = {'soin': 'heal', 'rage': 'rage', 'gel': 'freeze'}
            key = target_map.get(spell_name, 'heal')
            x, y = targets[key]
        else:
            x, y = self._village_center or (960, 500)

        self._adb_tap(x, y)
        time.sleep(0.3)

        self._remaining_troops[spell_idx] = max(
            0, self._remaining_troops[spell_idx] - 1
        )
        self._troop_mgr._last_troop_name = None

        return f"🧪{spell_name} → ({x}, {y})"

    def _execute_ability(self, hero_idx):
        """Active l'ability d'un héros."""
        hero_name = HERO_NAMES[hero_idx]

        # Scanner si l'icône n'est pas encore trouvée
        if hero_name not in self._hero_manager._icon_positions:
            screenshot = self._adb_screenshot()
            if screenshot is not None and self._hero_manager.has_templates():
                self._hero_manager.scan(screenshot)

        success = self._hero_manager.activate(hero_name, self._adb_tap)
        time.sleep(DELAY_ABILITY)

        if success:
            return f"⚡ {hero_name} ability activée"
        return f"⚠️ {hero_name} ability échouée"

    # -----------------------------------------------------------------
    #  Reward shaping V4
    # -----------------------------------------------------------------

    def _compute_shaping_reward(self, action_idx):
        """Reward shaping V4.2 — dispatch sur action_type au lieu de self._phase."""
        action_type, idx1, idx2 = decode_action(action_idx)

        if action_type == 'deploy':
            reward = compute_deploy_reward(
                action_type, idx1, idx2,
                self._tanks_deployed, self._troops_deployed,
                self._last_sector, self._combat_features,
            )
            if idx1 is not None:
                role = DEPLOY_ROLES[idx1]
                if role == 'tank':
                    self._tanks_deployed += 1
                self._troops_deployed += 1

        elif action_type in ('spell', 'ability', 'observe', 'wait_short', 'wait_long'):
            spell_name = idx1 if action_type == 'spell' else None
            hero_idx = idx1 if action_type == 'ability' else None
            reward = compute_combat_reward(
                action_type, spell_name, hero_idx,
                self._combat_features,
                self._combat_step_count,
                HERO_NAMES,
            )
            # Bonus destruction bâtiments détectée sur ce step (observe uniquement)
            if action_type == 'observe' and self._buildings_destroyed_total > 0:
                new_destroyed = self._buildings_destroyed_total - getattr(
                    self, '_last_rewarded_destroyed', 0
                )
                if new_destroyed > 0:
                    reward += 2.0 * new_destroyed
                    self._last_rewarded_destroyed = self._buildings_destroyed_total

        elif action_type == 'done':
            reward = compute_leftover_penalty(self._remaining_troops, TROOP_TYPES)
            reward += compute_spell_leftover_penalty(self._remaining_troops, TROOP_TYPES)

        else:
            reward = 0.0

        self._step_rewards.append(reward)
        return reward

    # -----------------------------------------------------------------
    #  Fin naturelle d'épisode
    # -----------------------------------------------------------------

    def _all_resources_exhausted(self):
        """
        Retourne True quand l'agent n'a plus rien à faire.

        Quand le compteur dit zéro, on fait UN rescan physique de la barre
        pour attraper les stragglers avant de conclure. Évite le cleanup
        post-episode tardif : les troupes résiduelles sont découvertes ici
        et déployées naturellement au prochain step.
        """
        role_counts, _ = build_role_inventory(self._remaining_troops, TROOP_TYPES)
        spell_counts = build_spell_inventory(self._remaining_troops, TROOP_TYPES)
        hero_mask = self._hero_manager.get_ability_mask()

        no_troops = all(v == 0 for v in role_counts.values())
        no_spells = all(v == 0 for v in spell_counts.values())
        no_abilities = all(hero_mask[i] == 0 for i in range(NUM_HEROES))

        if not (no_troops and no_spells and no_abilities):
            return False

        # Compteur à zéro — vérifier physiquement la barre une seule fois
        if not self._exhaustion_rescanned:
            self._exhaustion_rescanned = True
            self._troop_mgr.rescan(self._remaining_troops)
            # Re-évaluer après rescan
            role_counts, _ = build_role_inventory(self._remaining_troops, TROOP_TYPES)
            spell_counts = build_spell_inventory(self._remaining_troops, TROOP_TYPES)
            no_troops = all(v == 0 for v in role_counts.values())
            no_spells = all(v == 0 for v in spell_counts.values())
            if not (no_troops and no_spells):
                if self.verbose:
                    print("      ♻️  Stragglers détectés après rescan — épisode continue")
                return False

        return True

    # -----------------------------------------------------------------
    #  Step V4
    # -----------------------------------------------------------------

    def step(self, action_idx):
        """Exécute un step V4.2 — sans phases rigides."""
        self._step_count += 1
        action_type, _, _ = decode_action(action_idx)

        shaping = self._compute_shaping_reward(action_idx)
        action_desc = self._execute_action(action_idx)

        # Compteur proxy "combat" — s'incrémente dès qu'on ne déploie pas
        if action_type != 'deploy':
            self._combat_step_count += 1

        if self.verbose:
            sh = f" ({shaping:+.0f})" if shaping != 0 else ""
            print(f"   Step {self._step_count:2d} [{action_type}]: {action_desc}{sh}")

        # Rescan périodique
        if (self._step_count % RESCAN_EVERY_N_STEPS == 0
                and action_type != 'done'):
            self._troop_mgr.rescan(self._remaining_troops)

        # Fin d'épisode
        is_over = self._check_battle_end()
        is_done = (
            is_over
            or action_type == 'done'
            or self._all_resources_exhausted()
            or self._step_count >= MAX_STEPS_SAFETY
        )

        if is_done:
            # Déployer les troupes restantes si l'agent a choisi done prématurément
            any_troops_left = any(
                self._remaining_troops[i] > 0
                for i, t in enumerate(TROOP_TYPES)
                if t['role'] != 'spell'
            )
            if any_troops_left:
                self._troop_mgr.cleanup(
                    self._remaining_troops,
                    self._deploy_positions,
                    self._village_center,
                )
                time.sleep(1.0)  # laisser le jeu traiter les troupes

            reward, info = self._finish_episode()
            info['step'] = self._step_count
            info['combat_steps'] = self._combat_step_count
            info['abilities_used'] = self._hero_manager.num_activated()
            info['buildings_destroyed'] = self._buildings_destroyed_total
            spell_penalty = compute_spell_leftover_penalty(
                self._remaining_troops, TROOP_TYPES
            )
            if spell_penalty != 0:
                reward += spell_penalty
                if self.verbose:
                    print(f"   Malus sorts non utilises: {spell_penalty:.0f}")
            hero_bonus = compute_hero_survival_bonus(self._combat_features)
            if hero_bonus > 0:
                reward += hero_bonus
                if self.verbose:
                    heroes_alive = round((self._combat_features[4] if self._combat_features is not None else 0) * 5)
                    print(f"   Bonus héros survivants: +{hero_bonus:.0f} ({heroes_alive} héros)")
            return self._get_obs(), self._get_mask(), reward, True, info

        return self._get_obs(), self._get_mask(), shaping, False, {
            'step': self._step_count,
            'combat_step': self._combat_step_count,
        }

    # -----------------------------------------------------------------
    #  Reset override (V4 specific state)
    # -----------------------------------------------------------------

    def reset(self):
        """Reset V4 — appelle le reset V3 puis adapte."""
        super().reset()

        # V4.2 : forcer _phase='combat' pour les méthodes héritées V3
        # (_check_battle_end, _update_combat_observation utilisent self._phase)
        self._phase = 'combat'
        self._episode_start_time = time.time()

        # V4.2 : compteurs YOLO continu bâtiments
        self._buildings_destroyed_total = 0
        self._prev_building_count = len(self._buildings) if self._buildings else 0
        self._last_rewarded_destroyed = 0
        self._exhaustion_rescanned = False
        # Passer le compte initial au combat observer pour buildings_remaining_ratio
        self._combat_observer.start_combat(
            initial_building_count=self._prev_building_count
        )

        # Reset V4 state
        self._sector_map = np.zeros(NUM_SECTORS, dtype=np.float32)
        self._last_sector = None
        self._troop_mgr.reset()

        # V4.2 : recalculer les positions de déploiement depuis les bbox YOLO
        # Plus fiable que la détection HSV (overlay rouge parfois faible)
        if self._buildings:
            from clashai.perception.deploy_zone import (
                get_perimeter_from_buildings, save_deploy_debug_image,
            )
            from clashai.combat.state_encoder import find_best_attack_side

            debug_screenshot = self._adb_screenshot()

            result = get_perimeter_from_buildings(
                self._buildings, num_points=NUM_POSITIONS, return_debug=True,
                screenshot_pil=debug_screenshot,
            )
            yolo_positions, yolo_center, yolo_ok, deploy_debug = result
            if yolo_ok and yolo_positions:
                self._deploy_positions = yolo_positions
                self._village_center = yolo_center

            best_dir = find_best_attack_side(self._buildings, verbose=False)
            self._center_pos = int(best_dir / 8 * NUM_POSITIONS) % NUM_POSITIONS

            try:
                if debug_screenshot is not None:
                    extra = f'center_pos={self._center_pos} (dir best)'
                    path = save_deploy_debug_image(
                        debug_screenshot,
                        self._buildings,
                        self._deploy_positions or [],
                        self._village_center or (960, 500),
                        episode=self._episode_count,
                        extra_info=extra,
                        rejected_rays=deploy_debug.get('rejected_rays') if deploy_debug else None,
                    )
                    if path and self.verbose:
                        print(f"   📸 Debug deploy : {path}")
            except Exception as e:
                if self.verbose:
                    print(f"   ⚠️ Debug deploy image : {e}")
        else:
            self._center_pos = NUM_POSITIONS // 2

        # Rebuilds V4 obs
        return self._get_obs(), self._get_mask()

    # -----------------------------------------------------------------
    #  YOLO continu V4.2
    # -----------------------------------------------------------------

    def _update_combat_observation(self):
        """
        Override V4.2 — YOLO continu : bâtiments + troupes à chaque observe.
        Met à jour grid, features village ET features combat depuis un screenshot frais.
        """
        import time as _time
        screenshot = self._adb_screenshot()
        if screenshot is None:
            return

        # 1. YOLO bâtiments → rafraîchir la grille
        t0 = _time.time()
        new_buildings = self._analyze_village(screenshot, self.models)
        t_buildings = (_time.time() - t0) * 1000  # ms

        if new_buildings:
            from clashai.combat.state_encoder import encode_state
            state = encode_state(new_buildings)
            self._grid = state['grid']
            self._features = state['features']

            # Diff pour détecter les destructions
            curr_count = len(new_buildings)
            destroyed = max(0, self._prev_building_count - curr_count)
            self._buildings_destroyed_total += destroyed
            self._prev_building_count = curr_count
            self._buildings = new_buildings

            if destroyed > 0 and self.verbose:
                print(f"      💥 {destroyed} détruit(s) — "
                      f"total: {self._buildings_destroyed_total} "
                      f"({curr_count} restants)")

        # 2. YOLO troupes → rafraîchir combat features
        t0 = _time.time()
        spells_remaining = build_spell_inventory(self._remaining_troops, TROOP_TYPES)
        features, _ = self._combat_observer.observe(
            screenshot,
            village_center_adb=self._village_center,
            spells_remaining=spells_remaining,
            phase='combat',
            buildings_count=len(new_buildings) if new_buildings else self._prev_building_count,
        )
        t_troops = (_time.time() - t0) * 1000  # ms

        self._combat_features = features

        if self.verbose:
            print(f"      ⏱️  YOLO buildings: {t_buildings:.0f}ms | "
                  f"troops: {t_troops:.0f}ms")

    # -----------------------------------------------------------------
    #  Heuristic V4
    # -----------------------------------------------------------------

    def get_heuristic_sequence(self):
        """
        Séquence heuristique V4 — 1 action par unité.

        Chaque troupe = 1 action deploy(rôle, secteur).
        Les secteurs cyclent pour répartir les troupes.
        """
        from clashai.combat.action_space import encode_action as enc

        FAR_LEFT, LEFT, CENTER, RIGHT, FAR_RIGHT = 0, 1, 2, 3, 4
        TANK, RANGED, MELEE, HERO, SIEGE = 0, 1, 2, 3, 4

        actions = []
        role_inv, _ = build_role_inventory(self._remaining_troops, TROOP_TYPES)
        spell_inv = build_spell_inventory(self._remaining_troops, TROOP_TYPES)

        if self.verbose:
            print(f"   📋 Inventaire V4 : {dict(role_inv)} | sorts: {dict(spell_inv)}")

        # 1. TANKS — spread aux extrémités puis centre
        tank_sectors = [FAR_LEFT, FAR_RIGHT, CENTER, LEFT, RIGHT]
        for i in range(role_inv.get('tank', 0)):
            actions.append(enc('deploy', TANK, tank_sectors[i % len(tank_sectors)]))

        actions.append(enc('wait_long'))

        # 2. FUNNEL — 2 ranged aux extrémités
        funnel = min(role_inv.get('ranged', 0), 2)
        funnel_secs = [FAR_LEFT, FAR_RIGHT]
        for i in range(funnel):
            actions.append(enc('deploy', RANGED, funnel_secs[i]))

        actions.append(enc('wait_short'))

        # 3. RANGED — le reste en ligne (left, center, right)
        ranged_remaining = max(0, role_inv.get('ranged', 0) - funnel)
        ranged_sectors = [LEFT, CENTER, RIGHT]
        for i in range(ranged_remaining):
            actions.append(enc('deploy', RANGED, ranged_sectors[i % len(ranged_sectors)]))

        actions.append(enc('wait_long'))

        # 4. MELEE — au centre
        melee_sectors = [CENTER, LEFT, RIGHT]
        for i in range(role_inv.get('melee', 0)):
            actions.append(enc('deploy', MELEE, melee_sectors[i % len(melee_sectors)]))

        # 5. SIEGE — centre (V4.1: siège AVANT héros)
        for _ in range(role_inv.get('siege', 0)):
            actions.append(enc('deploy', SIEGE, CENTER))

        # 6. HEROES — centre
        for _ in range(role_inv.get('hero', 0)):
            actions.append(enc('deploy', HERO, CENTER))

        # V4.2 : pas de done intermédiaire — done = fin d'épisode
        # Les sorts et abilities suivent directement le deploy.
        actions.append(enc('observe'))

        # Sorts en priorité — rage, gel, soin (ordre tactique)
        spell_priority = ['rage', 'gel', 'soin']
        for spell_name in spell_priority:
            count = spell_inv.get(spell_name, 0)
            for _ in range(count):
                actions.append(enc('observe'))
                actions.append(enc('spell', spell_name))

        # Abilities — uniquement les héros déployés (skip championne/PG si absents)
        for i, hero_name in enumerate(HERO_NAMES):
            if self._hero_manager.is_deployed(hero_name):
                actions.append(enc('observe'))
                actions.append(enc('ability', i))

        actions.append(enc('observe'))
        actions.append(enc('done'))  # fin explicite pour BC pre-training

        return actions