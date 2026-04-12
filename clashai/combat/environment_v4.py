# clashai/combat/environment_v4.py
# Environnement V4 pour ClashAI.
#
# Hérite de ClashEnvV3 pour réutiliser toute la navigation ADB
# et override les méthodes clés pour le nouvel action space.
#
# Changements vs V3 :
#   - 37 actions (rôle × secteur + sorts auto-ciblés)
#   - Observation compactée (55 dims au lieu de 76)
#   - TroopManager pour le deploy par rôle
#   - Reward shaping centralisé
#   - L'agent choisit librement l'ordre des sorts/abilities

import time
import numpy as np

from clashai.combat.environment import ClashEnvV3
from clashai.combat.action_space import (
    TOTAL_ACTIONS, NUM_ROLES, NUM_SECTORS, NUM_HEROES,
    DEPLOY_ROLES, HERO_NAMES, SPELL_NAMES,
    MAX_STEPS_PER_EPISODE, MAX_COMBAT_STEPS, NUM_POSITIONS,
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
    compute_leftover_penalty,
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
    - Observation (55 dims)
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

        if verbose:
            print("\n🎮 ClashEnv V4 initialisé")
            print(f"   Actions     : {TOTAL_ACTIONS} "
                  f"({NUM_ROLES}×{NUM_SECTORS} deploy + "
                  f"{len(SPELL_NAMES)} sorts + {NUM_HEROES} abilities)")
            print(f"   Vector      : {VECTOR_SIZE} dims")
            print("   Phases      : deploy → combat")
            print(f"   Max steps   : {MAX_STEPS_PER_EPISODE} "
                  f"(dont {MAX_COMBAT_STEPS} combat)")

    # -----------------------------------------------------------------
    #  Observation V4 (55 dims)
    # -----------------------------------------------------------------

    def _get_obs(self):
        """Construit l'observation V4 : grid + vector 55 dims."""
        step_norm = np.array(
            [self._step_count / MAX_STEPS_PER_EPISODE],
            dtype=np.float32
        )
        phase_indicator = np.array(
            [1.0 if self._phase == 'combat' else 0.0],
            dtype=np.float32
        )

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
            step_norm,                # (1,)
            combat_feats,             # (15,)
            hero_status,              # (5,)
            phase_indicator,          # (1,)
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
            phase=self._phase,
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
            if self._phase == 'deploy':
                return "DONE (deploy → combat)"
            return "DONE (fin combat)"

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

        if self._deploy_positions and abs_pos < len(self._deploy_positions):
            x, y = self._deploy_positions[abs_pos]
        else:
            x, y = self._village_center or (960, 500)

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
        """Reward shaping V4 — délègue au module reward_shaping."""
        action_type, idx1, idx2 = decode_action(action_idx)

        if self._phase == 'deploy':
            reward = compute_deploy_reward(
                action_type, idx1, idx2,
                self._tanks_deployed, self._troops_deployed,
                self._last_sector, self._combat_features,
            )
            # Compteurs
            if action_type == 'deploy' and idx1 is not None:
                role = DEPLOY_ROLES[idx1]
                if role == 'tank':
                    self._tanks_deployed += 1
                self._troops_deployed += 1

            if action_type == 'done':
                reward += compute_leftover_penalty(
                    self._remaining_troops, TROOP_TYPES
                )

        elif self._phase == 'combat':
            reward = compute_combat_reward(
                action_type, idx1, idx2,
                self._combat_features,
                self._combat_step_count,
                HERO_NAMES,
            )
        else:
            reward = 0.0

        self._step_rewards.append(reward)
        return reward

    # -----------------------------------------------------------------
    #  Step V4
    # -----------------------------------------------------------------

    def step(self, action_idx):
        """Exécute un step V4."""
        self._step_count += 1

        shaping = self._compute_shaping_reward(action_idx)
        action_desc = self._execute_action(action_idx)
        action_type, _, _ = decode_action(action_idx)

        if self.verbose:
            tag = "🏗️" if self._phase == 'deploy' else "⚔️"
            sh = f" ({shaping:+.0f})" if shaping != 0 else ""
            print(f"   {tag} Step {self._step_count:2d}: {action_desc}{sh}")

        # Rescan périodique
        if (self._phase == 'deploy'
                and self._step_count % RESCAN_EVERY_N_STEPS == 0
                and action_type != 'done'):
            self._troop_mgr.rescan(self._remaining_troops)

        # Transition deploy → combat
        if action_type == 'done' and self._phase == 'deploy':
            self._troop_mgr.cleanup(
                self._remaining_troops,
                self._deploy_positions,
                self._village_center,
            )
            self._phase = 'combat'
            self._combat_step_count = 0
            self._combat_observer.start_combat()

            if self.verbose:
                print("\n   ⚔️ ═══ PHASE COMBAT ═══")
                print(f"   Héros déployés : {self._hero_manager.num_deployed()}")

            self._update_combat_observation()
            return self._get_obs(), self._get_mask(), shaping, False, {
                'step': self._step_count, 'phase': 'combat'
            }

        # Phase combat
        if self._phase == 'combat':
            self._combat_step_count += 1
            is_over = self._check_battle_end()

            is_done = (
                is_over
                or action_type == 'done'
                or self._combat_step_count >= MAX_COMBAT_STEPS
                or self._step_count >= MAX_STEPS_PER_EPISODE
            )

            if is_done:
                reward, info = self._finish_episode()
                info['combat_steps'] = self._combat_step_count
                info['abilities_used'] = self._hero_manager.num_activated()
                return self._get_obs(), self._get_mask(), reward, True, info

            return self._get_obs(), self._get_mask(), shaping, False, {
                'step': self._step_count,
                'combat_step': self._combat_step_count,
                'phase': 'combat',
            }

        # Phase deploy
        is_done = self._step_count >= MAX_STEPS_PER_EPISODE
        if is_done:
            reward, info = self._finish_episode()
            return self._get_obs(), self._get_mask(), reward, True, info

        return self._get_obs(), self._get_mask(), shaping, False, {
            'step': self._step_count,
        }

    # -----------------------------------------------------------------
    #  Reset override (V4 specific state)
    # -----------------------------------------------------------------

    def reset(self):
        """Reset V4 — appelle le reset V3 puis adapte."""
        obs, mask = super().reset()

        # Reset V4 state
        self._sector_map = np.zeros(NUM_SECTORS, dtype=np.float32)
        self._last_sector = None
        self._troop_mgr.reset()

        # Calculer le center_pos depuis le meilleur côté d'attaque
        if self._buildings:
            from clashai.combat.state_encoder import find_best_attack_side
            best_dir = find_best_attack_side(
                self._buildings, verbose=False
            )
            self._center_pos = int(best_dir / 8 * NUM_POSITIONS) % NUM_POSITIONS
        else:
            self._center_pos = NUM_POSITIONS // 2

        # Rebuilds V4 obs
        return self._get_obs(), self._get_mask()

    # -----------------------------------------------------------------
    #  Heuristic V4
    # -----------------------------------------------------------------

    def get_heuristic_sequence(self):
        """
        Séquence heuristique V4 — rôle × secteur.

        Même stratégie que V3 mais avec les actions V4.
        """
        from clashai.combat.action_space import encode_action as enc

        # Secteurs relatifs
        FAR_LEFT, LEFT, CENTER, RIGHT, FAR_RIGHT = 0, 1, 2, 3, 4
        TANK, RANGED, MELEE, HERO, SIEGE = 0, 1, 2, 3, 4

        actions = []
        role_inv, _ = build_role_inventory(self._remaining_troops, TROOP_TYPES)
        spell_inv = build_spell_inventory(self._remaining_troops, TROOP_TYPES)

        if self.verbose:
            print(f"   📋 Inventaire V4 : {dict(role_inv)} | sorts: {dict(spell_inv)}")

        # DEPLOY : tanks → wait → funnel → ranged → melee → siege → heroes → done
        for _ in range(role_inv.get('tank', 0)):
            for sec in [FAR_LEFT, FAR_RIGHT, CENTER]:
                if role_inv['tank'] > 0:
                    actions.append(enc('deploy', TANK, sec))

        actions.append(enc('wait_long'))

        for sec in [FAR_LEFT, FAR_RIGHT]:
            if role_inv.get('ranged', 0) > 0:
                actions.append(enc('deploy', RANGED, sec))

        actions.append(enc('wait_short'))

        for _ in range(role_inv.get('ranged', 0)):
            for sec in [LEFT, CENTER, RIGHT]:
                actions.append(enc('deploy', RANGED, sec))

        actions.append(enc('wait_long'))

        for sec in [CENTER, LEFT, RIGHT]:
            if role_inv.get('melee', 0) > 0:
                actions.append(enc('deploy', MELEE, sec))

        for _ in range(role_inv.get('siege', 0)):
            actions.append(enc('deploy', SIEGE, CENTER))

        for _ in range(role_inv.get('hero', 0)):
            actions.append(enc('deploy', HERO, CENTER))

        actions.append(enc('done'))

        # COMBAT : sorts + abilities + observe
        for spell_name, count in spell_inv.items():
            for _ in range(count):
                actions.append(enc('spell', spell_name))

        for i, hero in enumerate(HERO_NAMES):
            if hero in TROOP_NAME_TO_IDX:
                idx = TROOP_NAME_TO_IDX[hero]
                if self._remaining_troops[idx] > 0 or True:
                    actions.append(enc('ability', i))

        for _ in range(5):
            actions.append(enc('observe'))

        return actions
