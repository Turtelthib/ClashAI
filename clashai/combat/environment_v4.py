# clashai/combat/environment_v4.py
# V4 environment for ClashAI.
#
# Inherits from ClashEnvV3 to reuse all ADB navigation
# and overrides key methods for the new action space.
#
# Changes vs V3:
# - 37 actions (role × sector + auto-targeted spells)
# - Compacted observation (54 dims instead of 76)
# - TroopManager for role-based deployment
# - Centralized reward shaping
# - Agent freely chooses the order of spells/abilities

import os
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

# V3 imports to access constants
from clashai.combat.agent import (
    TROOP_TYPES, TROOP_NAME_TO_IDX,
)

# Delays — re-imported from clashai/config/timing.py (Phase A).
from clashai.config import (
    DELAY_SWITCH_TROOP, DELAY_DEPLOY,
    DELAY_WAIT_SHORT, DELAY_WAIT_LONG,
    DELAY_OBSERVE, DELAY_ABILITY,
)  # noqa: E402

# V4.3: periodic step rescan removed — _sync_remaining_from_perception()
# now keeps _remaining_troops fresh from the YOLO troop bar in PerceptionThread.
NO_TROOPS_CHECKS_THRESHOLD = 3


class ClashEnvV4(ClashEnvV3):
    """
    V4 environment — simplified action space (37 actions).

    Inherits from V3 for ADB navigation; overrides:
    - Observation (54 dims)
    - Action mask (37 actions)
    - Action execution (role × sector)
    - Reward shaping (separate module)
    """

    def __init__(self, models, verbose=True, debug_overlay=False,
                 test_capture=None):
        super().__init__(models, verbose)
        self._debug_overlay = debug_overlay
        # TestRunCapture instance for --test mode. When set, screen-state
        # detections + observe steps will save the 5 diagnostic captures.
        self._test_capture = test_capture

        # TroopManager V4 (replaces direct V3 selection)
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
        self._center_pos = NUM_POSITIONS // 2
        # Initialized here so _get_obs() (called by super().reset()) does not crash
        self._episode_start_time = time.time()
        self._buildings_destroyed_total = 0
        self._prev_building_count = 0
        self._last_rewarded_destroyed = 0
        self._exhaustion_rescanned = False

        if verbose:
            print("\nClashEnv V4 initialisé")
            print(f" Actions : {TOTAL_ACTIONS} "
                  f"({NUM_ROLES}×{NUM_SECTORS} deploy + "
                  f"{len(SPELL_NAMES)} sorts + {NUM_HEROES} abilities)")
            print(f" Vector : {VECTOR_SIZE} dims")
            print(" Phases : fusionnees (V4.2)")
            print(f" Safety cap : {MAX_STEPS_SAFETY} steps")

    # -----------------------------------------------------------------
    # Observation V4 (54 dims)
    # -----------------------------------------------------------------

    def _get_obs(self):
        """Builds the V4 observation: grid + 54-dim vector."""
        # Elapsed time normalized over 180s (CoC 3-min timer) — more stable than step/MAX
        elapsed = time.time() - self._episode_start_time
        time_norm = np.array([min(elapsed / 180.0, 1.0)], dtype=np.float32)

        combat_feats = (self._combat_features
                        if self._combat_features is not None
                        else np.zeros(COMBAT_FEATURES_SIZE, dtype=np.float32))

        hero_status = self._hero_manager.get_status_vector()

        # V4: role counts instead of individual troop counts
        role_counts = np.zeros(ROLE_FEATURES, dtype=np.float32)
        for i, role in enumerate(DEPLOY_ROLES):
            for troop in TROOP_TYPES:
                if troop['role'] == role and troop['name'] in TROOP_NAME_TO_IDX:
                    idx = TROOP_NAME_TO_IDX[troop['name']]
                    role_counts[i] += self._remaining_troops[idx]
        role_counts = role_counts / 10.0

        # V4: spell counts
        spell_counts = np.zeros(SPELL_FEATURES, dtype=np.float32)
        for i, spell_name in enumerate(SPELL_NAMES):
            if spell_name in TROOP_NAME_TO_IDX:
                spell_counts[i] = self._remaining_troops[TROOP_NAME_TO_IDX[spell_name]]
        spell_counts = spell_counts / 3.0

        vector = np.concatenate([
            self._features,
            role_counts,
            spell_counts,
            self._sector_map,
            time_norm,
            combat_feats,
            hero_status,
        ])

        return self._grid, vector

    # -----------------------------------------------------------------
    # Action mask V4 (37 actions)
    # -----------------------------------------------------------------

    def _get_mask(self):
        """V4 action mask."""
        hero_mask = self._hero_manager.get_ability_mask()
        return compute_action_mask(
            self._remaining_troops,
            TROOP_TYPES,
            hero_ability_mask=hero_mask,
        )

    # -----------------------------------------------------------------
    # Execute action V4
    # -----------------------------------------------------------------

    def _execute_action(self, action_idx):
        """Executes a V4 action."""
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
            return f"observe ({DELAY_OBSERVE}s)"

        elif action_type == 'wait_short':
            time.sleep(DELAY_WAIT_SHORT)
            self._troop_mgr._last_troop_name = None
            return "wait 0.5s"

        elif action_type == 'wait_long':
            time.sleep(DELAY_WAIT_LONG)
            self._troop_mgr._last_troop_name = None
            return "wait 2.0s"

        elif action_type == 'done':
            return "DONE (episode end)"

        return "???"

    def _execute_deploy(self, role_idx, sector_idx):
        """Deploys a troop of the given role at the given sector."""
        role_name = DEPLOY_ROLES[role_idx]

        # TroopManager selects the next troop for the role
        troop_idx, troop_name = self._troop_mgr.select_next_for_role(
            role_name, self._remaining_troops
        )

        if troop_idx is None:
            return f"WARNING: {role_name} exhausted"

        # Only sleep when a real troop switch happened (TroopFinder.select
        # already waits 0.15s on tap; if same troop is already selected,
        # no delay needed at all)
        if troop_name != self._troop_mgr._last_troop_name:
            time.sleep(DELAY_SWITCH_TROOP)

        # Convert sector → absolute position
        abs_pos = TroopManager.sector_to_position(sector_idx, self._center_pos)

        # V4.2: wraparound on the REAL number of found positions
        # (get_perimeter_from_buildings may return < NUM_POSITIONS).
        # NEVER fall back on _village_center which is inside the TH.
        if self._deploy_positions and len(self._deploy_positions) > 0:
            abs_pos = abs_pos % len(self._deploy_positions)
            x, y = self._deploy_positions[abs_pos]
        else:
            # Only possible fallback: no positions found at all.
            # Tap near the left edge (safe), not at the center.
            x, y = (100, 400)

        self._adb_tap(x, y)
        time.sleep(DELAY_DEPLOY)

        # Track heroes
        troop = TROOP_TYPES[troop_idx]
        if troop['role'] == 'hero':
            self._hero_manager.mark_deployed(troop_name)

        # Update counters
        self._remaining_troops[troop_idx] = max(
            0, self._remaining_troops[troop_idx] - 1
        )
        self._sector_map[sector_idx] += 0.2
        self._last_sector = sector_idx

        return f"{troop_name} → {DEPLOY_ROLES[role_idx]}@{sector_idx}"

    def _execute_spell(self, spell_name):
        """Casts a spell with automatic targeting."""
        if spell_name not in TROOP_NAME_TO_IDX:
            return f"WARNING: spell {spell_name} unknown"

        spell_idx = TROOP_NAME_TO_IDX[spell_name]
        if self._remaining_troops[spell_idx] <= 0:
            return f"WARNING: {spell_name} exhausted"

        # Select the spell in the bar
        if not self._troop_mgr.select_troop(spell_name):
            return f"WARNING: {spell_name} not found"

        time.sleep(DELAY_SWITCH_TROOP)

        # Auto-targeting via SpellCaster
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

        return f"{spell_name} → ({x}, {y})"

    def _execute_ability(self, hero_idx):
        """Activates a hero's ability."""
        hero_name = HERO_NAMES[hero_idx]

        # Scan if the icon has not been found yet
        if hero_name not in self._hero_manager._icon_positions:
            screenshot = self._adb_screenshot()
            if screenshot is not None and self._hero_manager.has_templates():
                self._hero_manager.scan(screenshot)

        success = self._hero_manager.activate(hero_name, self._adb_tap)
        time.sleep(DELAY_ABILITY)

        if success:
            return f"{hero_name} ability activated"
        return f"WARNING: {hero_name} ability failed"

    # -----------------------------------------------------------------
    # Reward shaping V4
    # -----------------------------------------------------------------

    def _compute_shaping_reward(self, action_idx):
        """Reward shaping V4.2 — dispatches on action_type instead of self._phase."""
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
            # Building destruction bonus detected on this step (observe only)
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
    # Fin naturelle d'épisode
    # -----------------------------------------------------------------

    def _all_resources_exhausted(self):
        """
        Returns True when the agent has nothing left to do.

        When the counter reaches zero, one physical bar rescan is performed
        to catch stragglers before concluding. Avoids late post-episode
        cleanup: residual troops are discovered here and deployed naturally
        on the next step.
        """
        role_counts, _ = build_role_inventory(self._remaining_troops, TROOP_TYPES)
        spell_counts = build_spell_inventory(self._remaining_troops, TROOP_TYPES)
        hero_mask = self._hero_manager.get_ability_mask()

        no_troops = all(v == 0 for v in role_counts.values())
        no_spells = all(v == 0 for v in spell_counts.values())
        no_abilities = all(hero_mask[i] == 0 for i in range(NUM_HEROES))

        if not (no_troops and no_spells and no_abilities):
            return False

        # Counter at zero — physically check the bar once
        if not self._exhaustion_rescanned:
            self._exhaustion_rescanned = True
            self._troop_mgr.rescan(self._remaining_troops)
            # Re-evaluate after rescan
            role_counts, _ = build_role_inventory(self._remaining_troops, TROOP_TYPES)
            spell_counts = build_spell_inventory(self._remaining_troops, TROOP_TYPES)
            no_troops = all(v == 0 for v in role_counts.values())
            no_spells = all(v == 0 for v in spell_counts.values())
            if not (no_troops and no_spells):
                if self.verbose:
                    print(" Stragglers detected after rescan — episode continues")
                return False

        return True

    # -----------------------------------------------------------------
    # Step V4
    # -----------------------------------------------------------------

    def step(self, action_idx):
        """Executes a V4.2 step — no rigid phases."""
        self._step_count += 1
        action_type, _, _ = decode_action(action_idx)

        shaping = self._compute_shaping_reward(action_idx)
        action_desc = self._execute_action(action_idx)

        # Proxy "combat" counter — increments whenever not deploying
        if action_type != 'deploy':
            self._combat_step_count += 1

        if self.verbose:
            sh = f" ({shaping:+.0f})" if shaping != 0 else ""
            print(f" Step {self._step_count:2d} [{action_type}]: {action_desc}{sh}")

        # V4.3: periodic rescan removed — TroopBarDetector runs every frame
        # in PerceptionThread, and _sync_remaining_from_perception() is called
        # in _update_combat_observation() to keep _remaining_troops fresh.
        # Only the one-shot exhaustion sanity rescan in
        # _all_resources_exhausted() remains, as a safety net before declaring
        # the episode finished.

        # Episode end
        is_over = self._check_battle_end()
        is_done = (
            is_over
            or action_type == 'done'
            or self._all_resources_exhausted()
            or self._step_count >= MAX_STEPS_SAFETY
        )

        if is_done:
            # Deploy remaining troops if the agent chose done prematurely
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
                time.sleep(1.0)

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
                    print(f" Malus sorts non utilises: {spell_penalty:.0f}")
            hero_bonus = compute_hero_survival_bonus(self._combat_features)
            if hero_bonus > 0:
                reward += hero_bonus
                if self.verbose:
                    heroes_alive = round((self._combat_features[4] if self._combat_features is not None else 0) * 5)
                    print(f" Bonus héros survivants: +{hero_bonus:.0f} ({heroes_alive} héros)")
            return self._get_obs(), self._get_mask(), reward, True, info

        return self._get_obs(), self._get_mask(), shaping, False, {
            'step': self._step_count,
            'combat_step': self._combat_step_count,
        }

    # -----------------------------------------------------------------
    # Test-mode hook: notify TestRunCapture of every classified screen
    # -----------------------------------------------------------------

    def _get_screen_state(self):
        state, confidence, img_pil = super()._get_screen_state()
        if self._test_capture is not None and img_pil is not None:
            # Trace every CNN screen-state transition (dedup consecutive
            # duplicates) → logs/test_run/screens/NN_state_conf.png
            self._test_capture.trace_screen(
                state, confidence, img_pil, self.models, env=self
            )
            # The 5 main diagnostic captures (village_home / prep_attaque /
            # debut_attaque / 30s / 60s) are still saved by the more
            # detailed annotation path:
            if state == 'phase_attaque':
                self._test_capture.mark_attack_start()
            else:
                self._test_capture.maybe_save_screen(
                    state, img_pil, self.models, env=self
                )
        return state, confidence, img_pil

    # -----------------------------------------------------------------
    # Reset override (V4 specific state)
    # -----------------------------------------------------------------

    def reset(self):
        """Reset V4 — calls the V3 reset then adapts."""
        # V4.3: pause async perception during navigation (save GPU)
        _pt = self.models.get('perception_thread') if self.models else None
        if _pt is not None:
            _pt.pause()

        super().reset()

        # V4.3: resume perception once we're on the attack screen
        if _pt is not None:
            _pt.resume()

        # V4.2: force _phase='combat' for inherited V3 methods
        # (_check_battle_end, _update_combat_observation use self._phase)
        self._phase = 'combat'
        self._episode_start_time = time.time()

        # V4.2: continuous YOLO building counters
        self._buildings_destroyed_total = 0
        self._prev_building_count = len(self._buildings) if self._buildings else 0
        self._last_rewarded_destroyed = 0
        self._exhaustion_rescanned = False
        # Pass the initial count to the combat observer for buildings_remaining_ratio
        self._combat_observer.start_combat(
            initial_building_count=self._prev_building_count
        )

        # Reset V4 state
        self._sector_map = np.zeros(NUM_SECTORS, dtype=np.float32)
        self._last_sector = None
        self._troop_mgr.reset()

        # V4.3: deploy zone — walls segmentation (primary) then building hull (fallback)
        if self._buildings:
            from clashai.perception.deploy_zone import (
                get_perimeter_from_buildings, get_perimeter_from_walls,
            )
            from clashai.combat.state_encoder import find_best_attack_side

            debug_screenshot = self._adb_screenshot()
            yolo_ok = False

            # Primary: wall segmentation model (robust to theme/color changes)
            yolo_walls = self.models.get('yolo_walls') if self.models else None
            if yolo_walls is not None and debug_screenshot is not None:
                wall_positions, wall_center, wall_ok = get_perimeter_from_walls(
                    debug_screenshot, yolo_walls,
                    buildings=self._buildings,
                    num_points=NUM_POSITIONS,
                )
                if wall_ok and wall_positions:
                    self._deploy_positions = wall_positions
                    self._village_center = wall_center
                    yolo_ok = True

            # Fallback: building bbox hull (V4.2)
            if not yolo_ok:
                result = get_perimeter_from_buildings(
                    self._buildings, num_points=NUM_POSITIONS, return_debug=False,
                    screenshot_pil=debug_screenshot,
                )
                yolo_positions, yolo_center, yolo_ok = result
                if yolo_ok and yolo_positions:
                    self._deploy_positions = yolo_positions
                    self._village_center = yolo_center

            best_dir = find_best_attack_side(self._buildings, verbose=False)
            self._center_pos = int(best_dir / 8 * NUM_POSITIONS) % NUM_POSITIONS

            # Save start-of-episode annotated capture (replaces deploy_zone log)
            if debug_screenshot is not None:
                self._schedule_episode_captures(debug_screenshot)
        else:
            self._center_pos = NUM_POSITIONS // 2

        # Rebuilds V4 obs
        return self._get_obs(), self._get_mask()

    # -----------------------------------------------------------------
    # YOLO continu V4.2
    # -----------------------------------------------------------------

    def _sync_remaining_from_perception(self, troop_bar_detections):
        """
        V4.3 — sync `_remaining_troops` with the YOLO troop bar detector
        running in PerceptionThread. Replaces the periodic `rescan()` call.

        For each detected slot:
          - is_grayed → that troop is depleted, force count = 0
          - active   → use the OCR count (with hero hardcap at 1)

        Manual decrement after each deploy still happens; this just corrects
        drift when YOLO+OCR sees a different reality.
        """
        if not troop_bar_detections:
            return
        bar = self.models.get('troop_bar_detector') if self.models else None
        if bar is None:
            return

        try:
            from clashai.combat.troop_manager import ALIAS_MAP
        except ImportError:
            ALIAS_MAP = {}

        # Set of names actually visible in the bar (post-grey filter)
        visible_active = set()
        for d in troop_bar_detections:
            if d.get('no_tap'):
                continue
            name = ALIAS_MAP.get(d['name'], d['name'])
            if name not in TROOP_NAME_TO_IDX:
                continue
            idx = TROOP_NAME_TO_IDX[name]
            if d.get('is_grayed'):
                self._remaining_troops[idx] = 0
            else:
                visible_active.add(name)

        # Counts (with hero hardcap inside to_counts())
        try:
            counts = bar.to_counts(troop_bar_detections)
        except Exception:
            counts = {}
        for raw_name, cnt in counts.items():
            name = ALIAS_MAP.get(raw_name, raw_name)
            if name in TROOP_NAME_TO_IDX:
                self._remaining_troops[TROOP_NAME_TO_IDX[name]] = float(cnt)

    def _update_combat_observation(self):
        """
        V4.3 — reads from the async PerceptionThread (non-blocking).
        Falls back to V4.2 blocking YOLO if the thread is unavailable.
        """
        import time as _time

        #  Try async perception thread (V4.3) 
        perception = self.models.get('perception_thread') if self.models else None
        if perception is not None and perception.is_fresh(max_age_s=1.0):
            state = perception.get_latest()
            screenshot = state['frame']
            new_buildings = state['buildings']

            if new_buildings and screenshot is not None:
                from clashai.combat.state_encoder import encode_state
                enc = encode_state(new_buildings)
                self._grid = enc['grid']
                self._features = enc['features']

                curr_count = len(new_buildings)
                destroyed = max(0, self._prev_building_count - curr_count)
                self._buildings_destroyed_total += destroyed
                self._prev_building_count = curr_count
                self._buildings = new_buildings

                if destroyed > 0 and self.verbose:
                    print(f" {destroyed} destroyed — "
                          f"total: {self._buildings_destroyed_total} "
                          f"({curr_count} remaining)")

                if state['combat_features'] is not None:
                    self._combat_features = state['combat_features']

                # V4.3: keep _remaining_troops in sync with the YOLO troop bar
                # detector that runs in PerceptionThread (replaces the
                # periodic rescan that used to fire every 10 steps).
                self._sync_remaining_from_perception(state.get('troop_bar'))

                # Hero ability scan from the cached frame
                self._hero_manager.scan(screenshot)

                if self.verbose:
                    print(f" Perception cache: {len(new_buildings)} bldg | "
                          f"{state['inference_ms']:.0f}ms (async)")

                # Debug overlay
                if self._debug_overlay:
                    self._save_debug_overlay(screenshot, new_buildings)
                # Test mode: maybe save debut/30s/60s combat captures
                if self._test_capture is not None:
                    self._test_capture.maybe_save_combat(
                        screenshot, self.models, env=self
                    )
                return

        #  Fallback: V4.2 blocking YOLO 
        screenshot = self._adb_screenshot()
        if screenshot is None:
            return

        # 1. YOLO buildings → refresh the grid
        t0 = _time.time()
        new_buildings = self._analyze_village(screenshot, self.models)
        t_buildings = (_time.time() - t0) * 1000

        if new_buildings:
            from clashai.combat.state_encoder import encode_state
            state = encode_state(new_buildings)
            self._grid = state['grid']
            self._features = state['features']

            # Diff to detect destructions
            curr_count = len(new_buildings)
            destroyed = max(0, self._prev_building_count - curr_count)
            self._buildings_destroyed_total += destroyed
            self._prev_building_count = curr_count
            self._buildings = new_buildings

            if destroyed > 0 and self.verbose:
                print(f" {destroyed} destroyed — "
                      f"total: {self._buildings_destroyed_total} "
                      f"({curr_count} remaining)")

        # 2. Hero ability scan — populates _icon_positions so the mask can enable abilities.
        # Must run before get_ability_mask() is called in _get_mask().
        self._hero_manager.scan(screenshot)

        # 3. YOLO troops → refresh combat features
        t0 = _time.time()
        spells_remaining = build_spell_inventory(self._remaining_troops, TROOP_TYPES)
        features, _ = self._combat_observer.observe(
            screenshot,
            village_center_adb=self._village_center,
            spells_remaining=spells_remaining,
            phase='combat',
            buildings_count=len(new_buildings) if new_buildings else self._prev_building_count,
        )
        t_troops = (_time.time() - t0) * 1000

        self._combat_features = features

        if self.verbose:
            print(f" YOLO buildings: {t_buildings:.0f}ms | "
                  f"troops: {t_troops:.0f}ms")

        if self._debug_overlay:
            self._save_debug_overlay(screenshot, new_buildings)
        # Test mode: maybe save debut/30s/60s combat captures
        if self._test_capture is not None:
            self._test_capture.maybe_save_combat(
                screenshot, self.models, env=self
            )

    def _schedule_episode_captures(self, start_screenshot):
        """
        Saves annotated captures for the current episode:
          - t=0s  (start_screenshot already taken)
          - t=15s (scheduled via threading.Timer)
          - t=30s (scheduled via threading.Timer)
        All saved to logs/episode_NNNN/.
        """
        import threading
        ep = self._episode_count

        # t=0 — we already have the screenshot
        self._save_episode_capture(start_screenshot, ep, label='t0s')

        # t=15s and t=30s — capture in background (non-blocking)
        def _capture_later(delay, label):
            import time as _t
            _t.sleep(delay)
            try:
                img = self._adb_screenshot()
                if img:
                    self._save_episode_capture(img, ep, label=label)
            except Exception:
                pass

        threading.Thread(target=_capture_later, args=(15, 't15s'), daemon=True).start()
        threading.Thread(target=_capture_later, args=(30, 't30s'), daemon=True).start()

    def _save_episode_capture(self, screenshot, episode, label='t0s'):
        """Saves one annotated capture for the episode folder."""
        try:
            import cv2
            import numpy as np
            from clashai.navigation.game_loop import classify_screen, analyze_village

            ep_dir = os.path.join('logs', f'episode_{episode:04d}')
            os.makedirs(ep_dir, exist_ok=True)

            img_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            from clashai.perception.coord_utils import ImageScaler
            scaler = ImageScaler(img_cv)
            h = scaler.img_h
            sx, sy = scaler.sx, scaler.sy  # legacy names — used below

            # Screen state
            state, conf = classify_screen(screenshot, self.models)
            cv2.putText(img_cv, f"Screen: {state} ({conf:.0%})", (8, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0) if conf > 0.7 else (0, 165, 255), 2)

            # Buildings YOLO
            buildings = analyze_village(screenshot, self.models)
            for b in buildings:
                x1, y1, x2, y2 = b['bbox']
                cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 200, 0), 1)
                cv2.putText(img_cv, b['class'][:10], (x1, y1 - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.27, (0, 255, 0), 1)

            # Deploy positions
            if self._deploy_positions:
                for i, (px, py) in enumerate(self._deploy_positions):
                    cv2.circle(img_cv, (int(px * sx), int(py * sy)), 7, (0, 0, 220), -1)
                    cv2.putText(img_cv, str(i), (int(px * sx) - 4, int(py * sy) + 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

            # Troop bar detector
            bar_det = self.models.get('troop_bar_detector') if self.models else None
            if bar_det is not None:
                for d in bar_det.detect(screenshot):
                    x1, y1, x2, y2 = d['bbox']
                    color = (80, 80, 80) if d['is_grayed'] else (0, 200, 255)
                    cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img_cv, f"{d['name']} x{d['count']}", (x1, y1 - 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1)

            # Stats
            cv2.putText(img_cv, f"Ep {episode} | {label} | {len(buildings)} bldg",
                        (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(img_cv, f"Ep {episode} | {label} | {len(buildings)} bldg",
                        (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            path = os.path.join(ep_dir, f'{label}.jpg')
            cv2.imwrite(path, img_cv, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if self.verbose:
                print(f" Episode capture: {path}")
        except Exception as e:
            if self.verbose:
                print(f" WARNING: episode capture failed: {e}")

    def _save_debug_overlay(self, screenshot, buildings):
        """Saves a debug overlay image for the current observe step."""
        try:
            from clashai.perception.debug_overlay import save_debug_overlay
            troop_positions = getattr(self._troop_finder, 'positions', {})
            save_debug_overlay(
                screenshot_pil=screenshot,
                step=self._step_count,
                episode=self._episode_count,
                buildings=buildings,
                deploy_positions=self._deploy_positions,
                troop_positions=troop_positions,
                remaining_troops=self._remaining_troops,
                troop_types=TROOP_TYPES,
            )
        except Exception as e:
            if self.verbose:
                print(f" WARNING: debug overlay: {e}")

    # -----------------------------------------------------------------
    # Heuristic V4
    # -----------------------------------------------------------------

    def get_heuristic_sequence(self):
        """
        Heuristic sequence V4 — 1 action per unit.

        Each troop = 1 deploy(role, sector) action.
        Sectors cycle to spread troops evenly.
        """
        from clashai.combat.action_space import encode_action as enc

        FAR_LEFT, LEFT, CENTER, RIGHT, FAR_RIGHT = 0, 1, 2, 3, 4
        TANK, RANGED, MELEE, HERO, SIEGE = 0, 1, 2, 3, 4

        actions = []
        role_inv, _ = build_role_inventory(self._remaining_troops, TROOP_TYPES)
        spell_inv = build_spell_inventory(self._remaining_troops, TROOP_TYPES)

        if self.verbose:
            print(f" V4 inventory: {dict(role_inv)} | spells: {dict(spell_inv)}")

        # 1. TANKS — spread at extremes then center
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

        # 3. RANGED — remaining in a line (left, center, right)
        ranged_remaining = max(0, role_inv.get('ranged', 0) - funnel)
        ranged_sectors = [LEFT, CENTER, RIGHT]
        for i in range(ranged_remaining):
            actions.append(enc('deploy', RANGED, ranged_sectors[i % len(ranged_sectors)]))

        actions.append(enc('wait_long'))

        # 4. MELEE — at center
        melee_sectors = [CENTER, LEFT, RIGHT]
        for i in range(role_inv.get('melee', 0)):
            actions.append(enc('deploy', MELEE, melee_sectors[i % len(melee_sectors)]))

        # 5. SIEGE — center (V4.1: siege BEFORE heroes)
        for _ in range(role_inv.get('siege', 0)):
            actions.append(enc('deploy', SIEGE, CENTER))

        # 6. HEROES — centre
        for _ in range(role_inv.get('hero', 0)):
            actions.append(enc('deploy', HERO, CENTER))

        # V4.2: no intermediate done — done = end of episode
        # Spells and abilities follow directly after deploy.
        actions.append(enc('observe'))

        # Spells in priority order — rage, freeze, heal (tactical order)
        spell_priority = ['rage', 'gel', 'soin']
        for spell_name in spell_priority:
            count = spell_inv.get(spell_name, 0)
            for _ in range(count):
                actions.append(enc('observe'))
                actions.append(enc('spell', spell_name))

        # Abilities — deployed heroes only (skip championne/PG if absent)
        for i, hero_name in enumerate(HERO_NAMES):
            if self._hero_manager.is_deployed(hero_name):
                actions.append(enc('observe'))
                actions.append(enc('ability', i))

        actions.append(enc('observe'))
        actions.append(enc('done'))

        return actions