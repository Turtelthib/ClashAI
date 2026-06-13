# clashai/combat/environment_v4/core.py
# CoreMixin — V4 lifecycle: __init__, reset, step, episode-end, screen hook.

import time

import numpy as np

from clashai.combat.action_space import (
    TOTAL_ACTIONS, NUM_ROLES, NUM_SECTORS, NUM_HEROES,
    SPELL_NAMES, NUM_POSITIONS, MAX_STEPS_SAFETY,
    decode_action, build_role_inventory, build_spell_inventory,
)
from clashai.combat.agent_v4 import VECTOR_SIZE
from clashai.combat.troop_manager import TroopManager
from clashai.combat.reward_shaping import (
    compute_spell_leftover_penalty, compute_hero_survival_bonus,
)
from clashai.combat.legacy.agent import TROOP_TYPES, TROOP_NAME_TO_IDX


class CoreMixin:
    """Lifecycle of the V4 environment (init / reset / step / episode end)."""

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
    # Natural episode end
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
            from clashai.config.logging import pp
            sh = f" ({shaping:+.0f})" if shaping != 0 else ""
            tag = action_type if action_type in (
                'deploy', 'spell', 'ability', 'observe', 'done'
            ) else 'wait' if action_type.startswith('wait') else 'step'
            pp(f" Step {self._step_count:2d} [{action_type}]: {action_desc}{sh}", tag=tag)

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
