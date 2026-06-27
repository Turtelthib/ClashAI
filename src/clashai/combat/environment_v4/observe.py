# clashai/combat/environment_v4/observe.py
# ObserveMixin — refresh combat observation from PerceptionThread (async) or
# blocking YOLO (fallback), and keep troop counters in sync.

from clashai.combat.action_space import build_spell_inventory
from clashai.combat.legacy.agent import TROOP_TYPES, TROOP_NAME_TO_IDX


class ObserveMixin:
    """V4.3 perception sync: async PerceptionThread cache, blocking fallback."""

    def _sync_remaining_from_perception(self, troop_bar_detections):
        """
        V4.3 — sync `_remaining_troops` with the YOLO troop bar detector
        running in PerceptionThread. Replaces the periodic `rescan()` call.

        Session 13 cleanup: OCR-based count overwrite removed (counts
        were unreliable, e.g. "sorcier x74" misreads corrupted the
        correct manual-decrement counters and broke the cleanup phase).
        Now we only use YOLO's `is_grayed` signal to detect depletion;
        manual decrement after each deploy remains the authoritative
        source of remaining counts.
        """
        if not troop_bar_detections:
            return

        try:
            from clashai.combat.troop_manager import ALIAS_MAP
        except ImportError:
            ALIAS_MAP = {}

        for d in troop_bar_detections:
            if d.get('no_tap') or not d.get('is_grayed'):
                continue
            name = ALIAS_MAP.get(d['name'], d['name'])
            if name in TROOP_NAME_TO_IDX:
                self._remaining_troops[TROOP_NAME_TO_IDX[name]] = 0

    def _sync_grayed_from_cache(self):
        """Cheap grayed-only refresh from the async perception cache (no GPU).

        The heuristic/agent deploys in a BURST before the next `observe` step,
        so the grayed signal — only synced at observe time — is ignored during
        the burst and depleted troops get tapped until their (over-estimated)
        default_max counter drains. Reading the PerceptionThread cache here is
        free (no inference) and lets select_next_for_role() skip grayed troops
        mid-burst. Falsely-zeroed troops are still recovered by cleanup() at
        episode end (tap-until-gray).
        """
        perception = self.models.get('perception_thread') if self.models else None
        if perception is None or not perception.is_fresh(max_age_s=1.5):
            return
        self._sync_remaining_from_perception(perception.get_latest().get('troop_bar'))

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

                # Hero ability availability from the cached troop bar CNN
                self._hero_manager.update_from_troop_bar(state.get('troop_bar'))

                if self.verbose:
                    from clashai.config.logging import pp, styled
                    bldg_str = styled(f"{len(new_buildings)} bldg", 'yolo_alt')
                    pp(f" Perception cache: {bldg_str} | "
                       f"{state['inference_ms']:.0f}ms (async)",
                       tag='yolo_dim')

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

        # 2. Hero ability availability from the troop bar CNN — populates
        # _icon_positions so the mask can enable abilities. Must run before
        # get_ability_mask() is called in _get_mask().
        bar_det = self.models.get('troop_bar_detector') if self.models else None
        if bar_det is not None:
            self._hero_manager.update_from_troop_bar(bar_det.detect(screenshot))

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
