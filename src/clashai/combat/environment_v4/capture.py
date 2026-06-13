# clashai/combat/environment_v4/capture.py
# CaptureMixin — annotated episode captures + per-step debug overlays.

import os

from clashai.combat.legacy.agent import TROOP_TYPES


class CaptureMixin:
    """Saves annotated captures (episode timeline + debug overlays)."""

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
                from clashai.config.logging import pp
                pp(f" Episode capture: {path}", tag='init')
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
