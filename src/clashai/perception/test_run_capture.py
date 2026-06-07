# clashai/perception/test_run_capture.py
# Test-mode capture: saves 5 annotated screenshots during one episode so we
# can visually validate that all perception models (screen CNN, YOLO
# buildings, YOLO walls deploy zone, YOLO troop bar, EasyOCR counters,
# YOLO troops combat) see the game correctly before launching real training.
#
# Trigger: tools/train/train_rl_v4.py --test
#
# Output: logs/test_run/<label>.png
#   village_home.png  — village home screen, screen-state CNN overlay
#   prep_attaque.png  — army selection, + troop bar bboxes & counters
#   debut_attaque.png — t=0 of attack, all annotations
#   attaque_30s.png   — t≈30s into the attack
#   attaque_60s.png   — t≈60s into the attack
#
# Each file is written ONCE per episode (idempotent on label).

import os
import time
import threading
import cv2
import numpy as np


# Per-label annotation profile. Each profile is a set of annotation tags
# applied to the captured frame.
PROFILES = {
    'village_home':   {'screen', 'buildings'},
    'prep_attaque':   {'screen', 'troop_bar'},
    'debut_attaque':  {'screen', 'buildings', 'deploy', 'troop_bar'},
    'attaque_30s':    {'screen', 'buildings', 'deploy', 'troop_bar', 'troops_combat'},
    'attaque_60s':    {'screen', 'buildings', 'deploy', 'troop_bar', 'troops_combat'},
}

EXPECTED_LABELS = list(PROFILES.keys())


class TestRunCapture:
    """
    Coordinator for the 5 diagnostic captures of a `--test` run.

    Lifecycle:
        cap = TestRunCapture()
        # in env.reset() / navigation:
        cap.maybe_save_screen('village_home', img, models, env)
        cap.maybe_save_screen('prep_attaque', img, models, env)
        # once phase_attaque is reached:
        cap.mark_attack_start()
        cap.maybe_save_combat(img, models, env)  # → debut_attaque.png
        # later, called from each observe step:
        cap.maybe_save_combat(img, models, env)  # → attaque_30s / 60s
        # at the end:
        cap.report()
    """

    def __init__(self, output_dir='logs/test_run', verbose=True):
        self.output_dir = output_dir
        self.verbose = verbose
        self.screens_dir = os.path.join(output_dir, 'screens')
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.screens_dir, exist_ok=True)

        self._saved = set()
        self._attack_t0 = None
        self._initial_buildings = None  # snapshot for destruction diff
        self._lock = threading.Lock()

        # Trace of every CNN screen state transition during the run.
        # Lets the user see exactly what the screen classifier saw at each
        # navigation step, with confidence overlaid — useful to debug
        # mispredictions or unstable transitions.
        self._last_screen_state = None
        self._screen_seq = 0

    def mark_attack_start(self):
        """Called when phase_attaque first detected. Starts the 30s/60s timer."""
        if self._attack_t0 is None:
            self._attack_t0 = time.time()

    def maybe_save_screen(self, state, img_pil, models, env=None):
        """Saves village_home.png or prep_attaque.png if state matches and
        the corresponding file hasn't been written yet."""
        if state not in ('village_home', 'prep_attaque'):
            return
        with self._lock:
            if state in self._saved:
                return
            self._save_annotated(img_pil, state, models, env)

    def trace_screen(self, state, confidence, img_pil, models, env=None):
        """
        Saves a screenshot every time the CNN screen-state classifier
        produces a different prediction from the previous one. Useful to
        verify the CNN's view at each navigation step.

        Output: logs/test_run/screens/NN_state_conf.png (NN = sequence).
        Consecutive duplicates are deduped — if village_home is reported
        20 times in a row we keep only the first occurrence per run.
        """
        if img_pil is None or state is None:
            return
        with self._lock:
            if state == self._last_screen_state:
                return
            self._last_screen_state = state
            self._screen_seq += 1
            seq = self._screen_seq

            try:
                img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                _draw_screen_state_with_conf(img_cv, state, confidence, seq)
                safe_state = ''.join(c if c.isalnum() else '_' for c in state)
                fname = f"{seq:02d}_{safe_state}_{int(confidence * 100):02d}.png"
                path = os.path.join(self.screens_dir, fname)
                cv2.imwrite(path, img_cv)
                if self.verbose:
                    print(f" Screen trace: {path}")
            except Exception as e:
                if self.verbose:
                    print(f" WARNING: screen trace failed: {e}")

    def maybe_save_combat(self, img_pil, models, env):
        """
        Saves combat captures based on elapsed time since mark_attack_start():
          - first call after mark_attack_start  → debut_attaque.png
          - then t >= 30s                       → attaque_30s.png
          - then t >= 60s                       → attaque_60s.png
        """
        if self._attack_t0 is None:
            return
        t = time.time() - self._attack_t0
        with self._lock:
            if 'debut_attaque' not in self._saved:
                self._save_annotated(img_pil, 'debut_attaque', models, env)
                # snapshot initial buildings for later destruction overlay
                buildings = getattr(env, '_buildings', None)
                if buildings:
                    self._initial_buildings = list(buildings)
                return
            if t >= 30 and 'attaque_30s' not in self._saved:
                self._save_annotated(img_pil, 'attaque_30s', models, env)
                return
            if t >= 60 and 'attaque_60s' not in self._saved:
                self._save_annotated(img_pil, 'attaque_60s', models, env)
                return

    def is_complete(self):
        """True if all 5 expected captures have been written."""
        return all(lbl in self._saved for lbl in EXPECTED_LABELS)

    def report(self):
        """Print a summary of which captures were saved / skipped."""
        print(f"\n{'='*60}")
        print(f" Test run captures -> {os.path.abspath(self.output_dir)}")
        print(f"{'='*60}")
        for lbl in EXPECTED_LABELS:
            mark = '[OK]' if lbl in self._saved else '[--]'
            print(f" {mark}  {lbl}.png")
        print(f"{'-'*60}")
        print(f" Screen trace: {self._screen_seq} state transitions captured")
        print(f"   -> {os.path.abspath(self.screens_dir)}/")
        print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Internal: annotation
    # ------------------------------------------------------------------

    def _save_annotated(self, img_pil, label, models, env):
        """Render the requested annotations for `label` and save to disk."""
        if img_pil is None:
            if self.verbose:
                print(f" WARNING: test capture {label} skipped (no image)")
            return
        profile = PROFILES.get(label, set())
        try:
            img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            from clashai.perception.coord_utils import ImageScaler
            scaler = ImageScaler(img_cv)
            h = scaler.img_h
            sx, sy = scaler.sx, scaler.sy

            if 'screen' in profile:
                _draw_screen_state(img_cv, img_pil, models, label)
            if 'buildings' in profile:
                buildings = _annotate_buildings(img_cv, img_pil, models, env, sx, sy)
            else:
                buildings = []
            if 'deploy' in profile:
                _annotate_deploy_zone(img_cv, env, sx, sy)
            if 'troops_combat' in profile:
                _annotate_combat_troops(img_cv, env, sx, sy)
                _highlight_destroyed(img_cv, self._initial_buildings, buildings, sx, sy)
            if 'troop_bar' in profile:
                _annotate_troop_bar(img_cv, img_pil, models, sx, sy)

            # Footer
            attack_t = (time.time() - self._attack_t0) if self._attack_t0 else 0
            footer = f"{label}  |  {len(buildings)} bldg  |  attack_t={attack_t:.1f}s"
            _draw_text(img_cv, footer, (8, h - 10), 0.55, white=True)

            path = os.path.join(self.output_dir, f'{label}.png')
            cv2.imwrite(path, img_cv)
            self._saved.add(label)
            if self.verbose:
                print(f" Test capture saved: {path}")
        except Exception as e:
            if self.verbose:
                print(f" WARNING: test capture {label} failed: {e}")


# =============================================================================
# Annotation primitives (all operate on the cv2 BGR image in-place)
# =============================================================================

# Category → BGR color for building bboxes
_BUILDING_COLORS = {
    'defense':   (0, 0, 255),     # red
    'tour':      (0, 0, 255),
    'canon':     (0, 0, 255),
    'mortier':   (0, 0, 255),
    'inferno':   (0, 80, 255),
    'aigle':     (0, 100, 255),
    'tesla':     (0, 50, 255),
    'arbalete':  (0, 80, 255),
    'cdc':       (255, 200, 0),   # cyan-ish — clan castle
    'hdv':       (255, 0, 255),   # magenta — town hall
    'ressource': (0, 200, 0),     # green
    'or':        (0, 200, 0),
    'elixir':    (200, 0, 200),
    'mur':       (180, 180, 180),
    'default':   (200, 200, 200),
}


def _color_for_building(cls_name):
    cls = cls_name.lower()
    for key, color in _BUILDING_COLORS.items():
        if key in cls:
            return color
    return _BUILDING_COLORS['default']


def _draw_text(img, text, org, scale=0.6, white=True):
    """Text with a black outline so it's readable on any background."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), max(2, int(scale * 4)), cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255) if white else (0, 255, 0),
                max(1, int(scale * 2)), cv2.LINE_AA)


def _draw_screen_state(img_cv, img_pil, models, label):
    try:
        from clashai.navigation.game_loop import classify_screen
        state, conf = classify_screen(img_pil, models)
    except Exception:
        state, conf = '?', 0.0
    color = (0, 255, 0) if conf > 0.7 else (0, 165, 255)
    cv2.rectangle(img_cv, (4, 4), (520, 56), (0, 0, 0), -1)
    cv2.putText(img_cv, f"[{label}]  CNN: {state} ({conf:.0%})",
                (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)


def _draw_screen_state_with_conf(img_cv, state, conf, seq):
    """Variant for trace_screen: shows the CNN's own confidence (not a
    re-classification) plus a sequence index."""
    color = (0, 255, 0) if conf > 0.7 else (0, 165, 255)
    cv2.rectangle(img_cv, (4, 4), (640, 60), (0, 0, 0), -1)
    cv2.putText(img_cv, f"#{seq:02d}  CNN: {state} ({conf:.0%})",
                (12, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)


def _annotate_buildings(img_cv, img_pil, models, env, sx, sy):
    try:
        from clashai.navigation.game_loop import analyze_village
        buildings = analyze_village(img_pil, models)
    except Exception:
        buildings = []
    for b in buildings:
        x1, y1, x2, y2 = b['bbox']
        color = _color_for_building(b['class'])
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
        label = f"{b['class'][:14]} {b['confidence']:.2f}"
        _draw_text(img_cv, label, (x1, max(12, y1 - 4)), 0.35)
    return buildings


def _annotate_deploy_zone(img_cv, env, sx, sy):
    positions = getattr(env, '_deploy_positions', None) if env else None
    center = getattr(env, '_village_center', None) if env else None
    if not positions:
        return
    pts = [(int(x * sx), int(y * sy)) for x, y in positions]

    # Hull (closed polyline through the points)
    if len(pts) >= 3:
        hull_pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img_cv, [hull_pts], isClosed=True,
                      color=(0, 255, 0), thickness=2)

    # Numbered points
    for i, (x, y) in enumerate(pts):
        cv2.circle(img_cv, (x, y), 9, (0, 0, 220), -1)
        cv2.circle(img_cv, (x, y), 9, (255, 255, 255), 1)
        _draw_text(img_cv, str(i), (x - 5, y + 4), 0.40)

    # Village center marker
    if center:
        cx, cy = int(center[0] * sx), int(center[1] * sy)
        cv2.drawMarker(img_cv, (cx, cy), (0, 255, 255),
                       cv2.MARKER_CROSS, 18, 2)


def _annotate_troop_bar(img_cv, img_pil, models, sx, sy):
    bar = models.get('troop_bar_detector') if models else None
    if bar is None:
        return
    try:
        dets = bar.detect(img_pil)
    except Exception:
        return
    for d in dets:
        x1, y1, x2, y2 = d['bbox']
        is_gray = d.get('is_grayed', False)
        color = (90, 90, 90) if is_gray else (0, 200, 255)
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
        count = d.get('count', '?')
        text = f"{d['name'][:12]} x{count}"
        _draw_text(img_cv, text, (x1, max(12, y1 - 4)), 0.40)


def _annotate_combat_troops(img_cv, env, sx, sy):
    """Yellow circles on YOLO-detected combat troops (from troop_finder)."""
    finder = getattr(env, '_troop_finder', None) if env else None
    if finder is None:
        return
    positions = getattr(finder, 'positions', None) or {}
    for name, val in positions.items():
        if isinstance(val, (tuple, list)) and len(val) >= 2:
            x, y = val[0], val[1]
        elif isinstance(val, dict):
            x, y = val.get('x'), val.get('y')
        else:
            continue
        if x is None or y is None:
            continue
        cx, cy = int(x * sx), int(y * sy)
        cv2.circle(img_cv, (cx, cy), 14, (0, 255, 255), 2)
        _draw_text(img_cv, name[:8], (cx - 14, cy - 18), 0.4)


def _highlight_destroyed(img_cv, initial_buildings, current_buildings, sx, sy):
    """Overlays an X on bboxes that were present at debut_attaque but no
    longer detected. Approximates destruction by spatial proximity (~64px)."""
    if not initial_buildings or not current_buildings:
        return
    curr_centers = [
        ((b['bbox'][0] + b['bbox'][2]) // 2,
         (b['bbox'][1] + b['bbox'][3]) // 2)
        for b in current_buildings
    ]
    for b in initial_buildings:
        x1, y1, x2, y2 = b['bbox']
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        # Is there a current building roughly here?
        alive = any(abs(cx - x) < 64 and abs(cy - y) < 64
                    for x, y in curr_centers)
        if alive:
            continue
        cv2.line(img_cv, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.line(img_cv, (x1, y2), (x2, y1), (0, 0, 255), 3)
