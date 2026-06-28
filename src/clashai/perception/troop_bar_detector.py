# clashai/perception/troop_bar_detector.py
# YOLO-based troop bar detector — replaces TroopFinder template matching.
#
# Detects all troop/spell/hero/siege icons in the bottom bar via YOLO,
# then applies an HSV saturation filter to distinguish active vs grayed icons,
# and reads the counter (x2, x11...) via EasyOCR on the counter crop.
#
# Interface compatible with TroopFinder.positions:
#   positions = {name: (x, y, conf, count)}  — only active (non-grayed) slots

import cv2
import numpy as np

# HSV saturation below this threshold = icon is grayed out (depleted/upgrading)
GRAYED_SAT_THRESHOLD = 30

# Confidence threshold for YOLO detection.
# Session 13: lowered 0.45 → 0.40 after validating on real training frames.
# At 0.45, borderline icons (e.g. golem @ 0.41) were dropped → the agent
# missed troops the manual tool found. 0.40 catches all without false
# positives (the troop bar is a controlled UI region). 0.50 starts missing.
YOLO_CONF = 0.40
# Inference image size.
#
# Session 13 finding: setting this to 1600 (the value in
# tools/train/train_yolo_troop_bar.py::IMG_SIZE) TANKED detection — only 0-1
# icons found out of ~9 visible. At default 640 we recover 9/9 detections.
# Likely cause: double-resize (WGC 2451x1411 → LANCZOS 1920x1080 → YOLO
# letterbox 1600x1600) blurs small icons, OR the checkpoint was actually
# trained at a different imgsz than what's hardcoded in the train script.
#
# Sticking to 640 (Ultralytics' default) until next retrain validates a
# higher imgsz with explicit train→infer parity benchmarking.
YOLO_IMGSZ = 1088

# Classes that should NEVER be tapped even if detected (deployed siege machines)
NO_TAP_CLASSES = {
    'demolisseur_deploye', 'dirigeable_deploye', 'broyeur_pierre_deploye',
    'caserne_siege_deploye', 'lance_buche_deploye', 'catapulte_deploye',
    'foreuse_deploye', 'lance_troupe_deploye', 'fourgon_celeste_deploye',
}

# Classes that never have a counter (abilities, deployed siege)
NO_COUNTER_CLASSES = NO_TAP_CLASSES | {
    'roi_capa', 'reine_capa', 'grand_gardien_capa',
    'championne_capa', 'prince_gargouille_capa', 'duc_draconique_capa',
}

# Hero icons in the troop bar — always exactly 1 in the army (can't queue 2
# heroes). The OCR sometimes reads "11" or "23" on the small white badge,
# which is a misread; we cap to 1 to ignore those.
UNIQUE_HEROES = {
    'roi', 'reine', 'grand_gardien',
    'championne', 'prince_gargouille', 'duc_draconique',
}


class TroopBarDetector:
    """
    YOLO-based troop bar icon detector.

    Pipeline per frame:
      1. YOLO → bboxes + class names
      2. HSV saturation → is_grayed (depleted/upgrading)

    Returns presence + active/grayed state. EasyOCR-based counter reading
    was removed Session 13 (counts were unreliable — typical misreads like
    "sorcier x74" overwrote correct manual-decrement counters and broke
    the cleanup phase). Counts come from manual decrement in the env;
    F.3 mini-CNN (planned) will eventually provide reliable counts.

    Replaces TroopFinder (template matching).
    """

    def __init__(self, model_path, verbose=False):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.verbose = verbose
        self._last_detections = []
        from clashai.config.logging import pp
        pp(f" TroopBarDetector loaded: {len(self.model.names)} classes", tag='yolo')

    def detect(self, screenshot_pil, screen='combat', prev_counts=None):
        """
        Detects all icons in the troop bar.

        Args:
            screenshot_pil: PIL.Image of the full screen (any resolution)
            screen: kept for API back-compat; no longer affects detection
                    (was used for OCR crop position, now unused).
            prev_counts: kept for API back-compat; no longer used.

        Returns:
            List of dicts:
            {
                'name':      str   — class name ('golem', 'soin', 'roi_capa'...)
                'bbox':      (x1, y1, x2, y2)
                'center':    (cx, cy)
                'conf':      float
                'count':     int   — 0 if grayed, 1 otherwise (presence only)
                'is_grayed': bool  — depleted/unavailable
                'no_tap':    bool  — clicking would be destructive
            }
        """
        # CRITICAL (Session 13): pass the PIL image directly, NOT np.array(pil).
        # Ultralytics reads a numpy array as BGR (cv2 convention) but PIL as
        # RGB. Feeding np.array(rgb_pil) swapped the R/B channels → systematic
        # misclassification of color-dependent icons (gel↔poison, soin↔clone,
        # championne missed). PIL input keeps the channels correct.
        from clashai.perception.inference_lock import INFERENCE_LOCK
        with INFERENCE_LOCK:
            results = self.model.predict(
                screenshot_pil, conf=YOLO_CONF, imgsz=YOLO_IMGSZ, verbose=False
            )

        detections = []
        r = results[0]
        names = r.names

        for box in r.boxes:
            conf = float(box.conf[0])
            cls  = int(box.cls[0])
            name = names.get(cls, f'cls{cls}')
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            crop = screenshot_pil.crop((x1, y1, x2, y2))
            is_grayed = self._is_grayed(crop)

            # Count = presence flag only (OCR removed Session 13).
            # The env keeps the authoritative count via manual decrement.
            count = 0 if is_grayed else 1

            detections.append({
                'name':      name,
                'bbox':      (x1, y1, x2, y2),
                'center':    (cx, cy),
                'conf':      conf,
                'count':     count,
                'is_grayed': is_grayed,
                'no_tap':    name in NO_TAP_CLASSES,
            })

        self._last_detections = detections

        if self.verbose:
            active = [d for d in detections if not d['is_grayed'] and not d['no_tap']]
            summary = '  '.join(f"{d['name']}x{d['count']}" for d in active)
            print(f" TroopBar: {summary or 'none'}")

        return detections

    def to_positions(self, detections=None):
        """
        Converts detections to TroopFinder.positions format:
            {name: (x, y, conf)}

        Only includes ACTIVE (non-grayed, non-deploye) icons.

        The tap point is the icon's UPPER part, not its geometric center:
        siege machines and the Grand Warden carry a green mode/swap arrow at the
        bottom — tapping the center hits it and opens a sub-menu (the unit then
        fails to deploy). The upper part selects any icon safely.
        """
        if detections is None:
            detections = self._last_detections

        positions = {}
        for d in detections:
            if d['is_grayed'] or d['no_tap']:
                continue
            x1, y1, x2, y2 = d['bbox']
            tap_x = (x1 + x2) // 2
            tap_y = y1 + int((y2 - y1) * 0.35)   # upper part — avoids the bottom arrow
            positions[d['name']] = (tap_x, tap_y, d['conf'])

        return positions

    def to_counts(self, detections=None, prev_counts=None):
        """
        Returns {name: count} for all active icons. Since OCR was removed
        (Session 13), count is presence-based: 1 if visible & active,
        0 (omitted) if grayed.

        Args kept for API back-compat.
        """
        if detections is None:
            detections = self._last_detections

        counts = {}
        for d in detections:
            if d['is_grayed'] or d['no_tap']:
                continue
            counts[d['name']] = 1
        return counts

    def _is_grayed(self, crop_pil):
        """
        Returns True if the icon crop is grayed out (saturation < threshold).
        Works for:
          - Troops depleted during combat (stock = 0)
          - Heroes in upgrade (shown on prep_attaque screen)
          - Abilities already used (cooldown)
        """
        if crop_pil.width < 4 or crop_pil.height < 4:
            return False
        hsv = cv2.cvtColor(np.array(crop_pil), cv2.COLOR_RGB2HSV)
        avg_sat = float(np.mean(hsv[:, :, 1]))
        return avg_sat < GRAYED_SAT_THRESHOLD
