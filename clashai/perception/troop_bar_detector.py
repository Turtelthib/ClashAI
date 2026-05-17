# clashai/perception/troop_bar_detector.py
# YOLO-based troop bar detector — replaces TroopFinder template matching.
#
# Detects all troop/spell/hero/siege icons in the bottom bar via YOLO,
# then applies an HSV saturation filter to distinguish active vs grayed icons,
# and reads the counter (x2, x11...) via EasyOCR on the counter crop.
#
# Interface compatible with TroopFinder.positions:
#   positions = {name: (x, y, conf, count)}  — only active (non-grayed) slots

import os
import cv2
import numpy as np
from PIL import Image

# HSV saturation below this threshold = icon is grayed out (depleted/upgrading)
GRAYED_SAT_THRESHOLD = 30

# Confidence threshold for YOLO detection
YOLO_CONF = 0.30
# Match the training imgsz of the troop bar weights (see
# tools/train_yolo_troop_bar.py::IMG_SIZE). Ultralytics' default at predict
# is 640, which silently halves the resolution the model sees and tanks
# detection quality on small icons / counter badges.
YOLO_IMGSZ = 1600

# Counter position differs by screen:
#   prep_attaque  → top-LEFT  corner of the icon (army selection screen)
#   phase_attaque → top-RIGHT corner of the icon (battle bar during combat)
# Expressed as fraction of the bbox size.
COUNTER_CROP_Y_FRAC = 0.40   # top portion height (same for both)

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
    YOLO-based troop bar icon detector with counter reading.

    Pipeline per frame:
      1. YOLO → bboxes + class names
      2. HSV saturation → is_grayed (depleted/upgrading)
      3. EasyOCR on counter crop → count (how many left)

    Replaces both TroopFinder (template matching) and TroopCounter (OCR).
    """

    def __init__(self, model_path, verbose=False):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.verbose = verbose
        self._last_detections = []
        self._ocr = None  # lazy init — EasyOCR is slow to load
        print(f" TroopBarDetector loaded: {len(self.model.names)} classes")

    def _get_ocr(self):
        """Lazy-loads EasyOCR reader (only digits, fast inference)."""
        if self._ocr is None:
            import easyocr
            self._ocr = easyocr.Reader(['en'], gpu=True, verbose=False)
            print(" TroopBarDetector: EasyOCR loaded")
        return self._ocr

    def detect(self, screenshot_pil, screen='combat', prev_counts=None):
        """
        Detects all icons in the troop bar.

        Args:
            screenshot_pil: PIL.Image of the full screen (any resolution)
            screen: 'combat'  → counter at top-RIGHT of icon (battle bar)
                    'prep'    → counter at top-LEFT  of icon (army selection)
            prev_counts: dict {name: int} — last known counts (from remaining_troops).
                         Used as upper bound: OCR reading > prev+2 is rejected.
                         Enables monotonic validation without any hardcoded limits.

        Returns:
            List of dicts:
            {
                'name':      str   — class name ('golem', 'soin', 'roi_capa'...)
                'bbox':      (x1, y1, x2, y2)
                'center':    (cx, cy)
                'conf':      float
                'count':     int   — number remaining (1 if unreadable)
                'is_grayed': bool  — depleted/unavailable
                'no_tap':    bool  — clicking would be destructive
            }
        """
        img_arr = np.array(screenshot_pil)
        results = self.model.predict(
            img_arr, conf=YOLO_CONF, imgsz=YOLO_IMGSZ, verbose=False
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

            # Read counter — pass prev_count for monotonic validation
            prev = prev_counts.get(name) if prev_counts else None
            count = 0 if is_grayed else self._read_count(
                screenshot_pil, x1, y1, x2, y2, name,
                screen=screen, prev_count=prev
            )

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
        """
        if detections is None:
            detections = self._last_detections

        positions = {}
        for d in detections:
            if d['is_grayed'] or d['no_tap']:
                continue
            cx, cy = d['center']
            positions[d['name']] = (cx, cy, d['conf'])

        return positions

    def to_counts(self, detections=None, prev_counts=None):
        """
        Returns {name: count} for all active icons.
        Useful for updating remaining_troops in the environment.

        Example: {'golem': 2, 'sorcier': 12, 'soin': 2, 'rage': 3}
        """
        if detections is None:
            detections = self._last_detections

        counts = {}
        for d in detections:
            if d['is_grayed'] or d['no_tap']:
                continue
            name = d['name']
            # Hero abilities + heroes themselves are always exactly 1 (you
            # can only have 1 of each in your army) — clamp regardless of OCR
            if name in NO_COUNTER_CLASSES or name in UNIQUE_HEROES:
                counts[name] = 1
            else:
                counts[name] = max(counts.get(name, 0), d['count'])

        return counts

    def _read_count(self, screenshot_pil, x1, y1, x2, y2, name, screen='combat',
                    prev_count=None):
        """
        Reads the troop counter ("x2", "x11"...) from the icon.

        Production approach — no hardcoded limits:
          - 3 preprocessing strategies, consensus vote
          - Monotonic validation: counts only decrease during combat.
            If OCR reads > prev_count+2 it's a misread → keep prev_count.
          - Upper bound = prev_count (set dynamically from remaining_troops)
        """
        if name in NO_COUNTER_CLASSES or name in UNIQUE_HEROES:
            return 1

        w = x2 - x1
        h = y2 - y1
        cy2 = y1 + int(h * COUNTER_CROP_Y_FRAC)
        margin = 4

        if screen == 'prep':
            crop_x1 = max(0, x1 - margin)
            crop_x2 = min(screenshot_pil.width, x1 + int(w * 0.45) + margin)
        else:
            crop_x1 = max(0, x1 + int(w * 0.55) - margin)
            crop_x2 = min(screenshot_pil.width, x2 + margin)

        crop = screenshot_pil.crop((
            crop_x1,
            max(0, y1 - margin),
            crop_x2,
            min(screenshot_pil.height, cy2 + margin),
        ))

        if crop.width < 8 or crop.height < 8:
            return prev_count if prev_count is not None else 1

        try:
            crop_cv = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2BGR)
            crop_cv = cv2.resize(crop_cv, None, fx=4, fy=4,
                                 interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(crop_cv, cv2.COLOR_BGR2GRAY)

            # 3 preprocessing strategies for robustness
            _, s1 = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
            _, s2 = cv2.threshold(clahe.apply(gray), 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            s3 = cv2.bitwise_not(s1)

            ocr = self._get_ocr()
            candidates = []
            for preprocessed in (s1, s2, s3):
                result = ocr.readtext(preprocessed, allowlist='0123456789',
                                      detail=0, paragraph=False)
                if result:
                    text = ''.join(result).strip().lstrip('xX')
                    if text.isdigit():
                        val = int(text)
                        if val > 0:
                            candidates.append(val)

            if not candidates:
                return prev_count if prev_count is not None else 1

            from collections import Counter as _Counter
            best = _Counter(candidates).most_common(1)[0][0]

            # Monotonic validation (no hardcoded limits):
            # prev_count is the ground truth upper bound.
            if prev_count is not None and prev_count > 0:
                if best > prev_count + 2:
                    return prev_count   # misread — keep last known value
                return min(best, prev_count)

            return best

        except Exception:
            pass

        return prev_count if prev_count is not None else 1

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
