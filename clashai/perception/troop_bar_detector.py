# clashai/perception/troop_bar_detector.py
# YOLO-based troop bar detector — replaces TroopFinder template matching.
#
# Detects all troop/spell/hero/siege icons in the bottom bar via YOLO,
# then applies an HSV saturation filter to distinguish active vs grayed icons.
#
# Interface compatible with TroopFinder.positions:
#   positions = {name: (x, y, conf)}  — only active (non-grayed) slots
#
# Also provides raw detections with is_grayed flag for full info.

import os
import cv2
import numpy as np
from PIL import Image

# HSV saturation below this threshold = icon is grayed out (depleted/upgrading)
GRAYED_SAT_THRESHOLD = 30

# Confidence threshold for YOLO detection
YOLO_CONF = 0.30

# Classes that should NEVER be tapped even if detected (deployed siege machines)
NO_TAP_CLASSES = {
    'demolisseur_deploye', 'dirigeable_deploye', 'broyeur_pierre_deploye',
    'caserne_siege_deploye', 'lance_buche_deploye', 'catapulte_deploye',
    'foreuse_deploye', 'lance_troupe_deploye', 'fourgon_celeste_deploye',
}


class TroopBarDetector:
    """
    YOLO-based troop bar icon detector.

    Replaces TroopFinder's template matching with a single YOLO inference.
    Faster, more robust, works with any army composition.
    """

    def __init__(self, model_path, verbose=False):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.verbose = verbose
        self._last_detections = []
        print(f" TroopBarDetector loaded: {len(self.model.names)} classes")

    def detect(self, screenshot_pil):
        """
        Detects all icons in the troop bar.

        Args:
            screenshot_pil: PIL.Image of the full screen (any resolution)

        Returns:
            List of dicts:
            {
                'name':      str   — class name (e.g. 'golem', 'soin', 'roi_capa')
                'bbox':      (x1, y1, x2, y2)  — pixel coords in the image
                'center':    (cx, cy)
                'conf':      float
                'is_grayed': bool  — True if slot is depleted/unavailable
                'no_tap':    bool  — True if clicking would be destructive
            }
        """
        img_arr = np.array(screenshot_pil)
        results = self.model.predict(img_arr, conf=YOLO_CONF, verbose=False)

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

            # HSV saturation check on the icon crop
            crop = screenshot_pil.crop((x1, y1, x2, y2))
            is_grayed = self._is_grayed(crop)

            detections.append({
                'name':      name,
                'bbox':      (x1, y1, x2, y2),
                'center':    (cx, cy),
                'conf':      conf,
                'is_grayed': is_grayed,
                'no_tap':    name in NO_TAP_CLASSES,
            })

        self._last_detections = detections

        if self.verbose:
            active = [d for d in detections if not d['is_grayed'] and not d['no_tap']]
            grayed = [d for d in detections if d['is_grayed']]
            no_tap = [d for d in detections if d['no_tap']]
            print(f" TroopBar: {len(active)} active, {len(grayed)} grayed, "
                  f"{len(no_tap)} no-tap ({len(detections)} total)")

        return detections

    def to_positions(self, detections=None):
        """
        Converts detections to TroopFinder.positions format:
            {name: (x, y, conf)}

        Only includes ACTIVE (non-grayed, non-deploye) icons.
        Grayed and _deploye icons are excluded so the agent never taps them.
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
