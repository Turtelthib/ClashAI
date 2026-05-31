# clashai/perception/troop_detector.py
# Mid-combat troop detection via YOLO.
#
# Wrapper around the troop YOLO model (13 classes, mAP50=0.987).
# Provides structured detections usable by CombatObserver
# and SpellCaster.
#
# Usage:
# detector = TroopDetector()
# detections = detector.detect(screenshot_pil)
# # detections = list[Detection(class_name, x, y, w, h, conf)]

import os
from dataclasses import dataclass
from typing import Optional

from clashai.paths import WEIGHTS_DIR


# =============================================================================
# CONFIGURATION
# =============================================================================

YOLO_TROOPS_PATH = os.path.join(WEIGHTS_DIR, 'yolo_troops.pt')

# Troop YOLO model classes (order = index)
TROOP_CLASSES = [
    'golem', 'sorcier', 'sorciere', 'pekka', 'archere',
    'lance_buche', 'roi', 'reine', 'grand_gardien', 'championne',
    'demolisseur', 'bouliste', 'prince_gargouille',
]

# Categories for grouping
HERO_CLASSES = {'roi', 'reine', 'grand_gardien', 'championne', 'prince_gargouille'}
SIEGE_CLASSES = {'lance_buche', 'demolisseur', 'bouliste'}
TROOP_CLASSES_SET = set(TROOP_CLASSES) - HERO_CLASSES - SIEGE_CLASSES

# Confidence threshold
DEFAULT_CONF = 0.35
# YOLO combat troops trained at imgsz=640 (see tools/train/train_yolo_troops.py
# DEFAULT_IMG_SIZE). Set explicitly so a future retrain at 1280/1600 only
# requires updating this constant.
YOLO_TROOPS_IMGSZ = 640

# Re-imported from clashai/config/screen.py (Phase A).
from clashai.config import ADB_WIDTH, ADB_HEIGHT  # noqa: E402

# UI exclusion zone (do not detect in the troop bar / header)
UI_TOP_Y_RATIO = 0.06
UI_BOTTOM_Y_RATIO = 0.82


# =============================================================================
# DATACLASS
# =============================================================================

@dataclass
class Detection:
    """A YOLO troop detection."""
    class_name: str
    class_id: int
    x: int
    y: int
    w: int
    h: int
    conf: float

    @property
    def is_hero(self) -> bool:
        return self.class_name in HERO_CLASSES

    @property
    def is_siege(self) -> bool:
        return self.class_name in SIEGE_CLASSES

    @property
    def is_troop(self) -> bool:
        return self.class_name in TROOP_CLASSES_SET


# =============================================================================
# TROOP DETECTOR
# =============================================================================

class TroopDetector:
    """
    Mid-combat troop detector based on YOLO.

    Loads the troop YOLO model and provides structured detections
    with UI zone filtering.
    """

    def __init__(self, weights_path: Optional[str] = None, conf: float = DEFAULT_CONF,
                 verbose: bool = True):
        self.conf = conf
        self.verbose = verbose
        self._model = None
        self._weights_path = weights_path or YOLO_TROOPS_PATH

    def _load_model(self):
        """Loads the YOLO model (lazy loading)."""
        if self._model is not None:
            return

        if not os.path.exists(self._weights_path):
            raise FileNotFoundError(
                f"Modèle YOLO troupes introuvable : {self._weights_path}\n"
                f"Entraîne-le avec : python tools/train/train_yolo_troops.py"
            )

        from ultralytics import YOLO
        self._model = YOLO(self._weights_path)
        if self.verbose:
            from clashai.config.logging import pp
            pp(f" YOLO troupes chargé : {self._weights_path}", tag='yolo')

    def detect(self, screenshot_pil, filter_ui: bool = True) -> list[Detection]:
        """
        Detects troops in a screenshot.

        Args:
            screenshot_pil: PIL Image (RGB)
            filter_ui: filter detections in the UI zone

        Returns:
            List of Detection sorted by decreasing confidence
        """
        self._load_model()

        img_w, img_h = screenshot_pil.size
        scale_x = ADB_WIDTH / img_w
        scale_y = ADB_HEIGHT / img_h

        from clashai.perception.inference_lock import INFERENCE_LOCK
        with INFERENCE_LOCK:
            results = self._model(
                screenshot_pil, conf=self.conf,
                imgsz=YOLO_TROOPS_IMGSZ, verbose=False,
            )
        detections = []

        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # Center in ADB coordinates
                cx = int((x1 + x2) / 2 * scale_x)
                cy = int((y1 + y2) / 2 * scale_y)
                w = int((x2 - x1) * scale_x)
                h = int((y2 - y1) * scale_y)

                # Filter UI zone
                if filter_ui:
                    y_ratio = cy / ADB_HEIGHT
                    if y_ratio < UI_TOP_Y_RATIO or y_ratio > UI_BOTTOM_Y_RATIO:
                        continue

                class_name = TROOP_CLASSES[cls_id] if cls_id < len(TROOP_CLASSES) else f"unk_{cls_id}"

                detections.append(Detection(
                    class_name=class_name,
                    class_id=cls_id,
                    x=cx, y=cy, w=w, h=h,
                    conf=conf,
                ))

        detections.sort(key=lambda d: d.conf, reverse=True)

        if self.verbose and detections:
            from clashai.config.logging import pp, styled
            counts = {}
            for d in detections:
                counts[d.class_name] = counts.get(d.class_name, 0) + 1
            summary = ', '.join(f"{v}×{k}" for k, v in counts.items())
            pp(f" YOLO troupes: {styled(summary, 'yolo_alt')}", tag='yolo')

        return detections

    def detect_grouped(self, screenshot_pil) -> dict:
        """
        Detects and groups troops by category.

        Returns:
            dict with keys 'troops', 'heroes', 'sieges', 'all',
            each value being a list of Detection.
        """
        all_dets = self.detect(screenshot_pil)
        return {
            'troops': [d for d in all_dets if d.is_troop],
            'heroes': [d for d in all_dets if d.is_hero],
            'sieges': [d for d in all_dets if d.is_siege],
            'all': all_dets,
        }

    def get_positions(self, screenshot_pil, class_filter: set = None) -> list[tuple[int, int]]:
        """
        Returns the ADB (x, y) positions of detected troops.

        Args:
            class_filter: set of class names to include (None = all)
        """
        dets = self.detect(screenshot_pil)
        if class_filter:
            dets = [d for d in dets if d.class_name in class_filter]
        return [(d.x, d.y) for d in dets]

    def count_by_class(self, screenshot_pil) -> dict[str, int]:
        """Counts the number of detected troops per class."""
        dets = self.detect(screenshot_pil)
        counts = {}
        for d in dets:
            counts[d.class_name] = counts.get(d.class_name, 0) + 1
        return counts


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    import sys
    from PIL import Image

    print("Test TroopDetector\n")

    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        img = Image.open(img_path)
        detector = TroopDetector()
        grouped = detector.detect_grouped(img)
        print("\nRésultats:")
        print(f" Troupes : {len(grouped['troops'])}")
        print(f" Héros : {len(grouped['heroes'])}")
        print(f" Sièges : {len(grouped['sieges'])}")
        for d in grouped['all']:
            print(f" {d.class_name:20s} ({d.x:4d}, {d.y:4d}) conf={d.conf:.2f}")
    else:
        print("Usage: python -m clashai.perception.troop_detector <screenshot.png>")
