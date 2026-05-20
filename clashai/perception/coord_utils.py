# clashai/perception/coord_utils.py
# Centralised coordinate scaling between the canonical 1920x1080 game frame
# (ADB native / what models / button positions / YOLO bboxes are calibrated
# on) and the actual pixel size of a captured image (which can vary with
# WGC / DPI / window size).
#
# Replaces ~5 ad-hoc `scale_x = w / 1920` / `scale_y = h / 1080` patterns
# scattered across `environment_v4`, `debug_overlay`, `test_run_capture`,
# `troop_counter`, `perception_thread`.

from clashai.config import SCREEN_WIDTH, SCREEN_HEIGHT


class ImageScaler:
    """
    Scaler between canonical (1920x1080) and image pixel coordinates.

    Construction accepts a numpy array, PIL.Image, or (height, width) tuple.

    Usage:
        scaler = ImageScaler(img_cv)            # cv2/numpy
        scaler = ImageScaler(img_pil)           # PIL
        scaler = ImageScaler((1080, 1920))      # raw shape

        # canonical → image space (e.g. draw a 1920x1080-calibrated point
        # onto an image of arbitrary size):
        ix, iy = scaler.to_img(960, 540)

        # image → canonical space (e.g. interpret a YOLO bbox detected on
        # an image and remap to canonical coordinates):
        cx, cy = scaler.to_canonical(ix, iy)
    """

    __slots__ = ('img_w', 'img_h', 'sx', 'sy')

    def __init__(self, img_or_shape):
        if hasattr(img_or_shape, 'shape'):           # numpy array (HxWxC)
            h, w = img_or_shape.shape[:2]
        elif hasattr(img_or_shape, 'size'):          # PIL.Image (w, h)
            w, h = img_or_shape.size
        else:                                         # (height, width) tuple
            h, w = img_or_shape

        self.img_w = w
        self.img_h = h
        # sx, sy = factors that go from canonical → image
        self.sx = w / SCREEN_WIDTH
        self.sy = h / SCREEN_HEIGHT

    # ---- canonical → image ----------------------------------------------

    def to_img(self, x, y):
        """Map a (x, y) point from canonical 1920x1080 → image pixels."""
        return int(x * self.sx), int(y * self.sy)

    def to_img_x(self, x):
        return int(x * self.sx)

    def to_img_y(self, y):
        return int(y * self.sy)

    # ---- image → canonical ----------------------------------------------

    def to_canonical(self, x, y):
        """Map a (x, y) point from image pixels → canonical 1920x1080."""
        if self.sx == 0 or self.sy == 0:
            return x, y
        return int(x / self.sx), int(y / self.sy)

    def to_canonical_x(self, x):
        return int(x / self.sx) if self.sx else x

    def to_canonical_y(self, y):
        return int(y / self.sy) if self.sy else y

    # ---- convenience -----------------------------------------------------

    def is_identity(self) -> bool:
        """True if the image is already at canonical resolution (1.0 scale)."""
        return self.sx == 1.0 and self.sy == 1.0
