# clashai/perception/digit_reader.py
# Read a small troop-count badge ("x12") into an int — B2 approach.
#
# The badge is white text ("x" multiplier + the digits) over a colored slot.
# We:
#   1. isolate the white text (bright, low-saturation),
#   2. split it into glyph columns (vertical projection profile),
#   3. drop the leading 'x' glyph,
#   4. classify each remaining glyph 0-9 with a shared CNN,
#   5. read left -> right and concatenate.
#
# Shared by BOTH training-data generation (tools/data/build_digit_singles.py)
# and inference (TroopBarDetector._read_count) so the model sees the exact same
# glyph crops it was trained on. The classifier is optional: segment_glyphs()
# works standalone (used to build the per-digit dataset).

import numpy as np
import cv2
import torch
import torch.nn as nn

IMG_SIZE = 32

# Glyph segmentation tuning (relative to the crop, so size-independent).
TEXT_BAND_FRAC = 0.62      # text "xNN" sits in the upper band; ignore troop art below
V_MIN = 165                # white text: high value (brightness)
S_MAX = 80                 # white text: low saturation
COL_ON_FRAC = 0.16         # a column is "on" if its white-pixel sum > this * max
MIN_GLYPH_W = 3            # ignore spans thinner than this (noise)
GLYPH_MIN_H_FRAC = 0.5     # drop spans shorter than this * tallest span (art/noise)
GLYPH_PAD = 1


def _white_mask(band_bgr):
    """Binary mask of white text pixels (bright + desaturated)."""
    hsv = cv2.cvtColor(band_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]
    return ((v > V_MIN) & (s < S_MAX)).astype(np.uint8)


def _column_spans(mask):
    """Group consecutive 'on' columns into (x0, x1) glyph spans."""
    col = mask.sum(axis=0)
    if col.max() == 0:
        return []
    thr = col.max() * COL_ON_FRAC
    on = col > thr
    spans, start = [], None
    for x, v in enumerate(on):
        if v and start is None:
            start = x
        elif not v and start is not None:
            if x - start >= MIN_GLYPH_W:
                spans.append((start, x))
            start = None
    if start is not None and len(on) - start >= MIN_GLYPH_W:
        spans.append((start, len(on)))
    return spans


def segment_glyphs(crop_pil, drop_leading_x=True, return_mask=False):
    """Split a badge crop into individual digit glyph images (left -> right).

    Returns a list of grayscale np.uint8 glyph crops (the 'x' dropped). Empty
    list if nothing readable. With return_mask=True, also returns (mask, spans)
    for debugging.
    """
    rgb = np.array(crop_pil.convert('RGB'))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    band = bgr[0:max(1, int(h * TEXT_BAND_FRAC)), :]
    mask = _white_mask(band)

    # (x0, x1, y0, y1) per column span, with its tight vertical extent.
    spans = []
    for x0, x1 in _column_spans(mask):
        rows = np.where(mask[:, x0:x1].sum(axis=1) > 0)[0]
        if len(rows):
            spans.append((x0, x1, int(rows[0]), int(rows[-1]) + 1))

    # Drop short spans: digit glyphs span most of the text height; troop-art /
    # noise blobs are shorter. This is the main fix for a single '1' being
    # over-split into '12'/'121'.
    if spans:
        max_h = max(y1 - y0 for _, _, y0, y1 in spans)
        spans = [s for s in spans if (s[3] - s[2]) >= GLYPH_MIN_H_FRAC * max_h]

    if drop_leading_x and spans:
        spans = spans[1:]   # the leftmost glyph is the 'x' multiplier

    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    glyphs = []
    for x0, x1, y0, y1 in spans:
        gx0 = max(0, x0 - GLYPH_PAD)
        gx1 = min(band.shape[1], x1 + GLYPH_PAD)
        gy0 = max(0, y0 - GLYPH_PAD)
        gy1 = min(band.shape[0], y1 + GLYPH_PAD)
        glyphs.append(gray[gy0:gy1, gx0:gx1])

    if return_mask:
        return glyphs, mask, spans
    return glyphs


# =============================================================================
# CLASSIFIER (per-digit 0-9) — shared by train (tools/train/train_digit_cnn.py)
# and inference (read_count below / TroopBarDetector._read_count).
# =============================================================================

class DigitCNN(nn.Module):
    """LeNet-ish: ~60k params, plenty for 32x32 grayscale single-digit glyphs."""

    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 4 * 4, 128), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


_MODEL = None      # None = not loaded yet, False = unavailable, else the model
_CLASSES = None
_DEVICE = None


def _load_model():
    """Lazy-load weights/digit_cnn.pt (singleton). Returns the model or False."""
    global _MODEL, _CLASSES, _DEVICE
    if _MODEL is not None:
        return _MODEL
    import os
    from clashai.paths import WEIGHTS_DIR
    path = os.path.join(WEIGHTS_DIR, 'digit_cnn.pt')
    if not os.path.exists(path):
        _MODEL = False
        return False
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        _CLASSES = ckpt['classes']
        _DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        m = DigitCNN(len(_CLASSES)).to(_DEVICE)
        m.load_state_dict(ckpt['state_dict'])
        m.eval()
        _MODEL = m
    except Exception:
        _MODEL = False
    return _MODEL


def _glyph_to_tensor(glyph_gray):
    g = cv2.resize(glyph_gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return torch.from_numpy((g.astype(np.float32) / 255.0)).unsqueeze(0)  # (1,32,32)


def read_count(crop_pil, min_conf=0.6):
    """Read a troop-count badge crop into (int|None, confidence).

    Segments the badge into digit glyphs, classifies each 0-9, reads left→right.
    Returns (None, conf) if the model is unavailable, nothing was segmented, or
    the weakest glyph is below min_conf → the caller should fall back to EasyOCR.
    """
    model = _load_model()
    if not model:
        return None, 0.0
    glyphs = segment_glyphs(crop_pil)
    if not glyphs:
        return None, 0.0
    xs = torch.stack([_glyph_to_tensor(g) for g in glyphs]).to(_DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(xs), dim=1)
        conf, idx = probs.max(dim=1)
    digits = [_CLASSES[i] for i in idx.tolist()]
    weakest = float(conf.min().item())
    if weakest < min_conf:
        return None, weakest
    try:
        return int(''.join(digits)), weakest
    except ValueError:
        return None, weakest
