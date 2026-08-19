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

import cv2
import numpy as np
import torch
import torch.nn as nn

IMG_SIZE = 32

# Badge crop geometry — MUST match tools/data/collect_digit_crops.py (the model
# is trained on exactly these crops). Combat = top-right of the icon, prep =
# top-left.
COUNTER_CROP_Y_FRAC = 0.40
BADGE_MARGIN_PX = 4

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


# ── Lecture des nombres dans les WIDGETS d'UI (compteurs, ouvriers, prix) ────
# Différent des badges de troupes : le nombre est centré verticalement (pas de
# bande haute) et le crop contient des ICÔNES BLANCHES (épée, visage d'ouvrier,
# reflet d'une goutte) qui passent le masque "texte blanc" et se font segmenter
# comme des chiffres — c'est ce qui faisait lire "222" au lieu de "1/1".
# Parade : ne garder que le plus grand groupe de glyphes GÉOMÉTRIQUEMENT
# COHÉRENT (même hauteur, même ligne de base) = le nombre ; une icône a une
# autre hauteur et/ou un autre centre vertical, donc elle tombe hors du groupe.
# La boîte du CNN doit RESTER LARGE : c'est l'icône (visage d'ouvrier vs épée)
# qui distingue `nombre_ouvrier` de `place_labo` — deux "N/M" seuls seraient
# visuellement identiques et YOLO ne pourrait plus les séparer. C'est donc ICI
# qu'on isole le nombre, par COMPOSANTES CONNEXES : chaque chiffre est un blob
# compact, alors que le contour du cadre est un grand rectangle quasi vide et
# les icônes/reflets ont des proportions aberrantes.
# Seuils mesurés sur crops réels (village_principal.png) :
#   chiffres        -> remplissage 0.59-0.85, ratio h/w 1.0-2.6
#   contour cadre   -> remplissage 0.08-0.09  (130x60 px pour 666 px allumés)
#   épée / bandeau  -> ratio h/w 0.29-0.64    (large et plat)
#   traits, reflets -> ratio h/w 4.4-13.0     (fin et haut)
WIDGET_MIN_GLYPH_H = 10     # px : ignore le bruit
WIDGET_MIN_FILL = 0.25      # un chiffre remplit sa boîte, un contour non
WIDGET_ASPECT_MIN = 0.80    # écarte les éléments larges et plats
WIDGET_ASPECT_MAX = 5.0     # écarte les traits fins verticaux

WIDGET_GLYPH_H_TOL = 0.15   # écart de hauteur toléré dans un même nombre
WIDGET_GLYPH_C_TOL = 0.25   # écart de centre vertical toléré (× hauteur)

# ⚠️ GARDE-FOU : mieux vaut ne RIEN lire qu'un chiffre faux — un montant erroné
# fausse la décision d'achat. Un nombre correctement segmenté a des glyphes de
# hauteur quasi identique ; dès que ça part en vrille (contour du cadre, icône,
# fond de village capté dans une boîte trop large), on refuse la lecture.
# Observé sur crops réels : boîte serrée -> 7 glyphes tous à h=24 (lecture sûre) ;
# boîte large -> hauteurs 30..92 (à rejeter).
WIDGET_H_CONSISTENCY = 0.15   # tous les glyphes à ±15 % de la hauteur médiane
WIDGET_MIN_CONF = 0.75        # plus strict que les badges de troupes (0.60)


def _glyphs_consistent(glyphs):
    """True si les glyphes ont une hauteur homogène (= un vrai nombre)."""
    if len(glyphs) < 1:
        return False
    hs = sorted(g.shape[0] for g in glyphs)
    med = hs[len(hs) // 2]
    if med <= 0:
        return False
    return all(abs(h - med) <= WIDGET_H_CONSISTENCY * med for h in hs)


# Le texte d'un widget n'est pas toujours BLANC : sur l'écran de confirmation, le
# prix passe en ROUGE quand on n'a pas les moyens (c'est ainsi que le jeu le
# signale). Le masque "texte blanc" (V>165, S<80) est alors aveugle — mesuré sur
# capture réelle : 16 pixels allumés seulement → lecture impossible, statut
# `need_decision` alors que le prix était parfaitement lisible à l'œil.
# Couleur MESURÉE du prix "rouge" sur capture réelle : RGB (240,120,120) →
# HSV **H0 S128 V240** (un saumon, pas un rouge pur), variantes H6-H8.
# ⚠️ Ne pas re-mesurer sur une capture ANNOTÉE : le cadre magenta que dessine
# `UIDetector.annotate` (H≈169, S≈228) se superpose aux chiffres et fausse la
# calibration — piège rencontré en vrai.
# On accepte donc le texte rouge/saumon (teinte aux deux bouts de la roue, bien
# saturé et lumineux). La goutte d'élixir voisine (H≈150) reste dehors, et les
# formes aberrantes sont de toute façon filtrées ensuite.
RED_H_HIGH = 165      # teinte >= : rouge/cramoisi (la roue se referme à 179)
RED_H_LOW = 10        # teinte <= : rouge franc
RED_S_MIN = 120
RED_V_MIN = 150


def _widget_text_mask(bgr):
    """Masque du texte d'un widget : blanc/clair OU rouge (prix non payable)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    white = (v > V_MIN) & (s < S_MAX)
    red = (v > RED_V_MIN) & (s > RED_S_MIN) & ((h >= RED_H_HIGH) | (h <= RED_H_LOW))
    return (white | red).astype(np.uint8)


def _coherent_spans(spans):
    """Plus grand sous-ensemble de spans partageant hauteur + centre vertical."""
    best = []
    for ref in spans:
        rh = max(1, ref[3] - ref[2])
        rc = (ref[2] + ref[3]) / 2
        grp = [s for s in spans
               if abs((s[3] - s[2]) - rh) <= WIDGET_GLYPH_H_TOL * rh
               and abs(((s[2] + s[3]) / 2) - rc) <= WIDGET_GLYPH_C_TOL * rh]
        if len(grp) > len(best):
            best = grp
    return best


def _widget_spans(crop_pil):
    """(spans du nombre, image BGR) — composantes connexes filtrées + cohérence.

    La projection de colonnes (utilisée pour les badges) fusionne tout ce qui
    partage une colonne : dans un widget elle colle le contour du cadre aux
    chiffres. Les composantes connexes isolent chaque glyphe indépendamment.
    """
    rgb = np.array(crop_pil.convert('RGB'))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    # pleine hauteur (pas de TEXT_BAND_FRAC) + texte blanc OU rouge
    mask = _widget_text_mask(bgr)

    n, _labels, stats, _cent = cv2.connectedComponentsWithStats(mask, 8)
    spans = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i][:5])
        if h < WIDGET_MIN_GLYPH_H or w < 2:
            continue
        if area / float(w * h) < WIDGET_MIN_FILL:      # contour de cadre
            continue
        aspect = h / float(w)
        if not (WIDGET_ASPECT_MIN <= aspect <= WIDGET_ASPECT_MAX):
            continue                                    # icône plate / trait fin
        spans.append((x, x + w, y, y + h))

    spans.sort(key=lambda s: s[0])                      # gauche -> droite
    return _coherent_spans(spans), bgr


def segment_widget_glyphs(crop_pil):
    """Glyphes du nombre d'un widget (icônes exclues), de gauche à droite."""
    spans, bgr = _widget_spans(crop_pil)
    if not spans:
        return []
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    out = []
    for x0, x1, y0, y1 in spans:
        gx0, gx1 = max(0, x0 - GLYPH_PAD), min(w, x1 + GLYPH_PAD)
        gy0, gy1 = max(0, y0 - GLYPH_PAD), min(h, y1 + GLYPH_PAD)
        out.append(gray[gy0:gy1, gx0:gx1])
    return out


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


def read_number(crop_pil, drop_leading_x=False, min_conf=0.6):
    """Read pure digits from a crop into (int|None, confidence).

    Segments the crop into digit glyphs, classifies each 0-9, reads left→right.
    `drop_leading_x=True` for troop badges ("xNN"); False for plain numbers
    (ressources, compteur d'ouvriers, prix d'upgrade — pas de 'x' de tête).

    Returns (None, conf) if the model is unavailable, nothing was segmented, or
    the weakest glyph is below min_conf.
    """
    model = _load_model()
    if not model:
        return None, 0.0
    glyphs = segment_glyphs(crop_pil, drop_leading_x=drop_leading_x)
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


def read_count(crop_pil, min_conf=0.6):
    """Read a troop-count badge crop ("xNN") into (int|None, confidence).

    Thin wrapper over read_number that drops the leading 'x' multiplier glyph.
    """
    return read_number(crop_pil, drop_leading_x=True, min_conf=min_conf)


def classify_glyphs(glyphs, min_conf=0.6):
    """Classe une liste de glyphes 0-9 → (list[str] | None, confiance la + faible).

    None si le modèle est indisponible, la liste vide, ou un glyphe sous min_conf.
    """
    model = _load_model()
    if not model or not glyphs:
        return None, 0.0
    xs = torch.stack([_glyph_to_tensor(g) for g in glyphs]).to(_DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(xs), dim=1)
        conf, idx = probs.max(dim=1)
    weakest = float(conf.min().item())
    if weakest < min_conf:
        return None, weakest
    return [_CLASSES[i] for i in idx.tolist()], weakest


def read_widget_number(crop_pil, min_conf=WIDGET_MIN_CONF):
    """Lit l'entier d'un widget d'UI (compteur, prix) → (int|None, confiance).

    Refuse (None) si la segmentation n'est pas homogène — voir WIDGET_H_CONSISTENCY.
    """
    glyphs = segment_widget_glyphs(crop_pil)
    if not _glyphs_consistent(glyphs):
        return None, 0.0
    digits, conf = classify_glyphs(glyphs, min_conf)
    if digits is None:
        return None, conf
    try:
        return int(''.join(digits)), conf
    except ValueError:
        return None, conf


def read_widget_ratio(crop_pil, min_conf=WIDGET_MIN_CONF):
    """Lit un widget "N/M" → ((N, M) | None, confiance).

    Le '/' n'est pas une classe du CNN : on lit le **premier** et le **dernier**
    glyphe du groupe. Valide dans CoC où N et M sont des chiffres uniques
    (max 6 ouvriers, 1 labo) — "5/5", "1/6", "0/1". Même garde-fou d'homogénéité.
    """
    glyphs = segment_widget_glyphs(crop_pil)
    if len(glyphs) < 2 or not _glyphs_consistent(glyphs):
        return None, 0.0
    digits, conf = classify_glyphs([glyphs[0], glyphs[-1]], min_conf)
    if digits is None:
        return None, conf
    try:
        return (int(digits[0]), int(digits[1])), conf
    except ValueError:
        return None, conf


def crop_count_badge(img_pil, bbox, position='combat'):
    """Crop the count badge ("xNN") from an icon bbox. MUST stay identical to
    collect_digit_crops.crop_count_badge (the model trains on these pixels).
    Returns a PIL.Image or None if the box is degenerate."""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    if w < 16 or h < 16:
        return None
    cy2 = y1 + int(h * COUNTER_CROP_Y_FRAC)
    if position == 'prep':
        cx1 = max(0, x1 - BADGE_MARGIN_PX)
        cx2 = min(img_pil.width, x1 + int(w * 0.45) + BADGE_MARGIN_PX)
    else:  # combat
        cx1 = max(0, x1 + int(w * 0.55) - BADGE_MARGIN_PX)
        cx2 = min(img_pil.width, x2 + BADGE_MARGIN_PX)
    cy1 = max(0, y1 - BADGE_MARGIN_PX)
    cy2 = min(img_pil.height, cy2 + BADGE_MARGIN_PX)
    if cx2 - cx1 < 8 or cy2 - cy1 < 8:
        return None
    return img_pil.crop((cx1, cy1, cx2, cy2))


def read_bar_counts(screenshot_pil, detections, position='combat', min_conf=0.6):
    """Read the count of each active troop-bar icon from a full screenshot.

    Returns {class_name: int} only for icons read confidently (>= min_conf).
    Skips grayed/no-tap icons. Caller keeps its own value for anything absent.
    """
    out = {}
    for d in detections:
        if d.get('is_grayed') or d.get('no_tap'):
            continue
        badge = crop_count_badge(screenshot_pil, d['bbox'], position)
        if badge is None:
            continue
        n, _ = read_count(badge, min_conf=min_conf)
        if n is not None and n > 0:
            # SUM duplicates: the same troop/spell can appear twice in the bar
            # (army + clan-castle). They share one logical slot → total = sum.
            out[d['name']] = out.get(d['name'], 0) + n
    return out
