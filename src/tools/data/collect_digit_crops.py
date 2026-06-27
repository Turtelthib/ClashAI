# tools/data/collect_digit_crops.py
# Phase F.3 step 1 : walk every logs/episode_*/ frame, run YOLO troop bar
# on it, crop the count badge for each (countable) detection, and save
# the crop to needLabelisation/digits/. The labelisation pass (manual or
# semi-auto) comes next.
#
# Run:
#   uv run python tools/data/collect_digit_crops.py
#   uv run python tools/data/collect_digit_crops.py --logs-dir logs --out needLabelisation/digits --limit 200
#
# What gets cropped:
#   For each YOLO troop bar detection that is NOT in NO_COUNTER_CLASSES
#   nor in UNIQUE_HEROES (heroes are always 1, no counter to read), we
#   crop the badge at the position corresponding to the SOURCE FRAME's
#   screen state:
#     - prep_attaque   → top-LEFT badge (army selection screen)
#     - phase_attaque  → top-RIGHT badge (battle bar during combat)
#     - other screens  → skipped (no relevant badge)
#
#   With --position both, we ignore the screen state and emit both
#   crops — useful for max recall during data collection, but you'll
#   need to filter the empty ones during labelisation.
#
# Output filenames:
#   <class>_<frameid>_<bbox_hash>_<position>.png
#   e.g. golem_ep0042_t30s_2f8b_combat.png
#
# Idempotent: if a frame has already been processed (filename collision),
# the new crop overwrites — that's fine, deterministic.

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

from PIL import Image


# Crop box matches troop_bar_detector._read_count() exactly so the model
# is trained on the same pixels the inference step will see.
COUNTER_CROP_Y_FRAC = 0.40
MARGIN_PX = 4


def crop_count_badge(img_pil, bbox, position='combat'):
    """
    Reproduce the badge crop logic from
    clashai/perception/troop_bar_detector.py::_read_count.

    position : 'prep'   → top-LEFT  corner (army selection screen)
               'combat' → top-RIGHT corner (battle bar)

    Returns a PIL.Image (cropped) or None if the resulting box is degenerate.
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    if w < 16 or h < 16:
        return None

    cy2 = y1 + int(h * COUNTER_CROP_Y_FRAC)

    if position == 'prep':
        cx1 = max(0, x1 - MARGIN_PX)
        cx2 = min(img_pil.width, x1 + int(w * 0.45) + MARGIN_PX)
    else:  # combat
        cx1 = max(0, x1 + int(w * 0.55) - MARGIN_PX)
        cx2 = min(img_pil.width, x2 + MARGIN_PX)

    cy1 = max(0, y1 - MARGIN_PX)
    cy2 = min(img_pil.height, cy2 + MARGIN_PX)

    if cx2 - cx1 < 8 or cy2 - cy1 < 8:
        return None
    return img_pil.crop((cx1, cy1, cx2, cy2))


def short_hash(*items, n=4):
    """4-char hash of the joined items, for unique filenames."""
    h = hashlib.md5('|'.join(str(i) for i in items).encode()).hexdigest()
    return h[:n]


def safe_classname(name):
    return ''.join(c if c.isalnum() else '_' for c in name)[:30]


def walk_log_frames(logs_dir):
    """Yield every (episode_label, frame_path) under logs_dir.

    Looks at logs/episode_*/ and yields each .jpg/.png inside.
    """
    base = Path(logs_dir)
    if not base.exists():
        return
    for ep_dir in sorted(base.glob('episode_*')):
        ep_label = ep_dir.name
        for img_path in sorted(ep_dir.glob('*.jpg')) + sorted(ep_dir.glob('*.png')):
            yield ep_label, img_path
    # Also pick up test_run/ if present
    test_run = base / 'test_run'
    if test_run.exists():
        for img_path in sorted(test_run.glob('*.png')):
            yield 'test_run', img_path
    # And the accumulating prep_attaque captures (richest: full counts).
    digit_frames = base / 'digit_frames'
    if digit_frames.exists():
        for img_path in sorted(digit_frames.glob('*.png')):
            yield 'prep', img_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logs-dir', default='logs',
                    help="Directory to walk for episode_*/ subfolders")
    ap.add_argument('--out', default='needLabelisation/digits',
                    help="Where to save the cropped badges")
    ap.add_argument('--limit', type=int, default=0,
                    help="Stop after N frames (0 = all)")
    ap.add_argument('--position', choices=('auto', 'both', 'prep', 'combat'),
                    default='auto',
                    help="Which badge to crop. 'auto' = classify each "
                         "frame's screen state and pick the matching "
                         "position (recommended). 'both' = emit both "
                         "crops regardless (lots of junk to filter).")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Lazy-import the detector so the script can be inspected without GPU.
    from clashai.navigation import game_loop
    from clashai.perception.troop_bar_detector import (
        NO_COUNTER_CLASSES, UNIQUE_HEROES,
    )

    print(f" Loading TroopBarDetector...")
    models = game_loop.load_models()
    bar = models.get('troop_bar_detector')
    if bar is None:
        print("ERROR: TroopBarDetector not loaded. Check weights/yolo_troupes_barre/troop_bar.pt")
        sys.exit(1)
    print(f" → loaded ({len(bar.model.names)} classes)")

    skip_classes = NO_COUNTER_CLASSES | UNIQUE_HEROES

    # Map screen state → which crop position the badge is at.
    SCREEN_TO_POSITION = {
        'prep_attaque':  'prep',
        'phase_attaque': 'combat',
    }

    n_frames = 0
    n_classified = 0
    n_crops = 0
    n_skipped = 0
    n_skipped_screen = 0
    t0 = time.time()

    try:
        for ep_label, img_path in walk_log_frames(args.logs_dir):
            n_frames += 1
            if args.limit and n_frames > args.limit:
                break
            try:
                img = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"  skipping {img_path}: {e}")
                continue

            # Decide which crop position(s) to emit
            if args.position == 'auto':
                from clashai.navigation.game_loop import classify_screen
                screen, conf = classify_screen(img, models)
                n_classified += 1
                pos = SCREEN_TO_POSITION.get(screen)
                if pos is None or conf < 0.55:
                    # Frame isn't a prep/phase_attaque screen → no badge to read
                    n_skipped_screen += 1
                    continue
                positions = (pos,)
            elif args.position == 'both':
                positions = ('prep', 'combat')
            else:
                positions = (args.position,)

            detections = bar.detect(img, screen='combat')  # screen kw only affects OCR count, we ignore
            for d in detections:
                name = d['name']
                if name in skip_classes or d['is_grayed'] or d['no_tap']:
                    n_skipped += 1
                    continue
                bbox = d['bbox']
                frame_id = f"{ep_label}_{img_path.stem}"
                bbox_h = short_hash(*bbox)
                for pos in positions:
                    crop = crop_count_badge(img, bbox, position=pos)
                    if crop is None:
                        continue
                    fname = f"{safe_classname(name)}_{frame_id}_{bbox_h}_{pos}.png"
                    crop.save(os.path.join(args.out, fname))
                    n_crops += 1

            if n_frames % 10 == 0:
                elapsed = time.time() - t0
                rate = n_frames / elapsed if elapsed > 0 else 0
                print(f"  {n_frames} frames | {n_crops} crops | "
                      f"skipped {n_skipped} | {rate:.1f} f/s")

    except KeyboardInterrupt:
        print("\n Interrupted.")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f" Done.")
    print(f" Frames processed       : {n_frames}")
    print(f" Frames classified      : {n_classified}")
    print(f" Frames skipped (screen): {n_skipped_screen}  (not prep_attaque or phase_attaque)")
    print(f" Crops saved            : {n_crops}  → {os.path.abspath(args.out)}")
    print(f" Detections skipped     : {n_skipped}  (heroes / abilities / siege)")
    print(f" Time                   : {elapsed:.1f}s")
    print(f"{'='*60}")
    print(f"\n Next step: label each crop manually (or write a labelling")
    print(f" helper that pre-fills with EasyOCR's guess).")
    print(f" Suggested layout: needLabelisation/digits/<count>/<file>.png")


if __name__ == '__main__':
    main()
