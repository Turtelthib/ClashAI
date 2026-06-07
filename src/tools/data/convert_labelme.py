# scripts/rl/convert_labelme_troops.py
# Converts LabelMe annotations (JSON) to YOLO format (txt).
#
# LabelMe saves one .json file per image with polygons/rectangles.
# YOLO expects one .txt file per image with:
# class x_center y_center width height (normalized 0-1)
#
# Usage:
# python scripts/rl/convert_labelme_troops.py
# python scripts/rl/convert_labelme_troops.py --input combat_captures --output dataset_troops
# python scripts/rl/convert_labelme_troops.py --split 0.8

import os
import json
import shutil
import random
import argparse
from pathlib import Path

from clashai.paths import PROJECT_ROOT as project_root


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_INPUT_DIR = os.path.join(project_root, 'combat_captures')
DEFAULT_OUTPUT_DIR = os.path.join(project_root, 'dataset_troops')
DEFAULT_TRAIN_SPLIT = 0.8

# Class name → YOLO ID mapping
# MUST match coc_troops.yaml exactly
CLASS_MAP = {
    'golem': 0,
    'sorcier': 1,
    'sorciere': 2,
    'pekka': 3,
    'archere': 4,
    'lance_buche': 5,
    'roi': 6,
    'reine': 7,
    'grand_gardien': 8,
    'championne': 9,
    'demolisseur': 10,
    'bouliste': 11,
    'prince_gargouille': 12,
}

# Aliases (to tolerate naming variations)
ALIASES = {
    'Golem': 'golem',
    'GOLEM': 'golem',
    'Sorcier': 'sorcier',
    'wizard': 'sorcier',
    'Sorciere': 'sorciere',
    'sorcière': 'sorciere',
    'Sorcière': 'sorciere',
    'witch': 'sorciere',
    'Pekka': 'pekka',
    'PEKKA': 'pekka',
    'P.E.K.K.A': 'pekka',
    'Archere': 'archere',
    'archère': 'archere',
    'archer': 'archere',
    'Lance_buche': 'lance_buche',
    'lance_bûche': 'lance_buche',
    'log_launcher': 'lance_buche',
    'siege': 'lance_buche',
    'Roi': 'roi',
    'king': 'roi',
    'barbarian_king': 'roi',
    'Reine': 'reine',
    'queen': 'reine',
    'archer_queen': 'reine',
    'Grand_gardien': 'grand_gardien',
    'grand_warden': 'grand_gardien',
    'warden': 'grand_gardien',
    'GG': 'grand_gardien',
    'Championne': 'championne',
    'royal_champion': 'championne',
    'champion': 'championne',
    'RC': 'championne',
}


# =============================================================================
# CONVERSION
# =============================================================================

def convert_labelme_to_yolo(json_path, img_width, img_height):
    """
    Converts a LabelMe JSON file to YOLO lines.

    Returns:
        lines: list of strings in format "class x_center y_center w h"
        stats: dict {class: count}
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    lines = []
    stats = {}

    for shape in data.get('shapes', []):
        label = shape['label']

        # Resolve aliases
        if label in ALIASES:
            label = ALIASES[label]

        if label not in CLASS_MAP:
            print(f" WARNING: Classe inconnue ignorée : '{shape['label']}' "
                  f"dans {json_path}")
            continue

        class_id = CLASS_MAP[label]
        points = shape['points']

        if shape['shape_type'] == 'rectangle' or len(points) == 2:
            # Rectangle: 2 points (top-left corner, bottom-right corner)
            x1 = min(points[0][0], points[1][0])
            y1 = min(points[0][1], points[1][1])
            x2 = max(points[0][0], points[1][0])
            y2 = max(points[0][1], points[1][1])
        elif len(points) >= 3:
            # Polygon: take the enclosing bounding box
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            x1, y1 = min(xs), min(ys)
            x2, y2 = max(xs), max(ys)
        else:
            continue

        # Normalize (0-1)
        x_center = ((x1 + x2) / 2) / img_width
        y_center = ((y1 + y2) / 2) / img_height
        w = (x2 - x1) / img_width
        h = (y2 - y1) / img_height

        # Clamp
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        w = max(0.001, min(1, w))
        h = max(0.001, min(1, h))

        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
        stats[label] = stats.get(label, 0) + 1

    return lines, stats


def process_dataset(input_dir, output_dir, train_split=DEFAULT_TRAIN_SPLIT):
    """
    Converts the entire LabelMe folder to a YOLO dataset.
    """
    print(f"\n{'='*60}")
    print(" Conversion LabelMe -> YOLO")
    print(f"{'='*60}")
    print(f" Input : {input_dir}")
    print(f" Output : {output_dir}")
    print(f" Split : {train_split:.0%} train / {1-train_split:.0%} val")
    print(f"{'='*60}\n")

    if not os.path.exists(input_dir):
        print(f"ERROR: Dossier d'entrée introuvable : {input_dir}")
        return

    # Find all JSON files
    json_files = sorted(Path(input_dir).glob('*.json'))
    if not json_files:
        print(f"ERROR: Aucun fichier .json trouvé dans {input_dir}")
        return

    print(f" {len(json_files)} fichiers JSON trouvés")

    # Create output structure
    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

    # Shuffle and split
    random.seed(42)
    indices = list(range(len(json_files)))
    random.shuffle(indices)
    split_idx = int(len(indices) * train_split)
    train_indices = set(indices[:split_idx])

    total_stats = {}
    converted = 0
    skipped = 0

    for i, json_path in enumerate(json_files):
        # Find the corresponding image
        stem = json_path.stem
        img_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            candidate = json_path.with_suffix(ext)
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            print(f" WARNING: Image manquante pour {json_path.name}")
            skipped += 1
            continue

        # Read image dimensions
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        img_w = data.get('imageWidth')
        img_h = data.get('imageHeight')

        if not img_w or not img_h:
            from PIL import Image
            img = Image.open(img_path)
            img_w, img_h = img.size

        # Convert
        lines, stats = convert_labelme_to_yolo(json_path, img_w, img_h)

        if not lines:
            skipped += 1
            continue

        # Determine train or val
        split = 'train' if i in train_indices else 'val'

        # Copy image
        dst_img = os.path.join(output_dir, 'images', split, img_path.name)
        shutil.copy2(img_path, dst_img)

        # Write YOLO label
        label_name = stem + '.txt'
        dst_label = os.path.join(output_dir, 'labels', split, label_name)
        with open(dst_label, 'w') as f:
            f.write('\n'.join(lines) + '\n')

        # Stats
        for cls, count in stats.items():
            total_stats[cls] = total_stats.get(cls, 0) + count
        converted += 1

    # Summary
    print(f"\n{'='*60}")
    print(" Conversion terminée")
    print(f"{'='*60}")
    print(f" Convertis : {converted}")
    print(f" Ignorés : {skipped}")

    train_count = len(os.listdir(os.path.join(output_dir, 'images', 'train')))
    val_count = len(os.listdir(os.path.join(output_dir, 'images', 'val')))
    print(f" Train : {train_count} images")
    print(f" Val : {val_count} images")

    print("\n Annotations par classe :")
    for cls in sorted(total_stats.keys()):
        count = total_stats[cls]
        bar = '' * min(count // 5, 40)
        print(f" {cls:20s} : {count:5d} {bar}")

    total_annotations = sum(total_stats.values())
    print(f" {'TOTAL':20s} : {total_annotations:5d}")

    # Check for missing classes
    missing = set(CLASS_MAP.keys()) - set(total_stats.keys())
    if missing:
        print(f"\n WARNING: Classes sans annotation : {sorted(missing)}")
        print(" Assure-toi d'annoter ces troupes !")

    print("\n Prochaine étape :")
    print(" python scripts/rl/train_yolo_troops.py --data coc_troops.yaml")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Conversion LabelMe -> YOLO pour les troupes"
    )
    parser.add_argument('--input', type=str, default=DEFAULT_INPUT_DIR,
                        help="Dossier avec les images + JSON LabelMe")
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Dossier de sortie YOLO")
    parser.add_argument('--split', type=float, default=DEFAULT_TRAIN_SPLIT,
                        help="Train/total ratio (default: 0.8)")

    args = parser.parse_args()

    process_dataset(
        input_dir=args.input,
        output_dir=args.output,
        train_split=args.split,
    )
