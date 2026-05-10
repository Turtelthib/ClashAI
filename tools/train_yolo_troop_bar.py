# tools/train_yolo_troop_bar.py
# ============================================================
# ClashAI — YOLO Troop Bar Training (Kaggle-ready)
# ============================================================
# Paste this script directly into a Kaggle notebook cell.
# GPU: recommended T4 x2 or P100
#
# 78 classes:
#   Troops (31), Spells (17), Siege available (9),
#   Siege deployed (9), Heroes (6), Hero abilities (6)
#
# Grayed state detected by HSV saturation post-processing,
# not as separate classes (except siege _deploye which has
# destructive click consequence).
# ============================================================

# ── 1. Install dependencies ──────────────────────────────────
import subprocess
subprocess.run(['pip', 'install', 'ultralytics', '-q'], check=True)

# ── 2. Dataset — uploaded directly to Kaggle ─────────────────
# On Kaggle: + Add Data → upload your Roboflow zip → it appears at /kaggle/input/<dataset-name>/
# Find data.yaml path with: !find /kaggle/input -name 'data.yaml'
DATA_YAML = '/kaggle/input/YOUR_DATASET_NAME/data.yaml'  # adjust to your dataset name
print(f'Dataset: {DATA_YAML}')

# ── 3. Classes — read from data.yaml (Roboflow generates this) ─
# Class names and order come from data.yaml automatically.
# YOLO ignores any CLASSES list here — nothing to configure.
import yaml
with open(DATA_YAML) as f:
    meta = yaml.safe_load(f)
CLASSES = meta.get('names', [])
print(f'Classes from data.yaml: {len(CLASSES)} → {CLASSES}')

# ── 4. Training ───────────────────────────────────────────────
from ultralytics import YOLO
import shutil, os

MODEL      = 'yolo26m.pt'
EPOCHS     = 150
BATCH      = 16             # adjust if OOM: try 8
IMG_SIZE   = 1600
OUTPUT_DIR = '/kaggle/working/yolo_troop_bar'

yolo = YOLO(MODEL)

results = yolo.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    batch=BATCH,
    imgsz=IMG_SIZE,
    project=OUTPUT_DIR,
    name='train',
    exist_ok=True,
    patience=20,
    save=True,
    save_period=10,
    verbose=True,
    plots=True,

    optimizer='AdamW',
    lr0=0.001,
    lrf=0.01,
    weight_decay=0.0005,

    # Augmentations — light (Roboflow already augmented)
    mosaic=1.0,
    mixup=0.05,
    hsv_h=0.01,   # minimal hue shift (don't change icon colors)
    hsv_s=0.3,
    hsv_v=0.3,
    fliplr=0.0,   # no horizontal flip (asymmetric icons)
    flipud=0.0,
    degrees=3.0,
    scale=0.2,
    translate=0.1,
)

# ── 5. Save best model ────────────────────────────────────────
best_src = os.path.join(OUTPUT_DIR, 'train', 'weights', 'best.pt')
best_dst = '/kaggle/working/troop_bar.pt'

if os.path.exists(best_src):
    shutil.copy2(best_src, best_dst)
    print(f'\nBest model saved: {best_dst}')
    print('Download troop_bar.pt from Kaggle output tab')
    print('Then place it at: weights/yolo_troop_bar/troop_bar.pt')
else:
    print('WARNING: best.pt not found')

# ── 6. Quick validation summary ───────────────────────────────
print(f'\nFinal metrics:')
print(f'  mAP50    : {results.results_dict.get("metrics/mAP50(B)", 0):.3f}')
print(f'  mAP50-95 : {results.results_dict.get("metrics/mAP50-95(B)", 0):.3f}')
print(f'  Precision: {results.results_dict.get("metrics/precision(B)", 0):.3f}')
print(f'  Recall   : {results.results_dict.get("metrics/recall(B)", 0):.3f}')
