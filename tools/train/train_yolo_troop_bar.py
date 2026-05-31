# tools/train/train_yolo_troop_bar.py
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

import subprocess
import yaml

from ultralytics import YOLO
import shutil, os

DATA_YAML = './datasets/dataset_troupe_barre/dataa.yaml'
MODEL      = 'yolo26s.pt'
EPOCHS     = 100
BATCH      = 16
IMG_SIZE   = 1088
OUTPUT_DIR = './weights/yolo_troupes_barre/V2/'

if __name__ == '__main__':
    
    subprocess.run(['pip', 'install', 'ultralytics', '-q'], check=True)

    print(f'Dataset: {DATA_YAML}')

    with open(DATA_YAML) as f:
        meta = yaml.safe_load(f)
    CLASSES = meta.get('names', [])
    print(f'Classes from data.yaml: {len(CLASSES)} → {CLASSES}')

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
        device=0,

        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,

        mosaic=1.0,
        mixup=0.05,
        hsv_h=0.01,
        hsv_s=0.3,
        hsv_v=0.3,
        fliplr=0.0, 
        flipud=0.0,
        degrees=3.0,
        scale=0.2,
        translate=0.1,
    )

    best_src = os.path.join(OUTPUT_DIR, 'train', 'weights', 'best.pt')
    best_dst = './weights/yolo_troupes_barre/V2/best/troop_bar.pt'

    if os.path.exists(best_src):
        os.makedirs(os.path.dirname(best_dst), exist_ok=True)
        shutil.copy2(best_src, best_dst)
        print(f'\nBest model saved: {best_dst}')
        print('Download troop_bar.pt from Kaggle output tab')
        print('Then place it at: weights/yolo_troop_bar/troop_bar.pt')
    else:
        print('WARNING: best.pt not found')

    print(f'\nFinal metrics:')
    print(f'  mAP50    : {results.results_dict.get("metrics/mAP50(B)", 0):.3f}')
    print(f'  mAP50-95 : {results.results_dict.get("metrics/mAP50-95(B)", 0):.3f}')
    print(f'  Precision: {results.results_dict.get("metrics/precision(B)", 0):.3f}')
    print(f'  Recall   : {results.results_dict.get("metrics/recall(B)", 0):.3f}')
