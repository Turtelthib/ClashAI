# tools/train/train_yolo_walls_seg.py
# Training YOLO26 for WALL SEGMENTATION in Clash of Clans.
#
# Differences vs train_yolo_walls.py:
# - yolo26s-seg.pt model (segmentation instead of detection)
# - Expected annotations in YOLO segmentation format (polygons, not bboxes)
# - Inference returns pixel MASKS, not bboxes
#
# Why segmentation for walls?
# - Walls are thin and diagonal → rectangular bbox encompasses a lot of grass
# - Segmentation = captures the ACTUAL shape of the wall, pixel by pixel
# - Faster annotation with the Roboflow brush
# - Tighter convex hull over the actual wall area
#
# Why yolo26s-seg and not yolo26n-seg?
# - ‘s’ (small) = ~19 MB, better for fine details (level 1-3 walls = very few pixels)
# - ‘n’ (nano) = 5 MB, faster but less accurate on small objects
# - On RTX 5070, the speed difference is minimal (~30 ms)
#
# Expected structure (Roboflow YOLO Segmentation export):
#   dataset_walls/
#    data.yaml               (Roboflow config)
#    train/
#       images/
#       labels/             (.txt with normalized polygons)
#    valid/
#       images/
#       labels/
#    test/                   (optional)
#        images/
#        labels/

import os
import argparse

from clashai.paths import PROJECT_ROOT as project_root


# =============================================================================
#                         CONFIGURATION
# =============================================================================

DATASET_YAML = os.path.join(project_root, 'datasets', 'dataset_walls', 'data.yaml')
WEIGHTS_DIR = os.path.join(project_root, 'weights', 'yolo_walls_seg')
BEST_WEIGHTS = os.path.join(WEIGHTS_DIR, 'best.pt')

DEFAULT_EPOCHS = 100
DEFAULT_BATCH = 12
DEFAULT_IMG_SIZE = 640
DEFAULT_MODEL = 'yolo26m-seg.pt'


# =============================================================================
#                         ENTRAÎNEMENT
# =============================================================================

def train(epochs=DEFAULT_EPOCHS, batch=DEFAULT_BATCH, img_size=DEFAULT_IMG_SIZE,
          model=DEFAULT_MODEL, resume=False, data=None):
    """Entraîne YOLO26-seg pour segmenter les murs CoC."""
    from ultralytics import YOLO

    data_yaml = data or DATASET_YAML

    if not os.path.exists(data_yaml):
        print(f"ERROR: Dataset not found: {data_yaml}")
        print("   Expected structure:")
        print("   datasets/dataset_walls/")
        print("    data.yaml")
        print("    train/images/, train/labels/")
        print("    valid/images/, valid/labels/")
        return

    print(f"\n{'='*60}")
    print("   ClashAI — Entraînement YOLO26-SEG Murs")
    print(f"{'='*60}")
    print(f"  Dataset  : {data_yaml}")
    print(f"  Modèle   : {model}")
    print(f"  Epochs   : {epochs}")
    print(f"  Batch    : {batch}")
    print(f"  Img size : {img_size}")
    print(f"  Resume   : {resume}")
    print(f"{'='*60}\n")

    if resume and os.path.exists(BEST_WEIGHTS):
        print(f" Reprise depuis {BEST_WEIGHTS}")
        yolo = YOLO(BEST_WEIGHTS)
    else:
        print(f" Chargement du modèle pré-entraîné {model}")
        yolo = YOLO(model)

    results = yolo.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=img_size,
        project=WEIGHTS_DIR,
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

        box=7.5,
        cls=1.5,
        dfl=1.5,


        mosaic=1.0,
        mixup=0.05,            
        copy_paste=0.10,       
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        flipud=0.0,  
        degrees=5.0,
        scale=0.5,
        translate=0.1,
    )

    best_src = os.path.join(WEIGHTS_DIR, 'train', 'weights', 'best.pt')
    if os.path.exists(best_src):
        import shutil
        shutil.copy2(best_src, BEST_WEIGHTS)
        print(f"\n Meilleur modèle copié vers : {BEST_WEIGHTS}")

    print(f"\n Résultats dans : {os.path.join(WEIGHTS_DIR, 'train')}")
    return results


# =============================================================================
#                         TEST / INFÉRENCE
# =============================================================================

def test(image_path=None, conf=0.35, save=True):
    """Teste le modèle de segmentation sur une image ou un screenshot ADB."""
    from ultralytics import YOLO

    if not os.path.exists(BEST_WEIGHTS):
        print(f" Pas de modèle entraîné : {BEST_WEIGHTS}")
        print("   Lance d'abord : uv run python tools/train/train_yolo_walls_seg.py")
        return

    print("\n Test YOLO26-SEG Murs")
    print(f"   Modèle : {BEST_WEIGHTS}")
    print(f"   Seuil  : {conf}")

    yolo = YOLO(BEST_WEIGHTS)

    if image_path and os.path.exists(image_path):
        print(f"   Image  : {image_path}")
        source = image_path
    else:
        print("   Source : screenshot ADB")
        import subprocess
        import io
        from PIL import Image

        from clashai.paths import ADB_DEVICE
        result = subprocess.run(
            ["adb", "-s", ADB_DEVICE, "exec-out", "screencap", "-p"],
            capture_output=True, timeout=5
        )
        if result.returncode != 0 or len(result.stdout) < 100:
            print(" Screenshot ADB échoué")
            return

        img = Image.open(io.BytesIO(result.stdout)).convert("RGB")
        tmp_path = os.path.join(project_root, '_test_walls_screenshot.png')
        img.save(tmp_path)
        source = tmp_path

    results = yolo.predict(
        source=source,
        conf=conf,
        save=save,
        save_txt=True,
        retina_masks=True,  
        project=os.path.join(WEIGHTS_DIR, 'test'),
        name='predict',
        exist_ok=True,
    )

    for r in results:
        masks = r.masks
        if masks is None or len(masks) == 0:
            print("\n     Aucun mur détecté")
            continue

        boxes = r.boxes
        print(f"\n    {len(masks)} murs détectés :")
        for i, (box, mask) in enumerate(zip(boxes, masks)):
            conf_val = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            mask_pixels = int(mask.data.sum().item())
            print(f"      mur #{i:2d}  conf={conf_val:.2f}  "
                  f"bbox=({int(x1)},{int(y1)})-({int(x2)},{int(y2)})  "
                  f"surface={mask_pixels}px")

    if save:
        print(f"\n    Images annotées : {os.path.join(WEIGHTS_DIR, 'test', 'predict')}")


# =============================================================================
#                         VALIDATION
# =============================================================================

def validate(data=None):
    """Valide le modèle de segmentation sur le split valid."""
    from ultralytics import YOLO

    if not os.path.exists(BEST_WEIGHTS):
        print(f" Pas de modèle entraîné : {BEST_WEIGHTS}")
        return

    data_yaml = data or DATASET_YAML

    print("\n Validation YOLO26-SEG Murs")
    print(f"   Modèle  : {BEST_WEIGHTS}")
    print(f"   Dataset : {data_yaml}")

    yolo = YOLO(BEST_WEIGHTS)
    metrics = yolo.val(data=data_yaml, split='val')

    print(f"\n{'='*60}")
    print("   Résultats de validation")
    print(f"{'='*60}")
    print(f"  [Box] mAP50      : {metrics.box.map50:.3f}")
    print(f"  [Box] mAP50-95   : {metrics.box.map:.3f}")
    print(f"  [Box] Précision  : {metrics.box.mp:.3f}")
    print(f"  [Box] Rappel     : {metrics.box.mr:.3f}")
    print(f"  [Seg] mAP50      : {metrics.seg.map50:.3f}")
    print(f"  [Seg] mAP50-95   : {metrics.seg.map:.3f}")
    print(f"  [Seg] Précision  : {metrics.seg.mp:.3f}")
    print(f"  [Seg] Rappel     : {metrics.seg.mr:.3f}")
    print(f"{'='*60}")

    seg_map50 = metrics.seg.map50
    if seg_map50 >= 0.8:
        print("   Excellent — segmentation murs très fiable")
    elif seg_map50 >= 0.6:
        print("   Correct — utilisable avec quelques ratés")
    elif seg_map50 >= 0.4:
        print("   Moyen — plus d'images ou meilleure annotation nécessaire")
    else:
        print("   Faible — dataset insuffisant ou mal annoté")


# =============================================================================
#                         MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ClashAI — Entraînement YOLO26-SEG segmentation murs"
    )
    parser.add_argument('--test', action='store_true',
                        help="Tester le modèle sur une image ou screenshot ADB")
    parser.add_argument('--validate', action='store_true',
                        help="Valider sur le dataset de validation")
    parser.add_argument('--resume', action='store_true',
                        help="Reprendre l'entraînement depuis best.pt")
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                        help=f"Nombre d'epochs (défaut: {DEFAULT_EPOCHS})")
    parser.add_argument('--batch', type=int, default=DEFAULT_BATCH,
                        help=f"Batch size (défaut: {DEFAULT_BATCH})")
    parser.add_argument('--img-size', type=int, default=DEFAULT_IMG_SIZE,
                        help=f"Taille image (défaut: {DEFAULT_IMG_SIZE})")
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                        help=f"Modèle de base (défaut: {DEFAULT_MODEL})")
    parser.add_argument('--data', type=str, default=None,
                        help="Chemin custom vers le yaml du dataset")
    parser.add_argument('--image', type=str, default=None,
                        help="Image à tester (avec --test)")
    parser.add_argument('--conf', type=float, default=0.35,
                        help="Seuil de confiance pour le test")

    args = parser.parse_args()

    if args.test:
        test(image_path=args.image, conf=args.conf)
    elif args.validate:
        validate(data=args.data)
    else:
        train(
            epochs=args.epochs,
            batch=args.batch,
            img_size=args.img_size,
            model=args.model,
            resume=args.resume,
            data=args.data,
        )
