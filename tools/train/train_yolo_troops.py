# scripts/rl/train_yolo_troops.py
# YOLO11 training for troop detection in combat.
#
# Same approach as for buildings, but adapted for troops:
# - Troops move (unlike static buildings)
# - Troops are smaller (wizards, archers)
# - Troops often overlap (witch packs)
# - Background changes (different villages)
#
# Expected dataset structure:
# dataset_troops/
#  images/
#   train/ # 80% of images
#   val/ # 20% of images
#  labels/
#   train/ # YOLO labels (.txt)
#   val/
#  coc_troops.yaml
#
# Usage:
# python scripts/rl/train_yolo_troops.py
# python scripts/rl/train_yolo_troops.py --epochs 150 --batch 8
# python scripts/rl/train_yolo_troops.py --resume
# python scripts/rl/train_yolo_troops.py --test
# python scripts/rl/train_yolo_troops.py --test --image combat_captures/combat_001.png

import os
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# CONFIGURATION
# =============================================================================

DATASET_YAML = os.path.join(project_root, 'coc_troops.yaml')
WEIGHTS_DIR = os.path.join(project_root, 'weights', 'yolo_troops')
BEST_WEIGHTS = os.path.join(WEIGHTS_DIR, 'best.pt')

# Default hyperparameters
DEFAULT_EPOCHS = 100
DEFAULT_BATCH = 16
DEFAULT_IMG_SIZE = 640
DEFAULT_MODEL = 'yolo11m.pt'


# =============================================================================
# TRAINING
# =============================================================================

def train(epochs=DEFAULT_EPOCHS, batch=DEFAULT_BATCH, img_size=DEFAULT_IMG_SIZE,
          model=DEFAULT_MODEL, resume=False, data=None):
    """Trains the YOLO11 model to detect troops in combat."""
    from ultralytics import YOLO

    data_yaml = data or DATASET_YAML

    if not os.path.exists(data_yaml):
        print(f"ERROR: Dataset non trouvé : {data_yaml}")
        print(" Crée le dataset avec capture_combat.py + annotation LabelMe")
        print(" Puis convertis en format YOLO avec convert_labelme.py")
        return

    print(f"\n{'='*60}")
    print(" ClashAI — Entraînement YOLO Troupes")
    print(f"{'='*60}")
    print(f" Dataset : {data_yaml}")
    print(f" Modèle : {model}")
    print(f" Epochs : {epochs}")
    print(f" Batch : {batch}")
    print(f" Img size : {img_size}")
    print(f" Resume : {resume}")
    print(f"{'='*60}\n")

    if resume and os.path.exists(BEST_WEIGHTS):
        print(f"Reprise depuis {BEST_WEIGHTS}")
        yolo = YOLO(BEST_WEIGHTS)
    else:
        print(f"Chargement du modèle pré-entraîné {model}")
        yolo = YOLO(model)

    # Training
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

        # Optimizer
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,

        # Loss weights

        box=7.5,
        cls=1.0,
        dfl=1.5,

        # Augmentation (defined in the yaml, but can be overridden here)
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.15,
    )

    # Copy the best weights to a fixed location
    best_src = os.path.join(WEIGHTS_DIR, 'train', 'weights', 'best.pt')
    if os.path.exists(best_src):
        import shutil
        shutil.copy2(best_src, BEST_WEIGHTS)
        print(f"\nMeilleur modèle copié vers : {BEST_WEIGHTS}")

    print(f"\nRésultats dans : {os.path.join(WEIGHTS_DIR, 'train')}")
    return results


# =============================================================================
# TEST / INFERENCE
# =============================================================================

def test(image_path=None, conf=0.35, save=True):
    """Tests the model on an image or an ADB screenshot."""
    from ultralytics import YOLO

    if not os.path.exists(BEST_WEIGHTS):
        print(f"ERROR: Pas de modèle entraîné : {BEST_WEIGHTS}")
        print(" Lance d'abord : python train_yolo_troops.py")
        return

    print("\nTest YOLO Troupes")
    print(f" Modèle : {BEST_WEIGHTS}")
    print(f" Seuil : {conf}")

    yolo = YOLO(BEST_WEIGHTS)

    if image_path and os.path.exists(image_path):
        print(f" Image : {image_path}")
        source = image_path
    else:
        # ADB screenshot
        print(" Source : screenshot ADB")
        import subprocess
        import io
        from PIL import Image

        result = subprocess.run(
            ["adb", "exec-out", "screencap", "-p"],
            capture_output=True, timeout=5
        )
        if result.returncode != 0 or len(result.stdout) < 100:
            print("ERROR: Screenshot ADB échoué")
            return

        img = Image.open(io.BytesIO(result.stdout)).convert("RGB")
        tmp_path = os.path.join(project_root, '_test_troops_screenshot.png')
        img.save(tmp_path)
        source = tmp_path

    # Inference
    results = yolo.predict(
        source=source,
        conf=conf,
        save=save,
        save_txt=True,
        project=os.path.join(WEIGHTS_DIR, 'test'),
        name='predict',
        exist_ok=True,
    )

    # Display results
    for r in results:
        boxes = r.boxes
        if len(boxes) == 0:
            print("\n WARNING: Aucune troupe détectée")
            continue

        print(f"\n {len(boxes)} troupes détectées :")
        names = r.names
        for box in boxes:
            cls_id = int(box.cls[0])
            conf_val = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            name = names[cls_id]
            print(f" {name:20s} conf={conf_val:.2f} "
                  f"({int(x1)},{int(y1)})-({int(x2)},{int(y2)})")

    if save:
        print(f"\n Images annotées dans : {os.path.join(WEIGHTS_DIR, 'test', 'predict')}")


# =============================================================================
# VALIDATION
# =============================================================================

def validate(data=None):
    """Validates the model on the validation dataset."""
    from ultralytics import YOLO

    if not os.path.exists(BEST_WEIGHTS):
        print(f"ERROR: Pas de modèle entraîné : {BEST_WEIGHTS}")
        return

    data_yaml = data or DATASET_YAML

    print("\nValidation YOLO Troupes")
    print(f" Modèle : {BEST_WEIGHTS}")
    print(f" Dataset : {data_yaml}")

    yolo = YOLO(BEST_WEIGHTS)
    metrics = yolo.val(data=data_yaml, split='val')

    print(f"\n{'='*60}")
    print(" Résultats de validation")
    print(f"{'='*60}")
    print(f" mAP50 : {metrics.box.map50:.3f}")
    print(f" mAP50-95 : {metrics.box.map:.3f}")
    print(f" Précision : {metrics.box.mp:.3f}")
    print(f" Rappel : {metrics.box.mr:.3f}")
    print(f"{'='*60}")

    # Per class
    if hasattr(metrics.box, 'maps') and metrics.box.maps is not None:
        NAMES = ['golem', 'sorcier', 'sorciere', 'pekka', 'archere',
                 'lance_buche', 'roi', 'reine', 'grand_gardien', 'championne']
        print("\n Par classe :")
        for i, m in enumerate(metrics.box.maps):
            name = NAMES[i] if i < len(NAMES) else f"class_{i}"
            print(f" {name:20s} mAP50 = {m:.3f}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ClashAI — Entraînement YOLO détection troupes"
    )
    parser.add_argument('--test', action='store_true',
                        help="Test the model")
    parser.add_argument('--validate', action='store_true',
                        help="Validate on the validation dataset")
    parser.add_argument('--resume', action='store_true',
                        help="Resume training")
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                        help=f"Number of epochs (default: {DEFAULT_EPOCHS})")
    parser.add_argument('--batch', type=int, default=DEFAULT_BATCH,
                        help=f"Batch size (default: {DEFAULT_BATCH})")
    parser.add_argument('--img-size', type=int, default=DEFAULT_IMG_SIZE,
                        help=f"Image size (default: {DEFAULT_IMG_SIZE})")
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                        help=f"Base model (default: {DEFAULT_MODEL})")
    parser.add_argument('--data', type=str, default=None,
                        help="Path to the dataset yaml")
    parser.add_argument('--image', type=str, default=None,
                        help="Image to test (with --test)")
    parser.add_argument('--conf', type=float, default=0.35,
                        help="Confidence threshold for testing")

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
