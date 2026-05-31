# tools/debug/detect_troop_bar.py
# Run the troop bar YOLO model on a single image file and save the
# annotated result at the project root.
#
# Usage:
#   uv run python tools/debug/detect_troop_bar.py --file path/to/image.png
#   uv run python tools/debug/detect_troop_bar.py --file img.png --conf 0.3
#   uv run python tools/debug/detect_troop_bar.py --file img.png --out result.png
#
# Output (default): <project_root>/_troop_bar_detection.png

import argparse
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_PATH = PROJECT_ROOT / 'weights' / 'yolo_troupes_barre' / 'troop_bar.pt'

HEROES = {'roi', 'reine', 'grand_gardien', 'championne', 'prince_gargouille', 'duc_draconique'}
SPELLS = {'soin', 'rage', 'gel', 'zap', 'saut', 'clone', 'invisible', 'rappel',
          'resurrection', 'totem', 'poison', 'seisme', 'speed', 'squelette',
          'chauve_souris', 'floraison', 'bloc_glace'}


def get_color(name):
    if '_deploye' in name:
        return (255, 50, 50)
    if '_capa' in name:
        return (255, 165, 0)
    if name in HEROES:
        return (255, 215, 0)
    if name in SPELLS:
        return (180, 60, 255)
    return (50, 200, 50)


def annotate(pil_img, results, conf_threshold):
    draw = ImageDraw.Draw(pil_img)
    r = results[0]
    names = r.names
    detections = []

    for box in r.boxes:
        conf = float(box.conf[0])
        if conf < conf_threshold:
            continue
        cls = int(box.cls[0])
        name = names.get(cls, f'cls{cls}')
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        color = get_color(name)

        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f'{name} {conf:.0%}'
        tw = len(label) * 7 + 4
        draw.rectangle([x1, y1 - 18, x1 + tw, y1], fill=color)
        draw.text((x1 + 2, y1 - 16), label, fill=(255, 255, 255))
        detections.append((name, conf, (x1, y1, x2, y2)))

    return pil_img, detections


def main():
    parser = argparse.ArgumentParser(
        description="Run troop bar YOLO on an image and save the annotated result.")
    parser.add_argument('--file', required=True,
                        help="Path to the input image")
    parser.add_argument('--conf', type=float, default=0.40,
                        help="Confidence threshold (default 0.40, matches production)")
    parser.add_argument('--out', default=None,
                        help="Output path (default: <root>/_troop_bar_detection.png)")
    parser.add_argument('--imgsz', type=int, default=None,
                        help="Inference image size (default: model's trained imgsz)")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"ERROR: Image not found: {args.file}")
        sys.exit(1)
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found: {MODEL_PATH}")
        sys.exit(1)

    # Default imgsz = the value the detector uses in production.
    imgsz = args.imgsz
    if imgsz is None:
        try:
            from clashai.perception.troop_bar_detector import YOLO_IMGSZ
            imgsz = YOLO_IMGSZ
        except Exception:
            imgsz = 1088

    print(f"Loading model: {MODEL_PATH}")
    from ultralytics import YOLO
    model = YOLO(str(MODEL_PATH))
    print(f"Classes: {len(model.names)}")

    img = Image.open(args.file).convert('RGB')
    print(f"Image: {img.size[0]}x{img.size[1]}  |  conf={args.conf}  imgsz={imgsz}")

    results = model.predict(img, conf=args.conf, imgsz=imgsz, verbose=False)
    img, detections = annotate(img, results, args.conf)

    print(f"\nDetections: {len(detections)}")
    for name, conf, (x1, y1, x2, y2) in sorted(detections, key=lambda d: d[2][0]):
        print(f"  {name:25s} {conf:.0%}  ({x1},{y1})-({x2},{y2})")

    out = args.out or str(PROJECT_ROOT / '_troop_bar_detection.png')
    img.save(out)
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
