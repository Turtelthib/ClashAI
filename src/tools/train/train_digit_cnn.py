# tools/train/train_digit_cnn.py
# Phase F.3 step 3 — train a tiny CNN to read the troop-bar count badges,
# from the folder-per-label dataset produced by label_digit_crops.py
# (<data>/digits/<count>/*.png).
#
# Output: weights/digit_cnn.pt = {state_dict, classes, img_size}. Phase 4 will
# load it in TroopBarDetector._read_count() (fallback EasyOCR if conf low).
#
# Run:
#   uv run python tools/train/train_digit_cnn.py                 # train on real labeled data
#   uv run python tools/train/train_digit_cnn.py --epochs 30
#   uv run python tools/train/train_digit_cnn.py --smoke         # synthetic self-test (no data needed)

import argparse
import random
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from clashai.paths import DATASETS_DIR, WEIGHTS_DIR
# SSOT: the architecture + input size live with the inference reader.
from clashai.perception.digit_reader import DigitCNN, IMG_SIZE


def _augment_pil(img):
    """Light geometric jitter on a glyph (PIL 'L') — rotation/scale/shift.
    Crucial for the rare digits (0, 7) where we have few real samples."""
    w, h = img.size
    img = img.rotate(random.uniform(-12, 12), resample=Image.BILINEAR, fillcolor=0)
    sx, sy = random.uniform(0.85, 1.15), random.uniform(0.85, 1.15)
    tx, ty = random.uniform(-2.5, 2.5), random.uniform(-2.5, 2.5)
    img = img.transform((w, h), Image.AFFINE, (1 / sx, 0, tx, 0, 1 / sy, ty),
                        resample=Image.BILINEAR, fillcolor=0)
    return img


def load_image(path, augment=False):
    img = Image.open(path).convert('L')
    if augment:
        img = _augment_pil(img)
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if augment:
        arr = arr * random.uniform(0.85, 1.15) + np.random.normal(0, 0.04, arr.shape)
        arr = np.clip(arr, 0.0, 1.0)
    return torch.from_numpy(arr.astype(np.float32)).unsqueeze(0)  # (1, 32, 32)


def load_dataset(root):
    """root/<label>/*.png → (samples, classes). Ignores _-prefixed dirs."""
    root = Path(root)
    classes = sorted(
        [d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith('_')],
        key=lambda s: (len(s), s),   # "2" before "11"
    )
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    samples = []
    for c in classes:
        for p in (root / c).glob('*.png'):
            samples.append((p, cls_to_idx[c]))
    return samples, classes


def make_smoke_dataset(root, per_class=24):
    """Synthetic dataset (random noise) to verify the pipeline runs end-to-end."""
    for label in ('1', '2', '11'):
        d = root / label
        d.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            arr = (np.random.rand(20, 20) * 255).astype('uint8')
            Image.fromarray(arr, 'L').save(d / f'{label}_{i}.png')


def batches(samples, batch_size, device, shuffle=True, augment=False):
    idx = list(range(len(samples)))
    if shuffle:
        random.shuffle(idx)
    for s in range(0, len(idx), batch_size):
        chunk = idx[s:s + batch_size]
        x = torch.stack([load_image(samples[j][0], augment=augment) for j in chunk]).to(device)
        y = torch.tensor([samples[j][1] for j in chunk], dtype=torch.long).to(device)
        yield x, y


def oversample(train, classes, target=80):
    """Duplicate samples of under-represented classes up to ~target each, so the
    rare digits (0, 7) are seen as often as the common ones. Augmentation makes
    the duplicates differ at train time."""
    by_cls = {}
    for s in train:
        by_cls.setdefault(s[1], []).append(s)
    out = []
    for ci in range(len(classes)):
        pool = by_cls.get(ci, [])
        if not pool:
            continue
        out.extend(pool)
        while len([s for s in out if s[1] == ci]) < target:
            out.append(random.choice(pool))
    random.shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser(description="Train the troop-bar digit CNN (Phase F.3).")
    ap.add_argument('--data', default=str(Path(DATASETS_DIR) / 'digits_single'),
                    help="Per-digit dataset (0-9), from build_digit_singles.py")
    ap.add_argument('--no-aug', action='store_true', help="Disable augmentation")
    ap.add_argument('--out', default=str(Path(WEIGHTS_DIR) / 'digit_cnn.pt'))
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--val-frac', type=float, default=0.2)
    ap.add_argument('--smoke', action='store_true',
                    help="Run on a synthetic dataset (self-test, no real data)")
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tmp = None
    data_root = Path(args.data)
    if args.smoke:
        tmp = Path(tempfile.mkdtemp(prefix='digit_smoke_'))
        make_smoke_dataset(tmp)
        data_root = tmp
        args.epochs = min(args.epochs, 2)
        print(f"[smoke] synthetic dataset at {data_root}")

    samples, classes = load_dataset(data_root)
    if len(samples) < 4 or len(classes) < 2:
        print(f"ERROR: need >=2 classes with crops in {data_root} "
              f"(found {len(classes)} classes, {len(samples)} crops).")
        print("-> Label data first: uv run python tools/data/label_digit_crops.py")
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
        return

    random.seed(0)
    random.shuffle(samples)
    n_val = max(1, int(len(samples) * args.val_frac))
    val, train = samples[:n_val], samples[n_val:]
    aug = not args.no_aug and not args.smoke
    if aug:
        train = oversample(train, classes)
    print(f"Classes ({len(classes)}): {classes}")
    print(f"Train {len(train)} (aug={aug}) | Val {len(val)} | device {device}")

    model = DigitCNN(len(classes)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = nn.CrossEntropyLoss()
    best_acc = 0.0

    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for x, y in batches(train, args.batch_size, device, augment=aug):
            opt.zero_grad()
            loss = lossf(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item()

        # Validation (+ per-class hits/totals)
        model.eval()
        correct = 0
        per_cls_hit = [0] * len(classes)
        per_cls_tot = [0] * len(classes)
        with torch.no_grad():
            for x, y in batches(val, args.batch_size, device, shuffle=False):
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                for yi, pi in zip(y.tolist(), pred.tolist()):
                    per_cls_tot[yi] += 1
                    if yi == pi:
                        per_cls_hit[yi] += 1
        acc = correct / max(len(val), 1)
        print(f" epoch {ep:2d}/{args.epochs}  loss={tot/max(1,len(train)//args.batch_size+1):.3f}  val_acc={acc:.1%}")

        if acc >= best_acc:
            best_acc = acc
            best_per_cls = (per_cls_hit, per_cls_tot)
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({'state_dict': model.state_dict(),
                        'classes': classes, 'img_size': IMG_SIZE}, args.out)

    print(f"\nBest val_acc={best_acc:.1%} - saved -> {args.out}")
    if not args.smoke:
        hit, tot_c = best_per_cls
        print(" Accuracy par classe (val):")
        for i, c in enumerate(classes):
            t = tot_c[i]
            rate = f"{hit[i]}/{t} = {100*hit[i]/t:.0f}%" if t else "(aucun en val)"
            print(f"   {c}: {rate}")
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)
        print("[smoke] OK — pipeline runs end-to-end.")


if __name__ == '__main__':
    main()
