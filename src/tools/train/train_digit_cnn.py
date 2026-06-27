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

IMG_SIZE = 32


class DigitCNN(nn.Module):
    """LeNet-ish: ~60k params, plenty for 32x32 grayscale digit badges."""

    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 16x16
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 8x8
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 4x4
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 4 * 4, 128), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


def load_image(path):
    img = Image.open(path).convert('L').resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)  # (1, 32, 32)


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


def batches(samples, batch_size, device, shuffle=True):
    idx = list(range(len(samples)))
    if shuffle:
        random.shuffle(idx)
    for s in range(0, len(idx), batch_size):
        chunk = idx[s:s + batch_size]
        x = torch.stack([load_image(samples[j][0]) for j in chunk]).to(device)
        y = torch.tensor([samples[j][1] for j in chunk], dtype=torch.long).to(device)
        yield x, y


def main():
    ap = argparse.ArgumentParser(description="Train the troop-bar digit CNN (Phase F.3).")
    ap.add_argument('--data', default=str(Path(DATASETS_DIR) / 'digits'),
                    help="Dataset root (folder-per-label, from label_digit_crops.py)")
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
    print(f"Classes ({len(classes)}): {classes}")
    print(f"Train {len(train)} | Val {len(val)} | device {device}")

    model = DigitCNN(len(classes)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = nn.CrossEntropyLoss()
    best_acc = 0.0

    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for x, y in batches(train, args.batch_size, device):
            opt.zero_grad()
            loss = lossf(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item()

        # Validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in batches(val, args.batch_size, device, shuffle=False):
                correct += (model(x).argmax(1) == y).sum().item()
        acc = correct / max(len(val), 1)
        print(f" epoch {ep:2d}/{args.epochs}  loss={tot/max(1,len(train)//args.batch_size+1):.3f}  val_acc={acc:.1%}")

        if acc >= best_acc:
            best_acc = acc
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({'state_dict': model.state_dict(),
                        'classes': classes, 'img_size': IMG_SIZE}, args.out)

    print(f"\nBest val_acc={best_acc:.1%} - saved -> {args.out}")
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)
        print("[smoke] OK — pipeline runs end-to-end.")


if __name__ == '__main__':
    main()
