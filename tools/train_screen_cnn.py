# tools/train_screen_cnn.py
# Screen state CNN training.
# Uses the project's MyCustomCNN — simple, fast, proven working.
# Add more images to datasets/dataset_screen/<class>/ then retrain.

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from clashai.perception.screen_classifier import MyCustomCNN

from clashai.paths import PROJECT_ROOT as project_root

DATA_DIR    = os.path.join(project_root, 'datasets', 'dataset_screen')
WEIGHTS_DIR = os.path.join(project_root, 'weights')
SAVE_PATH   = os.path.join(WEIGHTS_DIR, 'classification', 'cnn_screen_classification.pt')
JSON_PATH   = os.path.join(WEIGHTS_DIR, 'classification', 'screen_classes.json')

IMG_SIZE      = 224
BATCH_SIZE    = 32
LEARNING_RATE = 0.001
EPOCHS        = 15
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Screen CNN training — MyCustomCNN")
print(f"Device: {DEVICE}")

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
classes      = full_dataset.classes
num_classes  = len(classes)

os.makedirs(os.path.dirname(JSON_PATH), exist_ok=True)
with open(JSON_PATH, 'w') as f:
    json.dump(classes, f)

print(f"{num_classes} screen states: {classes}")
print(f"Total images: {len(full_dataset)}")

counts = [0] * num_classes
for _, label in full_dataset:
    counts[label] += 1
for i, cls in enumerate(classes):
    flag = " <-- ADD MORE" if counts[i] < 20 else ""
    print(f"  {cls:<25s} {counts[i]} images{flag}")

train_size = int(0.8 * len(full_dataset))
val_size   = len(full_dataset) - train_size
train_set, val_set = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model     = MyCustomCNN(num_classes).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

print(f"\nTraining for {EPOCHS} epochs...")
best_val_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    correct, total = 0, 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for images, labels in loop:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs, 1)
        total   += labels.size(0)
        correct += (predicted == labels).sum().item()
        loop.set_postfix(loss=f'{loss.item():.3f}')

    train_acc = 100 * correct / total

    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            val_total   += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = 100 * val_correct / val_total
    print(f"  Train: {train_acc:.1f}%  Val: {val_acc:.1f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  Best model saved ({val_acc:.1f}%)")

    scheduler.step()

print(f"\nDone — best val accuracy: {best_val_acc:.1f}%")
print(f"Saved: {SAVE_PATH}")
