import os
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from clashai.perception.screen_classifier import MyCustomCNN

# --- CONFIGURATION ---
from clashai.paths import PROJECT_ROOT as project_root

DATA_DIR = os.path.join(project_root, 'dataset_cnn')
WEIGHTS_DIR = os.path.join(project_root, 'weights')
os.makedirs(WEIGHTS_DIR, exist_ok=True)

IMG_SIZE = 128
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 60
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Matériel : {DEVICE}")

# --- DATASET ---
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
classes = full_dataset.classes
num_classes = len(classes)

# Save the class list
json_path = os.path.join(WEIGHTS_DIR, 'classes.json')
print(f"Sauvegarde de la liste des classes dans '{json_path}'...")
with open(json_path, 'w') as f:
    json.dump(classes, f)

print(f"{num_classes} classes trouvées.")

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- MODEL ---
model = MyCustomCNN(num_classes).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS) 

# --- TRAINING ---
print("\n🔥 Démarrage de l'entraînement...")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
    for images, labels in loop:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        loop.set_postfix(loss=loss.item())

    print(f" --> Train Accuracy: {100 * correct / total:.2f}%")
    scheduler.step()

# --- SAVE ---
pth_path = os.path.join(WEIGHTS_DIR, "building_cnn.pth")
print(f"\nSauvegarde du modèle dans '{pth_path}'...")
torch.save(model.state_dict(), pth_path)
print("Terminé !")
