import os, json, torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from clashai.perception.screen_classifier import MyCustomCNN

# --- CONFIG ---
from clashai.paths import PROJECT_ROOT as project_root

DATA_DIR = os.path.join(project_root, 'dataset_screen')
WEIGHTS_DIR = os.path.join(project_root, 'weights')
os.makedirs(WEIGHTS_DIR, exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(" Entraînement du CNN de classification d'écran")
print(f"Matériel : {DEVICE}")

# --- DATASET ---
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
classes = full_dataset.classes
num_classes = len(classes)

# Save screen classes
json_path = os.path.join(WEIGHTS_DIR, 'screen_classes.json')
with open(json_path, 'w') as f:
    json.dump(classes, f)

print(f"{num_classes} états d'écran trouvés : {classes}")
print(f"{len(full_dataset)} images au total")

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_set, val_set = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

# --- MODEL ---
model = MyCustomCNN(num_classes).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# --- TRAINING ---
print(f"\n Démarrage de l'entraînement ({EPOCHS} epochs)...")

best_val_acc = 0.0
for epoch in range(EPOCHS):
    model.train()
    correct, total, running_loss = 0, 0, 0.0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for images, labels in loop:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        loop.set_postfix(loss=loss.item())

    train_acc = 100 * correct / total
    print(f" --> Train Accuracy: {train_acc:.2f}%")

    # Validation at each epoch
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = 100 * val_correct / val_total
    print(f" --> Val Accuracy: {val_acc:.2f}%")

    # Save the best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, "screen_cnn.pth"))
        print(f" Nouveau meilleur modèle sauvegardé ({val_acc:.2f}%)")

    scheduler.step()

print(f"\nEntraînement terminé ! Meilleure Val Accuracy : {best_val_acc:.2f}%")
print("Modèle sauvegardé : weights/screen_cnn.pth")