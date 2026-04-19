import os
import json
import random

import cv2
import torch
from torchvision import transforms
from ultralytics import YOLO
from PIL import Image

from clashai.perception.screen_classifier import MyCustomCNN

# --- CONFIGURATION ---
from clashai.paths import PROJECT_ROOT, WEIGHTS_DIR
# WEIGHTS_DIR imported from clashai.paths

YOLO_PATH = os.path.join(PROJECT_ROOT, 'runs', 'detect', 'FinishedTrain', 'weights', 'best.pt')
CNN_PATH = os.path.join(WEIGHTS_DIR, 'building_cnn.pth')
IMAGE_TO_TEST = 'test_img/hdv13.png'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- LOADING CLASSES ---
print("Chargement de la liste des classes depuis classes.json...")
try:
    json_path = os.path.join(WEIGHTS_DIR, 'classes.json')
    with open(json_path, 'r') as f:
        CLASSES_CNN = json.load(f)
    print(f"{len(CLASSES_CNN)} classes chargées.")
except FileNotFoundError:
    print("ERROR: ERREUR : Le fichier 'classes.json' est introuvable !")
    print(f" L'IA cherche dans : {os.path.abspath(json_path)}")
    print("👉 Lancez d'abord 'python main/train_all.py' pour générer ce fichier.")
    exit()

# --- LOADING MODELS ---
print("Chargement du système hybride...")
yolo_model = YOLO(YOLO_PATH)

cnn_model = MyCustomCNN(num_classes=len(CLASSES_CNN)).to(DEVICE)
cnn_model.load_state_dict(torch.load(CNN_PATH, map_location=DEVICE))
cnn_model.eval()

# --- TRANSFORMATION PIPELINE ---
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# --- INFERENCE ---
print(f"🕵 Analyse de {IMAGE_TO_TEST}...")

results = yolo_model.predict(IMAGE_TO_TEST, conf=0.25, iou=0.50, verbose=False)
result = results[0]

img_cv = cv2.imread(IMAGE_TO_TEST)
if img_cv is None:
    print(f"ERROR: Erreur : Impossible de lire l'image '{IMAGE_TO_TEST}'")
    exit()

img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
pil_image = Image.fromarray(img_rgb)

count = 0
for box in result.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    # Clamp to image dimensions
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_cv.shape[1], x2)
    y2 = min(img_cv.shape[0], y2)

    crop = pil_image.crop((x1, y1, x2, y2))
    crop_tensor = transform(crop).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = cnn_model(crop_tensor)
        _, predicted_idx = torch.max(outputs, 1)
        cnn_label = CLASSES_CNN[predicted_idx.item()]

        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence = probs[0][predicted_idx.item()].item()

    if cnn_label in ('useless', 'ignore'):
        continue

    if confidence > 0.5:
        count += 1

        # Deterministic color per class
        random.seed(predicted_idx.item() * 10)
        color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))

        cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_cv, f"{cnn_label} {confidence:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        print(f"Objet trouvé : {cnn_label} ({confidence:.0%})")

# --- SAVE ---
output_file = "Result.jpg"
cv2.imwrite(output_file, img_cv)
print(f"\nAnalyse terminée ! {count} objets détectés.")
print(f"🖼 Résultat sauvegardé sous : {output_file}")