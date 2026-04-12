from ultralytics import YOLO
import os
import shutil

# --- 1. CONFIGURATION DES CHEMINS ---
# On récupère le dossier où est ce script (scripts/)
# On remonte d'un cran pour trouver la racine (COCProj/)
from clashai.paths import PROJECT_ROOT as project_root

# On définit les chemins absolus
yaml_path = os.path.join(project_root, 'coc.yaml')
weights_dir = os.path.join(project_root, 'weights')

# --- 2. ENTRAÎNEMENT ---
# Charge un modèle pré-entraîné
model = YOLO('yolo11n.pt')

print("🚀 Lancement de l'entraînement YOLO...")
print(f"📂 Configuration : {yaml_path}")

results = model.train(
    data=yaml_path,
    epochs=500,
    patience=50,
    imgsz=1600,
    batch=4,
    cos_lr=True, 
    project=os.path.join(project_root, 'runs/detect'),
    name='FinishedTrain',
    mixup=0.1, 
    exist_ok=True,
    amp=True
)

os.makedirs(weights_dir, exist_ok=True)

source_best = os.path.join(project_root, 'runs/detect/FinishedTrain/weights/best.pt')
dest_best = os.path.join(weights_dir, 'best.pt')

if os.path.exists(source_best):
    shutil.copy(source_best, dest_best)
    print(f"✅ Modèle sauvegardé dans : {dest_best}")
else:
    print("⚠️ Attention : Le fichier best.pt n'a pas été trouvé.")