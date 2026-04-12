# ClashAI v4

IA autonome qui joue à Clash of Clans via ADB sur émulateur.

## Architecture

```
clashai/                    # Package principal
├── perception/             # Vision: YOLO, CNN, OCR, template matching
├── combat/                 # RL: agent PPO, environnement, sorts, héros
├── navigation/             # ADB, calibration UI, zoom, GdC
├── social/                 # Chat clan, commandes
└── brain.py                # Orchestrateur principal

tools/                      # Scripts d'entraînement et utilitaires
configs/                    # YAML configs (YOLO, UI positions)
weights/                    # Modèles entraînés (.pt, .pth)
templates/                  # Templates pour matching (troupes, héros, digits)
datasets/                   # Données d'entraînement
docs/                       # Documentation
```

## Quickstart

```bash
# Installation
uv sync

# Calibration UI
python -m clashai.navigation.calibrate_ui

# Lancer le brain (mode auto)
python -m clashai.brain --mode auto

# Entraînement RL
python tools/train_rl.py --heuristic --episodes 5
```

## Configuration

- **Émulateur** : Google Play Games Developer Emulator, 1920×1080
- **ADB** : `adb connect localhost:6520`
- **GPU** : RTX 5070 Laptop (8 Go VRAM)

## Modèles

| Modèle | Fichier | Description |
|--------|---------|-------------|
| YOLO bâtiments | `weights/yolo_buildings.pt` | 47 classes, détection village |
| YOLO troupes | `weights/yolo_troops.pt` | 13 classes, mid-combat (mAP50=0.987) |
| CNN écran | `weights/screen_cnn.pth` | 12 états d'écran |
| CNN bâtiments | `weights/building_cnn.pth` | Classification fine |
| RL agent | `weights/rl/` | PPO checkpoints |
