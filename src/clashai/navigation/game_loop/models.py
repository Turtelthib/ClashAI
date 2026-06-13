# clashai/navigation/game_loop/models.py
# load_models() — load screen CNN, YOLO buildings, building CNN, walls,
# troop bar detector + start the async PerceptionThread.

import os
import sys
import json

import torch
from ultralytics import YOLO

from clashai.paths import PROJECT_ROOT, WEIGHTS_DIR
from clashai.perception.screen_classifier import MyCustomCNN
from clashai.navigation.game_loop.constants import DEVICE


def load_models():
    """Loads the 3 models: screen CNN, YOLO, building CNN."""
    from clashai.config.logging import pp
    models = {}

    # --- 1) Screen state CNN ---
    pp(" Loading screen state CNN...", tag='init')
    screen_classes_path = os.path.join(WEIGHTS_DIR, 'classification', 'screen_classes.json')
    screen_weights_path = os.path.join(WEIGHTS_DIR, 'classification', 'cnn_screen_classification.pt')

    if not os.path.exists(screen_classes_path) or not os.path.exists(screen_weights_path):
        print("ERROR: cnn_screen_classification.pt or screen_classes.json not found!")
        print(" Run: uv run python tools/train/train_screen_cnn.py")
        sys.exit(1)

    with open(screen_classes_path) as f:
        meta = json.load(f)

    # Support both formats: old list and new list (both use MyCustomCNN)
    if isinstance(meta, list):
        models['screen_classes'] = meta
    else:
        models['screen_classes'] = meta.get('classes', meta)

    num_screen_classes = len(models['screen_classes'])
    screen_cnn = MyCustomCNN(num_classes=num_screen_classes)
    screen_cnn.load_state_dict(torch.load(screen_weights_path, map_location=DEVICE))
    screen_cnn = screen_cnn.to(DEVICE)
    screen_cnn.eval()
    models['screen_cnn'] = screen_cnn
    pp(f" {num_screen_classes} states loaded: {models['screen_classes']}", tag='init_done')

    # --- 2) YOLO Detection ---
    pp("Loading YOLO11...", tag='init')
    yolo_path = os.path.join(WEIGHTS_DIR, 'best.pt')
    if not os.path.exists(yolo_path):
        # Fallback: look in runs/
        yolo_path = os.path.join(PROJECT_ROOT, 'runs', 'detect', 'FinishedTrain', 'weights', 'best.pt')
    if not os.path.exists(yolo_path):
        print("ERROR: ERREUR : best.pt introuvable !")
        sys.exit(1)

    models['yolo'] = YOLO(yolo_path)
    pp(" YOLO chargé", tag='yolo')

    # --- 3) Building CNN ---
    pp("Loading building CNN...", tag='init')
    building_classes_path = os.path.join(WEIGHTS_DIR, 'classes.json')
    building_weights_path = os.path.join(WEIGHTS_DIR, 'building_cnn.pth')

    if not os.path.exists(building_classes_path) or not os.path.exists(building_weights_path):
        print("ERROR: ERREUR : building_cnn.pth ou classes.json introuvable !")
        sys.exit(1)

    with open(building_classes_path) as f:
        models['building_classes'] = json.load(f)

    building_cnn = MyCustomCNN(num_classes=len(models['building_classes'])).to(DEVICE)
    building_cnn.load_state_dict(torch.load(building_weights_path, map_location=DEVICE))
    building_cnn.eval()
    models['building_cnn'] = building_cnn
    pp(f" {len(models['building_classes'])} building classes loaded", tag='init_done')

    # --- 4) YOLO Walls segmentation ---
    walls_path = os.path.join(WEIGHTS_DIR, 'yolo_walls_seg', 'walls_detection.pt')
    if os.path.exists(walls_path):
        models['yolo_walls'] = YOLO(walls_path)
        pp(" YOLO walls loaded", tag='yolo')
    else:
        models['yolo_walls'] = None
        pp(f"WARNING: yolo_walls not found at {walls_path} — deploy zone will use building hull fallback", tag='warning')

    # --- 5) YOLO Troop Bar Detector ---
    troop_bar_path = os.path.join(WEIGHTS_DIR, 'yolo_troupes_barre', 'troop_bar.pt')
    if os.path.exists(troop_bar_path):
        from clashai.perception.troop_bar_detector import TroopBarDetector
        models['troop_bar_detector'] = TroopBarDetector(troop_bar_path)
    else:
        models['troop_bar_detector'] = None
        print(f"WARNING: troop_bar.pt not found — using template matching fallback")

    # --- 6) Async perception thread ---
    from clashai.perception.perception_thread import PerceptionThread
    models['perception_thread'] = PerceptionThread(models, verbose=False)
    models['perception_thread'].start()

    pp(f"\nTous les modèles sont chargés sur {DEVICE}\n", tag='init_done')
    return models

