# clashai/navigation/game_loop/constants.py
# Device, confidence thresholds, YOLO settings, torchvision transforms.

import torch
from torchvision import transforms

from clashai.config import (
    ADB_DELAY_TAP, ADB_DELAY_SCREENSHOT,
    ADB_DELAY_NAVIGATION, ADB_DELAY_MATCHMAKING,
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SCREEN_CONFIDENCE_THRESHOLD = 0.60
BUILDING_CONFIDENCE_THRESHOLD = 0.50
YOLO_CONF = 0.25
YOLO_IOU = 0.50
# YOLO buildings trained at imgsz=1600 (tools/train/train_yolo_buildings.py).
# Ultralytics defaults to 640 at predict if unset, halving resolution.
YOLO_BUILDINGS_IMGSZ = 1600

screen_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

building_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])
