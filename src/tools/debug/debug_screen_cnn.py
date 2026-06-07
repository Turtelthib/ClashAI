# tools/debug/debug_screen_cnn.py
# Shows exactly what the screen CNN sees and what it classifies.
# Run this to diagnose screen classification issues.
#
# Usage:
#   uv run python tools/debug/debug_screen_cnn.py

import os, sys, json, torch
from torchvision import transforms
from PIL import Image

from clashai.paths import PROJECT_ROOT as project_root

from clashai.paths import WEIGHTS_DIR
from clashai.perception.screen_classifier import MyCustomCNN

JSON_PATH  = os.path.join(WEIGHTS_DIR, 'classification', 'screen_classes.json')
MODEL_PATH = os.path.join(WEIGHTS_DIR, 'classification', 'cnn_screen_classification.pt')
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
with open(JSON_PATH) as f:
    meta = json.load(f)
classes = meta if isinstance(meta, list) else meta.get('classes', meta)

model = MyCustomCNN(num_classes=len(classes))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

# --- Capture ---
from clashai.perception.screen_capture import ScreenCapture, find_emulator_bbox

print("=== Screen capture debug ===\n")
bbox, title, _hwnd = find_emulator_bbox()
print(f"Window found : {title}")
print(f"Bbox         : {bbox}\n")

cap = ScreenCapture()
print(f"Backend      : {cap.backend}\n")

img = cap.grab()
if img is None:
    print("ERROR: capture returned None")
    sys.exit(1)

# Save captured frame
out = os.path.join(project_root, '_debug_capture.png')
img.save(out)
print(f"Captured frame saved: {out}")
print(f"Size: {img.size}\n")

# --- Classify ---
tensor = transform(img).unsqueeze(0).to(DEVICE)
with torch.no_grad():
    outputs = model(tensor)
    probs = torch.softmax(outputs, dim=1)[0]

# Show all class probabilities sorted
results = sorted(
    [(classes[i], float(probs[i])) for i in range(len(classes))],
    key=lambda x: -x[1]
)

print("=== Classification results ===")
for cls, prob in results:
    bar = '#' * int(prob * 40)
    marker = " <-- PREDICTED" if prob == results[0][1] else ""
    print(f"  {cls:<25s} {prob*100:5.1f}%  {bar}{marker}")
