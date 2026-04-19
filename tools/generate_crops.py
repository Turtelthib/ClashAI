from ultralytics import YOLO
import cv2
import os
import glob
from tqdm import tqdm
import shutil

from clashai.paths import PROJECT_ROOT as project_root

model_path = os.path.join(project_root, 'weights', 'best.pt')
source_images = os.path.join(project_root, 'dataset', 'images', 'train')
output_dir = os.path.join(project_root, 'dataset_cnn')

conf_threshold = 0.4

print(f"Loading model: {model_path}")
model = YOLO(model_path)

if os.path.exists(output_dir):
    print(f"Cleaning existing folder: {output_dir}")
    shutil.rmtree(output_dir)

os.makedirs(output_dir)

img_files = glob.glob(os.path.join(source_images, "*.jpg")) + glob.glob(os.path.join(source_images, "*.png"))
print(f"Extracting from {len(img_files)} images...")

count = 0
for img_path in tqdm(img_files):
    results = model.predict(img_path, conf=conf_threshold, verbose=False)
    result = results[0]
    img_cv = cv2.imread(img_path)
    
    if img_cv is None: continue

    filename = os.path.splitext(os.path.basename(img_path))[0]

    for i, box in enumerate(result.boxes):
        cls_id = int(box.cls[0])
        label_name = model.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        h, w, _ = img_cv.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        crop = img_cv[y1:y2, x1:x2]
        if crop.size == 0: continue
        
        class_dir = os.path.join(output_dir, label_name)
        os.makedirs(class_dir, exist_ok=True)
        
        save_path = os.path.join(class_dir, f"{filename}_{i}.jpg")
        cv2.imwrite(save_path, crop)
        count += 1

print(f"Done! {count} crops extracted to '{output_dir}'")