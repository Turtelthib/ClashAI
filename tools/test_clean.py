from ultralytics import YOLO
import cv2
import os
import random

model_path = os.path.join('runs', 'detect', 'FinishedTrain', 'weights', 'best.pt')
model = YOLO(model_path)

image_path = 'test_img/test_hdv18_1.png'
print(f"Analyse de {image_path}...")

results = model.predict(source=image_path, conf=0.50, iou=0.45, save=False, show=False)
result = results[0]

img = cv2.imread(image_path)

detected_items = {}
colors = {}

for box in result.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    class_id = int(box.cls[0])
    label_name = model.names[class_id]
    
    if class_id not in colors:
        random.seed(class_id * 5)
        colors[class_id] = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
    
    color = colors[class_id]
    detected_items[label_name] = color
    
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

if detected_items:
    padding = 10
    line_height = 30
    box_width = 300
    box_height = (len(detected_items) * line_height) + padding * 2
    
    overlay = img.copy()
    
    cv2.rectangle(overlay, (10, 10), (10 + box_width, 10 + box_height), (255, 255, 255), -1)
    
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    x_text = 20
    y_text = 40
    
    for name, color in sorted(detected_items.items()):
        cv2.rectangle(img, (x_text, y_text - 15), (x_text + 20, y_text + 5), color, -1)
        
        cv2.putText(img, name, (x_text + 30, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        y_text += line_height

output_path = 'Test_clean.png'
cv2.imwrite(output_path, img)

print("-" * 30)
print(f"Image générée : {output_path}")
print("-" * 30)