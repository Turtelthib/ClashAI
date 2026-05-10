# tools/test_troop_bar_webcam.py
# Live webcam test for the troop bar YOLO model.
# Uses tkinter + PIL for display (works with opencv-headless).
#
# Usage:
#   uv run python tools/test_troop_bar_webcam.py
#   uv run python tools/test_troop_bar_webcam.py --conf 0.3
#   uv run python tools/test_troop_bar_webcam.py --camera 1
#
# Controls:
#   Q / ESC : quit
#   S       : save current frame to _webcam_capture.png
#   +/-     : confidence up/down

import os
import sys
import argparse
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
import cv2

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

MODEL_PATH = os.path.join(project_root, 'weights', 'yolo_troupes_barre', 'troop_bar.pt')
# No forced INFER_SIZE — let YOLO use the native image resolution.
# Upscaling a low-res webcam to 1280 adds no information and hurts accuracy.

HEROES = {'roi', 'reine', 'grand_gardien', 'championne', 'prince_gargouille', 'duc_draconique'}
SPELLS = {'soin', 'rage', 'gel', 'zap', 'saut', 'clone', 'invisible', 'rappel',
          'resurrection', 'totem', 'poison', 'seisme', 'speed', 'squelette',
          'chauve_souris', 'floraison', 'bloc_glace'}


def get_color(name):
    if '_deploye' in name: return (255, 50, 50)
    if '_capa'    in name: return (255, 165, 0)
    if name in HEROES:     return (255, 215, 0)
    if name in SPELLS:     return (180, 60, 255)
    return (50, 200, 50)


def draw_on_pil(pil_img, results, conf_threshold):
    draw = ImageDraw.Draw(pil_img)
    r = results[0]
    names = r.names
    count = 0

    for box in r.boxes:
        conf = float(box.conf[0])
        if conf < conf_threshold:
            continue
        count += 1
        cls  = int(box.cls[0])
        name = names.get(cls, f'cls{cls}')
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        color = get_color(name)

        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f'{name} {conf:.0%}'
        tw = len(label) * 7 + 4
        draw.rectangle([x1, y1 - 18, x1 + tw, y1], fill=color)
        draw.text((x1 + 2, y1 - 16), label, fill=(255, 255, 255))

    return pil_img, count


class WebcamApp:
    def __init__(self, root, model, cap, args):
        self.root = root
        self.model = model
        self.cap = cap
        self.conf = args.conf
        self.running = True
        self.current_frame = None

        root.title('ClashAI - Troop Bar Detection')
        root.configure(bg='black')

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas = tk.Canvas(root, width=w, height=h, bg='black', highlightthickness=0)
        self.canvas.pack()

        self.status = tk.StringVar(value='Loading...')
        tk.Label(root, textvariable=self.status, bg='black', fg='white',
                 font=('Consolas', 11)).pack(fill='x')
        tk.Label(root, text='Q=quit  S=save  +/-=confidence',
                 bg='black', fg='gray', font=('Consolas', 9)).pack()

        root.bind('<q>', self.quit)
        root.bind('<Escape>', self.quit)
        root.bind('s', self.save)
        root.bind('+', self.conf_up)
        root.bind('=', self.conf_up)
        root.bind('-', self.conf_down)
        root.protocol('WM_DELETE_WINDOW', self.quit)

        self.update()

    def update(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.root.after(50, self.update)
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        results = self.model.predict(rgb, conf=self.conf, verbose=False)
        pil_img, count = draw_on_pil(pil_img, results, self.conf)

        self.current_frame = pil_img
        self.status.set(f'Detections: {count}   Confidence: {self.conf:.0%}')

        tk_img = ImageTk.PhotoImage(pil_img)
        self.canvas.create_image(0, 0, anchor='nw', image=tk_img)
        self.canvas._tk_img = tk_img

        self.root.after(30, self.update)

    def save(self, _=None):
        if self.current_frame:
            path = os.path.join(project_root, '_webcam_capture.png')
            self.current_frame.save(path)
            print(f'Saved: {path}')

    def conf_up(self, _=None):
        self.conf = min(0.95, round(self.conf + 0.05, 2))
        print(f'Confidence: {self.conf:.0%}')

    def conf_down(self, _=None):
        self.conf = max(0.05, round(self.conf - 0.05, 2))
        print(f'Confidence: {self.conf:.0%}')

    def quit(self, _=None):
        self.running = False
        self.cap.release()
        self.root.destroy()


def run_image_mode(model, image_path, conf):
    """Static image mode — shows detections on a single image, press any key to close."""
    if not os.path.exists(image_path):
        print(f'ERROR: Image not found: {image_path}')
        sys.exit(1)

    img = Image.open(image_path).convert('RGB')
    print(f'Image: {img.size[0]}x{img.size[1]}')

    results = model.predict(img, conf=conf, verbose=False)
    img, count = draw_on_pil(img, results, conf)

    print(f'Detections: {count}')
    for box in results[0].boxes:
        c = float(box.conf[0])
        if c < conf:
            continue
        cls  = int(box.cls[0])
        name = results[0].names.get(cls, f'cls{cls}')
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        print(f'  {name:25s} {c:.0%}  ({x1},{y1})-({x2},{y2})')

    # Save annotated result
    out = os.path.join(project_root, '_test_detection.png')
    img.save(out)
    print(f'Saved: {out}')

    # Show in tkinter window (press any key or close to quit)
    root = tk.Tk()
    root.title(f'ClashAI - {os.path.basename(image_path)} - {count} detections')
    tk_img = ImageTk.PhotoImage(img)
    tk.Label(root, image=tk_img).pack()
    tk.Label(root, text='Close window to quit', font=('Consolas', 9), fg='gray').pack()
    root.mainloop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf',   type=float, default=0.45)
    parser.add_argument('--camera', type=int,   default=0)
    parser.add_argument('--image',  type=str,   default=None,
                        help='Path to an image file — skips webcam, tests on static image')
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        print(f'ERROR: Model not found: {MODEL_PATH}')
        sys.exit(1)

    print(f'Loading model: {MODEL_PATH}')
    from ultralytics import YOLO
    model = YOLO(MODEL_PATH)
    print(f'Classes: {len(model.names)} -> {list(model.names.values())}')

    # Image mode
    if args.image:
        run_image_mode(model, args.image, args.conf)
        return

    # Webcam mode
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f'ERROR: Cannot open camera {args.camera}')
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'Camera: {w}x{h}')
    print('Controls: Q/ESC=quit  S=save  +/-=confidence')

    root = tk.Tk()
    WebcamApp(root, model, cap, args)
    root.mainloop()


if __name__ == '__main__':
    main()
