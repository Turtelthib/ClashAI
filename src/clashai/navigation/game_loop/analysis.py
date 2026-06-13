# clashai/navigation/game_loop/analysis.py
# classify_screen (screen-state CNN) + analyze_village (YOLO+building CNN).

import numpy as np
import torch

from clashai.perception.inference_lock import INFERENCE_LOCK
from clashai.navigation.game_loop.constants import (
    DEVICE, screen_transform, building_transform,
    YOLO_CONF, YOLO_IOU, YOLO_BUILDINGS_IMGSZ, BUILDING_CONFIDENCE_THRESHOLD,
)


def classify_screen(img_pil, models):
    """
    Determines the current screen state.
    Returns (state, confidence).
    """
    from clashai.perception.inference_lock import INFERENCE_LOCK
    tensor = screen_transform(img_pil).unsqueeze(0).to(DEVICE)
    with INFERENCE_LOCK, torch.no_grad():
        outputs = models['screen_cnn'](tensor)
        probs = torch.softmax(outputs, dim=1)
        idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][idx].item()

    state = models['screen_classes'][idx]
    return state, confidence


def analyze_village(img_pil, models):
    """
    Detects and classifies all buildings in the image.
    Returns a list of dicts {class, confidence, bbox, center}.
    """
    # YOLO detection — pass PIL directly. Ultralytics reads a numpy array
    # as BGR but a PIL image as RGB; np.array(rgb_pil) would swap R/B
    # channels. img_np is kept only for the clamp bounds below.
    img_np = np.array(img_pil)
    from clashai.perception.inference_lock import INFERENCE_LOCK
    with INFERENCE_LOCK:
        results = models['yolo'].predict(
            img_pil, conf=YOLO_CONF, iou=YOLO_IOU,
            imgsz=YOLO_BUILDINGS_IMGSZ, verbose=False,
        )
    
    buildings = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Clamp
        h, w = img_np.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # Crop and CNN classification
        crop = img_pil.crop((x1, y1, x2, y2))
        tensor = building_transform(crop).unsqueeze(0).to(DEVICE)

        from clashai.perception.inference_lock import INFERENCE_LOCK
        with INFERENCE_LOCK, torch.no_grad():
            outputs = models['building_cnn'](tensor)
            probs = torch.softmax(outputs, dim=1)
            idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0][idx].item()

        label = models['building_classes'][idx]

        # Filter out useless classes and low confidence
        if label in ('useless', 'ignore'):
            continue
        if confidence < BUILDING_CONFIDENCE_THRESHOLD:
            continue

        buildings.append({
            'class': label,
            'confidence': confidence,
            'bbox': (x1, y1, x2, y2),
            'center': ((x1 + x2) // 2, (y1 + y2) // 2)
        })

    return buildings


def get_village_summary(buildings):
    """Human-readable summary of detected buildings."""
    counts = {}
    for b in buildings:
        counts[b['class']] = counts.get(b['class'], 0) + 1

    # Sort by type: defenses first, then resources, then others
    defenses = ['hdv', 'tour_enfer_mono', 'tour_enfer_multiple', 'aigle_artilleur',
                'catapulte_erratique', 'arcX_sol', 'arcX_sol_air', 'monolithe',
                'tour_archere', 'canon', 'mortier', 'multi_mortier', 'tour_sorcier',
                'defense_antiaerienne', 'prop_air', 'tesla', 'canon_ricochet',
                'cracheur_feu', 'tour_runique_rage', 'tour_runique_poison',
                'tour_runique_invisible', 'tour_multi_equipe_rapide', 'tour_bombe',
                'tour_archere_multiple', 'tour_multi_equipe_lente', 'tour_archere_rapide',
                'canon_double', 'tour_vengeuse', 'super_tour_sorcier', 'gigabombe',
                'tour_runique_seisme', 'cabane_ouvrier_arme']

    ressources = ['reserve_or', 'reserve_elixir', 'reserve_noire', 'ressources']

    summary = {
        'total': len(buildings),
        'defenses': sum(counts.get(d, 0) for d in defenses),
        'ressources': sum(counts.get(r, 0) for r in ressources),
        'details': counts
    }
    return summary

