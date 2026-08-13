# clashai/perception/ui_detector.py
# Détecteur universel de boutons / éléments d'interface (CNN UI, V5.2).
#
# Le modèle YOLO "UI" (124 classes, mAP50 0.957, mAP50-95 0.822) reconnaît les
# boutons par leur apparence + leur TEXTE (entraîné à imgsz 1280). Ce module
# l'enveloppe et expose deux niveaux :
#
#   detect_raw(screenshot) -> {classe_cnn: [Detection, ...]}   (français, brut)
#   detect(screenshot)     -> {cle: (x, y, conf)}              (compat find_button)
#
# `detect()` est le contrat attendu par ui_buttons.set_detector() : find_button()
# l'essaie D'ABORD et retombe sur la position calibrée si rien (ou confiance <
# seuil). La bascule V5.2 est donc un seul set_detector() au démarrage — aucun
# appelant de find_button() à modifier (cf. ui_buttons.py).
#
# Désambiguïsation des boutons génériques
# ---------------------------------------
# `fermer` / `confirmer` / `suivant` apparaissent à plusieurs endroits selon
# l'écran (close_profil en haut-droite vs close_popup au centre). La classe CNN
# seule ne les distingue pas — elle voit "une croix". Pour une clé de
# calibration on choisit donc l'instance la PLUS PROCHE de sa position calibrée :
# close_profil et close_popup se démêlent tout seuls, sans classe dédiée.

import os
from dataclasses import dataclass

from clashai.config import ADB_HEIGHT, ADB_WIDTH
from clashai.navigation.calibrate_ui import DEFAULT_POSITIONS, get_position
from clashai.paths import WEIGHTS_DIR

# =============================================================================
# CONFIGURATION
# =============================================================================

# Le modèle est cherché à l'emplacement canonique puis là où l'entraînement
# Kaggle le dépose (weights/yolo_ui_cnn/yolo_ui_best.pt). Renomme-le en
# weights/yolo_ui.pt pour t'aligner sur yolo_troops.pt si tu veux.
_UI_WEIGHTS_CANDIDATES = [
    os.path.join(WEIGHTS_DIR, 'yolo_ui.pt'),
    os.path.join(WEIGHTS_DIR, 'yolo_ui_cnn', 'yolo_ui_best.pt'),
    os.path.join(WEIGHTS_DIR, 'yolo_ui_cnn', 'best.pt'),
]

# Seuil de confiance à l'inférence. La courbe F1 pique à ~0.48 (F1 0.83) ; on
# descend à 0.40 pour aussi nourrir detect_raw (les agents gatent eux-mêmes).
# find_button re-filtre à 0.60 (DETECTOR_MIN_CONFIDENCE) avant de faire confiance
# au modèle plutôt qu'à la calibration.
DEFAULT_CONF = 0.40

# Le modèle UI est entraîné à imgsz 1280 (texte des boutons). Le fixer ici : un
# futur ré-entraînement à une autre résolution ne touche que cette constante.
YOLO_UI_IMGSZ = 1280


# Table de correspondance : clé de calibration (anglais, historique) -> classe
# CNN (français, visuelle). Ne mappe QUE les boutons sûrs ; une clé absente
# retombe simplement sur sa position calibrée via find_button (pas de régression).
#
# ⚠️ Les entrées marquées "À CONFIRMER" sont mes hypothèses : à valider sur un
# vrai écran (le mauvais mapping = tap au mauvais endroit).
KEY_TO_CLASS = {
    # --- coeur de boucle : cycle d'attaque ---
    'attack_button':     'attaquer',              # ATTAQUER du village (bas-gauche)
    'start_attack':      'lancer_attaque',        # "LANCER L'ATTAQUE" (confirme)
    'return_home':       'rentrer',               # "RENTRER AU VILLAGE" (résultats)
    'find_match':        'trouver_partie_rapide',  # recherche rapide (confirmé)
    # --- retraite / abandon ---
    # NB: pas de mapping ff_button ici — l'abandon est state-dependent (capituler
    # si troupes vivantes, sinon terminer_bataille) et géré dans env._surrender()
    # qui lit directement les classes CNN. On garde juste la confirmation :
    'confirm_ff':        'confirmer',             # confirmation de l'abandon (popup)
    # --- château / renfort ---
    'cdc_confirmation':  'confirmer',             # confirmation demande de renfort
    # --- chat ---
    'chat_send':         'envoyer_message',       # flèche verte "envoyer"
    # --- GdC ---
    'gdc_open':          'guerre_clan',           # À CONFIRMER (accès menu GdC)
    'gdc_enemy_map':     'voir_enemis',           # carte ennemie
    'gdc_ally_map':      'voir_allie',            # carte alliée
    'gdc_attack_target': 'attaquer_guerre',       # ATTAQUER dans le popup GdC
    'gdc_village_next':  'village_suivant',       # flèche suivant (village n+1)
    'gdc_village_prev':  'village_precedent',     # flèche précédent (village n-1)
    # --- fermetures génériques (désambiguïsées par position) ---
    'close_profil':      'fermer',
    'close_menu':        'fermer',
    'close_popup':       'fermer',
}


# =============================================================================
# DATACLASS
# =============================================================================

@dataclass
class Detection:
    """Une détection de bouton/élément UI (coords en ADB 1920×1080)."""
    class_name: str
    class_id: int
    x: int
    y: int
    w: int
    h: int
    conf: float


# =============================================================================
# UI DETECTOR
# =============================================================================

class UIDetector:
    """Détecteur de boutons d'interface basé sur le CNN UI (YOLO 124 classes).

    Deux API :
      - detect_raw(img) -> {classe: [Detection]}  (brut, noms CNN français)
      - detect(img)     -> {cle: (x, y, conf)}    (contrat ui_buttons.set_detector)
    """

    def __init__(self, weights_path: str = None, conf: float = DEFAULT_CONF,
                 verbose: bool = True):
        self.conf = conf
        self.verbose = verbose
        self._model = None
        self._weights_path = weights_path or self._resolve_weights()

    @staticmethod
    def _resolve_weights() -> str:
        for p in _UI_WEIGHTS_CANDIDATES:
            if os.path.exists(p):
                return p
        return _UI_WEIGHTS_CANDIDATES[0]  # message d'erreur pointera dessus

    def _load_model(self):
        if self._model is not None:
            return
        if not os.path.exists(self._weights_path):
            raise FileNotFoundError(
                f"Modèle CNN UI introuvable : {self._weights_path}\n"
                f"Entraîne-le avec src/tools/train/kaggle_train_yolo_ui.py, "
                f"puis place le best.pt en weights/yolo_ui.pt."
            )
        from ultralytics import YOLO
        self._model = YOLO(self._weights_path)
        if self.verbose:
            from clashai.config.logging import pp
            pp(f" CNN UI chargé : {self._weights_path} "
               f"({len(self._model.names)} classes)", tag='yolo')

    # ---- inférence brute (partagée) ----------------------------------------

    def _infer(self, screenshot_pil):
        """Lance le modèle sous le lock d'inférence et rend les résultats YOLO."""
        self._load_model()
        from clashai.perception.inference_lock import INFERENCE_LOCK
        with INFERENCE_LOCK:
            return self._model(
                screenshot_pil, conf=self.conf,
                imgsz=YOLO_UI_IMGSZ, verbose=False,
            )

    # ---- niveau brut : par classe CNN (français) ---------------------------

    def detect_raw(self, screenshot_pil) -> dict:
        """Toutes les détections groupées par classe CNN.

        Returns:
            {classe: [Detection, ...]} — chaque liste triée par confiance ↓.
        """
        img_w, img_h = screenshot_pil.size
        scale_x = ADB_WIDTH / img_w
        scale_y = ADB_HEIGHT / img_h

        results = self._infer(screenshot_pil)

        out: dict[str, list[Detection]] = {}
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = int((x1 + x2) / 2 * scale_x)
                cy = int((y1 + y2) / 2 * scale_y)
                w = int((x2 - x1) * scale_x)
                h = int((y2 - y1) * scale_y)
                name = self._model.names.get(cls_id, f"unk_{cls_id}")
                out.setdefault(name, []).append(Detection(
                    class_name=name, class_id=cls_id,
                    x=cx, y=cy, w=w, h=h, conf=conf,
                ))

        for dets in out.values():
            dets.sort(key=lambda d: d.conf, reverse=True)

        if self.verbose and out:
            from clashai.config.logging import pp, styled
            summary = ', '.join(f"{len(v)}×{k}" for k, v in out.items())
            pp(f" CNN UI: {styled(summary, 'yolo_alt')}", tag='yolo')

        return out

    # ---- niveau compat : contrat find_button (cle -> (x,y,conf)) -----------

    def detect(self, screenshot_pil) -> dict:
        """Détections indexées à la fois par classe CNN ET par clé de calibration.

        - clé = nom de classe CNN  -> meilleure instance (conf max).
        - clé de calibration (KEY_TO_CLASS) -> instance désambiguïsée par
          proximité à la position calibrée (pour les boutons génériques).

        C'est le format attendu par ui_buttons.set_detector().
        """
        raw = self.detect_raw(screenshot_pil)
        out: dict[str, tuple] = {}

        # 1. accès direct par nom de classe (agents village/GdC en français)
        for cls, dets in raw.items():
            best = dets[0]  # déjà trié conf ↓
            out[cls] = (best.x, best.y, best.conf)

        # 2. clés de calibration historiques (compat find_button)
        for key, cls in KEY_TO_CLASS.items():
            dets = raw.get(cls)
            if not dets:
                continue
            if len(dets) == 1:
                d = dets[0]
            elif key in DEFAULT_POSITIONS:
                cx, cy = get_position(key)
                d = min(dets, key=lambda t: (t.x - cx) ** 2 + (t.y - cy) ** 2)
            else:
                d = dets[0]  # conf max faute de repère de position
            out[key] = (d.x, d.y, d.conf)

        return out

    # ---- helpers ------------------------------------------------------------

    def find(self, name: str, screenshot_pil):
        """(x, y, conf) du meilleur match pour `name` (classe OU clé), ou None."""
        return self.detect(screenshot_pil).get(name)

    def find_all(self, class_name: str, screenshot_pil) -> list:
        """Toutes les Detection d'une classe CNN (ex. tous les `membre_clan`)."""
        return self.detect_raw(screenshot_pil).get(class_name, [])

    def annotate(self, screenshot_pil, out_path: str) -> str:
        """Dessine les boîtes + classe + confiance sur l'image et l'enregistre.

        Utilise le rendu natif d'ultralytics (`results.plot()`, noms = model.names)
        pour visualiser exactement ce que le CNN voit. Renvoie `out_path`.
        """
        import cv2

        results = self._infer(screenshot_pil)
        annotated_bgr = results[0].plot(line_width=2)  # ndarray BGR, prêt à écrire
        cv2.imwrite(out_path, annotated_bgr)
        return out_path


def install(**kwargs) -> UIDetector:
    """Instancie le détecteur et le branche dans find_button (un seul appel).

    À placer au démarrage de l'app quand on veut activer la V5.2 :
        from clashai.perception.ui_detector import install
        install()
    """
    from clashai.perception import ui_buttons
    det = UIDetector(**kwargs)
    ui_buttons.set_detector(det)
    return det


# =============================================================================
# TEST MANUEL
# =============================================================================

if __name__ == "__main__":
    import sys

    from PIL import Image

    if len(sys.argv) < 2:
        print("Usage: python -m clashai.perception.ui_detector <screenshot.png>")
        sys.exit(0)

    img = Image.open(sys.argv[1]).convert('RGB')
    det = UIDetector()
    raw = det.detect_raw(img)

    print(f"\n{sum(len(v) for v in raw.values())} détections "
          f"sur {len(raw)} classes :\n")
    for cls in sorted(raw, key=lambda c: -raw[c][0].conf):
        for d in raw[cls]:
            print(f"  {d.class_name:28s} ({d.x:4d}, {d.y:4d})  conf={d.conf:.2f}")

    mapped = {k: v for k, v in det.detect(img).items() if k in KEY_TO_CLASS}
    if mapped:
        print("\nMapping clés de calibration -> détection :")
        for k, (x, y, c) in mapped.items():
            print(f"  {k:20s} = ({x:4d}, {y:4d})  conf={c:.2f}  [{KEY_TO_CLASS[k]}]")

    # Image annotée (boîtes + classe + conf) à côté de l'entrée
    base, _ = os.path.splitext(sys.argv[1])
    out_path = f"{base}_annotated.png"
    det.annotate(img, out_path)
    print(f"\nImage annotée -> {out_path}")
