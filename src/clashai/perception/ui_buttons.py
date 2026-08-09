# clashai/perception/ui_buttons.py
# Point d'acces UNIQUE aux positions de boutons de l'interface.
#
# Pourquoi ce module existe
# -------------------------
# Avant, chaque appelant se debrouillait : import direct de `get_position`,
# petite table de defauts locale (navigation/gdc/constants.py en avait une copie
# de 17 cles), ou coordonnees en dur. ~41 sites au total. Impossible d'y brancher
# un detecteur YOLO sans les toucher tous.
#
# Ici, un seul chemin : find_button(name) -> (x, y).
#
# La bascule vers le CNN UI (V5.2)
# --------------------------------
# `set_detector()` installe un detecteur ; `find_button` l'essaie D'ABORD et
# retombe sur la position calibree s'il ne trouve rien (ou si sa confiance est
# trop basse). Le jour ou le modele est pret, la migration est un seul appel a
# set_detector() au demarrage -- aucun appelant a modifier.
#
# Le fallback n'est pas transitoire : un bouton hors champ, un ecran inattendu ou
# un modele non charge doivent rester rattrapables par la calibration.

from clashai.navigation.calibrate_ui import DEFAULT_POSITIONS, get_position

# Confiance minimale pour faire confiance au detecteur plutot qu'a la calibration.
DETECTOR_MIN_CONFIDENCE = 0.60

_detector = None


def set_detector(detector):
    """Installe le detecteur de boutons (ou None pour revenir a la calibration).

    `detector` doit exposer :
        detect(screenshot) -> {name: (x, y, confidence)}

    C'est le seul point d'entree de la V5.2 : aucun appelant de find_button()
    n'a besoin de savoir qu'un modele existe.
    """
    global _detector
    _detector = detector


def get_detector():
    return _detector


def known_buttons():
    """Noms de boutons connus de la calibration (surface de reference)."""
    return set(DEFAULT_POSITIONS)


def find_button(name, screenshot=None, min_confidence=DETECTOR_MIN_CONFIDENCE):
    """Position (x, y) d'un bouton, detectee si possible, calibree sinon.

    Args:
        name: cle du bouton (ex. 'attack_button', 'chat_open')
        screenshot: frame a analyser. Sans elle, le detecteur est saute --
            find_button reste donc utilisable hors ecran, comme avant.
        min_confidence: seuil sous lequel une detection est ignoree.

    Returns:
        (x, y) — toujours. Un nom inconnu tombe sur le centre de l'ecran via
        get_position(), comportement historique conserve.
    """
    if _detector is not None and screenshot is not None:
        hit = _detect_one(name, screenshot)
        if hit is not None:
            x, y, confidence = hit
            if confidence >= min_confidence:
                return (int(x), int(y))

    return get_position(name)


def _detect_one(name, screenshot):
    """Interroge le detecteur pour un bouton. None si absent ou en echec.

    Un detecteur qui plante ne doit jamais bloquer la navigation : on log et on
    laisse find_button() retomber sur la calibration.
    """
    try:
        detections = _detector.detect(screenshot)
    except Exception as e:
        print(f"WARNING: UI detector failed on '{name}' ({e}) -> calibrated position")
        return None

    if not detections:
        return None
    return detections.get(name)
