"""Le point d'acces unique aux positions de boutons.

`perception/ui_buttons.find_button()` est le seam ou se branchera le CNN UI de
la V5.2. Ces tests verrouillent le contrat AVANT que le modele existe, pour que
la bascule soit un simple `set_detector()` et pas une migration de 41 appelants.

L'exigence centrale : le fallback calibre doit rattraper TOUS les cas de figure
du detecteur (absent, muet, pas confiant, en panne). Une navigation ne doit
jamais s'arreter parce qu'une inference a rate.
"""

import pytest

from clashai.navigation.calibrate_ui import DEFAULT_POSITIONS, get_position
from clashai.perception import ui_buttons


class FakeDetector:
    """Detecteur factice. `detections` = {name: (x, y, confidence)}."""

    def __init__(self, detections=None, raises=False):
        self._detections = detections or {}
        self._raises = raises
        self.calls = 0

    def detect(self, screenshot):
        self.calls += 1
        if self._raises:
            raise RuntimeError("modele non charge")
        return self._detections


@pytest.fixture(autouse=True)
def _reset_detector():
    """Le detecteur est un etat de module : on le remet a None apres chaque test."""
    yield
    ui_buttons.set_detector(None)


# ---------------------------------------------------------------------------
# Sans detecteur : comportement historique
# ---------------------------------------------------------------------------

def test_falls_back_to_the_calibrated_position():
    assert ui_buttons.find_button('attack_button') == get_position('attack_button')


def test_unknown_button_still_returns_a_usable_point():
    """get_position() renvoie le centre de l'ecran pour une cle inconnue.
    find_button ne doit pas lever : la navigation prefere un tap inutile a un crash."""
    position = ui_buttons.find_button('bouton_qui_nexiste_pas')
    assert isinstance(position, tuple) and len(position) == 2


def test_a_screenshot_without_detector_changes_nothing():
    assert ui_buttons.find_button('chat_open', screenshot=object()) == \
        get_position('chat_open')


# ---------------------------------------------------------------------------
# Avec detecteur
# ---------------------------------------------------------------------------

def test_confident_detection_wins_over_calibration():
    ui_buttons.set_detector(FakeDetector({'attack_button': (123, 456, 0.9)}))

    assert ui_buttons.find_button('attack_button', screenshot=object()) == (123, 456)


def test_detector_is_skipped_without_a_screenshot():
    """Beaucoup d'appelants n'ont pas de frame sous la main. Ils doivent continuer
    a fonctionner exactement comme avant."""
    detector = FakeDetector({'attack_button': (123, 456, 0.9)})
    ui_buttons.set_detector(detector)

    assert ui_buttons.find_button('attack_button') == get_position('attack_button')
    assert detector.calls == 0


def test_low_confidence_detection_is_ignored():
    ui_buttons.set_detector(FakeDetector({'attack_button': (123, 456, 0.1)}))

    assert ui_buttons.find_button('attack_button', screenshot=object()) == \
        get_position('attack_button')


def test_button_absent_from_the_detection_falls_back():
    ui_buttons.set_detector(FakeDetector({'chat_open': (1, 2, 0.99)}))

    assert ui_buttons.find_button('attack_button', screenshot=object()) == \
        get_position('attack_button')


def test_a_crashing_detector_never_breaks_navigation():
    ui_buttons.set_detector(FakeDetector(raises=True))

    assert ui_buttons.find_button('attack_button', screenshot=object()) == \
        get_position('attack_button')


def test_detection_coordinates_are_returned_as_ints():
    """Un tap ADB attend des entiers ; YOLO rend des flottants."""
    ui_buttons.set_detector(FakeDetector({'attack_button': (12.7, 34.2, 0.9)}))

    x, y = ui_buttons.find_button('attack_button', screenshot=object())
    assert isinstance(x, int) and isinstance(y, int)


def test_threshold_is_tunable_per_call():
    ui_buttons.set_detector(FakeDetector({'attack_button': (123, 456, 0.3)}))

    assert ui_buttons.find_button(
        'attack_button', screenshot=object(), min_confidence=0.2
    ) == (123, 456)


# ---------------------------------------------------------------------------
# Une seule table de defauts
# ---------------------------------------------------------------------------

def test_gdc_constants_delegate_instead_of_keeping_their_own_table():
    """navigation/gdc/constants.py recopiait 17 cles de DEFAULT_POSITIONS.
    C'etait le seul endroit ou les defauts pouvaient diverger en silence."""
    from clashai.navigation.gdc import constants

    for name in ('chat_open', 'gdc_open', 'attack_button', 'return_home'):
        assert constants._get_ui_pos(name) == get_position(name)


def test_cdc_confirmation_is_known_to_the_package():
    """Le bouton n'existait que dans le fork tools/setup : la production le lisait
    (social/clan_castle.py) sans jamais pouvoir le recalibrer."""
    assert 'cdc_confirmation' in DEFAULT_POSITIONS
    assert 'cdc_confirmation' in ui_buttons.known_buttons()


def test_every_calibratable_button_has_a_default():
    """Un bouton calibrable sans defaut tomberait sur le centre de l'ecran si le
    JSON venait a manquer."""
    from clashai.navigation.calibrate_ui import BUTTONS_TO_CALIBRATE

    missing = [
        entry[0]
        for group in BUTTONS_TO_CALIBRATE.values()
        for entry in group
        if entry[0] not in DEFAULT_POSITIONS
    ]
    assert not missing, f"boutons calibrables sans position par defaut : {missing}"
