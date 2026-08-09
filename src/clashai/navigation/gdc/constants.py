# clashai/navigation/gdc/constants.py
# UI button positions, target-list zone, scroll + wait timings.

# Re-imported from clashai/config (Phase A) — kept importable for back-compat.
from clashai.config import ADB_HEIGHT, ADB_WIDTH  # noqa: F401


def _get_ui_pos(name, screenshot=None):
    """Position d'un bouton de l'UI.

    Delegue a `perception.ui_buttons.find_button`, le point d'acces unique :
    detecteur CNN UI si installe ET si une frame est fournie, position calibree
    sinon.

    La table de defauts locale (17 cles recopiees a l'identique de
    `calibrate_ui.DEFAULT_POSITIONS`) a ete supprimee : duplication stricte, et
    le seul endroit du projet ou les defauts pouvaient diverger sans que
    personne ne le voie.
    """
    from clashai.perception.ui_buttons import find_button
    return find_button(name, screenshot=screenshot)


# Zone where enemy target numbers appear (the enemy list with their #).
TARGET_LIST_ZONE = {
    'left': 100,
    'right': 1820,
    'top': 150,
    'bottom': 850,
}

# Approximate number of CW targets visible at once on screen.
VISIBLE_TARGETS_PER_SCREEN = 5

# Scroll speed for navigating the list.
SCROLL_DISTANCE = 400
SCROLL_DURATION = 300

# Wait times between actions.
WAIT_NAVIGATION = 1.5
WAIT_MENU_LOAD = 2.0
WAIT_SCROLL = 1.0
WAIT_TARGET_LOAD = 2.0
WAIT_MATCHMAKING = 4.0

MAX_RETRIES = 15
