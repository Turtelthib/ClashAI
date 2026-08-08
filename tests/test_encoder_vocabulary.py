"""Coherence entre les noms de classes du CNN et ceux du code.

`weights/classes.json` est la seule verite : c'est ce que le modele emet
reellement. Un nom mal orthographie dans `encoder/constants.py` ne leve aucune
exception -- `CLASS_TO_CHANNEL` rate simplement la classe, le batiment n'entre
dans aucun canal de la grille RL et devient invisible pour l'agent.

C'est exactement ce qui est arrive a `double_canon`, ecrit `canon_double` dans
les CATEGORIES et dans DEFENSE_STATS : le canon double n'avait ni canal ni
statistique de danger. Ces tests empechent la recidive.
"""

import json

from clashai.combat.encoder.constants import (
    CATEGORIES,
    CLASS_TO_CHANNEL,
    DEFENSE_STATS,
)
from clashai.paths import WEIGHTS_DIR


def _cnn_classes():
    with open(f"{WEIGHTS_DIR}/classes.json", encoding='utf-8') as f:
        classes = json.load(f)
    return list(classes.values()) if isinstance(classes, dict) else list(classes)


def test_every_cnn_class_maps_to_a_grid_channel():
    """L'invariant qui compte : rien de ce que le CNN detecte ne doit tomber
    dans le vide."""
    unmapped = [c for c in _cnn_classes() if c not in CLASS_TO_CHANNEL]
    assert not unmapped, (
        f"classes emises par le CNN mais absentes de CLASS_TO_CHANNEL : "
        f"{unmapped} -- ces batiments sont invisibles pour l'agent RL"
    )


def test_defense_stats_cover_every_mapped_defense():
    """Une defense sans stats n'a pas de niveau de danger pour le ciblage."""
    defended = {
        name
        for category, names in CATEGORIES.items()
        if 'defense' in category
        for name in names
    }
    missing = sorted(n for n in defended if n not in DEFENSE_STATS)
    assert not missing, f"defenses sans DEFENSE_STATS : {missing}"


def test_category_names_are_not_duplicated_across_categories():
    """Un nom dans deux categories rendrait son canal ambigu."""
    seen = {}
    for category, names in CATEGORIES.items():
        for name in names:
            assert name not in seen, (
                f"'{name}' apparait dans '{seen.get(name)}' ET '{category}'"
            )
            seen[name] = category


def test_channel_indices_are_contiguous_from_zero():
    """Les canaux alimentent un tenseur : un trou decalerait la grille."""
    channels = sorted(set(CLASS_TO_CHANNEL.values()))
    assert channels == list(range(len(channels)))
