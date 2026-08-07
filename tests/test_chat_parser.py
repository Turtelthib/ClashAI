"""Parsing du chat de clan : commandes et horodatages.

`social/chat/parser.py` est 100 % pur (str -> dict/int, zero dependance hors `re`)
et c'est le point d'entree des ordres humains -- celui qui deviendra le canal du
LocalLLMBrain en V5.4. Les corrections d'erreurs OCR y sont faites par regex :
exactement le genre de code qui regresse sans bruit.

Toutes les valeurs attendues ci-dessous ont ete relevees sur le code actuel,
pas deduites de la docstring.
"""

import pytest

from clashai.social.chat.parser import (
    parse_all_commands,
    parse_command,
    parse_timestamp,
)

# ---------------------------------------------------------------------------
# Commandes d'attaque
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("line, target", [
    ('@mini_pekka 3', 3),              # numero seul
    ('@mini_pekka attack 3', 3),       # verbe anglais
    ('@mini_pekka attaque 12', 12),    # verbe francais
    ('@mini_pekka atk 7', 7),          # abreviations
    ('@mini_pekka att 9', 9),
    ('@mini pekka 3', 3),              # underscore ecrit en espace
    ('mini_pekka 3', 3),               # sans le @
    ('mini pekka 50', 50),             # borne haute acceptee
    ('  @MINI_PEKKA   ATTACK   4  ', 4),  # casse + espaces
])
def test_attack_commands(line, target):
    assert parse_command(line) == {'type': 'attack', 'target': target}


@pytest.mark.parametrize("word", ['stop', 'arret', 'pause'])
def test_stop_synonyms(word):
    assert parse_command(f'@mini_pekka {word}') == {'type': 'stop'}


@pytest.mark.parametrize("word", ['status', 'etat', 'info'])
def test_status_synonyms(word):
    assert parse_command(f'@mini_pekka {word}') == {'type': 'status'}


@pytest.mark.parametrize("word", ['reset', 'clear', 'oublie', 'nouveau', 'new'])
def test_reset_synonyms(word):
    assert parse_command(f'@mini_pekka {word}') == {'type': 'reset'}


# ---------------------------------------------------------------------------
# Rejets — le chat est un input hostile (cf. ROADMAP V5.4, injection)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("line", [
    'attaque 3',            # pas de mention du bot -> on ignore
    'salut les gars',
    '@mini_pekka',          # mention seule, pas de commande
    '@mini_pekka blabla',   # commande inconnue
    '@mini_pekka 0',        # hors bornes basse
    '@mini_pekka 51',       # hors bornes haute
])
def test_lines_that_must_not_produce_a_command(line):
    assert parse_command(line) is None


def test_target_bounds_are_1_to_50():
    assert parse_command('@mini_pekka 1')['target'] == 1
    assert parse_command('@mini_pekka 50')['target'] == 50
    assert parse_command('@mini_pekka 0') is None
    assert parse_command('@mini_pekka 51') is None


def test_custom_bot_name_is_honoured():
    assert parse_command('@robot 4', bot_name='robot') == {'type': 'attack', 'target': 4}
    assert parse_command('@mini_pekka 4', bot_name='robot') is None


# ---------------------------------------------------------------------------
# parse_all_commands
# ---------------------------------------------------------------------------

def test_parse_all_keeps_only_commands_and_attaches_raw_text():
    lines = ['salut', '@mini_pekka 3', 'blabla', '@mini_pekka stop']

    commands = parse_all_commands(lines)

    assert [c['type'] for c in commands] == ['attack', 'stop']
    assert commands[0]['raw_text'] == '@mini_pekka 3'
    assert commands[0]['target'] == 3


def test_parse_all_on_empty_input():
    assert parse_all_commands([]) == []


# ---------------------------------------------------------------------------
# Horodatages CoC -> age en minutes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text, minutes", [
    ("A l'instant", 0),
    ("a l instant", 0),
    ('1min', 1),
    ('5min', 5),
    ('1h', 60),
    ('1h 22min', 82),
    ('14h 8min', 848),
    ('2j', 2880),
    ('1j 3h', 1620),
])
def test_timestamp_formats(text, minutes):
    assert parse_timestamp(text) == minutes


@pytest.mark.parametrize("ocr, minutes", [
    ('IImin', 11),   # I majuscule lu a la place de 1
    ('l1min', 11),   # l minuscule
    ('Ih', 60),      # I devant une unite
])
def test_ocr_confusions_are_repaired(ocr, minutes):
    """Le CNN/OCR confond 1, I, l et |. Ces reparations sont faites par regex
    AVANT le passage en minuscules -- une inversion de l'ordre les casserait."""
    assert parse_timestamp(ocr) == minutes


@pytest.mark.parametrize("text", ['bonjour', '', '   '])
def test_non_timestamps_return_none(text):
    assert parse_timestamp(text) is None


def test_days_hours_and_minutes_add_up():
    """Les trois unites se cumulent : c'est une somme, pas un choix."""
    assert parse_timestamp('1j 3h') == 1 * 24 * 60 + 3 * 60
    assert parse_timestamp('14h 8min') == 14 * 60 + 8
