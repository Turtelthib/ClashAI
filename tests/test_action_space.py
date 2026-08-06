"""Bijection encode/decode de l'espace d'action.

Reflete le bloc __main__ de combat/action_space.py (laisse intact).

Pourquoi ca compte : tous les indices sont DERIVES
(ACTION_ABILITY_START = ACTION_SPELL_START + NUM_SPELLS). Ajouter un sort dans
configs/troops.json decale donc toute la fin de l'espace d'action. Si la bijection
casse, l'agent RL apprend sur des indices decales -- en silence, sans exception.
"""

from clashai.combat.action_space import (
    ACTION_DONE,
    ACTION_OBSERVE,
    ACTION_SPELL_START,
    DEPLOY_ROLES,
    DEPLOY_SECTORS,
    HERO_NAMES,
    NUM_SPELLS,
    TOTAL_ACTIONS,
    decode_action,
    encode_action,
)


def test_encode_decode_round_trip_over_the_whole_space():
    for action in range(TOTAL_ACTIONS):
        kind, i1, i2 = decode_action(action)
        assert encode_action(kind, i1, i2) == action, (
            f"aller-retour casse sur l'action {action} ({kind}, {i1}, {i2})"
        )


def test_every_action_decodes_to_a_known_kind():
    kinds = {decode_action(a)[0] for a in range(TOTAL_ACTIONS)}
    assert kinds <= {'deploy', 'spell', 'ability', 'observe', 'wait_short',
                     'wait_long', 'done'}


def test_layout_is_derived_not_hardcoded():
    """Les bornes doivent decouler des donnees, pas de constantes figees."""
    assert ACTION_SPELL_START == len(DEPLOY_ROLES) * len(DEPLOY_SECTORS)
    assert ACTION_OBSERVE == ACTION_SPELL_START + NUM_SPELLS + len(HERO_NAMES)
    assert TOTAL_ACTIONS == ACTION_DONE + 1


def test_deploy_block_covers_every_role_sector_pair():
    pairs = {
        (i1, i2)
        for a in range(TOTAL_ACTIONS)
        for kind, i1, i2 in [decode_action(a)]
        if kind == 'deploy'
    }
    expected = {
        (r, s) for r in range(len(DEPLOY_ROLES)) for s in range(len(DEPLOY_SECTORS))
    }
    assert pairs == expected


def test_one_action_per_spell_and_per_hero():
    kinds = [decode_action(a)[0] for a in range(TOTAL_ACTIONS)]
    assert kinds.count('spell') == NUM_SPELLS
    assert kinds.count('ability') == len(HERO_NAMES)


def test_control_actions_are_the_last_four():
    tail = [decode_action(a)[0] for a in range(ACTION_OBSERVE, TOTAL_ACTIONS)]
    assert tail == ['observe', 'wait_short', 'wait_long', 'done']
