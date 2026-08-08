"""La sequence heuristique — le prof du behavior cloning.

L'agent RL demarre en clonant cette sequence. Si l'ordre de deploiement y est
faux, le BC enseigne le mauvais comportement et le RL doit ensuite le desapprendre.
Ces tests verrouillent l'ordre, sans emulateur : le mixin ne lit que
`self._remaining_troops` et `self.verbose`.
"""

import numpy as np

from clashai.combat.action_space import DEPLOY_ROLES, decode_action
from clashai.combat.environment_v4.heuristic import HeuristicMixin
from clashai.combat.troop_registry import load_troop_types


class _FakeEnv(HeuristicMixin):
    """Env minimal : le mixin n'a besoin que de l'inventaire et de verbose."""

    verbose = False

    def __init__(self, remaining=None):
        troop_types = load_troop_types()
        self._remaining_troops = (
            remaining if remaining is not None
            else np.ones(len(troop_types), dtype=np.float32)
        )


def _deployed_roles(sequence):
    out = []
    for action in sequence:
        kind, role_idx, _ = decode_action(action)
        if kind == 'deploy':
            out.append(DEPLOY_ROLES[role_idx])
    return out


def test_sequence_is_not_empty_with_a_full_inventory():
    assert _FakeEnv().get_heuristic_sequence()


def test_tanks_open_the_attack():
    roles = _deployed_roles(_FakeEnv().get_heuristic_sequence())
    assert roles[0] == 'tank'


def test_heroes_come_after_tanks():
    roles = _deployed_roles(_FakeEnv().get_heuristic_sequence())
    assert roles.index('tank') < roles.index('hero')


def test_siege_goes_in_before_heroes():
    """Regle V4.1 : l'engin de siege ouvre la voie avant les heros."""
    roles = _deployed_roles(_FakeEnv().get_heuristic_sequence())
    assert roles.index('siege') < roles.index('hero')


def test_clean_is_the_very_last_deploy():
    """La sorciere des ruines n'invoque que sur batiment detruit : la poser avant
    que le push ait entame la base la gaspille."""
    roles = _deployed_roles(_FakeEnv().get_heuristic_sequence())
    assert roles[-1] == 'clean'


def test_empty_inventory_yields_no_deploy():
    troop_types = load_troop_types()
    empty = np.zeros(len(troop_types), dtype=np.float32)

    roles = _deployed_roles(_FakeEnv(remaining=empty).get_heuristic_sequence())

    assert roles == []


def test_every_emitted_action_is_within_the_action_space():
    from clashai.combat.action_space import TOTAL_ACTIONS

    for action in _FakeEnv().get_heuristic_sequence():
        assert 0 <= action < TOTAL_ACTIONS
