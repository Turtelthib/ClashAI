"""Reward shaping : les regles qui orientent l'apprentissage du RL.

Fonctions deterministes (etat) -> float, sans ADB ni GPU. Une erreur de signe ou
de seuil ici ne leve jamais d'exception : elle sabote l'entrainement en silence,
et ca ne se voit qu'apres des centaines d'episodes. D'ou ces tests.
"""

import numpy as np
import pytest

from clashai.combat import reward_shaping as R
from clashai.combat.action_space import DEPLOY_ROLES, HERO_NAMES

TANK = DEPLOY_ROLES.index('tank')
HERO = DEPLOY_ROLES.index('hero')
RANGED = DEPLOY_ROLES.index('ranged')


def _features(progress=0.0, troops_alive=0.5, hurt_ratio=0.0):
    """Vecteur combat (15,) du CombatObserver, aux seuls index lus par le module :
    [1] progression, [2] troupes vivantes, [10] ratio de blesses."""
    f = np.zeros(15, dtype=np.float32)
    f[1] = progress
    f[2] = troops_alive
    f[10] = hurt_ratio
    return f


# ---------------------------------------------------------------------------
# Signes des constantes — l'invariant le plus bete et le plus rentable
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    'REWARD_PER_STAR', 'REWARD_FIRST_STAR_BONUS', 'REWARD_THREE_STAR_BONUS',
    'REWARD_TANK_FIRST', 'REWARD_CONCENTRATION', 'REWARD_WAIT_AFTER_TANK',
    'REWARD_ABILITY_GOOD_TIMING', 'REWARD_GG_CLUTCH', 'REWARD_HERO_SURVIVAL',
])
def test_bonuses_are_positive(name):
    assert getattr(R, name) > 0, f"{name} recompense : doit etre > 0"


@pytest.mark.parametrize("name", [
    'REWARD_ZERO_STAR_PENALTY', 'REWARD_HERO_BEFORE_TANK',
    'REWARD_SPELL_TOO_EARLY', 'REWARD_SPREAD', 'REWARD_LEFTOVER_TROOPS',
    'REWARD_ABILITY_BAD_TIMING', 'REWARD_OVER_OBSERVE', 'REWARD_LEFTOVER_SPELLS',
])
def test_penalties_are_negative(name):
    assert getattr(R, name) < 0, f"{name} penalite : doit etre < 0"


# ---------------------------------------------------------------------------
# Phase de deploy
# ---------------------------------------------------------------------------

def test_tank_first_is_rewarded():
    reward = R.compute_deploy_reward('deploy', TANK, None, 0, 0, None, None)
    assert reward == R.REWARD_TANK_FIRST


def test_tank_bonus_stops_after_four_troops():
    """La regle "tanks d'abord" ne doit plus payer une fois le push lance."""
    assert R.compute_deploy_reward('deploy', TANK, None, 0, 4, None, None) == 0.0


def test_hero_before_any_tank_is_penalised():
    reward = R.compute_deploy_reward('deploy', HERO, None, 0, 0, None, None)
    assert reward == R.REWARD_HERO_BEFORE_TANK


def test_hero_after_a_tank_is_not_penalised():
    reward = R.compute_deploy_reward('deploy', HERO, None, 1, 1, None, None)
    assert reward == 0.0


@pytest.mark.parametrize("sector, last, expected_key", [
    (2, 2, 'REWARD_CONCENTRATION'),   # meme secteur
    (2, 3, 'REWARD_CONCENTRATION'),   # adjacent
    (0, 3, 'REWARD_SPREAD'),          # eparpille
    (0, 4, 'REWARD_SPREAD'),
])
def test_concentration_versus_spread(sector, last, expected_key):
    reward = R.compute_deploy_reward('deploy', RANGED, sector, 1, 1, last, None)
    assert reward == getattr(R, expected_key)


def test_distance_of_two_sectors_is_neutral():
    assert R.compute_deploy_reward('deploy', RANGED, 0, 1, 1, 2, None) == 0.0


def test_strategic_wait_after_tanks():
    assert R.compute_deploy_reward('wait_long', None, None, 1, 0, None, None) == \
        R.REWARD_WAIT_AFTER_TANK


def test_wait_without_any_tank_pays_nothing():
    assert R.compute_deploy_reward('wait_long', None, None, 0, 0, None, None) == 0.0


def test_done_carries_no_reward_of_its_own():
    """La penalite de troupes restantes vit dans compute_leftover_penalty()."""
    assert R.compute_deploy_reward('done', None, None, 1, 5, None, None) == 0.0


# ---------------------------------------------------------------------------
# Phase de combat — sorts
# ---------------------------------------------------------------------------

def test_rage_pays_only_with_troops_on_the_field():
    good = R.compute_combat_reward('spell', 'rage', None, _features(troops_alive=0.8),
                                   0, HERO_NAMES)
    bad = R.compute_combat_reward('spell', 'rage', None, _features(troops_alive=0.1),
                                  0, HERO_NAMES)
    assert good == R.REWARD_SPELL_RAGE_GOOD
    assert bad == R.REWARD_SPELL_RAGE_BAD
    assert good > bad


def test_heal_scales_with_how_hurt_the_troops_are():
    clutch = R.compute_combat_reward('spell', 'soin', None, _features(hurt_ratio=0.6),
                                     0, HERO_NAMES)
    useful = R.compute_combat_reward('spell', 'soin', None, _features(hurt_ratio=0.4),
                                     0, HERO_NAMES)
    wasted = R.compute_combat_reward('spell', 'soin', None, _features(hurt_ratio=0.0),
                                     0, HERO_NAMES)
    assert clutch == R.REWARD_SPELL_SOIN_GOOD + R.REWARD_COMBO_CLUTCH_HEAL
    assert useful == R.REWARD_SPELL_SOIN_GOOD
    assert wasted == R.REWARD_SPELL_SOIN_WASTED
    assert clutch > useful > wasted


def test_unknown_spell_falls_back_to_the_generic_reward():
    """Un sort ajoute au registre sans regle dediee ne doit pas valoir 0."""
    reward = R.compute_combat_reward('spell', 'totem', None, _features(), 0, HERO_NAMES)
    assert reward == R.REWARD_SPELL_IN_COMBAT


# ---------------------------------------------------------------------------
# Phase de combat — capacites de heros
# ---------------------------------------------------------------------------

def test_king_ability_timing():
    idx = HERO_NAMES.index('roi')
    good = R.compute_combat_reward('ability', None, idx, _features(progress=0.5),
                                   0, HERO_NAMES)
    too_early = R.compute_combat_reward('ability', None, idx, _features(progress=0.05),
                                        0, HERO_NAMES)
    assert good == R.REWARD_ABILITY_GOOD_TIMING
    assert too_early == R.REWARD_ABILITY_BAD_TIMING


def test_warden_rewards_saving_hurt_troops():
    idx = HERO_NAMES.index('grand_gardien')
    clutch = R.compute_combat_reward('ability', None, idx, _features(hurt_ratio=0.5),
                                     0, HERO_NAMES)
    wasted = R.compute_combat_reward('ability', None, idx, _features(hurt_ratio=0.05),
                                     0, HERO_NAMES)
    assert clutch == R.REWARD_GG_CLUTCH
    assert wasted == R.REWARD_ABILITY_BAD_TIMING


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

def test_observing_too_often_is_penalised():
    assert R.compute_combat_reward('observe', None, None, _features(), 11, HERO_NAMES) \
        == R.REWARD_OVER_OBSERVE


def test_observing_early_is_free():
    assert R.compute_combat_reward('observe', None, None, _features(), 3, HERO_NAMES) \
        == 0.0


# ---------------------------------------------------------------------------
# Robustesse : combat_features absent
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("action, spell, hero", [
    ('spell', 'rage', None),
    ('spell', 'soin', None),
    ('ability', None, 0),
    ('observe', None, None),
])
def test_none_features_never_raise(action, spell, hero):
    """La perception peut echouer : le reward doit degrader, pas planter."""
    assert isinstance(
        R.compute_combat_reward(action, spell, hero, None, 0, HERO_NAMES), float
    )
