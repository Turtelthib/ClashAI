"""Le registre data-driven : configs/troops.json -> troupes, roles, sorts.

C'est le SSOT de toute la strategie. Ajouter une troupe doit rester "1 ligne de
JSON, zero code" -- ces tests verrouillent les regles non triviales qui le
permettent, et surtout la coherence entre le JSON et le fallback code en dur.
"""

import json

import pytest

from clashai.combat.action_space import DEPLOY_ROLES
from clashai.combat.troop_registry import (
    _FALLBACK,
    DEFAULT_MAX_BY_ROLE,
    SPELL_TARGET_DEFAULTS,
    build_role_to_troops,
    load_troop_types,
)
from clashai.paths import CONFIGS_DIR


def _raw_troops():
    with open(f"{CONFIGS_DIR}/troops.json", encoding='utf-8') as f:
        return json.load(f)['troops']


# ---------------------------------------------------------------------------
# Forme des entrees
# ---------------------------------------------------------------------------

def test_every_entry_has_name_role_and_default_max():
    for troop in load_troop_types():
        assert set(troop) == {'name', 'role', 'default_max'}
        assert isinstance(troop['name'], str) and troop['name']
        assert isinstance(troop['default_max'], int)
        assert troop['default_max'] > 0


def test_names_are_unique():
    names = [t['name'] for t in load_troop_types()]
    assert len(names) == len(set(names)), "un nom en double masquerait une troupe"


# ---------------------------------------------------------------------------
# Les sorts : le "max" du JSON est deliberement ignore
# ---------------------------------------------------------------------------

def test_spells_ignore_the_json_max():
    """Regle "cast-until-grayed" : les sorts sont seedes genereusement et c'est
    le grise qui coupe. Un `max` dans le JSON ne doit PAS les brider -- c'etait
    la cause du sous-cast (gel=1, rage=3) corrige en Session 14."""
    spells = [t for t in load_troop_types() if t['role'] == 'spell']
    assert spells, "le registre doit contenir des sorts"
    for spell in spells:
        assert spell['default_max'] == DEFAULT_MAX_BY_ROLE['spell']


def test_non_spells_use_their_json_max_or_the_role_default():
    by_name = {t['name']: t for t in load_troop_types()}
    for raw in _raw_troops():
        if raw['role'] == 'spell':
            continue
        expected = int(raw.get('max', DEFAULT_MAX_BY_ROLE.get(raw['role'], 4)))
        assert by_name[raw['name']]['default_max'] == expected


# ---------------------------------------------------------------------------
# build_role_to_troops
# ---------------------------------------------------------------------------

def test_spells_are_excluded_from_the_deploy_mapping():
    role_map = build_role_to_troops()
    assert 'spell' not in role_map


def test_every_deploy_role_is_populated():
    """Un role de DEPLOY_ROLES sans aucune troupe rendrait 5 actions inertes."""
    role_map = build_role_to_troops()
    for role in DEPLOY_ROLES:
        assert role_map.get(role), f"aucune troupe pour le role '{role}'"


# ---------------------------------------------------------------------------
# Coherence du fallback code en dur
# ---------------------------------------------------------------------------

def test_fallback_entries_have_the_same_shape_as_the_json():
    for troop in _FALLBACK:
        assert 'name' in troop and 'role' in troop


def test_fallback_only_uses_known_deploy_roles():
    """Le fallback sert quand troops.json est illisible. S'il contenait un role
    inconnu, le mode degrade serait pire que la panne."""
    for troop in _FALLBACK:
        assert troop['role'] in set(DEPLOY_ROLES) | {'spell'}, (
            f"role inconnu dans _FALLBACK : {troop['name']} -> {troop['role']}"
        )


# ---------------------------------------------------------------------------
# Ciblage des sorts
# ---------------------------------------------------------------------------

def test_spell_targets_are_valid_categories():
    for spell, target in SPELL_TARGET_DEFAULTS.items():
        assert target in ('cluster', 'heal', 'defense'), (
            f"cible inconnue pour '{spell}' : {target!r} -- SpellCaster ne saura "
            f"pas quoi viser"
        )


# ---------------------------------------------------------------------------
# Bug connu, encore ouvert
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    strict=True,
    reason="Bug ouvert (audit 2026-08-05, item 1.4) : configs/troops.json donne "
           "le role 'clean' a sorciere_ruine, or 'clean' n'existe pas dans "
           "DEPLOY_ROLES -> l'unite n'est jamais deployable. Corriger le role "
           "dans le JSON, puis retirer ce marqueur.",
)
def test_every_role_in_the_json_is_a_known_deploy_role():
    unknown = {
        t['name']: t['role']
        for t in load_troop_types()
        if t['role'] not in set(DEPLOY_ROLES) | {'spell'}
    }
    assert not unknown, f"roles inconnus : {unknown}"
