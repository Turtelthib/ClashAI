# clashai/combat/troop_registry.py
# Data-driven troop registry (SSOT = configs/troops.json).
#
# Replaces the old hardcoded TROOP_TYPES (legacy/agent.py) + ROLE_TO_TROOPS
# (action_space.py). Adding a new troop/hero/siege = retrain the troop-bar CNN
# (perception) + ONE line in configs/troops.json (strategy) — zero code.
#
# The CNN gives the NAME; the role is the only strategic bit that must be
# declared. Eventually the LocalLLMBrain can fill unknown roles automatically.
#
# checkpoint-safe: adding a troop only grows a role's count sum — the V4 obs
# stays role-based (54 dims). (Adding a SPELL is NOT: it changes SPELL_FEATURES.)

import json
import os

# Generous-but-bounded fallback max per role when an entry omits "max".
# It's an UPPER bound (the grayed signal caps the real count below it); kept
# bounded so the heuristic doesn't waste the step budget tapping empty slots.
DEFAULT_MAX_BY_ROLE = {
    'tank': 4, 'ranged': 12, 'melee': 8, 'hero': 1, 'siege': 1, 'spell': 2,
}

# Used only if configs/troops.json is missing/broken — the historical 14.
_FALLBACK = [
    {'name': 'golem', 'role': 'tank', 'max': 2},
    {'name': 'sorcier', 'role': 'ranged', 'max': 6},
    {'name': 'sorciere', 'role': 'ranged', 'max': 10},
    {'name': 'pekka', 'role': 'melee', 'max': 2},
    {'name': 'archere', 'role': 'ranged', 'max': 5},
    {'name': 'roi', 'role': 'hero', 'max': 1},
    {'name': 'reine', 'role': 'hero', 'max': 1},
    {'name': 'grand_gardien', 'role': 'hero', 'max': 1},
    {'name': 'championne', 'role': 'hero', 'max': 1},
    {'name': 'prince_gargouille', 'role': 'hero', 'max': 1},
    {'name': 'lance_buche', 'role': 'siege', 'max': 1},
    {'name': 'soin', 'role': 'spell', 'max': 2},
    {'name': 'rage', 'role': 'spell', 'max': 3},
    {'name': 'gel', 'role': 'spell', 'max': 1},
]


def _registry_path():
    try:
        from clashai.paths import CONFIGS_DIR
        return os.path.join(CONFIGS_DIR, 'troops.json')
    except Exception:
        from clashai.paths import PROJECT_ROOT
        return os.path.join(PROJECT_ROOT, 'configs', 'troops.json')


def _load_raw():
    path = _registry_path()
    try:
        with open(path, encoding='utf-8') as f:
            troops = json.load(f).get('troops')
        if not troops:
            raise ValueError('empty "troops"')
        return troops
    except Exception as e:
        print(f"WARNING: troops.json unreadable ({e}) -> fallback hardcoded registry")
        return _FALLBACK


def load_troop_types():
    """TROOP_TYPES = [{'name', 'role', 'default_max'}] derived from the JSON.

    'max' in the JSON is optional → role-based default if omitted.
    """
    out = []
    for t in _load_raw():
        role = t['role']
        out.append({
            'name': t['name'],
            'role': role,
            'default_max': int(t.get('max', DEFAULT_MAX_BY_ROLE.get(role, 4))),
        })
    return out


def build_role_to_troops(troop_types=None):
    """ROLE_TO_TROOPS = {role: [names...]} (deploy priority order), spells excluded."""
    tt = troop_types if troop_types is not None else load_troop_types()
    out = {}
    for t in tt:
        if t['role'] == 'spell':
            continue
        out.setdefault(t['role'], []).append(t['name'])
    return out
