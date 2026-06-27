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
# checkpoint-safe: adding a TROOP only grows a role's count sum — the V4 obs
# stays role-based. Adding a SPELL is NOT checkpoint-safe (it grows
# SPELL_FEATURES + the action space → re-train), so spells are gated by the CNN:
# a spell pre-registered in troops.json but not yet a CNN class stays inert
# (no dead obs dim / needless re-train) until the CNN is retrained with it.

import json
import os

# Generous-but-bounded fallback max per role when an entry omits "max".
# It's an UPPER bound (the grayed signal caps the real count below it); kept
# bounded so the heuristic doesn't waste the step budget tapping empty slots.
DEFAULT_MAX_BY_ROLE = {
    'tank': 4, 'ranged': 12, 'melee': 8, 'hero': 1, 'siege': 1, 'spell': 8,
}

# Spells are cast-until-grayed: the per-attack count is unknown up front, so we
# always seed a generous upper bound (DEFAULT_MAX_BY_ROLE['spell']) and let the
# grayed signal cap it at the real count. The JSON "max" is IGNORED for spells
# (it under-counted, e.g. left 2 gel / 1 rage uncast).

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
        if role == 'spell':
            default_max = DEFAULT_MAX_BY_ROLE['spell']   # JSON "max" ignored (cast-until-grayed)
        else:
            default_max = int(t.get('max', DEFAULT_MAX_BY_ROLE.get(role, 4)))
        out.append({'name': t['name'], 'role': role, 'default_max': default_max})
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


# =============================================================================
# SPELLS (data-driven, gated by the CNN)
# =============================================================================

# Spell targeting category → which SpellCaster output to aim at.
#   'cluster' = your troop mass · 'heal' = your injured troops · 'defense' = the
#   enemy's most dangerous defense. Overridable per-spell via "target" in the
#   JSON. Best-guess for the spells you don't tune — edit as data, not code.
SPELL_TARGET_DEFAULTS = {
    'soin': 'heal', 'rage': 'cluster', 'gel': 'defense',
    'zap': 'defense', 'saut': 'defense', 'clone': 'cluster',
    'rappel': 'cluster', 'resurrection': 'heal', 'totem': 'cluster',
    'poison': 'defense', 'seisme': 'defense', 'speed': 'cluster',
    'squelette': 'defense', 'chauve_souris': 'defense', 'racine': 'cluster',
    'bloc_glace': 'defense', 'colere': 'cluster',
}

_CNN_CLASSES = None


def cnn_class_names():
    """Set of class names the troop-bar CNN can detect. Gates which spells are
    active in the action space (perception → strategy): a spell pre-registered in
    troops.json but absent from the CNN stays inert. Empty set if the model
    artifacts are unreadable (callers then skip the filter)."""
    global _CNN_CLASSES
    if _CNN_CLASSES is not None:
        return _CNN_CLASSES
    try:
        from clashai.paths import WEIGHTS_DIR
        p = os.path.join(WEIGHTS_DIR, 'yolo_troupes_barre', 'model_artifacts.json')
        with open(p, encoding='utf-8') as f:
            _CNN_CLASSES = set(json.load(f).get('names', []))
    except Exception:
        _CNN_CLASSES = set()
    return _CNN_CLASSES


def load_spell_names():
    """Ordered spell names = registry(role=spell) ∩ CNN classes (registry order).
    Falls back to all registry spells if the CNN class list is unavailable."""
    spells = [t['name'] for t in _load_raw() if t['role'] == 'spell']
    cnn = cnn_class_names()
    return [s for s in spells if s in cnn] if cnn else spells


def load_spell_targets():
    """{spell_name: target_category} for every registry spell. Uses the JSON
    'target' field if present, else SPELL_TARGET_DEFAULTS, else 'cluster'."""
    out = {}
    for t in _load_raw():
        if t['role'] != 'spell':
            continue
        out[t['name']] = t.get('target', SPELL_TARGET_DEFAULTS.get(t['name'], 'cluster'))
    return out
