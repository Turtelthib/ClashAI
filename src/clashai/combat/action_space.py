# clashai/combat/action_space.py
# Action space V4 for ClashAI.
#
# Compact action space (vs 289 in V3). The agent chooses a ROLE × SECTOR for
# deploy and high-level actions for combat. The environment translates these
# into concrete ADB taps. TOTAL_ACTIONS is DERIVED (not hardcoded) so the spell
# count can grow with the registry/CNN.
#
# Deploy (25) : 5 roles × 5 sectors relative to the attack side
# Spells (N)  : DATA-DRIVEN from configs/troops.json ∩ CNN classes, auto-targeted
# Abilities (5): roi, reine, grand_gardien, championne, prince_gargouille
# Observe (1) : screenshot + update features
# Control (3) : wait_short, wait_long, done

import numpy as np


# =============================================================================
# RÔLES DE DEPLOY
# =============================================================================

DEPLOY_ROLES = ['tank', 'ranged', 'melee', 'hero', 'siege']
NUM_ROLES = len(DEPLOY_ROLES)

# Mapping role → ordered list of troop names (deploy priority).
# DATA-DRIVEN: derived from configs/troops.json (clashai.combat.troop_registry),
# so a new troop in that JSON is automatically deployable — zero code here.
from clashai.combat.troop_registry import build_role_to_troops as _build_role_to_troops
ROLE_TO_TROOPS = _build_role_to_troops()
# Ensure every deploy role exists as a key, even if it has no troops yet.
for _r in DEPLOY_ROLES:
    ROLE_TO_TROOPS.setdefault(_r, [])


# =============================================================================
# SECTEURS DE DEPLOY
# =============================================================================

DEPLOY_SECTORS = ['far_left', 'left', 'center', 'right', 'far_right']
NUM_SECTORS = len(DEPLOY_SECTORS)

# Mapping sector → relative offset from the attack center
# (in number of positions out of the 20 on the perimeter)
SECTOR_OFFSETS = {
    'far_left': -4,
    'left': -2,
    'center': 0,
    'right': +2,
    'far_right': +4,
}


# =============================================================================
# ACTION SPACE
# =============================================================================

# Deploy: role × sector
NUM_DEPLOY_ACTIONS = NUM_ROLES * NUM_SECTORS

# Spells (auto-targeted by SpellCaster) — DATA-DRIVEN: registry ∩ CNN classes.
# One action per spell type; per-attack availability is handled by the mask, so
# an attack can carry 2 spells or 10 with no code change. No hardcoded count:
# the index layout below is computed from len(SPELL_NAMES). Adding a spell to
# troops.json (once the CNN detects it) grows this automatically → re-train.
from clashai.combat.troop_registry import load_spell_names as _load_spell_names
SPELL_NAMES = _load_spell_names()
NUM_SPELLS = len(SPELL_NAMES)
ACTION_SPELL_START = NUM_DEPLOY_ACTIONS

# Hero abilities
HERO_NAMES = ['roi', 'reine', 'grand_gardien', 'championne', 'prince_gargouille']
NUM_HEROES = len(HERO_NAMES)
ACTION_ABILITY_START = ACTION_SPELL_START + NUM_SPELLS
ACTION_ABILITY_ROI = ACTION_ABILITY_START
ACTION_ABILITY_REINE = ACTION_ABILITY_START + 1
ACTION_ABILITY_GG = ACTION_ABILITY_START + 2
ACTION_ABILITY_CHAMP = ACTION_ABILITY_START + 3
ACTION_ABILITY_PG = ACTION_ABILITY_START + 4

# Observe (screenshot + update features)
ACTION_OBSERVE = ACTION_ABILITY_START + NUM_HEROES

# Control
ACTION_WAIT_SHORT = ACTION_OBSERVE + 1
ACTION_WAIT_LONG = ACTION_OBSERVE + 2
ACTION_DONE = ACTION_OBSERVE + 3

TOTAL_ACTIONS = ACTION_DONE + 1

# Limits
MAX_STEPS_SAFETY = 200
                         # The true episode end is natural (_all_resources_exhausted)
NUM_POSITIONS = 20


# =============================================================================
# ENCODE / DECODE
# =============================================================================

def decode_action(action_idx):
    """
    Decodes a V4 action index.

    Returns:
        ('deploy', role_idx, sector_idx)
        ('spell', spell_name, None)
        ('ability', hero_idx, None)
        ('observe', None, None)
        ('wait_short', None, None)
        ('wait_long', None, None)
        ('done', None, None)
    """
    if action_idx < NUM_DEPLOY_ACTIONS:
        role_idx = action_idx // NUM_SECTORS
        sector_idx = action_idx % NUM_SECTORS
        return ('deploy', role_idx, sector_idx)
    elif ACTION_SPELL_START <= action_idx < ACTION_SPELL_START + len(SPELL_NAMES):
        spell_idx = action_idx - ACTION_SPELL_START
        return ('spell', SPELL_NAMES[spell_idx], None)
    elif ACTION_ABILITY_START <= action_idx < ACTION_ABILITY_START + NUM_HEROES:
        hero_idx = action_idx - ACTION_ABILITY_START
        return ('ability', hero_idx, None)
    elif action_idx == ACTION_OBSERVE:
        return ('observe', None, None)
    elif action_idx == ACTION_WAIT_SHORT:
        return ('wait_short', None, None)
    elif action_idx == ACTION_WAIT_LONG:
        return ('wait_long', None, None)
    elif action_idx == ACTION_DONE:
        return ('done', None, None)
    return ('done', None, None)


def encode_action(action_type, idx1=None, idx2=None):
    """Encodes an action as an index."""
    if action_type == 'deploy':
        return idx1 * NUM_SECTORS + idx2
    elif action_type == 'spell':
        return ACTION_SPELL_START + SPELL_NAMES.index(idx1)
    elif action_type == 'ability':
        return ACTION_ABILITY_START + idx1
    elif action_type == 'observe':
        return ACTION_OBSERVE
    elif action_type == 'wait_short':
        return ACTION_WAIT_SHORT
    elif action_type == 'wait_long':
        return ACTION_WAIT_LONG
    elif action_type == 'done':
        return ACTION_DONE
    return ACTION_DONE


# =============================================================================
# ROLE INVENTORY
# =============================================================================

def build_role_inventory(remaining_troops, troop_types):
    """
    Builds the per-role inventory from remaining troops.

    Args:
        remaining_troops: array (N,) — counter per troop type
        troop_types: list[dict] — TROOP_TYPES from V3

    Returns:
        role_counts: dict {role_name: total_count}
        role_queues: dict {role_name: [(troop_idx, count), ...]}
    """
    role_counts = {r: 0 for r in DEPLOY_ROLES}
    role_queues = {r: [] for r in DEPLOY_ROLES}

    for i, troop in enumerate(troop_types):
        count = int(remaining_troops[i])
        if count <= 0:
            continue
        role = troop['role']
        if role == 'spell':
            continue
        if role in role_counts:
            role_counts[role] += count
            role_queues[role].append((i, count))

    return role_counts, role_queues


def build_spell_inventory(remaining_troops, troop_types):
    """
    Builds the inventory of remaining spells.

    Returns:
        spell_counts: dict {'soin': n, 'rage': n, 'gel': n}
    """
    spell_counts = {s: 0 for s in SPELL_NAMES}
    for i, troop in enumerate(troop_types):
        if troop['role'] == 'spell' and troop['name'] in spell_counts:
            spell_counts[troop['name']] = int(remaining_troops[i])
    return spell_counts


# =============================================================================
# ACTION MASK
# =============================================================================

def compute_action_mask(remaining_troops, troop_types, hero_ability_mask=None):
    """
    Computes the valid action mask for V4.2.

    V4.2 : merged phases — no more binary deploy/combat phase.
    Masking is based solely on available resources.

    Args:
        remaining_troops: array (N,) — counter per type
        troop_types: list[dict] — TROOP_TYPES
        hero_ability_mask: array (5,) — 1.0 if ability available

    Returns:
        mask: array (37,) — 1.0 = valid action
    """
    mask = np.zeros(TOTAL_ACTIONS, dtype=np.float32)
    role_counts, _ = build_role_inventory(remaining_troops, troop_types)
    spell_counts = build_spell_inventory(remaining_troops, troop_types)

    # Deploy actions (0-24) : available if the role has remaining troops
    for role_idx, role_name in enumerate(DEPLOY_ROLES):
        if role_counts[role_name] > 0:
            start = role_idx * NUM_SECTORS
            mask[start:start + NUM_SECTORS] = 1.0

    # Spell actions (25-27) : available if the spell is still available
    for spell_idx, spell_name in enumerate(SPELL_NAMES):
        if spell_counts[spell_name] > 0:
            mask[ACTION_SPELL_START + spell_idx] = 1.0

    # Abilities (28-32) : available if the hero ability is ready
    if hero_ability_mask is not None:
        for i in range(NUM_HEROES):
            if hero_ability_mask[i] > 0:
                mask[ACTION_ABILITY_START + i] = 1.0

    # observe (33), wait_short (34), wait_long (35), done (36) : always available
    mask[ACTION_OBSERVE] = 1.0
    mask[ACTION_WAIT_SHORT] = 1.0
    mask[ACTION_WAIT_LONG] = 1.0
    mask[ACTION_DONE] = 1.0

    return mask


def sector_to_position(sector_idx, center_pos, num_positions=NUM_POSITIONS):
    """
    Converts a relative sector to an absolute position on the perimeter.

    Args:
        sector_idx: 0-4
        center_pos: center position (computed from the attack side)
        num_positions: total number of positions

    Returns:
        position: int (0 to num_positions-1)
    """
    sector_name = DEPLOY_SECTORS[sector_idx]
    offset = SECTOR_OFFSETS[sector_name]
    return (center_pos + offset) % num_positions


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Test Action Space V4\n")
    print(f"Total actions: {TOTAL_ACTIONS}\n")

    print("1. Decode all actions:")
    for a in range(TOTAL_ACTIONS):
        t, i1, i2 = decode_action(a)
        if t == 'deploy':
            print(f" [{a:2d}] deploy {DEPLOY_ROLES[i1]} @ {DEPLOY_SECTORS[i2]}")
        elif t == 'spell':
            print(f" [{a:2d}] spell {i1}")
        elif t == 'ability':
            print(f" [{a:2d}] ability {HERO_NAMES[i1]}")
        else:
            print(f" [{a:2d}] {t}")

    print("\n2. Encode/decode round-trip:")
    for a in range(TOTAL_ACTIONS):
        t, i1, i2 = decode_action(a)
        encoded = encode_action(t, i1, i2)
        assert encoded == a, f"Mismatch: {a} != {encoded}"
    print(" Round-trip OK")

    print(f"\nAction space V4 : {TOTAL_ACTIONS} actions")
