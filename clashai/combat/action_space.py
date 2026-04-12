# clashai/combat/action_space.py
# Action space V4 pour ClashAI.
#
# 37 actions au lieu de 289 (V3).
# L'agent choisit un RÔLE × SECTEUR pour le deploy,
# et des actions de haut niveau pour le combat.
# L'environnement traduit en taps ADB concrets.
#
# Deploy (25) : 5 rôles × 5 secteurs relatifs au côté d'attaque
# Sorts  (3)  : soin, rage, gel — ciblage automatique par SpellCaster
# Abilities (5): roi, reine, grand_gardien, championne, prince_gargouille
# Observe (1) : screenshot + mise à jour features
# Control (3) : wait_short, wait_long, done

import numpy as np


# =============================================================================
#                         RÔLES DE DEPLOY
# =============================================================================

DEPLOY_ROLES = ['tank', 'ranged', 'melee', 'hero', 'siege']
NUM_ROLES = len(DEPLOY_ROLES)  # 5

ROLE_TO_TROOPS = {
    'tank':   ['golem'],
    'ranged': ['sorcier', 'sorciere', 'archere'],
    'melee':  ['pekka'],
    'hero':   ['roi', 'reine', 'grand_gardien', 'championne', 'prince_gargouille'],
    'siege':  ['lance_buche'],
}


# =============================================================================
#                         SECTEURS DE DEPLOY
# =============================================================================

DEPLOY_SECTORS = ['far_left', 'left', 'center', 'right', 'far_right']
NUM_SECTORS = len(DEPLOY_SECTORS)  # 5

SECTOR_OFFSETS = {
    'far_left':  -4,
    'left':      -2,
    'center':     0,
    'right':     +2,
    'far_right': +4,
}


# =============================================================================
#                         ACTION SPACE
# =============================================================================

# Deploy: rôle × secteur
NUM_DEPLOY_ACTIONS = NUM_ROLES * NUM_SECTORS          # 25

# Sorts (auto-ciblés par SpellCaster)
SPELL_NAMES = ['soin', 'rage', 'gel']
ACTION_SPELL_START = NUM_DEPLOY_ACTIONS               # 25
ACTION_CAST_HEAL = ACTION_SPELL_START                  # 25
ACTION_CAST_RAGE = ACTION_SPELL_START + 1              # 26
ACTION_CAST_FREEZE = ACTION_SPELL_START + 2            # 27

# Abilities héros
HERO_NAMES = ['roi', 'reine', 'grand_gardien', 'championne', 'prince_gargouille']
NUM_HEROES = len(HERO_NAMES)
ACTION_ABILITY_START = ACTION_SPELL_START + 3          # 28
ACTION_ABILITY_ROI = ACTION_ABILITY_START              # 28
ACTION_ABILITY_REINE = ACTION_ABILITY_START + 1        # 29
ACTION_ABILITY_GG = ACTION_ABILITY_START + 2           # 30
ACTION_ABILITY_CHAMP = ACTION_ABILITY_START + 3        # 31
ACTION_ABILITY_PG = ACTION_ABILITY_START + 4           # 32

# Observe (screenshot + update features)
ACTION_OBSERVE = ACTION_ABILITY_START + NUM_HEROES     # 33

# Control
ACTION_WAIT_SHORT = ACTION_OBSERVE + 1                 # 34
ACTION_WAIT_LONG = ACTION_OBSERVE + 2                  # 35
ACTION_DONE = ACTION_OBSERVE + 3                       # 36

TOTAL_ACTIONS = ACTION_DONE + 1                        # 37

# Limits
MAX_STEPS_PER_EPISODE = 50
MAX_COMBAT_STEPS = 20
NUM_POSITIONS = 20  # positions sur le périmètre (hérité V3)


# =============================================================================
#                     ENCODE / DECODE
# =============================================================================

def decode_action(action_idx):
    """
    Décode un index d'action V4.

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
    """Encode une action en index."""
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
#                     ROLE INVENTORY
# =============================================================================

def build_role_inventory(remaining_troops, troop_types):
    """
    Construit l'inventaire par rôle à partir des troupes restantes.

    Args:
        remaining_troops: array (N,) — compteur par type de troupe
        troop_types: list[dict] — TROOP_TYPES du V3

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
            continue  # les sorts sont gérés séparément
        if role in role_counts:
            role_counts[role] += count
            role_queues[role].append((i, count))

    return role_counts, role_queues


def build_spell_inventory(remaining_troops, troop_types):
    """
    Construit l'inventaire des sorts restants.

    Returns:
        spell_counts: dict {'soin': n, 'rage': n, 'gel': n}
    """
    spell_counts = {s: 0 for s in SPELL_NAMES}
    for i, troop in enumerate(troop_types):
        if troop['role'] == 'spell' and troop['name'] in spell_counts:
            spell_counts[troop['name']] = int(remaining_troops[i])
    return spell_counts


# =============================================================================
#                     ACTION MASK
# =============================================================================

def compute_action_mask(remaining_troops, troop_types, phase='deploy',
                        hero_ability_mask=None):
    """
    Calcule le masque d'actions valides V4.

    Args:
        remaining_troops: array (N,) — compteur par type
        troop_types: list[dict] — TROOP_TYPES
        phase: 'deploy' ou 'combat'
        hero_ability_mask: array (5,) — 1.0 si ability dispo

    Returns:
        mask: array (37,) — 1.0 = action valide
    """
    mask = np.zeros(TOTAL_ACTIONS, dtype=np.float32)
    role_counts, _ = build_role_inventory(remaining_troops, troop_types)
    spell_counts = build_spell_inventory(remaining_troops, troop_types)

    if phase == 'deploy':
        # Deploy : chaque rôle avec des troupes restantes × tous les secteurs
        for role_idx, role_name in enumerate(DEPLOY_ROLES):
            if role_counts[role_name] > 0:
                start = role_idx * NUM_SECTORS
                mask[start:start + NUM_SECTORS] = 1.0

        # Control
        mask[ACTION_WAIT_SHORT] = 1.0
        mask[ACTION_WAIT_LONG] = 1.0
        mask[ACTION_DONE] = 1.0

    elif phase == 'combat':
        # Sorts restants
        for spell_idx, spell_name in enumerate(SPELL_NAMES):
            if spell_counts[spell_name] > 0:
                mask[ACTION_SPELL_START + spell_idx] = 1.0

        # Abilities héros
        if hero_ability_mask is not None:
            for i in range(NUM_HEROES):
                if hero_ability_mask[i] > 0:
                    mask[ACTION_ABILITY_START + i] = 1.0

        # Observe + control
        mask[ACTION_OBSERVE] = 1.0
        mask[ACTION_WAIT_SHORT] = 1.0
        mask[ACTION_WAIT_LONG] = 1.0

    return mask


def sector_to_position(sector_idx, center_pos, num_positions=NUM_POSITIONS):
    """
    Convertit un secteur relatif en position absolue sur le périmètre.

    Args:
        sector_idx: 0-4
        center_pos: position centrale (calculée depuis le côté d'attaque)
        num_positions: nombre total de positions

    Returns:
        position: int (0 to num_positions-1)
    """
    sector_name = DEPLOY_SECTORS[sector_idx]
    offset = SECTOR_OFFSETS[sector_name]
    return (center_pos + offset) % num_positions


# =============================================================================
#                         TEST
# =============================================================================

if __name__ == "__main__":
    print("🧪 Test Action Space V4\n")
    print(f"Total actions: {TOTAL_ACTIONS}\n")

    print("1. Decode toutes les actions:")
    for a in range(TOTAL_ACTIONS):
        t, i1, i2 = decode_action(a)
        if t == 'deploy':
            print(f"   [{a:2d}] deploy {DEPLOY_ROLES[i1]} @ {DEPLOY_SECTORS[i2]}")
        elif t == 'spell':
            print(f"   [{a:2d}] spell {i1}")
        elif t == 'ability':
            print(f"   [{a:2d}] ability {HERO_NAMES[i1]}")
        else:
            print(f"   [{a:2d}] {t}")

    print("\n2. Encode/decode roundtrip:")
    for a in range(TOTAL_ACTIONS):
        t, i1, i2 = decode_action(a)
        encoded = encode_action(t, i1, i2)
        assert encoded == a, f"Mismatch: {a} != {encoded}"
    print("   ✅ Roundtrip OK")

    print(f"\n✅ Action space V4 : {TOTAL_ACTIONS} actions")
