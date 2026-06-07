# clashai/combat/heuristic_v3.py
# V3 heuristic action sequence (DEPLOY phase troops + COMBAT phase spells & abilities).
#
# Extracted from environment.py::ClashEnvV3.get_heuristic_sequence() in
# Phase C.2-light to keep env.py focused on the gym-style interface.
# V4 has its own (different) heuristic in environment_v4.py.
#
# The sequence is pure logic from the env's current inventory snapshot —
# no I/O, no side effects on `env` other than reading three attributes:
#   env._buildings, env._remaining_troops, env.verbose

from clashai.combat.legacy.agent import (
    TROOP_TYPES, TROOP_NAME_TO_IDX,
    NUM_POSITIONS,
    ACTION_WAIT_SHORT, ACTION_WAIT_LONG, ACTION_WAIT_COMBAT, ACTION_DONE,
    ACTION_ABILITY_ROI, ACTION_ABILITY_REINE,
    ACTION_ABILITY_GG, ACTION_ABILITY_CHAMP, ACTION_ABILITY_PG,
)
from clashai.combat.state_encoder import find_best_attack_side


def build_heuristic_sequence(env):
    """
    Heuristic sequence V3 — fully dynamic.

    Adapts automatically to actually available troops/spells.

    DEPLOY phase: troops only (tanks → funnel → ranged → melee → siege → heroes)
    COMBAT phase: spells (with fresh screenshot per cast) + hero abilities

    Spells are in the combat phase because:
    - SpellCaster takes a screenshot per spell → precise targeting
    - We can see troops fighting → we know where to place heal/rage/freeze
    - Freeze can target infernos near troops in real time
    """
    if env._buildings:
        best_dir = find_best_attack_side(env._buildings, verbose=env.verbose)
    else:
        best_dir = 0

    center_pos = int(best_dir / 8 * NUM_POSITIONS) % NUM_POSITIONS
    positions = [(center_pos + i - 2) % NUM_POSITIONS for i in range(5)]

    actions = []
    remaining = env._remaining_troops.copy()

    def add(name, pos):
        """Adds an action if the troop is available."""
        if name not in TROOP_NAME_TO_IDX:
            return False
        idx = TROOP_NAME_TO_IDX[name]
        if remaining[idx] > 0:
            actions.append(idx * NUM_POSITIONS + pos)
            remaining[idx] -= 1
            return True
        return False

    def add_all(name, pos_list):
        """Deploys all units of a type across the given positions."""
        if name not in TROOP_NAME_TO_IDX:
            return 0
        idx = TROOP_NAME_TO_IDX[name]
        count = 0
        i = 0
        while remaining[idx] > 0:
            p = pos_list[i % len(pos_list)]
            actions.append(idx * NUM_POSITIONS + p)
            remaining[idx] -= 1
            count += 1
            i += 1
        return count

    def add_one_spell(spell_name, pos):
        """Casts ONE spell if available."""
        return add(spell_name, pos)

    # ============================================================
    # Dynamic inventory
    # ============================================================
    tanks = [t['name'] for t in TROOP_TYPES
             if t['role'] == 'tank' and remaining[TROOP_NAME_TO_IDX[t['name']]] > 0]
    ranged = [t['name'] for t in TROOP_TYPES
              if t['role'] == 'ranged' and remaining[TROOP_NAME_TO_IDX[t['name']]] > 0]
    melee = [t['name'] for t in TROOP_TYPES
             if t['role'] == 'melee' and remaining[TROOP_NAME_TO_IDX[t['name']]] > 0]
    heroes = [t['name'] for t in TROOP_TYPES
              if t['role'] == 'hero' and remaining[TROOP_NAME_TO_IDX[t['name']]] > 0]
    sieges = [t['name'] for t in TROOP_TYPES
              if t['role'] == 'siege' and remaining[TROOP_NAME_TO_IDX[t['name']]] > 0]
    spells = {}
    for t in TROOP_TYPES:
        if t['role'] == 'spell':
            idx = TROOP_NAME_TO_IDX[t['name']]
            if remaining[idx] > 0:
                spells[t['name']] = int(remaining[idx])

    total_spells = sum(spells.values())
    sp = 10

    if env.verbose:
        print(f" Inventory: "
              f"{len(tanks)} tanks, {len(ranged)} ranged, "
              f"{len(melee)} melee, {len(heroes)} heroes, "
              f"{len(sieges)} siege, {total_spells} spells {spells}")

    # ============================================================
    # DEPLOY PHASE: troops only
    # ============================================================

    # 1. TANKS at the edges
    for tank_name in tanks:
        idx = TROOP_NAME_TO_IDX[tank_name]
        n = int(remaining[idx])
        if n >= 2:
            add(tank_name, positions[0])
            add(tank_name, positions[4])
            add_all(tank_name, [positions[2]])
        elif n == 1:
            add(tank_name, positions[2])

    actions.append(ACTION_WAIT_LONG)

    # 2. FUNNEL — ranged at the edges
    funnel_count = 0
    for r_name in ranged:
        idx = TROOP_NAME_TO_IDX[r_name]
        if remaining[idx] >= 2 and funnel_count < 2:
            add(r_name, positions[0])
            add(r_name, positions[4])
            funnel_count += 1

    actions.append(ACTION_WAIT_SHORT)

    # 3. RANGED in a line
    for r_name in ranged:
        add_all(r_name, positions[1:4])

    actions.append(ACTION_WAIT_LONG)

    # 4. MELEE + SIEGE at the center
    for m_name in melee:
        add_all(m_name, [positions[2], positions[1], positions[3]])
    for s_name in sieges:
        add_all(s_name, [positions[2]])

    # 5. HEROES at the center
    for h_name in heroes:
        add(h_name, positions[2])

    # → DONE: transition to combat
    actions.append(ACTION_DONE)

    # ============================================================
    # COMBAT PHASE: spells + abilities (fresh screenshot)
    #
    # Each wait_combat = screenshot + SpellCaster recalculates
    # positions in real time. Spells are targeted on what the
    # AI SEES, not on static coordinates.
    #
    # Tactical sequence:
    # 1. Observe (troops engage)
    # 2. RAGE (DPS boost during engagement)
    # 3. Observe (see damage)
    # 4. FREEZE on inferno/eagle (protect troops)
    # 5. Hero abilities (GG first = invincibility)
    # 6. HEAL (troops have taken damage)
    # 7. Observe + alternate RAGE/HEAL remaining
    # 8. Remaining hero abilities
    # ============================================================
    ABILITY_ORDER = [
        ('grand_gardien', ACTION_ABILITY_GG),
        ('roi', ACTION_ABILITY_ROI),
        ('reine', ACTION_ABILITY_REINE),
        ('championne', ACTION_ABILITY_CHAMP),
        ('prince_gargouille', ACTION_ABILITY_PG),
    ]

    # Split abilities into two waves
    wave1_abilities = []
    wave2_abilities = []
    for hero_name, ability_action in ABILITY_ORDER:
        if hero_name in heroes:
            if hero_name in ('grand_gardien', 'roi'):
                wave1_abilities.append(ability_action)
            else:
                wave2_abilities.append(ability_action)

    # Sort spells by tactical priority
    # Freeze = urgent (protect from infernos), Rage = boost, Heal = sustain
    spell_queue = []
    # First: 1 rage (initial boost)
    if spells.get('rage', 0) > 0:
        spell_queue.append('rage')
        spells['rage'] -= 1
    # Then: all freezes (stop infernos)
    while spells.get('gel', 0) > 0:
        spell_queue.append('gel')
        spells['gel'] -= 1
    # Then alternate heal/rage
    while any(v > 0 for v in spells.values()):
        for spell_name in ['soin', 'rage']:
            if spells.get(spell_name, 0) > 0:
                spell_queue.append(spell_name)
                spells[spell_name] -= 1
                break
        else:
            # Remaining spells (other types)
            for spell_name in list(spells.keys()):
                if spells[spell_name] > 0:
                    spell_queue.append(spell_name)
                    spells[spell_name] -= 1
                    break
            else:
                break

    # --- Build the combat sequence ---

    # 1. Observe (troops engage defenses)
    actions.append(ACTION_WAIT_COMBAT)

    # 2. First spell (rage boost) + defensive abilities
    spell_idx = 0
    if spell_idx < len(spell_queue):
        add_one_spell(spell_queue[spell_idx], sp)
        spell_idx += 1

    for ability in wave1_abilities:
        actions.append(ability)

    # 3. Observe damage
    actions.append(ACTION_WAIT_COMBAT)

    # 4. Freeze + heal (protect and heal)
    spells_this_round = 0
    while spell_idx < len(spell_queue) and spells_this_round < 2:
        add_one_spell(spell_queue[spell_idx], sp + spells_this_round)
        spell_idx += 1
        spells_this_round += 1

    # 5. Observe
    actions.append(ACTION_WAIT_COMBAT)

    # 6. Offensive abilities
    for ability in wave2_abilities:
        actions.append(ability)

    # 7. Remaining spells (with observe between each for targeting)
    while spell_idx < len(spell_queue):
        actions.append(ACTION_WAIT_COMBAT)
        add_one_spell(spell_queue[spell_idx], sp + (spell_idx % 4))
        spell_idx += 1

    actions.append(ACTION_DONE)

    return actions
