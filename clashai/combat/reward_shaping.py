# clashai/combat/reward_shaping.py
# Reward shaping for ClashAI V4.
#
# All reward logic is centralized here.
# The environment calls compute_step_reward() and compute_final_reward().


from clashai.combat.action_space import DEPLOY_ROLES


# =============================================================================
# CONSTANTS
# =============================================================================

# Final rewards (based on attack results)
REWARD_PER_STAR = 100
REWARD_FIRST_STAR_BONUS = 50
REWARD_ZERO_STAR_PENALTY = -50
REWARD_THREE_STAR_BONUS = 50

# Deploy reward shaping
REWARD_TANK_FIRST = 5.0
REWARD_HERO_BEFORE_TANK = -3.0
REWARD_SPELL_TOO_EARLY = -8.0
REWARD_CONCENTRATION = 1.0
REWARD_SPREAD = -1.0
REWARD_WAIT_AFTER_TANK = 3.0
REWARD_LEFTOVER_TROOPS = -2.0

# Combat reward shaping
REWARD_ABILITY_GOOD_TIMING = 3.0
REWARD_ABILITY_BAD_TIMING = -2.0
REWARD_GG_CLUTCH = 5.0
REWARD_SPELL_IN_COMBAT = 1.0
REWARD_OVER_OBSERVE = -2.0
REWARD_LEFTOVER_SPELLS = -5.0

# Advanced reward shaping (V4.2)
REWARD_SPELL_RAGE_GOOD = 2.0
REWARD_SPELL_RAGE_BAD = 0.0
REWARD_SPELL_SOIN_GOOD = 3.0
REWARD_SPELL_SOIN_WASTED = 0.5
REWARD_SPELL_GEL = 1.5
REWARD_COMBO_CLUTCH_HEAL = 2.0
REWARD_HERO_SURVIVAL = 5.0


# =============================================================================
# STEP REWARD (during episode)
# =============================================================================

def compute_deploy_reward(action_type, role_idx, sector_idx,
                          tanks_deployed, troops_deployed,
                          last_sector, combat_features):
    """
    Reward shaping for the deploy phase.

    Args:
        action_type: 'deploy', 'wait_short', 'wait_long', 'done'
        role_idx: role index (0-4) if deploy
        sector_idx: sector index (0-4) if deploy
        tanks_deployed: int
        troops_deployed: int
        last_sector: int or None
        combat_features: array or None

    Returns:
        reward: float
    """
    reward = 0.0

    if action_type == 'deploy' and role_idx is not None:
        role_name = DEPLOY_ROLES[role_idx]

        # Rule 1: tanks first
        if role_name == 'tank' and troops_deployed < 4:
            reward += REWARD_TANK_FIRST

        # Rule 2: heroes not before tanks
        if role_name == 'hero' and tanks_deployed == 0:
            reward += REWARD_HERO_BEFORE_TANK

        # Rule 3: troop concentration
        if sector_idx is not None and last_sector is not None:
            dist = abs(sector_idx - last_sector)
            if dist <= 1:
                reward += REWARD_CONCENTRATION
            elif dist >= 3:
                reward += REWARD_SPREAD

    elif action_type == 'wait_long':
        # Rule 4: strategic wait after tanks
        if tanks_deployed > 0 and troops_deployed < 6:
            reward += REWARD_WAIT_AFTER_TANK

    elif action_type == 'done':
        # No penalty here — leftover troop penalty
        # is computed in compute_leftover_penalty()
        pass

    return reward


def compute_combat_reward(action_type, spell_name, hero_idx,
                          combat_features, combat_step_count,
                          hero_names):
    """
    Reward shaping for the combat phase.

    Args:
        action_type: 'spell', 'ability', 'observe', 'wait_short', etc.
        spell_name: str or None
        hero_idx: int or None
        combat_features: array (15,) from CombatObserver
        combat_step_count: int
        hero_names: list[str]

    Returns:
        reward: float
    """
    reward = 0.0

    if action_type == 'ability' and hero_idx is not None:
        hero_name = hero_names[hero_idx]
        progress = combat_features[1] if combat_features is not None else 0.0
        hurt_ratio = combat_features[10] if combat_features is not None else 0.0

        if hero_name == 'roi':
            if 0.3 <= progress <= 0.8:
                reward += REWARD_ABILITY_GOOD_TIMING
            elif progress < 0.1:
                reward += REWARD_ABILITY_BAD_TIMING

        elif hero_name == 'reine':
            if 0.2 <= progress <= 0.7:
                reward += REWARD_ABILITY_GOOD_TIMING

        elif hero_name == 'grand_gardien':
            if hurt_ratio > 0.3:
                reward += REWARD_GG_CLUTCH
            elif hurt_ratio < 0.1:
                reward += REWARD_ABILITY_BAD_TIMING

        elif hero_name in ('championne', 'prince_gargouille'):
            if progress > 0.2:
                reward += 1.0

    elif action_type == 'spell' and spell_name is not None:
        troops_alive = combat_features[2] if combat_features is not None else 0.5
        hurt_ratio = combat_features[10] if combat_features is not None else 0.0

        if spell_name == 'rage':
            reward += REWARD_SPELL_RAGE_GOOD if troops_alive > 0.3 else REWARD_SPELL_RAGE_BAD

        elif spell_name == 'soin':
            if hurt_ratio > 0.5:
                # Clutch combo: emergency heal
                reward += REWARD_SPELL_SOIN_GOOD + REWARD_COMBO_CLUTCH_HEAL
            elif hurt_ratio > 0.3:
                reward += REWARD_SPELL_SOIN_GOOD
            else:
                reward += REWARD_SPELL_SOIN_WASTED

        elif spell_name == 'gel':
            reward += REWARD_SPELL_GEL

        else:
            reward += REWARD_SPELL_IN_COMBAT

    elif action_type == 'observe':
        if combat_step_count > 10:
            reward += REWARD_OVER_OBSERVE

    return reward


def compute_hero_survival_bonus(combat_features):
    """
    End-of-episode bonus based on heroes still alive.

    Args:
        combat_features: array (15,) from the last observe, or None

    Returns:
        reward: float (0.0 if no observation available)
    """
    if combat_features is None:
        return 0.0
    heroes_alive_ratio = combat_features[4]
    num_alive = round(heroes_alive_ratio * 5)
    return REWARD_HERO_SURVIVAL * num_alive


def compute_leftover_penalty(remaining_troops, troop_types):
    """Penalty for troops not deployed at the end of deploy phase."""
    count = sum(
        int(remaining_troops[i])
        for i, t in enumerate(troop_types)
        if t['role'] != 'spell'
    )
    return REWARD_LEFTOVER_TROOPS * count if count > 0 else 0.0


def compute_spell_leftover_penalty(remaining_troops, troop_types):
    """Penalty for spells unused at the end of combat."""
    count = sum(
        int(remaining_troops[i])
        for i, t in enumerate(troop_types)
        if t['role'] == 'spell'
    )
    return REWARD_LEFTOVER_SPELLS * count if count > 0 else 0.0


# =============================================================================
# FINAL REWARD (end of episode)
# =============================================================================

def compute_final_reward(stars, percentage):
    """
    Final reward based on the attack result.

    Args:
        stars: int (0-3)
        percentage: int (0-100)

    Returns:
        reward: float
    """
    reward = (stars * REWARD_PER_STAR) + percentage

    if stars >= 1:
        reward += REWARD_FIRST_STAR_BONUS
    else:
        reward += REWARD_ZERO_STAR_PENALTY

    if stars == 3:
        reward += REWARD_THREE_STAR_BONUS

    return reward
