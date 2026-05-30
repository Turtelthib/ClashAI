# clashai/combat/episode_lifecycle.py
# Episode-end logic extracted from environment.py (Phase C.2-light).
#
# - check_battle_end(env)    : has the battle ended? (results screen or smart retreat)
# - wait_for_battle_end(env) : block until battle ends, possibly surrendering
# - finish_episode(env)      : aggregate stars / percentage / reward / info dict
# - compute_reward(stars, percentage) : pure reward computation
#
# All take `env` as the first argument so they can read / mutate the env
# state without being bound methods. env.py keeps thin shim methods that
# just delegate here.

import time

import cv2
import numpy as np

from clashai.config import (
    WAIT_BATTLE_MAX, WAIT_BATTLE_CHECK, WAIT_RESULT_SCREEN,
)
from clashai.perception.reward_reader import read_attack_results


# Reward weights — kept local to this module since the only callers are
# compute_reward + finish_episode. If a future PPO version needs to read
# them externally, move to config/rl.py.
REWARD_STAR_BONUS = 100
REWARD_ZERO_STAR_PENALTY = -50
REWARD_THREE_STAR_BONUS = 50
REWARD_FIRST_STAR_BONUS = 50

# Smart retreat detection (V3 era — kept for back-compat with the V3 env).
GREEN_DEAD_THRESHOLD = 2
NO_TROOPS_CHECKS_THRESHOLD = 3
NO_TROOPS_MIN_WAIT = 5.0


def compute_reward(stars: int, percentage: int) -> float:
    """Standard reward: stars × 100 + destruction percentage + bonuses."""
    reward = (stars * REWARD_STAR_BONUS) + percentage
    if stars >= 1:
        reward += REWARD_FIRST_STAR_BONUS
    if stars == 0:
        reward += REWARD_ZERO_STAR_PENALTY
    if stars == 3 and percentage == 100:
        reward += REWARD_THREE_STAR_BONUS
    return float(reward)


def check_battle_end(env) -> bool:
    """
    True if the battle is over.

    Two conditions:
      1. The screen shows the results screen.
      2. No living troops detected N times in a row (smart retreat) — the
         battle would end on its own in a few seconds anyway.
    """
    # Condition 1: results screen
    state, confidence, _ = env._get_screen_state()
    if state == 'resultats_attaque' and confidence > 0.6:
        return True

    # Condition 2: smart retreat (consecutive 0-troop checks)
    if env._no_troops_count >= NO_TROOPS_CHECKS_THRESHOLD:
        if env.verbose:
            print(f" Smart retreat: 0 troops for {env._no_troops_count} checks")
        return True

    return False


def wait_for_battle_end(env):
    """
    Block until the battle ends, optionally surrendering when troops die.

    Accelerated detection: if green troop bars stay ≤ GREEN_DEAD_THRESHOLD
    for NO_TROOPS_CHECKS_THRESHOLD consecutive checks → troops dead →
    surrender (white flag + confirmation) → results screen in ~5s.

    Note: green bars naturally decrease when troops take damage
    (green → orange). The threshold is therefore very low — an injured
    troop (orange bar) is still alive and fighting.

    Phase F.1 guard: if reset() couldn't reach phase_attaque (nav_failed
    flag set on env), the battle never happened. Returning None here lets
    finish_episode() short-circuit to a neutral 0.0 reward instead of
    surrendering on a UI it mistakes for a real fight.
    """
    if getattr(env, '_nav_failed', False):
        if env.verbose:
            print(" wait_for_battle_end: nav_failed=True → skipping battle wait")
        return None

    if env.verbose:
        print(" Waiting for battle end...")

    # Phase F.1: extra safety — verify we ARE in a battle before any
    # surrender / monitoring loop. If the env state lies and we're still
    # on village_home / recherche_adversaire / prep_attaque (i.e. the
    # match never launched), don't surrender on phantom UI bars.
    state, _, _ = env._get_screen_state()
    NON_BATTLE_STATES = {
        'village_home', 'recherche_adversaire', 'prep_attaque',
        'chargement', 'gdc_ally', 'gdc_enemy', 'gdc_ended',
        'menu_boutique', 'profil', 'chat_clan',
    }
    if state in NON_BATTLE_STATES:
        if env.verbose:
            print(f" wait_for_battle_end: screen='{state}' ≠ phase_attaque → "
                  f"no battle happened, returning None (no surrender)")
        return None

    surrendered = False
    if env._no_troops_count >= NO_TROOPS_CHECKS_THRESHOLD:
        env._surrender()
        surrendered = True
        min_wait = NO_TROOPS_MIN_WAIT
        if env.verbose:
            print(f" Active retreat → reduced wait ({min_wait:.0f}s min)")
    else:
        min_wait = 15.0 if env._phase == 'combat' else 30.0

    start_time = time.time()
    no_troops_consecutive = 0

    while time.time() - start_time < WAIT_BATTLE_MAX:
        elapsed = time.time() - start_time

        # 1. Check screen state
        state, confidence, img_pil = env._get_screen_state()

        if env.verbose and int(elapsed) % 10 == 0:
            print(f" {elapsed:.0f}s — screen: {state} ({confidence:.0%})")

        if state == 'resultats_attaque' and elapsed >= min_wait:
            if env.verbose:
                print(f" Battle ended after {elapsed:.0f}s")
            time.sleep(WAIT_RESULT_SCREEN)
            final_img = env._adb_screenshot()
            return final_img if final_img else img_pil

        # 2. Scan GREEN troop / hero bars (orange/red = injured or enemy)
        if img_pil is not None and not surrendered:
            try:
                img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                from clashai.combat.combat_observer import detect_troop_bars, detect_hero_bars
                green_pos, _ = detect_troop_bars(img_cv)
                hero_pos = detect_hero_bars(img_cv)
                green_count = len(green_pos) + len(hero_pos)

                if env.verbose:
                    print(f" Scan: {len(green_pos)} green, "
                          f"{len(hero_pos)} heroes → alive={green_count}")

                if green_count <= GREEN_DEAD_THRESHOLD:
                    no_troops_consecutive += 1
                    if env.verbose:
                        print(f" Below threshold "
                              f"({no_troops_consecutive}/{NO_TROOPS_CHECKS_THRESHOLD})")
                else:
                    no_troops_consecutive = 0

                if no_troops_consecutive >= NO_TROOPS_CHECKS_THRESHOLD:
                    if env.verbose:
                        print(f" Troops dead "
                              f"(green<={GREEN_DEAD_THRESHOLD} "
                              f"x{NO_TROOPS_CHECKS_THRESHOLD})")
                    env._surrender()
                    surrendered = True
                    min_wait = NO_TROOPS_MIN_WAIT

            except Exception as e:
                if env.verbose:
                    print(f" WARNING: Troop scan failed: {e}")

        time.sleep(WAIT_BATTLE_CHECK)

    state, _, img_pil = env._get_screen_state()
    if state == 'resultats_attaque':
        return img_pil
    return None


def finish_episode(env):
    """
    Block until the battle ends, read results, compute reward + info dict.
    Returns (reward, info).

    Phase F.1: if `env._nav_failed` is set (reset() couldn't reach the
    attack screen), short-circuit to a NEUTRAL outcome:
      - reward = 0.0  (NOT -50: the agent isn't responsible for nav bugs)
      - stars = 0, percentage = 0
      - info['nav_failed'] = True so training scripts can filter these
        episodes out of stats.
    """
    if env.verbose:
        from clashai.config.logging import pp
        remaining = int(np.sum(env._remaining_troops))
        pp(f" Episode over: {env._step_count} steps, "
           f"{env._combat_step_count} combat, "
           f"{remaining} remaining, "
           f"{env._hero_manager.num_activated()} abilities",
           tag='done')

    # Short-circuit for failed navigation — no battle happened.
    if getattr(env, '_nav_failed', False):
        if env.verbose:
            from clashai.config.logging import pp
            pp(" nav_failed=True → returning neutral reward 0.0 "
               "(no -50 penalty, no result read)", tag='warning')
        info = {
            'stars': 0, 'percentage': 0, 'reward': 0.0,
            'success': False,
            'steps': env._step_count,
            'deploy_steps': env._step_count - env._combat_step_count,
            'combat_steps': env._combat_step_count,
            'troops_remaining': int(np.sum(env._remaining_troops)),
            'abilities_used': env._hero_manager.num_activated(),
            'episode': env._episode_count,
            'early_retreat': False,
            'nav_failed': True,
        }
        return 0.0, info

    # If in combat phase, battle may already be over; otherwise wait passively
    result_img = wait_for_battle_end(env)

    if result_img is not None:
        results = read_attack_results(result_img, debug=False)
        stars = results['stars']
        percentage = results['percentage']
        success = results['success']
    else:
        if env.verbose:
            print(" WARNING: Unable to read results")
        stars = 0
        percentage = 0
        success = False

    reward = compute_reward(stars, percentage)

    if env.verbose:
        from clashai.config.logging import banner, pp
        retreat_str = "  (retreat)" if env._no_troops_count >= NO_TROOPS_CHECKS_THRESHOLD else ""
        banner(
            f"RESULTS{retreat_str}",
            f"⭐ {stars}/3   Destruction: {percentage}%   Reward: {reward:.0f}",
            tag='reward',
        )
        pp(" Returning to village...", tag='observe')
    env._return_to_village()

    info = {
        'stars': stars,
        'percentage': percentage,
        'reward': reward,
        'success': success,
        'steps': env._step_count,
        'deploy_steps': env._step_count - env._combat_step_count,
        'combat_steps': env._combat_step_count,
        'troops_remaining': int(np.sum(env._remaining_troops)),
        'abilities_used': env._hero_manager.num_activated(),
        'episode': env._episode_count,
        'early_retreat': env._no_troops_count >= NO_TROOPS_CHECKS_THRESHOLD,
        'nav_failed': False,
    }

    return reward, info
