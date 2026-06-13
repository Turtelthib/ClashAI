# clashai/combat/episode_runner.py
# Single attack-episode runner — the SSOT used by both the brain (farm + CW)
# and the V5.1 CombatAgent, so the deploy/combat loop lives in exactly one place.

def run_attack_episode(models, agent=None, use_heuristic=True, verbose=True):
    """
    Run one complete attack episode with ClashEnvV4.

    env.reset() handles the navigation to phase_attaque; this function drives
    the deploy/combat loop (scripted heuristic, or the RL policy if an agent
    is provided and use_heuristic is False).

    Args:
        models: loaded models dict (perception, troop_bar_detector, etc.)
        agent: PPOAgentV4 instance, or None for heuristic mode.
        use_heuristic: True → scripted sequence; False → agent.select_action.
        verbose: forwarded to the env.

    Returns:
        info: dict with episode results (stars, percentage, …), or None on failure.
    """
    from clashai.combat.environment_v4 import ClashEnvV4
    from clashai.combat.action_space import MAX_STEPS_SAFETY

    try:
        env = ClashEnvV4(models=models, verbose=verbose)
        obs, mask = env.reset()
        grid, vector = obs
        info = None

        if use_heuristic or agent is None:
            # Heuristic mode (scripted sequence)
            actions = env.get_heuristic_sequence()
            for action in actions:
                obs, mask, reward, done, info = env.step(action)
                grid, vector = obs
                if done:
                    break
        else:
            # RL mode
            for step in range(MAX_STEPS_SAFETY):
                action, _, _ = agent.select_action(grid, vector, mask)
                obs, mask, reward, done, info = env.step(action)
                grid, vector = obs
                if done:
                    break

        env.close()
        return info

    except Exception as e:
        print(f" ERROR: Erreur pendant l'attaque : {e}")
        import traceback
        traceback.print_exc()
        return None
