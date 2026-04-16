# tools/train_rl_v4.py
# Entraînement RL V4 pour ClashAI.
#
# Usage :
#   uv run python tools/train_rl_v4.py --heuristic --episodes 5
#   uv run python tools/train_rl_v4.py --episodes 100
#   uv run python tools/train_rl_v4.py --resume --episodes 50
#   uv run python tools/train_rl_v4.py --pretrain 15 --episodes 500
#   uv run python tools/train_rl_v4.py --pretrain 15 --bc-epochs 15 --episodes 500

import os
import json
import time
import argparse
from datetime import datetime

from clashai.paths import RL_WEIGHTS_DIR

from clashai.combat.environment_v4 import ClashEnvV4
from clashai.combat.agent_v4 import (
    PPOAgentV4, BATCH_SIZE, VECTOR_SIZE,
)
from clashai.combat.action_space import (
    MAX_STEPS_PER_EPISODE, TOTAL_ACTIONS,
)


def main():
    parser = argparse.ArgumentParser(description="ClashAI V4 RL Training")
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--heuristic', action='store_true',
                        help="Utiliser l'heuristique au lieu du PPO")
    parser.add_argument('--pretrain', type=int, default=0,
                        help="Pré-entraîner avec N épisodes heuristiques (behavioral cloning)")
    parser.add_argument('--bc-epochs', type=int, default=10,
                        help="Nombre d'epochs pour le behavioral cloning")
    parser.add_argument('--resume', action='store_true',
                        help="Reprendre depuis le dernier checkpoint")
    parser.add_argument('--verbose', action='store_true', default=True)
    args = parser.parse_args()

    mode = 'heuristique' if args.heuristic else 'PPO'
    if args.pretrain > 0:
        mode = f'pretrain BC ({args.pretrain} demos) + PPO'

    print(f"\n{'='*60}")
    print("  🎮 ClashAI V4 — Entraînement RL")
    print(f"  Mode       : {mode}")
    print(f"  Episodes   : {args.episodes}")
    print(f"  Actions    : {TOTAL_ACTIONS}")
    print(f"  Vector     : {VECTOR_SIZE} dims")
    print(f"{'='*60}\n")

    # Charger les modèles de perception
    print("📦 Chargement des modèles de perception...")
    from clashai.navigation import game_loop
    models = game_loop.load_models()

    # Créer l'environnement V4
    env = ClashEnvV4(models=models, verbose=args.verbose)

    # Clan Castle Manager (V4.1 — demande de troupes pendant le training)
    from clashai.social.clan_castle import ClanCastleManager
    cc_manager = ClanCastleManager(
        building_detector=models.get('building_detector'),
        verbose=False,  # Moins de logs pendant le training
    )

    # Créer l'agent
    agent = PPOAgentV4()

    os.makedirs(RL_WEIGHTS_DIR, exist_ok=True)
    checkpoint_path = os.path.join(RL_WEIGHTS_DIR, 'agent_v4_checkpoint.pth')
    best_path = os.path.join(RL_WEIGHTS_DIR, 'agent_v4_best.pth')
    log_path = os.path.join(RL_WEIGHTS_DIR, 'training_log_v4.json')

    # Resume
    if args.resume and os.path.exists(checkpoint_path):
        agent.load(checkpoint_path)

    # Training log
    training_log = []
    if os.path.exists(log_path):
        with open(log_path) as f:
            training_log = json.load(f)

    best_reward = max((e['reward'] for e in training_log), default=-999)

    # =====================================================================
    #  Behavioral Cloning (V4.1) — pré-entraînement sur heuristique
    # =====================================================================
    if args.pretrain > 0:
        demonstrations = []

        for ep in range(1, args.pretrain + 1):
            print(f"\n  🎓 BC collecte {ep}/{args.pretrain}")

            obs, mask = env.reset()
            grid, vector = obs
            actions = env.get_heuristic_sequence()

            ep_reward = 0.0
            start_time = time.time()

            for action in actions:
                # Collecter la démonstration (obs → action)
                demonstrations.append((
                    grid.copy(), vector.copy(), action, mask.copy()
                ))

                obs, mask, reward, done, info = env.step(action)
                grid, vector = obs
                ep_reward += reward
                if done:
                    break

            # V4.1 fix: attendre la fin du combat si les troupes
            # sont encore en vie après la séquence heuristique
            if not done:
                from clashai.combat.action_space import ACTION_OBSERVE, ACTION_WAIT_LONG
                for _ in range(60):  # Max 60 steps de wind-down
                    # Alterner observe et wait pour suivre le combat
                    for wind_action in [ACTION_OBSERVE, ACTION_WAIT_LONG]:
                        demonstrations.append((
                            grid.copy(), vector.copy(), wind_action, mask.copy()
                        ))
                        obs, mask, reward, done, info = env.step(wind_action)
                        grid, vector = obs
                        ep_reward += reward
                        if done:
                            break
                    if done:
                        break

            elapsed = time.time() - start_time
            stars = info.get('stars', 0)
            pct = info.get('percentage', 0)

            entry = {
                'episode': len(training_log) + 1,
                'stars': stars,
                'percentage': pct,
                'reward': ep_reward,
                'steps': info.get('step', 0),
                'combat_steps': info.get('combat_steps', 0),
                'abilities': info.get('abilities_used', 0),
                'time': round(elapsed, 1),
                'timestamp': datetime.now().isoformat(),
                'mode': 'heuristic_bc',
            }
            training_log.append(entry)

            print(f"   📊 {stars}⭐ {pct}% | Reward: {ep_reward:.0f} | "
                  f"Demos: {len(demonstrations)}")

        # Entraîner le réseau par behavioral cloning
        accuracy = agent.pretrain_bc(
            demonstrations,
            epochs=args.bc_epochs,
        )

        # Sauvegarder le checkpoint pré-entraîné
        bc_path = os.path.join(RL_WEIGHTS_DIR, 'agent_v4_pretrained_bc.pth')
        agent.save(bc_path)
        agent.save(checkpoint_path)

        with open(log_path, 'w') as f:
            json.dump(training_log, f, indent=2)

        print(f"\n  ✅ BC terminé — {len(demonstrations)} démos, "
              f"accuracy {accuracy:.1%}")
        print(f"  Checkpoint sauvegardé → {bc_path}")

    # =====================================================================
    #  Training loop (heuristique ou PPO)
    # =====================================================================
    for episode in range(1, args.episodes + 1):
        print(f"\n{'='*60}")
        print(f"  ⚔️  Épisode {episode}/{args.episodes}")
        print(f"  {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}")

        # V4.1: demander des troupes CC si cooldown passé
        try:
            cc_manager.request_if_needed(
                game_loop.adb_screenshot, game_loop.adb_tap
            )
        except Exception:
            pass  # Non-bloquant

        obs, mask = env.reset()
        grid, vector = obs

        episode_reward = 0.0
        start_time = time.time()

        if args.heuristic:
            # Mode heuristique (collecte de démonstrations)
            actions = env.get_heuristic_sequence()
            for action in actions:
                _, log_prob, value = agent.select_action(grid, vector, mask)
                agent.store_step(grid, vector, action, log_prob, value, mask)

                obs, mask, reward, done, info = env.step(action)
                grid, vector = obs
                episode_reward += reward
                if done:
                    break

            # V4.1 fix: attendre la fin du combat
            if not done:
                from clashai.combat.action_space import ACTION_OBSERVE, ACTION_WAIT_LONG
                for _ in range(60):
                    for wind_action in [ACTION_OBSERVE, ACTION_WAIT_LONG]:
                        _, log_prob, value = agent.select_action(grid, vector, mask)
                        agent.store_step(grid, vector, wind_action, log_prob, value, mask)
                        obs, mask, reward, done, info = env.step(wind_action)
                        grid, vector = obs
                        episode_reward += reward
                        if done:
                            break
                    if done:
                        break
        else:
            # Mode PPO
            for step in range(MAX_STEPS_PER_EPISODE):
                action, log_prob, value = agent.select_action(grid, vector, mask)
                agent.store_step(grid, vector, action, log_prob, value, mask)

                obs, mask, reward, done, info = env.step(action)
                grid, vector = obs
                episode_reward += reward
                if done:
                    break

        # Fin d'épisode
        agent.buffer.end_episode(
            final_reward=episode_reward,
            step_rewards=env.get_step_rewards(),
        )

        elapsed = time.time() - start_time
        stars = info.get('stars', 0)
        pct = info.get('percentage', 0)

        entry = {
            'episode': len(training_log) + 1,
            'stars': stars,
            'percentage': pct,
            'reward': episode_reward,
            'steps': info.get('step', 0),
            'combat_steps': info.get('combat_steps', 0),
            'abilities': info.get('abilities_used', 0),
            'time': round(elapsed, 1),
            'timestamp': datetime.now().isoformat(),
        }
        training_log.append(entry)

        print(f"\n   📊 Résultat: {stars}⭐ {pct}% | "
              f"Reward: {episode_reward:.0f} | "
              f"Temps: {elapsed:.0f}s")

        # PPO update
        if agent.buffer_ready():
            stats = agent.update()
            if stats:
                print(f"   📈 PPO update #{stats['update']}: "
                      f"policy={stats['policy_loss']:.4f} "
                      f"value={stats['value_loss']:.4f} "
                      f"entropy={stats['entropy']:.4f}")

        # Checkpoint
        if episode % BATCH_SIZE == 0 or episode == args.episodes:
            agent.save(checkpoint_path)

            if episode_reward > best_reward:
                best_reward = episode_reward
                agent.save(best_path)
                print(f"   🏆 Nouveau meilleur : {best_reward:.0f}")

            with open(log_path, 'w') as f:
                json.dump(training_log, f, indent=2)

    # Summary
    env.close()
    print(f"\n{'='*60}")
    print("  🏁 Entraînement terminé")
    print(f"  Episodes : {args.episodes}")
    avg_stars = sum(e['stars'] for e in training_log[-args.episodes:]) / args.episodes
    avg_pct = sum(e['percentage'] for e in training_log[-args.episodes:]) / args.episodes
    print(f"  Étoiles moy : {avg_stars:.2f}")
    print(f"  Destruction moy : {avg_pct:.1f}%")
    print(f"  Meilleur reward : {best_reward:.0f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()