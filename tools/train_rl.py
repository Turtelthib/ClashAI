# scripts/rl/train_rl_v3.py
# Boucle d'entraînement V3 pour ClashAI — IA réactive mid-combat.
#
# Changements vs V2 :
#   - L'agent opère en 2 phases (deploy + combat)
#   - Logs enrichis avec stats combat (abilities, combat steps)
#   - Compatible avec les checkpoints V2 (chargement partiel)
#
# Usage :
#   python scripts/rl/train_rl_v3.py                     (RL)
#   python scripts/rl/train_rl_v3.py --heuristic          (baseline)
#   python scripts/rl/train_rl_v3.py --resume              (reprendre)
#   python scripts/rl/train_rl_v3.py --episodes 50
#   python scripts/rl/train_rl_v3.py --from-v2 weights/rl_v2/agent_v2_best.pth

import os
import time
import json
import argparse
from datetime import datetime

import numpy as np

# Setup
from clashai.paths import PROJECT_ROOT as project_root

from clashai.combat.environment import ClashEnvV3
from clashai.combat.agent import (
    PPOAgentV3, MAX_STEPS_PER_EPISODE,
)


# =============================================================================
#                         CONFIGURATION
# =============================================================================

DEFAULT_EPISODES = 100
CHECKPOINT_EVERY = 8
LOG_EVERY = 1

WEIGHTS_DIR = os.path.join(project_root, 'weights')
RL_DIR = os.path.join(WEIGHTS_DIR, 'rl_v3')
CHECKPOINT_PATH = os.path.join(RL_DIR, 'agent_v3_checkpoint.pth')
BEST_PATH = os.path.join(RL_DIR, 'agent_v3_best.pth')
LOG_PATH = os.path.join(RL_DIR, 'training_log_v3.json')


# =============================================================================
#                        LOGGING
# =============================================================================

class TrainingLogger:
    def __init__(self, log_path=LOG_PATH):
        self.log_path = log_path
        self.episodes = []
        self.updates = []
        self.best_reward = -float('inf')
        self.start_time = None

        if os.path.exists(log_path):
            try:
                with open(log_path, 'r') as f:
                    data = json.load(f)
                self.episodes = data.get('episodes', [])
                self.updates = data.get('updates', [])
                self.best_reward = data.get('best_reward', -float('inf'))
                print(f"📊 Log chargé : {len(self.episodes)} épisodes, "
                      f"best reward = {self.best_reward:.0f}")
            except (json.JSONDecodeError, Exception) as e:
                print(f"⚠️  Log corrompu ({e}), nouveau log")

    def start(self):
        self.start_time = time.time()

    def log_episode(self, ep_num, info):
        """Log un épisode. Retourne True si c'est un nouveau record."""
        entry = {
            'episode': ep_num,
            'timestamp': datetime.now().isoformat(),
            'stars': info.get('stars', 0),
            'percentage': info.get('percentage', 0),
            'reward': info.get('reward', 0),
            'combat_reward': info.get('combat_reward', 0),
            'shaping_total': info.get('shaping_total', 0),
            'steps': info.get('steps', 0),
            'deploy_steps': info.get('deploy_steps', 0),
            'combat_steps': info.get('combat_steps', 0),
            'abilities_used': info.get('abilities_used', 0),
            'troops_remaining': info.get('troops_remaining', 0),
        }
        self.episodes.append(entry)

        is_best = False
        reward = info.get('reward', 0)
        if reward > self.best_reward:
            self.best_reward = reward
            is_best = True

        self._save()
        return is_best

    def log_update(self, stats):
        """Log une PPO update."""
        entry = {
            'update': stats['update'],
            'timestamp': datetime.now().isoformat(),
            'policy_loss': stats['policy_loss'],
            'value_loss': stats['value_loss'],
            'entropy': stats['entropy'],
            'total_steps': stats['total_steps'],
        }
        self.updates.append(entry)
        self._save()

    def _save(self):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        tmp_path = self.log_path + '.tmp'
        with open(tmp_path, 'w') as f:
            json.dump({
                'episodes': self.episodes,
                'updates': self.updates,
                'best_reward': self.best_reward,
                'version': 'v3',
            }, f, indent=2)
        os.replace(tmp_path, self.log_path)

    def print_summary(self, last_n=10):
        """Affiche un résumé des derniers épisodes."""
        recent = self.episodes[-last_n:]
        if not recent:
            return

        rewards = [e['reward'] for e in recent]
        stars = [e['stars'] for e in recent]
        pcts = [e['percentage'] for e in recent]
        abilities = [e.get('abilities_used', 0) for e in recent]
        combat_steps = [e.get('combat_steps', 0) for e in recent]

        print(f"\n   📈 Derniers {len(recent)} épisodes :")
        print(f"      Reward moyen  : {np.mean(rewards):.0f} "
              f"(min={min(rewards):.0f}, max={max(rewards):.0f})")
        print(f"      Stars         : {dict(zip(*np.unique(stars, return_counts=True)))}")
        print(f"      Destruction   : {np.mean(pcts):.1f}%")
        print(f"      Abilities/ep  : {np.mean(abilities):.1f}")
        print(f"      Combat steps  : {np.mean(combat_steps):.1f}")
        print(f"      Best all-time : {self.best_reward:.0f}")


# =============================================================================
#                       MAIN TRAINING LOOP
# =============================================================================

def train(num_episodes=DEFAULT_EPISODES, mode='rl', resume=False,
          from_v2=None):
    """
    Boucle principale d'entraînement V3.
    
    Args:
        num_episodes: nombre d'épisodes à jouer
        mode: 'rl' ou 'heuristic'
        resume: reprendre depuis le checkpoint V3
        from_v2: chemin vers un checkpoint V2 (chargement partiel)
    """
    os.makedirs(RL_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  ClashAI V3 — Entraînement {mode.upper()}")
    print(f"  {num_episodes} épisodes")
    print(f"{'='*60}\n")

    # 1. Charger les modèles de perception
    print("📦 Chargement des modèles de perception...")
    from clashai.navigation import game_loop
    models = game_loop.load_models()

    # 2. Créer l'environnement V3
    env = ClashEnvV3(models=models, verbose=True)

    # 3. Créer l'agent V3
    agent = PPOAgentV3()

    if resume:
        if agent.load(CHECKPOINT_PATH):
            print("✅ Reprise depuis le checkpoint V3")
        else:
            print("⚠️  Pas de checkpoint V3, démarrage from scratch")
    elif from_v2:
        if agent.load(from_v2):
            print(f"✅ Chargement partiel depuis V2 : {from_v2}")
        else:
            print("⚠️  Chargement V2 échoué, démarrage from scratch")

    # 4. Logger
    logger = TrainingLogger()
    logger.start()
    start_episode = len(logger.episodes)

    print(f"\n🚀 Début V3 (épisode {start_episode+1} → "
          f"{start_episode+num_episodes})\n")

    try:
        for ep_offset in range(num_episodes):
            ep_num = start_episode + ep_offset + 1
            ep_start = time.time()

            print(f"\n{'='*60}")
            print(f"  ÉPISODE #{ep_num} ({mode})")
            print(f"{'='*60}")

            # Reset
            obs, mask = env.reset()
            grid, vector = obs

            if mode == 'rl':
                agent.buffer.start_episode()

            # --- Boucle de décisions (deploy + combat) ---
            if mode == 'heuristic':
                heuristic_actions = env.get_heuristic_sequence()
                print(f"   🧠 Heuristique : {len(heuristic_actions)} actions")

                for action in heuristic_actions:
                    obs, mask, reward, done, info = env.step(action)
                    grid, vector = obs
                    if done:
                        break
            else:
                # RL : l'agent décide dans les deux phases
                for step in range(MAX_STEPS_PER_EPISODE):
                    action, log_prob, value = agent.select_action(
                        grid, vector, mask
                    )
                    agent.store_step(grid, vector, action, log_prob, value, mask)

                    obs, mask, reward, done, info = env.step(action)
                    grid, vector = obs

                    if done:
                        break

                # Terminer l'épisode
                combat_reward = info.get('combat_reward', reward)
                step_rewards = env.get_step_rewards()
                agent.buffer.end_episode(combat_reward, step_rewards)
                agent.total_episodes += 1

            # --- Logger ---
            is_best = logger.log_episode(ep_num, info)
            ep_duration = time.time() - ep_start

            # Affichage
            shaping_str = ""
            if 'shaping_total' in info:
                shaping_str = f" shaping={info['shaping_total']:+.0f}"

            combat_str = ""
            if info.get('combat_steps', 0) > 0:
                combat_str = (f" | combat={info['combat_steps']}steps"
                             f" abilities={info.get('abilities_used', 0)}")

            best_marker = " 🏆 RECORD !" if is_best else ""

            print(f"\n   📊 Ep #{ep_num}: "
                  f"{info.get('stars', 0)}⭐ {info.get('percentage', 0)}% "
                  f"reward={info.get('reward', 0):.0f}{shaping_str}"
                  f"{combat_str}"
                  f" ({ep_duration:.0f}s){best_marker}")

            # --- PPO Update ---
            if mode == 'rl' and agent.buffer_ready():
                print(f"\n   🔄 PPO Update "
                      f"({agent.buffer.total_steps()} steps)...")
                stats = agent.update()
                if stats:
                    logger.log_update(stats)
                    print(f"   ✅ Update #{stats['update']}: "
                          f"policy={stats['policy_loss']:.4f} "
                          f"value={stats['value_loss']:.4f} "
                          f"entropy={stats['entropy']:.4f}")

            # --- Checkpoint ---
            if ep_num % CHECKPOINT_EVERY == 0 and mode == 'rl':
                agent.save(CHECKPOINT_PATH)
                if is_best:
                    agent.save(BEST_PATH)

            # --- Résumé périodique ---
            if ep_num % 10 == 0:
                logger.print_summary()

    except KeyboardInterrupt:
        print("\n\n⛔ Arrêt demandé (Ctrl+C)")
        if mode == 'rl':
            agent.save(CHECKPOINT_PATH)
            print("💾 Checkpoint sauvegardé")

    finally:
        env.close()

    # Résumé final
    logger.print_summary(last_n=20)
    print("\n✅ Entraînement V3 terminé !")
    print(f"   Épisodes     : {num_episodes}")
    print(f"   Best reward  : {logger.best_reward:.0f}")
    print(f"   Checkpoint   : {CHECKPOINT_PATH}")
    print(f"   Best agent   : {BEST_PATH}")
    print(f"   Logs         : {LOG_PATH}")


# =============================================================================
#                        MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ClashAI V3 Training")
    parser.add_argument('--episodes', type=int, default=DEFAULT_EPISODES)
    parser.add_argument('--heuristic', action='store_true',
                        help="Mode heuristique (baseline)")
    parser.add_argument('--resume', action='store_true',
                        help="Reprendre depuis le checkpoint V3")
    parser.add_argument('--from-v2', type=str, default=None,
                        help="Charger un checkpoint V2 (partiel)")

    args = parser.parse_args()

    mode = 'heuristic' if args.heuristic else 'rl'

    train(
        num_episodes=args.episodes,
        mode=mode,
        resume=args.resume,
        from_v2=args.from_v2,
    )