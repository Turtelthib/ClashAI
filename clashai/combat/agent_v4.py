# clashai/combat/agent_v4.py
# Agent PPO V4 pour ClashAI.
#
# Changements vs V3 :
#   - 37 actions au lieu de 289 (rôle × secteur + sorts auto-ciblés)
#   - Observation compactée : role_counts(5) au lieu de troop_counts(14)
#   - Réseau plus petit → entraînement plus rapide
#   - Même architecture CNN + MLP → shared → actor/critic
#
# Usage :
#   agent = PPOAgentV4()
#   action, log_prob, value = agent.select_action(grid, vector, mask)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from clashai.combat.action_space import (
    TOTAL_ACTIONS, NUM_ROLES, NUM_SECTORS, NUM_HEROES,
    SPELL_NAMES,
    decode_action,
)
from clashai.combat.combat_observer import COMBAT_FEATURES_SIZE


# =============================================================================
#                         OBSERVATION SPACE
# =============================================================================

GRID_CHANNELS = 12
GRID_SIZE = 40
VILLAGE_FEATURES = 20

# V4 : vecteur compacté
ROLE_FEATURES = NUM_ROLES           # 5 (tank, ranged, melee, hero, siege)
SPELL_FEATURES = len(SPELL_NAMES)   # 3 (soin, rage, gel)
SECTOR_MAP_SIZE = NUM_SECTORS       # 5 (deploy density par secteur)
STEP_FEATURES = 1
HERO_STATUS_SIZE = NUM_HEROES       # 5
PHASE_SIZE = 1                      # 0=deploy, 1=combat

VECTOR_SIZE = (VILLAGE_FEATURES + ROLE_FEATURES + SPELL_FEATURES
               + SECTOR_MAP_SIZE + STEP_FEATURES
               + COMBAT_FEATURES_SIZE + HERO_STATUS_SIZE
               + PHASE_SIZE)
# 20 + 5 + 3 + 5 + 1 + 15 + 5 + 1 = 55

# PPO Hyperparamètres
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
ENTROPY_COEF = 0.04          # Plus élevé (moins d'actions → plus d'exploration)
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5
LEARNING_RATE = 3e-4          # Légèrement plus élevé (réseau plus petit)
PPO_EPOCHS = 4
BATCH_SIZE = 8


# =============================================================================
#                     RÉSEAU ACTOR-CRITIC V4
# =============================================================================

class ActorCriticV4(nn.Module):
    """
    Réseau Actor-Critic V4.

    Plus compact que V3 :
        - Vecteur 55 dims (vs 76)
        - Actor output 37 actions (vs 289)
        - Backbone partagé 192 dims (vs 256)
        - ~400K paramètres (vs 1.2M)
    """

    def __init__(self):
        super().__init__()

        self.grid_cnn = nn.Sequential(
            nn.Conv2d(GRID_CHANNELS, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.grid_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 5 * 5, 192),
            nn.ReLU(),
        )

        self.vector_fc = nn.Sequential(
            nn.Linear(VECTOR_SIZE, 96),
            nn.ReLU(),
            nn.Linear(96, 64),
            nn.ReLU(),
        )

        self.shared = nn.Sequential(
            nn.Linear(256, 192),
            nn.ReLU(),
            nn.Linear(192, 192),
            nn.ReLU(),
        )

        self.actor = nn.Sequential(
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Linear(128, TOTAL_ACTIONS),
        )

        self.critic = nn.Sequential(
            nn.Linear(192, 96),
            nn.ReLU(),
            nn.Linear(96, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def forward(self, grid, vector, action_mask=None):
        """
        Args:
            grid: (batch, 12, 40, 40)
            vector: (batch, 55)
            action_mask: (batch, 37) — 1.0 = valide
        """
        g = self.grid_cnn(grid)
        g = self.grid_fc(g)
        v = self.vector_fc(vector)

        combined = torch.cat([g, v], dim=1)
        shared = self.shared(combined)

        logits = self.actor(shared)
        value = self.critic(shared)

        if action_mask is not None:
            logits = logits + (action_mask - 1.0) * 1e8

        return logits, value


# =============================================================================
#                         ROLLOUT BUFFER
# =============================================================================

class RolloutBuffer:
    """Buffer PPO pour stocker les trajectoires."""

    def __init__(self):
        self.episodes = []
        self._current = []

    def start_episode(self):
        self._current = []

    def store_step(self, grid, vector, action, log_prob, value, action_mask):
        self._current.append({
            'grid': grid.copy() if isinstance(grid, np.ndarray) else grid,
            'vector': vector.copy() if isinstance(vector, np.ndarray) else vector,
            'action': action,
            'log_prob': log_prob,
            'value': value,
            'action_mask': action_mask.copy() if isinstance(action_mask, np.ndarray) else action_mask,
        })

    def end_episode(self, final_reward, step_rewards=None):
        if not self._current:
            return
        n = len(self._current)

        if step_rewards and len(step_rewards) >= n:
            rewards = [float(step_rewards[i]) for i in range(n)]
            rewards[-1] += final_reward
        else:
            rewards = [0.0] * n
            rewards[-1] = final_reward

        for i, step in enumerate(self._current):
            step['reward'] = rewards[i]
            step['done'] = (i == n - 1)

        self.episodes.append(self._current)
        self._current = []

    def num_episodes(self):
        return len(self.episodes)

    def total_steps(self):
        return sum(len(ep) for ep in self.episodes)

    def clear(self):
        self.episodes.clear()
        self._current = []

    def get_batch(self, device):
        all_g, all_v, all_a, all_lp, all_val = [], [], [], [], []
        all_m, all_adv, all_ret = [], [], []

        for episode in self.episodes:
            n = len(episode)
            rewards = [s['reward'] for s in episode]
            values = [s['value'] for s in episode]

            advantages, returns = [], []
            gae = 0.0
            for t in reversed(range(n)):
                next_val = values[t + 1] if t < n - 1 else 0.0
                delta = rewards[t] + GAMMA * next_val - values[t]
                gae = delta + GAMMA * GAE_LAMBDA * gae
                advantages.insert(0, gae)
                returns.insert(0, gae + values[t])

            for i, step in enumerate(episode):
                all_g.append(step['grid'])
                all_v.append(step['vector'])
                all_a.append(step['action'])
                all_lp.append(step['log_prob'])
                all_val.append(step['value'])
                all_m.append(step['action_mask'])
                all_adv.append(advantages[i])
                all_ret.append(returns[i])

        batch = {
            'grids': torch.FloatTensor(np.array(all_g)).to(device),
            'vectors': torch.FloatTensor(np.array(all_v)).to(device),
            'actions': torch.LongTensor(all_a).to(device),
            'log_probs': torch.FloatTensor(all_lp).to(device),
            'values': torch.FloatTensor(all_val).to(device),
            'masks': torch.FloatTensor(np.array(all_m)).to(device),
            'advantages': torch.FloatTensor(all_adv).to(device),
            'returns': torch.FloatTensor(all_ret).to(device),
        }

        adv = batch['advantages']
        if len(adv) > 1:
            batch['advantages'] = (adv - adv.mean()) / (adv.std() + 1e-8)

        return batch


# =============================================================================
#                         AGENT PPO V4
# =============================================================================

class PPOAgentV4:
    """
    Agent PPO V4 — action space simplifié.

    37 actions au lieu de 289.
    ~400K paramètres au lieu de 1.2M.
    Convergence estimée ~10× plus rapide.
    """

    def __init__(self, device=None, lr=LEARNING_RATE):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.network = ActorCriticV4().to(self.device)
        self.optimizer = optim.Adam(
            self.network.parameters(), lr=lr, eps=1e-5
        )
        self.buffer = RolloutBuffer()

        self.update_count = 0
        self.total_episodes = 0

        n_params = sum(p.numel() for p in self.network.parameters())
        print("🤖 Agent PPO V4 initialisé")
        print(f"   Device      : {self.device}")
        print(f"   Actions     : {TOTAL_ACTIONS} "
              f"({NUM_ROLES}×{NUM_SECTORS} deploy + "
              f"{len(SPELL_NAMES)} sorts + "
              f"{NUM_HEROES} abilities + observe + 3 ctrl)")
        print(f"   Vector      : {VECTOR_SIZE} dims")
        print(f"   Paramètres  : {n_params:,}")
        print(f"   Batch size  : {BATCH_SIZE} épisodes")

    def select_action(self, grid, vector, action_mask):
        """
        Choisit une action.

        Args:
            grid: np.array (12, 40, 40)
            vector: np.array (VECTOR_SIZE,)
            action_mask: np.array (37,)

        Returns:
            action: int
            log_prob: float
            value: float
        """
        grid_t = torch.FloatTensor(grid).unsqueeze(0).to(self.device)
        vec_t = torch.FloatTensor(vector).unsqueeze(0).to(self.device)
        mask_t = torch.FloatTensor(action_mask).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, value = self.network(grid_t, vec_t, mask_t)

        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.squeeze().item()

    def store_step(self, grid, vector, action, log_prob, value, action_mask):
        self.buffer.store_step(grid, vector, action, log_prob, value, action_mask)

    def buffer_ready(self):
        return self.buffer.num_episodes() >= BATCH_SIZE

    def update(self):
        """PPO update sur le batch accumulé."""
        if not self.buffer_ready():
            return None

        self.update_count += 1
        batch = self.buffer.get_batch(self.device)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for _ in range(PPO_EPOCHS):
            logits, values = self.network(
                batch['grids'], batch['vectors'], batch['masks']
            )
            dist = Categorical(logits=logits)

            new_log_probs = dist.log_prob(batch['actions'])
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - batch['log_probs'])
            surr1 = ratio * batch['advantages']
            surr2 = torch.clamp(
                ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON
            ) * batch['advantages']
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = nn.MSELoss()(values.squeeze(), batch['returns'])

            loss = (policy_loss
                    + VALUE_COEF * value_loss
                    - ENTROPY_COEF * entropy)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.network.parameters(), MAX_GRAD_NORM
            )
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()

        self.total_episodes += self.buffer.num_episodes()
        stats = {
            'policy_loss': total_policy_loss / PPO_EPOCHS,
            'value_loss': total_value_loss / PPO_EPOCHS,
            'entropy': total_entropy / PPO_EPOCHS,
            'update': self.update_count,
            'total_episodes': self.total_episodes,
            'batch_steps': len(batch['actions']),
        }

        self.buffer.clear()
        return stats

    def save(self, path):
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'update_count': self.update_count,
            'total_episodes': self.total_episodes,
        }, path)
        print(f"💾 Agent V4 sauvegardé → {path}")

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.update_count = checkpoint.get('update_count', 0)
        self.total_episodes = checkpoint.get('total_episodes', 0)
        print(f"📦 Agent V4 chargé ← {path}")
        print(f"   Updates: {self.update_count}, Episodes: {self.total_episodes}")


# =============================================================================
#                         TEST
# =============================================================================

if __name__ == "__main__":
    print("🧪 Test Agent V4\n")

    agent = PPOAgentV4()

    grid = np.random.randn(GRID_CHANNELS, GRID_SIZE, GRID_SIZE).astype(np.float32)
    vector = np.random.randn(VECTOR_SIZE).astype(np.float32)
    mask = np.ones(TOTAL_ACTIONS, dtype=np.float32)

    action, log_prob, value = agent.select_action(grid, vector, mask)
    action_type, idx1, idx2 = decode_action(action)
    print(f"\n   Action: {action} → {action_type} {idx1} {idx2}")
    print(f"   Log prob: {log_prob:.4f}")
    print(f"   Value: {value:.4f}")

    print("\n✅ Agent V4 OK")
