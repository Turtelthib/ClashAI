# clashai/combat/agent_v4/agent.py
# PPOAgentV4 — action selection + PPO update + checkpoint I/O (+ BC mixin).

import torch
import torch.optim as optim
import torch.nn as nn
from torch.distributions import Categorical

from clashai.combat.action_space import (
    TOTAL_ACTIONS, NUM_ROLES, NUM_SECTORS, NUM_HEROES, SPELL_NAMES,
)
from clashai.combat.agent_v4.constants import (
    VECTOR_SIZE, LEARNING_RATE, BATCH_SIZE, PPO_EPOCHS,
    CLIP_EPSILON, ENTROPY_COEF, VALUE_COEF, MAX_GRAD_NORM,
)
from clashai.combat.agent_v4.network import ActorCriticV4
from clashai.combat.agent_v4.buffer import RolloutBuffer
from clashai.combat.agent_v4.bc import BehavioralCloningMixin


class PPOAgentV4(BehavioralCloningMixin):
    """
    PPO Agent V4 — simplified action space.

    37 actions instead of 289.
    ~400K parameters instead of 1.2M.
    Estimated convergence ~10× faster.
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
        print("Agent PPO V4 initialisé")
        print(f" Device : {self.device}")
        print(f" Actions : {TOTAL_ACTIONS} "
              f"({NUM_ROLES}×{NUM_SECTORS} deploy + "
              f"{len(SPELL_NAMES)} sorts + "
              f"{NUM_HEROES} abilities + observe + 3 ctrl)")
        print(f" Vector : {VECTOR_SIZE} dims")
        print(f" Parameters : {n_params:,}")
        print(f" Batch size : {BATCH_SIZE} episodes")

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

            # Normalize returns before MSE to prevent value loss explosion
            # when rewards have high variance (e.g. [-50 ... +641]).
            ret = batch['returns']
            ret_norm = (ret - ret.mean()) / (ret.std() + 1e-8)
            value_loss = nn.MSELoss()(values.squeeze(), ret_norm)

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
        print(f"Agent V4 sauvegardé → {path}")

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.update_count = checkpoint.get('update_count', 0)
        self.total_episodes = checkpoint.get('total_episodes', 0)
        print(f"Agent V4 chargé ← {path}")
        print(f" Updates: {self.update_count}, Episodes: {self.total_episodes}")
