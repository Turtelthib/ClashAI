# clashai/combat/agent_v4/buffer.py
# RolloutBuffer — stores PPO trajectories + computes GAE advantages/returns.

import numpy as np
import torch

from clashai.combat.agent_v4.constants import GAMMA, GAE_LAMBDA


class RolloutBuffer:
    """PPO buffer for storing trajectories."""

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
