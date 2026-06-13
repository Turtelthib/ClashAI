# clashai/combat/agent_v4/network.py
# ActorCriticV4 — shared CNN+MLP backbone with actor (37 actions) + critic heads.

import numpy as np
import torch
import torch.nn as nn

from clashai.combat.action_space import TOTAL_ACTIONS
from clashai.combat.agent_v4.constants import GRID_CHANNELS, VECTOR_SIZE


class ActorCriticV4(nn.Module):
    """
    Actor-Critic Network V4.

    More compact than V3 :
        - 54-dim vector (vs 76) — V4.2: 55→54, merged phase
        - Actor output 37 actions (vs 289)
        - 192-dim shared backbone (vs 256)
        - ~400K parameters (vs 1.2M)
    """

    def __init__(self):
        super().__init__()

        # CNN for the village grid
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

        # MLP for the vector
        self.vector_fc = nn.Sequential(
            nn.Linear(VECTOR_SIZE, 96),
            nn.ReLU(),
            nn.Linear(96, 64),
            nn.ReLU(),
        )

        # Fusion → shared backbone
        # 192 (grid) + 64 (vector) = 256
        self.shared = nn.Sequential(
            nn.Linear(256, 192),
            nn.ReLU(),
            nn.Linear(192, 192),
            nn.ReLU(),
        )

        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Linear(128, TOTAL_ACTIONS),
        )

        # Critic head
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
            vector: (batch, 54)
            action_mask: (batch, 37) — 1.0 = valid
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
