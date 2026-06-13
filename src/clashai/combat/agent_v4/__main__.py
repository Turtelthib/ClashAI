# clashai/combat/agent_v4/__main__.py
# Smoke test: `uv run python -m clashai.combat.agent_v4`

import numpy as np

from clashai.combat.action_space import TOTAL_ACTIONS, decode_action
from clashai.combat.agent_v4.constants import GRID_CHANNELS, GRID_SIZE, VECTOR_SIZE
from clashai.combat.agent_v4.agent import PPOAgentV4


def main():
    print("Test Agent V4\n")

    agent = PPOAgentV4()

    grid = np.random.randn(GRID_CHANNELS, GRID_SIZE, GRID_SIZE).astype(np.float32)
    vector = np.random.randn(VECTOR_SIZE).astype(np.float32)
    mask = np.ones(TOTAL_ACTIONS, dtype=np.float32)

    action, log_prob, value = agent.select_action(grid, vector, mask)
    action_type, idx1, idx2 = decode_action(action)
    print(f"\n Action: {action} → {action_type} {idx1} {idx2}")
    print(f" Log prob: {log_prob:.4f}")
    print(f" Value: {value:.4f}")

    print("\nAgent V4 OK")


if __name__ == "__main__":
    main()
