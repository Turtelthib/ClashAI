# clashai/agents/__init__.py
# Multi-agent framework (V5.1+).
#
# An "agent" is a unit of behaviour the brain can run on demand:
#   - combat (attack a base)
#   - clan castle (request troops every 15 min)
#   - clan chat (read & respond to commands)
#   - GdC (clan war attack)
#   - village management / clan games (V5.2)
#
# Each agent inherits BaseAgent and is registered with the
# AgentScheduler, which decides who runs next based on priority +
# cooldown + can_run() votes.

from clashai.agents.base import BaseAgent, RunState, AgentResult
from clashai.agents.scheduler import AgentScheduler
from clashai.agents.world import build_world, WORLD_KEYS
from clashai.agents.clan_castle_agent import ClanCastleAgent
from clashai.agents.combat_agent import CombatAgent
from clashai.agents.gdc_agent import GdCAgent
from clashai.agents.chat_agent import ChatAgent

__all__ = [
    'BaseAgent', 'RunState', 'AgentResult', 'AgentScheduler',
    'build_world', 'WORLD_KEYS',
    'ClanCastleAgent', 'CombatAgent', 'GdCAgent', 'ChatAgent',
]
