# clashai/agents/scheduler.py
# Simple priority + cooldown scheduler for sub-agents.
#
# The brain owns one AgentScheduler. Each tick:
#   1. Build a `world` snapshot (latest perception + bookkeeping flags)
#   2. Ask every registered agent if it's ready (state OK + can_run() vote)
#   3. Pick the highest-priority ready agent
#   4. Run it synchronously
#   5. Record telemetry, repeat
#
# This is intentionally synchronous and single-agent-at-a-time — multiple
# agents running in parallel would fight over ADB / screen / window. If
# we ever need parallelism (e.g. perception in background while combat
# runs), it lives in the agent itself (e.g. PerceptionThread inside the
# CombatAgent), not in the scheduler.

import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

from clashai.agents.base import BaseAgent, AgentResult


class AgentScheduler:
    """
    Registry + dispatcher for sub-agents.

    Usage:
        sched = AgentScheduler()
        sched.register(CombatAgent())
        sched.register(ClanCastleAgent())
        sched.register(ClanChatAgent())

        while True:
            world = build_world_snapshot()
            picked = sched.pick(world)
            if picked is None:
                time.sleep(1)
                continue
            result = sched.run(picked)
            log(picked.name, result)
    """

    def __init__(self, history_size: int = 64):
        self._agents: List[BaseAgent] = []
        self._history: Deque[Tuple[str, AgentResult]] = deque(maxlen=history_size)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, agent: BaseAgent) -> None:
        if agent in self._agents:
            return
        self._agents.append(agent)
        agent.on_register(self)
        # Re-sort by priority (descending) so pick() is just a scan.
        self._agents.sort(key=lambda a: -a.priority)

    def unregister(self, agent: BaseAgent) -> None:
        if agent in self._agents:
            agent.on_unregister()
            self._agents.remove(agent)

    def get(self, name: str) -> Optional[BaseAgent]:
        for a in self._agents:
            if a.name == name:
                return a
        return None

    def all_agents(self) -> List[BaseAgent]:
        return list(self._agents)

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------

    def pick(self, world: Dict[str, Any]) -> Optional[BaseAgent]:
        """
        Pick the highest-priority agent that says it's ready.

        Returns None if no agent is currently runnable (all on cooldown
        or all voting no on `can_run`).
        """
        # _agents is already sorted by descending priority.
        for agent in self._agents:
            if agent.is_ready(world):
                return agent
        return None

    def run(self, agent: BaseAgent) -> AgentResult:
        """Synchronously execute `agent`. Records telemetry."""
        result = agent._execute()
        self._history.append((agent.name, result))
        return result

    def tick(self, world: Dict[str, Any]) -> Optional[Tuple[BaseAgent, AgentResult]]:
        """One full pick+run cycle. Returns (agent, result) or None."""
        agent = self.pick(world)
        if agent is None:
            return None
        return agent, self.run(agent)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def status(self) -> List[Dict[str, Any]]:
        """Snapshot of every registered agent — feeds the dashboard."""
        now = time.time()
        out = []
        for a in self._agents:
            out.append({
                'name': a.name,
                'priority': a.priority,
                'state': a.get_state().value,
                'cooldown_remaining': round(a.remaining_cooldown(), 1),
                'last_run_at': a._last_run_at,
                'seconds_since_last_run': (now - a._last_run_at) if a._last_run_at else None,
                'consecutive_errors': a._consecutive_errors,
            })
        return out

    def history(self, n: Optional[int] = None) -> List[Tuple[str, AgentResult]]:
        """Last `n` runs (or all if None) — most recent last."""
        if n is None:
            return list(self._history)
        return list(self._history)[-n:]

    def shutdown(self) -> None:
        """Cleanly shut down every registered agent."""
        for a in self._agents:
            try:
                a.shutdown()
            except Exception as e:
                print(f"WARNING: shutdown of {a.name} raised: {e}")
        self._agents.clear()
