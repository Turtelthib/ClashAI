# clashai/agents/base.py
# Abstract base class for every sub-agent.
#
# Concrete agents (CombatAgent, ClanCastleAgent, ClanChatAgent, GdcAgent…)
# override:
#   - name            : str — unique identifier for logging / dashboard
#   - priority        : int — higher = preempts lower
#   - cooldown_seconds: float — minimum gap between two runs
#   - can_run(world)  : bool — voting function consulted by the scheduler
#   - run()           : AgentResult — actually do the work
#
# The scheduler handles cooldown bookkeeping (last_run_at), so concrete
# agents don't have to remember when they last ran.

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class RunState(Enum):
    """Lifecycle state of an agent at a given point in time."""

    IDLE = 'idle'              # registered but not currently running
    RUNNING = 'running'        # run() is in progress on a worker thread
    COOLDOWN = 'cooldown'      # finished recently, can't run again yet
    DISABLED = 'disabled'      # manually disabled or repeatedly failing
    ERRORED = 'errored'        # last run() raised; needs investigation


@dataclass
class AgentResult:
    """
    Telemetry returned by run().

    The scheduler stores this in history so the dashboard can show "what
    happened in the last 5 runs of CombatAgent". `data` is a free-form
    dict — agents put whatever they want (stars, troops requested, etc.).
    """

    ok: bool                              # True if the run succeeded
    duration_s: float                     # wall-clock seconds spent
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None           # short error string if ok=False


class BaseAgent(ABC):
    """
    Abstract base class for every sub-agent.

    Subclasses must define `name`, `priority`, `cooldown_seconds` as class
    attributes (or pass them to __init__), and implement `can_run` + `run`.

    Concrete example:

        class ClanCastleAgent(BaseAgent):
            name = 'clan_castle'
            priority = 20
            cooldown_seconds = 15 * 60   # 15 min

            def can_run(self, world):
                return world.get('on_village_home', False)

            def run(self):
                self._cc_manager.request_troops()
                return AgentResult(ok=True, duration_s=0.0)
    """

    name: str = 'base'
    priority: int = 0
    cooldown_seconds: float = 0.0

    def __init__(self, **kwargs):
        # Allow per-instance overrides without subclassing for one-offs.
        for k, v in kwargs.items():
            if hasattr(type(self), k) or k in ('name', 'priority', 'cooldown_seconds'):
                setattr(self, k, v)

        self._last_run_at: Optional[float] = None
        self._state: RunState = RunState.IDLE
        self._consecutive_errors: int = 0
        # Tracker — set by the scheduler when it picks this agent.
        self._scheduler = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @abstractmethod
    def can_run(self, world: Dict[str, Any]) -> bool:
        """
        Return True if this agent is willing to execute right now.

        `world` is whatever the scheduler considers shared context —
        typically the latest perception snapshot, plus flags like
        `attacks_done_in_a_row`. Agents should NOT use this method to do
        I/O — keep it cheap.
        """

    @abstractmethod
    def run(self) -> AgentResult:
        """Actually do the work. Returns telemetry."""

    # ------------------------------------------------------------------
    # State helpers (manipulated by the scheduler — agents shouldn't call)
    # ------------------------------------------------------------------

    def get_state(self) -> RunState:
        if self._state == RunState.IDLE and self.remaining_cooldown() > 0:
            return RunState.COOLDOWN
        return self._state

    def remaining_cooldown(self) -> float:
        if self._last_run_at is None or self.cooldown_seconds <= 0:
            return 0.0
        elapsed = time.time() - self._last_run_at
        return max(0.0, self.cooldown_seconds - elapsed)

    def is_ready(self, world: Dict[str, Any]) -> bool:
        """True if state allows running AND can_run says yes."""
        if self._state == RunState.DISABLED:
            return False
        if self.remaining_cooldown() > 0:
            return False
        try:
            return self.can_run(world)
        except Exception:
            return False

    # Default hooks — subclasses override if needed
    def on_register(self, scheduler) -> None:
        """Called when the scheduler picks up the agent."""
        self._scheduler = scheduler

    def on_unregister(self) -> None:
        """Called when removed from the scheduler."""
        self._scheduler = None

    def shutdown(self) -> None:
        """Release resources (close env, stop threads, …). Default no-op."""

    # ------------------------------------------------------------------
    # Internal — used by AgentScheduler.run_one()
    # ------------------------------------------------------------------

    def _execute(self) -> AgentResult:
        """Wrap run() with state + timing + error bookkeeping."""
        self._state = RunState.RUNNING
        start = time.time()
        try:
            result = self.run()
            self._consecutive_errors = 0
            self._state = RunState.IDLE
        except Exception as e:
            self._consecutive_errors += 1
            self._state = RunState.ERRORED
            result = AgentResult(
                ok=False,
                duration_s=time.time() - start,
                error=f'{type(e).__name__}: {e}',
            )
        finally:
            self._last_run_at = time.time()
        return result

    def __repr__(self) -> str:
        return (f'<{type(self).__name__} name={self.name!r} '
                f'prio={self.priority} state={self.get_state().value}>')
