# clashai/brain/interface.py
# Brain — swappable decision layer. Given the world snapshot, choose which
# agent runs next (or None to idle). This is THE seam for the project's
# end goal: today HeuristicBrain (mechanical priority + cooldown + can_run via
# the AgentScheduler); later LocalLLMBrain (a local LLM that reasons in natural
# language, with a blackboard of goals/resources). See [[project_llm_brain_vision]].

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from clashai.agents.base import BaseAgent
from clashai.agents.scheduler import AgentScheduler


class Brain(ABC):
    """Decides what the bot does next, given the perceived world."""

    @abstractmethod
    def decide(self, world: Dict[str, Any]) -> Optional[BaseAgent]:
        """Return the agent to run this tick, or None to idle."""


class HeuristicBrain(Brain):
    """
    Mechanical brain: delegates the decision to the AgentScheduler
    (highest-priority agent whose cooldown elapsed and can_run(world) votes yes).

    This is the V5.1 default. A future LocalLLMBrain implements the same
    `decide(world)` interface but replaces the mechanical pick with LLM
    reasoning over goals — without touching the agents.
    """

    def __init__(self, scheduler: AgentScheduler):
        self._scheduler = scheduler

    def decide(self, world: Dict[str, Any]) -> Optional[BaseAgent]:
        return self._scheduler.pick(world)


# =============================================================================
# Offline demo — the Brain routes the right agent per world state (no emulator)
# =============================================================================

if __name__ == "__main__":
    from clashai.agents import (
        AgentScheduler, CombatAgent, GdCAgent, ClanCastleAgent,
    )

    class _FakeCC:
        def time_until_next_request(self):
            return 0.0
        def request_if_needed(self, s, t):
            pass

    print("HeuristicBrain offline demo\n")
    sched = AgentScheduler()
    combat = CombatAgent(models=None)
    gdc = GdCAgent(models=None)
    cc = ClanCastleAgent(manager=_FakeCC(),
                         screenshot_fn=lambda: None, tap_fn=lambda *a, **k: None)
    for a in (combat, gdc, cc):
        sched.register(a)
    brain = HeuristicBrain(sched)

    auto = {'mode': 'auto', 'on_village_home': True}

    # No war target → clan_castle (prio 20) over combat (prio 10)
    print(f"1. auto, village, no target -> {brain.decide(auto).name}")
    assert brain.decide(auto).name == 'clan_castle'

    # War target queued → gdc (prio 25) wins
    gdc.enqueue_target(5)
    print(f"2. auto + war target        -> {brain.decide(auto).name}")
    assert brain.decide(auto).name == 'gdc'

    # farm-only mode, CC not at village → only combat is eligible
    print(f"3. farm mode, not village   -> {brain.decide({'mode': 'farm', 'on_village_home': False}).name}")
    assert brain.decide({'mode': 'farm', 'on_village_home': False}).name == 'combat'

    print("\nOffline demo OK — HeuristicBrain.decide() routes via the scheduler")
