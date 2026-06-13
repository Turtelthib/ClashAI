# clashai/agents/gdc_agent.py
# GdCAgent — V5.1 agent for Clan War (GdC) attacks on a specific target.
#
# Unlike CombatAgent (farm, no target), a war attack needs a target number.
# Targets come from clan-chat commands ("@bot attaque le 5"): the future
# ChatAgent parses them and calls enqueue_target(n); GdCAgent consumes the
# queue. Higher priority than CombatAgent so a queued war order preempts farm.
#
# DRY: navigation = GdCNavigator.attack_target(); deploy/combat = the shared
# run_attack_episode() (same SSOT as CombatAgent and the brain).

import time
from collections import deque

from clashai.agents.base import BaseAgent, AgentResult


class GdCAgent(BaseAgent):
    """
    Executes a Clan War attack on a queued target.

    can_run: mode allows war (gdc / auto) AND at least one target is queued.
    run:     navigate to the target, attack, run the episode, return home.
    """

    name = 'gdc'
    priority = 25            # war orders preempt farm (combat=10) and CC (20)
    cooldown_seconds = 0.0

    def __init__(self, models=None, navigator=None, agent=None, use_heuristic=True,
                 modes=('gdc', 'auto'), verbose=True, **kwargs):
        super().__init__(**kwargs)
        self._models = models
        self._navigator = navigator
        self._agent = agent
        self._use_heuristic = use_heuristic
        self._modes = set(modes)
        self.verbose = verbose
        self._targets = deque()

    # ------------------------------------------------------------------
    # Target queue (filled by the chat command path)
    # ------------------------------------------------------------------
    def enqueue_target(self, n):
        """Queue a war target (#1..#50). Ignores invalids and duplicates."""
        if isinstance(n, int) and 1 <= n <= 50 and n not in self._targets:
            self._targets.append(n)
            return True
        return False

    def pending(self):
        return list(self._targets)

    def _nav(self):
        if self._navigator is None:
            from clashai.navigation.gdc_navigator import GdCNavigator
            self._navigator = GdCNavigator(self._models, verbose=self.verbose)
        return self._navigator

    # ------------------------------------------------------------------
    # BaseAgent API
    # ------------------------------------------------------------------
    def can_run(self, world):
        if world.get('mode', 'auto') not in self._modes:
            return False
        return len(self._targets) > 0

    def run(self):
        from clashai.combat.episode_runner import run_attack_episode
        start = time.time()
        target = self._targets.popleft()
        nav = self._nav()

        # 1. Navigate → select target → launch (up to phase_attaque)
        if not nav.attack_target(target):
            nav.return_to_village()
            return AgentResult(
                ok=False, duration_s=time.time() - start,
                data={'target': target}, error='navigation/select failed',
            )

        # 2. Deploy/combat via the shared runner
        info = run_attack_episode(
            self._models, agent=self._agent,
            use_heuristic=self._use_heuristic, verbose=self.verbose,
        )

        # 3. Always return home afterwards
        nav.return_to_village()

        return AgentResult(
            ok=info is not None, duration_s=time.time() - start,
            data={
                'target': target,
                'stars': info.get('stars') if info else None,
                'percentage': info.get('percentage') if info else None,
            },
            error=None if info else 'attack episode failed',
        )


# =============================================================================
# Offline demo — target queue + priority over farm + mode gating
# =============================================================================

if __name__ == "__main__":
    from clashai.agents.scheduler import AgentScheduler
    from clashai.agents.combat_agent import CombatAgent

    print("GdCAgent offline demo (queue + priority + mode gating)\n")
    sched = AgentScheduler()
    gdc = GdCAgent(models=None)
    combat = CombatAgent(models=None)
    sched.register(gdc)
    sched.register(combat)

    auto = {'mode': 'auto', 'on_village_home': True}

    # 1. No war target queued → combat is the default
    print(f"1. no target            -> {sched.pick(auto).name}")
    assert sched.pick(auto).name == 'combat'

    # 2. Queue a war target → GdC (prio 25) preempts combat (prio 10)
    assert gdc.enqueue_target(5)
    assert not gdc.enqueue_target(5)    # duplicate ignored
    assert not gdc.enqueue_target(99)   # out of range ignored
    print(f"2. target #5 queued     -> {sched.pick(auto).name}  pending={gdc.pending()}")
    assert sched.pick(auto).name == 'gdc'

    # 3. farm-only mode → GdC gated off even with a target → combat
    print(f"3. farm mode + target   -> {sched.pick({'mode': 'farm', 'on_village_home': True}).name}")
    assert sched.pick({'mode': 'farm', 'on_village_home': True}).name == 'combat'

    print("\nstatus:", [(a['name'], a['priority']) for a in sched.status()])
    print("\nOffline demo OK — war target queue + priority + mode gating validated")
