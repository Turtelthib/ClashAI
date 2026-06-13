# clashai/agents/combat_agent.py
# CombatAgent — V5.1 agent wrapping one farm/multiplayer attack episode.
#
# This is the "default activity": when no higher-priority agent (chat command,
# clan-castle request, war order) is ready, the scheduler runs a farm attack.
# The deploy/combat loop itself lives in clashai.combat.episode_runner (shared
# SSOT with the brain) — this agent is just the BaseAgent wrapper around it.

import time

from clashai.agents.base import BaseAgent, AgentResult


class CombatAgent(BaseAgent):
    """
    Runs one farm attack episode.

    can_run: the brain is in a mode that allows farming (farm / auto).
    run:     delegate to run_attack_episode() (env.reset handles navigation).
    """

    name = 'combat'
    priority = 10            # low — the idle/default activity; others preempt
    cooldown_seconds = 0.0

    def __init__(self, models=None, agent=None, use_heuristic=True,
                 modes=('farm', 'auto'), verbose=True, **kwargs):
        super().__init__(**kwargs)
        self._models = models
        self._agent = agent
        self._use_heuristic = use_heuristic
        self._modes = set(modes)
        self.verbose = verbose

    def can_run(self, world):
        # Farm only happens in farm/auto mode. (gdc mode waits for war orders.)
        return world.get('mode', 'auto') in self._modes

    def run(self):
        from clashai.combat.episode_runner import run_attack_episode
        start = time.time()
        info = run_attack_episode(
            self._models,
            agent=self._agent,
            use_heuristic=self._use_heuristic,
            verbose=self.verbose,
        )
        if info is None:
            return AgentResult(
                ok=False, duration_s=time.time() - start,
                error='attack episode failed',
            )
        return AgentResult(
            ok=True, duration_s=time.time() - start,
            data={
                'stars': info.get('stars'),
                'percentage': info.get('percentage'),
            },
        )


# =============================================================================
# Offline demo — priority preemption + mode gating (no emulator)
# =============================================================================

if __name__ == "__main__":
    from clashai.agents.scheduler import AgentScheduler
    from clashai.agents.clan_castle_agent import ClanCastleAgent

    class _FakeCC:
        def __init__(self):
            self._n = 0.0
        def time_until_next_request(self):
            return self._n
        def request_if_needed(self, s, t):
            self._n = 900.0

    print("CombatAgent offline demo (priority + mode gating)\n")
    sched = AgentScheduler()
    cc = ClanCastleAgent(manager=_FakeCC(),
                         screenshot_fn=lambda: None, tap_fn=lambda *a, **k: None)
    combat = CombatAgent(models=None, use_heuristic=True)  # run() not called here
    sched.register(cc)
    sched.register(combat)

    farm_world = {'on_village_home': True, 'mode': 'farm'}

    # 1. CC ready (prio 20) preempts combat (prio 10)
    print(f"1. village + CC ready   -> {sched.pick(farm_world).name}")
    assert sched.pick(farm_world).name == 'clan_castle'

    # 2. CC on cooldown -> combat is the default activity
    cc._cc._n = 900.0
    print(f"2. CC on cooldown       -> {sched.pick(farm_world).name}")
    assert sched.pick(farm_world).name == 'combat'

    # 3. gdc mode -> combat gated off, CC still on cooldown -> nothing to do
    print(f"3. gdc mode             -> {sched.pick({'on_village_home': True, 'mode': 'gdc'})}")
    assert sched.pick({'on_village_home': True, 'mode': 'gdc'}) is None

    print("\nstatus:", [a['name'] for a in sched.status()])
    print("\nOffline demo OK — priority preemption + mode gating validated")
