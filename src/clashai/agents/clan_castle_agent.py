# clashai/agents/clan_castle_agent.py
# ClanCastleAgent — V5.1 pilot agent wrapping the existing ClanCastleManager.
#
# This is the first concrete BaseAgent: it proves the
# world -> can_run -> run -> scheduler pattern end to end without rewriting any
# of the working clan-castle logic. The manager already owns its own request
# cooldown (time_until_next_request()), so the agent delegates cooldown to it
# rather than duplicating BaseAgent.cooldown_seconds.

import time

from clashai.agents.base import BaseAgent, AgentResult
from clashai.config import REQUEST_COOLDOWN


class ClanCastleAgent(BaseAgent):
    """
    Requests clan castle troops when idle at the village.

    can_run: we're on village_home AND the manager's request cooldown elapsed.
    run:     delegate to ClanCastleManager.request_if_needed().
    """

    name = 'clan_castle'
    priority = 20            # low — farm/war combat preempts this
    # Scheduler-level cooldown = the request interval. CRITICAL: the manager
    # only advances ITS cooldown on a SUCCESSFUL request; if a run no-ops
    # (template missing, CC full, failure), the manager stays "ready" forever
    # → this prio-20 agent would starve CombatAgent (prio 10) every tick.
    # The scheduler cooldown fires after EVERY run (success or not), so a
    # failed/no-op CC run still yields the floor to combat for 15 min.
    cooldown_seconds = REQUEST_COOLDOWN

    def __init__(self, manager=None, models=None,
                 screenshot_fn=None, tap_fn=None, verbose=True, **kwargs):
        super().__init__(**kwargs)
        if manager is None:
            from clashai.social.clan_castle import ClanCastleManager
            manager = ClanCastleManager(models=models, verbose=verbose)
        self._cc = manager
        self._screenshot_fn = screenshot_fn
        self._tap_fn = tap_fn

    def _io(self):
        """Lazily resolve the canonical ADB I/O fns (WGC-routed screenshot)."""
        if self._screenshot_fn is None or self._tap_fn is None:
            from clashai.navigation import game_loop as gl
            self._screenshot_fn = self._screenshot_fn or gl.adb_screenshot
            self._tap_fn = self._tap_fn or gl.adb_tap
        return self._screenshot_fn, self._tap_fn

    def can_run(self, world):
        if not world.get('on_village_home', False):
            return False
        try:
            return self._cc.time_until_next_request() <= 0
        except Exception:
            # Cooldown unknown → allow; request_if_needed re-checks internally.
            return True

    def run(self):
        start = time.time()
        screenshot_fn, tap_fn = self._io()
        self._cc.request_if_needed(screenshot_fn, tap_fn)
        try:
            next_in = round(self._cc.time_until_next_request(), 0)
        except Exception:
            next_in = None
        return AgentResult(
            ok=True,
            duration_s=time.time() - start,
            data={'next_request_in_s': next_in},
        )


# =============================================================================
# Offline demo / smoke test (no emulator, no models needed)
# =============================================================================

if __name__ == "__main__":
    from clashai.agents.scheduler import AgentScheduler

    class _FakeCC:
        """Stand-in for ClanCastleManager — no ADB, no YOLO."""
        def __init__(self):
            self._next = 0.0
        def time_until_next_request(self):
            return self._next
        def request_if_needed(self, screenshot_fn, tap_fn):
            print("   -> ClanCastleManager.request_if_needed() called")
            self._next = 900.0  # simulate 15-min cooldown after a request

    print("ClanCastleAgent offline demo\n")
    sched = AgentScheduler()
    agent = ClanCastleAgent(manager=_FakeCC(),
                            screenshot_fn=lambda: None, tap_fn=lambda *a, **k: None)
    sched.register(agent)

    # 1. Not on village_home → not picked
    picked = sched.pick({'on_village_home': False})
    print(f"1. not village        -> picked={picked}")
    assert picked is None

    # 2. On village_home + cooldown ready → picked
    picked = sched.pick({'on_village_home': True})
    print(f"2. village + ready    -> picked={picked.name if picked else None}")
    assert picked is agent

    # 3. Run it (fires request, sets cooldown)
    result = sched.run(picked)
    print(f"3. run                -> ok={result.ok} data={result.data}")
    assert result.ok

    # 4. Now on cooldown → not picked again
    picked = sched.pick({'on_village_home': True})
    print(f"4. village + cooldown -> picked={picked}")
    assert picked is None

    print("\nstatus:", sched.status())
    print("\nOffline demo OK — world -> can_run -> pick -> run pattern validated")
