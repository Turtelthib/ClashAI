# clashai/agents/chat_agent.py
# ChatAgent — V5.1 agent that reads the clan chat, parses commands, dispatches
# them to other agents, and replies. This is the NATURAL-LANGUAGE INPUT CHANNEL:
# today it consumes the regex parser inside ClanChatMonitor; tomorrow the
# LocalLLMBrain plugs in here to interpret free-form chat (see
# [[project_llm_brain_vision]]).
#
# Highest priority so commands are seen before the bot wanders off to farm.
# Cooldown gates how often we interrupt to glance at the chat.

import time

from clashai.agents.base import BaseAgent, AgentResult


class ChatAgent(BaseAgent):
    """
    Glances at the clan chat, dispatches commands, replies.

    can_run: at the village, in a mode that listens to the clan (gdc / auto).
             Cooldown (default = CHAT_CHECK_INTERVAL) gates the frequency.
    run:     open chat → read commands → dispatch (attack → on_attack, etc.)
             + ack while the chat is open → close.
    """

    name = 'chat'
    priority = 30            # highest — check orders before doing anything else

    def __init__(self, monitor=None, models=None,
                 on_attack=None, on_stop=None,
                 screenshot_fn=None, classify_screen_fn=None,
                 modes=('gdc', 'auto'), cooldown_seconds=None,
                 verbose=True, **kwargs):
        super().__init__(**kwargs)
        # Cooldown: default to the configured chat-check interval (SSOT).
        if cooldown_seconds is None:
            try:
                from clashai.config import CHAT_CHECK_INTERVAL
                cooldown_seconds = float(CHAT_CHECK_INTERVAL)
            except Exception:
                cooldown_seconds = 30.0
        self.cooldown_seconds = cooldown_seconds

        self._monitor = monitor
        self._models = models
        self._on_attack = on_attack      # callable(target:int) — e.g. gdc.enqueue_target
        self._on_stop = on_stop          # callable() — optional
        self._screenshot_fn = screenshot_fn
        self._classify_screen_fn = classify_screen_fn
        self._modes = set(modes)
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Lazy dependency resolution (so the agent is cheap to construct/test)
    # ------------------------------------------------------------------
    def _get_monitor(self):
        if self._monitor is None:
            from clashai.social.clan_chat_monitor import ClanChatMonitor
            self._monitor = ClanChatMonitor(verbose=self.verbose)
        return self._monitor

    def _io(self):
        if self._screenshot_fn is None or self._classify_screen_fn is None:
            from clashai.navigation import game_loop as gl
            self._screenshot_fn = self._screenshot_fn or gl.adb_screenshot
            self._classify_screen_fn = self._classify_screen_fn or gl.classify_screen
        return self._screenshot_fn, self._classify_screen_fn

    # ------------------------------------------------------------------
    # BaseAgent API
    # ------------------------------------------------------------------
    def can_run(self, world):
        if world.get('mode', 'auto') not in self._modes:
            return False
        return world.get('on_village_home', False)

    def run(self):
        start = time.time()
        monitor = self._get_monitor()
        screenshot_fn, classify_screen_fn = self._io()

        if not monitor.open_chat(classify_screen_fn, self._models):
            return AgentResult(ok=False, duration_s=time.time() - start,
                               error='could not open chat')

        time.sleep(0.5)
        img = screenshot_fn()
        commands = monitor.check_once(img) if img is not None else []

        dispatched = []
        for cmd in commands:
            reply = self._dispatch(cmd, monitor)
            dispatched.append({'type': cmd.get('type'),
                               'target': cmd.get('target'),
                               'reply': reply})

        monitor.close_chat()

        return AgentResult(
            ok=True, duration_s=time.time() - start,
            data={'commands': dispatched, 'count': len(dispatched)},
        )

    def _dispatch(self, cmd, monitor):
        """Act on one parsed command. Returns the ack text sent (or None)."""
        ctype = cmd.get('type')

        if ctype == 'attack':
            target = cmd.get('target')
            if self._on_attack and target is not None:
                self._on_attack(target)
            ack = f"IA - jattaque le {target}"
            self._safe_send(monitor, ack)
            return ack

        if ctype == 'stop':
            if self._on_stop:
                self._on_stop()
            return None

        if ctype == 'status':
            ack = "IA - ok je suis la"
            self._safe_send(monitor, ack)
            return ack

        return None

    def _safe_send(self, monitor, message):
        try:
            monitor.send_chat_message(message)
        except Exception as e:
            if self.verbose:
                print(f" WARNING: chat send failed: {e}")


# =============================================================================
# Offline demo — full multi-agent flow: chat command -> enqueue -> GdC picked
# =============================================================================

if __name__ == "__main__":
    from clashai.agents.scheduler import AgentScheduler
    from clashai.agents.combat_agent import CombatAgent
    from clashai.agents.gdc_agent import GdCAgent

    class _FakeMonitor:
        """Stand-in for ClanChatMonitor — returns one attack command, no ADB."""
        def __init__(self, cmds):
            self._cmds = cmds
        def open_chat(self, classify_fn, models):
            return True
        def check_once(self, img):
            return self._cmds
        def close_chat(self):
            pass
        def send_chat_message(self, msg):
            print(f"   chat reply: {msg!r}")

    print("ChatAgent offline demo (chat -> enqueue -> GdC picked)\n")
    sched = AgentScheduler()
    gdc = GdCAgent(models=None)
    combat = CombatAgent(models=None)
    chat = ChatAgent(
        monitor=_FakeMonitor([{'type': 'attack', 'target': 5}]),
        on_attack=gdc.enqueue_target,
        screenshot_fn=lambda: object(),       # non-None so check_once runs
        classify_screen_fn=lambda *a, **k: ('village_home', 1.0),
        cooldown_seconds=0.0,
    )
    sched.register(chat)
    sched.register(gdc)
    sched.register(combat)

    auto = {'mode': 'auto', 'on_village_home': True}

    # 1. ChatAgent (prio 30) is picked first
    picked = sched.pick(auto)
    print(f"1. first pick           -> {picked.name}")
    assert picked.name == 'chat'

    # 2. Run it: reads "attack 5", enqueues to GdC, acks in chat
    result = sched.run(picked)
    print(f"2. chat run             -> ok={result.ok} commands={result.data['commands']}")
    print(f"   gdc pending          -> {gdc.pending()}")
    assert gdc.pending() == [5]

    # 3. Next pick: GdC (prio 25) now has a target → preempts combat
    #    (chat is on its cooldown here only if >0; we set 0, so force it)
    chat._last_run_at = time.time()
    chat.cooldown_seconds = 999
    picked = sched.pick(auto)
    print(f"3. next pick            -> {picked.name}  (target {gdc.pending()})")
    assert picked.name == 'gdc'

    print("\nstatus:", [(a['name'], a['priority']) for a in sched.status()])
    print("\nOffline demo OK — chat command -> GdC enqueue -> scheduler routes to war")
