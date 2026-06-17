# clashai/brain/loop.py
# BrainLoopMixin — main decision loop (V5.1: scheduler-driven via the Brain).

import time

# ATTACKS_BEFORE_CHAT_CHECK is only used by the [DEAD-CODE-V5.1] _should_check_chat
# below — kept imported so the dead method stays valid until Étape B removes it.
from clashai.config import CHAT_CHECK_INTERVAL, ATTACKS_BEFORE_CHAT_CHECK


class BrainLoopMixin:
    """The heart of the Brain: each tick, ask the Brain which agent to run."""

    def _main_loop(self, max_episodes=None):
        """
        V5.1 main loop — scheduler-driven.

        Cycle:
          1. Ensure we're at the village (recovery)
          2. Build the world snapshot (perception + mode)
          3. Brain.decide(world) → which agent runs (priority + cooldown + can_run)
          4. Run it, update stats
          5. Human pause / idle
        """
        from clashai.agents import build_world

        while self._running:
            # --- Episode limit (counts farm attacks, like before) ---
            if max_episodes and self._attacks_done >= max_episodes:
                print(f"\n{max_episodes} attacks completed")
                break

            # --- 1. Return to village (recovery; also gives us the screen) ---
            if not self._ensure_at_village():
                print(" WARNING: Unable to return to village, retry...")
                time.sleep(5)
                continue

            # --- 2. World snapshot (perception cache + mode + village flag) ---
            world = build_world(
                self._models,
                mode=self.mode,
                on_village_home=True,  # we just ensured it
            )

            # --- 3. Decide via the (swappable) Brain ---
            agent = self._brain.decide(world)

            if agent is None:
                # Nothing ready. In gdc-only mode, wait for commands; else pause.
                if self.mode == 'gdc':
                    if self.verbose:
                        print(f"  Waiting for CW commands... "
                              f"(next check in {CHAT_CHECK_INTERVAL}s)")
                    time.sleep(CHAT_CHECK_INTERVAL)
                else:
                    self._human_pause()
                continue

            # --- 4. Run the chosen agent + update stats ---
            result = self._scheduler.run(agent)
            self._update_stats(agent, result)

            # --- 5. Human pause ---
            if self._running:
                self._human_pause()

    def _update_stats(self, agent, result):
        """Aggregate brain-level stats from an agent's AgentResult."""
        if not result.ok:
            if self.verbose and result.error:
                print(f" {agent.name} failed: {result.error}")
            return
        data = result.data or {}

        if agent.name == 'combat':
            self._attacks_done += 1
            self._total_stars += data.get('stars') or 0
            self._total_destruction += data.get('percentage') or 0
            self._attacks_since_chat_check += 1
            if self.verbose:
                avg = self._total_destruction / max(self._attacks_done, 1)
                print(f"\n Farm #{self._attacks_done}: "
                      f"{data.get('stars')}* {data.get('percentage')}% | "
                      f"Average: {avg:.1f}%")

        elif agent.name == 'gdc':
            self._gdc_attacks_done += 1
            if self.verbose:
                print(f"\n GdC #{data.get('target')}: "
                      f"{data.get('stars')}* {data.get('percentage')}%")

    # =====================================================================
    # [DEAD-CODE-V5.1] superseded by the scheduler + agents — remove in Étape B
    # (grep "DEAD-CODE-V5.1" to find every method to delete)
    # =====================================================================

    def _execute_task(self, task):  # [DEAD-CODE-V5.1] → CombatAgent / GdCAgent
        """Executes a task (farm attack or CW attack)."""
        task_type = task['type']

        if task_type == 'farm_attack':
            self._do_farm_attack()

        elif task_type == 'gdc_attack':
            target = task['target']
            original_cmd = task.get('original_cmd')

            # Acknowledgement BEFORE the attack
            if self._chat_monitor:
                self._send_chat_ack(target, before=True)

            # Attack
            info = self._do_gdc_attack(target)

            # Mark as executed
            if original_cmd and self._chat_monitor:
                self._chat_monitor.mark_executed(original_cmd)

            # Acknowledgement AFTER the attack (with result)
            if self._chat_monitor and info:
                self._send_chat_ack(target, before=False, result=info)

    def _should_check_chat(self):  # [DEAD-CODE-V5.1] → ChatAgent.can_run cooldown
        """Determines whether the chat should be checked now."""
        if self.mode == 'farm':
            return False

        if self._chat_monitor is None:
            return False

        # Check after N attacks or after a time interval
        now = time.time()
        time_since_check = now - self._last_chat_check

        if self._attacks_since_chat_check >= ATTACKS_BEFORE_CHAT_CHECK:
            return True
        if time_since_check >= CHAT_CHECK_INTERVAL:
            return True

        return False
