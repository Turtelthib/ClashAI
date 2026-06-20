# clashai/brain/loop.py
# BrainLoopMixin — main decision loop (V5.1: scheduler-driven via the Brain).

import time

from clashai.config import CHAT_CHECK_INTERVAL


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
