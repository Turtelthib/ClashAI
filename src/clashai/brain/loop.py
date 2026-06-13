# clashai/brain/loop.py
# BrainLoopMixin — main decision loop + task dispatch + chat-check timing.

import time

from clashai.config import (
    CHAT_CHECK_INTERVAL, ATTACKS_BEFORE_CHAT_CHECK,
    PRIORITY_GDC_COMMAND, PRIORITY_FARM_ATTACK,
)


class BrainLoopMixin:
    """The heart of the Brain: decides what to do at every moment."""

    def _main_loop(self, max_episodes=None):
        """
        The heart of the Brain. Decides what to do at every moment.

        Cycle:
          1. Make sure we are at the village
          2. Check chat commands (if it is time)
          3. Decide the next action
          4. Execute
          5. Human pause
          6. Repeat
        """
        while self._running:
            # --- Episode limit ---
            if max_episodes and self._attacks_done >= max_episodes:
                print(f"\n{max_episodes} attacks completed")
                break

            # --- 1. Return to village ---
            if not self._ensure_at_village():
                print(" WARNING: Unable to return to village, retry...")
                time.sleep(5)
                continue

            # --- 2. Check clan chat ---
            if self._should_check_chat():
                commands = self._check_clan_chat()

                # Process commands by priority
                for cmd in commands:
                    if cmd['type'] == 'attack':
                        self._task_queue.append({
                            'type': 'gdc_attack',
                            'target': cmd['target'],
                            'priority': PRIORITY_GDC_COMMAND,
                            'original_cmd': cmd,
                        })
                    elif cmd['type'] == 'stop':
                        print(" Stop command received")
                        if self._chat_monitor:
                            self._chat_monitor.mark_executed(cmd)
                        self._running = False
                        return

                # Sort by priority
                self._task_queue.sort(key=lambda t: t['priority'], reverse=True)

            # --- 3. Decide the next action ---
            if self._task_queue:
                # Execute the most urgent task
                task = self._task_queue.pop(0)
                self._execute_task(task)
            elif self.mode in ('farm', 'auto'):
                # No urgent task → farm
                self._execute_task({
                    'type': 'farm_attack',
                    'priority': PRIORITY_FARM_ATTACK,
                })
            elif self.mode == 'gdc':
                # CW mode: wait for commands
                if self.verbose:
                    print(f"  Waiting for CW commands... "
                          f"(next check in {CHAT_CHECK_INTERVAL}s)")
                time.sleep(CHAT_CHECK_INTERVAL)
                continue

            # --- 4. Human pause ---
            if self._running:
                self._human_pause()

    def _execute_task(self, task):
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

    def _should_check_chat(self):
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
