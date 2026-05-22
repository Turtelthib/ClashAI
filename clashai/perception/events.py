# clashai/perception/events.py
# Publish-subscribe event bus for perception updates.
#
# Today every consumer reads PerceptionThread.get_latest() in a polling
# loop. That works but couples the consumer's tick rate to the env step
# loop. V5.3 needs the agent to react to *events* (new screen state, new
# building destroyed, hero ability now usable…) instead of polling.
#
# This bus is the foundation: PerceptionThread emits events after each
# inference cycle, and consumers subscribe to the ones they care about.
# `get_latest()` is preserved for synchronous snapshot use cases.
#
# Threading model: emit() runs subscribers synchronously on the caller's
# thread (= the inference thread). Subscribers MUST be fast — read state,
# enqueue work, return. Heavy work should hop to another thread.

import threading
from collections import defaultdict
from typing import Callable, Dict, List


# Canonical event names. Use these constants rather than raw strings so
# typos surface as AttributeError at the call site.
EVENT_PERCEPTION_UPDATED = 'perception_updated'  # any inference cycle finished
EVENT_SCREEN_STATE_CHANGED = 'screen_state_changed'  # CNN screen state changed
EVENT_BUILDINGS_DESTROYED = 'buildings_destroyed'    # one or more bldgs gone
EVENT_TROOP_BAR_CHANGED = 'troop_bar_changed'        # bar contents changed

ALL_EVENTS = (
    EVENT_PERCEPTION_UPDATED,
    EVENT_SCREEN_STATE_CHANGED,
    EVENT_BUILDINGS_DESTROYED,
    EVENT_TROOP_BAR_CHANGED,
)


EventCallback = Callable[[dict], None]


class PerceptionEventBus:
    """
    Tiny synchronous pub-sub for perception events.

    Usage:
        bus = PerceptionEventBus()

        def on_screen(data):
            print(f"screen now: {data['state']} ({data['conf']:.0%})")

        bus.subscribe(EVENT_SCREEN_STATE_CHANGED, on_screen)
        # ...
        bus.emit(EVENT_SCREEN_STATE_CHANGED, {'state': 'village_home', 'conf': 0.95})

    Subscribers fire on the emitter's thread. Exceptions raised by one
    subscriber don't propagate — they're caught and logged so a buggy
    consumer can't kill the perception loop.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._subscribers: Dict[str, List[EventCallback]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Subscription
    # ------------------------------------------------------------------

    def subscribe(self, event_type: str, callback: EventCallback) -> None:
        with self._lock:
            if callback not in self._subscribers[event_type]:
                self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: str, callback: EventCallback) -> bool:
        """True if the callback was registered and got removed."""
        with self._lock:
            try:
                self._subscribers[event_type].remove(callback)
                return True
            except ValueError:
                return False

    def clear(self, event_type: str = None) -> None:
        """Clear all subscribers, or just those of one event type."""
        with self._lock:
            if event_type is None:
                self._subscribers.clear()
            else:
                self._subscribers.pop(event_type, None)

    def num_subscribers(self, event_type: str = None) -> int:
        with self._lock:
            if event_type is None:
                return sum(len(v) for v in self._subscribers.values())
            return len(self._subscribers.get(event_type, ()))

    # ------------------------------------------------------------------
    # Emission
    # ------------------------------------------------------------------

    def emit(self, event_type: str, data: dict) -> None:
        """
        Synchronously notify every subscriber of `event_type`.

        A snapshot of the subscriber list is taken under lock, so a
        callback can safely unsubscribe itself during emit.
        """
        with self._lock:
            callbacks = list(self._subscribers.get(event_type, ()))
        for cb in callbacks:
            try:
                cb(data)
            except Exception as e:
                # Don't let a buggy subscriber crash the producer thread.
                cb_name = getattr(cb, '__name__', repr(cb))
                print(f"WARNING: PerceptionEventBus subscriber "
                      f"{cb_name} for '{event_type}' raised: {e}")


# -----------------------------------------------------------------------------
# Singleton helper
# -----------------------------------------------------------------------------
# Most code will reach the bus via PerceptionThread.events. This singleton
# exists for stand-alone consumers that need an event bus without a thread.

_singleton: PerceptionEventBus = None


def get_bus() -> PerceptionEventBus:
    """Process-wide PerceptionEventBus singleton."""
    global _singleton
    if _singleton is None:
        _singleton = PerceptionEventBus()
    return _singleton
