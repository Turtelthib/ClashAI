# clashai/perception/perception_thread.py
# Async perception pipeline — runs YOLO + CNNs in background threads.
#
# Architecture V5.3 (push-based, was polling):
#
#   Backend push → FrameCallback → frame queue → InferenceWorker
#
#   On WGC: WGC's Rust thread fires `on_frame_arrived` at the emulator's
#           native render rate (~30-60fps). ScreenCapture relays the
#           frame to every subscriber. PerceptionThread is one such
#           subscriber.
#   On other backends: ScreenCapture spins a fallback polling thread at
#           ~30fps to emulate the push API. Same consumer code.
#
# The main agent thread never blocks on perception:
#   - env.step(action) executes ADB tap immediately (~70ms)
#   - env._update_combat_observation() reads cached state (instant)
#   - DELAY_OBSERVE drops from 2.0s → 0.1s
#
# In addition to the get_latest() pull API, this thread emits perception
# events via clashai.perception.events (PerceptionEventBus) so consumers
# can react to changes without polling.
#
# Usage:
#   pt = PerceptionThread(models)
#   pt.start()
#   state = pt.get_latest()   # {'frame', 'buildings', 'combat_features', ...}
#   pt.pause()   # during navigation (no need to run YOLO)
#   pt.resume()  # when attack starts
#   pt.stop()    # clean shutdown

import threading
import queue
import time
import traceback


class PerceptionThread:
    """
    Background perception pipeline.

    Two daemon threads:
      - FrameCapture: grabs frames from the emulator window at ~30fps.
      - InferenceWorker: runs YOLO buildings, YOLO troops, screen CNN.

    The main thread calls get_latest() to read the most recent result
    without blocking. The cached state is updated ~4-7fps (GPU inference).
    """

    # Target capture rate (fps)
    CAPTURE_FPS = 20
    # Minimum interval between inference runs (don't hammer GPU unnecessarily)
    INFERENCE_MIN_INTERVAL = 0.05  # 50ms → max 20fps inference

    def __init__(self, models, verbose=False, event_bus=None):
        self.models = models
        self.verbose = verbose

        # Frame queue: FrameCapture → InferenceWorker
        # maxsize=1: InferenceWorker always processes the LATEST frame
        self._frame_q = queue.Queue(maxsize=1)

        # Shared state: InferenceWorker → main thread
        self._lock = threading.Lock()
        self._state = {
            'frame': None,
            'buildings': [],
            'combat_features': None,
            'screen_state': None,
            'screen_conf': 0.0,
            'troop_bar': [],
            'troop_positions': {},
            'timestamp': -999.0,  # sentinel: never fresh until first inference
            'inference_ms': 0.0,
        }

        # Phase C.3: event bus so consumers can subscribe to push updates
        # instead of polling get_latest(). If no bus is passed, we use the
        # process-wide singleton — that lets multiple PerceptionThread
        # instances share subscribers, and lets consumers wire up before
        # the thread starts.
        if event_bus is None:
            from clashai.perception.events import get_bus
            event_bus = get_bus()
        self.events = event_bus

        # Previous screen state — used to detect transitions so we only
        # emit EVENT_SCREEN_STATE_CHANGED on a real change, not every tick.
        self._prev_screen_state = None
        self._prev_building_count = None

        # Control
        self._running = False
        self._pause_event = threading.Event()
        self._pause_event.set()  # start unpaused

        self._capture_thread = None
        self._inference_thread = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """Start both background threads."""
        self._running = True

        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name='PerceptionCapture',
            daemon=True,
        )
        self._inference_thread = threading.Thread(
            target=self._inference_loop,
            name='PerceptionInference',
            daemon=True,
        )

        self._capture_thread.start()
        self._inference_thread.start()
        print(f" PerceptionThread started (capture + inference on {self._backend_name()})")

    def stop(self):
        """Graceful shutdown — waits up to 2s per thread."""
        self._running = False
        self._pause_event.set()  # unblock if paused
        # Unblock frame queue reader
        try:
            self._frame_q.put_nowait(None)
        except queue.Full:
            pass
        if self._capture_thread:
            self._capture_thread.join(timeout=2.0)
        if self._inference_thread:
            self._inference_thread.join(timeout=2.0)

    def pause(self):
        """
        Pause inference (e.g. during navigation).
        Capture keeps running at low rate so the frame is fresh when resumed.
        """
        self._pause_event.clear()

    def resume(self):
        """Resume inference after pause."""
        self._pause_event.set()

    def get_latest(self):
        """
        Returns the latest perception state. Non-blocking.

        Returns a dict with:
          frame          : PIL.Image or None
          buildings      : list of {class, confidence, bbox, center}
          combat_features: np.array (15,) or None
          screen_state   : str or None
          screen_conf    : float
          timestamp      : float (time.time() of last inference)
          inference_ms   : float (last inference duration)
        """
        with self._lock:
            return dict(self._state)

    def is_fresh(self, max_age_s=0.5):
        """True if the last inference was less than max_age_s seconds ago."""
        with self._lock:
            return (time.time() - self._state['timestamp']) < max_age_s

    @property
    def running(self):
        return self._running

    # ------------------------------------------------------------------
    # Thread 1: Frame capture (V5.3 — push-based, was polling)
    # ------------------------------------------------------------------

    def _capture_loop(self):
        """
        V5.3 — subscribes to ScreenCapture's frame stream instead of
        polling.

        On WGC backend, frames arrive natively from the Rust capture
        thread at the emulator's render rate (~30-60fps).
        On other backends, ScreenCapture spins a fallback polling thread
        at ~30fps and pushes to subscribers — same API for the consumer.

        This thread now just registers the callback, waits for stop, and
        unsubscribes on shutdown. The actual frame handling happens
        inside `_on_new_frame` on whichever thread the capture backend
        uses.
        """
        from clashai.perception.screen_capture import get_capture
        cap = get_capture()
        cap.subscribe_to_frames(self._on_new_frame)

        if self.verbose:
            print(f"PerceptionCapture: subscribed to {cap.backend} push stream")

        # Block until stop. The capture work happens on the backend's
        # own thread via _on_new_frame.
        while self._running:
            time.sleep(0.1)

        cap.unsubscribe_from_frames(self._on_new_frame)

    def _on_new_frame(self, frame):
        """Frame callback fired by ScreenCapture on each new frame.

        Pushes the frame to the inference queue, dropping the previous
        one if the inference worker is lagging behind. Must be fast.
        """
        if not self._running:
            return
        try:
            self._frame_q.put_nowait(frame)
        except queue.Full:
            # Inference is slower than capture — discard the stale frame
            # and replace with the fresh one.
            try:
                self._frame_q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frame_q.put_nowait(frame)
            except queue.Full:
                pass

    # ------------------------------------------------------------------
    # Thread 2: Inference
    # ------------------------------------------------------------------

    def _inference_loop(self):
        """Reads latest frames and runs all YOLO + CNN inferences."""
        from clashai.navigation.game_loop import analyze_village, classify_screen

        last_inference = 0.0

        while self._running:
            # Wait if paused (e.g. during navigation)
            self._pause_event.wait(timeout=1.0)
            if not self._running:
                break

            # Throttle: enforce minimum inference interval
            now = time.time()
            wait = self.INFERENCE_MIN_INTERVAL - (now - last_inference)
            if wait > 0:
                time.sleep(wait)

            # Get latest frame (block up to 0.5s)
            try:
                frame = self._frame_q.get(timeout=0.5)
            except queue.Empty:
                continue

            if frame is None:
                continue

            t0 = time.time()
            buildings = []
            combat_features = None
            screen_state = None
            screen_conf = 0.0
            troop_bar = []       # raw detections from TroopBarDetector
            troop_positions = {} # active positions {name: (x, y, conf)}

            #  1. YOLO buildings + building CNN 
            try:
                buildings = analyze_village(frame, self.models)
            except Exception as e:
                if self.verbose:
                    traceback.print_exc()

            #  2. YOLO troops → combat features
            try:
                troop_detector = self.models.get('troop_detector')
                combat_obs = self.models.get('combat_observer')
                if troop_detector is not None and combat_obs is not None:
                    # combat_obs handles PIL→cv2 conversion + scaling internally.
                    combat_features, _ = combat_obs.observe(
                        frame,
                        buildings_count=len(buildings) if buildings else None,
                    )
            except Exception as e:
                if self.verbose:
                    traceback.print_exc()

            #  3. Troop bar detector (YOLO + HSV grayed filter)
            try:
                bar_detector = self.models.get('troop_bar_detector')
                if bar_detector is not None:
                    troop_bar = bar_detector.detect(frame)
                    troop_positions = bar_detector.to_positions(troop_bar)
            except Exception:
                if self.verbose:
                    traceback.print_exc()

            #  4. Screen classifier
            try:
                screen_state, screen_conf = classify_screen(frame, self.models)
            except Exception as e:
                if self.verbose:
                    traceback.print_exc()

            elapsed_ms = (time.time() - t0) * 1000
            last_inference = time.time()

            #  Update shared state
            new_state = {
                'frame': frame,
                'buildings': buildings,
                'combat_features': combat_features,
                'screen_state': screen_state,
                'screen_conf': screen_conf,
                'troop_bar': troop_bar,         # raw detections with is_grayed
                'troop_positions': troop_positions,  # {name: (x,y,conf)} active only
                'timestamp': last_inference,
                'inference_ms': elapsed_ms,
            }
            with self._lock:
                self._state = new_state

            if self.verbose:
                print(f" Perception: {len(buildings)} buildings | "
                      f"screen={screen_state} ({screen_conf:.0%}) | "
                      f"{elapsed_ms:.0f}ms")

            # Phase C.3: emit events for push-based consumers.
            # Subscribers run synchronously on this inference thread —
            # keep callbacks fast.
            self._emit_events(new_state)

    def _emit_events(self, state):
        """Fire push events derived from the latest inference cycle."""
        from clashai.perception.events import (
            EVENT_PERCEPTION_UPDATED,
            EVENT_SCREEN_STATE_CHANGED,
            EVENT_BUILDINGS_DESTROYED,
            EVENT_TROOP_BAR_CHANGED,
        )

        # Always: full state available
        self.events.emit(EVENT_PERCEPTION_UPDATED, state)

        # Screen state transitions
        ss = state['screen_state']
        if ss is not None and ss != self._prev_screen_state:
            self.events.emit(EVENT_SCREEN_STATE_CHANGED, {
                'state': ss,
                'conf': state['screen_conf'],
                'previous': self._prev_screen_state,
            })
            self._prev_screen_state = ss

        # Building destruction (compared to previous cycle)
        curr_buildings = state['buildings'] or []
        curr_count = len(curr_buildings)
        if self._prev_building_count is not None:
            destroyed = self._prev_building_count - curr_count
            if destroyed > 0:
                self.events.emit(EVENT_BUILDINGS_DESTROYED, {
                    'destroyed': destroyed,
                    'remaining': curr_count,
                })
        self._prev_building_count = curr_count

        # Troop bar updated (we don't diff content, just signal a refresh).
        # Consumers can filter on actual changes if needed.
        if state['troop_bar']:
            self.events.emit(EVENT_TROOP_BAR_CHANGED, {
                'detections': state['troop_bar'],
                'positions': state['troop_positions'],
            })

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _backend_name(self):
        try:
            from clashai.perception.screen_capture import get_capture
            return get_capture().backend
        except Exception:
            return 'adb'
