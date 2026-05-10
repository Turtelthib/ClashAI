# clashai/perception/perception_thread.py
# Async perception pipeline — runs YOLO + CNNs in background threads.
#
# Architecture (exploits the 20-core i7 Ultra 25Hx):
#
#   Thread 1 — FrameCapture  : mss.grab() at ~30fps → frame queue
#   Thread 2 — InferenceWorker: YOLO buildings + YOLO troops + CNN screen
#                               (GPU, releases Python GIL → true parallel)
#
# The main agent thread never blocks on perception:
#   - env.step(action) executes ADB tap immediately (~70ms)
#   - env._update_combat_observation() reads cached state (instant)
#   - DELAY_OBSERVE drops from 2.0s → 0.1s
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

    def __init__(self, models, verbose=False):
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
    # Thread 1: Frame capture
    # ------------------------------------------------------------------

    def _capture_loop(self):
        """Captures frames at ~CAPTURE_FPS and pushes to the inference queue."""
        from clashai.perception.screen_capture import get_capture
        cap = get_capture()
        interval = 1.0 / self.CAPTURE_FPS

        while self._running:
            t0 = time.time()

            try:
                frame = cap.grab()
                if frame is not None:
                    # Drop stale frame if inference is slower than capture
                    try:
                        self._frame_q.put_nowait(frame)
                    except queue.Full:
                        try:
                            self._frame_q.get_nowait()  # discard old
                        except queue.Empty:
                            pass
                        self._frame_q.put_nowait(frame)
            except Exception as e:
                if self.verbose:
                    print(f"WARNING: PerceptionCapture error: {e}")

            # Rate-limit to target fps
            elapsed = time.time() - t0
            sleep = max(0.0, interval - elapsed)
            if sleep > 0:
                time.sleep(sleep)

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
                    import cv2, numpy as np
                    img_cv = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
                    scale_x = 1920 / frame.width
                    scale_y = 1080 / frame.height
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
            with self._lock:
                self._state = {
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

            if self.verbose:
                print(f" Perception: {len(buildings)} buildings | "
                      f"screen={screen_state} ({screen_conf:.0%}) | "
                      f"{elapsed_ms:.0f}ms")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _backend_name(self):
        try:
            from clashai.perception.screen_capture import get_capture
            return get_capture().backend
        except Exception:
            return 'adb'
