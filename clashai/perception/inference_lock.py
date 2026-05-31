# clashai/perception/inference_lock.py
# Global lock serialising all YOLO / CNN inference calls.
#
# Ultralytics YOLO models (and torch modules in general) are NOT
# thread-safe: two concurrent .predict() / forward() calls on the same
# model object corrupt each other's internal state, producing wrong or
# degraded predictions.
#
# In ClashAI the PerceptionThread runs inference in a background thread,
# while test_run_capture / debug_overlay can call the SAME detector
# objects from the main thread during a --test run. Without this lock
# those concurrent calls race and the captured detections are garbage
# (e.g. gel misread as poison, championne missed).
#
# Every detector wraps its model call with `with INFERENCE_LOCK:` so any
# caller from any thread is automatically serialised. Contention is
# negligible — GPU inference is fast and the only concurrent caller is
# the rare capture path.

import threading

# RLock so a single thread can re-enter (e.g. a detector that internally
# calls another locked detector) without deadlocking.
INFERENCE_LOCK = threading.RLock()
