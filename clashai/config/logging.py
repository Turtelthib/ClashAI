# clashai/config/logging.py
# Centralised logging configuration.
#
# Two outputs:
#   1. Console — short format, level-coloured-ish via prefix
#   2. File   — logs/clashai.log, rotated at ~5 MB, 3 backups
#
# Level configurable via env var CLASHAI_LOG_LEVEL (DEBUG/INFO/WARNING/ERROR).
# Defaults to INFO.
#
# Usage in modules:
#     from clashai.config.logging import get_logger
#     log = get_logger(__name__)
#     log.info("starting attack")
#     log.warning("YOLO returned 0 detections")
#
# Migration policy (Phase C.5):
#   - brain.py + agents/*           → migrate to logging (high value for dashboard)
#   - perception_thread             → migrate (background thread, want timestamps)
#   - environment.py / env_v4.py    → KEEP print() for now (verbose=False can silence
#                                      already; full migration is mechanical, low ROI)
#   - tools/train_rl_v4.py reports  → KEEP print() (CLI user-facing output)

import logging
import os
from logging.handlers import RotatingFileHandler


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DEFAULT_LEVEL = os.environ.get('CLASHAI_LOG_LEVEL', 'INFO').upper()
LOG_DIR = os.environ.get('CLASHAI_LOG_DIR', 'logs')
LOG_FILE = os.path.join(LOG_DIR, 'clashai.log')
LOG_MAX_BYTES = 5 * 1024 * 1024   # 5 MB per file
LOG_BACKUP_COUNT = 3

# Console : compact (one-line per message)
CONSOLE_FMT = '[%(levelname).1s %(asctime)s %(name)s] %(message)s'
CONSOLE_DATEFMT = '%H:%M:%S'

# File : full info for post-mortem
FILE_FMT = '%(asctime)s %(levelname)-7s %(name)s | %(message)s'
FILE_DATEFMT = '%Y-%m-%d %H:%M:%S'


# -----------------------------------------------------------------------------
# Setup (idempotent)
# -----------------------------------------------------------------------------

_configured = False


def _configure_once() -> None:
    """Install the console + file handlers on the root logger. Idempotent."""
    global _configured
    if _configured:
        return

    root = logging.getLogger('clashai')
    root.setLevel(DEFAULT_LEVEL)
    root.propagate = False

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(DEFAULT_LEVEL)
    console.setFormatter(logging.Formatter(CONSOLE_FMT, CONSOLE_DATEFMT))
    root.addHandler(console)

    # File handler — best effort, don't crash if logs/ isn't writable
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        fh = RotatingFileHandler(
            LOG_FILE,
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding='utf-8',
        )
        fh.setLevel(DEFAULT_LEVEL)
        fh.setFormatter(logging.Formatter(FILE_FMT, FILE_DATEFMT))
        root.addHandler(fh)
    except OSError as e:
        root.warning(f"Could not open log file {LOG_FILE}: {e}")

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger named under the `clashai` hierarchy.

    Pass `__name__` from the calling module — Python's logging treats
    dotted names hierarchically, so e.g. `clashai.brain` is a child of
    `clashai`. That lets us tune verbosity per sub-package later
    (e.g. silence perception while keeping brain at INFO).
    """
    _configure_once()
    # Ensure the name lives under our 'clashai' root.
    if not name.startswith('clashai'):
        name = f'clashai.{name}'
    return logging.getLogger(name)


def set_level(level: str) -> None:
    """Change the log level at runtime (e.g. from a CLI flag)."""
    _configure_once()
    root = logging.getLogger('clashai')
    root.setLevel(level)
    for h in root.handlers:
        h.setLevel(level)
