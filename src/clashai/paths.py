"""Centralized paths and ADB configuration for the ClashAI project.

Single source of truth (SSOT) for every filesystem path. All modules
import their paths from here instead of recomputing project_root with
os.path.dirname() each time.
"""
import os
from pathlib import Path


def _find_project_root() -> str:
    """Walk up from this file until we find pyproject.toml = repo root.

    Robust to where the package lives (src/clashai/, clashai/, …) — moving
    the package only requires that pyproject.toml stays at the repo root.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / 'pyproject.toml').exists():
            return str(parent)
    # Fallback: legacy layout (clashai/ directly under root)
    return str(here.parents[1])


# Project root — located via the pyproject.toml marker (SSOT).
PROJECT_ROOT = _find_project_root()

# =============================================================================
# ADB DEVICE
# =============================================================================
# Serial of the target emulator (output of `adb devices`).
# Update this when switching emulators.
#   Google Play Games (old): localhost:6520
#   LDPlayer / BlueStacks / new emulator: 127.0.0.1:5555
ADB_DEVICE = os.environ.get("ADB_DEVICE", "localhost:6520")

# Configs — stays at repo root (user decision).
CONFIGS_DIR = os.path.join(PROJECT_ROOT, 'configs')
UI_POSITIONS_FILE = os.path.join(CONFIGS_DIR, 'ui_positions.json')

# Weights — stays at repo root (user decision).
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, 'weights')
RL_WEIGHTS_DIR = os.path.join(WEIGHTS_DIR, 'rl')

# Data root — datasets, templates, captures, debug output all live here now.
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

# Templates (under data/)
TEMPLATES_DIR = os.path.join(DATA_DIR, 'templates')
TROOP_TEMPLATES_DIR = os.path.join(TEMPLATES_DIR, 'troops')
HERO_TEMPLATES_DIR = os.path.join(TEMPLATES_DIR, 'hero_abilities')
REWARD_TEMPLATES_DIR = os.path.join(TEMPLATES_DIR, 'reward_digits')
REWARD_DIGITS_DIR = os.path.join(REWARD_TEMPLATES_DIR, 'digits')

# Datasets (under data/)
DATASETS_DIR = os.path.join(DATA_DIR, 'datasets')

# Debug output (under data/)
DEBUG_DIR = os.path.join(DATA_DIR, 'debug_reward')
