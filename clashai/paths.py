"""Centralized paths and ADB configuration for the ClashAI project.

All modules import their paths from here instead of
recomputing project_root with os.path.dirname() each time.
"""
import os

# Project root = parent of the clashai/ package
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =============================================================================
# ADB DEVICE
# =============================================================================
# Serial of the target emulator (output of `adb devices`).
# Update this when switching emulators.
#   Google Play Games (old): localhost:6520
#   LDPlayer / BlueStacks / new emulator: 127.0.0.1:5555
ADB_DEVICE = os.environ.get("ADB_DEVICE", "localhost:6520")

# Configs
CONFIGS_DIR = os.path.join(PROJECT_ROOT, 'configs')
UI_POSITIONS_FILE = os.path.join(CONFIGS_DIR, 'ui_positions.json')

# Templates
TEMPLATES_DIR = os.path.join(PROJECT_ROOT, 'templates')
TROOP_TEMPLATES_DIR = os.path.join(TEMPLATES_DIR, 'troops')
HERO_TEMPLATES_DIR = os.path.join(TEMPLATES_DIR, 'hero_abilities')
REWARD_TEMPLATES_DIR = os.path.join(TEMPLATES_DIR, 'reward_digits')
REWARD_DIGITS_DIR = os.path.join(REWARD_TEMPLATES_DIR, 'digits')

# Weights
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, 'weights')
RL_WEIGHTS_DIR = os.path.join(WEIGHTS_DIR, 'rl')

# Datasets
DATASETS_DIR = os.path.join(PROJECT_ROOT, 'datasets')

# Debug
DEBUG_DIR = os.path.join(PROJECT_ROOT, 'debug_reward')
