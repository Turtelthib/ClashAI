"""Centralized paths for the ClashAI project.

All modules import their paths from here instead of
recomputing project_root with os.path.dirname() each time.
"""
import os

# Project root = parent of the clashai/ package
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
