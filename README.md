# ClashAI

> Autonomous AI agent that plays Clash of Clans via ADB on an Android emulator,
> powered by PPO reinforcement learning, YOLO object detection, and CNN perception.

![Version](https://img.shields.io/badge/version-4.2-blue)
![Python](https://img.shields.io/badge/python-3.12+-green)
![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange)
![License](https://img.shields.io/badge/license-Proprietary-red)

---

## Overview

ClashAI observes the game screen through ADB screenshots, detects buildings and troops
using YOLO models, and takes actions (deploy troops, cast spells, activate hero abilities)
using a PPO (Proximal Policy Optimization) agent trained end-to-end on real gameplay.

**Key capabilities:**
- Fully autonomous attack loop: village scan → troop deployment → mid-combat decisions → result collection
- YOLO-based building and troop detection running on every observation step
- Behavioral Cloning (BC) pre-training from heuristic demonstrations before PPO
- Context-aware spell casting (heal on injured clusters, rage on large groups, freeze on infernos)
- Hero ability timing based on combat progress and troop health ratios
- Clan castle troop requests with cooldown management
- GdC (Guerre de Clan) navigator with auto-scouting and attack sequencing

---

## Architecture

```
ClashAI/
├── clashai/
│   ├── brain.py                    # Main orchestrator — farm loop, GdC mode
│   ├── combat/
│   │   ├── environment_v4.py       # RL environment (V4.2 — fused phase)
│   │   ├── agent_v4.py             # PPO agent — CNN grid + MLP vector head
│   │   ├── action_space.py         # 37 actions: deploy × role × sector, spells, abilities
│   │   ├── state_encoder.py        # YOLO output → (12, 40, 40) grid + (20,) feature vector
│   │   ├── combat_observer.py      # Mid-combat YOLO troop tracking + HSV health bars
│   │   ├── reward_shaping.py       # Centralized reward logic (step + final)
│   │   ├── hero_ability.py         # Hero ability timing and state tracking
│   │   ├── spell_caster.py         # Intelligent spell targeting (V2)
│   │   └── troop_manager.py        # Troop bar scan, selection, cleanup
│   ├── perception/
│   │   ├── building_detector.py    # YOLO buildings inference
│   │   ├── troop_detector.py       # YOLO troops inference
│   │   ├── deploy_zone.py          # Convex hull deploy perimeter from YOLO bboxes
│   │   ├── screen_classifier.py    # CNN — 12 screen states (village, combat, results…)
│   │   ├── reward_reader.py        # Star / percentage extraction from result screen
│   │   ├── troop_finder.py         # Template matching — troop bar slots
│   │   └── troop_counter.py        # OCR — troop counters (x2, x11…)
│   ├── navigation/
│   │   ├── brain.py / game_loop.py # ADB helpers, model loader, screen routing
│   │   ├── gdc_navigator.py        # Guerre de Clan — scout + attack flow
│   │   ├── calibrate_ui.py         # Interactive UI position calibration
│   │   └── zoom_control.py         # ADB pinch zoom
│   └── social/
│       ├── clan_castle.py          # CC troop request automation
│       └── clan_chat_monitor.py    # Chat command listener
│
├── tools/
│   ├── train_rl_v4.py              # V4 PPO training script (+ BC pre-training)
│   ├── train_yolo_buildings.py     # YOLO buildings fine-tuning
│   ├── train_yolo_troops.py        # YOLO troops fine-tuning
│   ├── train_cnn.py                # Building CNN classifier training
│   ├── train_screen_cnn.py         # Screen state CNN training
│   ├── capture_combat.py           # Dataset capture tool (live ADB screenshots)
│   ├── prepare_dataset.py          # Dataset split and formatting
│   ├── convert_labelme.py          # LabelMe JSON → YOLO TXT conversion
│   └── calibrate_ui.py             # Standalone UI calibration runner
│
├── configs/
│   ├── coc.yaml                    # YOLO buildings dataset config (48 classes)
│   ├── coc_troops.yaml             # YOLO troops dataset config
│   └── ui_positions.json           # Calibrated UI element positions
│
├── weights/                        # Trained model files (not committed)
│   ├── best.pt                     # YOLO buildings (48 classes)
│   ├── yolo_troops.pt              # YOLO troops (mAP50=0.987)
│   ├── screen_cnn.pth              # Screen state classifier
│   ├── building_cnn.pth            # Building classifier
│   └── rl/                         # PPO checkpoints
│
└── templates/                      # Template images for matching
    ├── troops/                     # Troop bar icons
    ├── hero_abilities/             # Hero ability icons
    ├── reward_digits/              # Digit templates for OCR
    └── clan_castle/                # CC request button template
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| RL Agent | PPO (PyTorch) — CNN grid encoder + MLP vector head |
| Object Detection | YOLOv11n (Ultralytics) — buildings + troops |
| Screen Classification | ResNet-based CNN (PyTorch) |
| Template Matching | OpenCV TM_CCOEFF_NORMED |
| OCR | EasyOCR — troop counters |
| Game Interface | ADB (Android Debug Bridge) via subprocess |
| Emulator | Google Play Games Developer Emulator 1920×1080 |
| Package Manager | uv |

---

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- CUDA-capable GPU (tested on RTX 5070 Laptop, 8 GB VRAM)
- Android emulator running Clash of Clans at **1920×1080**
- ADB installed and accessible in PATH

---

## Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd COCProj

# 2. Install dependencies (includes PyTorch CUDA 12.8)
uv sync

# 3. Connect to the emulator
adb connect localhost:6520
adb devices  # verify connection
```

---

## Configuration

### UI Calibration (required on first run)

The agent uses pixel-precise UI positions that vary between emulators.
Run the calibration tool and follow the on-screen instructions:

```bash
uv run python -m clashai.navigation.calibrate_ui
```

This saves positions to `configs/ui_positions.json`.

### Required weights

Place trained model files in `weights/`:

| File | Description |
|------|-------------|
| `best.pt` | YOLO buildings (48 classes incl. walls) |
| `yolo_troops.pt` | YOLO troops (13 classes, mAP50=0.987) |
| `screen_cnn.pth` | Screen state classifier (12 classes) |
| `building_cnn.pth` | Building fine-classifier |

---

## Usage

### Autonomous Farm Mode

```bash
uv run python -m clashai.brain --mode farm
```

The agent will loop: find a base → attack → collect resources → repeat.

### Guerre de Clan (Clan War) Mode

```bash
uv run python -m clashai.brain --mode gdc
```

Scouts all available bases, picks the best target, and attacks.

### Single Attack (test / debug)

```bash
uv run python -m clashai.brain --mode farm --episodes 1
```

---

## Training

### PPO Agent (V4)

```bash
# Heuristic baseline (no learning)
uv run python tools/train/train_rl_v4.py --heuristic --episodes 5

# Behavioral Cloning pre-training + PPO
uv run python tools/train/train_rl_v4.py --pretrain 15 --bc-epochs 15 --episodes 500

# Resume from checkpoint
uv run python tools/train/train_rl_v4.py --resume --episodes 200
```

### YOLO Buildings

```bash
uv run python tools/train/train_yolo_buildings.py
# Config: configs/coc.yaml — 48 classes
# Output: runs/detect/FinishedTrain/weights/best.pt → weights/best.pt
```

### YOLO Troops

```bash
uv run python tools/train/train_yolo_troops.py --epochs 300 --batch 16
# Config: configs/coc_troops.yaml — 13 classes
# Output: weights/yolo_troops.pt
```

### Screen Classifier CNN

```bash
uv run python tools/train/train_screen_cnn.py
# Dataset: datasets/screen_states/
# Output: weights/screen_cnn.pth
```

### Dataset Capture

```bash
# Capture live screenshots during combat for training data
uv run python tools/data/capture_combat.py --max 300 --interval 2.0
```

---

## Observation Space

The PPO agent receives a dual observation at each step:

| Input | Shape | Description |
|-------|-------|-------------|
| Grid | `(12, 40, 40)` | Spatial map — building types, danger zones, deploy perimeter |
| Vector | `(54,)` | Village features, role inventory, spell counts, combat features, hero status |

**Grid channels:** building types (one-hot by category), danger ground, danger air, deploy zone mask, destroyed markers.

**Vector breakdown:**
- `[0:20]` — village features (building counts, threat scores, side strengths)
- `[20:25]` — role inventory (tanks, ranged, melee, heroes, siege available)
- `[25:28]` — spell inventory (heal, rage, freeze)
- `[28:33]` — sector map (which sectors have been deployed to)
- `[33:34]` — time norm (elapsed / 180s)
- `[34:49]` — combat features (buildings remaining, troops alive, hurt ratio, clusters…)
- `[49:54]` — hero status (alive/deployed/ability ready per hero)

---

## Action Space

37 discrete actions:

| Range | Type | Description |
|-------|------|-------------|
| `[0–24]` | Deploy | 5 roles × 5 sectors (tank/ranged/melee/hero/siege × left/center-left/center/center-right/right) |
| `[25–27]` | Spell | heal / rage / freeze |
| `[28–32]` | Ability | 5 hero ability slots |
| `[33]` | Observe | Screenshot + YOLO inference |
| `[34]` | Wait short | 1s pause |
| `[35]` | Wait long | 3s pause |
| `[36]` | Done | End episode |

All actions are available at every step; invalid ones are masked by the action mask.

---

## Versioning

| Version | Status | Summary |
|---------|--------|---------|
| V1–V2 | Done | Single-decision and basic sequential deployment |
| V3 | Done | Full deploy+combat pipeline, 289 actions, 1.2M params |
| V4.0 | Done | Simplified action space (37 actions), YOLO troops |
| V4.1 | Done | BC pre-training, bug fixes, 192-episode validation run |
| V4.2 | Done | Fused deploy/combat phases, continuous YOLO, YOLO deploy zone, advanced reward shaping |
| V4.3 | Planned | CNN troop bar (replaces template matching + OCR) |
| V5 | Vision | Camera scroll, multi-composition, hero equipment awareness |

---

## License

Copyright (c) 2024–2026 Turtelthib. All Rights Reserved.

See [LICENSE](LICENSE) for full terms. Unauthorized use, copying, or distribution is prohibited.
