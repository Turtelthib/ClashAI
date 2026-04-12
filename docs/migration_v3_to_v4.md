# Migration V3 → V4

## Fichiers supprimés (legacy)
- `agent.py` (V1), `agent_v2.py` (V2)
- `environment.py` (V1), `environment_v2.py` (V2)
- `train_rl.py` (V1), `train_rl_v2.py` (V2)
- `deploy_scripts.py` (V1)
- `lunch.py`, `test_screen.py`, `train_all.py`

## Renommages
| Ancien | Nouveau |
|--------|---------|
| `agent_v3.py` | `clashai/combat/agent.py` |
| `environment_v3.py` | `clashai/combat/environment.py` |
| `train_rl_v3.py` | `tools/train_rl.py` |
| `troop_count_reader.py` | `clashai/perception/troop_counter.py` |
| `model.py` | `clashai/perception/screen_classifier.py` |
| `combine.py` | `clashai/perception/building_detector.py` |

## Nouveaux fichiers V4
| Fichier | Rôle |
|---------|------|
| `clashai/paths.py` | Chemins centralisés (PROJECT_ROOT, WEIGHTS_DIR, etc.) |
| `clashai/perception/troop_detector.py` | Wrapper YOLO troupes (13 classes) |
| `configs/coc_buildings.yaml` | Config YOLO bâtiments |
| `configs/coc_troops.yaml` | Config YOLO troupes |
| `pyproject.toml` | Config uv (remplace requirements.txt) |

## Intégration YOLO troupes V4
- **TroopDetector** : charge `weights/yolo_troops.pt`, retourne des `Detection` typées
- **CombatObserver** : mode YOLO (positions exactes par classe) + fallback barres HSV
- **SpellCaster** : `analyze_from_yolo()` cible les sorts via détections YOLO
- **HeroAbilityManager** : `update_battlefield_positions()` suit les héros via YOLO
- **Environment** : retraite basée sur le comptage YOLO (plus fiable que barres vertes)

## Imports
Tous les imports utilisent maintenant le package `clashai`:
```python
# Avant
from state_encoder import encode_state
from calibrate_ui import get_position

# Après
from clashai.combat.state_encoder import encode_state
from clashai.navigation.calibrate_ui import get_position
```

## Commandes
```bash
# Avant
python scripts/rl/brain.py --mode auto
python scripts/rl/train_rl_v3.py --episodes 100

# Après
python -m clashai.brain --mode auto
python tools/train_rl.py --episodes 100
```
