# Baselines RL — références gelées

> Chaque run baseline est **figé** ici (stats) + **archivé** sur disque (log + checkpoint) pour comparer les runs futurs **directement, sans re-déduire à la main**.

## Comment comparer un nouveau run

```bash
uv run python src/tools/train/compare_baseline.py
# ou en précisant :
uv run python src/tools/train/compare_baseline.py \
  --log weights/rl/training_log_v4.json \
  --baseline weights/baselines/v4.4-ppo-350ep/stats.json
```
→ affiche baseline vs courant côte à côte + delta (`[+]` = mieux, `[-]` = moins bien).

## Baselines

### `v4.4-ppo-350ep` — *2026-07-12*

PPO brut (pretrain BC 30 + PPO), obs **67 dims / 50 actions** (post rework sorts). **Convergé / plateau.**

| Métrique | Valeur |
|---|---|
| Épisodes | 350 |
| Reward moyen | 281.5 (min −103, max 682) |
| **Étoiles moy** | **1.49** |
| % 2★ et + | 52.9 % |
| % 3★ | 13.1 % |
| % 0★ (ratés) | 16.9 % |
| % destruction moy | 64.9 % |

> ⚠️ Ce sont les moyennes **globales** (incluent le creux du début). Sur les ~70 derniers épisodes : ~1.67★ / 72 % / 0 raté.

**Verdict** : plafonne au **niveau BC/heuristique** (~1.7★). Le RL brut ne casse pas le plafond → le levier est le **cerveau LLM** (stratégie), pas plus d'épisodes RL. Les ~17 % de 0★ = **bugs deploy/nav**, pas la politique.

**Fichiers archivés** (locaux, `weights/` est gitignoré — non pushés mais préservés sur disque) :
`weights/baselines/v4.4-ppo-350ep/` → `training_log_v4.json`, `agent_v4_checkpoint.pth`, `agent_v4_pretrained_bc.pth`, `stats.json`.

**Prochain run "propre"** : après les fixes deploy (taps invalides) + capas héros tardifs + une **contrainte KL-vers-BC** (empêcher PPO de redescendre sous l'heuristique). Comparer avec l'outil ci-dessus.
