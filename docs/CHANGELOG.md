# ClashAI — Changelog (tout ce qui est fait)

Historique chronologique des features livrées, du plus récent au plus ancien.

> Ce qui reste à faire : [ROADMAP.md](ROADMAP.md). Blocs de fix détaillés : [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

## Légende
✅ livré · 🐛 bug corrigé au passage · 🔧 → bloc détaillé dans TROUBLESHOOTING.md

---

## V5.1 — Foundation multi-agents (en cours)

Plomberie pour les sous-agents. Posé à côté du système existant → le bot tourne identique tant que `brain.py` n'est pas branché sur le scheduler.

- ✅ `BaseAgent` (`agents/base.py`) : `can_run(world)`, `run()`, `priority`, `cooldown_seconds`, état/erreurs/telemetry (Session 13).
- ✅ `AgentScheduler` (`agents/scheduler.py`) : registry + `pick(world)` (prio + cooldown + can_run) + `tick()` + history + status (Session 13).
- ✅ `build_world(models, **flags)` (`agents/world.py`) : snapshot SSOT lu par tous les `can_run()`, alimenté par le cache `PerceptionThread` (zéro screenshot bloquant) + flag `on_village_home`. Marche à vide.
- ✅ **4 agents concrets** enveloppant les capacités existantes (logique non réécrite) :
  - `ClanCastleAgent` (prio 20) — demande de troupes, cooldown délégué au manager.
  - `CombatAgent` (prio 10) — farm, activité par défaut. **DRY** : extraction de `combat/episode_runner.py::run_attack_episode()` (SSOT partagé avec `brain/farm.py`).
  - `GdCAgent` (prio 25) — guerre sur cible queuée (`enqueue_target`). 🐛 `GdCOrchestrator._run_attack` délègue aussi au runner (corrige un override de `heuristic_mode` qui tentait le RL sur un réseau non chargé).
  - `ChatAgent` (prio 30) — lit le chat, dispatche (`attack N → gdc.enqueue_target`), répond. **Canal d'entrée NL du futur LocalLLMBrain.**
- ✅ Chaque agent a une démo offline (sans émulateur) prouvant `world → can_run → pick → run` + préemption de priorité + mode gating.
- 🐛 **Fix famine d'agent** (révélé au 1er run réel) : `ClanCastleAgent` (prio 20, cooldown 0 + `can_run` toujours vrai car template manquant) monopolisait le scheduler → `CombatAgent` jamais lancé (que des pauses). Fix : `cooldown_seconds = REQUEST_COOLDOWN` (le scheduler pose le cooldown après chaque run, succès ou échec). Voir TROUBLESHOOTING.
- ✅ **Interface `Brain` + brain branché sur le scheduler (Étape A)** : `brain/interface.py` (`Brain` ABC + `HeuristicBrain` = `scheduler.pick`). `brain.py` enregistre les 4 agents dans un `AgentScheduler` et son `_main_loop` est réécrit (`world → brain.decide → scheduler.run → stats`). Vieilles méthodes taguées `[DEAD-CODE-V5.1]` (revert-safe, à supprimer en Étape B). **Première étape qui change le comportement runtime** (le bot route via le scheduler).

---

## Refacto architecture repo (src/ layout + split gros fichiers) — Session 13

> Plan : `.claude/plans/okk-maintenant-grosse-modification-jolly-manatee.md`. Rythme : 1 fichier = 1 test (`--test`) = 1 commit. Chaque split = sous-dossier par domaine + `__init__.py` ré-exportant l'API (back-compat). Vérif : compileall + scan AST + import test des importeurs.

- ✅ **Phase 1** : `clashai/`+`tools/` → `src/` ; data → `data/` ; `paths.py` en résolution SSOT par marqueur `pyproject.toml` ; hatchling src-layout.
- ✅ **Phase 2** : V3 déprécié isolé dans `combat/legacy/`.
- ✅ **Phase 3 — 13 splits** (12 du plan + spell_caster bonus) :

| # | Fichier | → Cible |
|---|---|---|
| 1 | `perception/screen_capture.py` | `screen_capture/` |
| 2 | `perception/deploy_zone.py` | `deploy/` (+ shim) |
| 3 | `perception/reward_reader.py` | `reward_reader/` |
| 4 | `combat/state_encoder.py` | `encoder/` (+ shim) |
| 5 | `navigation/game_loop.py` | `game_loop/` |
| 6 | `combat/hero_ability.py` | `hero/` (+ shim) |
| 7 | `social/clan_chat_monitor.py` | `social/chat/` (+ shim) |
| 8 | `navigation/gdc_navigator.py` | `navigation/gdc/` (+ shim) |
| 9 | `brain.py` | `brain/` (mixins) |
| 10 | `combat/environment_v4.py` | `environment_v4/` (mixins + MRO) |
| 11 | `combat/agent_v4.py` | `agent_v4/` |
| 12 | `combat/combat_observer.py` | `combat_observer/` |
| + | `combat/spell_caster.py` | `spell_caster/` |

- ✅ **Critère atteint : 0 fichier >500L hors `legacy/`.**
- 🐛 3 bugs préexistants corrigés : `weights_dir` GdC (pointait `src/weights/rl`), entry point `clashai-brain` cassé (pas de `main()`), `NO_TROOPS_CHECKS_THRESHOLD` dupliqué.
- 🔧 Migration capacités héros template → CNN (voir TROUBLESHOOTING) + fix capas jamais déclenchées (heuristique).

---

## V5.0 — Mode "en direct" (push pipeline) — Session 13

Phases 1-2 livrées (3-4 optionnelles, voir ROADMAP).

- ✅ **Phase 1** : `ScreenCapture.subscribe_to_frames(callback)` — API push universelle. WGC fire nativement sur `on_frame_arrived` (30-60fps) ; fallback poller 30fps pour les autres backends. `_fire_frame_callbacks_from_bgra()` convertit BGRA→PIL+normalize une fois pour tous.
- ✅ **Phase 2** : `PerceptionThread._capture_loop` ne polle plus — s'abonne via `subscribe_to_frames` et bloque sur un wait. `_on_new_frame` push dans la queue avec dédup (max 1 frame en attente).

---

## V4.3 — Perception + Vitesse — Session 12

- ✅ YOLO walls segmentation → deploy zone précise ; `get_perimeter_from_walls()`.
- ✅ Capture directe fenêtre (mss puis WGC) ~20ms vs 150ms ADB ; `adb_screenshot()` WGC d'abord, ADB fallback.
- ✅ `PerceptionThread` async (capture + YOLO en fond) ; `_update_combat_observation()` lit le cache (non-bloquant) ; `DELAY_OBSERVE` 2.5s→0.15s ; délais deploy −65%.
- ✅ YOLO barre de troupes (78 classes) ; `TroopBarDetector` + filtre HSV grisé ; `TroopFinder.update()` YOLO d'abord.
- ✅ OCR compteurs (EasyOCR, upscale ×3) ; hard cap héros uniques à 1 (`UNIQUE_HEROES`) ; suppression rescan périodique → `_sync_remaining_from_perception()` lit le cache.
- ✅ Mode `--test` : 1 épisode + 5 captures annotées dans `logs/test_run/` (`test_run_capture.py::TestRunCapture`). Debug overlay `--debug-overlay`.
- ✅ Bug séquence de récupération supprimée (l'agent ne panique plus sur état imprévu).
- 🔧 Fix capture émulateur occluded (WGC) ; fix `atexit` WGC (`Fatal Python error` au Ctrl+C) ; alignement `imgsz` par modèle ; fix demande CC (5 bugs) ; bug RGB/BGR YOLO → tous dans TROUBLESHOOTING.

---

## V4.2 — Refonte architecture combat — Session 8-11

> Suppression des phases rigides : l'agent devient réactif comme un humain.

- ✅ **Fusion phases deploy/combat** : `phase_indicator` supprimé (`PHASE_SIZE=0`), les 37 actions dispo à chaque step, masking sur ressources restantes (plus sur la phase). `VECTOR_SIZE` 55→54.
- ✅ **Suppression limite de steps** : `MAX_STEPS_PER_EPISODE` → `MAX_STEPS_SAFETY=200` (filet) ; fin naturelle via `_all_resources_exhausted()` ; `step_norm`→`time_norm` (timer CoC réel 180s).
- ✅ **YOLO continu** : bâtiments + troupes à chaque step ; détection destruction par diff (`_buildings_destroyed_total`, +2.0/bâtiment) ; `feature[0]` = `buildings_remaining_ratio`.
- ✅ **Zone de déploiement** : `get_perimeter_from_buildings()` (hull convexe + offset 35px) ; côté faible via `find_best_attack_side()`.
- ✅ **Reward shaping** : destruction sec/sec, survie héros (+5/héros), sorts contextuels (rage/soin/gel), combo clutch.
- 🔧 Bug échec navigation → faux -50 reward (voir TROUBLESHOOTING).
- ✅ (V4.2.1) Fixes : PPO value loss, BC loss, ability deadlock, deploy zone walls seg.

---

## V4.1 — Quick wins & analyse post-training — Session 7

> Run validation 192 épisodes PPO + 15 BC. ⭐ moy 1.34 (vs 1.16 V4.0), 2+⭐ 42.7%.

- ✅ Analyse 333 épisodes : PPO n'a pas convergé (bug reward + entropy + pas d'imitation).
- 🐛 **BUG CRITIQUE** : `_compute_shaping_reward()` passait `hero_idx` dans `spell_name` → abilities jamais récompensées.
- ✅ Imitation learning (behavioral cloning) ; `ENTROPY_COEF` 0.04→0.02 ; malus sorts non utilisés (−5/sort) ; fix double appel YOLO `CombatObserver`.
- ✅ Feature CC : détection château (YOLO), CC plein, template "Demande", cooldown 15min, intégration `brain.py`.
- Commande : `uv run python tools/train/train_rl_v4.py --pretrain 15 --episodes 200`.

---

## Versions antérieures

| Version | Résumé |
|---|---|
| V1 | Une seule décision par attaque |
| V2 | Améliorations intermédiaires |
| V3 | Déploiement séquentiel + combat réactif (289 actions, 1.2M params) |
| V4.0 | Action space simplifié 37 actions + YOLO troupes (Session 6) |
