# ClashAI — Changelog (tout ce qui est fait)

Historique chronologique des features livrées, du plus récent au plus ancien.

> Ce qui reste à faire : [ROADMAP.md](ROADMAP.md). Blocs de fix détaillés : [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

## Légende
✅ livré · 🐛 bug corrigé au passage · 🔧 → bloc détaillé dans TROUBLESHOOTING.md

---

## V5.1 — Foundation multi-agents (en cours)

Plomberie pour les sous-agents. Posé à côté du système existant → le bot tourne identique tant que `brain.py` n'est pas branché sur le scheduler.

- ✅ **Baseline RL figé + outil de comparaison** (Session 15) : le run PPO brut (350 ép, obs 67/actions 50) a **plateauté ~1.7★ / 53% de 2★+** = niveau BC/heuristique → le RL brut ne casse pas le plafond (confirme que le levier est le cerveau LLM, pas plus d'épisodes). Archivé `weights/baselines/v4.4-ppo-350ep/` (log+checkpoint+`stats.json`, local car `weights/` gitignoré) ; trace git `docs/baselines.md` ; outil `tools/train/compare_baseline.py` compare un run au baseline (côte à côte + delta), sans re-déduire.
- ✅ **Validation en conditions réelles (Session 15)** : rework 16 sorts + seed digit-CNN testés en run réel → OK. **Re-train sur la nouvelle obs 67 dims / 50 actions lancé** (gros run = baseline avant la suite).
- ✅ **Décision (Session 15) — chantier "deploy-until-grayed" requalifié en hardening** : sa prémisse (« pas de compteur fiable ») est obsolète depuis le digit-CNN (seed reset + re-lecture live). La refonte obs "présence-par-rôle" est **abandonnée** (perdrait de l'info vs les vrais comptes, désormais fiables). Architecture actée : **compteurs digit-CNN = source primaire, grisé = autorité de fin / filet**. Restent des petits items (mask "deploy si non-grisé", validation des rôles best-guess) → voir backlog ROADMAP.
- ✅🐛 **Sorts non tous lancés + rage mal placé** (Session 14, suite du rework) :
  - *Leftover* : `_execute_spell` plafonnait au compteur seedé à `default_max` = `max` JSON (gel=1, rage=3) → laissait 2 gel / 1 rage. Fix : les sorts **ignorent le `max` JSON** et sont seedés généreux (`DEFAULT_MAX_BY_ROLE['spell']=8`, cast-until-grayed) → le grisé coupe au vrai compte.
  - *Rage au centre* : la détection terrain (`yolo_troops.pt`, sous-entraînée) trouve souvent 0 troupe → `main_cluster` tombe sur le fallback `village_center`. Fix : support spells (cluster/heal) visent le **chemin de marche** (`_troop_march_point`, côté attaque→cœur) quand `num_troops==0`, + **spread** des casts cluster consécutifs (`_spread_cluster_point`) pour ne plus empiler les rages. (Le gel marchait déjà : `_find_freeze_target` cherche une défense proche.) Fix de fond = retrain `yolo_troops.pt` (ROADMAP).
- ✅ **Rework complet des sorts (data-driven)** (Session 14) : `SPELL_NAMES` dérivé du registre `troops.json` **∩ classes du CNN** (`troop_registry.load_spell_names`) — plus de `['soin','rage','gel']` ni de `+3` codés en dur (`ACTION_CAST_*` retirées, `ACTION_ABILITY_START` dérivé). 3→**16 sorts** ; un sort pré-enregistré mais pas encore dans le CNN reste inerte (pas de dim morte / re-train inutile). Ciblage **data-driven** (`SPELL_TARGET_DEFAULTS` cluster/heal/defense, overridable via `target` JSON) mappé sur SpellCaster. Heuristique caste tous les sorts présents (rage/gel/soin d'abord). `PPOAgentV4.load()` tolère le mismatch de dims. **obs 54→67, actions 37→50 → re-train** (heuristique fonctionne sans entraînement).
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
- ✅ **Interface `Brain` + brain branché sur le scheduler (Étape A)** : `brain/interface.py` (`Brain` ABC + `HeuristicBrain` = `scheduler.pick`). `brain.py` enregistre les 4 agents dans un `AgentScheduler` et son `_main_loop` est réécrit (`world → brain.decide → scheduler.run → stats`). **Première étape qui change le comportement runtime** ; validée en run réel (CombatAgent attaque via le scheduler, `--mode farm`).
- ✅ **Cleanup brain (Étape B)** : suppression des méthodes mortes + fichiers mixins `farm.py`/`war.py`/`chat.py` (logique désormais 100% dans les agents). `ClashBrain` = `core` + `loop` + `navigation`. Compteurs morts retirés.

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

## V4.4 — Polish perception (en cours) — Session 14-15

- ✅ **Digit CNN validé en conditions réelles (Session 15)** : seed au reset + re-lecture live confirmés en run réel → le mini-CNN chiffres est clos (reste le renfort data au fil de l'eau). Dernière étape V4.4 = le gros run (re-train 67 dims, en cours).

- ✅🐛 **Sorts château de clan écrasés + flèche siège/gardien** (Session 14, avant gros run) :
  - *Doublons château* : un sort présent 2× (armée x3 + château x1). 3 bugs combinés : (a) `read_bar_counts`/`to_positions` keyés par nom → compteur+position **écrasés** → fix **somme** + positions en liste/refresh ; (b) le **vrai bloqueur** : `_sync_remaining_from_perception` zérotait par nom → l'icône armée grisée mettait `rage=0` alors que le château était actif → 4e cast refusé. Fix : dépletion seulement si **toutes** les icônes du nom sont grisées ; (c) `finder.positions` rafraîchi **avant chaque deploy/sort** → une fois l'armée grisée, `select()` tape l'icône château.
  - *Flèche verte* : `to_positions` tapait le **centre** de l'icône → sur les engins de siège et le grand gardien, ça touchait la flèche de mode (bas) → ouvre un sous-menu → la troupe ne se déployait pas (déployée seulement au rescan). Fix : taper le **haut de l'icône** (`y1 + 0.35·h`).
- ✅ **Digit CNN — intégration reset-seeding + live re-read (Phase 4)** : `core._seed_counts_from_digits()` lit la barre de combat au début de l'attaque et seede `_remaining_troops` avec les **vrais compteurs** (troupes **ET sorts** ; fallback `default_max` généreux si conf basse). En cours d'attaque, `_sync_remaining_from_perception` **re-lit les compteurs (digit-CNN) à chaque `observe`** (frame fraîche, hors burst) → corrige la dérive du décrément manuel quand un deploy rate. Pré-deploy reste grisé-only (sûr, le cache laggé re-gonflerait un compteur → re-deploy). `digit_reader.crop_count_badge`/`read_bar_counts` = SSOT partagé avec collect. Modèle retrainé sur data 0/7 enrichie : **100% val acc/classe**. À valider émulateur.
- ✅ **Digit CNN par-chiffre (B2)** : lecture des compteurs troop bar ("x12" → 12). `clashai/perception/digit_reader.py` (SSOT : segmentation par projection-profile + drop du "x" + filtre hauteur, `DigitCNN`, `read_count` avec conf-gating). `tools/data/build_digit_singles.py` convertit le labeling whole-number en dataset par-chiffre 0-9 (réutilisé, 730→634 crops). `train_digit_cnn.py` adapté (augmentation + oversampling des classes rares + acc/classe). **Modèle 98% val acc**, read e2e 83.7% brut (conf-gating → ~8% fallback). Longueur variable (7, 79, 200…). À renforcer : `0`/`7` rares. Intégration (Phase 4) = à brancher.

- ✅🐛 **Deploy de troupes grisées pendant le burst** (`_sync_grayed_from_cache` dans `_execute_deploy`) : l'heuristique sur-estimait les comptes (`default_max`) et tapait les icônes grisées en boucle car le filtre grisé ne tournait qu'aux steps `observe` (après le burst de deploy). Fix : lecture gratuite du cache PerceptionThread avant chaque deploy → grisé respecté en plein burst. Bloc détaillé dans TROUBLESHOOTING.
- ✅🐛 **Registre de troupes data-driven** (`configs/troops.json` + `combat/troop_registry.py`) : `TROOP_TYPES` (legacy/agent.py) + `ROLE_TO_TROOPS` (action_space.py) en **dérivent** (loader). **Corrige le bug critique "les troupes non hardcodées ne se déploient pas"** : le CNN voyait golem_glace/bebe_dragon/gargouille/yeti mais l'agent ne les jouait pas (absentes du registre codé en dur). Registre 14 → **47 troupes** (toutes les classes déployables du CNN). Ajouter une troupe = 1 ligne JSON + retrain CNN, **zéro code Python**. Existantes préservées à l'identique, obs 54 dims (checkpoint-safe, pas de re-train). `max` = borne haute optionnelle par troupe (défaut par rôle). Rôles des troupes récentes = best-guess éditables. (Le "zéro compteur" total = gros chantier deploy-grisé, à part.)

- ✅ **Capture accumulante prep_attaque** (`env_v4._save_digit_frame`) : 1 frame `prep_attaque` (armée pleine, compteurs complets, aucun grisé) sauvée par épisode dans `logs/digit_frames/` (horodatée → s'accumule), sur tout run (pas que `--test`). Source la plus riche pour le dataset digit-CNN ; `collect_digit_crops` la lit.
- ✅ **Mini-CNN chiffres — outillage Phase 2+3** : `tools/data/label_digit_crops.py` (labelisation semi-auto : crop affiché + pré-remplissage EasyOCR + Enter/num/s/u/q, rangement `<count>/`, resumable) + `tools/train/train_digit_cnn.py` (mini-CNN ~60k params, dataset folder-per-label, `--smoke` self-test). Reste côté user : collecter+labéliser les crops puis entraîner ; puis Phase 4 (intégration). Complémentaire du deploy-grisé (compteurs précis quand fiables, grisé en fallback).

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
