# PROJECT_MAP — ClashAI

> Cartographie du code. Consultée en premier pour trouver un fichier ou comprendre l'architecture.
> **À régénérer après tout déplacement/création/suppression de fichier** (skill `project-mapper`).
> Généré le 5 août 2026 — 172 fichiers `.py` (144 dans `clashai/`, 28 dans `tools/`).

## Sens des dépendances

```
brain → agents → {combat, navigation, social} → perception → adb
```

Vérifié : aucune couche basse n'importe `brain` ni `agents`. Deux inversions connues à surveiller —
`perception/perception_thread.py` importe `navigation.game_loop`, et `config/perception.py` importe `perception`/`navigation` en late-import.

---

## `src/clashai/` — le package

```
clashai/
├── paths.py                      # SSOT filesystem : racine trouvée via le marqueur pyproject.toml
├── exceptions.py                 # Hiérarchie d'exceptions (⚠️ 11 des 13 classes ne sont ni levées ni attrapées)
├── _core_exceptions.py           # ClashAIError racine, isolé pour casser un cycle d'import
│
├── adb/                          # Couche la plus basse : parle à l'émulateur
│   ├── client.py                 # ADBClient : tap/swipe/keyevent/screencap. Singleton via get_client()
│   └── exceptions.py             # ADBNotFoundError / ADBTimeoutError (les 2 seules exceptions vivantes)
│
├── config/                       # Constantes centralisées (adoption incomplète, cf. AUDIT §4.2)
│   ├── screen.py                 # Résolution ADB/écran — SSOT des 1920×1080
│   ├── timing.py                 # Tous les DELAY_* et WAIT_*
│   ├── perception.py             # Seuils de confiance YOLO/CNN (⚠️ doublonné par game_loop/constants.py)
│   ├── rl.py                     # Hyperparamètres RL, HERO_NAMES (⚠️ TOTAL_ACTIONS_V4=37 périmé, non consommé)
│   ├── brain.py                  # Config de la boucle brain (⚠️ PRIORITY_* morts)
│   ├── window.py                 # Détection/géométrie de la fenêtre émulateur
│   └── logging.py                # Logger Rich + rotation (⚠️ get_logger() appelé NULLE PART — 661 print à la place)
│
├── perception/                   # Vision : de l'image aux faits
│   ├── perception_thread.py      # Thread d'inférence continu + cache d'état partagé (cœur temps réel)
│   ├── events.py                 # PerceptionEventBus : abonnement aux changements d'état
│   ├── inference_lock.py         # INFERENCE_LOCK : sérialise les appels GPU entre threads
│   ├── screen_classifier.py      # CNN : à quel écran du jeu on est
│   ├── digit_reader.py           # Segmentation + CNN 0-9 : lit les compteurs de troupes (SSOT)
│   ├── troop_bar_detector.py     # YOLO : la barre de troupes en bas de l'écran de combat
│   ├── troop_finder.py           # Localise un slot de troupe donné dans la barre
│   ├── troop_detector.py         # YOLO : troupes sur le terrain (⚠️ hardcode 13 classes malgré troop_registry)
│   ├── troop_counter.py          # Lecture des compteurs par template matching (voie historique)
│   ├── building_detector.py      # ⚠️ MORT ET CASSÉ : crash à l'import (runs/ inexistant), zéro appelant
│   ├── coord_utils.py            # ImageScaler : conversion écran↔ADB (⚠️ contourné dans 6+ modules)
│   ├── debug_overlay.py          # Annotation d'images de debug (⚠️ vocabulaire de classes périmé)
│   ├── test_run_capture.py       # ⚠️ Script de debug DANS le package de prod (livré dans le wheel)
│   ├── screen_capture/           # Capture d'écran multi-backend
│   │   ├── capture.py            # Chaîne WGC > PrintWindow > dxcam > mss > ADB. Singleton get_capture()
│   │   ├── gdi_capture.py        # Backend PrintWindow/GDI
│   │   ├── window_detect.py      # Trouve la fenêtre de l'émulateur
│   │   └── normalize.py          # Normalisation des frames
│   ├── deploy/                   # Où peut-on déployer des troupes
│   │   ├── boundary.py           # Périmètre de la base
│   │   ├── positions.py          # Les 20 positions de deploy sur le périmètre
│   │   ├── yolo_zone.py          # Zone déployable par segmentation des murs
│   │   └── debug.py              # Visualisation de la zone
│   └── reward_reader/            # Lecture de l'écran de fin de combat
│       ├── results.py            # Orchestration : étoiles + pourcentage
│       ├── stars.py              # Nombre d'étoiles
│       ├── percentage.py         # % de destruction (OCR chiffres)
│       └── green.py              # Détection des barres vertes
│
├── combat/                       # Le combat : RL + heuristique
│   ├── troop_registry.py         # ⭐ SSOT data-driven : lit configs/troops.json → troupes, rôles, sorts
│   ├── action_space.py           # ⭐ Espace d'action DÉRIVÉ (51 actions) : encode/decode/mask
│   ├── reward_shaping.py         # Constantes et fonctions de reward (⚠️ compute_final_reward() morte)
│   ├── episode_lifecycle.py      # Début/fin d'épisode, détection de fin de bataille, abandon
│   ├── episode_runner.py         # SSOT : lance un épisode complet (utilisé par brain ET agents)
│   ├── troop_manager.py          # Inventaire des troupes, sélection de slot, grisé
│   ├── state_encoder.py          # Shim de compat → encoder/
│   ├── hero_ability.py           # Shim de compat → hero/
│   ├── environment_v4/           # ⚠️ Env Gym V4 = 7 mixins sur un état mutable partagé (cf. AUDIT §3.5)
│   │   ├── env.py                # ClashEnvV4(7 mixins, ClashEnvV3) — l'assemblage
│   │   ├── core.py               # reset/step/fin d'épisode/seed des compteurs (6 responsabilités)
│   │   ├── observe.py            # Construction de l'observation + re-lecture live des compteurs
│   │   ├── actions.py            # Exécution des actions (deploy, sorts, capas)
│   │   ├── reward.py             # Reward par step
│   │   ├── heuristic.py          # Politique heuristique (prof du BC)
│   │   ├── observation.py        # Définition de l'espace d'observation
│   │   └── capture.py            # Sauvegarde des frames d'épisode
│   ├── encoder/                  # Image → tenseur pour le réseau
│   │   ├── grid.py               # Grille 40×40 × 12 canaux
│   │   ├── features.py           # Vecteur de features (68 dims)
│   │   ├── attack_side.py        # Choix géométrique du côté d'attaque
│   │   └── constants.py          # CATEGORIES + DEFENSE_STATS (⚠️ bug canon_double/double_canon)
│   ├── agent_v4/                 # L'agent RL courant
│   │   ├── agent.py              # PPO : act/update/save/load
│   │   ├── network.py            # CNN grille + MLP vecteur → actor/critic
│   │   ├── buffer.py             # Rollout buffer
│   │   ├── bc.py                 # Behavior cloning depuis l'heuristique
│   │   └── constants.py          # VECTOR_SIZE=68, hyperparamètres PPO
│   ├── spell_caster/             # Lancer les sorts au bon endroit
│   │   ├── caster.py             # Ciblage par type de sort (cluster/heal/defense)
│   │   ├── clustering.py         # ⚠️ BFS dupliqué avec combat_observer/
│   │   ├── health_bars.py        # ⚠️ Détection HSV dupliquée avec combat_observer/
│   │   └── constants.py          # ⚠️ Seuils HSV DIVERGENTS de combat_observer/
│   ├── combat_observer/          # État du champ de bataille (troupes vivantes, héros)
│   │   ├── observer.py           # Agrégation de l'observation combat
│   │   ├── clustering.py         # ⚠️ voir spell_caster/clustering.py
│   │   ├── health_bars.py        # ⚠️ voir spell_caster/health_bars.py
│   │   └── constants.py          # ⚠️ voir spell_caster/constants.py
│   ├── hero/                     # Héros et capacités
│   │   ├── manager.py            # Détection des capas dispo (grisé), déclenchement
│   │   ├── cli.py                # Outil de scan manuel
│   │   └── constants.py          # Positions/noms des héros
│   └── legacy/                   # ⚠️ MAL NOMMÉ : code VIVANT sur le chemin critique
│       ├── environment.py        # ClashEnvV3, 1472 L — ClashEnvV4 en HÉRITE, s'exécute à chaque attaque
│       ├── agent.py              # PPO V3 + TROOP_TYPES/TROOP_NAME_TO_IDX (importés par 7 modules v4)
│       └── heuristic_v3.py       # Heuristique V3
│
├── navigation/                   # Se déplacer dans l'UI du jeu
│   ├── calibrate_ui.py           # Calibration des positions UI + get_position() (⚠️ forké dans tools/setup/)
│   ├── zoom_control.py           # Contrôle du zoom
│   ├── gdc_navigator.py          # Shim de compat → gdc/
│   ├── game_loop/                # Boucle de jeu bas niveau
│   │   ├── models.py             # load_models() : charge les 5 modèles, produit le dict `models`
│   │   ├── controller.py         # run_live() : boucle autonome
│   │   ├── analysis.py           # analyze_village() / classify_screen()
│   │   ├── adb_io.py             # ⭐ adb_screenshot() canonique (réutilisé par gdc/ et chat/)
│   │   ├── constants.py          # ⚠️ Réhardcode 4 seuils déjà dans config/perception.py
│   │   └── models.py             # (cf. ci-dessus)
│   └── gdc/                      # Guerre de clans
│       ├── navigator.py          # Navigation dans la carte de guerre, choix de cible
│       ├── orchestrator.py       # Enchaînement complet d'une attaque de guerre
│       ├── ocr.py                # Lecture des numéros de cible
│       ├── adb_io.py             # Wrappers ADB locaux
│       └── constants.py          # ⭐ _get_ui_pos() = le seul embryon de point d'accès UI unique
│
├── social/                       # Chat et château de clan
│   ├── clan_castle.py            # Demander/recevoir des troupes du CC
│   ├── clan_chat_monitor.py      # Shim de compat → chat/
│   └── chat/
│       ├── monitor.py            # Lecture du chat de clan
│       ├── parser.py             # ⭐ 100% pur : timestamps + commandes (le meilleur candidat aux tests)
│       ├── ocr.py                # Moteur OCR (easyocr/tesseract)
│       ├── adb_io.py             # Wrappers ADB locaux
│       └── constants.py          # Zones de lecture du chat
│
├── agents/                       # Couche agents (V5.1)
│   ├── base.py                   # BaseAgent : can_run/run/is_ready + RunState + AgentResult
│   ├── scheduler.py              # AgentScheduler : priorité + cooldown, status() sérialisable
│   ├── world.py                  # build_world() : le dict d'état partagé par tous les agents
│   ├── combat_agent.py           # Attaque de farm
│   ├── gdc_agent.py              # Attaque de guerre (+ file enqueue_target)
│   ├── chat_agent.py             # Lecture/réponse au chat de clan
│   └── clan_castle_agent.py      # Demande de troupes
│
└── brain/                        # Orchestration haut niveau
    ├── app.py                    # ClashBrain = core + loop + navigation
    ├── core.py                   # Composition root : charge les modèles, câble agents/scheduler/Brain
    ├── loop.py                   # Boucle principale : world → brain.decide → scheduler.run → stats
    ├── interface.py              # ⭐ Brain (ABC) + HeuristicBrain — le seam du futur LocalLLMBrain
    ├── navigation.py             # Retour à un état connu (⚠️ cascade de 8 elif + coordonnées en dur)
    └── __main__.py               # `python -m clashai.brain --mode farm|gdc`
```

## `src/tools/` — outillage (hors package installé)

```
tools/
├── train/
│   ├── train_rl_v4.py            # Entraînement PPO de l'agent V4 (le script principal)
│   ├── train_rl.py               # Entraînement V3 (legacy)
│   ├── train_digit_cnn.py        # CNN chiffres 0-9
│   ├── train_screen_cnn.py       # CNN classification d'écran
│   ├── train_cnn.py              # CNN bâtiments
│   ├── train_yolo_troops.py      # YOLO troupes terrain
│   ├── train_yolo_troop_bar.py   # YOLO barre de troupes
│   ├── train_yolo_buildings.py   # YOLO bâtiments
│   ├── train_yolo_walls_seg.py   # Segmentation des murs
│   ├── kaggle_train_yolo_troops.py # ⭐ Script GÉNÉRIQUE Kaggle (lit n'importe quel dataset Roboflow)
│   └── compare_baseline.py       # Compare un run à une baseline archivée
├── data/
│   ├── collect_digit_crops.py    # Collecte de crops de chiffres
│   ├── label_digit_crops.py      # Labelisation semi-auto
│   ├── build_digit_singles.py    # Nombre entier → chiffres individuels
│   ├── capture_combat.py         # Capture de frames de combat
│   ├── capture_screen_state.py   # Capture d'écrans pour le CNN d'état
│   ├── capture_troop_bar.py      # Capture de la barre de troupes
│   ├── generate_crops.py         # Génération de crops
│   ├── prepare_dataset.py        # Préparation de dataset YOLO
│   ├── convert_labelme.py        # LabelMe → YOLO
│   └── convert_labelme_walls.py  # LabelMe → YOLO (murs)
├── debug/                        # ⚠️ Les test_*.py ici sont des scripts, PAS des tests pytest
│   ├── run_test.py               # Run de test annoté complet
│   ├── test_deploy.py            # ⚠️ Aucun if __name__ : s'exécute à l'import (GPU + émulateur)
│   ├── test_deploy_zone.py       # Visualisation de la zone de deploy
│   ├── test_screen_capture.py    # Test des backends de capture
│   ├── test_gpu.py               # ⚠️ torch.cuda au niveau module
│   ├── detect_troop_bar.py       # ⚠️ sys.exit() sans import sys (NameError)
│   ├── debug_screen_cnn.py       # Debug du CNN d'écran
│   └── inspect_emulator_window.py # Géométrie de la fenêtre
├── setup/
│   └── calibrate_ui.py           # ⚠️ FORK divergent de clashai/navigation/calibrate_ui.py
└── _archive/                     # ⚠️ Mort : zéro import dans tout le dépôt
```

## Données et config

```
configs/
├── troops.json                   # ⭐ SSOT troupes+sorts (63 entrées) — lu par troop_registry
├── ui_positions.json             # Positions UI calibrées (22 clés) — remplacé par le CNN UI en V5.2
├── coc.yaml                      # Dataset YOLO bâtiments (train uniquement)
├── coc_troops.yaml               # ⚠️ ORPHELIN : aucun lecteur, chemin absolu d'une autre machine
├── coc_walls.yaml                # ⚠️ ORPHELIN : aucun lecteur
└── logs/theme.yaml               # Thème Rich (⚠️ non versionné : .gitignore `logs/` le masque)

weights/                          # Gitignoré. ~250 Mo de poids orphelins à purger
├── classes.json                  # ⭐ Les 46 classes de bâtiments réellement émises par le CNN
├── best.pt                       # YOLO bâtiments
├── building_cnn.pth              # CNN classification bâtiment
├── digit_cnn.pt                  # CNN chiffres 0-9
├── yolo_troops.pt                # YOLO troupes terrain (51 classes, mAP50 0.82)
├── classification/               # CNN état d'écran
├── yolo_troupes_barre/           # YOLO barre + model_artifacts.json (classes troupes/sorts)
├── yolo_walls_seg/               # Segmentation murs
├── rl/                           # Checkpoints RL courants
└── baselines/v4.4-ppo-350ep/     # Baseline archivée (68 dims / 51 actions — compatible)

docs/
├── ROADMAP.md                    # Ce qui reste à faire
├── CHANGELOG.md                  # Ce qui est fait, par version
├── TROUBLESHOOTING.md            # Blocs de fix détaillés
├── AUDIT_2026-08-05.md           # Audit projet (temporaire, à dissoudre dans ROADMAP/TROUBLESHOOTING)
├── baselines.md                  # Résultats des runs de référence
├── dashboard_brief.md            # Spec du dashboard V6
└── vendor/                       # Docs tierces (non-skills)
```

## Points d'entrée

```bash
uv run python -m clashai.brain --mode farm      # boucle autonome (entrypoint déclaré : clashai-brain)
uv run python -m clashai.navigation.gdc         # guerre de clans
uv run python -m clashai.social.chat            # monitoring du chat
uv run python -m clashai.combat.spell_caster    # test du lanceur de sorts
uv run python -m clashai.combat.combat_observer # test de l'observateur
uv run python -m clashai.combat.agent_v4        # test de l'agent RL
uv run python src/tools/train/train_rl_v4.py    # entraînement (préfixe src/ obligatoire)
```

## Repères chiffrés (vérifiés le 5 août 2026)

| Quoi | Valeur | Source de vérité |
|---|---|---|
| Sorts actifs | **17** | `action_space.NUM_SPELLS` (registre ∩ classes CNN) |
| Dimensions d'observation | **68** | `combat/agent_v4/constants.VECTOR_SIZE` |
| Actions | **51** | `action_space.TOTAL_ACTIONS` |
| Entrées `troops.json` | 63 | `configs/troops.json` |
| Classes bâtiments | 46 | `weights/classes.json` |
| Modules importables | 137 / 138 | seul `perception/building_detector.py` échoue |
