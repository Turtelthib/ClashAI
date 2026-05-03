# ClashAI — Roadmap & Backlog

> Dernière mise à jour : 26 avril 2026 (Session 11 — multi-agents V5, vitesse deploy, navigation failure fix)  
> Ce document centralise **toutes** les modifications, features et idées prévues.  
> On pioche ici pour construire chaque version. Rien ne se perd.

---

## V4.1 — Quick wins & Analyse post-training

> **Objectif** : corriger les irritants, analyser les premiers résultats PPO, petites améliorations sans casser l'architecture.
> **Status** : ✅ Terminé (Session 7 + run validation 192 épisodes)

### Analyse (333 épisodes)

- [x] Analyser les résultats des 333 épisodes PPO (courbe reward, convergence, stratégies)
- [x] Comparer PPO vs heuristique baseline (2.0⭐ / 75.8% / reward 364)
- [x] Identifier les patterns : PPO n'a PAS convergé (1.16⭐ / 62% moy, sous la baseline)
- [x] Diagnostic : bug critique reward combat + entropy trop élevé + pas d'imitation learning

### Bugs trouvés et corrigés (Session 7)

- [x] **BUG CRITIQUE** : `_compute_shaping_reward()` passait hero_idx dans le paramètre spell_name → les abilities n'étaient JAMAIS récompensées en combat (`environment_v4.py`)
- [x] **BUG** : `steps` toujours 0 dans le log — `_finish_episode()` ne retournait pas `step` dans info (`environment_v4.py`)
- [x] Ajuster ordre heuristique : siège avant héros (`environment_v4.py`)
- [x] Augmenter `MAX_STEPS` de 50 à 65 (`action_space.py`)
- [x] Réduire `ENTROPY_COEF` de 0.04 à 0.02 — trop d'exploration (`agent_v4.py`)
- [x] Malus sorts non utilisés en fin de combat : `-5.0` par sort restant (`reward_shaping.py` + `environment_v4.py`)

### Corrections restantes

- [x] Fix double appel YOLO dans `CombatObserver` — `count_by_class()` relançait YOLO sur la même image (`combat_observer.py`)
- [x] Investiguer les 11 épisodes à 0% destruction → taps hors zone de déploiement (fix prévu en V4.2 avec contour YOLO bâtiments)

### Imitation learning

- [x] Pré-entraîner le PPO sur les épisodes heuristiques (behavioral cloning) — `agent_v4.py` + `train_rl_v4.py`
- [x] Relancer 5 épisodes heuristiques avec les fixes V4.1 pour une nouvelle baseline
- [x] **Run V4.1 de validation** : 192 épisodes PPO + 15 BC — ⭐ moy 1.34 (vs 1.16), 2+⭐ rate 42.7% (vs 29.3%), 0⭐ rate 19.3% (vs 25%). BC fonctionne (1.76⭐ sur ep 1-25), PPO plafonne ensuite → phases rigides sont le limitant → passer à V4.2
- [x] Commande : `uv run python tools/train_rl_v4.py --pretrain 15 --episodes 200`

### Stratégie d'entraînement (décidée Session 7)

- **Petits runs (~200 ep) à chaque patch** pour valider les fixes, pas besoin de convergence
- **Gros runs (500-1000+ ep) réservés à la fin de la V4** quand toute l'architecture est stable
- Logique : à chaque refonte majeure (V4.2 fusion phases, V4.3 CNN barre) on invalide le checkpoint précédent
- Investir 10-20h de training sur une archi finale >> 2h sur chaque version intermédiaire

### Feature indépendante : demande troupes château de clan

- [x] Détecter le château de clan via YOLO bâtiments (classe `clan_castle`)
- [x] Détecter si le CC est plein (heuristique pixels blancs "PLEIN" au-dessus)
- [x] Template matching pour le bouton "Demande" dans la barre du bas (position instable)
- [x] Bouton "Envoyer" via calibrate_ui `cdc_confirmation` (popup stable centré)
- [x] Cooldown 15 min entre chaque demande
- [x] Intégrer dans `brain.py` : demander des renforts avant chaque attaque (farm + GdC)
- [ ] **Setup requis** : `--capture` pour créer `templates/clan_castle/request.png` + calibrer `cdc_confirmation`
- [ ] Optionnel futur : achat de troupes avec points de capitales / gâteaux de CC

---

## V4.2 — Refonte architecture combat

> **Objectif** : supprimer les phases rigides, rendre l'agent vraiment réactif comme un joueur humain. C'est le plus gros changement architectural depuis V3→V4.

### Bug : échec navigation → faux -50 reward

> Symptôme observé session 11 (épisode 22) :
> Matchmaking bloqué (`recherche_adversaire`) → 3 recovery échouent → `ERROR: Unable to reach enemy village`
> → l'épisode continue quand même → `_wait_for_battle_end()` voit 1-2 barres vertes (UI) → croit aux troupes mortes → surrender → **-50 reward injuste**

- [ ] Dans `_wait_for_battle_end()` : si l'écran est `village_home` ou `recherche_adversaire` au lieu de `phase_attaque`, **ne pas surrendrer** — l'attaque n'a jamais eu lieu
- [ ] Dans `reset()` : si la navigation échoue (`ERROR: Unable to reach enemy village`), marquer l'épisode comme `nav_failed=True` et ne pas lancer `step()` du tout
- [ ] Reward d'un épisode `nav_failed` = **0.0** (pas de -50) — l'agent n'est pas responsable d'un bug de navigation
- [ ] Idéalement : retry automatique de la navigation (relancer `_navigate_to_attack()`) au lieu d'abandonner après 3 essais

### ⚠️ CRITIQUE : Amélioration heuristique sorts (apprentissages Session 7)

- [x] Sorts en priorité dans `get_heuristic_sequence()` — burst rage/gel/soin avant les abilities
- [x] Skip des abilities des héros non déployés — `is_deployed()` ajouté dans `hero_ability.py`
- [x] Burst initial de sorts avant les abilities pour garantir leur utilisation même si combat court
- [x] Augmenter `MAX_STEPS_PER_EPISODE` de 65 à 80 (couvrir le wind-down)
- `MAX_COMBAT_STEPS` déjà à 40 dans le code (> 35 requis)

### ⚠️ CRITIQUE : Bug clic centre écran avant attaque

- [x] Supprimé les 3 taps `(960, 400)` parasites dans `_navigate_to()` et entre épisodes
- [x] Remplacé les 3 taps `(960, 400)` dans `_recovery_sequence()` par `(30, 540)` (bord gauche sûr)

### ✅ Fix château de clan

- [x] `ClanCastleManager` utilisait `building_detector` inexistant dans `models` — remplacé par `models` dict + `analyze_village()`
- [x] Seuil `_is_castle_full` : pixels 200→230, ratio 0.15→0.30 (moins de faux positifs)
- [x] `_close_menu` : `tap(960,400)` → `tap(30,540)` (évite d'ouvrir un bâtiment)

### Fusion des phases deploy/combat

- [x] Supprimer le `phase_indicator` binaire (0=deploy, 1=combat) — `PHASE_SIZE = 0` dans `agent_v4.py`
- [x] Toutes les 37 actions disponibles à chaque step (deploy, sorts, abilities, observe)
- [x] L'agent peut faire : golem → attendre 3s → sorcières → rage → observer → gel → ability roi, le tout en continu
- [x] Modifier `action_space.py` : le masking ne dépend plus de la phase mais des ressources restantes — `compute_action_mask` signature simplifiée, phase supprimée
- [x] Modifier `agent_v4.py` : `VECTOR_SIZE` 55→54 dims, checkpoints V4.1 incompatibles
- [x] Modifier `environment_v4.py` : supprimer la logique de transition deploy→combat — phases fusionnées, `_get_obs()` 54 dims, `step()` unifié V4.2, `reset()` force `_phase='combat'`, heuristique sans `done` intermédiaire
- `state_encoder.py` : phase_indicator déjà absent du vecteur (54 dims confirmé)

### Suppression limite de steps

- [x] `MAX_STEPS_PER_EPISODE=80` → `MAX_STEPS_SAFETY=200` (filet de sécurité uniquement) dans `action_space.py`
- [x] Fin naturelle via `_all_resources_exhausted()` : plus de troupes + sorts + abilities dans `environment_v4.py`
- [x] `step_norm` (step/80) → `time_norm` (elapsed/180s) dans `_get_obs()` — timer COC réel, VECTOR_SIZE reste 54

### YOLO continu (bâtiments + troupes à chaque step)

- [x] Faire tourner YOLO bâtiments à chaque step d'observation — `_update_combat_observation()` override dans `environment_v4.py`
- [x] Faire tourner YOLO troupes du début à la fin de l'attaque — `_combat_observer.observe()` appelé à chaque `observe`
- [x] Benchmarker le temps d'inférence — logs `⏱️ YOLO buildings: Xms | troops: Xms` à chaque observe
- [x] Fusionner `CombatObserver` et `BuildingDetector` en pipeline unifié — `_update_combat_observation()` orchestre les deux en un seul appel
- [x] Produire un état riche à chaque step — grid + features village mis à jour + combat features YOLO troupes
- [x] Détection de destruction par diff entre deux scans YOLO bâtiments successifs — `_prev_building_count` → `_buildings_destroyed_total`, +2.0 reward/bâtiment
- [x] `feature[0]` du `CombatObserver` = `buildings_remaining_ratio` (était `phase` toujours 1.0) — VECTOR_SIZE reste 54

### Amélioration zone de déploiement

- [x] Calculer le contour de la base à partir des bounding boxes YOLO bâtiments — `get_perimeter_from_buildings()` dans `deploy_zone.py`
- [x] Placer les troupes juste en dehors de ce contour — hull convexe + offset 35px, filtre UI zones
- [x] Compléter `deploy_zone.py` — nouvelle fonction ajoutée, HSV conservé en fallback V3
- [x] Côté faible déjà géré par `find_best_attack_side()` dans `state_encoder.py`, branché sur `_center_pos` dans `reset()`

### Reward shaping avancé

- [x] Destruction seconde par seconde — diff YOLO bâtiments sur chaque `observe` (+2.0/bâtiment)
- [x] Survie des héros — `compute_hero_survival_bonus()` en fin d'épisode (+5.0/héros vivant via `combat_features[4]`)
- [x] Efficacité des sorts — rage contextuelle (+2.0 si troupes > 30%), soin contextuel (+3.0 si blessés, +5.0 clutch, +0.5 si gaspillé), gel (+1.5)
- [x] Combo clutch — soin quand `hurt_ratio > 0.5` → +2.0 bonus supplémentaire


---

## V4.3 — Perception + Vitesse

> **Objectif** : remplacer le template matching + OCR par des CNN plus fiables, améliorer la zone de déploiement, et réduire les délais pour une exécution plus fluide.

### CAUSE DE BUG: Séquence de récupération
Parfois si l'agent ne voit pas le village_home directement après une attaque il panique et active
la séquence de récupération et clique partout donc il arrive toujours pas a voir le village.
Solution juste supprimer la séquence de récupération l'agent est assez intelligent maintenant
pour reconnaitre les différents états d'écran.

### CRITIQUE : Remparts dans le YOLO bâtiments

- [ ] Labéliser les remparts via LabelMe dans `dataset_walls/` (en cours)
- [ ] Entraîner `yolo_walls` dédié via `tools/train_yolo_walls.py`
- [ ] Intégrer `yolo_walls` dans `get_perimeter_from_buildings()` pour exclure les remparts du hull convexe → deploy zone bien délimitée
- [ ] Ajouter la classe `rempart` dans `state_encoder.py` CATEGORIES (canal dédié dans la grille 12×40×40)

### Amélioration zone de déploiement

- [x] Entraîner `yolo_walls_seg` → `weights/yolo_walls_seg/walls_detection.pt`
- [x] Charger le modèle dans `load_models()` (`models['yolo_walls']`)
- [x] `get_perimeter_from_walls()` dans `deploy_zone.py` — masques segmentation → contour extérieur → positions de déploiement
- [x] Branché dans `environment_v4.py` reset() : walls en priorité, building hull en fallback
- [ ] Améliorer le fallback quand le raycast échoue (< 3 positions) : utiliser des positions fixes en bord d'écran propres
- [ ] Tester sur plusieurs bases (village zoomé, dézoomé, base compacte vs étalée, thème sombre CoC)

### Vitesse d'exécution

> Bottleneck principal : délais deploy redondants.
> Par troupe : TroopFinder.select (tap + 0.15s) + DELAY_SWITCH_TROOP (0.15s doublon) + tap deploy (0.1s) + DELAY_DEPLOY (0.08s) = **0.48s/troupe** × 20 troupes = 9.6s de pure attente.

- [x] Supprimer le doublon `DELAY_SWITCH_TROOP` quand la même troupe est déjà sélectionnée (`environment_v4.py`)
- [x] `DELAY_SWITCH_TROOP` 0.15s → 0.10s
- [x] `DELAY_DEPLOY` 0.08s → 0.05s
- [x] `ADB_DELAY_TAP` 0.1s → 0.07s
- [x] `DELAY_OBSERVE` 2.5s → 2.0s
- [x] `RESCAN_EVERY_N_STEPS` 8 → 10 (moins de rescans = moins de screenshots)
- [x] Capture directe fenêtre émulateur via `mss` (~20ms vs 150ms ADB) — `clashai/perception/screen_capture.py`
- [x] `adb_screenshot()` utilise mss en priorité, ADB en fallback — transparent pour tout le code existant
- [x] Thread de perception asynchrone — `clashai/perception/perception_thread.py` (Thread capture 20fps + Thread inference YOLO+CNN)
- [x] `_update_combat_observation()` lit depuis le cache du thread (non-bloquant) → fallback V4.2 si pas de cache frais
- [x] `DELAY_OBSERVE` 2.0s → 0.15s (le thread a déjà les résultats prêts)
- [x] Thread pausé pendant la navigation, repris au début du combat

### Capture directe fenêtre émulateur (zéro ADB pour la perception)

> Au lieu de passer par ADB pour chaque frame, capturer directement la fenêtre Windows de l'émulateur.
> Exactement le même principe que faire tourner un CNN sur une vidéo ou une webcam.
> Les taps restent en ADB (seul moyen d'envoyer des inputs), seule la **perception** bypass ADB.

Comparaison latences :
- ADB screencap PNG : ~150ms
- ADB screencap raw : ~40ms
- `dxcam` / `mss` (capture fenêtre) : ~5-10ms, 60fps possible

```python
# dxcam — capture GPU directe (Windows, DXGI)
import dxcam
camera = dxcam.create()
frame = camera.grab(region=emulator_window_bbox)  # numpy array, ~5ms

# mss — alternative légère sans dépendances GPU
import mss
with mss.mss() as sct:
    frame = np.array(sct.grab(emulator_bbox))
```

Pipeline cible (comme le projet détection de feu) :
```
Thread A — Capture : grab() à 15-30fps depuis la fenêtre émulateur
Thread B — YOLO   : buildings + troops sur chaque frame capturée
Thread C — CNN    : screen classifier sur chaque frame
Agent             : lit l'état le plus récent, décide en <100ms
```

- [ ] Ajouter `dxcam` ou `mss` comme dépendance (`uv add dxcam` ou `uv add mss`)
- [ ] Créer `clashai/perception/screen_capture.py` — wrapper unifié (dxcam en priorité, mss en fallback)
- [ ] Trouver les coordonnées de la fenêtre émulateur au démarrage (via pygetwindow ou EnumWindows)
- [ ] Adapter le thread de perception pour lire depuis `screen_capture` au lieu de `adb_screenshot`
- [ ] `DELAY_OBSERVE` → 0.1s (le thread tourne déjà en fond, observe = juste lire le buffer)
- [ ] Objectif : perception temps réel 15-30fps, même réactivité que le projet détection feu

### Debug overlay (visualisation de ce que l'agent voit)

> But : faciliter le debug sans attendre le dashboard V5.
> À chaque action `observe`, générer une image annotée dans `logs/debug/` montrant
> exactement l'état perçu par l'agent.

- [ ] Image annotée par observe : bâtiments YOLO (bbox colorés par catégorie), hull convexe de la zone de déploiement, positions de déploiement (points numérotés), cluster de troupes YOLO, sorts restants (overlay texte)
- [ ] Réutilise la logique de `save_deploy_debug_image()` déjà en place
- [ ] Option `--debug-overlay` dans `train_rl_v4.py` pour activer/désactiver (évite de ralentir le training normal)
Pour cette étape mettre les fichiers dans le dossier logs et dedans crée un dossier nommé episode_NUMERO_EPISODE/ et mettre tout les fichiers de l'épisode en question dedans et ainsi de suite chaque épisode = un dossier et pas tout en vrac

### Thread de perception asynchrone (gros gain de réactivité)

> Actuellement : agent décide → attend 2s → screenshot → YOLO 230ms → décide → ...
> Avec async : un thread tourne en fond en continu, l'agent lit l'état le plus récent instantanément.

Architecture cible :
- `PerceptionThread` (daemon thread) : capture screenshot toutes les ~0.5s + YOLO buildings + troops en continu
- Résultat stocké dans un buffer partagé protégé par `threading.Lock`
- L'action `observe` ne fait plus que lire ce buffer (quasi instantané)
- Pendant les taps de deploy, le thread a déjà calculé la prochaine observation
- `DELAY_OBSERVE` → 0s (ou 0.1s pour laisser le temps au thread de refresher)

Bénéfices :
- L'agent voit le champ de bataille se mettre à jour **pendant** qu'il déploie
- Réactivité proche d'un humain (décision toutes les 0.5s au lieu de 2s+)
- Plus de temps "mort" à attendre un screenshot bloquant

- [ ] Créer `clashai/perception/perception_thread.py` — thread daemon avec buffer partagé
- [ ] Modifier `_update_combat_observation()` pour lire depuis le buffer au lieu de bloquer
- [ ] Adapter `_execute_observe()` : `DELAY_OBSERVE` → 0.1s (juste pour que le PPO ne spam pas)
- [ ] Gérer proprement le stop du thread en fin d'épisode (`episode_done` event)

### CNN classification icônes troupes (remplace template matching)

- [ ] Entraîner un classifieur (ResNet18 fine-tuné) sur des crops de chaque slot de la barre
- [ ] Classes : chaque type de troupe + héros + sorts + siège
- [ ] Capturer un dataset de screenshots de la barre pendant les combats
- [ ] Annoter les crops (type de troupe par slot)
- [ ] Détecter l'état des abilities : grisé (mort), brillant (prêt), utilisé (cooldown)
- [ ] Remplacer le template matching dans `troop_finder.py`

### CNN lecture compteurs (remplace OCR)

- [ ] Entraîner un petit CNN type MNIST sur les compteurs (x2, x3, x11...)
- [ ] Capturer des crops de chaque compteur et annoter les chiffres
- [ ] Remplacer l'OCR dans `troop_counter.py`
- [ ] Tester la fiabilité sur tous les cas (x1 pas affiché, x10+, etc.)

---

## V5 — Architecture multi-agents

> **Objectif** : formaliser l'architecture en orchestrateur + sous-agents spécialisés.
> La structure existe déjà en partie — il s'agit de la formaliser, d'ajouter les agents manquants, et de rendre l'orchestrateur plus intelligent.

### État actuel (déjà présent, non formalisé)

| Sous-agent | Fichier actuel | Type |
|---|---|---|
| Orchestrateur | `brain.py` | Heuristique |
| Attaque (farm) | `combat/environment_v4.py` + PPO | RL (PPO) |
| Guerre de clan | `navigation/gdc_navigator.py` | Heuristique |
| Chat clan | `social/clan_chat_monitor.py` | Règles |
| Château de clan | `social/clan_castle.py` | Règles |

### Interface commune à créer

- [ ] Définir une classe de base `BaseAgent` avec interface `run()`, `can_run()`, `priority()`
- [ ] L'orchestrateur `brain.py` interroge chaque agent avec `can_run()` et délègue
- [ ] Chaque agent est isolé : son propre état, ses propres actions ADB, ses propres décisions
- [ ] Système de priorités : attaque > GdC > jeux de clan > gestion village > idle

### Nouveaux agents à créer

**Agent jeux de clan** (`clashai/clan_games/`)
- Détecter si des jeux de clan sont actifs (CNN écran)
- Identifier les tâches disponibles (template matching sur les cartes)
- Exécuter les tâches répétitives (attaque, don de troupes, etc.)
- Type : règles + heuristiques (pas besoin de RL)

**Agent gestion village** (`clashai/village/`)
- Détecter les constructeurs libres (template matching)
- Queue de priorité : améliorer selon un ordre défini (murs → défenses → ressources)
- Détecter les laboratoires libres → lancer une recherche
- Collecter les ressources (mines, coffres)
- Type : règles + queue de priorité (pas besoin de RL)

### Orchestrateur amélioré

- [ ] Boucle principale `brain.py` : checker chaque agent toutes les N minutes selon un schedule
- [ ] Gestion des cooldowns : ne pas relancer un agent qui vient de tourner
- [ ] Logging centralisé : chaque agent log ses actions dans un fichier commun

### Dashboard web temps réel (V5)

> Une app web légère (FastAPI + HTML/JS) qui tourne en parallèle du brain.
> Se met à jour en temps réel via WebSocket ou polling JSON.

- [ ] Page principale : état de chaque sous-agent (actif / idle / cooldown), dernière action, dernière attaque
- [ ] Onglet training : courbe reward/étoiles en temps réel, dernière image de debug overlay, PPO stats (value_loss, entropy, policy_loss)
- [ ] Onglet replay : les 5 dernières images de debug overlay par épisode (timeline visuelle de l'attaque)
- [ ] Onglet village : état des constructeurs, labo, ressources (lecture depuis les logs agents)
- [ ] Accessible depuis le réseau local (pratique pour suivre depuis un autre écran)
- [ ] **Page dédiée "Vision agent"** : flux vidéo en temps réel de ce que l'agent perçoit, avec overlays annotés (bboxes YOLO buildings colorées par classe, masques de segmentation remparts, zone de déploiement hull convexe + points numérotés, positions troupes YOLO, cluster principal, sorts restants en overlay texte). Alimentée par le thread de capture directe fenêtre émulateur — latence ~100ms entre le jeu réel et la page dashboard.

---

## V6 — Nouvelles capacités de combat

> **Objectif** : étendre les capacités de l'agent au-delà du combat pur.

### Caméra / scroll

- [ ] Ajouter des actions scroll/pan pour suivre les troupes hors écran
- [ ] Problème actuel : les troupes sortent de l'écran → YOLO ne les voit plus → retraite déclenchée trop tôt
- [ ] Nouvelles actions : scroll_left, scroll_right, scroll_up, scroll_down, center_on_troops
- [ ] Le vecteur d'observation doit inclure la position de la caméra

### Multi-compo

- [ ] Supporter d'autres armées que GoWitch (LavaLoon, Hybrid, QC, etc.)
- [ ] Adapter le `TroopManager` pour des rôles différents selon la compo
- [ ] Nouveaux templates/CNN pour les troupes non GoWitch
- [ ] L'agent doit apprendre des stratégies différentes selon la compo

### Équipements héros

- [ ] Détecter les équipements actifs sur chaque héros
- [ ] Adapter la stratégie selon l'équipement (ex: bouclier barbare vs cape d'invisibilité)
- [ ] Enrichir le vecteur d'observation avec l'info équipement

### Self-play / curriculum

- [ ] Entraîner sur des bases de difficulté croissante
- [ ] Commencer par des bases faciles (HDV 10-11), monter progressivement
- [ ] Potentiellement : utiliser le matchmaking du jeu pour trouver des bases adaptées

### Gestion du village
- [] amélioration troupes/héros/batiments



### OBJECTIF FINAL

- UNE IA AUTONOME INTELLIGENTE QUI JOUE COMME UN HUMAIN.

---

## Backlog — Idées non planifiées

> Idées intéressantes mais pas encore assignées à une version. On pioche ici quand on a du temps.

- [ ] Dashboard web temps réel (stats, replay, courbes de training)
- [ ] Replay vidéo des attaques (enregistrer ADB screen pendant le combat)
- [ ] Mode "coaching" : l'IA analyse une attaque humaine et donne des conseils
- [ ] Détection du type de base adverse (war base, farming base, troll base)
- [ ] Gestion automatique des boucliers et de la connexion
- [ ] Optimisation de la compo d'armée (quelle armée construire selon la base cible)
- [ ] Multi-compte : gérer plusieurs comptes sur plusieurs émulateurs
- [ ] Ligue farming automatique (monter/descendre en ligue selon l'objectif)

---

## Historique des versions

| Version | Status | Résumé |
|---------|--------|--------|
| V1 | ✅ Terminé | Une seule décision par attaque |
| V2 | ✅ Terminé | Améliorations intermédiaires |
| V3 | ✅ Terminé | Déploiement séquentiel + combat réactif (289 actions, 1.2M params) |
| V4.0 | ✅ Terminé | Action space simplifié 37 actions + YOLO troupes (Session 6) |
| V4.1 | ✅ Terminé | Fix bugs critiques + BC + run validation 192 ep — CC troops non fonctionnel (Session 7) |
| V4.2 | ✅ Terminé | Fusion phases, YOLO continu, zone deploy murs+bâtiments (segmentation), reward shaping, logs pro (Session 8-11) |
| V4.2.1 | ✅ Fix | PPO value loss, BC loss, ability deadlock, deploy zone walls seg (Session 10-11) |
| V4.3 | 📋 Planifié | CNN barre de troupes, vitesse d'exécution, deploy zone remparts |
| V5 | 💡 Vision | Architecture multi-agents (orchestrateur + spécialisés) |
| V6 | 💡 Vision | Caméra, multi-compo, équipements, self-play |
| V END | 🎯 Objectif | Jouer comme un humain