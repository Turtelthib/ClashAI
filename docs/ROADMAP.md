# ClashAI — Roadmap & Backlog

> Dernière mise à jour : 18 avril 2026 (Session 8 — V4.2 fusion phases environment_v4.py)  
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

## V4.3 — Perception améliorée (barre de troupes)

> **Objectif** : remplacer le template matching + OCR actuel par des CNN plus fiables pour lire la barre de troupes.

CRITIQUE: les remparts fausse les resultats de deploy_zone donc faut les labéliser via labelme. Et les rajouter au cnn batiment

### CNN classification icônes troupes

- [ ] Entraîner un classifieur (ResNet18 fine-tuné) sur des crops de chaque slot de la barre
- [ ] Classes : chaque type de troupe + héros + sorts + siège
- [ ] Capturer un dataset de screenshots de la barre pendant les combats
- [ ] Annoter les crops (type de troupe par slot)
- [ ] Détecter l'état des abilities : grisé (mort), brillant (prêt), utilisé (cooldown)
- [ ] Remplacer le template matching dans `troop_finder.py`

### CNN lecture compteurs

- [ ] Entraîner un petit CNN type MNIST sur les compteurs (x2, x3, x11...)
- [ ] Capturer des crops de chaque compteur et annoter les chiffres
- [ ] Remplacer l'OCR dans `troop_counter.py`
- [ ] Tester la fiabilité sur tous les cas (x1 pas affiché, x10+, etc.)

---

## V5 — Nouvelles capacités

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
| V4.2 | 🔄 En cours | Refonte combat (fusion phases ✅, YOLO continu ✅, zone deploy) |
| V4.3 | 📋 Planifié | CNN barre de troupes (remplace template matching + OCR) |
| V5 | 💡 Vision | Caméra, multi-compo, équipements, self-play |
| V END | 🎯 Objectif | Jouer comme un humain