# ClashAI — Roadmap & Backlog

> Dernière mise à jour : 15 avril 2026 (Session 7)  
> Ce document centralise **toutes** les modifications, features et idées prévues.  
> On pioche ici pour construire chaque version. Rien ne se perd.

---

## V4.1 — Quick wins & Analyse post-training

> **Objectif** : corriger les irritants, analyser les premiers résultats PPO, petites améliorations sans casser l'architecture.

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
- [ ] Relancer 5 épisodes heuristiques avec les fixes V4.1 pour une nouvelle baseline
- [ ] **Run V4.1 de validation** : 200 épisodes (pas besoin de convergence, juste valider que les fixes marchent)
- [ ] Commande : `uv run python tools/train_rl_v4.py --pretrain 15 --episodes 200`

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

⚠️ check demande de chateau de clan, ne marche pas dans la version actuelle..

### ⚠️ CRITIQUE : Amélioration heuristique sorts (apprentissages Session 7)

- [ ] Sorts en priorité dans `get_heuristic_sequence()` (malus -5 si non utilisés → l'heuristique en laisse souvent)
- [ ] Skip des abilities des héros non déployés (championne, prince_gargouille parfois absents)
- [ ] Burst initial de sorts avant les abilities pour garantir leur utilisation même si combat court
- [ ] Augmenter `MAX_COMBAT_STEPS` de 20 à 35 (temps de placer tous les sorts)
- [ ] Augmenter `MAX_STEPS_PER_EPISODE` de 65 à 80 (couvrir le wind-down)

### ⚠️ CRITIQUE : Bug clic centre écran avant attaque

- [ ] L'agent clique au centre de l'écran avant de cliquer sur le bouton attaquer (navigation V3 héritée)
- [ ] Parfois ça ouvre un menu inconnu → boucle infinie → épisode perdu
- [ ] Investiguer dans `environment.py` (V3 parent) le flow `reset()` / navigation vers `phase_attaque`
- [ ] Fix : soit supprimer le clic parasite, soit ajouter une détection des menus inconnus + fermeture

### Fusion des phases deploy/combat

- [ ] Supprimer le `phase_indicator` binaire (0=deploy, 1=combat)
- [ ] Toutes les 37 actions disponibles à chaque step (deploy, sorts, abilities, observe)
- [ ] L'agent peut faire : golem → attendre 3s → sorcières → rage → observer → gel → ability roi, le tout en continu
- [ ] Modifier `action_space.py` : le masking ne dépend plus de la phase mais des ressources restantes
- [ ] Modifier `environment_v4.py` : supprimer la logique de transition deploy→combat
- [ ] Modifier `state_encoder.py` : retirer phase_indicator du vecteur obs (55→54 dims, ou remplacer par autre chose)

### Suppression limite de steps

- [ ] Remplacer `MAX_STEPS = 50` par une condition de fin naturelle
- [ ] Conditions de fin : plus de troupes ET plus de sorts ET plus d'abilities ET (troupes mortes OU timer 3min)
- [ ] Garder un plafond de sécurité très haut (~200 steps) pour éviter les boucles infinies
- [ ] Adapter le `step_norm` dans le vecteur d'observation (normaliser par rapport au temps réel plutôt qu'au nombre de steps)

### YOLO continu (bâtiments + troupes à chaque step)

- [ ] Faire tourner YOLO bâtiments à chaque step d'observation (pas seulement au scan initial)
- [ ] Faire tourner YOLO troupes du début à la fin de l'attaque (pas seulement en phase combat)
- [ ] Benchmarker le temps d'inférence sur RTX 5070 (objectif : <100ms pour les deux modèles)
- [ ] Fusionner `CombatObserver` et `BuildingDetector` en un pipeline de perception unifié
- [ ] Produire un état riche à chaque step : positions troupes, positions bâtiments restants, défenses proches
- [ ] Détection de destruction par diff entre deux scans YOLO bâtiments successifs
- [ ] Le vecteur d'observation grossit mais les features sont bien plus riches pour le PPO

### Amélioration zone de déploiement

- [ ] Calculer le contour de la base à partir des bounding boxes YOLO bâtiments
- [ ] Placer les troupes juste en dehors de ce contour (plus de taps en zone rouge)
- [ ] Compléter ou remplacer `deploy_zone.py` actuel
- [ ] Bonus : identifier les côtés faibles (moins de défenses) pour orienter l'attaque

### Reward shaping avancé

- [ ] Destruction seconde par seconde (diff YOLO bâtiments entre screenshots)
- [ ] Survie des héros (YOLO héros visible en fin de combat = bonus)
- [ ] Efficacité des sorts (troupes détectées dans le rayon du sort = bonus, sort dans le vide = malus)
- [ ] Reward pour combos intelligentes (gel sur tour inferno, soin quand HP bas)


---

## V4.3 — Perception améliorée (barre de troupes)

> **Objectif** : remplacer le template matching + OCR actuel par des CNN plus fiables pour lire la barre de troupes.

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

### Détection Prince Gargouille fiable

- [ ] Le Prince Gargouille n'est pas toujours détecté par template matching
- [ ] Le CNN icônes devrait régler ce problème (c'est une classe comme une autre)

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
| V4.1 | 🔧 En cours | Fix bug critique reward combat + analyse 333 épisodes (Session 7) |
| V4.2 | 📋 Planifié | Refonte combat (fusion phases, YOLO continu, zone deploy) |
| V4.3 | 📋 Planifié | CNN barre de troupes (remplace template matching + OCR) |
| V5 | 💡 Vision | Caméra, multi-compo, équipements, self-play |
| V END | 🎯 Objectif | Jouer comme un humain