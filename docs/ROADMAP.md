# ClashAI — Roadmap

> **OBJECTIF FINAL** : une IA autonome intelligente qui joue comme un humain — joue, gère, recrute, s'améliore seule, et qu'on **pilote en langage naturel via le chat clan** (cerveau LLM local orchestrant des sous-agents).

**Statut** : `[ ]` à faire · `[x]` fait (détail → [CHANGELOG](CHANGELOG.md)) · 🔧 bug documenté → [TROUBLESHOOTING](TROUBLESHOOTING.md)
**Mise à jour** : 14 juin 2026 (Session 14 — V5.1 foundation multi-agents)

📂 **Ce doc** = ce qui reste à faire. · ✅ Fait → [CHANGELOG.md](CHANGELOG.md) · 🔧 Fix détaillés → [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

## Sommaire

- [📊 État des versions](#-état-des-versions)
- [🚀 En cours](#-en-cours)
  - [V4.4 — Polish perception](#v44--polish-perception)
  - [V5.1 — Foundation multi-agents](#v51--foundation-multi-agents)
  - [V5.0 — Mode live (phases optionnelles)](#v50--mode-live-phases-optionnelles)
- [📅 À venir](#-à-venir)
  - [V5.2 — Nouveaux agents + orchestrateur](#v52--nouveaux-agents--orchestrateur)
  - [V5.3 — Dashboard web temps réel](#v53--dashboard-web-temps-réel)
- [🔮 Vision long terme](#-vision-long-terme)
  - [V6 — Combat avancé](#v6--combat-avancé)
  - [Cerveau LLM local (coach + parole + RAG)](#cerveau-llm-local-coach--parole--rag)
- [🗃️ Backlog (non planifié)](#️-backlog-non-planifié)

---

## 📊 État des versions

| Version | Statut | Résumé |
|---|---|---|
| V1–V4.0 | ✅ | Décision unique → 37 actions + YOLO troupes (voir CHANGELOG) |
| V4.1 | ✅ | Fix bugs critiques + BC + run validation 192 ep |
| V4.2 | ✅ | Fusion phases, YOLO continu, zone deploy, reward shaping |
| V4.3 | ✅ | YOLO barre troupes, perception async, WGC, mode `--test` |
| V4.4 | 🔄 **en cours** | Polish perception : mini-CNN chiffres + gros run final |
| V5.0 | ✅ Ph.1-2 | Push pipeline WGC → PerceptionThread (Ph.3-4 optionnelles) |
| Refacto | ✅ | src/ layout + 13 splits (0 fichier >500L hors legacy) |
| V5.1 | 🔄 **en cours** | Foundation multi-agents : 4 agents faits, reste interface Brain |
| V5.2 | 💡 | Nouveaux agents (jeux clan, village) + orchestrateur + CNN options bar |
| V5.3 | 💡 | Dashboard web temps réel (FastAPI + WebSocket) |
| V6+ | 💡 | Combat avancé, cerveau LLM local, multi-compte |
| V END | 🎯 | IA autonome complète |

---

## 🚀 En cours

### V4.4 — Polish perception

> Clore les derniers irritants de perception, puis un gros run de validation avant la suite.

- [ ] **Mini-CNN classificateur de chiffres** (compteurs troop bar fiables — le "vrai truc") :
  - [x] Phase 1 — outil de collecte (`tools/data/collect_digit_crops.py`, mode `--position auto`)
  - [~] Phase 2 — **outils faits** : (a) capture accumulante d'une frame `prep_attaque` par épisode (`env_v4._save_digit_frame` → `logs/digit_frames/`, armée pleine = data la plus riche, lue par `collect_digit_crops`) ; (b) labelisation semi-auto (`tools/data/label_digit_crops.py` : crop affiché, pré-remplissage EasyOCR, Enter/num/s/u/q, range en `<count>/`, resumable). **Reste (ton côté)** : lancer des épisodes + labéliser 500-1000 crops.
  - [~] Phase 3 — **À REWORK en PAR-CHIFFRE — version B2 (segmentation + classifieur 0-9 partagé)** *(plan validé Session 14)*. Le 1er `train_digit_cnn.py` était par-nombre-entier → ne peut prédire QUE les nombres vus (88 jamais vu → faux). **Décision : B2 direct** (pas multi-tête) car les camps font ~200-320 places → un compte peut atteindre **3 chiffres** (ex. 200 barbares), que le multi-tête 2 chiffres (1-99) ne couvre pas. B2 = **longueur variable** + meilleure généralisation (un `8` appris en unités sert en centaines).
    - **Pipeline** : (1) segmenter le badge en chiffres individuels (projection-profile / composantes connexes) ; (2) classifieur **0-9** (10 classes) ; (3) inférence = segmenter → classer chaque chiffre → concaténer gauche→droite.
    - **Réutilise ton labeling** : un crop labellisé "200" → segmenté → `2`,`0`,`0`. Zéro re-labelisation.
    - Le `0` = classe normale (apprise des 10/20/200…) ; un compte de 0 = grisé (jamais lu).
    - **Risque** : fiabilité de la segmentation sur petits badges stylisés → **fallback EasyOCR** si segmentation/conf douteuse ; CRNN/CTC en upgrade si la segmentation déçoit.
  - [ ] Phase 4 — intégrer dans `TroopBarDetector._read_count()` (charger `weights/digit_cnn.pt`, fallback EasyOCR si conf basse) — **après** l'entraînement réel.
  - *Pourquoi* : EasyOCR peu fiable sur les petits badges ; le "snapshot OCR + manual decrement" drift quand un tap tombe hors zone de deploy.
  - **Relation avec le deploy-grisé** (gros chantier backlog) : **complémentaires, pas contradictoires**. Le deploy-grisé est le fallback **robuste** (zéro compteur, marche toujours). Ce mini-CNN est l'**upgrade précis** : compteurs exacts → l'agent sait *combien* il lui reste (meilleure stratégie). Cible : compteurs CNN quand fiables, grisé en fallback.
- [ ] **Gros run V4 final** : 300-500 épisodes une fois tous les fixes en place → baseline solide avant V5.

### V5.1 — Foundation multi-agents

> Plomberie pour le futur cerveau. **4 agents déjà faits** (voir CHANGELOG) ; reste l'orchestration.

- [x] **Interface `Brain`** (`brain/interface.py`) : `Brain` ABC + `HeuristicBrain` (= `scheduler.pick`). Seam pour le futur `LocalLLMBrain`.
- [x] **`brain.py` utilise `AgentScheduler`** (Étape A) : `_load_modules` enregistre les 4 agents + crée le `HeuristicBrain` ; `_main_loop` réécrit (`world → brain.decide → scheduler.run → stats`). Vieilles méthodes gardées et taguées `[DEAD-CODE-V5.1]` (revert-safe). ⚠️ **change le comportement** → test réel requis.
- [x] **Étape B** : run réel validé (CombatAgent attaque via le scheduler) → méthodes `[DEAD-CODE-V5.1]` supprimées + fichiers mixins `farm.py`/`war.py`/`chat.py` retirés (logique portée par les agents). Brain = `core` + `loop` + `navigation`. Compteurs morts (`_task_queue`/`_last_chat_check`/`_attacks_since_chat_check`) nettoyés.
- [ ] **ADB zéro screenshot (résiduel)** : faire lire le cache `PerceptionThread` aux consommateurs *live* (`gdc/navigator`, `social/chat`, `clan_castle`). En partie absorbé par le `world`. Le RAW `screencap` ne subsiste que comme fallback documenté (OK).
- [ ] Stop le sanity-rescan dans `environment_v4._all_resources_exhausted()` (redondant avec `_sync_remaining_from_perception()`).
- [ ] **Flag perception `chat_unread`** (badge `!`/rouge près du bouton chat) → `ChatAgent.can_run` ne check qu'en présence du signal (au lieu d'ouvrir périodiquement). Cf vision communication inter-agents.
- [ ] **🔨 Rework COMPLET des sorts (data-driven)** — *plan validé Session 14, à coder*. Aujourd'hui `SPELL_NAMES=['soin','rage','gel']` codé en dur + `ACTION_ABILITY_START = ACTION_SPELL_START + 3` (le `+3` hardcode le nb de sorts). Objectif : zéro nb de sorts hardcodé, tous les sorts via le registre + CNN, un combat peut avoir 10 sorts ou 2 (géré par le **mask**, l'action space est dimensionné à TOUS les types).
  - Registre : `troops.json` role=spell + champ **`target`** par sort (`cluster`=support→troupes / `defense`=offensif→défense ennemie). Lister tous les sorts castables du CNN.
  - `action_space` dérivé : `SPELL_NAMES` du registre ; `ACTION_ABILITY_START = ACTION_SPELL_START + len(SPELL_NAMES)` ; decode/encode + `TOTAL_ACTIONS` dérivés.
  - `mask` : sort actif ssi présent + non grisé. obs `SPELL_FEATURES = len(SPELL_NAMES)`.
  - Ciblage data-driven : `_execute_spell` lit `target` → SpellCaster (sait déjà cibler défense=freeze / cluster=heal,rage).
  - Heuristique : caste les sorts présents (plus de liste figée). **⚠️ re-train** (obs+actions changent).

### V5.0 — Mode live (phases optionnelles)

> Phases 1-2 livrées (voir CHANGELOG). Le reste est optionnel.

- [ ] **Phase 3** : decision tick agent event-driven (thread réagissant aux events `PerceptionEventBus`). Mode prod uniquement (RL training reste sur steps discrets).
- [ ] **Phase 4** : mesurer latency end-to-end (event → action). Cible ~150ms.
- *Avant de coder Ph.3* : définir avec l'utilisateur les critères de "changement significatif", le comportement idle, et l'impact sur le RL.

---

## 📅 À venir

### V5.2 — Nouveaux agents + orchestrateur

> Ajouter les agents manquants + un brain capable de décider QUOI faire et QUAND.

**Agent jeux de clan** (`clan_games/`) — détecter si actifs, identifier les tâches, exécuter (règles, pas de RL).
**Agent gestion village** (`village/`) — constructeurs libres, queue d'amélioration (murs→défenses→ressources), labo libre, collecte ressources.

**🔧 CNN barre d'options bâtiment** (perception robuste) — quand on tape un bâtiment, une barre de ~6-8 boutons apparaît (Demander, Renforcer, Améliorer…). Le template matching actuel sur "Demande" est fragile (~50%).
- [ ] CNN options bar : input = crop barre bas (y~860-1080), output = `{name, x, y, conf}` des boutons.
- [ ] Classes : `demander, renforcer, ameliorer, tresorerie, dormir, infos, rechercher, collecter, acheter…`
- [ ] Pipeline : tap bâtiment → CNN → bouton selon l'intention. **Unlock** l'agent gestion village.
- [ ] Data : ~200-500 crops annotés ; Model : YOLO nano sur la zone barre.

**Orchestrateur `brain.py`** — boucle `priority()`+`can_run()`, gestion cooldowns, logging centralisé, schedule par type d'agent.

### V5.3 — Dashboard web temps réel

> Suivre multi-agents / training / vision agent depuis une page web sur le réseau local. **Pas encore commencé.**

- [ ] **Maquette + spec** des pages AVANT le code (composants, sources, refresh, endpoints REST/WS).
- [ ] Pages : principale (état agents), Training (reward/PPO stats), Replay (overlays par épisode), Village, **Vision Agent** (flux vidéo annoté temps réel via le push pipeline V5.0).
- [ ] Stack : FastAPI + WebSocket ; front HTML/JS ou htmx.
- [ ] **Bonus pré-dashboard** : commande `--live` (fenêtre OpenCV temps réel) pour débugger la vision sans attendre le web.

---

## 🔮 Vision long terme

### V6 — Combat avancé

- [ ] **Caméra / scroll** : actions scroll/pan pour suivre les troupes hors écran (sinon retraite déclenchée trop tôt) ; position caméra dans l'observation.
- [ ] **Multi-compo** : supporter d'autres armées (LavaLoon, Hybrid, QC…) ; adapter `TroopManager` + templates/CNN.
- [ ] **Équipements héros** : détecter l'équipement actif, adapter la stratégie.
- [ ] **Self-play / curriculum** : bases de difficulté croissante (HDV10→12).

### Cerveau LLM local (coach + parole + RAG)

> 100% local, 0€/mois. **Ollama** + Llama 3.1 8B / Mistral 7B / Qwen sur le RTX 5070 (8 Go VRAM). C'est l'aboutissement de la vision : on parle à l'IA en langage naturel via le chat clan, elle supervise les sous-agents. Voir mémoire `project_llm_brain_vision`.

- [ ] **Intégration Ollama** (`uv add ollama`) → `LocalLLMBrain` derrière l'interface `Brain` (V5.1).
- [ ] **Mode coach** : après chaque attaque, contexte → analyse NL → log ou chat clan.
- [ ] **Parole autonome** : commente ses attaques dans le chat ("3★ 100% ! Compo parfaite").
- [ ] **Conseils GdC** : "2 infernos single → soin plutôt que rage".
- [ ] **Rapport quotidien** dans le chat.
- [ ] **RAG** (ChromaDB + SentenceTransformers) — base de connaissance pour donner au LLM le contexte du jeu :
  - **Mécaniques de jeu** : synergies sorts↔troupes (rage = +dégâts/+vitesse ; gel fige les défenses ; soin), rôles (golem = tank lent/PV, démolisseur = fonce sur les murs, etc.) → le cerveau raisonne stratégie.
  - **Stats exactes par niveau** (wiki CoC scrappé) → pas d'hallucination sur les chiffres.
  - **Historique d'attaques** (auto-alimenté) → mémoire épisodique ("la semaine dernière sur une base similaire…").
  - **Meta + données clan** (compos populaires, membres, règles).
  - **Le jargon/contexte = RAG, PAS fine-tuning** (le fine-tune apprend le style, pas les faits → hallucinations sinon). MAJ CoC → mettre à jour la base RAG seulement, zéro ré-entraînement.
- [ ] **Fine-tuning optionnel** (LoRA) : uniquement pour le *style* (parler comme un membre du clan), à partir des vrais logs de chat.
- [ ] ⚠️ **Sécurité** : le chat clan est un input HOSTILE (injection de prompt) → whitelist des donneurs d'ordres + actions destructives derrière confirmation. Séparer cerveau (décide QUOI) et RL (exécute COMMENT).

---

## 🗃️ Backlog (non planifié)

> Idées pas encore assignées à une version. On pioche ici quand on a du temps.

### 🎯🔨 GROS CHANTIER — Inventaire & déploiement pilotés par le grisé (zéro compteur)

> **Décidé Session 14, à faire (assez gros, ne pas oublier).** Objectif final : ajouter une troupe/engin/sort = **retrain le CNN + 1 ligne de data**, JAMAIS toucher au code Python. Et un déploiement robuste à la taille des camps / au changement de compo.

**Pourquoi** : il n'existe **pas de compteur fiable** (l'OCR des compteurs a été retiré Session 13). Aujourd'hui `_remaining_troops` est initialisé à `default_max` (par troupe, codé en dur dans `TROOP_TYPES`) — fragile : les camps grossissent, on change de compo. Le **seul signal fiable = `is_grayed`** du CNN troop bar (déjà exploité par `_sync_remaining_from_perception` qui met à 0 les grisés). Le `max` et la logique compteur sont **couplés** → on ne peut pas juste retirer `max` (l'heuristique construit sa séquence à l'avance à partir des compteurs).

**À faire (cohérent, un seul chantier)** :
- [x] **Registre data-driven** (Session 14) : `configs/troops.json` = SSOT `{name, role, max?}` ; `TROOP_TYPES` (`legacy/agent.py`) + `ROLE_TO_TROOPS` (`action_space.py`, group-by rôle) en **dérivent** via `combat/troop_registry.py`. **47 troupes** (toutes les classes déployables du CNN). Ajouter une troupe = 1 ligne JSON + retrain CNN, **zéro code**. Existantes préservées à l'identique, obs 54 dims (checkpoint-safe). → **corrige l'urgence "nouvelles troupes pas déployées"** (golem_glace, bebe_dragon, gargouille, yeti, etc.). `max` gardé comme borne haute optionnelle (défaut par rôle) — pas encore "zéro compteur".
- [ ] **Deploy-until-grayed** (reste du chantier) : `_execute_deploy(role)` déploie la troupe non-grisée du rôle ; le **mask** active `deploy(role)` tant qu'une troupe du rôle est non-grisée ; l'heuristique = "déploie ce rôle tant que pas grisé" (boucle runtime). → supprime `max`/`default_max` définitivement + rend le sanity-rescan inutile.
- [ ] **Rôles best-guess à valider** : les rôles des troupes récentes dans `troops.json` sont des estimations (éditables sans code). Vérifier en jeu et ajuster.
- [ ] **Impact RL** : change la sémantique de l'obs (role_counts → présence par rôle) → **re-train** (acceptable, checkpoint actuel faible : 0★ 27%, 80 ep). **Test émulateur requis.**
- [ ] **Sorts** : ajouter un sort change la dim d'obs (`SPELL_FEATURES`) → pas checkpoint-safe (à gérer à part des troupes).
- [ ] **Full-auto (horizon LLM)** : classe CNN inconnue → l'orchestrateur LLM déduit le rôle (connaissance jeu + RAG) et remplit le registre tout seul. Rejoint *Apprentissage continu*.

**Autre ajustement combat (non-critique, vu au 1er run)**
- [ ] **Spam de sorts** : l'heuristique balance tous les sorts d'affilée (3 rages en ~5s). Court terme : espacer dans `get_heuristic_sequence`. Long terme : timing géré par l'orchestrateur LLM.

**Combat intelligent**
- [ ] Estimation loot avant attaque (OCR ressources adverses → skip si pas rentable).
- [ ] Classification de base (farming/war/anti-3★) pour adapter la stratégie.
- [ ] Analyse des replays (extraire des patterns d'erreur).
- [ ] Ligue auto (monter/descendre selon objectif) ; combats classés.

**Gestion village**
- [ ] Queue recherche labo ; overflow ressources ; queue d'amélioration bâtiments ; gestion bouclier ; don de troupes auto (depuis le chat).

**Recrutement & social**
- [ ] Annonces de recrutement (chat global) ; réponses aux commandes membres ; rejoindre les guerres auto.

**Infrastructure & UX**
- [ ] Calibration UI automatique (remplacer `ui_positions.json` par détection YOLO/template).
- [ ] Replay vidéo des attaques (enregistrement ADB) ; multi-compte (plusieurs émulateurs) ; comportement humain (délais/patterns) ; mode coaching.

**ML & training**
- [ ] Curriculum learning ; self-play ; transfer learning ; estimation pré-attaque (% destruction prédit).

**Apprentissage continu (adaptation aux MAJ CoC)**
> Human-in-the-loop, 0€, ~1 semaine de maintenance par MAJ majeure CoC.
- [ ] Détection d'inconnus (YOLO conf < seuil → `unknown_X` + crop auto dans `needLabelisation/`).
- [ ] Maintenance mode (labéliser → réentraîner sur Kaggle) ; notification claire des inconnus détectés.
