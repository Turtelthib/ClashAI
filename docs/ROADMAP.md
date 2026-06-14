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

- [ ] **Mini-CNN classificateur de chiffres** (remplace EasyOCR pour les compteurs troop bar) :
  - [x] Phase 1 — collecte (`tools/data/collect_digit_crops.py`, mode `--position auto`)
  - [ ] Phase 2 — labelisation (500-1000 crops, split 80/20)
  - [ ] Phase 3 — entraîner un mini-CNN (LeNet/MobileNetV3-Small, `tools/train/train_digit_cnn.py`)
  - [ ] Phase 4 — intégrer dans `TroopBarDetector._read_count()` (fallback EasyOCR si conf basse)
  - *Pourquoi* : EasyOCR peu fiable sur les petits badges ; le "snapshot OCR + manual decrement" drift quand un tap tombe hors zone de deploy.
- [ ] **Gros run V4 final** : 300-500 épisodes une fois tous les fixes en place → baseline solide avant V5.

### V5.1 — Foundation multi-agents

> Plomberie pour le futur cerveau. **4 agents déjà faits** (voir CHANGELOG) ; reste l'orchestration.

- [ ] **Interface `Brain`** (seam pour cerveau swappable) : `brain.py` actuel → `HeuristicBrain` ; futur `LocalLLMBrain` (LLM local + RAG jargon clan, voir [Cerveau LLM](#cerveau-llm-local-coach--parole--rag)).
- [ ] **`brain.py` utilise `AgentScheduler`** au lieu de sa logique if/else en dur (enregistre les 4 agents, boucle sur `pick(world)`). ⚠️ **Première étape qui change le comportement** → test réel important.
- [ ] **ADB zéro screenshot (résiduel)** : faire lire le cache `PerceptionThread` aux consommateurs *live* (`gdc/navigator`, `social/chat`, `clan_castle`) au lieu de leur propre grab. *Note* : en grande partie absorbé par le `world` quand le brain passe sur le scheduler ; le RAW `adb exec-out screencap` ne subsiste que comme fallback documenté (OK).
- [ ] Stop le sanity-rescan dans `environment_v4._all_resources_exhausted()` (redondant avec `_sync_remaining_from_perception()`).

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
- [ ] **RAG** (ChromaDB + SentenceTransformers) : wiki CoC (stats exactes) + historique attaques (mémoire épisodique) + meta + données clan. **Le jargon/contexte = RAG, PAS fine-tuning** (le fine-tune apprend le style, pas les faits → hallucinations sinon).
- [ ] **Fine-tuning optionnel** (LoRA) : uniquement pour le *style* (parler comme un membre du clan), à partir des vrais logs de chat.
- [ ] ⚠️ **Sécurité** : le chat clan est un input HOSTILE (injection de prompt) → whitelist des donneurs d'ordres + actions destructives derrière confirmation. Séparer cerveau (décide QUOI) et RL (exécute COMMENT).

---

## 🗃️ Backlog (non planifié)

> Idées pas encore assignées à une version. On pioche ici quand on a du temps.

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
