# ClashAI — Roadmap

> **OBJECTIF FINAL** : une IA autonome intelligente qui joue comme un humain — joue, gère, recrute, s'améliore seule, et qu'on **pilote en langage naturel via le chat clan** (cerveau LLM local orchestrant des sous-agents).

**Statut** : `[ ]` à faire · `[x]` fait (détail → [CHANGELOG](CHANGELOG.md)) · 🔧 bug documenté → [TROUBLESHOOTING](TROUBLESHOOTING.md)
**Mise à jour** : 4 juillet 2026 (Session 15 — digit CNN + sorts + fixes sorts-château validés en réel ; gros run en cours ; plans V5.2→V7 & stack LLM figés)

📂 **Ce doc** = ce qui reste à faire. · ✅ Fait → [CHANGELOG.md](CHANGELOG.md) · 🔧 Fix détaillés → [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

## Sommaire

- [📊 État des versions](#-état-des-versions)
- [🚀 En cours](#-en-cours)
  - [V4.4 — Polish perception](#v44--polish-perception)
  - [V5.1 — Foundation multi-agents](#v51--foundation-multi-agents)
  - [V5.0 — Mode live (phases optionnelles)](#v50--mode-live-phases-optionnelles)
- [📅 À venir](#-à-venir)
  - [V5.2 — Perception + agents (règles)](#v52--perception--agents-règles)
  - [V5.3 — Cerveau LLM v1 (orchestrateur)](#v53--cerveau-llm-v1-orchestrateur)
  - [V5.4 — Pilotage chat + RAG complet](#v54--pilotage-chat--rag-complet)
- [🔮 Vision long terme](#-vision-long-terme)
  - [V6 — Dashboard web complet](#v6--dashboard-web-complet)
  - [V7+ — Automatisation & intelligence](#v7--automatisation--intelligence)
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
| V4.4 | 🔄 **en cours** | Digit CNN ✅ validé en réel ; reste le gros run (re-train 67 dims **en cours**) |
| V5.0 | ✅ Ph.1-2 | Push pipeline WGC → PerceptionThread (Ph.3-4 optionnelles) |
| Refacto | ✅ | src/ layout + 13 splits (0 fichier >500L hors legacy) |
| V5.1 | 🔄 **en cours** | Brain + scheduler + 4 agents ✅ ; restent les résiduels (ADB cache, sanity-rescan, chat_unread) |
| V5.2 | 💡 | CNN barre d'options → Agent village + Agent jeux de clan |
| V5.3 | 💡 | **Cerveau LLM v1** (orchestrateur) : `LocalLLMBrain` décide quel agent lancer |
| V5.4 | 💡 | **Pilotage chat + RAG complet** : parler à l'IA via le chat clan |
| V6 | 💡 | **Dashboard web complet** : visualise cerveau + agents + perfs + contrôle |
| V7+ | 💡 | **Automatisation & intelligence** : combat réactif, village intelligent, amélioration continue, multi-compte |
| V END | 🎯 | IA autonome complète |

> **Séquence figée (Session 14)** : V5.2 (perception+agents) → V5.3 (cerveau LLM) → V5.4 (chat+RAG) → **V6 (dashboard, une fois le LLM en place)** → V7+ (automatisation/intelligence). Détail « comment + avec quoi » dans *À venir* / *Vision*.

---

## 🚀 En cours

### V4.4 — Polish perception

> Clore les derniers irritants de perception, puis un gros run de validation avant la suite.

- [x] **Mini-CNN classificateur de chiffres** (compteurs troop bar fiables) — **FAIT & validé en conditions réelles (Session 15)** :
  - [x] Phase 1 — outil de collecte (`tools/data/collect_digit_crops.py`, mode `--position auto`)
  - [x] Phase 2 — **collecte + labelisation FAITES (Session 15)** : outils (capture prep_attaque `env_v4._save_digit_frame`, labelisation semi-auto `tools/data/label_digit_crops.py`) + **~956 crops labélisés** par l'utilisateur (dont renfort `0`/`7`).
  - [x] Phase 3 — **PAR-CHIFFRE B2 (segmentation + classifieur 0-9 partagé) — FAIT Session 14**. `clashai/perception/digit_reader.py` (SSOT segmentation + `DigitCNN` + `read_count`), `tools/data/build_digit_singles.py` (whole-number→per-digit, **réutilise ton labeling**, 730 crops→634 used), `train_digit_cnn.py` adapté (augmentation + oversampling + acc/classe). **Modèle : 100% val acc/classe** après renfort `0`/`7` (Session 15 ; était 98%) (`weights/digit_cnn.pt`). Read e2e : ~83% brut, **conf-gating** rejette ~14% (erreurs basse-conf → fallback). Longueur variable (gère 7, 79, 200…).
    - Segmentation ~87% exact (un "1" se sur-découpe parfois) → CRNN/CTC en upgrade si besoin.
  - [x] **Phase 4 — intégration (reset-seeding) — FAIT Session 14**. `digit_reader.crop_count_badge` + `read_bar_counts` (SSOT, partagé avec collect). `core._seed_counts_from_digits()` appelé au reset : lit la **barre de combat au début de l'attaque** (compteurs pleins, position combat = matché à l'entraînement → pas de souci prep) et seede `_remaining_troops` avec les **vrais compteurs**. Fallback par troupe = `default_max` si non lu (conf < 0.6). **Troupes ET sorts seedés** (Session 15) + **re-lecture live à chaque `observe`** (`_sync_remaining_from_perception`, frame fraîche hors burst) → corrige la dérive du décrément manuel. Log `digit-CNN seed: ...`. **✅ Validé en conditions réelles (Session 15).**
    - [x] *Renfort données `0`/`7`* fait (47/46 ex., 100% val) ; collecter d'autres nombres au fil de l'eau améliore encore (ongoing).
    - Doublons château (armée + CC) : compteurs **sommés** + dépletion seulement si **toutes** les icônes du nom grisées + positions rafraîchies (→ TROUBLESHOOTING).
  - *Pourquoi* : EasyOCR peu fiable sur les petits badges ; le "snapshot OCR + manual decrement" drift quand un tap tombe hors zone de deploy. → **résolu** par le digit CNN (reset-seeding + re-lecture live).
  - **Relation avec le deploy-grisé** : architecture actée Session 15 — **compteurs digit-CNN = source primaire, grisé = autorité de fin / filet de sécurité**. Le "gros chantier zéro compteur" a été **requalifié en petit item de hardening** (sa prémisse "pas de compteur fiable" est obsolète depuis ce CNN), voir backlog.
- [~] **Gros run final (re-train)** : **EN COURS (Session 15)** — re-train sur la nouvelle obs **67 dims / 50 actions** (imposé par le rework sorts). C'est l'**obs définitive** (refonte "présence-par-rôle" abandonnée, voir backlog) → ce run sert de baseline solide avant la suite.

### V5.1 — Foundation multi-agents

> Plomberie pour le futur cerveau. **4 agents déjà faits** (voir CHANGELOG) ; reste l'orchestration.

- [x] **Interface `Brain`** (`brain/interface.py`) : `Brain` ABC + `HeuristicBrain` (= `scheduler.pick`). Seam pour le futur `LocalLLMBrain`.
- [x] **`brain.py` utilise `AgentScheduler`** (Étape A) : `_load_modules` enregistre les 4 agents + crée le `HeuristicBrain` ; `_main_loop` réécrit (`world → brain.decide → scheduler.run → stats`). Vieilles méthodes gardées et taguées `[DEAD-CODE-V5.1]` (revert-safe). ⚠️ **change le comportement** → test réel requis.
- [x] **Étape B** : run réel validé (CombatAgent attaque via le scheduler) → méthodes `[DEAD-CODE-V5.1]` supprimées + fichiers mixins `farm.py`/`war.py`/`chat.py` retirés (logique portée par les agents). Brain = `core` + `loop` + `navigation`. Compteurs morts (`_task_queue`/`_last_chat_check`/`_attacks_since_chat_check`) nettoyés.
- [ ] **ADB zéro screenshot (résiduel)** : faire lire le cache `PerceptionThread` aux consommateurs *live* (`gdc/navigator`, `social/chat`, `clan_castle`). En partie absorbé par le `world`. Le RAW `screencap` ne subsiste que comme fallback documenté (OK).
- [ ] Stop le sanity-rescan dans `environment_v4._all_resources_exhausted()` (redondant avec `_sync_remaining_from_perception()`).
- [ ] **Flag perception `chat_unread`** (badge `!`/rouge près du bouton chat) → `ChatAgent.can_run` ne check qu'en présence du signal (au lieu d'ouvrir périodiquement). Cf vision communication inter-agents.
- [x] **🔨 Rework COMPLET des sorts (data-driven)** — *fait Session 14*. `SPELL_NAMES` dérivé du registre **∩ classes CNN** (`troop_registry.load_spell_names`), plus de `+3` hardcodé (`ACTION_ABILITY_START = ACTION_SPELL_START + len(SPELL_NAMES)`), constantes `ACTION_CAST_*` retirées. **16 sorts** (vs 3) ; un sort pré-enregistré mais pas encore dans le CNN (ex. `colere`) reste **inerte** (pas de dim morte / re-train inutile). Ciblage data-driven : `SPELL_TARGET_DEFAULTS` (cluster/heal/defense) overridable via `target` dans le JSON, mappé sur SpellCaster. Heuristique caste tous les sorts présents (mains d'abord). `load()` tolère le mismatch de dims. obs **54→67**, actions **37→50** → **re-train** (heuristique OK direct). **✅ Testé en conditions réelles (Session 15) ; re-train en cours.**

### V5.0 — Mode live (phases optionnelles)

> Phases 1-2 livrées (voir CHANGELOG). Le reste est optionnel.

- [ ] **Phase 3** : decision tick agent event-driven (thread réagissant aux events `PerceptionEventBus`). Mode prod uniquement (RL training reste sur steps discrets).
- [ ] **Phase 4** : mesurer latency end-to-end (event → action). Cible ~150ms.
- *Avant de coder Ph.3* : définir avec l'utilisateur les critères de "changement significatif", le comportement idle, et l'impact sur le RL.

---

## 📅 À venir

### V5.2 — Perception + agents (règles)

> CNN barre d'options + 2 agents à base de règles (pas de RL). Pipeline perception **identique au digit/troop bar**.

**🔧 CNN barre d'options bâtiment** — taper un bâtiment ouvre une barre de ~6-8 boutons (Demander, Renforcer, Améliorer…). Le template matching sur "Demande" est fragile (~50%).
- [ ] Collect crops (barre bas y~860-1080 quand un bâtiment est tapé) → label boutons (`demander/renforcer/ameliorer/tresorerie/collecter/rechercher…`) → train **YOLO nano** → `OptionsBarDetector` `{name,x,y,conf}`. Data ~200-500 crops. **Unlock** l'agent village.

**Agent village** (`village/`, `VillageAgent(BaseAgent)`, règles) — constructeurs libres, queue d'upgrade (murs→défenses→ressources), labo, collecte ; clique via `OptionsBarDetector`. State machine simple.
**Agent jeux de clan** (`clan_games/`) — détecter si actifs, lire les tâches (OCR), exécuter.

### V5.3 — Cerveau LLM v1 (orchestrateur)

> `LocalLLMBrain(Brain)` remplace `HeuristicBrain` : décide QUEL agent lancer selon le `world`. Le seam `Brain` existe déjà (V5.1).

- [ ] `LocalLLMBrain.decide(world)` → prompt (world JSON + agents-tools + RAG minimal) → Ollama **tool-call** → agent choisi.
- [ ] Stack : Ollama + **Mistral 7B** + `ollama-python` (détail figé → *Cerveau LLM local*).

### V5.4 — Pilotage chat + RAG complet

> Parler à l'IA via le chat clan ; elle comprend le jargon CoC + le contexte du clan.

- [ ] `ChatAgent` (déjà là) → `LocalLLMBrain` (avec RAG) → répond / exécute / rapporte.
- [ ] RAG : **Chroma** + `nomic-embed-text` (jargon/méca CoC + contexte clan + préférences) → *Cerveau LLM local*.
- [ ] ⚠️ Sécurité : chat = input **hostile** (injection) → whitelist des donneurs d'ordres + actions destructives derrière confirmation.

---

## 🔮 Vision long terme

### V6 — Dashboard web complet

> Prend tout son sens une fois le LLM en place : visualiser le raisonnement du cerveau + l'activité des agents + les perfs, et **contrôler**.

- [~] **Maquette + spec** des pages AVANT le code — **brief complet écrit** : [`docs/dashboard_brief.md`](dashboard_brief.md) (contexte, panneaux, sources, direction visuelle « poste de commande », tech cible). Maquette v1 faite (skill artifact-design) ; version finale à générer via **Claude Design** (produit séparé) → docs → dev.
- [ ] Pages/panneaux : **Cerveau** (décisions LLM + tool-calls), **Agents** (scheduler), **État des CNN** (statut live + ms/inférence par modèle ; `yolo_troops` en alerte), **Vision** (flux annoté temps réel), **Combat/RL** (reward + stats), **Chat**, **Village**, **Journal** (.md), **Contrôle**.
- [ ] Contrôle : start/stop, commandes manuelles, override.
- [ ] Stack : **FastAPI + WebSocket** ; front HTML/JS ou htmx (ou React).
- [ ] **Bonus pré-dashboard** : commande `--live` (fenêtre OpenCV temps réel) pour débugger la vision sans attendre le web.

### V7+ — Automatisation & intelligence

- [ ] **Combat réactif** (cf section combat) : obs tactique post-`yolo_troops` + reward de timing → l'agent joue libre, pas scripté.
- [ ] **Gestion village intelligente** : priorisation upgrades pilotée par le LLM (méta + objectifs), gestion bouclier, dons auto.
- [ ] **Communication inter-agents** : l'attack agent demande des troupes au CC agent, le village négocie les ressources → bus de messages + arbitrage LLM.
- [ ] **RL — efficacité échantillons** (épisodes coûteux) : PPO on-policy = peu efficace. Alternatives → **off-policy** (Rainbow/DQN, SAC discret, replay buffer) ou **model-based DreamerV3** (entraînement "en imagination"). À évaluer si la convergence est trop lente.
- [ ] **Amélioration continue** : self-play / curriculum (HDV croissants), analyse de replays (patterns d'erreur), multi-compo (LavaLoon, Hybrid, QC…), équipements héros.
- [ ] **Caméra / scroll** : suivre les troupes hors écran (sinon retraite trop tôt) ; position caméra dans l'obs.
- [ ] **Multi-compte**.

### Cerveau LLM local (coach + parole + RAG)

> 100% local, 0€/mois. C'est l'aboutissement de la vision : on parle à l'IA en langage naturel via le chat clan, elle supervise les sous-agents. Voir mémoire `project_llm_brain_vision`.

**🔧 Division du travail — archi figée (Session 15, validée par le schéma utilisateur)** :
- **LLM = manager/stratège** : vue globale (via `build_world` + RAG), décide **QUOI/QUAND** (attaquer, up quel bâtiment, quelle compo), **coache le RL** (debrief post-attaque → reward shaping + mémoire), parle au clan.
- **Sous-agents = yeux+mains+experts** : chacun (1) exécute une tâche, (2) **rapporte** ce qu'il a vu/fait, (3) **escalade les décisions** au LLM. Pattern : l'agent fait le check *pas cher* (perception), le LLM tranche le *cher* (raisonnement).
- **Agent combat/RL** : reçoit compo+cible du LLM → exécute le **micro** (temps réel) → rapporte le résultat → LLM debriefe. Le LLM **ne remplace pas** le RL (trop lent pour le micro) ; le RL **ne remplace pas** le LLM (pas de raisonnement/stratégie).
- **Exécution : heuristique-guidée-par-LLM d'abord** (pragmatique, marche tout de suite), RL pour l'optim micro **quand** il apporte un gain (le run baseline plafonne ~1.4★ → RL pas prioritaire).
- **Seuil de décision** : l'agent décide seul le routinier ; escalade au LLM le stratégique/ambigu (évite d'appeler le LLM 1000×/min).
- **Canaux** : dialogue live agent↔LLM = **tool-calls** ; **`.md` = carnet durable** (log décisions lisible par l'humain + mémoire RAG + canal d'instructions humaines). Inter-agents via le LLM au début (bus direct → V7+).
- **4/6 agents déjà faits** (V5.1 : Combat/Chat/GdC/ClanCastle) ; restent Village + JeuxClan (V5.2) + le LLM (V5.3).

**🔧 Stack figé (Session 14)** :
- **Runtime** : **Ollama** (serveur local, offload GPU/RAM auto, tool-calling), appelé via `ollama-python` (HTTP `localhost:11434`).
- **Modèle** : **Mistral 7B Instruct** (labo 🇫🇷, Apache 2.0, FR natif, tool-calling) Q4 par défaut (~4.5 Go → tient sur GPU à côté des CNN ~1-2 Go ; rapide). Upgrade : **Mistral Nemo 12B** (Mistral+NVIDIA, Apache 2.0, 128k ctx) en offload partiel. Alt US : Llama 3.1 8B (Meta). → prendre le dernier petit Mistral instruct. **(Qwen écarté — préférence US/EU de l'utilisateur.)**
- **Où** : **GPU** (l'orchestrateur décide toutes les qq s, pas de latence critique → 7-8B sur GPU = sweet spot). RAM (64 Go) = réservé à un éventuel "penseur lent" offline (70B, 1-3 tok/s) plus tard.
- **Sortie** : **tool-calling** (agents = tools) → décision structurée, pas de parsing fragile.
- **RAG** (perso) : **Chroma** + embedder `nomic-embed-text` (Ollama). Indexe jargon/méca CoC + contexte clan + préférences. MAJ = ajouter des docs, zéro ré-entraînement.
- **Découpage** : V5.3 = `LocalLLMBrain.decide(world)` (orchestre, RAG minimal) ; V5.4 = ChatAgent↔LLM + RAG complet (parole/ordres).

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

### 🔨 Hardening grisé (ex-"GROS CHANTIER zéro compteur", requalifié Session 15)

> **Requalifié Session 15.** La prémisse du chantier (« il n'existe pas de compteur fiable ») est **obsolète depuis le digit-CNN** (seed au reset + re-lecture live à chaque `observe`, validé en réel). La refonte "zéro compteur" (obs présence-par-rôle) est **abandonnée** : elle dégraderait l'observation — moins d'info que les vrais comptes, désormais fiables. Architecture actée : **compteurs digit-CNN = source primaire, grisé = autorité de fin / filet de sécurité** (`default_max` ne subsiste que comme fallback basse-confiance ~8% des lectures, corrigé au runtime par le grisé).

**Ce qui reste (petits items)** :
- [x] **Registre data-driven** (Session 14) : `configs/troops.json` = SSOT `{name, role, max?}` ; `TROOP_TYPES` + `ROLE_TO_TROOPS` en dérivent via `combat/troop_registry.py`. **47 troupes**, ajouter une troupe = 1 ligne JSON + retrain CNN, **zéro code**. (Détail → CHANGELOG.)
- [ ] **Mask ceinture-bretelles** : autoriser `deploy(role)` tant qu'une troupe du rôle est **non-grisée**, même si le compteur lu dit 0 (protège d'une lecture basse erronée). Petit changement, pas une refonte.
- [ ] **Rôles best-guess à valider** : les rôles des troupes récentes dans `troops.json` sont des estimations (éditables sans code). Vérifier en jeu et ajuster.
- [ ] **Sorts** : ajouter un sort change la dim d'obs (`SPELL_FEATURES`) → pas checkpoint-safe (à gérer à part des troupes).
- [ ] **Full-auto (horizon LLM)** : classe CNN inconnue → l'orchestrateur LLM déduit le rôle (connaissance jeu + RAG) et remplit le registre tout seul. Rejoint *Apprentissage continu*.

**Autre ajustement combat (non-critique, vu au 1er run)**
- [ ] **🔨 Retrain `yolo_troops.pt` (CNN troupes terrain)** — *root cause du rage mal placé*. Le modèle est **sous-entraîné** (peu de classes) → ne reconnaît pas la plupart des troupes déployées → `main_cluster` vide → support spells au fallback. Workaround en place (`_troop_march_point`), mais le vrai fix = ré-entraîner avec toutes les troupes (comme le CNN troop bar). Débloque rage/heal **précis** + features combat fiables.
- [~] **Spam de sorts** : l'heuristique balance tous les sorts d'affilée. **Atténué** Session 14 : `_spread_cluster_point` étale les casts cluster (plus d'empilement spatial). **Reste (vu run Session 14)** :
  - Espacement **temporel** (l'heuristique enchaîne les casts ; timing géré par l'orchestrateur LLM à terme).
  - **Gel re-gèle la même défense déjà gelée** → `SpellCaster` doit mémoriser les défenses gelées récemment (cooldown ~5s) et viser la suivante. Petit fix dédié possible.

**🔨 Robustesse du déploiement** *(2 bugs liés, vus run Session 14)*
- [ ] **Taps de deploy invalides** : `_execute_deploy` tape `self._deploy_positions[i]` (périmètre murs/bâtiments) — parfois le point tombe sur un bâtiment / dans la base / zone rouge non-déployable → tap sans effet, mais le compteur décrémente quand même. Fixes (simple→robuste) :
  1. **Push outward** : décaler les positions du périmètre vers l'extérieur (loin de la base, sur l'herbe) d'une marge.
  2. **Snap zone déployable** : masque herbe verte (HSV) OU détection de l'overlay rouge (frame avec troupe sélectionnée) → snapper chaque position au point valide le plus proche. Pas de train.
  3. **Validation post-deploy** : après le tap, vérifier qu'une unité est apparue (compteur baissé / YOLO troupe) ; sinon retry à un offset. Filet de sécurité.
- [ ] **Capacités des héros déployés au rescan jamais jouées** : l'heuristique file les `ability(i)` selon l'inventaire de DÉPART. Un héros déployé **tard** (au `cleanup`, car son deploy initial a raté — cf ci-dessus) n'a pas sa capa dans la séquence → oubliée. Fix : déclenchement **piloté par perception** — après `cleanup()` (et périodiquement), passe "fire abilities" qui, pour chaque héros dont le CNN voit la capa dispo (non grisée) + non utilisée → `_execute_ability`. Le `hero_manager` détecte déjà les `*_capa`. *(Lien : fixer les taps invalides réduit les héros déployés tard ; cette passe = filet.)*

**🔨 Combat réactif / autonome (moins scripté)** *(objectif clé, noté Session 14)*
> Le déroulé `deploy→sorts→rescan→observe` visible = l'**heuristique** (prof BC + fallback). L'agent RL décide déjà action-par-action selon l'obs, mais 3 choses le brident → le rendre vraiment libre/réactif :
- [ ] **Obs tactique riche** : savoir *où sont mes troupes* (→ **retrain `yolo_troops`**), où le push progresse/bloque, position relative des défenses. Sans ça l'agent est aveugle → reste proche du script. **Prérequis #1.**
- [ ] **Reward de timing** : récompenser rage sur troupes engagées, soin sur troupes blessées, gel sur défense dangereuse active, renfort là où ça bloque. Aujourd'hui reward ≈ destruction/étoiles → signal trop pauvre pour la tactique réactive.
- [ ] **Moins d'ancrage BC** : l'agent démarre scripté (clone heuristique) ; avec bon reward + assez d'épisodes, PPO s'en écarte. Option : réduire le poids/durée du BC après un premier baseline.
- [ ] **Horizon LLM (V6)** : le cerveau LLM raisonne sur la bataille en temps réel = l'autonomie ultime. Rejoint *Cerveau LLM local*.

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
