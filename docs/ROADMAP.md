# ClashAI — Roadmap

> **OBJECTIF FINAL** : une IA autonome intelligente qui joue comme un humain — joue, gère, recrute, s'améliore seule, et qu'on **pilote en langage naturel via le chat clan** (cerveau LLM local orchestrant des sous-agents).

**Statut** : `[ ]` à faire · `[~]` partiel · `[x]` fait (détail → [CHANGELOG](CHANGELOG.md)) · 🚫 bloqué · 🔧 bug documenté → [TROUBLESHOOTING](TROUBLESHOOTING.md)
**Mise à jour** : 17 août 2026 — CNN UI livré & branché, agent village (récolte + upgrades) livré, re-train UI en cours.

📂 **Ce doc** = ce qui reste à faire. · ✅ Fait → [CHANGELOG.md](CHANGELOG.md) · 🔧 Fix détaillés → [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

**Chiffres actuels (vérifiés dans le code)** : 17 sorts · obs **69 dims** / **56 actions** · 63 entrées `troops.json` · CNN UI **124 classes** (re-train en cours) · **184 tests**.

---

## Sommaire

- [📊 État des versions](#-état-des-versions)
- [⏳ En attente de validation réelle](#-en-attente-de-validation-réelle)
- [🚀 En cours](#-en-cours)
  - [V5.2 — CNN UI + agents village](#v52--cnn-ui--agents-village)
  - [V4.4 — Polish perception](#v44--polish-perception)
  - [V5.1 — Résiduels multi-agents](#v51--résiduels-multi-agents)
  - [V5.0 — Mode live (phases optionnelles)](#v50--mode-live-phases-optionnelles)
- [📅 À venir](#-à-venir)
  - [V5.3 — Cerveau LLM v1 (orchestrateur)](#v53--cerveau-llm-v1-orchestrateur)
  - [V5.4 — Pilotage chat + RAG complet](#v54--pilotage-chat--rag-complet)
- [🔮 Vision long terme](#-vision-long-terme)
  - [V6 — Dashboard web complet](#v6--dashboard-web-complet)
  - [V7+ — Automatisation & intelligence](#v7--automatisation--intelligence)
  - [Cerveau LLM local (archi + stack figés)](#cerveau-llm-local-archi--stack-figés)
- [🗃️ Backlog (non planifié)](#️-backlog-non-planifié)

---

## 📊 État des versions

| Version | Statut | Résumé |
|---|---|---|
| V1–V4.3 | ✅ | Décision unique → obs/actions, YOLO troupes+barre, perception async, WGC (voir CHANGELOG) |
| V4.4 | 🔄 | Digit CNN ✅ validé en réel ; **reste la baseline RL en 69/56** |
| V5.0 | ✅ Ph.1-2 | Push pipeline WGC → PerceptionThread (Ph.3-4 optionnelles) |
| Refacto | ✅ | src/ layout + 13 splits (0 fichier >500L hors legacy) |
| V5.1 | 🔄 | Brain + scheduler + 4 agents ✅ ; 3 résiduels (ADB cache, sanity-rescan, chat_unread) |
| **V5.2** | 🔄 **en cours** | CNN UI ✅ livré/branché · Agent village : récolte ✅, upgrades ⏳ (attend re-train) · labo + dons à faire · jeux de clan 🚫 |
| V5.3 | 💡 | **Cerveau LLM v1** (orchestrateur) : `LocalLLMBrain` décide quel agent lancer |
| V5.4 | 💡 | **Pilotage chat + RAG complet** : parler à l'IA via le chat clan |
| V6 | 💡 | **Dashboard web** (maquette ✅, build réel à faire) |
| V7+ | 💡 | Combat réactif, village intelligent, amélioration continue, multi-compte |
| V END | 🎯 | IA autonome complète |

> **Séquence figée** : V5.2 (perception+agents) → V5.3 (cerveau LLM) → V5.4 (chat+RAG) → **V6 (dashboard, une fois le LLM en place)** → V7+.

---

## ⏳ En attente de validation réelle

> Code livré + testé unitairement, **pas encore confirmé en jeu**. À vider au fil des runs.

- [ ] **Re-train CNN UI** (en cours) : nouveau dataset avec `compteur_or/elixir/elixir_noire`, `nombre_ouvrier`, `place_labo`, `prix_upgrade`, `confirmer_upgrade`, `donner`. Au retour : déposer en `weights/yolo_ui.pt`, checker le **mAP par classe** (les nouvelles ont peu d'exemples).
- [ ] **Mappings CNN ambigus** : `find_match` (→ `trouver_partie_rapide`) et `gdc_open` (→ `guerre_clan`) à confirmer sur captures réelles.
- [ ] **Flux upgrade en réel** : `tools/debug/village_upgrade_demo.py` (sans `--confirm` = annule, zéro dépense).
- [ ] **Récolte en réel** : vérifier la boucle re-scan sur un village plein.
- [ ] **Abandon state-dependent** : `capituler`+`confirmer` vs `terminer_bataille`.
- [ ] **Lecture digit CNN sur les gros nombres** (ressources ~10M) : police différente des badges de troupes → si la segmentation cale, ajouter des samples + re-train (même pipeline).

---

## 🚀 En cours

### V5.2 — CNN UI + agents village

> CNN UI **livré** (détecteur universel de boutons) + agents à base de règles. Détail du fait → [CHANGELOG](CHANGELOG.md).

**CNN UI** — le socle est en place (`UIDetector` branché au démarrage, `find_button()` = point d'accès unique).
- [ ] **Migrer les appelants restants vers `find_button()`** (17 appels directs à `get_position` + 18 taps en dur hors JSON) : retraite → dialogues confirmer/annuler → demander château → menus bâtiment. Sans urgence : ils marchent déjà via la calibration.
- [ ] **Renfort dataset** : classes rares (1-2 exemples) ratées, confondues avec `background` → ajouter des captures des cas ratés.

**Agent village** (`village/`, `VillageAgent`, règles ; clique via `UIDetector`) — par incréments :
- [x] **Incr. 1 — Récolte** : boucle re-scan (taper une icône en récolte d'autres). Prio 15, cooldown 5 min.
- [~] **Incr. 2 — Upgrades** : `widget_reader` (CNN localise → digit CNN lit) + `VillageUpgrader` (gating ouvriers → `ameliorer` → confirmation → affordabilité → `confirmer_upgrade`/`annuler`). Sûr par défaut (anti-gemmes). **Attend** : re-train (classes) + LLM (décision du *quoi*).
- [ ] **Incr. 3 — Labo** : capteur prêt (`read_labs()` lit `place_labo`). Restent : figer la sémantique libre/occupé + le flux (ouvrir labo → choisir recherche → confirmer, réutilise `upgrade_building`).
- [ ] **Incr. 4 — Dons** *(nouveau)* : le chat clan liste les demandes avec un bouton `donner` (classe au dataset). Flux : détecter les `donner` **actifs** (grisé/actif par saturation, comme les sorts) → taper → pop-up → **CNN troops** reconnaît les cartes → donner. Demandes verrouillées = le jeu n'offre que les bonnes troupes, rien à lire de plus. Action bénigne (pas de whitelist nécessaire).

**Agent jeux de clan** (`clan_games/`) — 🚫 **BLOQUÉ** : les jeux de clan ne sont pas actifs en ce moment → rien à observer/labéliser/tester. À reprendre quand ils reviennent (détecter si actifs → lire les tâches → exécuter).

### V4.4 — Polish perception

- [ ] **Baseline RL en 69/56** : le checkpoint archivé `v4.4-ppo-350ep` est en 68/51 → **ne se recharge plus** (`PPOAgentV4.load()` repart de zéro **en silence**, vérifier les logs de démarrage). Refaire un run propre après les fixes deploy → nouvelle baseline (`docs/baselines.md`, `compare_baseline.py`).

### V5.1 — Résiduels multi-agents

- [ ] **ADB zéro screenshot (résiduel)** : faire lire le cache `PerceptionThread` aux consommateurs *live* (`gdc/navigator`, `social/chat`, `clan_castle`). En partie absorbé par le `world`.
- [ ] Stop le sanity-rescan dans `environment_v4._all_resources_exhausted()` (redondant avec `_sync_remaining_from_perception()`).
- [ ] **Flag perception `chat_unread`** (badge `!` près du bouton chat) → `ChatAgent.can_run` ne check qu'en présence du signal, au lieu d'ouvrir périodiquement.

### V5.0 — Mode live (phases optionnelles)

- [ ] **Phase 3** : decision tick event-driven (thread réagissant aux events `PerceptionEventBus`). Mode prod only (le RL reste sur steps discrets).
- [ ] **Phase 4** : mesurer la latence end-to-end (event → action). Cible ~150 ms.
- *Avant Ph.3* : définir les critères de « changement significatif », le comportement idle, et l'impact RL.

---

## 📅 À venir

### V5.3 — Cerveau LLM v1 (orchestrateur)

> `LocalLLMBrain(Brain)` remplace `HeuristicBrain` : décide QUEL agent lancer selon le `world`. Le seam `Brain` existe déjà (V5.1).

- [ ] `LocalLLMBrain.decide(world)` → prompt (world JSON + agents-tools + RAG minimal) → Ollama **tool-call** → agent choisi.
- [ ] **Outils village déjà prêts** : `free_builders()`, `resources()`, `upgrade_building()` (retour structuré `UpgradeResult`) → à exposer comme tools + dans le `world`.
- [ ] Stack : Ollama + **Mistral 7B** + `ollama-python` (détail → *Cerveau LLM local*).

### V5.4 — Pilotage chat + RAG complet

- [ ] `ChatAgent` (déjà là) → `LocalLLMBrain` (avec RAG) → répond / exécute / rapporte.
- [ ] RAG : **Chroma** + `nomic-embed-text` (jargon/méca CoC + contexte clan + préférences).
- [ ] ⚠️ Sécurité : chat = input **hostile** (injection) → whitelist des donneurs d'ordres + actions destructives (`exclure`, `promouvoir`, `retrograder`) derrière confirmation.

---

## 🔮 Vision long terme

### V6 — Dashboard web complet

> Prend tout son sens une fois le LLM en place : visualiser le raisonnement du cerveau + l'activité des agents + les perfs, et **contrôler**.

- [x] **Maquette + spec** : [`docs/dashboard_brief.md`](dashboard_brief.md) + [`docs/Dashboard_design_project/`](Dashboard_design_project/). ⚠️ Format Claude Design = **référence, pas déployable**.
- [ ] **Build réel** : (1) réécrire la maquette en front autonome (self-contained, sans runtime Claude Design) ; (2) **backend FastAPI + WebSocket** branché sur `build_world` / `AgentScheduler.status()` / `training_log_v4.json` + `compare_baseline.py` / cache PerceptionThread.
- [ ] Contrôle : start/stop, commandes manuelles, override.
- [ ] **Bonus pré-dashboard** : commande `--live` (fenêtre OpenCV temps réel) pour débugger la vision sans attendre le web.

### V7+ — Automatisation & intelligence

- [ ] **Combat réactif** : obs tactique post-`yolo_troops` + reward de timing → l'agent joue libre, pas scripté (détail → backlog).
- [ ] **Gestion village intelligente** : priorisation upgrades pilotée par le LLM (méta + objectifs), gestion bouclier.
- [ ] **Communication inter-agents** : l'attack agent demande des troupes au CC agent, le village négocie les ressources → bus de messages + arbitrage LLM.
- [ ] **RL — efficacité échantillons** : PPO on-policy peu efficace → **off-policy** (Rainbow/DQN, SAC discret) ou **model-based DreamerV3**. À évaluer si la convergence traîne.
- [ ] **Amélioration continue** : self-play / curriculum (HDV croissants), analyse de replays, multi-compo (LavaLoon, Hybrid, QC…), équipements héros.
- [ ] **Caméra / scroll** : suivre les troupes hors écran (sinon retraite trop tôt) ; position caméra dans l'obs.
- [ ] **Recrutement** (`inviter`) : annonces + réponses aux candidats. ⚠️ actions destructives clan exclues du périmètre auto.
- [ ] **Multi-compte**.

### Cerveau LLM local (archi + stack figés)

> 100 % local, 0 €/mois. Aboutissement de la vision : on parle à l'IA en langage naturel via le chat clan, elle supervise les sous-agents.

**Division du travail (figée)** :
- **LLM = manager/stratège** : vue globale (`build_world` + RAG), décide **QUOI/QUAND**, **coache le RL** (debrief post-attaque), parle au clan.
- **Sous-agents = yeux+mains+experts** : exécutent, **rapportent**, **escaladent** les décisions. L'agent fait le check *pas cher* (perception), le LLM tranche le *cher* (raisonnement).
- **Agent combat/RL** : reçoit compo+cible → exécute le **micro** (temps réel) → rapporte. Le LLM ne remplace pas le RL (trop lent), le RL ne remplace pas le LLM (pas de stratégie).
- **Exécution** : heuristique-guidée-par-LLM d'abord (marche tout de suite) ; RL pour l'optim micro **quand** il apporte un gain (baseline plafonne ~1.5★).
- **Canaux** : dialogue agent↔LLM = **tool-calls** ; **`.md` = carnet durable** (log décisions + mémoire RAG + instructions humaines).
- **Agents** : 5/7 faits (Combat/Chat/GdC/ClanCastle/Village) ; restent JeuxClan (🚫 bloqué) + le LLM.

**Stack (figé)** :
- **Runtime** : **Ollama** (local, offload GPU auto, tool-calling) via `ollama-python` (`localhost:11434`).
- **Modèle** : **Mistral 7B Instruct** Q4 (🇫🇷, Apache 2.0, FR natif, ~4.5 Go → tient sur GPU à côté des CNN). Upgrade : **Mistral Nemo 12B** (128k ctx). *(Qwen écarté — préférence US/EU.)*
- **Où** : **GPU** (décision toutes les qq s, pas de latence critique). RAM 64 Go réservée à un éventuel « penseur lent » (70B) plus tard.
- **Sortie** : **tool-calling** (agents = tools) → décision structurée, pas de parsing fragile.
- **RAG** : **Chroma** + `nomic-embed-text`. Indexe méca CoC (synergies sorts↔troupes, rôles), stats par niveau (wiki scrappé → pas d'hallucination), historique d'attaques (auto-alimenté), meta + données clan.
- ⚠️ **RAG, PAS fine-tuning** : le fine-tune apprend le style, pas les faits. MAJ CoC → mettre à jour la base, zéro ré-entraînement. LoRA optionnel plus tard **pour le style seulement**.

- [ ] **Intégration Ollama** (`uv add ollama`) → `LocalLLMBrain` derrière l'interface `Brain`.
- [ ] **Mode coach** : après chaque attaque, contexte → analyse NL → log ou chat clan.
- [ ] **Parole autonome** + **conseils GdC** + **rapport quotidien** dans le chat.

---

## 🗃️ Backlog (non planifié)

> Idées pas encore assignées à une version. On pioche ici quand on a du temps.

### 🔨 Hardening grisé / registre

> Architecture actée : **compteurs digit-CNN = source primaire, grisé = autorité de fin / filet**. La refonte « zéro compteur » (obs présence-par-rôle) est **abandonnée** (elle dégraderait l'obs).

- [ ] **Mask ceinture-bretelles** : autoriser `deploy(role)` tant qu'une troupe du rôle est **non-grisée**, même si le compteur lu dit 0 (protège d'une lecture basse erronée).
- [ ] **Rôles best-guess à valider** : les rôles des troupes récentes de `troops.json` sont des estimations (éditables sans code). Vérifier en jeu.
- [ ] **Sorts** : ajouter un sort change `SPELL_FEATURES` → **pas checkpoint-safe** (à gérer à part des troupes ; ajouter une troupe à un rôle existant l'est).
- [ ] **Full-auto (horizon LLM)** : classe CNN inconnue → le LLM déduit le rôle (RAG) et remplit le registre tout seul.

### 🔨 Combat — sorts & déploiement

- [~] **Spam de sorts** : `_spread_cluster_point` étale les casts cluster (fait). Restent :
  - [ ] Espacement **temporel** (l'heuristique enchaîne ; timing géré par le LLM à terme).
  - [ ] **Gel re-gèle la même défense** → `SpellCaster` doit mémoriser les défenses gelées (cooldown ~5 s) et viser la suivante.
- [ ] **Taps de deploy invalides** : le point du périmètre tombe parfois sur un bâtiment / zone rouge → tap sans effet mais compteur décrémenté. Fixes (simple→robuste) : (1) push outward, (2) snap zone déployable (masque herbe HSV / overlay rouge), (3) validation post-deploy + retry à un offset.
- [ ] **Capas des héros déployés tard jamais jouées** : l'heuristique file les `ability(i)` selon l'inventaire de départ. Fix : passe « fire abilities » **pilotée par perception** (capa non grisée + non utilisée → tirer), après `cleanup()` et périodiquement.

### 🔨 Combat réactif (moins scripté)

- [ ] **Obs tactique riche** : où sont mes troupes, où le push bloque, position relative des défenses. **Prérequis #1** (yolo_troops retrainé ✅ → débloqué).
- [ ] **Reward de timing** : rage sur troupes engagées, soin sur blessées, gel sur défense active. Aujourd'hui reward ≈ destruction/étoiles → trop pauvre pour la tactique.
- [ ] **Moins d'ancrage BC** : réduire le poids/durée du BC après un premier baseline.

### Combat intelligent

- [ ] Estimation loot avant attaque (skip si pas rentable) ; classification de base (farming/war/anti-3★) ; analyse de replays ; ligue auto / combats classés.

### Gestion village

- [ ] Queue recherche labo ; overflow ressources ; queue d'amélioration bâtiments ; gestion bouclier.

### Infrastructure & UX

- [~] Calibration UI automatique → **fusionnée dans le CNN UI V5.2** (le fallback calibré reste, par design).
- [ ] Replay vidéo des attaques (enregistrement ADB) ; comportement humain (délais/patterns) ; mode coaching.

### ML & training

- [ ] Curriculum learning ; self-play ; transfer learning ; estimation pré-attaque (% destruction prédit).

### Apprentissage continu (adaptation aux MAJ CoC)

> Human-in-the-loop, 0 €, ~1 semaine de maintenance par MAJ majeure CoC.

- [ ] Détection d'inconnus (YOLO conf < seuil → `unknown_X` + crop auto dans `needLabelisation/`).
- [ ] Maintenance mode (labéliser → réentraîner sur Kaggle) + notification des inconnus détectés.
