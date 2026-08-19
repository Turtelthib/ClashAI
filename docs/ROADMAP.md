# ClashAI — Roadmap

> **OBJECTIF FINAL** : une IA autonome intelligente qui joue comme un humain — joue, gère, recrute, s'améliore seule, et qu'on **pilote en langage naturel via le chat clan** (cerveau LLM local orchestrant des sous-agents).

**Statut** : `[ ]` à faire · `[~]` partiel · `[x]` fait (détail → [CHANGELOG](CHANGELOG.md)) · 🚫 bloqué · 🔧 bug documenté → [TROUBLESHOOTING](TROUBLESHOOTING.md)
**Mise à jour** : 19 août 2026 — **V5.2 close côté code** (CNN UI, récolte, upgrades, labo, dons validés en réel ; migration `find_button` terminée). Reste 2 validations en jeu + le renfort dataset.

📂 **Ce doc** = ce qui reste à faire. · ✅ Fait → [CHANGELOG.md](CHANGELOG.md) · 🔧 Fix détaillés → [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

**Chiffres actuels (vérifiés dans le code)** : **18 sorts** · obs **70 dims** / **57 actions** · 63 entrées `troops.json` · CNN UI **140 classes** (v4) · CNN barre **83 classes** (v2) · **236 tests**.

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
| V4.4 | 🔄 | Digit CNN ✅ validé en réel ; **reste la baseline RL en 70/57** |
| V5.0 | ✅ Ph.1-2 | Push pipeline WGC → PerceptionThread (Ph.3-4 optionnelles) |
| Refacto | ✅ | src/ layout + 13 splits (0 fichier >500L hors legacy) |
| V5.1 | 🔄 | Brain + scheduler + 4 agents ✅ ; 3 résiduels (ADB cache, sanity-rescan, chat_unread) |
| **V5.2** | 🔄 **en cours** | CNN UI ✅ (140 cl., mAP50 0.972) · Agent village : récolte ✅, upgrades ✅ **validés en réel** (le LLM décidera du QUOI) · labo ✅ **validé en réel** · dons ✅ **validés en réel** · **code V5.2 terminé** · jeux de clan 🚫 |
| V5.3 | 💡 | **Cerveau LLM v1** (orchestrateur) : `LocalLLMBrain` décide quel agent lancer |
| V5.4 | 💡 | **Pilotage chat + RAG complet** : parler à l'IA via le chat clan |
| V6 | 💡 | **Dashboard web** (maquette ✅, build réel à faire) |
| V7+ | 💡 | Combat réactif, village intelligent, amélioration continue, multi-compte |
| V END | 🎯 | IA autonome complète |

> **Séquence figée** : V5.2 (perception+agents) → V5.3 (cerveau LLM) → V5.4 (chat+RAG) → **V6 (dashboard, une fois le LLM en place)** → V7+.

---

## ⏳ En attente de validation réelle

> Code livré + testé unitairement, **pas encore confirmé en jeu**. À vider au fil des runs.

- [x] **Re-train CNN UI (17 août 2026)** : **140 classes**, mAP50 0.981 → **0.976 (v3)** après renfort des confirmations de **bâtiment** (v2 n'avait que celles du labo). Déployé en `weights/yolo_ui.pt`. **Les 27 classes référencées par le code sont toutes présentes** (vérifié) → lecture ressources/ouvriers/labo/prix + `confirmer_upgrade` + `donner` actifs.
- [ ] **Mappings CNN ambigus** : `find_match` (→ `trouver_partie_rapide`) et `gdc_open` (→ `guerre_clan`) à confirmer sur captures réelles.
- [x] **Lecture des widgets — validée en run réel (17 août 2026)** : `Ouvriers libres : 5` + `{'or': 2742878, 'elixir': 2876593, 'elixir_noire': 49739}`. Segmentation par composantes connexes, **dataset inchangé** (🔧 [TROUBLESHOOTING](TROUBLESHOOTING.md)).
- [x] **Flux upgrade — début validé en réel** : bâtiment tapé → `ameliorer` détecté → tapé → **annulation sûre** (zéro dépense, garde-fou anti-gemmes confirmé).
- [x] **Flux upgrade validé DE BOUT EN BOUT en réel (18 août 2026)** : capteurs → `ameliorer` → `prix_upgrade`/`confirmer_upgrade` détectés → prix 1 900 000 lu → ressource (or) identifiée → décision `ok` → **`--confirm` a réellement lancé l'amélioration**. 🔧 un faux `cant_afford` (couleur en RGB) corrigé au passage → [TROUBLESHOOTING](TROUBLESHOOTING.md).
- [x] **Récolte validée en réel (19 août 2026)** : `Village : récolte effectuée (3 tap(s))` — la boucle re-scan s'arrête bien quand il n'y a plus d'icône.
- [ ] **Abandon state-dependent** : `capituler`+`confirmer` vs `terminer_bataille`.
- [x] **Digit CNN sur les gros nombres — validé (19 août 2026)** : 7 chiffres lus correctement en run réel (4 540 057 · 3 919 089 · 2 742 878). La crainte d'une police différente des badges de troupes ne s'est pas matérialisée — le chemin widget (composantes connexes) la gère.

---

## 🚀 En cours

### V5.2 — CNN UI + agents village

> CNN UI **livré** (détecteur universel de boutons) + agents à base de règles. Détail du fait → [CHANGELOG](CHANGELOG.md).

**CNN UI** — le socle est en place (`UIDetector` branché au démarrage, `find_button()` = point d'accès unique).
- [x] **Migration vers `find_button()` terminée (19 août 2026)** : **plus aucun `get_position()` direct** hors de la calibration elle-même. Les `try/except ImportError` avec tables de coordonnées de secours dupliquées sont supprimés (`find_button` ne lève jamais et porte déjà les défauts). **`brain/navigation` passe au CNN** (`screenshot=img`) : le retour au village tape le vrai bouton détecté — notamment `rentrer` sur l'écran de résultats, où l'on tapait **4 hauteurs à l'aveugle**. Ailleurs, migration **neutre** (sans screenshot) : vérifié `find_button(k) == get_position(k)` sur les 11 clés.
- [ ] **Renfort dataset** : classes rares (1-2 exemples) ratées, confondues avec `background` → ajouter des captures des cas ratés.

**Agent village** (`village/`, `VillageAgent`, règles ; clique via `UIDetector`) — par incréments :
- [x] **Incr. 1 — Récolte** : boucle re-scan (taper une icône en récolte d'autres). Prio 15, cooldown 5 min.
- [x] **Incr. 2 — Upgrades (mécanisme complet, validé en réel)** : `widget_reader` (CNN localise → digit CNN lit) + `VillageUpgrader` (gating ouvriers → `ameliorer` → confirmation → affordabilité → `confirmer_upgrade`/`annuler`). Sûr par défaut (anti-gemmes). **Reste au LLM de décider QUOI améliorer** (V5.3) — les outils sont prêts.
- [x] **Incr. 3 — Labo (validé en réel, 18 août 2026)** : `village/lab.py` (`VillageLab`) — labo localisé par le **CNN bâtiments** (aucune coordonnée en dur) · `rechercher` ouvre la grille · les cartes sont **nommées par le CNN barre de troupes** et triées couleur/gris par saturation · la confirmation est **déléguée à `VillageUpgrader.confirm_step`** (garde-fou anti-gemmes partagé). Run réel : 7 améliorables, sapeur cliqué, annulation sûre — puis **`--confirm --troupe gobelin` a réellement lancé la recherche**, prix et troupes corrects dans les logs. 🐛 prix **rouge** (solde insuffisant) illisible → masque étendu + garde-fou autoritatif `price_is_red`. Démo `tools/debug/village_lab_demo.py` (`--scan`). 16 tests.
  - [x] **CNN barre de troupes validé sur l'écran labo** : 6 vignettes reconnues (conf 0.81-0.94) — `geant`, `barbare`, `sapeur`, `guerrisseuse`, `gobelin`, `mineur`. **Aucun labeling nécessaire.**
  - [x] **Prix des cartes du labo : résolu (19 août 2026)**, confirmé en réel. Les 3 lectures ratées étaient les prix **non payables**, donc écrits en **rouge** → invisibles pour l'ancien masque « texte blanc ». Les correctifs faits depuis (**composantes connexes** + **masque rouge**) portent sur ce même chemin `read_widget_number` et l'ont réparé sans travail dédié. Rien à cadrer, rien à re-labelliser.
  - [x] **`rechercher` renforcé (CNN UI v4, 18 août 2026)** : mAP50 0.972, F1 **0.91** (pic à conf 0.635 — le seuil d'action 0.60 tombe pile dessus). 140 classes, les 28 utilisées par le code présentes. → **flux labo complet à re-tester**.
  - [x] **Re-train barre de troupes déployé (18 août 2026)** : **83 classes** (79 → 83). Le sort **`colere`**, jusque-là pré-enregistré mais absent du CNN (donc **inerte**), est désormais reconnu → il s'active tout seul : **18 sorts**, obs **69→70**, actions **56→57**. Le design « registre ∩ classes CNN » a fonctionné exactement comme prévu, sans une ligne de code. ⚠️ **Re-train RL requis** — mais aucune perte : tous les checkpoints existants étaient déjà en 68/51, donc périmés avant ce changement.
- [x] **Incr. 4 — Dons (validé en réel, 19 août 2026)** : `social/donations.py` (`DonationManager`) — boutons `donner` **actifs** seulement (les demandes satisfaites sont grisées) · cartes du pop-up nommées par le CNN barre de troupes · **géométrie du pop-up confirmée** (le filtre `min_x` sépare bien les cartes à donner des icônes de la demande) · dons **répartis** entre les troupes proposées. Run réel : chat ouvert, onglet gratuit sélectionné, 6 dons effectués. Démo `tools/debug/donations_demo.py` (`--scan`). 18 tests.
  - 🛡️ **Sécurité gemmes** : le pop-up a deux onglets, `dons_normaux` (gratuit) et `dons_gemme` (**coûte des gemmes**). On ne *devine* pas lequel est actif : on **tape explicitement `dons_normaux`** (gratuit, idempotent) et on **abandonne sans rien donner** s'il est introuvable. `dons_gemme` n'est jamais tapé — testé même quand il est détecté avec une confiance supérieure.
  - [x] **Validé en réel (18 août 2026)** : chat ouvert, 1 demande détectée, onglet gratuit sélectionné, **6 dons effectués**.
  - 📌 **Filtrage par icônes abandonné** : quand une demande verrouille des troupes, **le jeu l'impose déjà** (les autres cartes sont grisées) → filtrer par-dessus n'ajoute rien et peut retrancher à tort. `read_request()` reste un **capteur informatif** (utile au LLM) ; le vrai besoin est l'OCR du message (→ V5.4).
  - [x] **Dons répartis entre les troupes proposées** : la politique « toujours la 1ʳᵉ carte » martelait une seule troupe (sur « ballon + sorcière », que des ballons). Remplacée par « la troupe la **moins donnée** jusqu'ici » → couvre les demandes mixtes sans lire les quantités. `MAX_TAPS_PER_REQUEST` 6 → **30** (une demande peut réclamer ~45 places d'armée ; 6 tronquait « 2 ballons + 3 sorcières + 2 zap »). Garde-fou de stagnation.
  - [x] **Fin de don gérée par le jeu** : quand le château du membre n'a plus la place pour une troupe (un électro-dragon prend 30 places, il en reste 20 → il se grise), le jeu grise cette carte ; **tout grisé = château plein**. Notre boucle s'arrête déjà sur « plus rien de donnable » → condition de fin correcte **sans code supplémentaire**, et c'est une raison de plus de filtrer sur le grisé.

**Agent jeux de clan** (`clan_games/`) — 🚫 **BLOQUÉ** : les jeux de clan ne sont pas actifs en ce moment → rien à observer/labéliser/tester. À reprendre quand ils reviennent (détecter si actifs → lire les tâches → exécuter).

### V4.4 — Polish perception

- [ ] **Baseline RL en 70/57** : le checkpoint archivé `v4.4-ppo-350ep` est en 68/51 → **ne se recharge plus** (`PPOAgentV4.load()` repart de zéro **en silence**, vérifier les logs de démarrage). Refaire un run propre après les fixes deploy → nouvelle baseline (`docs/baselines.md`, `compare_baseline.py`).

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
- [ ] **OCR par MESSAGE plutôt que par zone** — *prérequis d'un vrai dialogue*. Aujourd'hui l'OCR lit **un seul rectangle en dur** (`chat/constants.py` : 0-850 × 60-980) et rend un bloc de texte indifférencié : impossible de savoir qui a dit quoi. Il faut **une nouvelle classe CNN pour les bulles de message des membres** (à labelliser) — ⚠️ `message_clan` **n'est PAS** ça : c'est le bouton du menu château pour écrire au clan. Ensuite : détecter chaque bulle → OCR individuel → auteur + texte + ordre. Même pattern « CNN localise → lecteur lit » que les compteurs/prix/cartes labo.
  - **Débloque aussi les dons intelligents** : une demande écrite en toutes lettres (« il me faut des sapeurs et des ballons ») sans verrouillage laisse le jeu tout accepter → seul le texte dit quoi envoyer. `DonationManager.donate_to_request(wanted=…)` attend déjà cette liste.
- [ ] RAG : **Chroma** + `nomic-embed-text` (jargon/méca CoC + contexte clan + préférences).
- [ ] **Dons intelligents sur demande ÉCRITE** : quand le membre écrit « 3 sorcières + 2 ballons + 1 soin + 1 zap » **sans que le jeu verrouille** ces troupes, seul l'OCR du message dit quoi envoyer — et **en quelle quantité**. Le LLM doit fournir exactement ça, ni plus ni moins. `DonationManager.donate_to_request(wanted=…)` attend déjà la liste ; reste à lire le texte (dépend de l'OCR-par-message ci-dessus) et à **respecter les quantités** (aujourd'hui on répartit à l'aveugle jusqu'à ce que le jeu refuse).
- [ ] **Scroll horizontal** — grille du **labo** ET pop-up des **dons** : des troupes/sorts restent hors écran, donc jamais choisis. Même mécanisme pour les deux (swipe + déduplication entre pages).
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
