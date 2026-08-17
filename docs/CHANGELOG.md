# ClashAI — Changelog (tout ce qui est fait)

Historique chronologique des features livrées, du plus récent au plus ancien.

> Ce qui reste à faire : [ROADMAP.md](ROADMAP.md). Blocs de fix détaillés : [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

## Légende
✅ livré · 🐛 bug corrigé au passage · 🔧 → bloc détaillé dans TROUBLESHOOTING.md

---

## V5.2 — CNN UI + agents village (en cours)

> **CNN UI livré et branché** · **Agent village** : récolte livrée, upgrades livrés (attendent le re-train des classes + le LLM pour décider). Restent : labo, dons, et l'agent jeux de clan (🚫 jeux inactifs) → [ROADMAP](ROADMAP.md). *Cette section couvre aussi l'outillage & l'audit qui ont ouvert le cycle.*

- ✅ **Agent village — incrément 2 : upgrades (mains + yeux pour le LLM)** — la **décision** (quoi améliorer) reviendra au LLM (V5.3) ; on livre le **geste** + les **capteurs proactifs**, conçus comme des outils appelables.
  - **Pattern unifié « CNN localise → digit CNN lit »** (`perception/widget_reader.py`) : le CNN UI donne la position d'un widget, le digit CNN lit le nombre dedans. `read_resources` (`compteur_or/elixir/elixir_noire`), `read_builders` (`nombre_ouvrier`, "N/M" → coupe au milieu pour éviter le `/`), `read_widget_number` (générique, ex. `prix_upgrade`). Réutilise `digit_reader` — nouveau `read_number(drop_leading_x=…)` généralise `read_count` (les nombres hors badges de troupes n'ont pas de `x` de tête). Convertit les coords ADB du détecteur → pixels image pour cropper.
  - **Exécuteur** (`village/upgrader.py`, `VillageUpgrader.upgrade_building`) : constructeur libre ? → tap bâtiment → bouton `ameliorer` ? → écran de confirmation (lit prix + ressources) → **décision d'affordabilité** → `confirmer_upgrade` / `annuler`. `confirmer_upgrade` = le bouton de confirmation avec prix (labo + bâtiment), **distinct** du `ameliorer` du menu et du `confirmer` générique. Statuts : `ok` / `no_builder` / `not_upgradeable` / `cant_afford` / `need_decision`.
  - **Sûr par défaut (anti-gemmes)** : sans affordabilité **prouvée** (prix lu + ressource cible, ou `confirm_decider` fourni par le LLM), on **annule** — jamais de `confirmer` à l'aveugle qui déclencherait le pop-up « acheter des gemmes ». `read_builders=None` (classe absente) n'infère **pas** 0 → ne bloque pas le flux.
  - **Ressource d'un prix lue par COULEUR** (`classify_resource_color` / `read_price_resource`) : l'icône (pièce/goutte) à droite du prix apparaît **partout** dans le jeu → en faire une classe CNN obligerait à la labéliser sur tous les écrans (même visuel = même classe) pour un gain nul. On compte les pixels proches de 3 couleurs de référence (doré / rose / violet sombre) ; la plus représentée gagne, le fond (bouton vert, texte blanc) ne matche rien et s'ignore. Validé sur variations d'ombrage claires/ombrées des 3 ressources. Conséquence : **l'affordabilité devient autonome** — `upgrade_building` sans `resource_type` lit la ressource à l'écran et décide seul (avant : `need_decision`).
  - **Capteurs = futurs outils LLM** : `free_builders()`, `resources()`, `upgrade_building()` avec retour structuré (`UpgradeResult`). Démo codée en dur `tools/debug/village_upgrade_demo.py` (sans `--confirm` : va jusqu'à l'écran de confirmation puis annule → teste le flux **sans dépenser**).
  - **Dépend d'un re-train** : les classes `compteur_*`, `nombre_ouvrier`, `prix_upgrade`, `confirmer_upgrade` arrivent côté dataset → le code est écrit à ces noms et **s'allume au re-train** (lecture = None/{} d'ici là, aucun plantage). `ameliorer`/`annuler` déjà dans les 124 → le flux jusqu'à l'écran de confirmation (+ annulation sûre) est **testable en réel dès maintenant** ; la confirmation réelle attend `confirmer_upgrade`.
  - **22 tests** (`test_widget_reader.py` + `test_village_upgrader.py`, digit CNN + I/O mockés) : lecture ressources/ouvriers/labo, scaling ADB→image, classification couleur (3 ressources + rejet du fond), affordabilité autonome (accepte/refuse), chaque statut, et l'invariant **anti-gemmes** (`confirmer_upgrade` jamais tapé sans décision). **184 tests.**
- ✅ **Agent village — incrément 1 : récolte des ressources** — premier agent V5.2 à base de règles, exploite directement le CNN UI. `village/collector.py` (`VillageCollector`) détecte les icônes de récolte (`recolter_or` / `recolter_elixir` / `recolter_elixir_noire`) ; `agents/village_agent.py` (`VillageAgent(BaseAgent)`) l'ordonnance.
  - **Boucle re-scan (méca CoC)** : taper **une** icône de récolte en récolte automatiquement d'autres → on ne tape PAS toutes les positions détectées d'un coup (après le 1er tap, les autres icônes ont disparu et leur ancienne position taperait un bâtiment au hasard). On boucle : capture **fraîche** → tape 1 icône → attend (0.6 s) → re-scanne, jusqu'à écran vide. Robuste aux deux cas (un tap vide tout → 1 passe ; un tap vide un seul → plusieurs). Garde-fous : 8 passes max + arrêt si la même icône réapparaît au même endroit (tap sans effet).
  - **Sûr par construction** : ne fait que taper des collecteurs pleins (aucune dépense, idempotent). No-op propre si le CNN UI est absent (`detect_raw` vide → 0 tap, aucune erreur) ou si pas de frame.
  - Priorité **15** — entre `clan_castle` (20) et `combat` (10) : la récolte passe avant une attaque quand son cooldown (5 min) est écoulé, puis rend le sol au combat. Même raisonnement que le cooldown de `ClanCastleAgent`.
  - Réutilise le détecteur déjà branché (`ui_buttons.get_detector()`) — pas de second chargement du YOLO ; fallback instance dédiée. Pattern `ClanCastleAgent`→`ClanCastleManager` : wrapper mince dans `agents/`, logique dans `village/` (extensible upgrades/labo). Enregistré dans `brain/core` (tous modes).
  - **10 tests** (`test_village_agent.py`, détecteur *séquencé par frames* qui simule la disparition des icônes) : tap-clears-all → 1 tap, tap-clears-one → N taps, arrêt icône bloquée, no-op écran vide / frame absente, ignore les non-`recolter_*`, gating `village_home`, cooldown, priorité village↔combat. **162 tests.**
- ✅ **CNN UI universel livré + branché** — le YOLO « boutons / éléments d'interface » (**124 classes**, mAP50 **0.957** / mAP50-95 **0.822**, imgsz 1280) reconnaît les boutons par leur apparence + leur **texte**. Matrice de confusion à **diagonale propre : aucune confusion bouton↔bouton** (les rares ratés partent en `background` = classes à 1-2 exemples). Seuil de conf optimal ~0.48 (courbe F1 = 0.83).
  - `perception/ui_detector.py` : `UIDetector` à deux niveaux — `detect_raw()` → `{classe_cnn: [Detection]}` (brut, noms **français**, primitive des futurs agents village/GdC) et `detect()` → `{clé: (x,y,conf)}` (contrat `set_detector`). Noms lus **depuis le modèle** (`model.names`), jamais une liste en dur.
  - **Désambiguïsation des boutons génériques par position** : `fermer`/`confirmer` apparaissent à plusieurs endroits ; pour une clé de calibration on prend l'instance **la plus proche** de sa position calibrée (`close_profil` haut-droite vs `close_popup` centre se démêlent sans classe dédiée).
  - **Branché au démarrage** (`brain/core._load_modules`) : `set_detector(UIDetector())` si `weights/yolo_ui*.pt` présent, sinon **calibration seule** (aucune régression). Le YOLO se charge en **lazy** au 1er `detect()`. Le `find_button()` existant l'utilise sans qu'aucun appelant ne change — la bascule promise en un seul appel a tenu.
  - Mapping clé↔classe (16 sûrs) : `find_match→trouver_partie_rapide`, `gdc_attack_target→attaquer_guerre`, `gdc_enemy_map→voir_enemis`, `gdc_ally_map→voir_allie`, `gdc_village_next/prev→village_suivant/precedent`… Clés ambiguës laissées au fallback (pas de tap au hasard).
- ✅ **Abandon *state-dependent* (CNN)** — `env._surrender()` lit l'écran : troupes vivantes → `capituler` + `confirmer` (popup) ; aucune troupe → `terminer_bataille` (direct, sans popup). Remplace le double-tap en dur `ff_button`/`confirm_ff` ; la calibration reste le **filet** si détecteur absent / sous le seuil de confiance. 152 tests toujours verts.
- ✅ **Prep CNN UI : point d'accès unique `find_button()`** — dérisque V5.2 **avant** d'entraîner le YOLO. Aucun changement de comportement (mêmes positions rendues), **~400 lignes de duplication supprimées**.
  - `perception/ui_buttons.py` : `find_button(name, screenshot=None)` essaie le détecteur (installé par `set_detector()`) puis retombe sur la position calibrée. Fallback couvrant **les quatre** cas d'échec : détecteur absent, bouton non détecté, confiance < 0.60, détecteur qui lève. Une inférence ratée ne doit jamais arrêter une navigation. Coordonnées converties en `int` (YOLO rend des flottants, ADB veut des entiers).
  - **La bascule V5.2 sera un seul `set_detector()` au démarrage**, zéro appelant à modifier — c'était tout l'objectif.
  - `navigation/gdc/constants._get_ui_pos` (consommé 23× par `gdc/navigator.py`) délègue désormais ; sa table de **17 défauts recopiés** est supprimée. C'était le seul endroit du projet où les défauts pouvaient diverger sans que personne ne le voie.
  - 🐛 **Fork `tools/setup/calibrate_ui.py` éliminé** (404 lignes → lanceur de 20). Il avait un groupe `cdc` **absent** de la version package — or c'est la version package que la prod importe (`social/clan_castle.py` lit `get_position('cdc_confirmation')`), donc recalibrer via `python -m clashai.navigation.calibrate_ui` ne pouvait **jamais** recalibrer ce bouton, alors que les deux écrivaient le même JSON. Groupe `cdc` porté dans le package + défaut ajouté ; `main()` extrait pour que le lanceur l'appelle.
  - ⚡ `load_positions()` mis en **cache sur le mtime** : `get_position()` relisait et re-parsait le JSON à **chaque appel** (~24 lectures disque pour une seule navigation GdC). Mesuré : 2000 appels en **13 ms**. Une recalibration change le mtime → prise en compte sans redémarrage (vérifié).
  - 13 tests (`test_ui_buttons.py`), dont la dégradation sur détecteur en panne et l'invariant « tout bouton calibrable a un défaut ». **152 tests.**
  - 📌 *Correction de l'audit* : le §3.6 annonçait « 3 tables de défauts concurrentes et divergentes ». Vérifié : elles **ne divergent pas** — les 17 clés communes sont identiques et celle de `gdc/` est un sous-ensemble strict. L'écart cité (`chat_open`) comparait le JSON *calibré* aux *défauts*, ce qui est le fonctionnement normal. C'était de la duplication, pas un bug de correction.
- ✅🐛 **Rôle `clean` + fix `double_canon`** — ⚠️ **les deux changent l'observation → re-train requis**. Groupés exprès dans un seul commit pour n'imposer **qu'un** re-train au lieu de deux.
  - **Nouveau rôle de deploy `clean`** (6ᵉ) : `sorciere_ruine` portait le rôle `clean`, absent de `DEPLOY_ROLES` → l'unité n'était **jamais déployable** (item 1.4 de l'audit). Le rôle devient légitime, car c'est la seule troupe du jeu qui **n'attaque pas** : elle invoque des troupes quand un bâtiment tombe, donc la poser au début la gaspille.
  - **Choix d'archi (utilisateur)** : **pas de masque**. L'agent reste libre de la déployer quand il veut ; c'est `reward_shaping` qui oriente. Malus **proportionnel à la précocité** (−6 à 0 % de destruction, atténué jusqu'au seuil `CLEAN_DESTRUCTION_MIN = 0.20`, +4 au-delà) plutôt qu'une falaise → PPO a un gradient exploitable. Cohérent avec la ligne « moins d'ancrage BC / agent libre » de la ROADMAP. Signal lu dans `combat_features[0]` (bâtiments restants/initiaux) — **aucun changement de signature**.
  - **Heuristique** : deploy `clean` en dernier, après une pause, pour que le BC enseigne le bon timing.
  - Dimensions : actions **51 → 56**, obs **68 → 69**, tout dérivé automatiquement (le design data-driven a tenu, seul `DEPLOY_ROLES` a été touché). Nouveaux rôles à ajouter **en fin de liste** : l'heuristique code en dur les indices 0-4.
  - 🐛 **`canon_double` → `double_canon`** (item 1.2) : le nom du code ne correspondait pas à ce que le CNN émet (`weights/classes.json`) → le canon double n'était mappé sur **aucun canal** de la grille RL et n'avait **aucune stat de danger**. Invisible pour l'agent. Vérifié : 0 classe CNN non mappée après fix, nombre de canaux inchangé (9).
  - Tests : `test_encoder_vocabulary.py` (toute classe émise par le CNN doit être mappée — l'invariant qui aurait attrapé le bug), règles de timing du `clean`, et `test_heuristic_sequence.py` (ordre de deploy du prof du BC). Le `xfail(strict=True)` posé la veille est passé en `XPASS` en échec et a forcé le retrait du marqueur, comme prévu. **139 tests.**
- ✅ **Tests de logique pure** (`tests/`, **26 → 122 tests**, 3,7 s, toujours zéro matériel). Couvre les 3 cibles à plus fort ROI de l'audit :
  - `social/chat/parser.py` — 9 formats de commande (avec/sans `@`, underscore ou espace, `attack`/`attaque`/`atk`/`att`, casse), les 11 synonymes `stop`/`status`/`reset`, les rejets (pas de mention du bot, hors bornes 1-50, commande inconnue), les 9 formats d'horodatage CoC, et les **corrections d'erreurs OCR** (`IImin`→11, `l1min`→11, `Ih`→60) qui dépendent d'un nettoyage regex fait **avant** le passage en minuscules.
  - `combat/troop_registry.py` — règle « les sorts ignorent le `max` du JSON » (cause du sous-cast gel=1/rage=3 corrigé en Session 14), exclusion des sorts du mapping de deploy, unicité des noms, cohérence du `_FALLBACK` codé en dur, et validité des cibles de `SPELL_TARGET_DEFAULTS`.
  - `combat/reward_shaping.py` — signe de 17 constantes (une inversion sabote le RL sans jamais lever), règles tank-first/héros-après-tank/concentration-vs-spread, paliers du soin, rage conditionnée aux troupes vivantes, timing des capas roi/gardien, et robustesse à `combat_features=None`.
  - **Bug connu encodé en `xfail(strict=True)`** : `sorciere_ruine` a le rôle `clean` dans `configs/troops.json`, absent de `DEPLOY_ROLES` → unité jamais déployable (item 1.4, toujours ouvert car il change le comportement). Le jour où le rôle est corrigé, pytest signale un `XPASS` en échec et force à retirer le marqueur.
  - Les 3 fichiers ont été **validés par mutation** (signe d'une reward, borne du parser, bridage des sorts) : chaque mutation produit bien un échec, et chacune a été revertée.
- ✅ **Première suite de tests automatisés** (`tests/`, **26 tests, 1,6 s**, zéro émulateur/ADB/GPU/poids). Le projet déclarait `pytest` en dev-dep depuis toujours sans qu'un seul test existe.
  - **Aucun fichier de production modifié.** Les blocs `if __name__ == "__main__"` des agents contenaient déjà de vrais `assert` et de vrais fakes, mais rien ne les exécutait. Ils sont **reflétés** dans `tests/`, pas déplacés — ils restent utilisables en démo manuelle (`python -m clashai.agents.chat_agent`). Contrepartie assumée : un changement de comportement d'agent est à répercuter à deux endroits.
  - Couvre l'ordonnancement (priorités chat 30 > gdc 25 > cc 20 > combat 10, cooldowns, gating par mode), le cycle `world → can_run → pick → run`, le seam `Brain.decide()` (contrat que devra respecter `LocalLLMBrain` en V5.3), le flux chat → file GdC → routage (futur canal d'entrée V5.4), la sérialisabilité JSON de `scheduler.status()` (contrat dashboard V6) et la **bijection `encode_action`/`decode_action`** sur les 51 actions.
  - **Piège pytest neutralisé sans renommer un seul fichier** : `[tool.pytest.ini_options] testpaths = ["tests"]`. Sans ça, un `pytest` à la racine collecte `src/tools/debug/test_deploy.py`, qui n'a **aucune garde `if __name__`** → l'import charge les modèles sur GPU puis pilote l'émulateur. Le renommage en `debug_*.py` reste souhaitable mais n'est plus urgent.
  - 🐛 **Défaut de conception corrigé dans les tests eux-mêmes** : la suite a été validée par mutation (`combat.priority` 10 → 99). Premier essai → pytest **a bloqué 2 minutes**, parce que `scheduler.run(scheduler.pick(...))` sélectionnait `CombatAgent`, dont le `run()` lance une vraie attaque et attend ADB. D'où `conftest.no_hardware()`, qui neutralise **le seul `run()`** en gardant `priority`/`cooldown_seconds`/`can_run()` réels. Même mutation désormais : **7 échecs d'assertion en 0,10 s**. Procédure notée dans `tests/README.md`.
- ✅🐛 **Lot « bugs neutres » de l'audit** (items 1.1, 1.5, 1.6, 1.7, 1.8) — aucun effet sur le comportement, le balayage d'imports passe de **137 ok / 1 fail à 137 ok / 0 fail** :
  - **1.1** `perception/building_detector.py` **supprimé** : orphelin (zéro import dans tout le dépôt) et déjà cassé — il exécutait `print`/`exit()`/`YOLO()`/`torch.load`/`predict`/`imwrite` **au niveau module**, sur un chemin `runs/detect/FinishedTrain/` inexistant. C'était l'unique échec du balayage d'imports. Il dupliquait par ailleurs intégralement `game_loop/models.py`.
  - **1.5** `import sys` manquant dans `tools/debug/detect_troop_bar.py` → les deux `sys.exit()` levaient un `NameError` (ruff `F821`).
  - **1.6** `uv add rich` : `config/logging.py` importait `rich` en top-level sans qu'il soit déclaré (présent seulement en transitif via `typer` — un `uv lock` pouvait le faire disparaître et casser les 17 modules qui en dépendent). `uv.lock` ne bouge que de 2 lignes, aucun paquet ajouté.
  - **1.7** `.gitignore` : `logs/` → `/logs/`. Le motif non ancré masquait aussi `configs/logs/theme.yaml`, le thème Rich lu par `config/logging.py:53` → désormais versionné. `logs/` à la racine reste bien ignoré.
  - **1.8** `.gitignore` : `test_deploy.py`/`test_gpu.py` → `/test_deploy.py`, `/test_gpu.py`. Les motifs non ancrés masquaient `src/tools/debug/`, où ces deux outils existaient sur disque sans jamais être versionnés. **Effet de bord utile** : ruff respecte le `.gitignore`, ces fichiers n'avaient donc **jamais** été lintés — 5 violations mécaniques corrigées dans la foulée. ⚠️ Leur renommage en `debug_*.py` (piège pytest) reste à faire.
- ✅🐛🔧 **Passe ruff sur `src/`** : 456 → **168 violations** (285 corrigées : tri d'imports, espaces, `f""` sans placeholder). Le linter était configuré depuis toujours et n'avait jamais tourné. **Deux régressions découvertes et évitées au passage** → bloc [TROUBLESHOOTING](TROUBLESHOOTING.md#-ruff---fix-casse-le-code) : (1) `F401` **est** auto-corrigé malgré le marqueur `[-]`, et a supprimé un ré-export (`DEBUG_DIR` dans `reward_reader/constants.py`) consommé par `percentage.py`/`stars.py` → F401 exclu de la passe, à traiter fichier par fichier plus tard ; (2) le fix `F811` sur `game_loop/analysis.py` a réexposé un import local enfoui en fin de fonction qui rendait `INFERENCE_LOCK` **local à toute la fonction** → `UnboundLocalError` dans `analyze_village()`, la perception plantait à la première frame. Corrigé en retirant le dernier import résiduel. Restent 168 violations dont **115 F401** volontairement différées ; le reste = 38 `E501`, 9 `E402`, 2 `F821` (vrai `NameError`, item 1.5 de l'audit), 2 `E741`, 1 `E722`, 1 `E731`.
- ✅ **Audit projet complet** (5 août 2026) : 4 sous-agents en parallèle (duplication/SSOT, architecture/couplage, qualité Python, config/doc) → [`docs/AUDIT_2026-08-05.md`](AUDIT_2026-08-05.md), ~60 constats classés par gravité + plan en 7 phases. Fichier **temporaire**, à dissoudre dans ROADMAP/TROUBLESHOOTING au fil du tri. Chiffres corrigés au passage : les **17 sorts / 68 dims / 51 actions** réels (la doc annonçait 16/67/50) ; la baseline `v4.4-ppo-350ep` est bien **compatible** (checkpoint vérifié en 68/51), seul son texte de description était faux.
- ✅ **Refonte des skills Claude Code** : les 8 `.md` à plat dans `.claude/skills/` n'étaient **chargés par personne** (Claude Code exige `<nom>/SKILL.md` + frontmatter YAML). 7 skills convertis et désormais actifs — `project-mapper`, `refacto-architect`, `roadmap-manager`, `uv-workflow`, `yolo-expert`, `stop-slop`, `context-engineering`. `UI-UX.md` (README d'un installeur tiers, pas un skill) déplacé en `docs/vendor/`. Liens morts retirés (`references/*.md` de stop-slop, 15 sous-skills de context-engineering). Chaque skill projet gagne une commande de vérification manuelle.
- ✅ **`PROJECT_MAP.md`** créé à la racine (attendu par `project-mapper` depuis toujours) : arborescence annotée de `src/`, sens des dépendances, points d'entrée, repères chiffrés vérifiés.
- ✅ **`.claude/skills/` versionné** (`.gitignore` ignorait tout `.claude/`) ; `settings.local.json` reste exclu.

---

## V5.1 — Foundation multi-agents (en cours)

Plomberie pour les sous-agents. Posé à côté du système existant → le bot tourne identique tant que `brain.py` n'est pas branché sur le scheduler.

- ✅🐛 **yolo_troops retrainé + abandon basé CNN** (Session 15) : nouveau modèle troupes terrain (408 img / 51 classes, Kaggle, **mAP50 0.82**, troupes courantes 0.75-0.95) déployé en `weights/yolo_troops.pt` → **corrige le placement rage** (cluster réel détecté, validé en run : 3 rages sur les dragons ✅). Et `wait_for_battle_end` remplace le comptage des barres de vie vertes (troupe blessée = comptée morte → abandon à 99%) par le comptage **yolo_troops** (`_count_field_troops`, exclut ennemis, fallback barres) → abandon seulement après 4 scans à 0 troupe. Script Kaggle standalone `tools/train/kaggle_train_yolo_troops.py`. **+ fix** : `TroopDetector` lit les noms de classe depuis le **modèle** (`self._model.names`) au lieu d'une liste codée en dur → les 51 classes sont bien nommées (avant, indices >12 en `unk_N` → split héros/troupe + exclusion ennemis cassés).

- ✅ **Baseline RL figé + outil de comparaison** (Session 15) : le run PPO brut (350 ép, obs 67/actions 50) a **plateauté ~1.7★ / 53% de 2★+** = niveau BC/heuristique → le RL brut ne casse pas le plafond (confirme que le levier est le cerveau LLM, pas plus d'épisodes). Archivé `weights/baselines/v4.4-ppo-350ep/` (log+checkpoint+`stats.json`, local car `weights/` gitignoré) ; trace git `docs/baselines.md` ; outil `tools/train/compare_baseline.py` compare un run au baseline (côte à côte + delta), sans re-déduire.
- ✅ **Validation en conditions réelles (Session 15)** : rework des sorts + seed digit-CNN testés en run réel → OK. Re-train lancé sur la nouvelle obs d'alors. ⚠️ *Chiffres d'époque périmés : ce run visait 67/50 ; le code est passé à **68/51** (rework sorts) puis **69 dims / 56 actions** (rôle `clean`). Valeurs courantes toujours à relire dans le code, jamais à recopier d'ici.*
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
