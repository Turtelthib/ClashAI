# ClashAI — Roadmap

> **OBJECTIF FINAL** : une IA autonome intelligente qui joue comme un humain — joue, gère, recrute, s'améliore seule, et qu'on **pilote en langage naturel via le chat clan** (cerveau LLM local orchestrant des sous-agents).

**Statut** : `[ ]` à faire · `[~]` partiel · `[x]` fait (détail → [CHANGELOG](CHANGELOG.md)) · 🚫 bloqué · 🔧 bug documenté → [TROUBLESHOOTING](TROUBLESHOOTING.md)
**Mise à jour** : 27 août 2026 — **agent jeux de clan livré, 3 chemins à valider en jeu** (CNN UI v5 + reader livrés). **V5.3 démarrée** : cerveau LLM + CNN UI continu livrés. **V5.2 close côté code** (CNN UI, récolte, upgrades, labo, dons validés en réel ; migration `find_button` terminée). Reste 2 validations en jeu + le renfort dataset.

📂 **Ce doc** = ce qui reste à faire. · ✅ Fait → [CHANGELOG.md](CHANGELOG.md) · 🔧 Fix détaillés → [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

**Chiffres actuels (vérifiés dans le code)** : **18 sorts** · obs **70 dims** / **57 actions** · 63 entrées `troops.json` · CNN UI **155 classes** (v5) · CNN barre **83 classes** (v2) · **479 tests**.

---

## Sommaire

- [📊 État des versions](#-état-des-versions)
- [⏳ En attente de validation réelle](#-en-attente-de-validation-réelle)
- [🚀 En cours](#-en-cours)
  - [V5.2 — CNN UI + agents village](#v52--cnn-ui--agents-village)
  - [V4.4 — Polish perception](#v44--polish-perception)
  - [V5.1 — Résiduels multi-agents](#v51--résiduels-multi-agents)
  - [V5.3 — Cerveau LLM v1 (orchestrateur)](#v53--cerveau-llm-v1-orchestrateur)
  - [V5.0 — Mode live (phases optionnelles)](#v50--mode-live-phases-optionnelles)
- [📅 À venir](#-à-venir)
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
| **V5.2** | 🔄 **en cours** | CNN UI ✅ (140 cl., mAP50 0.972) · Agent village : récolte ✅, upgrades ✅ **validés en réel** (le LLM décidera du QUOI) · labo ✅ **validé en réel** · dons ✅ **validés en réel** · **code V5.2 terminé** · **jeux de clan 🔄 en cours** : CNN UI v5 (155 cl.) + reader + catalogue + selector + agent ✅, **démarrage du bot OK**, reste la VALIDATION EN JEU |
| V5.3 | 🔄 **en cours** | Cerveau LLM **actif en réel** (Mistral 7B, décisions 0,4-2,5 s) + console de discussion + CNN UI continu ; reste à enrichir le `world` |
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
- [ ] 🔴 **Re-calibrer `DETECTOR_MIN_CONFIDENCE` pour le CNN v5** — le seuil d'action global vaut 0.60 et son commentaire dans `ui_detector.py` le justifie par le pic F1 du **v4 (0.635)**. **Le v5 a son pic à 0.332** : le raisonnement est périmé, 0.60 est passé du centre du plateau à son bord droit. Déjà coûté un bug (🔧 `commencer_introuvable`, où la classe sortait à 0.496 en visant le bouton exactement). **D'autres agents peuvent rater des boutons en silence pour la même raison** — les jeux de clan ne l'ont vu que parce qu'on mesurait. Reprendre le seuil avec une validation par classe, pas au jugé.

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

**Agent jeux de clan** (`clan_games/`) — 🚀 **LIVRÉ** (développé du 22 au 27 août 2026, pendant la fenêtre des jeux). ⏳ **Reprise des jeux ~20 septembre 2026** : c'est là que se feront les 3 validations restantes.

> 💡 **L'agent ne JOUE pas les défis, il les CHOISIT.** La progression s'incrémente toute seule pendant que `CombatAgent` farme → pas d'exécuteur par type de défi à écrire. « Puis-je le réussir ? » n'est pas un raisonnement mais un **filtre de capacités** (pas de village des ouvriers, pas de compo d'armée choisie → ces défis sont éliminés d'office).
>
> ⚠️ **Pas droit à l'essai-erreur** : accepter un défi puis le `Rejeter` a une pénalité en jeu. Si le scorer n'est sûr de rien, il ne prend **rien** et retente au cooldown suivant (même principe que l'anti-gemmes : sans preuve, on n'agit pas). **`rejeter_defi` est détecté, jamais tapé.**
>
> 📌 `defi` et `evenement` (classes CNN UI existantes, inutilisées) **n'ont rien à voir** : `defi` = défi amical entre membres, `evenement` = événements du jeu (ligue de clan, durée limitée).

**🔎 Ce que les captures du 22 août 2026 ont établi** (→ `data/captures/jeux_clan/`) :

1. **La grille ne porte AUCUN titre** — chaque carte = icône + points (4×2, scrollable). Le nom et la description n'existent **que dans le pop-up de détail**. Lire les défis depuis la grille = reconnaître des **icônes**, pas du texte.
2. **Le pop-up se lit bien à l'OCR** : `Gagnez une étoile en combat multijoueur en utilisant au moins 1 Dragon.` à 0.59-0.99 (2 fautes cosmétiques). C'est la **ligne sémantique**. Le titre stylisé, lui, est massacré (`Ghaos draconien`) — sans importance, il n'ajoute rien.
3. **Taper une carte est SÛR** : ça ouvre le pop-up, `COMMENCER` est un bouton séparé → on inspecte les 8 défis sans jamais en engager un.
4. **Un défi actif grise TOUTES les autres cartes** ; l'active passe en cadre doré + horloge, son badge points devient une progression (`0/1`), et son pop-up affiche `Rejeter` (rouge) au lieu de `Commencer`. → « ai-je un défi en cours ? » se détecte **par la saturation**, avec le `_is_grayed` déjà écrit dans `donations.py`/`lab.py`. Zéro classe, zéro modèle.
5. Chaque défi porte aussi une **limite de temps** (ex. `3H`) → contrainte du scorer, pas seulement les points.

> 🎯 **Décision d'archi : le CNN UI v5 apprend le MOBILIER, pas les icônes de défi.** Une classe par icône (~40/saison, renouvelées à chaque saison) obligerait à re-labéliser tous les mois, et une icône inconnue = défi invisible. Le **sens** vient du pop-up par OCR + catalogue de motifs → marche sur les défis d'aujourd'hui **et** de la saison prochaine, sans re-train. Coût : ~15 s pour croiser les 8 cartes, quelques fois par jour. Acceptable.
>
> **10 classes à labéliser** : `jeux_clan` (l'entrée sur village_home ≈ (495,1020), le barbu roux — **la seule qui exige vraiment le CNN**) · `carte_defi` (générique, donne les 8 boîtes quel que soit le scroll) · `points_defi` · `commencer_defi` · `rejeter_defi` · `score_jeux_clan` (`0/10000`, plafond perso) · `temps_restant_jeux` · `onglet_defis` / `onglet_clan` / `onglet_recompenses`.
>
> 🏷️ **Deux classes labélisées dans DEUX états**, triées par saturation dans le code (jamais par une classe dédiée — précédent `donner`) :
> - `carte_defi` : colorée (disponible) **et** grisée (un autre défi est actif) **et** cadre doré (c'est celle en cours).
> - `points_defi` : bandeau vert `400` (points à gagner) **et** bandeau gris `0/1` (progression du défi en cours). Ce second état est ce qui dira à l'agent « c'est terminé, va en chercher un autre » — sans ce crop, la progression n'est pas lisible. Localisé par le CNN puis lu par le digit CNN, comme `compteur_or` / `prix_upgrade` (`widget_reader`) : pas de bandeau déduit géométriquement depuis `carte_defi`, donc pas de coordonnée en dur.

- [x] **Incr. 0 — Capture + dump OCR** (22 août 2026) : `tools/debug/clan_games_capture.py`, **lecture seule absolue** (ne tape jamais, n'accepte aucun défi). Mode guidé → `_raw.png` / `_ocr.json` / `_ocr.png` (+ `_cnn.png`) sur 7 écrans ; mode `--rafale N` → captures brutes pour le dataset de labeling. Les PNG bruts sont l'**actif durable** : les jeux durent 1 semaine, le reste se développe hors ligne après.
- [x] **Incr. 1 — CNN UI v5 livré (22 août 2026)** : **155 classes**, mAP50 0.978, F1 0.90. **15 nouvelles, 0 perdue.** Déployé en `weights/yolo_ui.pt`. Noms réels : `raccourci_jdc`, `carte_defi`, `point_defi`, `progression_defi`, `commencer_defi`, `rejeter`, `onglet_defi_jdc`, `recompense_jdc`, `classement_clan_jdc`, `score_personnel`, `score_clan`, `palier`, `temps_restant_jeux`, `temps_avant_expiration`, `jeux_de_clans`.
- [ ] 🔧 **Renfort dataset jeux de clan** — 3 trous mesurés sur les 7 captures réelles, par ordre d'importance :
  - [ ] **`raccourci_jdc` à 0.41-0.44** : l'entrée sur village_home, **sous le seuil d'action (0.60)**. C'est la classe la plus importante de toutes et la plus faible. Contournée par un `SEUIL_ENTREE = 0.35` local dans `clan_games/reader.py` — **pansement à retirer** une fois la classe renforcée. ~30-50 captures du village, zooms et positions variés.
  - [ ] **`rejeter` : 0 détection** alors que le bouton rouge est bien visible sur `defi_en_cours`. Sans conséquence (on ne le tape jamais, et `engage` a un second chemin), mais c'est un signal perdu.
  - [ ] **Faux `carte_defi` sur l'onglet Récompenses** : les tuiles de récompense passent pour des cartes de défi (13 détections à 0.50-0.81, ramenées à 4 par les filtres taille/seuil). Renforcer l'onglet Récompenses en négatif.
  - [ ] *(mineur)* `onglet_defi_jdc` instable à 0.46-0.66 — l'onglet **actif**, en surbrillance, est sous-représenté. Contourné : `menu_ouvert()` teste les deux onglets voisins (0.91-0.97) au lieu de celui-là.
  - [ ] **`jeux_de_clans` : 0 détection, même au seuil plancher 0.05.** C'est la **tente** qui apparaît sur la carte du village pendant les jeux ; la taper ouvre le menu, exactement comme `raccourci_jdc` (la petite icône à côté des événements). **Deux entrées indépendantes vers le même menu** — le reader doit essayer les deux, et une tente bien détectée réglerait le problème du raccourci à 0.41 sans re-train. Reste à savoir si elle était simplement **hors champ** sur les 2 captures village (village scrollé) ou si le modèle ne la reconnaît pas : à trancher en direct, c'est 30 s.
- [x] **Incr. 2 — `clan_games/reader.py` livré (22 août 2026)** : 8/8 cartes avec les bons points, score personnel `(0, 10000)`, défi actif + progression `(0, 1)`, cartes grisées correctes. **Une inférence par frame** (badges appariés par inclusion géométrique). 🐛 `read_widget_ratio` est mono-chiffre **par conception** → `lire_ratio()` aiguille digit CNN / OCR sur le **nombre de glyphes**. ♻️ `widget_reader.read_number_in/read_ratio_in` séparent localisation et lecture. Démo `tools/debug/clan_games_reader_demo.py` (hors ligne ou `--live`). → détail [CHANGELOG](CHANGELOG.md)
- [ ] **Tests unitaires du reader** : appariement badge/carte, tri par rangée, aiguillage `lire_ratio`, `engage` sans classe `rejeter`. Sur détections factices, sans modèle.
- [x] **Incr. 3 — Catalogue par CONTRAINTES (22 août 2026, v2)** : on ne catalogue **pas** les défis (des dizaines, renouvelés chaque saison) — on décrit le **terrain** (où ça se joue) et les **modificateurs** (ce que ça impose). L'objectif n'entre pas dans la faisabilité. Vérifié sur **5 familles jamais vues** : toutes comprises ; la v1 par gabarits les aurait toutes refusées. ~~Incr. 3 v1 (gabarits de phrase)~~ : `configs/clan_games.json` + `catalog.py`, **100 % pur**, **25 tests sur les 7 descriptions réelles**. 4 familles relevées en jeu. 🔑 « utilisant au moins 1 Golem » = il suffit que la troupe soit **déjà dans la barre** (capteur existant) → ces défis deviennent conditionnels, pas impossibles. 🛡️ Refus par défaut sur tout ce qui n'est pas certain. Vérifié bout en bout : **7/7 reconnus, 0 inconnu**. → détail [CHANGELOG](CHANGELOG.md)
- [ ] **Croisement : 1 défi sur 8 manquant** — le crawler en a lu 7. À diagnostiquer sur les captures de `data/captures/jeux_clan/defis/` (veto déclenché ? grille plus courte sur une frame ? pop-up resté ouvert ?).
- [x] **Incr. 4 — `clan_games/selector.py` livré (22 août 2026)** : ouvrir → croiser → choisir → engager → fermer. Liste de taps exhaustive en tête de module ; `rejeter` jamais tapé ; `confirmer=False` par défaut. → [CHANGELOG](CHANGELOG.md)
- [x] **Incr. 5 — `ClanGamesAgent` livré (22 août 2026)** : branché dans `brain/core.py`, prio 16, cooldown 30 min, `can_run` gratuit (lit `world['buttons']`), plafond mémorisé. CLI `--jeux-clan-confirmer`. **17 tests** sur les garde-fous.
**📋 État : l'agent est COMPLET et sûr, mais 3 chemins n'ont jamais été vus fonctionner en jeu.** Ce qui est validé en réel : ouverture, croisement (8/8 cartes, 8/8 descriptions), interprétation, refus motivé, remontée des besoins, et **un engagement réel** (« Détruisez Canon 10 fois »). Ce qui ne l'est pas ↓

- [ ] 🎯 **`statut: ok` jamais observé.** Le seul engagement réel a eu lieu **avant** le correctif de vérification et s'est soldé par `engagement_non_confirme`. La boucle de scrutation `_attendre_engagement()` n'a donc **jamais tourné pour de vrai** — seulement en test. À voir au premier défi sans contrainte.
- [ ] 🎯 **`deja_engage` jamais observé** : détecter un défi déjà en cours et ne rien faire. Se vérifie en relançant la démo juste après un engagement réussi.
- [ ] 🎯 **L'agent n'a jamais tourné DANS le bot.** Il a été éligible une fois, le LLM a choisi `combat`, et le run s'est arrêté. Le chemin `world → can_run → run → AgentResult` n'est validé qu'en tests. La 1ʳᵉ ligne du docstring a été réécrite depuis pour donner l'enjeu au modèle — effet non mesuré.
- [ ] 🎯 **VALIDATION EN JEU — détail.** Rien n'a encore été engagé en réel. **Passer par la démo directe**, pas par le bot : un run réel a montré que le LLM peut ne jamais choisir l'agent (3,5 min = une seule décision, partie à `combat`).
  1. `uv run python -m tools.debug.clan_games_demo --scan` → l'entrée est-elle vue, la grille lue ?
  2. `uv run python -m tools.debug.clan_games_demo` → flux complet, **sans engager**. Vérifier le défi désigné.
  3. [x] `clan_games_demo --confirmer` → **défi réellement engagé en jeu (22 août 2026)** : « Détruisez Canon 10 fois », 300 pts, choisi et lancé. Deux bugs corrigés au passage (🔧 seuil `commencer_defi`, puis vérification trop hâtive). **Reste à revoir un run complet de bout en bout avec le statut `ok`.**
  - [x] **Croisement validé sans doublon (22 août 2026)** : 7 défis distincts, 7 descriptions lues, tous les verdicts justes, 3 besoins remontés. La carte ratée par le CNN est désormais rattrapée par son badge.
  4. Enfin `uv run python -m clashai.brain --mode farm --jeux-clan-confirmer` pour valider l'intégration.
- [x] **Croisement instable — instrumenté et stabilisé (22 août 2026)** : cause identifiée, le **pop-up recouvre des cartes** (5/6/8 selon la frame). `_grille_stable()` ferme et attend le retour à la taille de référence. L'OCR est désormais journalisé défi par défi, et le refus donne une raison par défi.
- [x] **`aucun_defi_sur` diagnostiqué (22 août 2026)** : run réel avec les nouveaux logs → **7/7 descriptions lues**, refus tous légitimes (4 défis « avec unité » sur une barre vide, 3 au village des ouvriers). Le comportement était correct ; il manquait la **remontée du besoin**, désormais livrée (`defis_bloques`).
- [ ] 🔗 **Brancher `defis_bloques` sur le cerveau** : l'agent dit maintenant « golem → 150 pts » ; le `world`/prompt ne le transporte pas encore. À faire avec 5.3.2/5.3.3 (registre d'outils) — c'est exactement le genre de besoin qu'un outil `adapter_armee(troupe)` consommera.
- [ ] 🎖️ **Composition d'armée pilotée — LE déblocage à fort levier.** Le bot ne choisit pas encore ses troupes. Mesuré sur les 7 défis réels du 22 août : **0 faisable aujourd'hui**, **4 faisables** en basculant le seul drapeau `Capacites.composition_armee` (les 3 autres sont au village des ouvriers, définitivement hors de portée).
  - 📌 **Beaucoup moins cher que je ne le croyais** : depuis la refonte du jeu, **former des troupes est instantané et gratuit**. Pas de file d'attente à surveiller, pas de coût en élixir à arbitrer, pas d'ordre contraint avec la limite de 3H d'un défi. Il ne reste que le geste : ouvrir le menu d'armée, choisir, former.
  - Le catalogue est **déjà prêt** : `evaluer()` accepte le drapeau, `compo_armee_non_pilotee:<troupe>` nomme la troupe exacte, `defis_a_une_troupe_pres()` remonte les besoins triés. Rien à changer côté jeux de clan le jour où l'outil existe.
  - ⚠️ Seul coût résiduel identifié : imposer une troupe **modifie la composition sur laquelle l'agent de combat a été réglé**. C'est ce que représente `effort_ajoute: 1` — ni un délai, ni une dépense.

### V4.4 — Polish perception

- [ ] **Baseline RL en 70/57** : le checkpoint archivé `v4.4-ppo-350ep` est en 68/51 → **ne se recharge plus** (`PPOAgentV4.load()` repart de zéro **en silence**, vérifier les logs de démarrage). Refaire un run propre après les fixes deploy → nouvelle baseline (`docs/baselines.md`, `compare_baseline.py`).

### V5.1 — Résiduels multi-agents

- [ ] **ADB zéro screenshot (résiduel)** : faire lire le cache `PerceptionThread` aux consommateurs *live* (`gdc/navigator`, `social/chat`, `clan_castle`). En partie absorbé par le `world`.
- [ ] Stop le sanity-rescan dans `environment_v4._all_resources_exhausted()` (redondant avec `_sync_remaining_from_perception()`).
- [ ] **Flag perception `chat_unread`** (badge `!` près du bouton chat) → `ChatAgent.can_run` ne check qu'en présence du signal, au lieu d'ouvrir périodiquement.

### V5.3 — Cerveau LLM v1 (orchestrateur)

> `LocalLLMBrain(Brain)` remplace `HeuristicBrain` : décide QUEL agent lancer selon le `world`. Détail du fait → [CHANGELOG](CHANGELOG.md).

- [x] **CNN UI continu, cadence réduite (19 août 2026)** : le `PerceptionThread` détecte les boutons **1 cycle sur 5** (~4 Hz au lieu de 20 — les boutons bougent lentement, le combat a besoin des ms), résultat **conservé** entre deux passages. Exposé dans le `world` : `buttons` + `buttons_age_s`.
- [x] **`LocalLLMBrain` livré (19 août 2026)** : même contrat `decide(world)`, choisit **parmi les agents éligibles** (il ne contourne ni cooldowns ni `can_run`). **Repli heuristique systématique** : Ollama absent, timeout, JSON cassé, agent halluciné. **Actif par défaut** (`--no-llm` pour forcer l'heuristique). 19 tests, client Ollama injecté (aucun serveur requis).
- [x] **Stack Ollama opérationnelle (19 août 2026)** : client `ollama` + serveur (lancé automatiquement par l'installeur Windows — `ollama serve` dit « adresse déjà utilisée », c'est normal) + `mistral:latest`. Validé de bout en bout : préchauffage OK, décisions en 0,4-2,5 s, discussion qui décrit correctement l'état du jeu.
- [x] **`world` enrichi (19 août 2026)** : ressources, ouvriers et labo lus **dans le même cycle** que la détection UI (une seule inférence, cache de détections) et exposés sous `readings`. Les valeurs non lisibles sont annoncées « NON LUE » au modèle — corrige une hallucination observée en réel (« tu as 15568 or »).
- [x] **Console de discussion opérateur (19 août 2026)** : `tools/debug/llm_chat.py` — on parle au cerveau dans un terminal, il voit l'état réel du jeu et répond. Commandes `/etat`, `/oubli`, `/quit` ; `--sans-jeu` pour discuter sans charger la perception. *Sera intégrée au bot en 5.3.3.*

**Objectif de fin de V5.3** : le LLM pilote les agents, en autonomie **et** sur instruction admin en langage naturel (« donne-moi 3 ballons et 2 yéti »). Six incréments, un commit et une validation en jeu chacun.

| # | Incrément | Ce que ça livre |
|---|---|---|
| ~~5.3.0~~ ✅ | ~~Banc d'essai de prompts~~ | plusieurs formulations par intention + score attendu, en une commande |
| ~~5.3.1~~ ✅ | ~~Enrichir le `world`~~ | collecteurs pleins + dons en attente (les **coûts** passent en outil, 5.3.2) |
| ~~5.3.2~~ ✅ | ~~Registre d'outils~~ (lecture seule) | 3 garde-fous dans le code : autorité, dépense, arguments |
| **5.3.3** | **Console intégrée + outils qui agissent** | 4 étapes testées **une par une** en jeu : 3a plomberie · 3b branchement · 3c LLM · 3d bâtiments par nom |
| **5.3.4** | **Quantités dans le chat admin** | « 3 ballons et 2 yéti » → exactement ça |
| **5.3.5** | **Boucle autonome avec mémoire** | le LLM voit le RÉSULTAT de ses actions |

- [x] **5.3.0 — Banc d'essai de prompts (19 août 2026)** — `uv run python -m tools.eval.prompt_bench`. Référence **Mistral 7B : 52/56 (93 %)**, seuil 90 %. Deux défauts du modèle isolés et documentés (déterminant de la question recopié comme quantité ; formulation elliptique → nombre bouche-trou). *Placé en premier après le bug du 19 août : un défaut de prompt est invisible aux tests unitaires (le code est juste, c'est le modèle qui interprète mal) et **intermittent selon la formulation** — un essai manuel avait conclu « ça marche » la veille.* Toute la V5.3 fait interpréter du langage naturel à un 7B : c'est le mode d'échec dominant, pas les bugs de code. Le banc évalue **dans les deux sens** (rappel *et* retenue) et devient le critère de recette des incréments suivants. Il répond aussi par un chiffre, et non par une opinion, à « Mistral 7B tient-il le tool-calling ? ».
- [x] **5.3.1 — `world` enrichi : ce qui se COMPTE (19 août 2026)** — collecteurs pleins par ressource + demandes de dons en attente, tirés des détections DÉJÀ faites (zéro inférence en plus), comptés au seuil d'action. Banc : **98/102 (96 %)**. 🔎 **Périmètre corrigé** : les **coûts d'amélioration ne sont pas lisibles passivement** (il faut taper le bâtiment puis `ameliorer`) → déplacés en 5.3.2/5.3.3 comme **outil** `cout_amelioration(batiment)`. 🐛 Deux « défauts du modèle » se sont révélés être des défauts de prompt (ligne à deux nombres, liste à virgules).
- [x] **5.3.2 — Registre d'outils, lecture seule (19 août 2026)** — `Tool` + `ToolRegistry.call()`. 🛡️ **La sécurité est dans `call()`, pas dans le prompt** : autorité par source (`clan` ne voit ni n'appelle un outil qui agit), dépense = `confirm=True` obligatoire (**socle anti-gemmes, écrit une fois**), arguments validés contre le schéma. Journal de tous les appels. Outils livrés : `etat_du_village`, `lister_troupes_disponibles`. 🐛 « zéro » vs « je ne vois pas d'ici » désormais distingués dans le prompt. Banc : **104/108 (96 %)**.
  - [ ] Reste à ajouter en **5.3.3d** : `cout_amelioration(batiment)` (déplacé de 5.3.1 — un coût exige de taper le bâtiment).
- [ ] **5.3.3 — Console admin intégrée + outils qui agissent**, en **4 étapes, chacune testée en jeu avant la suivante** (pas toutes à la fin) :
  - [ ] **3a — Plomberie** (fichiers neufs) : file d'instructions (l'admin passe avant le clan) · outils qui agissent (récolte, attaque, renforts, dons, labo) · console `/commande` → outil (lecture = immédiat, action = en file, dépense = `o/n`) · noms de troupes validés **à la saisie** · démo sans émulateur.
  - [ ] **3b — Branchement au bot** : `--console` = **thread du bot**, pas un programme à côté (`llm_chat.py` démarre son propre `PerceptionThread` : deux pipelines sur 8 Go, deux `world` divergents, taps sans ordre). La boucle vide la file entre deux agents et pendant les pauses ; « je termine d'abord : combat ».
  - [ ] **3c — Le LLM appelle les outils** : tool-calling Ollama + cas au banc (phrase → bon outil).
  - [ ] **3d — Bâtiments par nom** : `ameliorer_batiment(nom)` + `cout_amelioration(nom)`.
  - 🛡️ **Deux verrous par dépense** : le `o/n` = l'**intention** de l'opérateur (`confirm=True` du registre) ; la **preuve d'affordabilité** reste au module. Aucun outil ne passe de `confirm_decider` (🔧 trou corrigé le 12 sept. 2026 : un décideur remplaçait la preuve).
- [ ] **5.3.4 — Quantités dans le chat admin** : « donne-moi 3 ballons et 2 yéti » → exactement ça. `donate_to_request(wanted=…)` existe déjà et refuse proprement (`no_match`) plutôt que de donner n'importe quoi ; il lui manque le **compte** (passer d'un ensemble à un quota `{ballon: 3, yeti: 2}`). Tout nom de troupe est **validé contre `troops.json`** : un nom inconnu = refus, jamais d'approximation. Pré-parseur déterministe **seulement si** le banc 5.3.0 montre que le 7B décroche — pas de béquille avant la preuve.
- [ ] **5.3.5 — Boucle autonome avec mémoire des résultats** : aujourd'hui `AgentResult` se perd, donc le LLM re-décide à l'identique après un échec. Mémoire courte des N dernières actions **et de leur issue**. Le plus risqué, donc en dernier.

**🛡️ Autorité : deux entrées, deux pouvoirs.** La file d'instructions porte **qui parle**.
- **`admin`** (toi, terminal) → peut déclencher les outils qui agissent, et **passe devant** le chat de clan si les deux attendent.
- **`clan`** (membres) → **conversation uniquement**. Un « @mini_pekka lance une attaque » écrit par un membre est du *texte*, pas une instruction.

⚠️ La sécurité tient à l'**architecture, pas au prompt** : aucun outil n'est branché sur la source `clan`, donc il ne *peut pas* agir — on ne compte pas sur le modèle pour reconnaître un ordre et le refuser (une consigne à un 7B s'applique de travers, cf. le bug du 19 août). Le prompt sert seulement à ce qu'il l'explique poliment **et ne mente pas** (jamais « ok j'attaque ! » sans rien faire). Pire cas d'une injection de prompt dans le chat clan : il dit une bêtise. Jamais : il joue.

- [ ] **Exposer les agents comme vrais tool-calls** Ollama (aujourd'hui : prompt + JSON, ce qui marche déjà et reste plus portable entre modèles). À trancher avec le banc 5.3.0.
- [ ] **Mode coach** : après chaque attaque, contexte → analyse NL → log ou chat clan.

### V5.0 — Mode live (phases optionnelles)

- [ ] **Phase 3** : decision tick event-driven (thread réagissant aux events `PerceptionEventBus`). Mode prod only (le RL reste sur steps discrets).
- [ ] **Phase 4** : mesurer la latence end-to-end (event → action). Cible ~150 ms.
- *Avant Ph.3* : définir les critères de « changement significatif », le comportement idle, et l'impact RL.

---

## 📅 À venir

### V5.4 — Pilotage chat + RAG complet

- [ ] `ChatAgent` (déjà là) → `LocalLLMBrain` (avec RAG) → répond / exécute / rapporte.
- [ ] **OCR par MESSAGE plutôt que par zone** — *prérequis d'un vrai dialogue*. Aujourd'hui l'OCR lit **un seul rectangle en dur** (`chat/constants.py` : 0-850 × 60-980) et rend un bloc de texte indifférencié : impossible de savoir qui a dit quoi. Il faut **une nouvelle classe CNN pour les bulles de message des membres** (à labelliser) — ⚠️ `message_clan` **n'est PAS** ça : c'est le bouton du menu château pour écrire au clan. Ensuite : détecter chaque bulle → OCR individuel → auteur + texte + ordre. Même pattern « CNN localise → lecteur lit » que les compteurs/prix/cartes labo.
  - **Débloque aussi les dons intelligents** : une demande écrite en toutes lettres (« il me faut des sapeurs et des ballons ») sans verrouillage laisse le jeu tout accepter → seul le texte dit quoi envoyer. `DonationManager.donate_to_request(wanted=…)` attend déjà cette liste.
- [ ] RAG : **Chroma** + `nomic-embed-text` (jargon/méca CoC + contexte clan + préférences).
- [ ] **Dons intelligents sur demande ÉCRITE** : quand le membre écrit « 3 sorcières + 2 ballons + 1 soin + 1 zap » **sans que le jeu verrouille** ces troupes, seul l'OCR du message dit quoi envoyer — et **en quelle quantité**. Le LLM doit fournir exactement ça, ni plus ni moins. `DonationManager.donate_to_request(wanted=…)` attend déjà la liste ; reste à lire le texte (dépend de l'OCR-par-message ci-dessus) et à **respecter les quantités** (aujourd'hui on répartit à l'aveugle jusqu'à ce que le jeu refuse).
- [ ] **Relayer les demandes des membres à l'admin** : un « @mini_pekka lance une attaque » écrit dans le chat de clan ne se perd pas — il remonte dans la console admin (« *un membre demande une attaque* ») et **tu** tranches. Le membre est entendu, l'autorité reste d'un seul côté. Dépend de l'OCR-par-message ci-dessus.
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
