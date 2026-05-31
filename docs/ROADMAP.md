# ClashAI — Roadmap & Backlog

## 🎯 Vision & Objectif Global

- **OBJECTIF FINAL** : UNE IA AUTONOME INTELLIGENTE QUI JOUE COMME UN HUMAIN. IA autonome complète — joue, gère, recrute, s'améliore seule.

> Dernière mise à jour : 26 avril 2026 (Session 11 — multi-agents V5, vitesse deploy, navigation failure fix)  
> Ce document centralise **toutes** les modifications, features et idées prévues.  
> On pioche ici pour construire chaque version. Rien ne se perd.

---

## 🚀 En Cours (Current & Next Release)

### V4.3 — Perception + Vitesse

> **Objectif** : remplacer le template matching + OCR par des CNN plus fiables, améliorer la zone de déploiement, et réduire les délais pour une exécution plus fluide.

#### Fait (V4.3)

- [x] YOLO walls segmentation → deploy zone précise (`weights/yolo_walls_seg/walls_detection.pt`)
- [x] `get_perimeter_from_walls()` — masques seg + bboxes bâtiments → positions de déploiement
- [x] Capture directe fenêtre émulateur via `mss` (~20ms vs 150ms ADB) — `clashai/perception/screen_capture.py`
- [x] `adb_screenshot()` → mss en priorité, ADB en fallback
- [x] Thread perception asynchrone — `clashai/perception/perception_thread.py` (capture 20fps + YOLO en fond)
- [x] `_update_combat_observation()` lit depuis le cache (non-bloquant)
- [x] `DELAY_OBSERVE` 2.5s → 0.15s
- [x] Délais deploy réduits (~65% plus rapide) : DELAY_SWITCH_TROOP, DELAY_DEPLOY, ADB_DELAY_TAP
- [x] YOLO barre de troupes — 78 classes, `tools/train/train_yolo_troop_bar.py` (Kaggle-ready)
- [x] `TroopBarDetector` + filtre HSV grisé — `clashai/perception/troop_bar_detector.py`
- [x] `TroopFinder.update()` → YOLO en priorité, template matching en fallback
- [x] Chargé dans `load_models()` + `perception_thread` → détection fréquente sans bloquer

#### Reste à faire (V4.3)

- [x] **Bug séquence de récupération** : séquence **supprimée** purement et simplement (pas remplacée par un sleep). Testé : l'agent ne panique plus quand il rencontre un état imprévu, il continue son flow normal
- [x] **Debug overlay basique** : flag `--debug-overlay` dans `tools/train/train_rl_v4.py`, `_save_debug_overlay()` dans `clashai/combat/environment_v4.py` save `logs/episode_N/tXs.jpg` à chaque `observe`. Vu en run Session 12 : `Episode capture: logs\episode_0001\t0s.jpg`. La version Phase D détaillée (village_home.png / prep_attaque.png / debut_attaque.png / attaque_30s.png avec bboxes annotées colorées) est l'objet du mode `--test` (item suivant, à faire)
- [x] **OCR compteurs** : EasyOCR dans `TroopBarDetector._read_count()` — crop top-RIGHT (combat) ou top-LEFT (prep) selon l'écran, upscale ×3 + threshold → lecture du chiffre. `to_counts()` → `{name: count}`. `TroopManager.rescan()` utilise YOLO+OCR en priorité, legacy en fallback.
- [x] **Fix capture fenêtre émulateur occluded** (Session 12) — voir bloc dédié ci-dessous
- [x] **Mode `--test`** (Session 12) : `uv run python tools/train/train_rl_v4.py --test` lance 1 épisode heuristique et sauvegarde 5 captures annotées dans `logs/test_run/` : `village_home.png` (état CNN), `prep_attaque.png` (+ barre troupes YOLO + compteurs OCR), `debut_attaque.png` (+ bboxes bâtiments colorées par catégorie + hull deploy + points numérotés), `attaque_30s.png` / `attaque_60s.png` (+ bâtiments détruits surlignés en croix rouge + troupes YOLO combat). Module : `clashai/perception/test_run_capture.py::TestRunCapture`. Hooks dans `environment_v4.py` (`_get_screen_state` override + `_update_combat_observation`). Rapport `[OK]/[--]` à la fin pour voir quelles captures ont été atteintes
- [ ] Tester + valider en conditions réelles, ajuster conf YOLO si faux positifs
- [x] **Fix demande de troupes château de clan** (Session 13, **5 bugs**) :
  - `verbose=False` sur le CC manager → toutes les failures silencieuses (CC pas trouvé, cooldown, CC FULL — aucun log visible). Fix : `verbose=True`.
  - Pas de check `screen == village_home` avant l'appel → YOLO ne trouvait pas le château quand on était sur `resultats_attaque` ou `recherche_adversaire`. Fix : guard `classify_screen() == 'village_home'` avant `request_if_needed`.
  - `try/except Exception: pass` → toute exception silencieusement avalée. Fix : log explicite avec nom de la classe + message.
  - **Mismatch nom de classe YOLO** : `ClanCastleManager._find_clan_castle()` cherchait `b['class'] == 'clan_castle'` (anglais), mais le modèle YOLO bâtiments utilise `'chateau_clan'` (français, cf `weights/classes.json`). → Match jamais → CC jamais trouvé. **C'était la vraie raison pour laquelle l'agent ignorait totalement le CC.** Fix : `CC_CLASSES = ('chateau_clan', 'clan_castle')` (tuple, accepte les deux pour robustesse aux retrains futurs).
  - **`_close_menu` tape l'icône chat** : la fonction de fermeture tapait à `(30, 540)` ce qui correspond EXACTEMENT au bouton chat de clan (icône orange à gauche). Quand le template "Demande" n'était pas trouvé, `_close_menu` ouvrait le chat au lieu de fermer. Tentative 1 : `KEYCODE_BACK` (Android back) — **REJETÉ** car sur l'émulateur Google Play Games, BACK sur l'écran `village_home` déclenche le dialog "quitter le jeu" (pas le même comportement qu'un téléphone). Tentative 2 (en place) : tap à `(5, 5)` — coin haut-gauche garanti hors-UI, ferme tout menu ouvert sans risque de tap parasite.
  - Setup (`templates/clan_castle/request.png` + `cdc_confirmation` dans `configs/ui_positions.json`) vérifié OK Session 13.
  - `brain.py` faisait déjà bien (`_ensure_at_village()` avant l'appel) — mais souffrait aussi du bug class name → CC ignoré là aussi.
- [x] **Hard cap héros uniques à 1** (Session 12) : `UNIQUE_HEROES = {roi, reine, grand_gardien, championne, prince_gargouille, duc_draconique}` dans `troop_bar_detector.py`. `to_counts()` et `_read_count()` forcent count=1 pour ces classes peu importe ce que l'OCR lit (les badges héros affichent un nombre que l'OCR confond souvent avec 11/23). Évite que `_remaining_troops['reine'] = 23` après une mauvaise lecture
- [x] **Suppression rescan périodique** (Session 12) : avec YOLO troop bar tournant à chaque frame dans `PerceptionThread`, plus besoin du rescan tous les 10 steps qui prenait un screenshot dédié + relançait YOLO. Nouveau : `_sync_remaining_from_perception()` lit `state['troop_bar']` du cache (gratuit, déjà calculé) et met à jour `_remaining_troops` à chaque `_update_combat_observation()`. Plus réactif (chaque observe vs tous les 10 steps) et plus rapide (0 screenshot supplémentaire). Le rescan one-shot dans `_all_resources_exhausted()` est conservé comme sanity avant déclaration de fin d'épisode

#### 🔧 Fix capture fenêtre émulateur occluded (Session 12)

> Si le problème revient (capture montre l'écran du PC au lieu du jeu, CNN écran déraille, agent voit le bureau, etc.) → relire ce bloc.

**Symptômes**
- `_debug_capture.png` contient VS Code / le terminal / le bureau au lieu du contenu de l'émulateur
- Le CNN classificateur d'écran prédit toujours `chargement` ou des écrans aléatoires
- L'agent RL ne reconnaît pas l'état du jeu et fait n'importe quoi
- `ScreenCapture` log un backend `dxcam` ou `mss`

**Cause racine**
- Google Play Games rend en **DirectX/Vulkan dans une surface GPU accélérée**, pas dans la couche GDI lisible par PrintWindow
- `dxcam` et `mss` lisent les pixels de **l'écran physique** → si VS Code est devant, ils capturent VS Code
- `PrintWindow` + `PW_RENDERFULLCONTENT` retourne le contenu de l'écran (pas du buffer de la fenêtre) pour ces émulateurs hardware-accélérés

**Solution (en place)**
- Backend **WGC (Windows.Graphics.Capture)** via le package `windows-capture` (Rust wrapper, déjà dans `pyproject.toml`)
- C'est l'API que OBS / Snipping Tool utilisent, conçue exactement pour les apps DirectX
- Ordre des backends dans `clashai/perception/screen_capture.py::ScreenCapture._init_backend()` :
  **`wgc` → `printwindow` → `dxcam` → `mss` → `adb`**
- WGC tourne en background thread (`start_free_threaded`) et met à jour `self._wgc_latest` (BGRA numpy) à chaque frame ; `_grab_wgc()` lit ce buffer sans latence
- **Routing de tout le code via WGC** : `game_loop.adb_screenshot()` essaie d'abord `get_capture().grab()` (WGC) puis tombe sur ADB si KO. Comme `environment.py`, `environment_v4.py`, `brain.py` et `PerceptionThread` passent tous par `game_loop.adb_screenshot()` (ou directement `get_capture()`), training + brain + perception sont tous sur WGC sans changement supplémentaire
- **Normalisation 1920x1080** (`ScreenCapture._normalize_to_canonical()`) : WGC/PrintWindow/dxcam/mss capturent toute la fenêtre (titlebar Windows + bordures) à la résolution OS (DPI-scalée, ex 2560x1528). Le CNN d'écran + YOLO bâtiments + positions UI sont tous calibrés sur la sortie ADB native 1920x1080. La normalisation utilise `GetClientRect` + `ClientToScreen` pour calculer l'offset du contenu jeu dans la fenêtre capturée, applique le facteur DPI, crop le titlebar/bordures, puis resize en 1920x1080. **Sans ce step, le CNN d'écran prédit n'importe quoi en training (la barre noire en haut le déstabilise complètement)**

**Pièges déjà rencontrés**
1. **VS Code match** : le keyword `"Google Play"` matchait le titre de VS Code `"Fix Google Play emulator - COCProj - Visual Studio Code"` → liste `EXCLUDED_TITLE_SUBSTRINGS` (Visual Studio Code, navigateurs, JetBrains, Discord, Slack, etc.)
2. **adbproxy.exe match** : titre style chemin `C:\Program Files\...\adbproxy.exe` matchait aussi → filtre `\\` et `.exe` dans `find_emulator_bbox` + `_find_hwnd`
3. **Fenêtre minimisée** : `_find_hwnd` rejetait à cause du filtre taille (158x26 sous le seuil 400x300) → **la fenêtre de l'émulateur ne doit pas être minimisée** (peut être derrière, c'est OK)
4. **Cleanup thread Python** : "Fatal Python error" au shutdown — bénin (WGC thread cleanup), pas bloquant
5. **`adb_screenshot()` reverté en pur ADB** : pendant le cycle de debug, le wrapper qui routait `adb_screenshot()` via `get_capture()` avait été supprimé → training tournait sur pur ADB pendant que les tests tournaient sur WGC, donc *test OK / training KO*. Restauré : essaie WGC d'abord, fallback ADB
6. **Mismatch résolution** : sans `_normalize_to_canonical()`, WGC renvoyait 1283x751 (ou 2560x1528 en DPI 200%) avec titlebar Windows en haut. Le CNN d'écran (entraîné sur ADB 1920x1080 sans chrome) délirait → croyait être en `chargement` ou `recherche_adversaire` partout

**Commandes de test**
```bash
# Smoke test rapide — vérifie le backend sélectionné et capture 1 frame
uv run python -c "from clashai.perception.screen_capture import ScreenCapture; c = ScreenCapture(); print('backend=', c.backend); img = c.grab(); img.save('_wgc_smoketest.png') if img else print('FAIL')"

# Test interactif 3 scénarios (visible / derrière / au choix)
uv run python tools/debug/test_screen_capture.py

# Diagnostic complet : énumère parent + tous les enfants, PrintWindow sur chacun
uv run python tools/debug/inspect_emulator_window.py
```

**Si WGC casse encore**
1. Vérifier que la fenêtre émulateur n'est pas minimisée (`uv run python -c "import ctypes; print(ctypes.windll.user32.IsIconic(<hwnd>))"`)
2. Vérifier que `windows-capture` est bien installé (`uv pip list | grep windows-capture`)
3. Lire le log d'init : si `WGC backend (XXX)` montre un titre suspect (VS Code, navigateur), ajouter une exclusion dans `EXCLUDED_TITLE_SUBSTRINGS`
4. Lancer `tools/debug/inspect_emulator_window.py` → si TOUTES les fenêtres enfants capturent l'écran, l'émulateur a changé sa pile de rendu → vérifier la doc `windows-capture` pour un mode alternatif

#### Mode `--test` (diagnostic visuel — à implémenter)

```
uv run python tools/train/train_rl_v4.py --test
```

Lance exactement **1 épisode** (pas d'entraînement PPO) avec captures annotées sauvegardées dans `logs/test_run/` :

| Fichier | Moment | Contenu |
|---|---|---|
| `village_home.png` | Détection écran village | Screenshot brut + état CNN écran |
| `prep_attaque.png` | Écran de sélection d'armée | Screenshot + bboxes YOLO barre de troupes (avec compteurs x2/x11) + état `screen='prep'` |
| `debut_attaque.png` | t=0 de l'attaque | Screenshot + bboxes YOLO bâtiments colorées par catégorie + zone deploy (hull + points numérotés) + barre de troupes YOLO |
| `attaque_30s.png` | t=30s pendant l'attaque | Même chose + bâtiments détruits surlignés + troupes YOLO détectées |
| `attaque_60s.png` | t=60s pendant l'attaque | Même chose |

Objectif : valider visuellement que **tous les CNN voient correctement** avant de lancer un vrai entraînement.
- Buildings YOLO → bonnes classes, bonnes bboxes ?
- Deploy zone → positions bien en dehors des murs ?
- Troop bar → bonnes classes, bons compteurs ?
- Combat features → troupes bien trackées par YOLO troops ?

### V4.4 — Polish perception (avant la refonte V5)

> **Objectif** : clore proprement le cluster V4 avec les derniers irritants de perception, puis lancer un gros run de validation avant d'attaquer la refonte architecturale V5.

- [ ] **Mini CNN classificateur de chiffres** (remplace EasyOCR pour compteurs troop bar) — découpé en 4 phases :
  - [x] **Phase 1** (Session 13) : tool de collecte `tools/data/collect_digit_crops.py` — walk `logs/episode_*/` + `logs/test_run/`, run YOLO troop bar sur chaque jpg, crop le badge compteur de chaque détection countable (skip héros + abilities + siege déployés), save dans `needLabelisation/digits/<class>_<frameid>_<bbox>_<position>.png`. **Mode `--position auto` (défaut)** classifie l'écran source (CNN screen) et crop UNIQUEMENT la position pertinente : `prep_attaque` → badge top-LEFT, `phase_attaque` → badge top-RIGHT. Mode `--position both` disponible pour max recall (mais 50% des crops sont vides). Idempotent. Commande : `uv run python tools/data/collect_digit_crops.py --limit 200`
  - [ ] **Phase 2** : labelisation (manuelle ou semi-auto avec EasyOCR comme 1ère estimation). Cible : 500-1000 crops annotés, split 80/20 train/val.
  - [ ] **Phase 3** : entraîner un mini CNN (LeNet ou MobileNetV3-Small, ~50-200k params, 100 classes 0-99 ou regression). Notebook ou `tools/train/train_digit_cnn.py`.
  - [ ] **Phase 4** : intégrer dans `TroopBarDetector._read_count()` avec fallback EasyOCR si confiance basse.
  - **Pourquoi** : EasyOCR est générique et peu fiable sur les petits badges blanc/noir des icônes. Le `snapshot OCR + manual decrement` ne marche pas non plus car parfois l'agent tape hors zone de déploiement → la troupe n'est PAS déployée mais le compteur manuel décrémente → drift inverse.
- [x] **Fix `Fatal Python error: PyInterpreterState_Delete: remaining threads`** au Ctrl+C (Session 12)
  - Cause : `windows_capture.start_free_threaded()` lance un thread Rust qui ne se termine pas quand Python finalise
  - Fix : `start_free_threaded()` retourne un `CaptureControl` ; on stocke `self._wgc_control` et on enregistre `atexit.register(self._stop_wgc)` qui appelle `ctrl.stop()` + `ctrl.wait()` avec try/except (par défense, certains modules peuvent déjà être partiellement déchargés à ce stade)
- [x] **Aligner `imgsz` sur tous les YOLO** (Session 12, corrigé Session 13)
  - Constatation : `model.predict()` sans `imgsz=` → Ultralytics utilise 640 par défaut, peu importe l'imgsz d'entraînement → perte de détail silencieuse
  - Constantes ajoutées par modèle dans leur module respectif :

    | Modèle | Constante | Valeur | Module | Note |
    |---|---|---|---|---|
    | troop bar | `YOLO_IMGSZ` | **1088** | `troop_bar_detector.py` | Session 13 : retrain dédié à imgsz=1088 → détection mieux qu'à 640 ou 1600. Voir historique ci-dessous. |
    | bâtiments | `YOLO_BUILDINGS_IMGSZ` | 1600 | `game_loop.py` | OK à 1600 |
    | troupes combat | `YOLO_TROOPS_IMGSZ` | 640 | `troop_detector.py` | Default Ultralytics |
    | walls seg | `YOLO_WALLS_IMGSZ` | 640 | `deploy_zone.py` | Default Ultralytics |

  **📜 Historique imgsz troop bar**
  - 1ère tentative : ROADMAP avait noté `1600` (cf `tools/train/train_yolo_troop_bar.py::IMG_SIZE`) → en prod le YOLO ne trouvait que 0-1 icônes / 9 (conf~0.39). Cause probable : double-resize WGC 2451x1411 → LANCZOS 1920x1080 → YOLO letterbox 1600x1600 trop blur.
  - 2ème tentative (Session 13) : revert à `640` (default Ultralytics) → 9/9 détections (conf 0.35-0.96). Empiriquement OK mais qualité moyenne.
  - 3ème tentative (Session 13, fin) : retrain dédié à `imgsz=1088` → mieux que 640 mais pas encore parfait. Le modèle mérite plus de data / plus d'epochs (item séparé hors refactor).
- [x] **Fix demande de troupes château de clan** (Session 13, fait — voir détail dans V4.3)
- [ ] **Bug `grand_gardien` tap mode toggle** (à fixer côté data, pas code) : le bouton vert mode air/sol sur l'icône du GG n'a pas de classe YOLO dédiée → bbox `grand_gardien` inclut ce bouton → le tap au centre du bbox peut tomber sur le toggle au lieu de l'icône à déployer. **Fix recommandé** : retrain YOLO troop bar avec une nouvelle classe `grand_gardien_mode` (ou `mode_toggle_generic` partagé entre héros). Log diagnostic ajouté dans `troop_finder.select()` (warning si y hors range 950-1080).
- [x] **Conf YOLO troop bar 0.45 → 0.40** (Session 13) : 0.45 était pile sur le fil (golem @ 0.41 droppé). Validé empiriquement sur vraies frames : 0.40 = rien loupé, 0.50 = ça loupe. `YOLO_CONF` dans `troop_bar_detector.py` + défaut du tool `detect_troop_bar.py`.

#### 🔧 Bug RGB/BGR inversé sur l'input YOLO (Session 13) — LE vrai bug

> Si une détection YOLO se trompe systématiquement sur des classes dépendantes de la couleur (gel↔poison, soin↔clone, troupes mal classées) alors que le tool manuel `detect_troop_bar.py` sur la MÊME image est parfait → relire ce bloc.

**Symptôme**
- Détection troop bar fausse de façon **systématique** sur les classes couleur : `gel` (bleu) ↔ `poison` (violet), `soin` (jaune) ↔ `clone` (violet), championne manquée
- Le tool manuel `tools/debug/detect_troop_bar.py` sur exactement la même image → 100% correct
- Même modèle + conf + imgsz + image → résultats différents

**Cause racine**
- **Ultralytics lit un `np.ndarray` comme du BGR** (convention cv2), mais un `PIL.Image` comme du RGB.
- La prod faisait `model.predict(np.array(screenshot_pil))` → envoie des octets **RGB** que YOLO interprète comme **BGR** → **canaux Rouge/Bleu inversés** → classes couleur confondues.
- Le tool manuel faisait `model.predict(img_pil)` (PIL) → canaux corrects → toujours bon.
- La SEULE différence de code entre les deux chemins = `np.array(pil)` vs `pil`. C'était ça.

**Solution (en place)**
- Passer le `PIL.Image` directement à `.predict()` (ultralytics gère le RGB) — JAMAIS `np.array(pil)` brut.
- Corrigé dans : `troop_bar_detector.detect` + `analyze_village` (YOLO bâtiments).
- Déjà corrects : `troop_detector.detect` (passe PIL), `deploy_zone` walls (`cv2.cvtColor(..., RGB2BGR)` explicite avant predict).
- **Règle** : pour passer un numpy à ultralytics, TOUJOURS le convertir en BGR (`cv2.cvtColor(arr, COLOR_RGB2BGR)`). Sinon passer le PIL directement.

**Bonus défensif — verrou d'inférence**
- `clashai/perception/inference_lock.py` : `INFERENCE_LOCK = threading.RLock()` global, acquis autour de chaque appel modèle (`classify_screen`, `analyze_village`, building CNN, `troop_bar_detector.detect`, `troop_detector.detect`).
- Raison : les modèles ultralytics/torch ne sont pas thread-safe. Le `PerceptionThread` (fond) et `test_run_capture` (`--test`, thread principal) appellent les mêmes objets → on sérialise par sécurité. (Ce n'était pas LE bug couleur, mais c'est une bonne pratique pour le multi-agents V5.)
- [ ] **Migration capacités héros : template matching → CNN** (À FAIRE AVANT LE GROS RUN). Actuellement `clashai/combat/hero_ability.py` détecte les abilities dispo via `cv2.matchTemplate` (5 templates `ability_*.png`). Le YOLO troop bar connaît déjà les classes `*_capa` (`roi_capa`, `reine_capa`, `grand_gardien_capa`, `championne_capa`, `prince_gargouille_capa`, `duc_draconique_capa`). Migrer `hero_ability.scan()` pour lire ces détections depuis le `PerceptionThread` au lieu du template matching → plus robuste, plus rapide, un détecteur de moins à maintenir. Supprimer ensuite le chargement des templates `ability_*.png` (le log `5 ability templates loaded`).
- [ ] **Gros run V4 final** : 300-500 épisodes une fois tous les fixes ci-dessus en place. Baseline solide avant V5.

### V5.0 — Mode "en direct" (réactif temps réel)

> **Note** : numéroté V5.0 car livré AVANT V5.1/V5.2 dans l'ordre chronologique (Session 13). Le push pipeline était la priorité #1 pour la suite multi-agents.
>
> **Objectif** : passer du modèle actuel `screenshot → décision → tap → sleep` à un vrai pipeline vidéo continu où l'agent voit le jeu comme une vidéo, pas comme des photos.

#### Différence à clarifier

> **Aujourd'hui** : `PerceptionThread._capture_loop` appelle `cap.grab()` 20 fois par seconde. C'est 20 screenshots/s, **pas** un stream vidéo. Le screenshot est instantané mais ça reste une succession de captures discrètes commandées par le code.
>
> **V5.0** : la fenêtre émulateur est vue comme un **flux vidéo continu** (WGC fournit déjà ça via `on_frame_arrived` — chaque frame de l'émulateur déclenche un callback). On bascule en mode **push** : chaque frame WGC qui arrive déclenche directement le pipeline d'inférence. Plus de `grab() + sleep`, juste un consommateur qui traite ce qui arrive.

#### Tâches

- [x] **Phase 1** (Session 13) : `ScreenCapture.subscribe_to_frames(callback)` — API push universelle. Sur WGC, callbacks fire nativement sur le thread Rust quand `on_frame_arrived` se déclenche (= rythme natif de l'émulateur, 30-60fps). Sur autres backends (PrintWindow/dxcam/mss/ADB), un thread fallback polling à 30fps émule l'API push. `unsubscribe_from_frames()` + `num_frame_subscribers()` aussi exposés. `_fire_frame_callbacks_from_bgra()` fait la conversion BGRA→PIL+normalize une seule fois pour tous les subscribers.
- [x] **Phase 2** (Session 13) : `PerceptionThread._capture_loop` ne polle plus — il s'abonne via `cap.subscribe_to_frames(self._on_new_frame)` et bloque sur un wait. Les frames arrivent directement du thread WGC (ou du fallback poller). `_on_new_frame` est rapide : juste push dans la queue d'inférence avec dédup (max 1 frame en attente, on jette les vieilles).
- [ ] **Phase 3 (optionnel)** : decision tick agent event-driven (separate thread qui réagit aux events PerceptionEventBus au lieu de step discrets). Pour le mode prod uniquement, RL training reste sur steps discrets.
- [ ] **Phase 4 (optionnel)** : mesurer latency end-to-end (event → action). Cible : ~150ms vs ~500ms avant V5.0.

#### Avant de coder V5.0 Phase 3 (decision tick event-driven)

À définir clairement avec l'utilisateur :
- Critères de "changement significatif" déclenchant une décision
- Comportement si l'agent ne veut rien faire pendant longtemps (action `idle`/`observe` vs no-op)
- Impact sur le RL training (ou alors on garde discrete steps en training et continuous en inférence — à valider)

---

## 📅 À Venir (Up Next)

### V5.1 — Foundation multi-agents (ADB zero screenshot + BaseAgent)

> **Objectif** : préparer le terrain pour les nouveaux sous-agents. Aucun nouvel agent ici, juste de la plomberie.

#### ADB zéro screenshot — perception 100% directe

> V4.3 a branché le PerceptionThread sur la perception combat. V5.1 étend ça à TOUT le code — plus aucun `adb exec-out screencap` nulle part. ADB = uniquement pour les inputs (taps, swipes).

- [ ] Migrer `clashai/navigation/gdc_navigator.py` → lit depuis `PerceptionThread.get_latest()['screen_state']`
- [ ] Migrer `clashai/social/clan_castle.py` + `clan_chat_monitor.py` → même thread
- [ ] Migrer `clashai/combat/hero_ability.py` → scan abilities depuis le frame du thread
- [ ] Supprimer **tous** les appels directs `adb exec-out screencap` restants dans `clashai/` (grep doit retourner 0)
- [ ] `PerceptionThread` devient le **seul point d'entrée** pour la vision de l'agent
- [ ] Stop le sanity-rescan dans `environment_v4._all_resources_exhausted()` (redondant avec `_sync_remaining_from_perception()` qui tourne à chaque observe)

#### Interface commune `BaseAgent`

- [x] Définir classe abstraite `BaseAgent` : `can_run(world)`, `run()`, `priority`, `cooldown_seconds` (Session 13, Phase C.4 — `clashai/agents/base.py`)
- [x] `AgentScheduler` : registry + `pick(world)` (prio + cooldown + can_run) + `tick(world)` + history (Session 13, `clashai/agents/scheduler.py`)
- [ ] Chaque agent existant (combat, GdC, château, chat) implémente l'interface — fait en V5.2
- [ ] L'orchestrateur `brain.py` utilise `AgentScheduler` au lieu de la logique ad-hoc actuelle — fait en V5.2

#### État actuel (à formaliser)

| Sous-agent | Fichier actuel | Type | V5.1 status |
|---|---|---|---|
| Orchestrateur | `brain.py` | Heuristique | À refondre V5.2 |
| Attaque (farm) | `combat/environment_v4.py` + PPO | RL | Déjà OK, juste wrapper BaseAgent |
| Guerre de clan | `navigation/gdc_navigator.py` | Heuristique | Wrapper + migration PerceptionThread |
| Chat clan | `social/clan_chat_monitor.py` | Règles | Wrapper + migration |
| Château de clan | `social/clan_castle.py` | Règles | Wrapper + migration + fix V4.4 |

### V5.2 — Nouveaux sous-agents + orchestrateur intelligent

> **Objectif** : ajouter les agents qui manquent et un brain orchestrateur capable de décider QUOI faire et QUAND.

#### Nouveaux agents

**Agent jeux de clan** (`clashai/clan_games/`)
- Détecter si jeux de clan actifs (CNN écran ou template menu)
- Identifier tâches disponibles (template matching sur cartes)
- Exécuter tâches répétitives (attaque, don de troupes, etc.)
- Type : règles + heuristiques (pas besoin de RL)

**Agent gestion village** (`clashai/village/`)
- Détecter constructeurs libres (template matching ou YOLO dédié)
- Queue de priorité d'amélioration (murs → défenses → ressources)
- Détecter labo libre → lancer recherche
- Collecter ressources (mines, coffres)
- Type : règles + queue de priorité

#### 🔧 CNN barre d'options bâtiment (perception robuste)

> Idée Session 13 : quand on tape n'importe quel bâtiment du village (CC, mine, défense, labo, etc.), une **barre d'options** apparaît en bas de l'écran avec ~6-8 boutons (`Demander`, `Renforcer`, `Améliorer`, `Trésorerie`, `Dormir`, `Infos`, etc.). Le contenu et l'ordre des boutons varient selon le type de bâtiment. Le template matching actuel sur le bouton `Demande` du CC est fragile (~50% de réussite).

- [ ] **CNN options bar** dédié : input = crop de la barre du bas (y~860-1080), output = liste `{name, x, y, conf}` des boutons détectés
- [ ] Classes : `demander`, `renforcer`, `ameliorer`, `tresorerie`, `dormir`, `infos`, `rechercher`, `collecter`, `acheter`, ... (à compléter avec inventaire complet en parcourant chaque type de bâtiment)
- [ ] Pipeline : tap bâtiment → screenshot → CNN options bar → décider quel bouton presser selon l'intention de l'agent
- [ ] **Unlock** : remplace le template matching CC fragile + débloque l'agent gestion village (gérer constructeurs, labo, ressources sans hardcoded coords)
- [ ] **Data** : collecter ~200-500 crops de barres d'options annotés (script de capture interactive : "tape ce bâtiment, je save la barre", puis labellisation manuelle)
- [ ] **Model** : YOLO petit (nano) sur la zone de la barre, similaire à `troop_bar_detector` mais sur une zone légèrement plus haute

#### Orchestrateur `brain.py`

- [ ] Boucle principale : check chaque agent selon `priority()` + `can_run()`
- [ ] Gestion cooldowns : ne pas relancer agent qui vient de tourner
- [ ] Logging centralisé : chaque agent log ses actions dans un fichier commun
- [ ] Schedule par N minutes selon le type d'agent (combat = quand armée prête, village = toutes les 30 min, etc.)

### V5.3 — Dashboard web temps réel

> **Objectif** : suivre tout ce qui se passe (multi-agents, training, vision agent) depuis une page web sur le réseau local.

#### Avant le code : maquette + spec pages

- [ ] **Maquette ASCII / Figma** des pages avant la moindre ligne de code
- [ ] Définir précisément le contenu de chaque page : composants, sources de données, fréquence de rafraîchissement
- [ ] Lister les endpoints API (REST + WebSocket) nécessaires

#### Pages prévues (à valider via maquette)

- [ ] **Page principale** — État de chaque sous-agent (actif / idle / cooldown), dernière action, dernière attaque
- [ ] **Onglet Training** — Courbe reward/étoiles en temps réel, dernière image de debug overlay, PPO stats (value_loss, entropy, policy_loss)
- [ ] **Onglet Replay** — Les N dernières images de debug overlay par épisode (timeline visuelle d'une attaque)
- [ ] **Onglet Village** — État constructeurs, labo, ressources (lecture depuis logs agents)
- [ ] **Onglet Vision Agent** — **Flux vidéo en temps réel** de ce que l'agent perçoit, avec overlays annotés (bboxes YOLO bâtiments colorées par classe, masques walls seg, hull deploy zone + points numérotés, positions troupes YOLO, cluster principal, sorts restants en overlay texte). Alimenté par le flux V5.0 (push pipeline) : subscribe au `frame_callback` de `ScreenCapture` + au `PerceptionEventBus`.

#### Stack proposée (à valider)

- Backend : **FastAPI** + WebSocket (frames vidéo + state updates)
- Frontend : HTML/JS vanilla ou htmx (pas besoin de framework lourd)
- Accessible sur le réseau local : pratique pour suivre depuis téléphone / autre écran

#### Bonus pré-dashboard

- [ ] Commande `--live` (en mode CLI, AVANT le dashboard web) : ouvre une fenêtre OpenCV qui affiche en temps réel ce que l'agent voit avec annotations. Permet de débugger la vision SANS attendre le dashboard complet.

---

## 🔮 Vision Long Terme (Future)

### V6 — Nouvelles capacités de combat

> **Objectif** : étendre les capacités de l'agent au-delà du combat pur.

#### Caméra / scroll

- [ ] Ajouter des actions scroll/pan pour suivre les troupes hors écran
- [ ] Problème actuel : les troupes sortent de l'écran → YOLO ne les voit plus → retraite déclenchée trop tôt
- [ ] Nouvelles actions : scroll_left, scroll_right, scroll_up, scroll_down, center_on_troops
- [ ] Le vecteur d'observation doit inclure la position de la caméra

#### Multi-compo

- [ ] Supporter d'autres armées que GoWitch (LavaLoon, Hybrid, QC, etc.)
- [ ] Adapter le `TroopManager` pour des rôles différents selon la compo
- [ ] Nouveaux templates/CNN pour les troupes non GoWitch
- [ ] L'agent doit apprendre des stratégies différentes selon la compo

#### Équipements héros

- [ ] Détecter les équipements actifs sur chaque héros
- [ ] Adapter la stratégie selon l'équipement (ex: bouclier barbare vs cape d'invisibilité)
- [ ] Enrichir le vecteur d'observation avec l'info équipement

#### Self-play / curriculum

- [ ] Entraîner sur des bases de difficulté croissante
- [ ] Commencer par des bases faciles (HDV 10-11), monter progressivement
- [ ] Potentiellement : utiliser le matchmaking du jeu pour trouver des bases adaptées

#### Gestion du village
- [ ] amélioration troupes/héros/batiments

---

## 🗃️ Backlog & Idées (Non planifiées)

> Idées intéressantes mais pas encore assignées à une version. On pioche ici quand on a du temps.

#### Combat intelligent
- [ ] **Estimation loot avant attaque** — OCR ressources du village adverse + estimation % destruction possible → skip si pas rentable
- [ ] **Classification de base** — détecter le type (farming, war, anti-3★, île) pour adapter la stratégie avant d'attaquer en guerre de clan/ligue de clan car impossible de prédire le village sur lequel on va tomber quand on lance une attaque de famr ou classé.
- [ ] **Analyse des replays** — lire les replays des propres attaques pour extraire des patterns d'erreur et améliorer l'heuristique
- [ ] **Ligue automatique** — monter/descendre en ligue selon l'objectif configuré (max loot, max trophées)
- [ ] **Combats classés** — gérer le mode ranked/ligue avec ses spécificités

#### Gestion village
- [ ] **Queue recherche labo** — détecter quand le labo est libre et lancer la prochaine recherche auto
- [ ] **Overflow ressources** — améliorer quand les reserves sont pleines pour pas se faire piller
- [ ] **Amélioration bâtiments** — queue de priorité d'amélioration (constructeur libre → améliore X selon stratégie)
- [ ] **Gestion bouclier** — acheter/maintenir un bouclier stratégiquement selon les ressources accumulées
- [ ] **Don de troupes automatique** — détecter les demandes du chat et répondre avec les bonnes troupes

#### Recrutement & Social
- [ ] **Recrutement clan** — poster des annonces dans le chat global selon des critères configurables (trophées, activité)
- [ ] **Chat de clan** — répondre aux commandes des membres, gérer les demandes de troupes
- [ ] **Participation guerres de clan** — détecter et rejoindre automatiquement les guerres de clan

#### Infrastructure & UX
- [ ] **Calibration UI automatique** — remplacer `ui_positions.json` par détection auto des boutons via YOLO/template → plus de calibration manuelle après changement d'émulateur ou mise à jour du jeu
- [ ] **Dashboard web temps réel** — page dédiée vision agent (flux vidéo annoté), stats training, courbes reward, état multi-agents (prévu V5)
- [ ] **Replay vidéo des attaques** — enregistrer l'écran ADB pendant le combat pour review
- [ ] **Multi-compte** — gérer plusieurs comptes sur plusieurs émulateurs en parallèle
- [ ] **Comportement humain** — délais aléatoires, patterns de tap naturels, pauses — réduire la détectabilité
- [ ] **Mode coaching** — l'IA analyse une attaque humaine et donne des conseils

#### ML & Training
- [ ] **Curriculum learning** — entraîner sur des bases de difficulté croissante (HDV10 → HDV11 → HDV12) pour convergence plus rapide
- [ ] **Self-play** — s'attaquer soi-même pour générer de la diversité dans le training sans dépendre du matchmaking
- [ ] **Transfer learning** — pré-entraîner sur un TH level, fine-tuner pour les autres
- [ ] **Estimation pré-attaque** — réseau qui prédit le % de destruction avant d'attaquer basé sur armée vs base détectée

#### Apprentissage continu (adaptation aux mises à jour CoC)
> Problème : quand CoC ajoute de nouvelles troupes/bâtiments, l'agent ne les reconnaît plus.
> Solution : human-in-the-loop — l'agent détecte l'inconnu, l'utilisateur labélise, retrain.
> Pas d'API externe, 0€ de coût, 1 semaine de maintenance par mise à jour majeure CoC.

- [ ] **Détection d'inconnus** — quand YOLO conf < seuil sur un objet → flag `unknown_X` + capture automatique du crop dans `needLabelisation/` avec timestamp
- [ ] **Maintenance mode** — l'utilisateur ouvre `needLabelisation/`, labélise les crops dans Roboflow, réentraîne sur Kaggle → agent à jour. ~1 semaine par update majeur CoC.
- [ ] **Notification** — log clair "WARNING: X objets inconnus détectés ce run → voir needLabelisation/"

#### Agent LLM — Coach & Parole autonome (100% local, 0€/mois)
> L'agent dispose de toutes les données : résultats, composition, base, actions...
> Solution locale : **Ollama** (gratuit, open-source) + Llama 3.1 8B ou Mistral 7B.
> Tourne sur le RTX 5070 8GB VRAM — aucun coût API, aucune dépendance externe.
> Fine-tuning optionnel sur RunPod (~20€ one-shot) si on veut du vocabulaire CoC spécifique.

- [ ] **Intégration Ollama** — `uv add ollama` + modèle local (Llama 3.1 8B ou Phi-3 mini ~4GB) → `ollama.generate(model='llama3', prompt=context)` → 0€/mois
- [ ] **Mode coach** — après chaque attaque, passer le contexte au LLM local : résultats, actions, composition → analyse en langage naturel → log ou chat clan
- [ ] **Parole autonome** — l'agent commente ses attaques dans le chat clan : "3★ 100% ! Compo parfaite" / "Base difficile, 2★ mais sorts de soin bien placés"
- [ ] **Conseils stratégiques GdC** — avant une attaque : "Cette base a 2 infernos single → je recommande soin plutôt que rage"
- [ ] **Rapport quotidien** — résumé posté dans le chat : N attaques, X étoiles moyennes, recommandations
- [ ] **Fine-tuning optionnel** — si le modèle de base manque de vocabulaire CoC, fine-tuner Mistral 7B sur RunPod (~20€ one-shot, pas récurrent)
- [ ] **RAG (Retrieval Augmented Generation)** — base de connaissance vectorielle consultée à chaque génération. Élimine les hallucinations sur les stats et permet une mémoire épisodique de l'agent.

  Base de connaissance RAG :
  - **Wiki CoC** (scraping) : stats troupes/bâtiments/sorts par niveau → réponses factuellement exactes
  - **Historique attaques** (auto-alimenté) : les N derniers épisodes → mémoire épisodique "la semaine passée sur une base similaire..."
  - **Meta actuel** : compositions populaires, synergies → conseils stratégiques à jour
  - **Données clan** : membres, activité, règles → contexte personnalisé

  Architecture : ChromaDB (vector store local, 0€) + SentenceTransformers pour les embeddings + Ollama pour la génération.
  Mise à jour CoC → mettre à jour la base RAG uniquement, pas le modèle → 0 ré-entraînement.

  Fine-tuning + RAG = le modèle parle CoC (fine-tuning) + ses réponses sont ancrées dans des faits réels (RAG).

---

## ✅ Versions Précédentes (Terminées)

### V4.2 — Refonte architecture combat

> **Objectif** : supprimer les phases rigides, rendre l'agent vraiment réactif comme un joueur humain. C'est le plus gros changement architectural depuis V3→V4.

#### Bug : échec navigation → faux -50 reward (Session 13, ✅ fix)

> Symptôme observé session 11 (épisode 22) :
> Matchmaking bloqué (`recherche_adversaire`) → 3 recovery échouent → `ERROR: Unable to reach enemy village`
> → l'épisode continue quand même → `_wait_for_battle_end()` voit 1-2 barres vertes (UI) → croit aux troupes mortes → surrender → **-50 reward injuste**

- [x] **`_wait_for_battle_end()` ne surrend plus si écran ≠ phase_attaque** : nouvelle garde dans `clashai/combat/episode_lifecycle.py::wait_for_battle_end()` — détecte les états non-battle (`village_home`, `recherche_adversaire`, `prep_attaque`, `chargement`, `gdc_*`, `menu_*`, `profil`, `chat_clan`) et retourne `None` au lieu de surrendrer
- [x] **`reset()` marque `self._nav_failed = True`** quand `_navigate_to('phase_attaque')` échoue après tous les retries — `clashai/combat/environment.py::reset()` (variable initialisée à `False` en début de reset, set à `True` après échec)
- [x] **Reward `nav_failed = 0.0`** au lieu de -50 — `finish_episode()` court-circuite si `env._nav_failed=True`, retourne `(0.0, info)` avec `info['nav_failed']=True` pour que les training scripts puissent filtrer ces épisodes des stats
- [x] **Retry auto navigation** : si `_navigate_to('phase_attaque')` échoue, `reset()` attend 3s et retente une fois avant de marquer `nav_failed`. Cause fréquente = matchmaker bloqué sur `recherche_adversaire`, le retry suffit souvent

#### ⚠️ CRITIQUE : Amélioration heuristique sorts (apprentissages Session 7)

- [x] Sorts en priorité dans `get_heuristic_sequence()` — burst rage/gel/soin avant les abilities
- [x] Skip des abilities des héros non déployés — `is_deployed()` ajouté dans `hero_ability.py`
- [x] Burst initial de sorts avant les abilities pour garantir leur utilisation même si combat court
- [x] Augmenter `MAX_STEPS_PER_EPISODE` de 65 à 80 (couvrir le wind-down)
- `MAX_COMBAT_STEPS` déjà à 40 dans le code (> 35 requis)

#### ⚠️ CRITIQUE : Bug clic centre écran avant attaque

- [x] Supprimé les 3 taps `(960, 400)` parasites dans `_navigate_to()` et entre épisodes
- [x] Remplacé les 3 taps `(960, 400)` dans `_recovery_sequence()` par `(30, 540)` (bord gauche sûr)

#### ✅ Fix château de clan

- [x] `ClanCastleManager` utilisait `building_detector` inexistant dans `models` — remplacé par `models` dict + `analyze_village()`
- [x] Seuil `_is_castle_full` : pixels 200→230, ratio 0.15→0.30 (moins de faux positifs)
- [x] `_close_menu` : `tap(960,400)` → `tap(30,540)` (évite d'ouvrir un bâtiment)

#### Fusion des phases deploy/combat

- [x] Supprimer le `phase_indicator` binaire (0=deploy, 1=combat) — `PHASE_SIZE = 0` dans `agent_v4.py`
- [x] Toutes les 37 actions disponibles à chaque step (deploy, sorts, abilities, observe)
- [x] L'agent peut faire : golem → attendre 3s → sorcières → rage → observer → gel → ability roi, le tout en continu
- [x] Modifier `action_space.py` : le masking ne dépend plus de la phase mais des ressources restantes — `compute_action_mask` signature simplifiée, phase supprimée
- [x] Modifier `agent_v4.py` : `VECTOR_SIZE` 55→54 dims, checkpoints V4.1 incompatibles
- [x] Modifier `environment_v4.py` : supprimer la logique de transition deploy→combat — phases fusionnées, `_get_obs()` 54 dims, `step()` unifié V4.2, `reset()` force `_phase='combat'`, heuristique sans `done` intermédiaire
- `state_encoder.py` : phase_indicator déjà absent du vecteur (54 dims confirmé)

#### Suppression limite de steps

- [x] `MAX_STEPS_PER_EPISODE=80` → `MAX_STEPS_SAFETY=200` (filet de sécurité uniquement) dans `action_space.py`
- [x] Fin naturelle via `_all_resources_exhausted()` : plus de troupes + sorts + abilities dans `environment_v4.py`
- [x] `step_norm` (step/80) → `time_norm` (elapsed/180s) dans `_get_obs()` — timer COC réel, VECTOR_SIZE reste 54

#### YOLO continu (bâtiments + troupes à chaque step)

- [x] Faire tourner YOLO bâtiments à chaque step d'observation — `_update_combat_observation()` override dans `environment_v4.py`
- [x] Faire tourner YOLO troupes du début à la fin de l'attaque — `_combat_observer.observe()` appelé à chaque `observe`
- [x] Benchmarker le temps d'inférence — logs `⏱️ YOLO buildings: Xms | troops: Xms` à chaque observe
- [x] Fusionner `CombatObserver` et `BuildingDetector` en pipeline unifié — `_update_combat_observation()` orchestre les deux en un seul appel
- [x] Produire un état riche à chaque step — grid + features village mis à jour + combat features YOLO troupes
- [x] Détection de destruction par diff entre deux scans YOLO bâtiments successifs — `_prev_building_count` → `_buildings_destroyed_total`, +2.0 reward/bâtiment
- [x] `feature[0]` du `CombatObserver` = `buildings_remaining_ratio` (était `phase` toujours 1.0) — VECTOR_SIZE reste 54

#### Amélioration zone de déploiement

- [x] Calculer le contour de la base à partir des bounding boxes YOLO bâtiments — `get_perimeter_from_buildings()` dans `deploy_zone.py`
- [x] Placer les troupes juste en dehors de ce contour — hull convexe + offset 35px, filtre UI zones
- [x] Compléter `deploy_zone.py` — nouvelle fonction ajoutée, HSV conservé en fallback V3
- [x] Côté faible déjà géré par `find_best_attack_side()` dans `state_encoder.py`, branché sur `_center_pos` dans `reset()`

#### Reward shaping avancé

- [x] Destruction seconde par seconde — diff YOLO bâtiments sur chaque `observe` (+2.0/bâtiment)
- [x] Survie des héros — `compute_hero_survival_bonus()` en fin d'épisode (+5.0/héros vivant via `combat_features[4]`)
- [x] Efficacité des sorts — rage contextuelle (+2.0 si troupes > 30%), soin contextuel (+3.0 si blessés, +5.0 clutch, +0.5 si gaspillé), gel (+1.5)
- [x] Combo clutch — soin quand `hurt_ratio > 0.5` → +2.0 bonus supplémentaire

### V4.1 — Quick wins & Analyse post-training

> **Objectif** : corriger les irritants, analyser les premiers résultats PPO, petites améliorations sans casser l'architecture.
> **Status** : ✅ Terminé (Session 7 + run validation 192 épisodes)

#### Analyse (333 épisodes)

- [x] Analyser les résultats des 333 épisodes PPO (courbe reward, convergence, stratégies)
- [x] Comparer PPO vs heuristique baseline (2.0⭐ / 75.8% / reward 364)
- [x] Identifier les patterns : PPO n'a PAS convergé (1.16⭐ / 62% moy, sous la baseline)
- [x] Diagnostic : bug critique reward combat + entropy trop élevé + pas d'imitation learning

#### Bugs trouvés et corrigés (Session 7)

- [x] **BUG CRITIQUE** : `_compute_shaping_reward()` passait hero_idx dans le paramètre spell_name → les abilities n'étaient JAMAIS récompensées en combat (`environment_v4.py`)
- [x] **BUG** : `steps` toujours 0 dans le log — `_finish_episode()` ne retournait pas `step` dans info (`environment_v4.py`)
- [x] Ajuster ordre heuristique : siège avant héros (`environment_v4.py`)
- [x] Augmenter `MAX_STEPS` de 50 à 65 (`action_space.py`)
- [x] Réduire `ENTROPY_COEF` de 0.04 à 0.02 — trop d'exploration (`agent_v4.py`)
- [x] Malus sorts non utilisés en fin de combat : `-5.0` par sort restant (`reward_shaping.py` + `environment_v4.py`)

#### Corrections restantes

- [x] Fix double appel YOLO dans `CombatObserver` — `count_by_class()` relançait YOLO sur la même image (`combat_observer.py`)
- [x] Investiguer les 11 épisodes à 0% destruction → taps hors zone de déploiement (fix prévu en V4.2 avec contour YOLO bâtiments)

#### Imitation learning

- [x] Pré-entraîner le PPO sur les épisodes heuristiques (behavioral cloning) — `agent_v4.py` + `train_rl_v4.py`
- [x] Relancer 5 épisodes heuristiques avec les fixes V4.1 pour une nouvelle baseline
- [x] **Run V4.1 de validation** : 192 épisodes PPO + 15 BC — ⭐ moy 1.34 (vs 1.16), 2+⭐ rate 42.7% (vs 29.3%), 0⭐ rate 19.3% (vs 25%). BC fonctionne (1.76⭐ sur ep 1-25), PPO plafonne ensuite → phases rigides sont le limitant → passer à V4.2
- [x] Commande : `uv run python tools/train/train_rl_v4.py --pretrain 15 --episodes 200`

#### Stratégie d'entraînement (décidée Session 7)

- **Petits runs (~200 ep) à chaque patch** pour valider les fixes, pas besoin de convergence
- **Gros runs (500-1000+ ep) réservés à la fin de la V4** quand toute l'architecture est stable
- Logique : à chaque refonte majeure (V4.2 fusion phases, V4.3 CNN barre) on invalide le checkpoint précédent
- Investir 10-20h de training sur une archi finale >> 2h sur chaque version intermédiaire

#### Feature indépendante : demande troupes château de clan

- [x] Détecter le château de clan via YOLO bâtiments (classe `clan_castle`)
- [x] Détecter si le CC est plein (heuristique pixels blancs "PLEIN" au-dessus)
- [x] Template matching pour le bouton "Demande" dans la barre du bas (position instable)
- [x] Bouton "Envoyer" via calibrate_ui `cdc_confirmation` (popup stable centré)
- [x] Cooldown 15 min entre chaque demande
- [x] Intégrer dans `brain.py` : demander des renforts avant chaque attaque (farm + GdC)
- [ ] **Setup requis** : `--capture` pour créer `templates/clan_castle/request.png` + calibrer `cdc_confirmation`
- [ ] Optionnel futur : achat de troupes avec points de capitales / gâteaux de CC

---

## 📜 Historique des versions

| Version | Status | Résumé |
|---------|--------|--------|
| V1 | ✅ Terminé | Une seule décision par attaque |
| V2 | ✅ Terminé | Améliorations intermédiaires |
| V3 | ✅ Terminé | Déploiement séquentiel + combat réactif (289 actions, 1.2M params) |
| V4.0 | ✅ Terminé | Action space simplifié 37 actions + YOLO troupes (Session 6) |
| V4.1 | ✅ Terminé | Fix bugs critiques + BC + run validation 192 ep — CC troops non fonctionnel (Session 7) |
| V4.2 | ✅ Terminé | Fusion phases, YOLO continu, zone deploy murs+bâtiments (segmentation), reward shaping, logs pro (Session 8-11) |
| V4.2.1 | ✅ Fix | PPO value loss, BC loss, ability deadlock, deploy zone walls seg (Session 10-11) |
| V4.3 | ✅ Terminé | YOLO barre troupes, perception async, WGC + normalize, debug overlay, mode --test, screen trace, hard cap héros, suppression rescan périodique (Session 12) |
| V4.4 | 🔄 À faire | Polish perception : mini CNN chiffres (remplace EasyOCR), fix atexit WGC, imgsz aligné par modèle, fix CC, gros run final |
| V5.0 | ✅ Phase 1+2 | Mode "en direct" : push pipeline WGC → PerceptionThread event-driven (Session 13). Phases 3-4 optionnelles. |
| V5.1 | 💡 Vision | Foundation multi-agents : ADB zéro screenshot + BaseAgent interface (BaseAgent déjà créé en Session 13 refactor) |
| V5.2 | 💡 Vision | Nouveaux agents : jeux de clan + gestion village + orchestrateur intelligent + CNN options bar bâtiment |
| V5.3 | 💡 Vision | Dashboard web temps réel (FastAPI + WebSocket + page Vision Agent) |
| V6 | 💡 Vision | Multi-compo, scouting base, ligue auto, curriculum learning |
| V7 | 💡 Vision | Loot decision, calibration auto UI, donation, comportement humain |
| V END | 🎯 Objectif | IA autonome complète — joue, gère, recrute, s'améliore seule |