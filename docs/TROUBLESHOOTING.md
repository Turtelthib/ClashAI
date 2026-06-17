# ClashAI — Troubleshooting (blocs de fix détaillés)

Référence des bugs non-triviaux déjà résolus : **symptômes → cause → fix → pièges → tests**.
Si un de ces problèmes réapparaît, relire le bloc correspondant avant de re-debugger.

> Pour la liste chronologique de tout ce qui est fait, voir [CHANGELOG.md](CHANGELOG.md).
> Pour ce qui reste à faire, voir [ROADMAP.md](ROADMAP.md).

## Sommaire

- [Capture fenêtre émulateur occluded (WGC)](#-capture-fenêtre-émulateur-occluded-wgc)
- [RGB/BGR inversé sur l'input YOLO](#-rgbbgr-inversé-sur-linput-yolo)
- [Capacités héros jamais déclenchées (mode heuristique)](#-capacités-héros-jamais-déclenchées-mode-heuristique)
- [Migration capacités héros : template → CNN](#-migration-capacités-héros--template--cnn)
- [Alignement `imgsz` par modèle YOLO (+ historique troop bar)](#-alignement-imgsz-par-modèle-yolo)
- [Demande de troupes château de clan (5 bugs)](#-demande-de-troupes-château-de-clan-5-bugs)
- [Échec navigation → faux -50 reward](#-échec-navigation--faux--50-reward)
- [Famine d'agent dans le scheduler (CC monopolise, combat ne tourne pas)](#-famine-dagent-dans-le-scheduler)

---

## 🔧 Capture fenêtre émulateur occluded (WGC)

> Si la capture montre l'écran du PC au lieu du jeu, le CNN écran déraille, ou l'agent voit le bureau → relire ce bloc.

**Symptômes**
- `_debug_capture.png` contient VS Code / le terminal / le bureau au lieu de l'émulateur
- Le CNN classificateur d'écran prédit toujours `chargement` ou des écrans aléatoires
- L'agent RL ne reconnaît pas l'état du jeu et fait n'importe quoi
- `ScreenCapture` log un backend `dxcam` ou `mss`

**Cause racine**
- Google Play Games rend en **DirectX/Vulkan dans une surface GPU accélérée**, pas dans la couche GDI lisible par PrintWindow
- `dxcam`/`mss` lisent les pixels de **l'écran physique** → si VS Code est devant, ils capturent VS Code
- `PrintWindow` + `PW_RENDERFULLCONTENT` retourne le contenu de l'écran (pas du buffer fenêtre) pour ces émulateurs hardware-accélérés

**Solution (en place)**
- Backend **WGC (Windows.Graphics.Capture)** via le package `windows-capture` (wrapper Rust). C'est l'API que OBS/Snipping Tool utilisent, conçue pour les apps DirectX.
- Ordre des backends (`perception/screen_capture/capture.py::_init_backend()`) : **`wgc → printwindow → dxcam → mss → adb`**
- WGC tourne en background thread (`start_free_threaded`) et met à jour `self._wgc_latest` (BGRA numpy) à chaque frame ; `_grab_wgc()` lit ce buffer sans latence.
- **Routing via WGC** : `game_loop.adb_screenshot()` essaie `get_capture().grab()` (WGC) puis fallback ADB. Tout le code passe par là → training + brain + perception sur WGC.
- **Normalisation 1920x1080** (`_normalize_to_canonical()`) : WGC/PrintWindow/dxcam/mss capturent toute la fenêtre (titlebar + bordures) à la résolution OS DPI-scalée. Le CNN écran + YOLO + positions UI sont calibrés sur la sortie ADB native 1920x1080. La normalisation crop le chrome via `GetClientRect`+`ClientToScreen`+facteur DPI puis resize en 1920x1080. **Sans ce step, le CNN écran délire (la barre noire en haut le déstabilise).**

**Pièges déjà rencontrés**
1. **VS Code match** : `"Google Play"` matchait `"Fix Google Play emulator - COCProj - Visual Studio Code"` → liste `EXCLUDED_TITLE_SUBSTRINGS`.
2. **adbproxy.exe match** : titre type chemin `C:\...\adbproxy.exe` → filtre `\\` et `.exe`.
3. **Fenêtre minimisée** : rejetée par le filtre taille → **l'émulateur ne doit pas être minimisé** (derrière, c'est OK).
4. **Cleanup thread** : "Fatal Python error" au shutdown — bénin (cf bloc atexit WGC dans CHANGELOG).
5. **`adb_screenshot()` reverté en pur ADB** pendant un cycle de debug → *test OK / training KO*. Restauré : WGC d'abord, fallback ADB.
6. **Mismatch résolution** sans normalisation → WGC renvoyait 1283x751 / 2560x1528 avec titlebar → CNN écran délirait.

**Commandes de test**
```bash
uv run python -c "from clashai.perception.screen_capture import ScreenCapture; c = ScreenCapture(); print('backend=', c.backend); img = c.grab(); img.save('_wgc_smoketest.png') if img else print('FAIL')"
uv run python tools/debug/test_screen_capture.py
uv run python tools/debug/inspect_emulator_window.py
```

**Si WGC casse encore** : vérifier que la fenêtre n'est pas minimisée ; que `windows-capture` est installé ; lire le log d'init (titre suspect → ajouter exclusion) ; `inspect_emulator_window.py` (si TOUTES les enfants capturent l'écran → l'émulateur a changé sa pile de rendu).

---

## 🔧 RGB/BGR inversé sur l'input YOLO

> Si une détection YOLO se trompe **systématiquement** sur des classes dépendantes de la couleur (gel↔poison, soin↔clone) alors que le tool manuel `detect_troop_bar.py` sur la MÊME image est parfait → relire ce bloc.

**Symptôme** : détection fausse systématique sur les classes couleur ; le tool manuel sur la même image = 100% correct ; même modèle+conf+imgsz+image → résultats différents.

**Cause racine** : **Ultralytics lit un `np.ndarray` comme du BGR** (convention cv2), mais un `PIL.Image` comme du RGB. La prod faisait `model.predict(np.array(screenshot_pil))` → octets RGB interprétés comme BGR → canaux R/B inversés. Le tool manuel passait le PIL directement. La SEULE différence = `np.array(pil)` vs `pil`.

**Solution (en place)** : passer le `PIL.Image` directement à `.predict()` — JAMAIS `np.array(pil)` brut. Corrigé dans `troop_bar_detector.detect` + `analyze_village`. **Règle** : pour passer un numpy à ultralytics, TOUJOURS `cv2.cvtColor(arr, COLOR_RGB2BGR)` d'abord ; sinon passer le PIL.

**Bonus défensif — verrou d'inférence** : `perception/inference_lock.py::INFERENCE_LOCK = threading.RLock()` global, acquis autour de chaque appel modèle. Les modèles ultralytics/torch ne sont pas thread-safe et `PerceptionThread` + `test_run_capture` appellent les mêmes objets → on sérialise.

---

## 🔧 Capacités héros jamais déclenchées (mode heuristique)

**Symptôme** : l'agent ne déclenche JAMAIS les capacités héros en combat, alors que les héros sont bien déployés.

**Cause racine** (mode heuristique = `--test` + brain sans checkpoint) : `get_heuristic_sequence()` construit toute la séquence **en une fois, juste après `reset()`** — avant tout deploy. La boucle abilities était gardée par `if self._hero_manager.is_deployed(hero_name)`, mais `reset()` vient de remettre `_deployed = {tous False}` → check toujours False au build → **aucune action `ability` ajoutée**. Les deploys suivants passaient `is_deployed=True` trop tard (séquence déjà figée).

**Fix** (`combat/environment_v4/heuristic.py`) : gate sur l'**inventaire build-time** (`TROOP_TYPES[i]['role']=='hero'` et `_remaining_troops[i]>0`) au lieu de `is_deployed()`. + `wait_long` avant le bloc abilities pour laisser la capacité se charger (sinon `*_capa` grisé → exclu du mask).

**Pièges** :
- `is_deployed()` = état runtime, inutilisable dans un plan construit à l'avance. Raisonner **inventaire (build-time)** vs **état (runtime)**.
- Le `*_capa` est grisé ~quelques sec après deploy (cooldown de charge) → exclu tant que grisé. D'où le `wait_long`.
- En mode RL le chemin était OK (le mask s'ouvre quand le CNN voit un `*_capa` non-grisé). La séquence corrigée sert aussi de démos BC.

**Test** :
```bash
uv run python -c "
import numpy as np
from clashai.combat.environment_v4.heuristic import HeuristicMixin
from clashai.combat.legacy.agent import TROOP_TYPES, TROOP_NAME_TO_IDX
from clashai.combat.action_space import decode_action, HERO_NAMES
from clashai.combat.hero import HeroAbilityManager
class F(HeuristicMixin):
    def __init__(s, r): s._remaining_troops=r; s.verbose=False; s._hero_manager=HeroAbilityManager(verbose=False); s._hero_manager.reset()
r=np.zeros(len(TROOP_TYPES),dtype=int); r[TROOP_NAME_TO_IDX['golem']]=2; r[TROOP_NAME_TO_IDX['roi']]=1; r[TROOP_NAME_TO_IDX['reine']]=1
seq=[decode_action(a) for a in F(r).get_heuristic_sequence()]
print('abilities:', [HERO_NAMES[d[1]] for d in seq if d[0]=='ability'])  # -> ['roi','reine']
"
uv run python tools/train/train_rl_v4.py --test   # guetter les logs '<hero> ability activated'
```

---

## 🔧 Migration capacités héros : template → CNN

**Motivation** : `HeroAbilityManager` détectait les capas par template matching (zone hardcodée `ABILITY_ZONE` y=850-1080) alors que le CNN troop bar détecte DÉJÀ les classes `<hero>_capa`. Deux systèmes redondants (violation DRY) + crops manuels `ability_*.png` fragiles.

**Fix** : suppression du template matching, remplacé par la lecture des `*_capa` du `TroopBarDetector` (qui tourne déjà dans `PerceptionThread` → **zéro inférence en plus**).
- `HeroAbilityManager.update_from_troop_bar(detections)` : mappe `<hero>_capa` → `<hero>`, garde les héros de `HERO_NAMES` (`duc_draconique` ignoré), lit le `center` comme position de tap. `is_grayed=True` (utilisée/cooldown) → exclu. Présence d'un `*_capa` = preuve de déploiement → marque `_deployed`.
- Supprimés : `template_match.py`, `_load_templates()`, `_templates`, `scan()` template, `ABILITY_ZONE_*`/`MATCH_THRESHOLD`/`TEMPLATES_DIR`.
- Câblage `environment_v4` (3 sites) : async path → `update_from_troop_bar(state['troop_bar'])` ; fallback → `bar_det.detect()` ; `_execute_ability` re-scan idem.

**Pièges/décisions** :
- `scan()` conservé en shim déprécié (signature `scan(screenshot_pil=None, troop_bar_detections=None)`) pour ne pas casser `legacy/`.
- `has_templates()` conservé → retourne toujours `False` (legacy `if has_templates(): scan()` devient un no-op propre).
- `prince_gargouille` dans `HERO_NAMES` (géré) ; `duc_draconique` a un `_capa` mais hors `HERO_NAMES` → ignoré.

**Tests** :
```bash
uv run python -m clashai.combat.hero.cli --file logs/test_run/attaque_30s.png
uv run python -c "from clashai.combat.hero_ability import HeroAbilityManager as M; m=M(verbose=True); m.reset(); print(m.update_from_troop_bar([{'name':'roi_capa','center':(300,980),'conf':.9,'is_grayed':False},{'name':'reine_capa','center':(380,980),'conf':.9,'is_grayed':True}])); print(m.get_ability_mask())"
```

---

## 🔧 Alignement `imgsz` par modèle YOLO

**Constat** : `model.predict()` sans `imgsz=` → Ultralytics utilise 640 par défaut, peu importe l'imgsz d'entraînement → perte de détail silencieuse. Constantes ajoutées par modèle :

| Modèle | Constante | Valeur | Module |
|---|---|---|---|
| troop bar | `YOLO_IMGSZ` | **1088** | `troop_bar_detector.py` |
| bâtiments | `YOLO_BUILDINGS_IMGSZ` | 1600 | `navigation/game_loop` |
| troupes combat | `YOLO_TROOPS_IMGSZ` | 640 | `troop_detector.py` |
| walls seg | `YOLO_WALLS_IMGSZ` | 640 | `perception/deploy` |

**📜 Historique imgsz troop bar** : (1) `1600` (valeur du script d'entraînement) → en prod 0-1 icône/9 (double-resize WGC→LANCZOS→letterbox trop blur). (2) `640` (default) → 9/9 mais qualité moyenne. (3) **retrain dédié `1088`** → mieux, mérite encore plus de data/epochs.

**Conf** : `YOLO_CONF` troop bar = **0.40** (0.45 droppait golem @0.41 ; 0.50 loupe).

---

## 🔧 Demande de troupes château de clan (5 bugs)

1. **`verbose=False`** sur le CC manager → toutes les failures silencieuses. Fix : `verbose=True`.
2. **Pas de check `screen == village_home`** avant l'appel → YOLO ne trouvait pas le château hors village. Fix : guard `classify_screen()=='village_home'`.
3. **`try/except: pass`** → exceptions avalées. Fix : log explicite.
4. **Mismatch nom de classe YOLO** : `_find_clan_castle()` cherchait `'clan_castle'` (anglais) mais le modèle utilise `'chateau_clan'` (français, cf `weights/classes.json`) → match jamais → **la vraie raison pour laquelle l'agent ignorait le CC**. Fix : `CC_CLASSES = ('chateau_clan', 'clan_castle')`.
5. **`_close_menu` tapait l'icône chat** : tap à `(30,540)` = bouton chat clan → ouvrait le chat au lieu de fermer. `KEYCODE_BACK` rejeté (déclenche "quitter le jeu" sur l'émulateur). Fix : tap à `(5,5)` (coin hors-UI).

---

## 🔧 Échec navigation → faux -50 reward

**Symptôme** : matchmaking bloqué (`recherche_adversaire`) → recovery échoue → l'épisode continue → `_wait_for_battle_end()` voit des barres vertes UI → croit aux troupes mortes → surrender → **-50 reward injuste**.

**Fix** :
- `wait_for_battle_end()` (`combat/episode_lifecycle.py`) ne surrend plus si écran ≠ `phase_attaque` (détecte les états non-battle → retourne `None`).
- `reset()` marque `self._nav_failed = True` si `_navigate_to('phase_attaque')` échoue après retries.
- `finish_episode()` court-circuite si `_nav_failed` → reward `0.0` (au lieu de -50) + `info['nav_failed']=True` pour filtrer ces épisodes.
- Retry auto : `reset()` attend 3s et retente une fois (matchmaker bloqué = cause fréquente, le retry suffit souvent).

---

## 🔧 Famine d'agent dans le scheduler

> Si un agent ne se déclenche jamais (`brain --mode farm` ne fait que des pauses, CombatAgent jamais lancé) → relire ce bloc.

**Symptôme** : en V5.1 (brain branché sur l'`AgentScheduler`), le bot ne fait que `_human_pause()` ; un agent prioritaire (ex. `ClanCastleAgent` prio 20) tourne en silence à chaque tick et l'agent moins prioritaire (`CombatAgent` prio 10) n'a jamais son tour.

**Cause racine** : un agent **prio-haute** dont `can_run` reste **True en permanence** et `cooldown_seconds = 0` **monopolise** le scheduler (`pick()` le renvoie à chaque tick). Cas concret : `ClanCastleAgent` avait délégué son cooldown au `ClanCastleManager` — mais le manager n'avance `_last_request_time` que sur une requête **réussie**. Template `request` manquant → requête échoue → `time_until_next_request()` reste à 0 → CC "prêt" en boucle → famine de `CombatAgent`.

**Fix** : donner un **cooldown scheduler** à l'agent (`ClanCastleAgent.cooldown_seconds = REQUEST_COOLDOWN`). Le scheduler pose `_last_run_at` après **chaque** `run()` (succès **ou** échec, cf `BaseAgent._execute` `finally`) → l'agent rend la main pour 15 min même si son run n'a rien fait.

**Règle générale** : tout agent prioritaire doit soit avoir un `cooldown_seconds > 0`, soit devenir `can_run=False` après avoir agi (ex. `GdCAgent` vide sa file). Sinon il affame les agents en dessous. `CombatAgent` (prio la plus basse) peut rester à cooldown 0 — il n'affame personne et son `run()` dure plusieurs minutes (une attaque complète).

**Test** :
```bash
uv run python -c "
from clashai.agents import AgentScheduler, CombatAgent, ClanCastleAgent
from clashai.social.clan_castle import ClanCastleManager
s=AgentScheduler(); c=CombatAgent(models=None,verbose=False)
cc=ClanCastleAgent(manager=ClanCastleManager(models=None,verbose=False),screenshot_fn=lambda:None,tap_fn=lambda *a,**k:None,verbose=False)
s.register(c); s.register(cc); w={'mode':'farm','on_village_home':True}
print('tick1', s.pick(w).name); cc._execute(); print('tick2', s.pick(w).name)  # clan_castle puis combat
"
```
