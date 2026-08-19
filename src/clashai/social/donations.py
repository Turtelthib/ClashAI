# clashai/social/donations.py
# Dons de troupes au clan (V5.2, increment 4).
#
# Les demandes de renfort s'affichent dans le chat de clan, chacune avec un
# bouton "Donner". Le taper ouvre un pop-up où l'on choisit les troupes/sorts.
#
# ⚠️⚠️ SÉCURITÉ — LE POP-UP A DEUX ONGLETS
#   - `dons_normaux` : don classique, GRATUIT ✅
#   - `dons_gemme`   : don payé en GEMMES ❌ (monnaie premium)
# On ne devine JAMAIS lequel est actif : on **tape explicitement
# `dons_normaux`** (gratuit, idempotent) avant de donner quoi que ce soit, et on
# **abandonne** si ce bouton est introuvable. `dons_gemme` n'est jamais tapé.
# Même philosophie que l'anti-gemmes de l'upgrader : sans preuve, on n'agit pas.
#
# Le don est une action BÉNIGNE (on donne, on ne dépense rien) → contrairement
# aux actions de gestion du clan (exclure/promouvoir), elle n'a pas besoin de
# liste blanche.
#
# Ce module fournit les capteurs + le geste ; combien donner et à qui pourra
# être piloté par le LLM (V5.3) via `choose`.

import time

import cv2
import numpy as np

from clashai.config import ADB_HEIGHT, ADB_WIDTH
from clashai.perception.ui_buttons import DETECTOR_MIN_CONFIDENCE as _MIN_CONF

DONATE_BUTTON = 'donner'          # un par demande dans le chat
FREE_TAB = 'dons_normaux'         # onglet GRATUIT — le seul qu'on tape
GEM_TAB = 'dons_gemme'            # onglet GEMMES — jamais tapé, jamais

# Saturation moyenne sous laquelle un bouton est considéré GRISÉ (demande déjà
# satisfaite, ex. "45/45"). Même seuil et même principe que les icônes de la
# barre de troupes (`troop_bar_detector.GRAYED_SAT_THRESHOLD`).
GRAYED_SAT_THRESHOLD = 30

# Le pop-up s'ouvre À DROITE du bouton tapé : on n'accepte que les cartes
# situées au-delà, pour ne pas confondre avec les icônes de troupes DEMANDÉES
# qui s'affichent dans le panneau de chat, à gauche.
POPUP_X_MARGIN = 40

# Bord droit du panneau de chat (aligné sur social/chat/constants.CHAT_ZONE_RIGHT).
# Les icônes des troupes DEMANDÉES sont dedans : c'est ce qui nous dit quoi donner.
CHAT_PANEL_RIGHT = 850

# Délais d'ouverture.
_D_POPUP = 1.0
_D_TAB = 0.5
_D_TAP = 0.6

# Garde-fou de boucle. Une demande peut réclamer jusqu'à ~45 places d'armée :
# avec des troupes bon marché (barbare = 1 place) ça fait beaucoup de taps. 6
# était très en dessous — une demande « 2 ballons + 3 sorcières + 2 zap » (7 dons)
# était tronquée. On monte franchement ; la boucle s'arrête d'elle-même dès qu'il
# n'y a plus rien à donner ou que ça n'avance plus.
MAX_TAPS_PER_REQUEST = 30


class DonationManager:
    """Répond aux demandes de renfort du chat de clan."""

    def __init__(self, detector=None, verbose=True, debug_dir=None):
        self._detector = detector
        self.verbose = verbose
        self._debug_dir = debug_dir

    def _det(self):
        if self._detector is None:
            from clashai.perception.ui_buttons import get_detector
            self._detector = get_detector()
            if self._detector is None:
                from clashai.perception.ui_detector import UIDetector
                self._detector = UIDetector(verbose=False)
        return self._detector

    # ---- capteurs ----------------------------------------------------------

    @staticmethod
    def _is_grayed(crop_pil):
        """True si le crop est désaturé (bouton inactif)."""
        if crop_pil.width < 4 or crop_pil.height < 4:
            return False
        hsv = cv2.cvtColor(np.asarray(crop_pil.convert('RGB')), cv2.COLOR_RGB2HSV)
        return float(np.mean(hsv[:, :, 1])) < GRAYED_SAT_THRESHOLD

    def find_donate_buttons(self, screenshot_pil):
        """Boutons `donner` ACTIFS, de haut en bas. [(x, y, conf), ...]

        Une demande déjà satisfaite (ex. "45/45") a son bouton grisé : le CNN le
        détecte quand même (il a été labellisé dans les deux états), donc on
        filtre nous-mêmes par saturation.
        """
        dets = self._det().detect_raw(screenshot_pil).get(DONATE_BUTTON) or []
        img_w, img_h = screenshot_pil.size
        sx, sy = img_w / ADB_WIDTH, img_h / ADB_HEIGHT

        out = []
        for d in dets:
            if d.conf < _MIN_CONF:
                continue
            x1 = max(0, int((d.x - d.w / 2) * sx))
            y1 = max(0, int((d.y - d.h / 2) * sy))
            x2 = min(img_w, int((d.x + d.w / 2) * sx))
            y2 = min(img_h, int((d.y + d.h / 2) * sy))
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            if self._is_grayed(screenshot_pil.crop((x1, y1, x2, y2))):
                continue                      # demande déjà satisfaite
            out.append((d.x, d.y, d.conf))
        out.sort(key=lambda t: t[1])          # de haut en bas
        return out

    def list_donatable(self, screenshot_pil, models, min_x=0):
        """Cartes donnables du pop-up : [(nom, x, y), ...] en coords ADB.

        Le CNN de la barre de troupes nomme les vignettes ; les grisées (stock
        vide / non autorisées) sont écartées. `min_x` borne à droite du bouton
        tapé pour ignorer les icônes de la demande, affichées dans le chat.
        """
        bar = (models or {}).get('troop_bar_detector')
        if bar is None:
            return []
        try:
            dets = bar.detect(screenshot_pil)
        except Exception:
            return []

        iw, ih = screenshot_pil.size
        sx, sy = ADB_WIDTH / iw, ADB_HEIGHT / ih
        out = []
        for d in dets:
            if d.get('is_grayed') or d.get('no_tap'):
                continue
            cx, cy = d['center']
            ax, ay = int(cx * sx), int(cy * sy)
            if ax < min_x:
                continue                      # panneau de chat, pas le pop-up
            out.append((d['name'], ax, ay))
        out.sort(key=lambda t: (t[2], t[1]))
        return out

    def read_request(self, screenshot_pil, models, button_y, block_top=0):
        """Troupes VERROUILLÉES par la demande (set) — **information**, pas filtre.

        ⚠️ Ne PAS s'en servir pour filtrer les dons : quand une demande verrouille
        des troupes, **le jeu l'impose déjà** (les autres cartes sont grisées dans
        le pop-up, donc `list_donatable` ne les voit pas). Filtrer par-dessus ne
        peut rien ajouter — mais peut retrancher à tort si la lecture d'icône se
        trompe, et on ne donnerait alors rien du tout.

        Ça reste un capteur utile : le LLM saura *ce que cette demande réclame*.

        Le cas qui, LUI, a besoin d'aide, c'est la demande écrite **en toutes
        lettres** (« il me faut des sapeurs et des ballons ») sans verrouillage :
        le jeu laisse alors tout donner, et seul l'OCR du message peut dire quoi
        envoyer. C'est l'objet de l'item OCR-par-message (V5.4) ; `wanted=` de
        `donate_to_request` est là pour recevoir cette liste-là.

        Le bloc est borné verticalement par le bouton précédent (`block_top`) et
        celui de cette demande (`button_y`) : les demandes sont empilées, donc
        cette fenêtre isole proprement une demande, sans géométrie devinée.
        """
        bar = (models or {}).get('troop_bar_detector')
        if bar is None:
            return set()
        try:
            dets = bar.detect(screenshot_pil)
        except Exception:
            return set()

        iw, ih = screenshot_pil.size
        sx, sy = ADB_WIDTH / iw, ADB_HEIGHT / ih
        wanted = set()
        for d in dets:
            cx, cy = d['center']
            ax, ay = cx * sx, cy * sy
            if ax > CHAT_PANEL_RIGHT:       # hors du panneau de chat
                continue
            if not (block_top < ay <= button_y):
                continue                     # appartient à une autre demande
            wanted.add(d['name'])
        return wanted

    # ---- exécuteur ---------------------------------------------------------

    def donate_to_request(self, button_xy, screenshot_fn, tap_fn, models,
                          choose=None, max_taps=MAX_TAPS_PER_REQUEST,
                          wanted=None):
        """Répond à UNE demande. Renvoie (statut, nb_dons).

        `wanted` : troupes à donner en priorité — destiné aux demandes formulées
        **en texte** (OCR, V5.4). Ne PAS y passer les icônes verrouillées : le jeu
        les impose déjà (cf. `read_request`). Si non vide,
        on ne donne **que** celles-là — envoyer 6 barbares à quelqu'un qui
        demande des ballons ne rend service à personne. Si on n'a aucune des
        troupes demandées, on ne donne **rien** (`no_match`) plutôt que
        n'importe quoi.

        Statuts : ok | free_tab_not_found | nothing_to_give | no_match |
                  no_screenshot
        """
        bx, by = button_xy
        tap_fn(bx, by)
        time.sleep(_D_POPUP)

        img = screenshot_fn()
        self._dump(img, 'dons_2_popup')
        if img is None:
            return ('no_screenshot', 0)

        # ⚠️ SÉCURITÉ : sélectionner explicitement l'onglet GRATUIT. Sans lui,
        # on ne peut pas garantir qu'on n'est pas sur l'onglet GEMMES → on sort
        # sans rien taper d'autre.
        free = self._det().detect(img).get(FREE_TAB)
        if free is None or free[2] < _MIN_CONF:
            self._close(img, tap_fn)
            return ('free_tab_not_found', 0)
        tap_fn(int(free[0]), int(free[1]))
        time.sleep(_D_TAB)

        given = 0
        donated = {}          # nom -> nombre de dons déjà faits
        stagnant = 0
        for _ in range(max_taps):
            img = screenshot_fn()
            if img is None:
                break
            cards = self.list_donatable(img, models, min_x=bx + POPUP_X_MARGIN)
            if wanted:
                cards = [c for c in cards if c[0] in wanted]
            if not cards:
                break                      # pop-up vide ou refermé = demande servie

            pick = (choose or self._least_donated)(cards, donated)                 if choose is None else choose(cards)
            if pick is None:
                break
            name, px, py = pick
            tap_fn(px, py)
            given += 1
            donated[name] = donated.get(name, 0) + 1
            time.sleep(_D_TAP)

            # Rien ne bouge (carte toujours là, aucune autre proposée) : on
            # évite de marteler la même troupe indéfiniment.
            stagnant = stagnant + 1 if len(cards) == 1 else 0
            if stagnant >= 3 and len(donated) == 1:
                break

        img = screenshot_fn()
        self._close(img, tap_fn)
        if self.verbose:
            from clashai.config.logging import pp
            detail = f" (demandé : {', '.join(sorted(wanted))})" if wanted else ""
            pp(f" Dons : {given} don(s) pour cette demande{detail}", tag='ok')
        if given:
            return ('ok', given)
        return ('no_match' if wanted else 'nothing_to_give', 0)

    @staticmethod
    def _least_donated(cards, donated):
        """Politique par défaut : la troupe la MOINS donnée jusqu'ici.

        L'ancienne (« toujours la première carte ») martelait une seule troupe :
        sur une demande « ballon + sorcière », elle n'envoyait que des ballons
        tant que le jeu ne les grisait pas. Répartir couvre naturellement les
        demandes mixtes, sans avoir à lire les quantités.

        (Lire les quantités exactes — « 2 ballons, 3 sorcières » — demanderait
        d'OCRiser les compteurs de la demande : voir ROADMAP.)
        """
        return min(cards, key=lambda c: (donated.get(c[0], 0), c[2], c[1]))

    def _close(self, img, tap_fn):
        """Ferme le pop-up (bouton `fermer`, sinon tap neutre)."""
        hit = None
        if img is not None:
            hit = self._det().detect(img).get('fermer')
        if hit is not None and hit[2] >= _MIN_CONF:
            tap_fn(int(hit[0]), int(hit[1]))
        else:
            tap_fn(ADB_WIDTH // 2, int(ADB_HEIGHT * 0.12))
        time.sleep(_D_TAP)

    def _dump(self, img, step):
        """Capture annotée + brute de l'étape (debug_dir seulement)."""
        if not self._debug_dir or img is None:
            return
        import os
        os.makedirs(self._debug_dir, exist_ok=True)
        try:
            img.save(os.path.join(self._debug_dir, f'{step}_raw.png'))
            self._det().annotate(img, os.path.join(self._debug_dir, f'{step}.png'))
        except Exception as e:
            print(f"WARNING: dump '{step}' impossible ({e})")
