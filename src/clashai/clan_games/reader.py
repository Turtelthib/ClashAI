# clashai/clan_games/reader.py
# Perception des jeux de clan (V5.2) — LIT, ne tape jamais.
#
# Tout part d'UNE inférence CNN UI par frame (`detect_raw`), puis on croise les
# boîtes entre elles. Rien n'est déduit d'une coordonnée en dur : la grille
# scrolle, le pop-up recouvre les cartes, et les positions bougent donc à chaque
# frame.
#
# Ce que le CNN UI v5 (155 classes, 15 nouvelles) fournit ici — confiances
# mesurées sur les captures réelles du 22 août 2026 :
#
#   carte_defi            0.88-0.97   les 8 cartes de la grille
#   point_defi            0.65-0.94   le bandeau vert "400" (points à gagner)
#   progression_defi      0.93        le bandeau gris "0/1" (défi EN COURS)
#   commencer_defi        0.50-0.98   le bouton vert (TRÈS variable, cf. SEUIL_COMMENCER)
#   temps_avant_expiration 0.90-0.91  la limite du défi (ex. "3H")
#   score_personnel       0.75-0.76   "0/10000" = plafond perso
#   score_clan            0.74-0.82   le score du clan
#   temps_restant_jeux    0.97        "5j 17h"
#   raccourci_jdc         0.41-0.53   entrée (icône d'interface)  <- point faible
#   jeux_de_clans         (jamais vu)  entrée (tente sur la carte du village)
#
# ── Le badge appartient à la carte, géométriquement ──────────────────────────
# Mesuré : une `carte_defi` fait ~213x288 (ex. x[684,896] y[517,805]) et son
# `point_defi` ~190x54 tombe DEDANS (y[735,789]). L'appariement se fait donc par
# INCLUSION du centre du badge dans la boîte de la carte — pas par « ~105 px
# sous le centre », qui casserait au premier changement de résolution.
#
# ── Défi actif : lu au gris, pas à une classe ────────────────────────────────
# Quand un défi est engagé, le jeu grise TOUTES les autres cartes et encadre
# l'active en doré. On ne labélise pas ces états : la saturation les sépare,
# comme pour les boutons `donner` (social/donations.py) et les vignettes de la
# barre de troupes. Une carte porteuse d'un `progression_defi` EST l'active.

import re
from dataclasses import dataclass

import cv2
import numpy as np

from clashai.config import ADB_HEIGHT, ADB_WIDTH
from clashai.perception.troop_bar_detector import GRAYED_SAT_THRESHOLD

# ── Classes CNN UI (v5) ─────────────────────────────────────────────────────
TENTE = 'jeux_de_clans'                # la tente posée sur la carte du village
RACCOURCI = 'raccourci_jdc'            # petite icône de l'interface, à côté des événements
CARTE = 'carte_defi'
POINTS = 'point_defi'                  # bandeau vert : points à gagner
PROGRESSION = 'progression_defi'       # bandeau gris "0/1" : défi en cours
COMMENCER = 'commencer_defi'           # bouton vert du pop-up
REJETER = 'rejeter'                    # bouton rouge — DÉTECTÉ, JAMAIS TAPÉ
ONGLET_DEFIS = 'onglet_defi_jdc'
ONGLET_RECOMPENSES = 'recompense_jdc'
ONGLET_CLASSEMENT = 'classement_clan_jdc'
SCORE_PERSO = 'score_personnel'        # "0/10000" = plafond individuel
SCORE_CLAN = 'score_clan'
TEMPS_JEUX = 'temps_restant_jeux'      # "5j 17h"
TEMPS_EXPIRATION = 'temps_avant_expiration'   # limite du défi ouvert

# Seuil d'EXISTENCE des éléments de la fenêtre. Plus bas que le seuil d'action
# global (`ui_buttons.DETECTOR_MIN_CONFIDENCE` = 0.60) parce qu'ici on ne fait
# que constater : mal lire une carte fait rater un défi, pas dépenser.
SEUIL_LECTURE = 0.50

# Seuil de l'ENTRÉE, à part et volontairement bas.
#
# `raccourci_jdc` plafonne à 0.41-0.44 sur les captures du 22 août — sous le
# seuil d'action de 0.60, alors que c'est la classe la plus importante. On
# l'abaisse ICI seulement, pour deux raisons : un faux positif coûte un tap
# inoffensif sur le village (au pire un menu s'ouvre, et `menu_ouvert()` le
# démentira tout de suite), et un faux négatif rend l'agent complètement
# aveugle. Asymétrie franche -> on privilégie le rappel.
#
# ⚠️ C'est un PANSEMENT en attendant le renfort dataset de cette classe. Quand
# `raccourci_jdc` remontera au-dessus de 0.60, supprimer cette constante et
# repasser par find_button().
SEUIL_ENTREE = 0.35

# Seuil d'ACTION du pop-up (taper `Commencer` engage vraiment le défi).
#
# ⚠️ 0.60 était le seuil global hérité du CNN UI **v4**, dont le pic F1 tombait
# à 0.635. Le **v5 a son pic à 0.332** : tout le modèle sort des confiances plus
# basses, et 0.60 est désormais au bord droit du plateau, pas en son centre.
#
# Mesuré sur 6 pop-ups réels (run du 22 août, captures `demo/`) :
#   commencer_defi  0.976 · 0.956 · 0.896 · 0.643 · 0.496 · (absent)
# La confiance dépend fortement de la position du pop-up, qui suit la carte
# tapée. Le run a échoué en `commencer_introuvable` sur le tirage à **0.496** —
# détection pourtant PARFAITEMENT centrée sur le bouton (vérifié à l'image).
#
# On descend donc sous le minimum observé. Le risque reste borné par ailleurs :
# `commencer_defi` n'existe que dans le pop-up de détail, il n'y en a qu'un à
# l'écran, et le selector VÉRIFIE APRÈS LE TAP que le défi est réellement
# devenu actif — la sûreté ne repose donc pas sur ce nombre.
SEUIL_COMMENCER = 0.45

# Conservé pour les autres éléments engageants.
SEUIL_ACTION = 0.60

# Le pop-up recouvre la grille : une carte à moitié cachée est détectée avec une
# boîte tronquée (mesuré : h=59 au lieu de 288). On l'écarte plutôt que de lire
# un badge qui appartient peut-être à la voisine.
HAUTEUR_CARTE_MIN = 150

# Deux cartes de la même rangée ne sont pas exactement à la même hauteur
# (mesuré : y=332 et y=333 sur la rangée du haut). Un tri brut par y intercale
# donc les colonnes. On regroupe par rangée sous cette tolérance.
TOLERANCE_RANGEE = 60

# Enveloppe d'appartenance d'un badge à sa carte. Deux pièges mesurés :
#
#  1. Le pop-up TRONQUE les boîtes de carte (h=59 au lieu de 288) et le badge
#     tombe alors HORS de la boîte réduite. L'inclusion stricte le déclarait
#     orphelin et fabriquait une carte en double (9 cibles pour 8 défis).
#     -> on teste la proximité, pas l'inclusion.
#  2. Une enveloppe SYMÉTRIQUE de 250 px attrape le badge de la rangée du
#     DESSUS : les rangées sont à ~328 px (y 333 et 661), les badges à ~102 px
#     sous leur carte (y 437 et 762) — donc 661-437 = 224 px seulement.
#     -> l'enveloppe est ASYMÉTRIQUE. Un badge est sous sa carte (dy ~ +102),
#        ou quasi confondu avec elle quand la boîte est tronquée (dy ~ -6).
#
# Colonnes espacées de 254 px : dx max 150 ne peut pas mordre sur la voisine.
_BADGE_DX_MAX = 150
_BADGE_DY_MIN = -80      # boîte tronquée : le centre remonte sur le badge
_BADGE_DY_MAX = 200      # < 224, donc jamais la rangée du dessus

# ── Lecture d'un "N/M" : quel lecteur, et pourquoi ──────────────────────────
# `digit_reader.read_widget_ratio` est **documenté** comme mono-chiffre (« valide
# dans CoC où N et M sont des chiffres uniques — 5/5, 1/6, 0/1 ») : il lit le
# premier et le dernier glyphe, le '/' n'étant pas une classe du CNN. Les jeux
# de clan sortent de ce cadre — "0/10000" pour le score, "0/300" pour un défi de
# murs. Mesuré sur les captures du 22 août :
#
#   widget                     digit CNN ratio   OCR
#   progression_defi "0/1"     (0, 1)  ✅        "07"        ❌
#   score_personnel "0/10000"  (0, 0)  ❌        "o/10000"   ✅ (0.96)
#   score_clan "600"           (6, 1)  ❌        "600"       ✅ (0.95)
#
# Aucun des deux ne gagne partout : le digit CNN est le bon sur les glyphes
# minuscules, l'OCR sur les nombres longs. Ce qui les départage sans deviner,
# c'est le NOMBRE DE GLYPHES du crop — "0/1" en fait 3 (chiffre, '/', chiffre),
# "0/10000" en fait 7. Trois glyphes = l'hypothèse mono-chiffre tient.
GLYPHES_RATIO_SIMPLE = 3

# Confusions d'OCR observées sur ces widgets (police du jeu, chiffres cerclés).
_CORRECTIONS_OCR = str.maketrans({'o': '0', 'O': '0', 'l': '1', 'I': '1'})


@dataclass
class Defi:
    """Une carte de la grille, telle qu'on la VOIT (rien d'interprété ici)."""

    x: int                      # centre de la carte (ADB) — le point à taper
    y: int
    w: int
    h: int
    conf: float
    points: int = None          # None = badge absent ou illisible (jamais deviné)
    progression: tuple = None   # (fait, total) si le défi est EN COURS
    grisee: bool = False        # un AUTRE défi est actif -> celle-ci est bloquée
    active: bool = False        # c'est le défi en cours
    # True si la carte a été récupérée depuis son badge de points, faute d'une
    # `carte_defi` détectée. La position est alors celle du BADGE — taper le
    # bandeau ouvre la même carte. Purement informatif (diagnostic, logs).
    depuis_badge: bool = False

    @property
    def disponible(self) -> bool:
        """Engageable maintenant : ni grisée, ni déjà en cours."""
        return not self.grisee and not self.active


class ClanGamesReader:
    """Capteurs des jeux de clan. Aucune méthode de cette classe ne tape."""

    def __init__(self, detector=None, widget_reader=None):
        self._detector = detector
        self._widgets = widget_reader

    def _det(self):
        """Détecteur branché en priorité, sinon instance dédiée (lazy).

        Même cascade que DonationManager / VillageLab : on réutilise le
        détecteur déjà chargé par l'app plutôt que d'en instancier un second.
        """
        if self._detector is None:
            from clashai.perception.ui_buttons import get_detector
            self._detector = get_detector()
            if self._detector is None:
                from clashai.perception.ui_detector import UIDetector
                self._detector = UIDetector(verbose=False)
        return self._detector

    def _wr(self):
        if self._widgets is None:
            from clashai.perception.widget_reader import WidgetReader
            self._widgets = WidgetReader(detector=self._det())
        return self._widgets

    # ------------------------------------------------------------------
    # Helpers géométriques (coords ADB partout)
    # ------------------------------------------------------------------

    @staticmethod
    def _boite(d):
        return (d.x - d.w // 2, d.y - d.h // 2, d.x + d.w // 2, d.y + d.h // 2)

    @classmethod
    def _dedans(cls, det, carte):
        """Le CENTRE de `det` tombe-t-il dans la boîte de `carte` ?"""
        x1, y1, x2, y2 = cls._boite(carte)
        return x1 <= det.x <= x2 and y1 <= det.y <= y2

    def lire_ratio(self, screenshot_pil, det):
        """Lit un widget "N/M" quelle que soit la longueur des nombres.

        Aiguillage par le nombre de glyphes du crop (cf. le bloc en tête de
        module) : 3 glyphes -> digit CNN (fiable sur les petits), sinon OCR
        (fiable sur les longs). Rend None plutôt qu'un nombre deviné.
        """
        from clashai.perception import digit_reader

        crop = self._wr()._widget_crop(screenshot_pil, det)
        if crop is None:
            return None

        if len(digit_reader.segment_widget_glyphs(crop)) <= GLYPHES_RATIO_SIMPLE:
            ratio, _ = digit_reader.read_widget_ratio(crop)
            if ratio is not None:
                return ratio

        for texte in _ocr_lignes(crop):
            m = re.search(r'(\d+)\s*/\s*(\d+)', texte.translate(_CORRECTIONS_OCR))
            if m:
                return (int(m.group(1)), int(m.group(2)))
        return None

    def _grisee(self, screenshot_pil, det):
        """True si le crop est désaturé (un autre défi est actif).

        Même seuil et même principe que `troop_bar_detector` et les boutons
        `donner` : on importe la constante plutôt que d'en recopier la valeur.
        """
        iw, ih = screenshot_pil.size
        sx, sy = iw / ADB_WIDTH, ih / ADB_HEIGHT
        x1, y1, x2, y2 = self._boite(det)
        box = (max(0, int(x1 * sx)), max(0, int(y1 * sy)),
               min(iw, int(x2 * sx)), min(ih, int(y2 * sy)))
        if box[2] - box[0] < 4 or box[3] - box[1] < 4:
            return False
        crop = screenshot_pil.crop(box).convert('RGB')
        hsv = cv2.cvtColor(np.asarray(crop), cv2.COLOR_RGB2HSV)
        return float(np.mean(hsv[:, :, 1])) < GRAYED_SAT_THRESHOLD

    # ------------------------------------------------------------------
    # Capteurs
    # ------------------------------------------------------------------

    def entree(self, screenshot_pil, raw=None):
        """(x, y, conf, classe) d'une entrée vers les jeux de clan, ou None.

        **Deux entrées mènent au même menu, mais elles ne se valent pas** :
          - `raccourci_jdc` : la petite icône de l'interface, à côté des
            événements. **Visible quels que soient le zoom et la position de la
            caméra** — c'est LE signal fiable. 0.41-0.53 en réel.
          - `jeux_de_clans` : la TENTE posée sur la carte du village. Mesuré en
            direct : **elle ne sort que si la caméra est zoomée dessus** (0.42),
            et pas du tout en vue large. Opportuniste, donc.

        On interroge donc le raccourci D'ABORD. Surtout : **l'absence de tente
        ne veut PAS dire que les jeux sont finis** — elle veut dire qu'on ne
        regarde pas au bon endroit. Inverser cet ordre ferait dépendre le
        déclenchement de l'agent du cadrage de la caméra, qu'on ne contrôle pas.

        La présence d'une des deux est le signal « les jeux de clan sont
        actifs » : hors période, le jeu n'affiche ni l'une ni l'autre.
        """
        raw = self._raw(raw, screenshot_pil)
        for classe in (RACCOURCI, TENTE):
            for d in raw.get(classe, []):
                if d.conf >= SEUIL_ENTREE:
                    return (d.x, d.y, d.conf, classe)
        return None

    def menu_ouvert(self, raw=None, screenshot_pil=None):
        """True si la fenêtre des jeux de clan est à l'écran.

        On teste les DEUX autres onglets (`recompense_jdc` 0.93-0.97,
        `classement_clan_jdc` 0.91-0.94) plutôt que l'onglet Défis lui-même :
        `onglet_defi_jdc` plafonne à 0.46-0.66 sur les captures réelles — c'est
        l'onglet ACTIF, donc en surbrillance, et il est sous-représenté dans le
        dataset sous cette forme. Les onglets voisins sont toujours visibles,
        quel que soit celui qui est sélectionné.
        """
        raw = self._raw(raw, screenshot_pil)
        for cls in (ONGLET_RECOMPENSES, ONGLET_CLASSEMENT):
            if any(d.conf >= SEUIL_LECTURE for d in raw.get(cls, [])):
                return True
        return False

    def lire_grille(self, screenshot_pil, raw=None):
        """Les cartes visibles, en ordre de lecture (haut→bas, gauche→droite).

        Une seule inférence : les badges de points et de progression sont
        appariés à leur carte par inclusion géométrique, pas re-détectés.

        ── Récupération par le badge ────────────────────────────────────────
        `point_defi` est DÉTECTÉ PLUS FIABLEMENT que `carte_defi`. Mesuré en run
        réel : `7×carte_defi, 8×point_defi` sur la même frame, et sur tout le
        run les badges restent à 8 pendant que les cartes oscillent entre 5 et 9.
        Une carte ratée fait sauter un défi pour de bon.

        Or un badge IMPLIQUE une carte. Tout badge sans carte à proximité devient
        donc une cible à part entière, positionnée SUR LE BADGE — taper le
        bandeau de points ouvre la même carte. On ne reconstruit pas la boîte de
        la carte par un décalage en pixels : ce serait une coordonnée en dur
        déguisée, cassée au premier changement de résolution.

        ⚠️ **Récupération DÉSACTIVÉE quand un pop-up est ouvert**, pour deux
        raisons mesurées :
          - le pop-up affiche SON PROPRE compteur de points, visuellement
            identique à un bandeau de carte : le CNN le classe `point_defi` et
            on fabriquait une carte fantôme au milieu du pop-up (663, 471) ;
          - le pop-up tronque les cartes qu'il recouvre, elles sont écartées par
            `HAUTEUR_CARTE_MIN`, et leurs badges devenaient orphelins à tort.
        Une carte cachée n'est pas perdue : `selector._grille_stable()` ferme le
        pop-up et relit. On préfère la revoir plus tard que l'inventer ici.
        """
        raw = self._raw(raw, screenshot_pil)

        toutes = [d for d in raw.get(CARTE, []) if d.conf >= SEUIL_LECTURE]
        # Les cartes tronquées par le pop-up sont écartées des CIBLES (on ne
        # tape pas une carte à moitié cachée) mais restent dans `toutes` : elles
        # servent à reconnaître leurs badges, qui sinon passeraient pour
        # orphelins et fabriqueraient des doublons.
        cartes = [d for d in toutes if d.h >= HAUTEUR_CARTE_MIN]
        badges = [d for d in raw.get(POINTS, []) if d.conf >= SEUIL_LECTURE]
        progs = [d for d in raw.get(PROGRESSION, []) if d.conf >= SEUIL_LECTURE]

        wr = self._wr()
        out = []
        for c in cartes:
            defi = Defi(x=c.x, y=c.y, w=c.w, h=c.h, conf=c.conf)

            prog = next((p for p in progs if self._dedans(p, c)), None)
            if prog is not None:
                defi.active = True
                defi.progression = self.lire_ratio(screenshot_pil, prog)
            else:
                badge = next((b for b in badges if self._dedans(b, c)), None)
                if badge is not None:
                    defi.points = wr.read_number_in(screenshot_pil, badge)
                defi.grisee = self._grisee(screenshot_pil, c)

            out.append(defi)

        # --- récupération : un badge orphelin = une carte que le CNN a ratée --
        for badge in (() if self.popup_ouvert(raw=raw) else badges):
            if any(abs(badge.x - c.x) <= _BADGE_DX_MAX
                   and _BADGE_DY_MIN <= (badge.y - c.y) <= _BADGE_DY_MAX
                   for c in toutes):
                continue          # ce badge appartient à une carte déjà vue
            defi = Defi(x=badge.x, y=badge.y, w=badge.w, h=badge.h,
                        conf=badge.conf, depuis_badge=True)
            defi.points = wr.read_number_in(screenshot_pil, badge)
            defi.grisee = self._grisee(screenshot_pil, badge)
            out.append(defi)

        # Rangée d'abord, colonne ensuite : deux cartes voisines diffèrent de
        # quelques pixels en y (332 vs 333), un tri brut par y les intercale.
        # Un badge orphelin s'insère naturellement : il est dans la rangée de sa
        # carte (le bandeau est inclus dans la boîte, pas en dessous).
        out.sort(key=lambda d: (d.y // TOLERANCE_RANGEE, d.x))
        return out

    @staticmethod
    def defi_actif(grille):
        """La carte en cours, ou None. `grille` vient de lire_grille().

        ⚠️ Passe par la GRILLE, donc inutilisable quand un pop-up est ouvert :
        il recouvre les cartes, elles sortent tronquées et `HAUTEUR_CARTE_MIN`
        les écarte — le `progression_defi` n'a alors plus de carte à laquelle
        s'attacher. Pour la simple question « un défi tourne-t-il ? », utiliser
        `defi_engage()`, qui ne dépend d'aucune géométrie.
        """
        return next((d for d in grille if d.active), None)

    def defi_engage(self, screenshot_pil=None, raw=None):
        """True si un défi est en cours — SANS passer par la grille.

        Vécu : après un `Commencer` réussi, le pop-up reste ouvert (il devient
        le pop-up « Rejeter ») et masque les cartes. `defi_actif()` rendait donc
        None alors que `progression_defi` ET `rejeter` étaient bien détectés sur
        la frame. On teste les deux signaux directement au niveau des classes.

        Deux signaux indépendants, c'est voulu : `rejeter` est mal détecté (0/7
        sur les captures du 22 août) et `progression_defi` a besoin que la carte
        active soit visible. L'un rattrape l'autre.
        """
        raw = self._raw(raw, screenshot_pil)
        for classe in (PROGRESSION, REJETER):
            if any(d.conf >= SEUIL_LECTURE for d in raw.get(classe, [])):
                return True
        return False

    def score_personnel(self, screenshot_pil, raw=None):
        """(points_gagnés, plafond) — ex. (0, 10000). None si illisible.

        Le plafond individuel est la condition d'arrêt de l'agent : une fois
        atteint, plus rien à gagner, inutile d'ouvrir la fenêtre.
        """
        raw = self._raw(raw, screenshot_pil)
        dets = raw.get(SCORE_PERSO) or []
        if not dets or dets[0].conf < SEUIL_LECTURE:
            return None
        return self.lire_ratio(screenshot_pil, dets[0])

    def popup_ouvert(self, raw=None, screenshot_pil=None):
        """Un pop-up de détail est-il à l'écran ? Test PUREMENT visuel.

        Séparé de `lire_popup()` parce que celui-ci lance l'OCR pour extraire la
        description : `lire_grille()` n'a besoin que du booléen, et lui faire
        payer un passage EasyOCR à chaque appel serait absurde.

        Trois signaux, dont un seul suffit — `commencer_defi` est très variable
        (0.50-0.98) et `rejeter` quasi jamais détecté, mais
        `temps_avant_expiration` sort à 0.90+ et n'existe QUE dans le pop-up.
        """
        raw = self._raw(raw, screenshot_pil)
        if any(d.conf >= SEUIL_COMMENCER for d in raw.get(COMMENCER, [])):
            return True
        for classe in (REJETER, TEMPS_EXPIRATION):
            if any(d.conf >= SEUIL_LECTURE for d in raw.get(classe, [])):
                return True
        return False

    def lire_popup(self, screenshot_pil, raw=None):
        """Le pop-up de détail d'un défi.

        Renvoie un dict :
            ouvert      : bool
            engage      : bool          — un défi est DÉJÀ en cours (bouton Rejeter)
            commencer   : (x, y) | None — seul point d'action autorisé
            description : str | None    — la phrase qui dit QUOI faire (OCR)

        La description est la seule source de sens : la grille ne porte ni titre
        ni texte (icône + points uniquement), et le titre stylisé du pop-up est
        illisible à l'OCR (`Chaos draconien` -> `Ghaos draconien`). Elle, en
        revanche, sort à 0.59-0.99.
        """
        raw = self._raw(raw, screenshot_pil)

        commencer = next((d for d in raw.get(COMMENCER, [])
                          if d.conf >= SEUIL_COMMENCER), None)
        rejeter = next((d for d in raw.get(REJETER, [])
                        if d.conf >= SEUIL_LECTURE), None)
        expiration = next((d for d in raw.get(TEMPS_EXPIRATION, [])
                           if d.conf >= SEUIL_LECTURE), None)

        ouvert = self.popup_ouvert(raw=raw)

        # `engage` ne repose PAS sur la seule détection de `rejeter` : sur les 7
        # captures du 22 août, le bouton rouge est bien à l'écran et le CNN ne
        # le sort jamais (0 détection). Le pop-up n'ayant que deux états, un
        # pop-up ouvert SANS `Commencer` est forcément l'état « déjà engagé ».
        # Deux chemins vers la même conclusion, dont un qui ne dépend pas d'une
        # classe défaillante.
        engage = rejeter is not None or (ouvert and commencer is None)

        return {
            'ouvert': ouvert,
            'engage': engage,
            'commencer': (commencer.x, commencer.y) if commencer else None,
            'description': self._description(screenshot_pil, expiration) if ouvert else None,
        }

    # ------------------------------------------------------------------
    # OCR de la description
    # ------------------------------------------------------------------

    def _description(self, screenshot_pil, expiration_det):
        """La phrase du pop-up, par OCR. None si l'OCR est indisponible.

        Le bloc de texte est borné par le pop-up lui-même : il tient entre le
        haut de la fenêtre et le bandeau points/durée, dont `temps_avant_
        expiration` donne le repère. Sans ce repère on OCRise plus large et on
        filtre sur la longueur — une description fait une phrase, les autres
        textes de l'écran sont des nombres ou des mots isolés.
        """
        try:
            import easyocr  # noqa: F401
        except ImportError:
            return None

        y_max = expiration_det.y if expiration_det is not None else int(ADB_HEIGHT * 0.45)
        iw, ih = screenshot_pil.size
        sy = ih / ADB_HEIGHT
        crop = screenshot_pil.crop((0, 0, iw, min(ih, int(y_max * sy))))

        lignes = _ocr_lignes(crop)
        # Une description = une phrase. Les nombres, titres et onglets alentour
        # sont courts ; on garde le bloc de texte le plus long.
        phrases = [t for t in lignes if len(t) >= 15]
        return ' '.join(phrases) if phrases else None

    # ------------------------------------------------------------------

    def _raw(self, raw, screenshot_pil):
        """Réutilise un detect_raw fourni, sinon en fait un. Une inférence max."""
        if raw is not None:
            return raw
        if screenshot_pil is None:
            raise ValueError("fournir `raw` ou `screenshot_pil`")
        return self._det().detect_raw(screenshot_pil)


# ── OCR (partagé, chargé une fois) ──────────────────────────────────────────
# social/chat/ocr.py existe mais jette les bbox et initialise en fr+en. Ici le
# jeu est 100 % français et on veut l'ordre de lecture : lecteur dédié.

_reader_ocr = None


def _ocr_lignes(img_pil):
    """Lignes de texte d'une image, dans l'ordre de lecture. [] si l'OCR échoue."""
    global _reader_ocr
    try:
        if _reader_ocr is None:
            import easyocr
            _reader_ocr = easyocr.Reader(['fr'], gpu=False, verbose=False)
        res = _reader_ocr.readtext(np.asarray(img_pil.convert('RGB')), paragraph=False)
    except Exception:
        return []
    res.sort(key=lambda r: (r[0][0][1], r[0][0][0]))
    return [t.strip() for _b, t, c in res if c > 0.3 and t.strip()]
