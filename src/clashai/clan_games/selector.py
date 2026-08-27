# clashai/clan_games/selector.py
# Le GESTE des jeux de clan : ouvrir -> croiser -> choisir -> engager -> fermer.
#
# La perception est dans reader.py, la décision dans catalog.py. Ici, rien que
# des taps et de l'orchestration — c'est le seul fichier du package qui touche
# au jeu, donc le seul à relire quand on se demande « qu'est-ce qu'il peut
# taper ? ».
#
# ⚠️⚠️ CE QUE CE MODULE PEUT TAPER — liste exhaustive
#
#   1. l'entrée des jeux de clan sur le village (`raccourci_jdc` / `jeux_de_clans`)
#   2. le centre d'une `carte_defi` détectée À L'INSTANT
#   3. `commencer_defi`, et UNIQUEMENT si `confirmer=True`
#   4. le point neutre de fermeture / le `fermer` de la fenêtre
#
# Aucune autre coordonnée n'est atteignable : il n'existe pas de branche du code
# qui tape autre chose. En particulier **`rejeter` n'est jamais tapé** — rejeter
# un défi engagé a une pénalité en jeu.
#
# ── Trois garde-fous, éprouvés hors ligne avant d'exister ici ────────────────
#   1. LISTE BLANCHE de points (ci-dessus).
#   2. VETO GÉOMÉTRIQUE : avant chaque tap de carte, on vérifie que le point ne
#      tombe pas dans la boîte de `commencer_defi` ni de `rejeter`. Le pop-up
#      recouvre la grille — sans ce veto, taper « la carte 5 » d'après une
#      lecture périmée pourrait atterrir sur COMMENCER. Le tap est ANNULÉ, pas
#      recalé : un point suspect ne se corrige pas, il s'abandonne.
#   3. RELECTURE à chaque tour : jamais d'action sur une position d'il y a une
#      seconde.
#
# ── Pourquoi `confirmer` est faux par défaut ────────────────────────────────
# Engager un défi est IRRÉVERSIBLE. Le mode par défaut fait donc tout le
# travail — ouvrir, croiser, interpréter, désigner le gagnant — et s'arrête
# juste avant le tap. Même philosophie que `VillageUpgrader` (annulation sûre
# par défaut) et `VillageLab` (`--confirm` pour dépenser).

import time

from clashai.clan_games.reader import COMMENCER, REJETER, ClanGamesReader

# Délais (s). Le pop-up s'ouvre avec une animation : trop court et on lit
# l'écran d'avant, en concluant à tort que la carte n'a pas de description.
_D_MENU = 1.5
_D_POPUP = 1.2
_D_FERMETURE = 0.8

# ── Vérification de l'engagement : on SCRUTE, on ne dort pas ────────────────
# Après le tap `Commencer`, le jeu met un temps variable à basculer (grisage des
# autres cartes + barre de progression). Vécu : la vérification à 1,2 s a conclu
# « pas engagé » alors que le défi était bien parti — la bascule s'est vue sur la
# frame suivante, ~1 s plus tard, pendant la fermeture. Un délai fixe ne fait que
# déplacer le problème : trop court on se trompe, trop long on ralentit tout.
# On interroge donc jusqu'à ce que ce soit vrai, avec une borne.
_VERIF_INTERVALLE = 0.5
_VERIF_TIMEOUT = 5.0

# ── Stabilisation de la grille ──────────────────────────────────────────────
# Le pop-up de détail RECOUVRE une partie des cartes : le CNN n'en voit alors
# plus que 5 ou 6 sur 8, et celles qui restent sont tronquées. Vécu : le
# croisement indexait dans la grille relue à chaque tour, donc dès qu'elle
# rétrécissait les dernières cartes devenaient inatteignables (« carte 8
# absente de la frame »). Fermer le pop-up suffit — mais la fermeture est
# animée, et relire trop tôt donne encore une grille tronquée. On attend donc
# que la grille soit REVENUE à sa taille de référence.
_GRILLE_INTERVALLE = 0.4
_GRILLE_TIMEOUT = 3.0

# La grille de RÉFÉRENCE ne peut pas se lire sur une seule frame : le compte
# varie (vu en réel 5, 6, 7, 8 et même 9 `carte_defi` sur le même écran). Une
# référence lue à 7 alors qu'il y en a 8 fait sauter une carte définitivement.
# On échantillonne et on garde la grille la PLUS COMPLÈTE.
_ECHANTILLONS_REFERENCE = 3

# Deux cartes voisines sont espacées d'environ 254 px. Une carte re-détectée à
# moins de cette distance de sa position de référence EST la même carte.
_TOLERANCE_APPARIEMENT = 120

# Point neutre de fermeture d'un pop-up : le panneau d'illustration à GAUCHE de
# la fenêtre. Choisi au-dessus du bandeau de score, qui porte un « + » d'achat.
POINT_NEUTRE = (370, 450)

# Garde-fou de boucle : la grille en montre 8, on ne croise jamais plus.
MAX_CARTES = 12


class ClanGamesSelector:
    """Ouvre les jeux de clan, lit les défis, engage le meilleur."""

    def __init__(self, reader=None, catalogue=None, verbose=True, debug_dir=None):
        self._reader = reader or ClanGamesReader()
        self._catalogue = catalogue
        self.verbose = verbose
        self._debug_dir = debug_dir

    def _cat(self):
        if self._catalogue is None:
            from clashai.clan_games.catalog import Catalogue
            self._catalogue = Catalogue()
        return self._catalogue

    def _log(self, msg, tag='info'):
        if not self.verbose:
            return
        from clashai.config.logging import pp
        pp(f" Jeux de clan : {msg}", tag=tag)

    # ------------------------------------------------------------------
    # Garde-fous
    # ------------------------------------------------------------------

    def _interdit(self, x, y, raw):
        """Nom du bouton engageant sur lequel (x, y) tombe, ou None."""
        for classe in (COMMENCER, REJETER):
            for d in raw.get(classe, []):
                x1, y1, x2, y2 = self._reader._boite(d)
                if x1 <= x <= x2 and y1 <= y <= y2:
                    return classe
        return None

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def ouvrir(self, screenshot_fn, tap_fn):
        """Ouvre la fenêtre depuis le village. True si elle est bien ouverte."""
        img = screenshot_fn()
        if img is None:
            return False

        if self._reader.menu_ouvert(screenshot_pil=img):
            return True                       # déjà ouverte, rien à taper

        entree = self._reader.entree(img)
        if entree is None:
            return False                      # jeux inactifs, ou entrée non vue

        x, y, conf, classe = entree
        self._log(f"entrée '{classe}' à ({x}, {y}) conf={conf:.2f}")
        tap_fn(x, y)
        time.sleep(_D_MENU)

        img = screenshot_fn()
        return img is not None and self._reader.menu_ouvert(screenshot_pil=img)

    def fermer(self, screenshot_fn, tap_fn):
        """Referme la fenêtre (bouton `fermer`, sinon tap neutre)."""
        img = screenshot_fn()
        cible = None
        if img is not None:
            hit = self._reader._det().detect(img).get('fermer')
            if hit is not None and hit[2] >= 0.60:
                cible = (int(hit[0]), int(hit[1]))
        tap_fn(*(cible or POINT_NEUTRE))
        time.sleep(_D_FERMETURE)

    # ------------------------------------------------------------------
    # Croisement
    # ------------------------------------------------------------------

    def croiser(self, screenshot_fn, tap_fn):
        """Ouvre chaque carte pour en lire la description. -> list[Defi].

        Chaque `Defi` rendu porte un attribut `description` (str | None) : la
        grille n'ayant AUCUN texte, c'est le seul moyen de savoir ce que fait
        un défi. Ne tape que des centres de cartes et le point neutre.
        """
        img = screenshot_fn()
        if img is None or not self._reader.menu_ouvert(screenshot_pil=img):
            return []

        depart = self._grille_reference(screenshot_fn)
        reference = len(depart)
        self._log(f"{reference} carte(s) dans la grille de référence")
        resultats = []

        for i, cible in enumerate(depart[:MAX_CARTES]):
            cartes, raw = self._grille_stable(screenshot_fn, tap_fn, reference)
            if cartes is None:
                break

            # ⚠️ On cible une POSITION mémorisée, pas un indice. Vécu : la
            # grille relue changeait de taille (7 puis 8 cartes), donc
            # `cartes[i]` ne désignait plus la même carte — un run a ouvert
            # DEUX FOIS le même défi et en a sauté deux autres.
            carte = self._apparier(cible, cartes)
            if carte is None:
                self._log(f"carte {i + 1} introuvable près de "
                          f"({cible.x},{cible.y}), ignorée", tag='warn')
                continue

            faute = self._interdit(carte.x, carte.y, raw)
            if faute:
                self._log(f"tap annulé : ({carte.x},{carte.y}) tombe sur '{faute}'",
                          tag='warn')
                continue

            tap_fn(carte.x, carte.y)
            time.sleep(_D_POPUP)

            img = screenshot_fn()
            if img is None:
                break
            carte.description = self._reader.lire_popup(img)['description']
            resultats.append(carte)
            self._dump(img, f'defi_{i + 1:02d}')

            # Journaliser CE QUE L'OCR A LU, carte par carte. Sans ça, un
            # « aucun défi sûr » est indébogable : impossible de distinguer
            # « la description n'a pas été lue » de « elle a été lue et
            # refusée ». C'est l'information la plus utile du croisement.
            pts = carte.points if carte.points is not None else '?'
            if carte.description:
                self._log(f"défi {i + 1}/{len(depart)} ({pts} pts) : "
                          f"{carte.description}")
            else:
                self._log(f"défi {i + 1}/{len(depart)} ({pts} pts) : "
                          f"DESCRIPTION NON LUE", tag='warn')

        tap_fn(*POINT_NEUTRE)
        time.sleep(_D_FERMETURE)
        return resultats

    def _grille_reference(self, screenshot_fn):
        """La grille la plus complète vue sur plusieurs échantillons.

        Une seule frame ne suffit pas : le compte de `carte_defi` varie d'un
        relevé à l'autre sur un écran pourtant statique. Sous-estimer la
        référence fait sauter des défis pour de bon.
        """
        meilleure = []
        for _ in range(_ECHANTILLONS_REFERENCE):
            img = screenshot_fn()
            if img is not None:
                cartes = self._reader.lire_grille(img)
                if len(cartes) > len(meilleure):
                    meilleure = cartes
            time.sleep(_GRILLE_INTERVALLE)
        return meilleure

    @staticmethod
    def _apparier(cible, cartes):
        """La carte la plus proche de `cible`, ou None si trop loin."""
        if not cartes:
            return None
        proche = min(cartes, key=lambda c: (c.x - cible.x) ** 2 + (c.y - cible.y) ** 2)
        distance = ((proche.x - cible.x) ** 2 + (proche.y - cible.y) ** 2) ** 0.5
        return proche if distance <= _TOLERANCE_APPARIEMENT else None

    def _grille_stable(self, screenshot_fn, tap_fn, reference):
        """Ferme tout pop-up et attend que la grille retrouve `reference` cartes.

        Renvoie (cartes, raw) — `cartes` est None si l'écran est perdu. Après le
        timeout on rend la MEILLEURE grille vue plutôt que rien : une grille
        partielle vaut mieux qu'un abandon, l'appelant sait la gérer.
        """
        limite = time.time() + _GRILLE_TIMEOUT
        meilleur = (None, None)
        ferme = False

        while True:
            img = screenshot_fn()
            if img is None:
                return meilleur
            raw = self._reader._det().detect_raw(img)

            # Un pop-up ouvert recouvre la grille -> le fermer, une seule fois.
            # ⚠️ C'est un INDICE, pas une condition de lecture : si la détection
            # du pop-up restait bloquée sur « ouvert », gater la lecture dessus
            # ferait tout caler sans rien lire. On tente la fermeture, puis on
            # lit de toute façon — c'est le NOMBRE DE CARTES qui tranche.
            if not ferme and self._reader.lire_popup(img, raw=raw)['ouvert']:
                tap_fn(*POINT_NEUTRE)
                ferme = True
                time.sleep(_D_FERMETURE)
                continue

            cartes = self._reader.lire_grille(img, raw=raw)
            if len(cartes) >= reference:
                return (cartes, raw)
            if meilleur[0] is None or len(cartes) > len(meilleur[0]):
                meilleur = (cartes, raw)

            if time.time() >= limite:
                if meilleur[0] is not None and len(meilleur[0]) < reference:
                    self._log(f"grille incomplète : {len(meilleur[0])}/{reference} "
                              f"cartes après {_GRILLE_TIMEOUT:.0f} s", tag='warn')
                return meilleur
            time.sleep(_GRILLE_INTERVALLE)

    # ------------------------------------------------------------------
    # Flux complet
    # ------------------------------------------------------------------

    def engager_le_meilleur(self, screenshot_fn, tap_fn, capacites,
                            confirmer=False):
        """Ouvre, croise, choisit, engage. -> (statut, détails).

        Statuts :
            ok                  un défi a été engagé ET vérifié actif à l'écran
            engagement_non_confirme  `Commencer` tapé, mais le défi n'apparaît
                                pas actif ensuite -> à vérifier à la main
            choisi_non_engage   un gagnant est désigné, tap retenu (confirmer=False)
            deja_engage         un défi est déjà en cours -> rien à faire
            plafond_atteint     le score personnel est au maximum
            menu_introuvable    jeux inactifs, ou entrée non détectée
            aucun_defi_lu       la grille n'a rien donné
            aucun_defi_sur      rien n'est faisable avec certitude -> on ne tente rien
            commencer_introuvable  le bouton n'est pas détecté au seuil d'action
        """
        if not self.ouvrir(screenshot_fn, tap_fn):
            return ('menu_introuvable', {})

        img = screenshot_fn()
        raw = self._reader._det().detect_raw(img) if img is not None else {}

        # 1. Plafond personnel : au max, plus rien à gagner, on n'ouvre plus.
        score = self._reader.score_personnel(img, raw=raw) if img is not None else None
        if score and score[1] and score[0] >= score[1]:
            self.fermer(screenshot_fn, tap_fn)
            return ('plafond_atteint', {'score': score})

        # 2. Un défi déjà engagé ? On ne le rejette JAMAIS pour en prendre un
        #    autre — la pénalité en jeu coûte plus que le gain espéré.
        grille = self._reader.lire_grille(img, raw=raw) if img is not None else []
        actif = ClanGamesReader.defi_actif(grille)
        if actif is not None:
            self.fermer(screenshot_fn, tap_fn)
            return ('deja_engage', {'progression': actif.progression, 'score': score})

        # 3. Croiser pour connaître le SENS de chaque carte.
        defis = self.croiser(screenshot_fn, tap_fn)
        if not defis:
            self.fermer(screenshot_fn, tap_fn)
            return ('aucun_defi_lu', {'score': score})

        lues = sum(1 for d in defis if getattr(d, 'description', None))
        self._log(f"{len(defis)} défi(s) croisé(s), {lues} description(s) lue(s)")

        # Ce qu'il MANQUE, remonté dans tous les cas — y compris quand un défi
        # est engagé : le cerveau peut vouloir préparer le suivant.
        bloques = self._cat().defis_a_une_troupe_pres(defis, capacites)
        if bloques:
            manque = ', '.join(f"{d['unite']} ({d['points']} pts)" for d in bloques)
            self._log(f"débloquables si le bot savait composer son armée "
                      f"(instantané et gratuit en jeu) : {manque}")

        choix = self._cat().choisir(defis, capacites)
        if choix is None:
            self.fermer(screenshot_fn, tap_fn)
            # POURQUOI rien n'est passé : une raison par défi. Sans ce
            # détail, « aucun défi sûr » ne dit pas s'il faut corriger le
            # catalogue, l'OCR, ou simplement entraîner d'autres troupes.
            self._log("aucun défi sûr — on ne tente rien, on réessaiera", tag='warn')
            dispo = ', '.join(sorted(capacites.unites_disponibles)) or 'AUCUNE'
            self._log(f"troupes disponibles : {dispo}")
            for defi, analyse, _ok, raison in self._cat().diagnostiquer(defis, capacites):
                pts = defi.points if defi.points is not None else '?'
                self._log(f"   {pts} pts · terrain={analyse.type} · {raison}")
            return ('aucun_defi_sur',
                    {'lus': len(defis), 'descriptions_lues': lues, 'score': score,
                     'defis_bloques': bloques})

        defi, analyse = choix
        details = {
            'points': defi.points,
            'terrain': analyse.type,
            'contraintes': analyse.contraintes,
            'unite': analyse.unite,
            'description': defi.description,
            'score': score,
            'defis_bloques': bloques,
        }
        self._log(f"choix : {defi.points} pts — terrain {analyse.type}"
                  + (f", contraintes {analyse.contraintes}" if analyse.contraintes else ""),
                  tag='ok')

        if not confirmer:
            self.fermer(screenshot_fn, tap_fn)
            return ('choisi_non_engage', details)

        return self._engager(defi, screenshot_fn, tap_fn, details)

    def _engager(self, defi, screenshot_fn, tap_fn, details):
        """Rouvre la carte gagnante, tape `Commencer`, puis VÉRIFIE. IRRÉVERSIBLE.

        La vérification finale est le vrai garde-fou. `commencer_defi` sort entre
        0.50 et 0.98 selon la position du pop-up : aucun seuil ne peut à la fois
        ne jamais rater un vrai bouton et ne jamais en inventer un. On ne cherche
        donc pas à être sûr AVANT le tap — on constate APRÈS.

        Un défi engagé se voit sans ambiguïté : `progression_defi` apparaît et
        le pop-up passe en `rejeter`. Si ce n'est pas le cas, on le DIT
        (`engagement_non_confirme`) au lieu d'annoncer un succès non constaté.

        ⚠️ La vérification NE passe PAS par `lire_grille` : après le tap, le
        pop-up reste ouvert (il devient celui du rejet), recouvre les cartes,
        et `HAUTEUR_CARTE_MIN` les écarte — un engagement réussi était alors
        rapporté comme non confirmé. `defi_engage()` lit les classes
        directement, sans géométrie.
        """
        img = screenshot_fn()
        if img is None:
            return ('commencer_introuvable', details)

        raw = self._reader._det().detect_raw(img)
        if self._interdit(defi.x, defi.y, raw):
            return ('commencer_introuvable', details)

        tap_fn(defi.x, defi.y)
        time.sleep(_D_POPUP)

        img = screenshot_fn()
        if img is None:
            return ('commencer_introuvable', details)

        pop = self._reader.lire_popup(img)
        self._dump(img, 'engagement')

        # `commencer` est renseigné SEULEMENT au seuil d'action (0.60) : sans
        # détection franche, on n'engage pas à l'aveugle.
        if pop['engage'] or pop['commencer'] is None:
            self.fermer(screenshot_fn, tap_fn)
            return ('commencer_introuvable', details)

        tap_fn(*pop['commencer'])
        engage = self._attendre_engagement(screenshot_fn, details)
        self.fermer(screenshot_fn, tap_fn)

        if not engage:
            self._log("`Commencer` tapé mais aucun défi actif à l'écran — "
                      "à vérifier à la main", tag='warn')
            return ('engagement_non_confirme', details)

        self._log(f"défi engagé et vérifié actif : {details['points']} pts"
                  + (f" (progression {details['progression']})"
                     if details.get('progression') else ""), tag='ok')
        return ('ok', details)

    def _attendre_engagement(self, screenshot_fn, details):
        """Scrute jusqu'à voir le défi engagé, ou abandonne après le timeout.

        Renseigne `details['progression']` au passage si la grille est lisible
        (elle ne l'est pas tant qu'un pop-up la recouvre) — c'est un bonus pour
        le journal, jamais une condition.
        """
        limite = time.time() + _VERIF_TIMEOUT
        derniere = None
        essai = 0
        while True:
            img = screenshot_fn()
            if img is not None:
                derniere = img
                essai += 1
                raw = self._reader._det().detect_raw(img)
                if self._reader.defi_engage(raw=raw):
                    actif = ClanGamesReader.defi_actif(
                        self._reader.lire_grille(img, raw=raw))
                    if actif is not None:
                        details['progression'] = actif.progression
                    self._dump(img, 'apres_engagement')
                    return True
            if time.time() >= limite:
                break
            time.sleep(_VERIF_INTERVALLE)

        # Échec : on garde la DERNIÈRE frame vue, celle qui servira au debug.
        self._dump(derniere, 'apres_engagement')
        self._log(f"engagement non constaté après {_VERIF_TIMEOUT:.0f} s "
                  f"({essai} relevés)", tag='warn')
        return False

    # ------------------------------------------------------------------

    def _dump(self, img, etape):
        """Capture annotée de l'étape (debug_dir seulement)."""
        if not self._debug_dir or img is None:
            return
        import os
        os.makedirs(self._debug_dir, exist_ok=True)
        try:
            img.save(os.path.join(self._debug_dir, f'{etape}.png'))
        except Exception as e:
            print(f"WARNING: dump '{etape}' impossible ({e})")
