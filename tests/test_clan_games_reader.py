# tests/test_clan_games_reader.py
# La lecture de la grille, sur détections factices — aucun modèle, aucun jeu.
#
# Les chiffres viennent de mesures RÉELLES (captures et runs du 22 août 2026) :
# cartes ~213x288 espacées de 254 px, rangées à y 333 et 661, badges ~102 px
# sous leur carte, cartes tronquées par le pop-up à h=59.

import pytest

from clashai.clan_games.reader import ClanGamesReader

COLONNES = (790, 1045, 1299, 1553)


class _D:
    """Une Detection minimale (l'interface qu'attend le reader)."""

    def __init__(self, x, y, w=213, h=288, conf=0.95):
        self.x, self.y, self.w, self.h, self.conf = x, y, w, h, conf


def _carte(x, y, h=288):
    return _D(x, y, w=213, h=h)


def _badge(x, y):
    return _D(x, y, w=190, h=54, conf=0.90)


class _FauxDetecteur:
    def __init__(self, raw):
        self._raw = raw

    def detect_raw(self, img):
        return self._raw


class _FauxWidgets:
    """Lecteur de nombres neutre : la grille se teste sans le digit CNN."""

    def read_number_in(self, img, det):
        return 150

    def read_ratio_in(self, img, det):
        return (0, 1)


def _reader(raw):
    r = ClanGamesReader(detector=_FauxDetecteur(raw), widget_reader=_FauxWidgets())
    # Le gris se mesure sur des pixels : neutralisé ici, il a ses propres tests
    # ailleurs (troop_bar_detector) et n'est pas le sujet.
    r._grisee = lambda img, det: False
    return r


def test_grille_complete_sans_recuperation():
    raw = {
        'carte_defi': [_carte(x, y) for y in (333, 661) for x in COLONNES],
        'point_defi': [_badge(x, y + 102) for y in (333, 661) for x in COLONNES],
    }
    grille = _reader(raw).lire_grille(object(), raw=raw)
    assert len(grille) == 8
    assert not any(d.depuis_badge for d in grille)


def test_une_carte_ratee_est_recuperee_par_son_badge():
    """Le cas exact du run du 22 août : `7×carte_defi, 8×point_defi`.

    `point_defi` est détecté plus fidèlement que `carte_defi` (badges à 8 sur
    tout le run, cartes oscillant entre 5 et 9). Sans récupération, le défi de
    la carte manquante n'est JAMAIS croisé.
    """
    cartes = [_carte(x, y) for y in (333, 661) for x in COLONNES]
    del cartes[3]                                     # le CNN rate une carte
    raw = {
        'carte_defi': cartes,
        'point_defi': [_badge(x, y + 102) for y in (333, 661) for x in COLONNES],
    }
    grille = _reader(raw).lire_grille(object(), raw=raw)
    assert len(grille) == 8
    recuperees = [d for d in grille if d.depuis_badge]
    assert len(recuperees) == 1
    assert (recuperees[0].x, recuperees[0].y) == (1553, 435)   # SUR le badge


def test_le_badge_de_la_rangee_du_dessus_ne_vole_pas_une_carte():
    """Piège mesuré : les rangées sont à 328 px (333 et 661) et les badges à
    ~102 px sous leur carte (437 et 762). Une enveloppe symétrique de 250 px
    rattache donc le badge de la rangée du DESSUS à la carte du dessous
    (661 - 437 = 224 < 250), et la vraie carte manquante n'est plus récupérée.
    L'enveloppe est asymétrique pour ça.
    """
    raw = {
        'carte_defi': [_carte(790, 661)],             # rangée du bas seulement
        'point_defi': [_badge(790, 435), _badge(790, 763)],
    }
    grille = _reader(raw).lire_grille(object(), raw=raw)
    # Le badge du haut (435) appartient à une carte non détectée -> récupéré.
    # Celui du bas (763) appartient à la carte présente -> pas de doublon.
    assert len(grille) == 2
    assert sum(d.depuis_badge for d in grille) == 1


def test_une_carte_tronquee_absorbe_quand_meme_son_badge():
    """Une carte recouverte par le pop-up sort à h=59 et est écartée des CIBLES
    (on ne tape pas une carte à moitié cachée). Mais elle doit rester connue,
    sinon son badge passe pour orphelin et fabrique un doublon — vécu : 9 cibles
    pour 8 défis.
    """
    raw = {
        'carte_defi': [_carte(1297, 192, h=59)],      # tronquée
        'point_defi': [_badge(1294, 185)],            # dy = -7
    }
    grille = _reader(raw).lire_grille(object(), raw=raw)
    assert grille == []            # ni cible tronquée, ni doublon fabriqué


def test_pas_de_recuperation_quand_un_popup_est_ouvert():
    """Le pop-up affiche SON PROPRE compteur de points, visuellement identique
    à un bandeau de carte : le CNN le classe `point_defi`. Sans ce garde-fou on
    fabriquait une carte fantôme au milieu du pop-up (mesuré en 663, 471).
    """
    raw = {
        'carte_defi': [_carte(1299, 333)],
        'point_defi': [_badge(1299, 435), _badge(663, 471)],   # le 2e = le pop-up
        'commencer_defi': [_D(976, 510, w=250, h=90, conf=0.9)],
    }
    grille = _reader(raw).lire_grille(object(), raw=raw)
    assert [(d.x, d.y) for d in grille] == [(1299, 333)]


def test_ordre_de_lecture_par_rangee_puis_colonne():
    # Deux cartes de la même rangée diffèrent de quelques pixels en y (332 vs
    # 333) : un tri brut par y intercalerait les colonnes.
    raw = {'carte_defi': [_carte(1553, 332), _carte(790, 333),
                          _carte(790, 661), _carte(1553, 662)],
           'point_defi': []}
    grille = _reader(raw).lire_grille(object(), raw=raw)
    assert [(d.x, d.y) for d in grille] == [(790, 333), (1553, 332),
                                            (790, 661), (1553, 662)]


@pytest.mark.parametrize('classe', ['progression_defi', 'rejeter'])
def test_defi_engage_ne_depend_pas_de_la_grille(classe):
    """Après un `Commencer` réussi le pop-up reste ouvert et masque les cartes :
    `defi_actif()` (qui passe par la grille) rendait None alors que le défi
    était bien lancé. `defi_engage()` lit les classes directement.
    """
    raw = {classe: [_D(1300, 506, conf=0.90)]}
    assert _reader(raw).defi_engage(raw=raw) is True


def test_defi_engage_est_faux_sur_une_grille_normale():
    raw = {'carte_defi': [_carte(790, 333)], 'point_defi': [_badge(790, 435)]}
    assert _reader(raw).defi_engage(raw=raw) is False
