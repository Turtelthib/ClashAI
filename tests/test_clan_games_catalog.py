# tests/test_clan_games_catalog.py
# Le catalogue des jeux de clan — logique pure, aucun modèle, aucun émulateur.
#
# Les descriptions ci-dessous sont les VRAIES sorties OCR relevées en jeu le
# 22 août 2026 (tools/debug/clan_games_croiser.py), fautes comprises : titres
# massacrés en préfixe, « I0 Fois » pour « 10 fois ». C'est le point du fichier
# — tester sur du texte propre inventé ne prouverait rien du chemin réel.

import pytest

from clashai.clan_games.catalog import (
    Analyse,
    Capacites,
    Catalogue,
    capacites_depuis_world,
    normaliser,
)
from clashai.clan_games.reader import Defi

# --- descriptions réelles, telles que l'OCR les a rendues --------------------

D_BEBE_DRAGON = ("Garderie PQuR bêbês dRagons Gagnez une étoile en combat "
                 "multijoueur en utilisant au moins 1 Bébé dragon.")
D_SAUT = ("SurMonter les obstacles Gagnez une étoile en combat multijoueur "
          "en utilisant au moins 1 Sort de saut.")
D_VALKYRIE = ("chevaucuee des valkyries Gagnez une étoile en combat "
              "multijoueur en utilisant au moins 1 Valkyrie.")
D_GOLEM = "Gagnez une étoile en combat multijoueur en utilisant au moins 1 Golem."
D_OUVRIERS_UNITE = ("Éclaf PYROêlectrique Gagnez une étoile en combat des "
                    "ouvriers en utilisant au moins 1 Sorcier pyroélectrique.")
D_OUVRIERS_ETOILES = "Gagnez 6 étoiles en combat des ouvriers."
D_DETRUIRE = "Détruisez Canon I0 Fois en combat"


@pytest.fixture
def cat():
    # Registre figé : le catalogue ne doit pas dépendre de l'état de troops.json
    # pour que ces tests restent stables.
    return Catalogue(unites_connues={'bebe_dragon', 'valkyrie', 'golem', 'saut',
                                     'dragon', 'barbare'})


def _defi(points=150, description=None, grisee=False, active=False):
    d = Defi(x=800, y=330, w=213, h=288, conf=0.95, points=points,
             grisee=grisee, active=active)
    d.description = description
    return d


# =============================================================================
# Normalisation
# =============================================================================

def test_normalise_accents_et_casse():
    assert normaliser("Gagnez une ÉTOILE") == "gagnez une etoile"


def test_corrige_les_chiffres_ocr_seulement_dans_les_tokens_chiffres():
    # « I0 Fois » -> « 10 fois » : sans ça, la quantité est illisible et le défi
    # serait refusé à tort.
    assert normaliser("Canon I0 Fois") == "canon 10 fois"


def test_ne_massacre_pas_les_mots_sans_chiffre():
    # La correction I->1 / O->0 est bornée aux tokens contenant un chiffre,
    # sinon « Golem » deviendrait « g0lem » et ne matcherait plus le registre.
    assert normaliser("Golem Ile") == "golem ile"


# =============================================================================
# Interprétation des 4 familles réelles
# =============================================================================

@pytest.mark.parametrize('description,unite', [
    (D_BEBE_DRAGON, 'bebe_dragon'),
    (D_VALKYRIE, 'valkyrie'),
    (D_GOLEM, 'golem'),
    (D_SAUT, 'saut'),          # « Sort de saut » -> saut (préfixe retiré)
])
def test_multijoueur_avec_unite(cat, description, unite):
    a = cat.interpreter(description)
    assert a.type == 'multijoueur'
    assert a.contraintes == ['unite_imposee']
    assert a.unite == unite
    assert a.quantite == 1


def test_le_titre_massacre_en_prefixe_ne_gene_pas(cat):
    # « Garderie PQuR bêbês dRagons » précède la phrase utile : on cherche le
    # motif n'importe où dans la ligne, on ne l'ancre pas au début.
    assert cat.interpreter(D_BEBE_DRAGON).type == 'multijoueur'


def test_village_des_ouvriers_est_un_terrain_distinct(cat):
    # « combat des ouvriers » CONTIENT « en combat » (le repli large). L'ordre
    # du JSON doit donc le faire gagner, sinon un défi hors de portée passerait
    # pour un défi du village principal — et serait engagé à tort.
    assert cat.interpreter(D_OUVRIERS_UNITE).type == 'village_ouvriers'
    assert cat.interpreter(D_OUVRIERS_ETOILES).type == 'village_ouvriers'


def test_terrain_implicite_en_combat(cat):
    # « Détruisez Canon 10 fois en combat » ne nomme pas son terrain : c'est le
    # repli `village_principal`. Aucune contrainte -> progresse pendant le farm.
    a = cat.interpreter(D_DETRUIRE)
    assert a.type == 'village_principal'
    assert a.contraintes == []
    assert a.effort == 1


@pytest.mark.parametrize('description', [
    # Familles JAMAIS observées : elles doivent être comprises quand même, car
    # seul le TERRAIN compte pour la faisabilité, pas l'objectif.
    "Pillez 500 000 pieces d'or en combat multijoueur.",
    "Detruisez 300 murs en combat.",
    "Gagnez 5 etoiles en combat multijoueur.",
    "Infligez 1 000 000 de degats avec vos heros en combat.",
    "Eliminez 250 troupes ennemies en combat multijoueur.",
])
def test_un_defi_jamais_vu_reste_lisible(cat, description):
    # LA raison d'être du design par contraintes : aucun de ces défis n'était
    # dans les 7 relevés, et pourtant ils sont tous jouables. La version par
    # gabarits de phrase les aurait TOUS refusés.
    a = cat.interpreter(description)
    assert a.reconnu
    assert cat.evaluer(a, Capacites())[0] is True


def test_un_defi_jamais_vu_hors_de_portee_est_bien_refuse(cat):
    a = cat.interpreter("Gagnez 3 etoiles en guerre de clans.")
    assert a.type == 'guerre'
    assert cat.evaluer(a, Capacites())[1] == 'guerre_non_jouee'


def test_sans_terrain_reconnu_on_refuse(cat):
    # Le garde-fou du design par contraintes : si on ne sait même pas OÙ ça se
    # joue, on ne peut pas juger. Pas de terrain -> pas de faisabilité.
    a = cat.interpreter("Faites trois fois le tour du village en sifflant.")
    assert not a.reconnu
    assert a.type is None
    assert cat.evaluer(a, Capacites())[1] == 'description_non_reconnue'


def test_description_absente(cat):
    assert not cat.interpreter(None).reconnu
    assert not cat.interpreter('').reconnu


# =============================================================================
# Faisabilité
# =============================================================================

def test_unite_presente_dans_la_barre_rend_le_defi_faisable(cat):
    caps = Capacites(unites_disponibles={'golem'})
    assert cat.evaluer(cat.interpreter(D_GOLEM), caps) == (True, 'ok')


def test_unite_absente_refuse_faute_de_savoir_composer(cat):
    # Ce qui bloque n'est PAS un coût — former des troupes est instantané et
    # gratuit en jeu — mais une capacité que le bot n'a pas encore.
    caps = Capacites(unites_disponibles={'barbare'})
    faisable, raison = cat.evaluer(cat.interpreter(D_GOLEM), caps)
    assert not faisable
    assert raison == 'compo_armee_non_pilotee:golem'


def test_savoir_composer_rend_le_defi_faisable_sans_la_troupe(cat):
    """Le jour où le bot choisit ses troupes, un seul drapeau débloque la
    moitié des défis d'une saison (4 sur 7 au relevé du 22 août 2026).

    Former est instantané et gratuit : il n'y a rien d'autre à modéliser, ni
    délai ni dépense. La barre vide ne prouve donc plus rien.
    """
    caps = Capacites(unites_disponibles=set(), composition_armee=True)
    assert cat.evaluer(cat.interpreter(D_GOLEM), caps) == (True, 'ok')


def test_savoir_composer_ne_sauve_pas_une_unite_inconnue(cat):
    # « Sorcier pyroélectrique » n'est pas au registre : on ne peut pas le
    # former, quoi qu'on sache faire. Le refus tient.
    caps = Capacites(unites_disponibles=set(), composition_armee=True)
    faisable, raison = cat.evaluer(cat.interpreter(D_OUVRIERS_UNITE), caps)
    assert not faisable


def test_savoir_composer_ne_debloque_pas_le_village_des_ouvriers(cat):
    caps = Capacites(unites_disponibles=set(), composition_armee=True)
    faisable, raison = cat.evaluer(cat.interpreter(D_OUVRIERS_ETOILES), caps)
    assert not faisable
    assert raison == 'village_des_ouvriers_non_joue'


def test_unite_hors_registre_refuse_sans_approximer(cat):
    # « Sorcier pyroélectrique » n'existe pas dans troops.json (village des
    # ouvriers) : on refuse, on ne rabat pas sur « sorcier ».
    a = cat.interpreter(D_OUVRIERS_UNITE)
    assert a.unite is None
    assert a.unite_brute is not None


def test_village_des_ouvriers_toujours_refuse(cat):
    # Même avec toutes les troupes du monde : le bot n'y joue pas.
    caps = Capacites(unites_disponibles={'golem', 'valkyrie', 'bebe_dragon'})
    for d in (D_OUVRIERS_UNITE, D_OUVRIERS_ETOILES):
        faisable, raison = cat.evaluer(cat.interpreter(d), caps)
        assert not faisable
        assert raison == 'village_des_ouvriers_non_joue'


def test_terrain_sans_modificateur_ne_depend_daucune_troupe(cat):
    # Aucune contrainte de composition -> faisable avec une barre vide.
    caps = Capacites(unites_disponibles=set())
    assert cat.evaluer(cat.interpreter(D_DETRUIRE), caps) == (True, 'ok')


def test_inconnu_nest_jamais_faisable(cat):
    faisable, raison = cat.evaluer(Analyse(), Capacites())
    assert not faisable
    assert raison == 'description_non_reconnue'


# =============================================================================
# Choix
# =============================================================================

def test_choisit_le_meilleur_rendement(cat):
    caps = Capacites(unites_disponibles={'golem'})
    defis = [
        _defi(150, D_GOLEM),        # multijoueur(1) + unite(1) = effort 2 -> 75
        _defi(300, D_DETRUIRE),     # village_principal, effort 1 -> 300  <- gagnant
    ]
    choix = cat.choisir(defis, caps)
    assert choix is not None
    assert choix[0].points == 300


def test_ne_choisit_rien_si_aucun_defi_nest_sur(cat):
    # Tout est soit hors de portée, soit non reconnu -> None, et l'agent
    # réessaiera. Ne JAMAIS engager un défi irréversible par défaut.
    caps = Capacites(unites_disponibles=set())
    defis = [_defi(150, D_GOLEM), _defi(200, D_OUVRIERS_ETOILES),
             _defi(400, "Charabia inconnu du catalogue")]
    assert cat.choisir(defis, caps) is None


def test_ignore_les_cartes_grisees_et_active(cat):
    caps = Capacites(unites_disponibles={'golem'})
    assert cat.choisir([_defi(150, D_GOLEM, grisee=True)], caps) is None
    assert cat.choisir([_defi(150, D_GOLEM, active=True)], caps) is None


def test_ignore_une_carte_dont_les_points_sont_illisibles(cat):
    # On ne compare pas ce qu'on n'a pas lu.
    caps = Capacites(unites_disponibles={'golem'})
    assert cat.choisir([_defi(None, D_GOLEM)], caps) is None


def test_diagnostiquer_donne_une_raison_pour_chaque_defi(cat):
    caps = Capacites(unites_disponibles={'golem'})
    lignes = cat.diagnostiquer([_defi(150, D_GOLEM), _defi(200, D_OUVRIERS_ETOILES)], caps)
    assert [ligne[2] for ligne in lignes] == [True, False]
    assert all(ligne[3] for ligne in lignes)


# =============================================================================
# Capacités depuis le world
# =============================================================================

def test_capacites_lisent_la_barre_de_troupes():
    world = {'troop_bar': [{'name': 'golem', 'is_grayed': False},
                           {'name': 'barbare', 'is_grayed': True}]}
    caps = capacites_depuis_world(world)
    assert caps.unites_disponibles == {'golem'}     # la grisée est épuisée
    assert caps.village_ouvriers is False


def test_barre_non_lue_ne_suppose_aucune_troupe():
    # Un world vide ne doit pas rendre les défis « avec unité » faisables.
    assert capacites_depuis_world({}).unites_disponibles == set()
    assert capacites_depuis_world(None).unites_disponibles == set()


# =============================================================================
# Besoins : ce qui manque, pas seulement que ça manque
# =============================================================================

def test_defis_a_une_troupe_pres_nomme_ce_qui_manque(cat):
    caps = Capacites(unites_disponibles=set())
    besoins = cat.defis_a_une_troupe_pres(
        [_defi(150, D_GOLEM), _defi(150, D_BEBE_DRAGON)], caps)
    assert {b['unite'] for b in besoins} == {'golem', 'bebe_dragon'}
    assert all(b['points'] == 150 for b in besoins)


def test_le_village_des_ouvriers_nest_PAS_un_besoin(cat):
    """Aucune troupe ne débloquera jamais le village des ouvriers.

    Le faire remonter comme un besoin enverrait le cerveau adapter une compo
    pour rien. On sépare l'actionnable du définitif.
    """
    caps = Capacites(unites_disponibles=set())
    besoins = cat.defis_a_une_troupe_pres(
        [_defi(200, D_OUVRIERS_ETOILES), _defi(100, D_OUVRIERS_UNITE)], caps)
    assert besoins == []


def test_un_defi_deja_faisable_nest_pas_un_besoin(cat):
    caps = Capacites(unites_disponibles={'golem'})
    assert cat.defis_a_une_troupe_pres([_defi(150, D_GOLEM)], caps) == []


def test_les_besoins_sont_tries_par_points_decroissants(cat):
    """Le cerveau arbitre : adapter la compo coûte du temps et de l'élixir.
    Lui présenter le plus rentable en premier.
    """
    caps = Capacites(unites_disponibles=set())
    besoins = cat.defis_a_une_troupe_pres(
        [_defi(150, D_GOLEM), _defi(400, D_BEBE_DRAGON)], caps)
    assert [b['points'] for b in besoins] == [400, 150]
