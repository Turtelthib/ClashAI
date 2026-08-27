# tests/test_clan_games_selector.py
# Le GESTE des jeux de clan, sur détections factices — aucun modèle, aucun jeu.
#
# Ce fichier existe surtout pour les GARDE-FOUS. Engager un défi est
# irréversible et rejeter un défi engagé a une pénalité : les invariants
# « `rejeter` n'est jamais tapé » et « `commencer` seulement si confirmer »
# doivent être vérifiés mécaniquement, pas relus à l'œil.

import pytest

from clashai.agents.clan_games_agent import ClanGamesAgent
from clashai.clan_games.catalog import Capacites
from clashai.clan_games.reader import Defi
from clashai.clan_games.selector import POINT_NEUTRE, ClanGamesSelector

D_GOLEM = "Gagnez une étoile en combat multijoueur en utilisant au moins 1 Golem."
D_DETRUIRE = "Détruisez Canon 10 fois en combat"


class FauxReader:
    """Reader scripté : rend ce qu'on lui dit, sans modèle ni image."""

    def __init__(self, grille=None, popup=None, entree=None, menu=True,
                 score=None, interdits=None, grille_apres=None,
                 releves_avant_bascule=0):
        self._grille = grille if grille is not None else []
        # Ce que le jeu montre APRÈS un `Commencer` réussi : le défi devient
        # actif. Le test bascule `engage_tape` depuis son tap_fn.
        self._grille_apres = grille_apres
        self.engage_tape = False
        # Nombre de relevés pendant lesquels le jeu n'a PAS encore basculé
        # après le tap. Reproduit le bug réel : la bascule arrive en retard.
        self._releves_avant_bascule = releves_avant_bascule
        self.releves = 0
        self._popup = popup or {'ouvert': False, 'engage': False,
                                'commencer': None, 'description': None}
        self._entree = entree
        self._menu = menu
        self._score = score
        self._interdits = interdits or {}

    def entree(self, img, raw=None):
        return self._entree

    def menu_ouvert(self, raw=None, screenshot_pil=None):
        return self._menu

    def defi_engage(self, screenshot_pil=None, raw=None):
        if not self.engage_tape:
            return False
        self.releves += 1
        return self.releves > self._releves_avant_bascule

    def lire_grille(self, img, raw=None):
        if self.engage_tape and self._grille_apres is not None:
            return list(self._grille_apres)
        return list(self._grille)

    def score_personnel(self, img, raw=None):
        return self._score

    def lire_popup(self, img, raw=None):
        return dict(self._popup)

    def _det(self):
        faux = self

        class _D:
            def detect_raw(self, img):
                return faux._interdits

            def detect(self, img):
                return {}
        return _D()

    @staticmethod
    def _boite(d):
        return (d.x - d.w // 2, d.y - d.h // 2, d.x + d.w // 2, d.y + d.h // 2)

    @staticmethod
    def defi_actif(grille):
        return next((d for d in grille if d.active), None)


class _Det:
    """Détection minimale (l'interface qu'attend `_interdit`)."""

    def __init__(self, x, y, w=200, h=60):
        self.x, self.y, self.w, self.h = x, y, w, h


def _defi(points=300, description=D_DETRUIRE, x=800, y=330, **kw):
    d = Defi(x=x, y=y, w=213, h=288, conf=0.95, points=points, **kw)
    d.description = description
    return d


@pytest.fixture(autouse=True)
def horloges_rapides(monkeypatch):
    """Neutralise les délais réels du selector.

    Ils existent pour laisser au JEU le temps d'animer ; sur des détections
    factices ils ne font que rallonger la suite (20 s pour ce fichier). On garde
    un timeout de vérification non nul pour que la boucle de scrutation soit
    réellement exercée — c'est elle qu'on teste.
    """
    import clashai.clan_games.selector as sel
    for nom in ('_D_MENU', '_D_POPUP', '_D_FERMETURE',
                '_VERIF_INTERVALLE', '_GRILLE_INTERVALLE'):
        monkeypatch.setattr(sel, nom, 0.001)
    monkeypatch.setattr(sel, '_VERIF_TIMEOUT', 0.2)
    monkeypatch.setattr(sel, '_GRILLE_TIMEOUT', 0.1)


@pytest.fixture
def io():
    """(screenshot_fn, tap_fn, taps) — l'image est un jeton, jamais lue."""
    taps = []
    return (lambda: object()), (lambda x, y: taps.append((x, y))), taps


@pytest.fixture
def caps():
    return Capacites(unites_disponibles={'golem'})


# =============================================================================
# Garde-fous — les tests qui comptent
# =============================================================================

def test_sans_confirmer_le_bouton_commencer_nest_jamais_tape(io, caps):
    screenshot_fn, tap_fn, taps = io
    reader = FauxReader(
        grille=[_defi()],
        popup={'ouvert': True, 'engage': False, 'commencer': (976, 510),
               'description': D_DETRUIRE},
    )
    sel = ClanGamesSelector(reader=reader, verbose=False)
    statut, details = sel.engager_le_meilleur(screenshot_fn, tap_fn, caps,
                                              confirmer=False)
    assert statut == 'choisi_non_engage'
    assert details['points'] == 300
    assert (976, 510) not in taps          # l'invariant


def test_le_veto_annule_un_tap_qui_tombe_sur_commencer(io, caps):
    # Le pop-up recouvre la grille : la « carte » est en fait sous le bouton.
    # Le tap doit être ABANDONNÉ, pas recalé ailleurs.
    screenshot_fn, tap_fn, taps = io
    reader = FauxReader(
        grille=[_defi(x=976, y=510)],
        interdits={'commencer_defi': [_Det(976, 510)]},
    )
    sel = ClanGamesSelector(reader=reader, verbose=False)
    resultats = sel.croiser(screenshot_fn, tap_fn)
    assert resultats == []
    assert (976, 510) not in taps
    assert taps == [POINT_NEUTRE]           # seule la fermeture a été émise


def test_une_grille_tronquee_par_le_popup_est_reessayee(io):
    """Le pop-up de détail recouvre des cartes : le CNN n'en voit plus que 5 sur
    8. Vécu : le croisement indexait dans cette grille rétrécie, donc les
    dernières cartes devenaient inatteignables (« carte 8 absente de la frame »).
    On ferme le pop-up et on attend que la grille soit revenue.
    """
    screenshot_fn, tap_fn, taps = io
    pleine = [_defi(x=800 + 250 * i, y=330) for i in range(4)]
    reader = FauxReader(grille=pleine,
                        popup={'ouvert': False, 'engage': False,
                               'commencer': None, 'description': D_DETRUIRE})

    # Après le 1er relevé, la grille rétrécit (pop-up ouvert), puis revient.
    releves = {'n': 0}
    pleine_ref = list(pleine)

    def lire_grille(img, raw=None):
        releves['n'] += 1
        if 2 <= releves['n'] <= 4:
            return pleine_ref[:2]          # tronquée par le pop-up
        return list(pleine_ref)

    reader.lire_grille = lire_grille

    sel = ClanGamesSelector(reader=reader, verbose=False)
    resultats = sel.croiser(screenshot_fn, tap_fn)
    # Les 4 cartes sont croisées malgré la grille tronquée entre-temps.
    assert len(resultats) == 4


def test_le_veto_couvre_aussi_rejeter(io):
    screenshot_fn, tap_fn, taps = io
    reader = FauxReader(grille=[_defi(x=990, y=520)],
                        interdits={'rejeter': [_Det(990, 520)]})
    sel = ClanGamesSelector(reader=reader, verbose=False)
    sel.croiser(screenshot_fn, tap_fn)
    assert (990, 520) not in taps


def test_un_defi_deja_engage_nest_jamais_rejete(io, caps):
    # Le rejeter pour en prendre un « meilleur » coûte une pénalité en jeu.
    screenshot_fn, tap_fn, taps = io
    actif = _defi(points=150, active=True)
    actif.progression = (0, 1)
    reader = FauxReader(grille=[actif],
                        interdits={'rejeter': [_Det(990, 520)]})
    sel = ClanGamesSelector(reader=reader, verbose=False)
    statut, details = sel.engager_le_meilleur(screenshot_fn, tap_fn, caps)
    assert statut == 'deja_engage'
    assert details['progression'] == (0, 1)
    assert (990, 520) not in taps


def test_rien_de_sur_ne_declenche_aucun_engagement(io):
    # Barre vide + défi qui exige un golem -> aucun candidat -> on ne tente rien.
    screenshot_fn, tap_fn, taps = io
    reader = FauxReader(
        grille=[_defi(points=150, description=D_GOLEM)],
        popup={'ouvert': True, 'engage': False, 'commencer': (976, 510),
               'description': D_GOLEM},
    )
    sel = ClanGamesSelector(reader=reader, verbose=False)
    statut, _ = sel.engager_le_meilleur(screenshot_fn, tap_fn,
                                        Capacites(unites_disponibles=set()),
                                        confirmer=True)
    assert statut == 'aucun_defi_sur'
    assert (976, 510) not in taps


def test_commencer_non_detecte_nengage_pas_a_laveugle(io, caps):
    screenshot_fn, tap_fn, taps = io
    reader = FauxReader(
        grille=[_defi()],
        popup={'ouvert': True, 'engage': False, 'commencer': None,
               'description': D_DETRUIRE},
    )
    sel = ClanGamesSelector(reader=reader, verbose=False)
    statut, _ = sel.engager_le_meilleur(screenshot_fn, tap_fn, caps,
                                        confirmer=True)
    assert statut == 'commencer_introuvable'


def test_avec_confirmer_le_defi_est_engage(io, caps):
    # Succès = `Commencer` tapé ET le défi constaté actif ensuite.
    screenshot_fn, _tap, taps = io
    engage = _defi(active=True)
    engage.progression = (0, 10)
    reader = FauxReader(
        grille=[_defi()],
        grille_apres=[engage],
        popup={'ouvert': True, 'engage': False, 'commencer': (976, 510),
               'description': D_DETRUIRE},
    )

    def tap_fn(x, y):
        taps.append((x, y))
        if (x, y) == (976, 510):
            reader.engage_tape = True      # le jeu bascule : défi actif

    sel = ClanGamesSelector(reader=reader, verbose=False)
    statut, details = sel.engager_le_meilleur(screenshot_fn, tap_fn, caps,
                                              confirmer=True)
    assert statut == 'ok'
    assert (976, 510) in taps
    assert details['terrain'] == 'village_principal'
    assert details['progression'] == (0, 10)


def test_la_bascule_tardive_du_jeu_est_attendue_pas_manquee(io, caps):
    """Vécu : la vérification à 1,2 s a conclu « pas engagé » alors que le défi
    était bien parti — le jeu a basculé ~1 s plus tard. Un délai fixe ne fait que
    déplacer le problème ; on scrute jusqu'à la borne.
    """
    screenshot_fn, _tap, taps = io
    engage = _defi(active=True)
    engage.progression = (0, 10)
    reader = FauxReader(
        grille=[_defi()],
        grille_apres=[engage],
        popup={'ouvert': True, 'engage': False, 'commencer': (976, 510),
               'description': D_DETRUIRE},
        releves_avant_bascule=3,       # le jeu traîne 3 relevés
    )

    def tap_fn(x, y):
        taps.append((x, y))
        if (x, y) == (976, 510):
            reader.engage_tape = True

    sel = ClanGamesSelector(reader=reader, verbose=False)
    statut, details = sel.engager_le_meilleur(screenshot_fn, tap_fn, caps,
                                              confirmer=True)
    assert statut == 'ok'
    assert reader.releves > 3          # il a bien fallu insister


def test_un_engagement_non_constate_nest_pas_annonce_comme_reussi(io, caps):
    """Le vrai garde-fou : `commencer_defi` sort entre 0.50 et 0.98 selon la
    position du pop-up, donc aucun seuil ne garantit que le tap a porté. On tape,
    puis on CONSTATE — et si le défi n'est pas actif, on le dit.

    Vécu : un run réel a échoué en `commencer_introuvable` sur une détection à
    0.496 qui visait pourtant le bouton exactement. Baisser le seuil ne suffit
    pas ; il faut aussi cesser d'annoncer un succès non vérifié.
    """
    screenshot_fn, tap_fn, taps = io
    reader = FauxReader(               # le jeu ne bascule PAS : rien d'actif
        grille=[_defi()],
        popup={'ouvert': True, 'engage': False, 'commencer': (976, 510),
               'description': D_DETRUIRE},
    )
    sel = ClanGamesSelector(reader=reader, verbose=False)
    statut, _ = sel.engager_le_meilleur(screenshot_fn, tap_fn, caps,
                                        confirmer=True)
    assert (976, 510) in taps                     # on a bien tapé
    assert statut == 'engagement_non_confirme'    # mais on ne le revendique pas


def test_un_engagement_non_confirme_remonte_comme_echec():
    # Contrairement à `aucun_defi_sur` (refus voulu), ne pas savoir si le tap a
    # porté est un vrai échec : ça doit apparaître dans AgentResult.error.
    class _Sel:
        def engager_le_meilleur(self, *a, **kw):
            return ('engagement_non_confirme', {'points': 300})

    agent = ClanGamesAgent(selector=_Sel(), screenshot_fn=lambda: None,
                           tap_fn=lambda x, y: None, verbose=False)
    res = agent.run()
    assert res.ok is False
    assert res.error == 'engagement_non_confirme'


# =============================================================================
# Plafond et navigation
# =============================================================================

def test_plafond_atteint_arrete_tout(io, caps):
    screenshot_fn, tap_fn, taps = io
    reader = FauxReader(grille=[_defi()], score=(10000, 10000))
    sel = ClanGamesSelector(reader=reader, verbose=False)
    statut, _ = sel.engager_le_meilleur(screenshot_fn, tap_fn, caps, confirmer=True)
    assert statut == 'plafond_atteint'


def test_menu_deja_ouvert_ne_tape_rien_pour_louvrir(io):
    screenshot_fn, tap_fn, taps = io
    sel = ClanGamesSelector(reader=FauxReader(menu=True), verbose=False)
    assert sel.ouvrir(screenshot_fn, tap_fn) is True
    assert taps == []


def test_sans_entree_visible_le_menu_est_introuvable(io, caps):
    screenshot_fn, tap_fn, taps = io
    sel = ClanGamesSelector(reader=FauxReader(menu=False, entree=None), verbose=False)
    statut, _ = sel.engager_le_meilleur(screenshot_fn, tap_fn, caps)
    assert statut == 'menu_introuvable'
    assert taps == []


# =============================================================================
# Agent
# =============================================================================

def test_agent_ne_tourne_pas_hors_du_village():
    agent = ClanGamesAgent(selector=object(), verbose=False)
    assert agent.can_run({'on_village_home': False,
                          'buttons': {'raccourci_jdc': (494, 1022, 0.5)}}) is False


def test_agent_ne_tourne_pas_sans_entree_visible():
    # Pas de raccourci ni de tente = les jeux ne sont pas actifs.
    agent = ClanGamesAgent(selector=object(), verbose=False)
    assert agent.can_run({'on_village_home': True, 'buttons': {}}) is False


@pytest.mark.parametrize('classe', ['raccourci_jdc', 'jeux_de_clans'])
def test_agent_tourne_sur_lune_ou_lautre_des_entrees(classe):
    agent = ClanGamesAgent(selector=object(), verbose=False)
    assert agent.can_run({'on_village_home': True,
                          'buttons': {classe: (494, 1022, 0.5)}}) is True


def test_agent_arrete_definitivement_au_plafond():
    class _Sel:
        def engager_le_meilleur(self, *a, **kw):
            return ('plafond_atteint', {'score': (10000, 10000)})

    agent = ClanGamesAgent(selector=_Sel(), screenshot_fn=lambda: None,
                           tap_fn=lambda x, y: None, verbose=False)
    world = {'on_village_home': True, 'buttons': {'raccourci_jdc': (1, 1, 0.5)}}
    assert agent.can_run(world) is True
    assert agent.run().ok is True
    # Plus rien à gagner de la semaine : inutile de rouvrir le menu.
    assert agent.can_run(world) is False


def test_aucun_defi_sur_nest_pas_un_echec():
    # Le refus est le comportement voulu, pas une erreur à remonter.
    class _Sel:
        def engager_le_meilleur(self, *a, **kw):
            return ('aucun_defi_sur', {'lus': 8})

    agent = ClanGamesAgent(selector=_Sel(), screenshot_fn=lambda: None,
                           tap_fn=lambda x, y: None, verbose=False)
    res = agent.run()
    assert res.ok is True
    assert res.error is None
    assert res.data['statut'] == 'aucun_defi_sur'


def test_tous_les_agents_sont_exportes_par_le_package():
    """`brain/core.py` importe depuis `clashai.agents`, pas depuis les modules.

    Vécu : `ClanGamesAgent` écrit, testé (17 tests au vert) et branché dans
    `core.py`… mais absent de `agents/__init__.py`. Les tests importaient le
    MODULE directement, donc aucun ne passait par le chemin que le bot utilise
    réellement — ImportError au premier démarrage.

    Ce test parcourt les modules `*_agent.py` et vérifie que chaque classe
    d'agent est bien exportée par le package. Il vaut pour tous les agents à
    venir, pas seulement celui-ci.
    """
    import importlib
    import pkgutil

    import clashai.agents as pkg
    from clashai.agents.base import BaseAgent

    manquants = []
    for info in pkgutil.iter_modules(pkg.__path__):
        if not info.name.endswith('_agent'):
            continue
        module = importlib.import_module(f'clashai.agents.{info.name}')
        for nom, objet in vars(module).items():
            est_agent = (isinstance(objet, type)
                         and issubclass(objet, BaseAgent)
                         and objet is not BaseAgent
                         and objet.__module__ == module.__name__)
            if est_agent and getattr(pkg, nom, None) is not objet:
                manquants.append(f'{nom} ({info.name})')

    assert not manquants, (
        f"agents non exportés par clashai/agents/__init__.py : {manquants}")


def test_agent_confirmer_est_faux_par_defaut():
    vus = {}

    class _Sel:
        def engager_le_meilleur(self, s, t, caps, confirmer=False):
            vus['confirmer'] = confirmer
            return ('choisi_non_engage', {})

    ClanGamesAgent(selector=_Sel(), screenshot_fn=lambda: None,
                   tap_fn=lambda x, y: None, verbose=False).run()
    assert vus['confirmer'] is False


def test_le_besoin_de_troupe_remonte_dans_le_resultat(io):
    """L'agent ne compose pas l'armée — il signale ce qui la débloquerait.

    Sans ça, `aucun_defi_sur` jette l'information la plus utile qu'il possède :
    « entraîne un golem et 150 points deviennent atteignables ». Le cerveau en a
    besoin pour arbitrer (adapter la compo a un coût, un défi sans contrainte
    n'en a pas).
    """
    screenshot_fn, tap_fn, _taps = io
    reader = FauxReader(
        grille=[_defi(points=150, description=D_GOLEM)],
        popup={'ouvert': True, 'engage': False, 'commencer': (976, 510),
               'description': D_GOLEM},
    )
    sel = ClanGamesSelector(reader=reader, verbose=False)
    statut, details = sel.engager_le_meilleur(
        screenshot_fn, tap_fn, Capacites(unites_disponibles=set()), confirmer=True)

    assert statut == 'aucun_defi_sur'
    assert details['defis_bloques'] == [
        {'points': 150, 'unite': 'golem', 'terrain': 'multijoueur',
         'description': D_GOLEM},
    ]
