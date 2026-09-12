"""Diagnostic d'echec de navigation (V5.3).

Vecu le 19 aout 2026 : le bot annonce « Unable to return to village » — un
probleme de NAVIGATION — alors qu'il n'avait recu AUCUNE image (emulateur
minimise + `adb devices` vide). On a cherche le defaut dans le code de
navigation, qui etait parfaitement sain.

Un message d'erreur qui designe le mauvais coupable coute plus cher qu'une
absence de message : il envoie chercher au mauvais endroit.
"""

from clashai.brain.navigation import SCREEN_REGION_BACKENDS, navigation_diagnosis


def test_no_capture_at_all_is_not_called_a_navigation_problem():
    msg = navigation_diagnosis(no_capture=15, attempts=15, states=[])
    assert "aucune capture" in msg
    assert "PAS un problème de navigation" in msg


def test_no_capture_message_names_both_possible_causes():
    """Emulateur minimise OU ADB deconnecte : l'utilisateur doit savoir quoi
    verifier, sans relire le code."""
    msg = navigation_diagnosis(no_capture=15, attempts=15, states=[])
    assert 'MINIMISÉ' in msg
    assert 'adb devices' in msg


def test_a_screen_region_backend_stuck_on_loading_is_a_desktop_capture():
    """La signature exacte du bug : mss + « chargement » en boucle. Le CNN voit
    une fenetre d'IDE sombre et la classe « chargement »."""
    msg = navigation_diagnosis(no_capture=0, attempts=15,
                               states=['chargement'] * 15, backend='mss')
    assert 'BUREAU' in msg and 'minimisé' in msg
    assert 'TROUBLESHOOTING' in msg


def test_the_desktop_diagnosis_applies_to_every_screen_region_backend():
    for backend in SCREEN_REGION_BACKENDS:
        msg = navigation_diagnosis(0, 15, ['chargement'] * 15, backend=backend)
        assert 'BUREAU' in msg, backend


def test_wgc_stuck_on_loading_is_not_blamed_on_the_desktop():
    """WGC capture la FENETRE : « chargement » en boucle y signifie vraiment que
    le jeu charge. Accuser le bureau enverrait chercher au mauvais endroit."""
    msg = navigation_diagnosis(0, 15, ['chargement'] * 15, backend='wgc')
    assert 'BUREAU' not in msg
    assert 'chargement' in msg


def test_a_real_navigation_failure_lists_the_screens_seen():
    """Quand la capture marche, le message doit dire OU on etait bloque."""
    msg = navigation_diagnosis(0, 15, ['profil'] * 8 + ['menu_boutique'] * 7)
    assert 'profil' in msg and 'menu_boutique' in msg
    assert 'aucune capture' not in msg


def test_screens_are_deduplicated_and_sorted():
    msg = navigation_diagnosis(0, 3, ['profil', 'profil', 'chat_clan'])
    assert msg.count('profil') == 1
    assert msg.index('chat_clan') < msg.index('profil')


def test_a_partial_capture_loss_is_not_reported_as_total():
    """Quelques images perdues n'expliquent pas l'echec : ne pas envoyer
    l'utilisateur verifier son emulateur pour rien."""
    msg = navigation_diagnosis(no_capture=3, attempts=15,
                               states=['profil'] * 12)
    assert 'aucune capture' not in msg
    assert 'profil' in msg


def test_no_state_and_no_backend_still_gives_a_readable_message():
    msg = navigation_diagnosis(no_capture=0, attempts=15, states=[])
    assert 'aucun' in msg and msg.strip()
