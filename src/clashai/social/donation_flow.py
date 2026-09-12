# clashai/social/donation_flow.py
# Parcours complet des dons : ouvrir le chat -> servir les demandes -> refermer.
#
# Extrait de `tools/debug/donations_demo.py` (validé en réel en V5.2) pour que le
# cerveau puisse l'appeler comme un outil (V5.3, étape 5.3.3a). Le geste est le
# même ; seules les dépendances sont injectées, pour être testable sans ADB.
#
# La sécurité anti-gemmes reste là où elle était : `DonationManager.
# donate_to_request` sélectionne explicitement l'onglet GRATUIT et abandonne s'il
# ne le trouve pas. Ce module n'y touche pas.
#
# ⚠️ Limites connues, héritées de la démo :
#   - on sert les demandes VISIBLES, de haut en bas, jusqu'à `max_requests` —
#     sans savoir QUI demande (il faut l'OCR par message, V5.4) ;
#   - les positions des boutons viennent de la première capture ; si le chat
#     défile entre deux demandes, les suivantes peuvent être décalées.

DEFAULT_MAX_REQUESTS = 5


def donate_visible_requests(manager, monitor, screenshot_fn, tap_fn, classify_fn,
                            models, wanted=None, max_requests=DEFAULT_MAX_REQUESTS):
    """Répond aux demandes de dons visibles. Rend un résumé (dict).

    Args:
        manager: DonationManager (trouve les boutons, donne à une demande).
        monitor: objet avec `open_chat(classify_fn, models) -> bool` et
                 `close_chat()` (ClanChatMonitor en production).
        wanted: ensemble de noms de troupes à donner EXCLUSIVEMENT, ou None
                pour laisser la politique par défaut choisir.

    Lève RuntimeError si le chat ne s'ouvre pas. Le chat est TOUJOURS refermé
    une fois ouvert, même si un don plante en cours de route — sinon le bot
    reprendrait sa boucle avec le chat par-dessus le village.
    """
    if not monitor.open_chat(classify_fn, models):
        raise RuntimeError("impossible d'ouvrir le chat de clan "
                           "(le bot est-il bien sur le village ?)")
    try:
        img = screenshot_fn()
        buttons = manager.find_donate_buttons(img) if img is not None else []
        statuses = []
        total = 0
        for x, y, _conf in buttons[:max_requests]:
            status, given = manager.donate_to_request(
                (x, y), screenshot_fn, tap_fn, models, wanted=wanted)
            statuses.append(status)
            total += given
        return {
            'demandes_vues': len(buttons),
            'demandes_servies': sum(1 for s in statuses if s == 'ok'),
            'dons': total,
            'statuts': statuses,
            'troupes_voulues': sorted(wanted) if wanted else None,
        }
    finally:
        monitor.close_chat()
