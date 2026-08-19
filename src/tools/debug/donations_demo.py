# src/tools/debug/donations_demo.py
# Démo de test pour les dons de troupes au clan (V5.2, increment 4).
#
# SÛR PAR CONSTRUCTION : donner ne coûte RIEN (on donne ses propres troupes).
# Le seul vrai risque serait l'onglet `dons_gemme` (don payé en gemmes) — le
# code tape explicitement `dons_normaux` et abandonne s'il ne le trouve pas.
#
# Usage (émulateur branché) :
#   # 1) LECTURE SEULE — ouvre TOI-MÊME le chat clan, puis :
#   uv run python -m tools.debug.donations_demo --scan
#
#   # 2) Flux complet depuis le village (ouvre le chat tout seul)
#   uv run python -m tools.debug.donations_demo
#   uv run python -m tools.debug.donations_demo --max-requests 1

import argparse


def main():
    ap = argparse.ArgumentParser(description="Démo dons de troupes au clan")
    ap.add_argument('--scan', action='store_true',
                    help="N'agit PAS : liste ce que le CNN voit sur l'écran "
                         "courant (à lancer sur le chat de clan).")
    ap.add_argument('--max-requests', type=int, default=3,
                    help="Nombre max de demandes traitées (défaut 3).")
    ap.add_argument('--debug-dir', default='debug_dons',
                    help="Dossier des captures. Vide pour désactiver.")
    args = ap.parse_args()

    from clashai.navigation import game_loop as gl
    from clashai.perception.ui_detector import install
    from clashai.social.donations import DonationManager

    install(verbose=True)
    models = gl.load_models()
    mgr = DonationManager(verbose=True, debug_dir=args.debug_dir or None)

    # --- mode scan : diagnostic pur, aucun tap ------------------------------
    if args.scan:
        img = gl.adb_screenshot()
        if img is None:
            print("ERREUR: pas de capture (émulateur branché ?)")
            return
        buttons = mgr.find_donate_buttons(img)
        print(f"\nBoutons 'donner' ACTIFS : {len(buttons)}")
        for x, y, c in buttons:
            print(f"   ({x}, {y})  conf={c:.2f}")
        print("\n(les demandes déjà satisfaites ont un bouton grisé → écartées)")

        prev = 0
        for x, y, _c in buttons:
            w = mgr.read_request(img, models, button_y=y, block_top=prev)
            reclame = ", ".join(sorted(w)) if w else "(rien de lisible)"
            print(f"   demande @({x}, {y}) réclame : {reclame}")
            prev = y

        cards = mgr.list_donatable(img, models)
        print(f"\nVignettes de troupes non grisées visibles : {len(cards)}")
        for name, x, y in cards[:15]:
            print(f"   {name:22s} ({x}, {y})")
        print("\n⚠️ Sur l'écran du CHAT (pop-up fermé), ces vignettes sont celles")
        print("   des demandes — c'est normal. Le filtre `min_x` les écarte une")
        print("   fois le pop-up ouvert.")
        if args.debug_dir:
            mgr._dump(img, 'dons_scan')
            print(f"\nCaptures dans : {args.debug_dir}/ "
                  "(dons_scan.png annotée + dons_scan_raw.png brute)")
        return

    # --- flux complet -------------------------------------------------------
    from clashai.social.clan_chat_monitor import ClanChatMonitor
    monitor = ClanChatMonitor(verbose=True)
    if not monitor.open_chat(gl.classify_screen, models):
        print("ERREUR: impossible d'ouvrir le chat de clan "
              "(es-tu bien sur l'écran du village ?)")
        return

    img = gl.adb_screenshot()
    mgr._dump(img, 'dons_1_chat')
    buttons = mgr.find_donate_buttons(img) if img is not None else []
    print(f"\n{len(buttons)} demande(s) à satisfaire")
    if not buttons:
        print("(aucun bouton 'donner' actif — rien à donner pour l'instant)")
        monitor.close_chat()
        return

    total = 0
    prev_y = 0
    for i, (x, y, _c) in enumerate(buttons[:args.max_requests], 1):
        print(f"\n--- demande {i}/{min(len(buttons), args.max_requests)} "
              f"@({x}, {y}) ---")
        # Info seulement : le jeu FORCE deja les troupes quand la demande
        # les verrouille (les autres sont grisees) -> on ne filtre PAS dessus.
        wanted = mgr.read_request(img, models, button_y=y, block_top=prev_y)
        demande = ", ".join(sorted(wanted)) if wanted else "(rien de verrouille)"
        print(f"    demande verrouillee : {demande}")
        status, given = mgr.donate_to_request(
            (x, y), gl.adb_screenshot, gl.adb_tap, models)
        print(f"    statut={status}  dons={given}")
        total += given
        prev_y = y

    monitor.close_chat()
    print(f"\nTotal : {total} don(s)")
    if args.debug_dir:
        print(f"Captures dans : {args.debug_dir}/")


if __name__ == "__main__":
    main()
