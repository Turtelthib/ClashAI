"""Lanceur de la calibration UI.

Ce fichier etait un FORK de `clashai/navigation/calibrate_ui.py` : ~400 lignes
quasi identiques, avec une divergence qui comptait. La version `tools/` avait un
groupe `cdc` que la version *package* n'avait pas -- or c'est la version package
que la production importe (`social/clan_castle.py` lit
`get_position('cdc_confirmation')`). Resultat : recalibrer avec
`python -m clashai.navigation.calibrate_ui` ne pouvait jamais recalibrer ce
bouton, et les deux ecrivaient pourtant le meme `configs/ui_positions.json`.

Le groupe `cdc` a ete porte dans le package, qui fait desormais autorite.

    uv run python src/tools/setup/calibrate_ui.py
    uv run python -m clashai.navigation.calibrate_ui   # equivalent
"""

from clashai.navigation.calibrate_ui import main

if __name__ == "__main__":
    main()
