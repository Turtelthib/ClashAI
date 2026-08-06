---
name: uv-workflow
description: Gestion de l'environnement Python avec uv (Astral). À utiliser pour toute tâche impliquant l'installation de dépendances, la gestion du virtualenv, ou l'exécution de scripts Python dans ce projet. Se déclenche sur "installe", "ajoute une dépendance", "lance le script", "pip", "venv", "requirements".
---

# uv Workflow

Tu es un expert Python travaillant sur un projet moderne qui utilise exclusivement `uv` (par Astral) comme gestionnaire de paquets et d'environnement.

## Règles strictes pour ce projet

1. **INTERDICTION D'UTILISER PIP :** Ne suggère jamais `pip install`, `pip freeze` ou `virtualenv`.
2. **AJOUT DE DÉPENDANCES :** Pour ajouter un paquet, utilise uniquement `uv add <package>`. Pour les dépendances de développement, utilise `uv add --dev <package>`.
3. **EXÉCUTION DE SCRIPTS :** Pour exécuter du code ou un script, utilise toujours `uv run python <script.py>` ou `uv run <commande>` pour t'assurer qu'il tourne dans le bon environnement.
4. **SYNCHRONISATION :** Si tu modifies manuellement le fichier `pyproject.toml`, rappelle-toi d'utiliser `uv sync` ensuite.
5. Ne propose pas de créer un `.venv` avec le module `venv` natif, `uv` le gère automatiquement.

## Spécificités de cet environnement

- **Deux venvs coexistent** à la racine : `.venv/` (le bon) et `.labelme_env/` (héritage labelme). La variable `VIRTUAL_ENV` pointe parfois vers `.labelme_env`, ce qui déclenche l'avertissement `does not match the project environment path` — inoffensif, `uv` utilise bien `.venv`. Utilise `--active` seulement si tu veux explicitement l'autre.
- **`ruff` n'est pas installé dans `.venv`** : passe par `uv run --with ruff ruff check src/`.
- **torch/torchvision viennent d'un index dédié** (`pytorch-cu128`, cf. `[tool.uv.sources]`). Ne les réinstalle jamais depuis PyPI standard.
- Toute dépendance importée en **top-level** dans `src/clashai/` doit être déclarée dans `pyproject.toml`. Une dépendance seulement transitive peut disparaître au prochain `uv lock`.

## Vérifier qu'un import n'est pas une dépendance fantôme

```bash
uv run python -c "import <paquet>; print('<paquet>', <paquet>.__version__)"
grep -n "<paquet>" pyproject.toml || echo "NON DECLARE dans pyproject.toml"
```
