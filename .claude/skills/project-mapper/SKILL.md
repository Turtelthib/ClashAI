---
name: project-mapper
description: GPS du projet ClashAI. À utiliser en priorité pour comprendre l'architecture, trouver un fichier, ou mettre à jour la cartographie globale du code (PROJECT_MAP.md à la racine). Se déclenche sur "où est", "quel fichier fait quoi", "architecture du projet", "cartographie", "mets à jour la map".
---

# Project Mapper

Tu agis comme le GPS de ce projet. Pour éviter de consommer des tokens en utilisant des commandes de recherche (`ls`, `find`, `grep`) à l'aveugle, tu dois t'appuyer sur le fichier `PROJECT_MAP.md` situé à la racine.

## Règles du graphe de projet

1. **CONSULTATION :** Si on te pose une question sur l'architecture ou qu'on cherche un fichier, lis `PROJECT_MAP.md` en premier.
2. **GÉNÉRATION :** Si le fichier n'existe pas ou qu'on te demande de le régénérer, crée-le. Il doit contenir une arborescence claire (type commande `tree`) des dossiers.
3. **EXCLUSIONS STRICTES :** Le graphe ne doit jamais lister les dossiers ignorés (`node_modules`, `dist`, `__pycache__`, `.venv`, `.labelme_env`), les poids de modèles (`.pt`, `.pth`, `.onnx`) ni les images de test.
4. **ANNOTATIONS :** À côté de chaque fichier important dans le graphe, ajoute un commentaire court (1 ligne max) expliquant son rôle métier (ex: `├── train.py # Script d'entraînement principal du modèle`).
5. **MISE À JOUR AUTOMATIQUE :** À chaque fois que tu crées, renommes ou supprimes un fichier dans le cadre d'un refactoring, tu dois obligatoirement mettre à jour le `PROJECT_MAP.md` pour que le graphe reste juste.

## Vérifier que la map est à jour

```bash
# compte les .py réels vs ceux listés dans la map
find src -name "*.py" -not -path "*__pycache__*" | wc -l
grep -c "\.py" PROJECT_MAP.md
```
