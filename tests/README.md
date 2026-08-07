# tests/

Suite de non-régression **offline** : aucun émulateur, aucun ADB, aucun GPU, aucun poids de modèle.
Tout tourne sur des fakes, en quelques dixièmes de seconde.

```bash
uv run pytest              # tout
uv run pytest -v           # avec le détail par test
uv run pytest -k scheduler # un sous-ensemble
```

## Origine

Ces tests **reflètent** les blocs `if __name__ == "__main__"` déjà présents dans le code de
production (`agents/*.py`, `brain/interface.py`, `combat/action_space.py`). Ces blocs contenaient
déjà de vrais `assert` et de vrais fakes, mais rien ne les exécutait automatiquement.

**Les blocs d'origine n'ont pas été déplacés ni modifiés.** Ils restent utilisables en démo
manuelle (`uv run python -m clashai.agents.chat_agent`) et affichent un déroulé lisible.
Les tests ici en sont la version automatisée.

> Conséquence : si tu changes le comportement d'un agent, **deux** endroits sont à mettre à jour —
> le bloc `__main__` du module et le test correspondant. C'est le prix de ne pas toucher au code
> de production. Si un jour la duplication gêne, le bloc `__main__` est le candidat à supprimer,
> pas le test.

## Pourquoi `no_hardware()`

`conftest.no_hardware(agent)` remplace **uniquement** le `run()` d'un agent ; `priority`,
`cooldown_seconds` et `can_run()` restent ceux de la vraie classe, donc c'est bien le vrai
ordonnancement qui est testé.

Ce n'est pas de la prudence théorique. `CombatAgent.run()` et `GdCAgent.run()` lancent un vrai
épisode d'attaque et **bloquent** en attendant l'émulateur. En mutant `combat.priority` de 10 à 99
pour vérifier que la suite avait des dents, `scheduler.run(scheduler.pick(...))` a sélectionné
`CombatAgent` et pytest a tourné **2 minutes sans rendre la main** — au lieu d'échouer.

Avec le garde-fou, la même mutation produit 7 échecs d'assertion en 0,10 s.

> Règle : tout agent enregistré dans un test dont le `run()` peut toucher ADB ou le GPU passe par
> `no_hardware()`. Seuls `ClanCastleAgent` (manager falsifié) et `ChatAgent` (monitor falsifié)
> tournent avec leur vrai `run()`.

## Vérifier que la suite a des dents

```bash
sed -i '23s/priority = 10/priority = 99/' src/clashai/agents/combat_agent.py
uv run pytest                                   # doit donner 7 echecs
git checkout -- src/clashai/agents/combat_agent.py
```

## Pourquoi `testpaths`

`pyproject.toml` fixe `testpaths = ["tests"]`. Sans ça, un `pytest` lancé à la racine collecte
`src/tools/debug/test_deploy.py`, qui **n'a pas de garde `if __name__`** : l'import seul charge
tous les modèles YOLO/CNN sur GPU puis pilote l'émulateur. Ne pas retirer ce réglage tant que ces
scripts n'ont pas été renommés en `debug_*.py`.

## Bugs connus encodés en `xfail`

Un bug ouvert peut être verrouillé sans rendre la suite rouge :

```python
@pytest.mark.xfail(strict=True, reason="Bug ouvert (audit item 1.4) : ...")
def test_every_role_in_the_json_is_a_known_deploy_role():
    ...
```

`strict=True` compte double : tant que le bug est là, le test passe en `xfailed` ; le jour où
quelqu'un le corrige, pytest signale un `XPASS` **en échec** et force à retirer le marqueur.
L'invariant ne peut donc pas être oublié.

Actuellement un seul : `sorciere_ruine` a le rôle `clean` dans `configs/troops.json`, or `clean`
n'existe pas dans `DEPLOY_ROLES` — l'unité n'est jamais déployable.

## Ce qui n'est pas couvert

Reste à tester (voir `docs/AUDIT_2026-08-05.md` §2) : `perception/digit_reader.py`
(segmentation — demande de construire des crops synthétiques) et `combat/encoder/`
(shape/dtype du tenseur, invariants entre entraînement et inférence).
