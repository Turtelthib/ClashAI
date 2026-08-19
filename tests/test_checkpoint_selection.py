"""Selection du checkpoint RL au demarrage : ne JAMAIS croire un chargement rate.

Bug reel (19 aout 2026) : `PPOAgentV4.load()` ne leve PAS sur un mismatch de
dimensions — il log un WARNING et renvoie False, en laissant le reseau
fraichement initialise. Le brain ignorait ce retour et passait quand meme en
"Mode RL" -> le bot attaquait avec une politique ALEATOIRE au lieu de
l'heuristique prevue comme repli.

Ce cas est FREQUENT : obs et actions sont derivees du registre, donc ajouter un
role (`clean`) ou un sort (`colere`) change les dimensions et perime tous les
checkpoints.
"""

from clashai.brain.core import load_first_compatible_checkpoint


class _Agent:
    """load() renvoie ce qu'on lui dit, comme le vrai PPOAgentV4."""

    def __init__(self, results):
        self._results = list(results)
        self.loaded = []

    def load(self, path):
        self.loaded.append(path)
        return self._results.pop(0)


_ALL_EXIST = staticmethod(lambda p: True)


def test_incompatible_checkpoint_does_not_enable_rl():
    """LE bug : load() -> False doit laisser le mode heuristique."""
    agent = _Agent([False])
    assert load_first_compatible_checkpoint(
        agent, ['ckpt.pth'], exists=lambda p: True) is False


def test_compatible_checkpoint_enables_rl():
    agent = _Agent([True])
    assert load_first_compatible_checkpoint(
        agent, ['ckpt.pth'], exists=lambda p: True) is True


def test_falls_back_to_the_next_checkpoint():
    """`best` incompatible mais `checkpoint` bon -> on prend le second."""
    agent = _Agent([False, True])
    assert load_first_compatible_checkpoint(
        agent, ['best.pth', 'ckpt.pth'], exists=lambda p: True) is True
    assert agent.loaded == ['best.pth', 'ckpt.pth']


def test_missing_files_are_skipped_without_loading():
    agent = _Agent([])
    assert load_first_compatible_checkpoint(
        agent, ['absent.pth'], exists=lambda p: False) is False
    assert agent.loaded == []


def test_no_checkpoint_at_all_means_heuristic():
    assert load_first_compatible_checkpoint(_Agent([]), []) is False


def test_a_raising_load_is_caught_and_does_not_enable_rl():
    """Meme un load() qui leve ne doit pas faire croire au mode RL."""
    class _Boom:
        def load(self, path):
            raise RuntimeError('taille incompatible')
    assert load_first_compatible_checkpoint(
        _Boom(), ['ckpt.pth'], exists=lambda p: True) is False
