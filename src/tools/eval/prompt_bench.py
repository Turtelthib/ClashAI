# src/tools/eval/prompt_bench.py
# Banc d'essai des prompts du cerveau (V5.3, incrément 5.3.0).
#
# Mesure un TAUX sur plusieurs formulations par intention, dans les DEUX SENS
# (rappel : il doit donner la valeur / retenue : il doit refuser sans inventer).
# C'est le critère de recette des incréments suivants de la V5.3.
#
# Aucune perception, aucun émulateur : les `world` sont figés. Il faut seulement
# Ollama en marche.
#
# Usage :
#   uv run python -m tools.eval.prompt_bench                 # la suite complète
#   uv run python -m tools.eval.prompt_bench --liste         # sans rien lancer
#   uv run python -m tools.eval.prompt_bench -k elixir_noire # filtrer
#   uv run python -m tools.eval.prompt_bench --repeat 3      # 3 tirages/formulation
#   uv run python -m tools.eval.prompt_bench --model mistral-nemo
#   uv run python -m tools.eval.prompt_bench --out bench.json
#
# Durée : ~1 à 3 s par appel. La suite complète à --repeat 3 ≈ 84 appels, ~1 min.
#
# ⚠️ PIÈGE MESURÉ (19 août 2026) : à --repeat 1, deux exécutions du MÊME prompt
# donnent des scores différents (la discussion tourne à température 0.7). J'ai
# ainsi conclu à tort qu'une modification de prompt avait cassé un invariant —
# le même échec était présent dans la référence. Pour COMPARER, --repeat 3
# minimum, et ne conclure que sur un écart net.

import argparse
import json
import sys
import time


def main():
    ap = argparse.ArgumentParser(
        description="Banc d'essai des prompts du cerveau LLM")
    ap.add_argument('--model', default=None, help="Modèle Ollama (défaut: mistral)")
    ap.add_argument('-k', '--filtre', default=None,
                    help="Ne garde que les intentions contenant ce texte.")
    ap.add_argument('--repeat', type=int, default=3,
                    help="Tirages par formulation (défaut: 3). ⚠️ NE PAS "
                         "descendre à 1 pour COMPARER deux prompts : la "
                         "discussion tourne à température 0.7, et un run à 1 "
                         "tirage fabrique de fausses régressions (mesuré). "
                         "--repeat 1 ne vaut que pour un coup d'œil rapide.")
    ap.add_argument('--seuil', type=float, default=0.9,
                    help="Score global minimal ; en dessous, code de sortie 1 "
                         "(défaut: 0.9).")
    ap.add_argument('--liste', action='store_true',
                    help="Affiche la suite et ce que chaque cas protège, sans "
                         "appeler le modèle.")
    ap.add_argument('--out', default=None,
                    help="Écrit le rapport en JSON (pour comparer deux runs).")
    ap.add_argument('--verbeux', action='store_true',
                    help="Affiche aussi les réponses réussies.")
    args = ap.parse_args()

    from clashai.agents.scheduler import AgentScheduler
    from clashai.brain.llm_brain import DEFAULT_MODEL, LocalLLMBrain
    from clashai.brain.prompt_eval import SUITE, run_suite

    suite = SUITE
    if args.filtre:
        suite = [c for c in SUITE if args.filtre.lower() in c.intent.lower()]
        if not suite:
            print(f"Aucune intention ne contient « {args.filtre} ».")
            print("Disponibles :")
            for c in SUITE:
                print(f"  {c.intent}")
            return 2

    if args.liste:
        _print_suite(suite)
        return 0

    brain = LocalLLMBrain(AgentScheduler(), model=args.model or DEFAULT_MODEL,
                          verbose=False)

    print(f"Modèle : {brain._model}   ·   "
          f"{len(suite)} intentions   ·   "
          f"{sum(len(c.phrasings) for c in suite) * max(1, args.repeat)} appels")
    print("Préchauffage…", flush=True)
    if not brain.warmup():
        print("\nOllama ne répond pas. Vérifie qu'il tourne et que "
              "`ollama pull mistral` est terminé.")
        return 2

    def ask(question, world):
        # Mémoire vidée entre chaque formulation : on mesure le PROMPT, pas la
        # capacité du modèle à se souvenir de la réponse précédente. Sans ça, la
        # 2e question hérite de la 1re et le banc se ment à lui-même.
        brain.reset_chat()
        return brain.chat(question, world)

    print()
    t0 = time.time()
    report = run_suite(suite, ask, repeat=args.repeat, on_case=_print_case)
    elapsed = time.time() - t0

    _print_failures(report, verbose=args.verbeux)
    _print_summary(report, elapsed, args.seuil)

    if args.out:
        _write_json(report, args.out, brain._model, args.repeat)
        print(f"Rapport JSON -> {args.out}")

    return 0 if report.meets(args.seuil) else 1


# ---------------------------------------------------------------------------
# Affichage
# ---------------------------------------------------------------------------

def _bar(result):
    return ''.join('#' if r.ok else '.' for r in result.results)


def _print_case(result):
    mark = 'OK  ' if result.score == 1.0 else 'RATE'
    print(f"  [{mark}] {result.case.intent:<38} "
          f"{result.passed}/{result.total}  {_bar(result)}")


def _print_suite(suite):
    print(f"\n{len(suite)} intentions, "
          f"{sum(len(c.phrasings) for c in suite)} formulations\n")
    for c in suite:
        print(f"  {c.intent}   ({len(c.phrasings)} formulations)")
        if c.why:
            print(f"      {c.why}")
        for p in c.phrasings:
            print(f"      · {p}")
        print()


def _print_failures(report, verbose=False):
    failed = [c for c in report.cases if c.failures]
    if not failed and not verbose:
        return
    print()
    print("=" * 74)
    print("DÉTAIL")
    print("=" * 74)
    for cr in report.cases:
        shown = cr.results if verbose else cr.failures
        if not shown:
            continue
        print(f"\n{cr.case.intent}   {cr.passed}/{cr.total}")
        if cr.failures and cr.case.why:
            print(f"  ce que ce cas protège : {cr.case.why}")
        for r in shown:
            tag = 'ok  ' if r.ok else 'RATE'
            answer = ' '.join((r.answer or '[aucune réponse]').split())
            print(f"  [{tag}] « {r.phrasing} »")
            print(f"         -> {answer[:150]}")


def _print_summary(report, elapsed, seuil):
    print()
    print("=" * 74)
    pct = report.score * 100
    verdict = 'OK' if report.meets(seuil) else 'SOUS LE SEUIL'
    print(f"GLOBAL  {report.passed}/{report.total}  ({pct:.0f}%)   "
          f"seuil {seuil:.0%}   -> {verdict}")
    print(f"{elapsed:.0f}s")
    print("=" * 74)
    if not report.meets(seuil):
        print("\nUn échec ici n'est PAS forcément « le modèle est trop petit ».")
        print("Regarde d'abord le prompt : un nom technique, deux libellés qui")
        print("se ressemblent, ou une consigne de prudence sans déclencheur")
        print("objectif suffisent à produire ces ratés.")


def _write_json(report, path, model, repeat):
    import os
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    data = {
        'model': model,
        'repeat': repeat,
        'score': report.score,
        'passed': report.passed,
        'total': report.total,
        'cases': [
            {
                'intent': cr.case.intent,
                'passed': cr.passed,
                'total': cr.total,
                'score': cr.score,
                'results': [
                    {'phrasing': r.phrasing, 'ok': r.ok, 'answer': r.answer}
                    for r in cr.results
                ],
            }
            for cr in report.cases
        ],
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    sys.exit(main())
