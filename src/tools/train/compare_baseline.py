# tools/train/compare_baseline.py
# Compare le run RL courant à un baseline gelé (stats côte à côte + delta).
# Sert à comparer DIRECTEMENT deux runs sans re-déduire à la main.
#
# Run :
#   uv run python src/tools/train/compare_baseline.py
#   uv run python src/tools/train/compare_baseline.py --log weights/rl/training_log_v4.json \
#       --baseline weights/baselines/v4.4-ppo-350ep/stats.json

import argparse
import json
import os

from clashai.paths import RL_WEIGHTS_DIR, WEIGHTS_DIR


def _num(e, *keys):
    for k in keys:
        if k in e and isinstance(e[k], (int, float)):
            return e[k]
    return 0


def compute_stats(log, name="courant"):
    """Mêmes métriques que le snapshot baseline (stats.json)."""
    n = len(log)
    if n == 0:
        return None
    R = [_num(e, 'reward') for e in log]
    S = [_num(e, 'stars') for e in log]
    P = [_num(e, 'percentage') for e in log]
    rate = lambda v: round(100 * sum(1 for s in S if s == v) / n, 1)
    return {
        "name": name,
        "episodes": n,
        "reward_mean": round(sum(R) / n, 1),
        "reward_max": round(max(R), 1),
        "stars_mean": round(sum(S) / n, 3),
        "pct_2plus": round(100 * sum(1 for s in S if s >= 2) / n, 1),
        "pct_3": rate(3),
        "pct_0": rate(0),
        "destruction_pct_mean": round(sum(P) / n, 1),
    }


def _flatten_baseline(b):
    """Adapte le stats.json gelé (structure imbriquée) au format plat."""
    return {
        "name": b.get("name", "baseline"),
        "episodes": b.get("episodes", 0),
        "reward_mean": b.get("reward", {}).get("mean", 0),
        "reward_max": b.get("reward", {}).get("max", 0),
        "stars_mean": b.get("stars", {}).get("mean", 0),
        "pct_2plus": b.get("stars", {}).get("pct_2plus", 0),
        "pct_3": b.get("stars", {}).get("pct_3", 0),
        "pct_0": b.get("stars", {}).get("pct_0", 0),
        "destruction_pct_mean": b.get("destruction_pct_mean", 0),
    }


ROWS = [
    ("Episodes", "episodes", 0, False),
    ("Reward moyen", "reward_mean", 1, True),
    ("Reward max", "reward_max", 1, True),
    ("Etoiles moy", "stars_mean", 3, True),
    ("% 2* et +", "pct_2plus", 1, True),
    ("% 3*", "pct_3", 1, True),
    ("% 0* (rates)", "pct_0", 1, False),   # plus bas = mieux
    ("% destruction", "destruction_pct_mean", 1, True),
]


def main():
    ap = argparse.ArgumentParser(description="Compare le run RL courant à un baseline gelé.")
    ap.add_argument('--log', default=os.path.join(RL_WEIGHTS_DIR, 'training_log_v4.json'))
    ap.add_argument('--baseline', default=os.path.join(
        WEIGHTS_DIR, 'baselines', 'v4.4-ppo-350ep', 'stats.json'))
    args = ap.parse_args()

    if not os.path.exists(args.baseline):
        print(f"Baseline introuvable : {args.baseline}")
        return
    if not os.path.exists(args.log):
        print(f"Log courant introuvable : {args.log}")
        return

    base = _flatten_baseline(json.load(open(args.baseline, encoding='utf-8')))
    cur = compute_stats(json.load(open(args.log, encoding='utf-8')))

    print(f"\n  {'':16s} {base['name']:>16s} {cur['name']:>12s} {'delta':>10s}")
    print(f"  {'-'*56}")
    for label, key, dec, higher_better in ROWS:
        b, c = base.get(key, 0), cur.get(key, 0)
        d = c - b
        arrow = ""
        if key != "episodes" and abs(d) > 1e-9:
            good = (d > 0) if higher_better else (d < 0)
            arrow = "  [+]" if good else "  [-]"
        fmt = f"{{:.{dec}f}}"
        print(f"  {label:16s} {fmt.format(b):>16s} {fmt.format(c):>12s} "
              f"{('+' if d >= 0 else '')}{fmt.format(d):>9s}{arrow}")
    print(f"\n  Baseline : {args.baseline}")
    print(f"  Courant  : {args.log}\n")


if __name__ == '__main__':
    main()
