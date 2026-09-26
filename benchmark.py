#!/usr/bin/env python3
"""Compare categorization strategies against the hand-labeled set in assets/gold.tsv.

Usage:
    python3 benchmark.py                               # all strategies
    python3 benchmark.py --strategies recursive beam --runs 3

Metrics per strategy:
    exact      share of products whose category is one of the accepted gold categories
    top-level  share with the right top-level category (e.g. "Electronics")
    path       partial credit: how much of the best-matching gold path was right, from the root.
               "Electronics > Audio" against gold "Electronics > Audio > Speakers" scores 2/3.
    cost, requests and seconds are per product; seconds is the latency of one product.
    conf ok/bad  mean confidence of right vs. wrong answers. A wide gap means the confidence can
               drive escalation (cascade) or human review; a narrow one means it cannot.

Jev is not fully deterministic, so --runs repeats everything and reports the mean.
Per-product predictions go to benchmark_results.csv for inspection.
"""

import argparse
import csv
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

from jev import ROOT, Node, Taxonomy, api_key, path_str
from strategies import STRATEGIES

GOLD_FILE = ROOT / "assets" / "gold.tsv"


def load_gold(taxonomy: Taxonomy) -> list[tuple[str, list[Node]]]:
    gold = []
    for line in GOLD_FILE.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        product, *paths = line.split("\t")
        gold.append((product, [taxonomy.node(p) for p in paths]))  # raises on a typo in a path
    return gold


def path_credit(predicted: Node, accepted: list[Node]) -> float:
    def shared(a: Node, b: Node) -> int:
        n = 0
        while n < min(len(a), len(b)) and a[n] == b[n]:
            n += 1
        return n

    return max(shared(predicted, g) / max(len(predicted), len(g)) for g in accepted)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark categorization strategies.")
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    parser.add_argument("--runs", type=int, default=1, help="repeat to average out Jev's variance")
    parser.add_argument("--workers", type=int, default=8, help="products in parallel")
    args = parser.parse_args()

    key = api_key()
    taxonomy = Taxonomy()
    gold = load_gold(taxonomy)
    print(f"{len(gold)} gold products, {args.runs} run(s) per strategy\n")

    summary, rows = [], []
    for name in args.strategies:
        strategy = STRATEGIES[name]
        runs = []
        for run in range(args.runs):
            def timed(item):
                started = time.time()
                result = strategy(item[0], taxonomy, key)
                return result, time.time() - started

            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                outcomes = list(pool.map(timed, gold))

            exact = top = credit = 0.0
            conf_ok, conf_bad = [], []
            for (product, accepted), (result, seconds) in zip(gold, outcomes):
                is_exact = result.node in accepted
                exact += is_exact
                (conf_ok if is_exact else conf_bad).append(result.confidence)
                top += any(result.node[:1] == g[:1] for g in accepted)
                credit += path_credit(result.node, accepted)
                rows.append({
                    "strategy": name,
                    "run": run + 1,
                    "product": product,
                    "exact": int(is_exact),
                    "predicted": path_str(result.node),
                    "accepted": " || ".join(path_str(g) for g in accepted),
                    "confidence": round(result.confidence, 2),
                    "seconds": round(seconds, 2),
                    "cost_usd": result.cost_usd,
                    "detail": result.detail,
                })
            n = len(gold)
            runs.append({
                "exact": exact / n,
                "top": top / n,
                "path": credit / n,
                "cost": sum(r.cost_usd for r, _ in outcomes) / n,
                "requests": sum(r.requests for r, _ in outcomes) / n,
                "seconds": statistics.median(s for _, s in outcomes),
                "conf_ok": statistics.mean(conf_ok) if conf_ok else float("nan"),
                "conf_bad": statistics.mean(conf_bad) if conf_bad else float("nan"),
            })
            print(f"  {name} run {run + 1}: exact {exact / n:.0%}")
        summary.append((name, {k: statistics.mean(r[k] for r in runs) for k in runs[0]}, runs))

    print(f"\n{'strategy':<12} {'exact':>7} {'top-level':>10} {'path':>7} {'$/product':>11} {'requests':>9} {'sec (median)':>13} {'conf ok/bad':>12}")
    for name, m, runs in summary:
        spread = f"  (runs: {', '.join(f'{r["exact"]:.0%}' for r in runs)})" if len(runs) > 1 else ""
        print(
            f"{name:<12} {m['exact']:>7.0%} {m['top']:>10.0%} {m['path']:>7.0%} "
            f"{m['cost']:>11.5f} {m['requests']:>9.1f} {m['seconds']:>13.2f} "
            f"{m['conf_ok']:>6.2f}/{m['conf_bad']:.2f}{spread}"
        )

    out = ROOT / "benchmark_results.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPer-product results: {out}")


if __name__ == "__main__":
    main()
