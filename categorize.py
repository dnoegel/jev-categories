#!/usr/bin/env python3
"""Categorize products into the Google Product Taxonomy with Jev.

Usage:
    python3 categorize.py                                   # assets/products.txt -> results.csv
    python3 categorize.py --strategy recursive --limit 20
    python3 categorize.py --product "Kong Classic Dog Chew Toy"

Strategies are in strategies.py; benchmark.py compares them against a hand-labeled set.

Results below --review-threshold get needs_review=1. On the benchmark, accepting only cascade
results with confidence >= 0.8 covered 79% of products at 98% precision.
"""

import argparse
import csv
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from jev import ROOT, Taxonomy, api_key, path_str
from strategies import STRATEGIES


def main() -> None:
    parser = argparse.ArgumentParser(description="Categorize products into the Google Product Taxonomy with Jev.")
    parser.add_argument("--strategy", choices=STRATEGIES, default="cascade")
    parser.add_argument("--product", help="categorize a single product name and print the result")
    parser.add_argument("--input", type=Path, default=ROOT / "assets" / "products.txt", help="one product per line")
    parser.add_argument("--output", type=Path, default=ROOT / "results.csv")
    parser.add_argument("--limit", type=int, help="only the first N products")
    parser.add_argument("--workers", type=int, default=16, help="products categorized in parallel")
    parser.add_argument("--review-threshold", type=float, default=0.8, help="flag results below this confidence")
    args = parser.parse_args()

    key = api_key()
    taxonomy = Taxonomy()
    strategy = STRATEGIES[args.strategy]

    if args.product:
        result = strategy(args.product, taxonomy, key)
        print(f"{path_str(result.node)}  (id {taxonomy.ids[result.node]})")
        print(f"confidence {result.confidence:.2f}, {result.requests} requests, cost ${result.cost_usd:.5f}")
        if result.detail:
            print(f"steps: {result.detail}")
        return

    products = [line.strip() for line in args.input.read_text().splitlines() if line.strip()]
    products = products[: args.limit] if args.limit else products

    started = time.time()
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, (product, result) in enumerate(
            zip(products, pool.map(lambda p: strategy(p, taxonomy, key), products)), 1
        ):
            needs_review = result.confidence < args.review_threshold
            flag = "REVIEW" if needs_review else "      "
            print(f"[{i}/{len(products)}] {flag} {result.confidence:.2f}  {product}  ->  {path_str(result.node)}")
            rows.append({
                "product": product,
                "category_id": taxonomy.ids[result.node],
                "category_path": path_str(result.node),
                "confidence": round(result.confidence, 2),
                "needs_review": int(needs_review),
                "detail": result.detail,
                "cost_usd": result.cost_usd,
            })

    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    total_cost = sum(r["cost_usd"] for r in rows)
    review = sum(r["needs_review"] for r in rows)
    print(f"\n{review} of {len(rows)} products need review (confidence < {args.review_threshold})")
    print(f"{len(rows)} products in {time.time() - started:.0f}s, cost ${total_cost:.4f}, written to {args.output}")


if __name__ == "__main__":
    main()
