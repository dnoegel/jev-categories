# jev-categories

Categorizes product names into the [Google Product Taxonomy](https://support.google.com/merchants/answer/6324436) with Jev. Provides and benchmark different strategies. Supports optional `need_review` decision if minimum threshold is not met. 

## Usage

```
python3 categorize.py
python3 categorize.py --strategy recursive --limit 20
python3 categorize.py --product "Kong Classic Dog Chew Toy"
python3 benchmark.py --runs 2
```

Needs `OPENROUTER=<key>` in `.env`. No dependencies beyond the Python standard library.

## Strategies (`strategies.py`)

| Strategy | Algorithm | How it presents the taxonomy | Requests per product |
|---|---|---|---|
| `recursive` | greedy descent | One level at a time: the children of the current node, each with its path and up to 30 subcategory names, plus `STAY` below the top level | 1 per level (about 3.6) |
| `beam` | beam search (width 3) | Like `recursive`, but keeps the 3 most probable paths per level, all asked in one request; a final question compares the ended paths | about 5.4 |
| `tournament` | brute force in batches | All 5595 categories as full paths, in 23 heats of 250 run in parallel; the top 3 of each heat go to a final | 24 (2 rounds) |
| `cascade` | cheap first, escalate | `beam`; if its confidence is below 0.6, `tournament` | about 8 on average |

## Benchmark results

73 hand-labeled products (`assets/gold.tsv`), 2 runs each:

| Strategy | Exact | Top level | Path credit | $/product | Median seconds | Confidence right/wrong |
|---|---|---|---|---|---|---|
| recursive | 72% | 82% | 76% | **0.00016** | **1.47** | 0.83 / 0.61 |
| beam | 91% | 95% | 94% | 0.00033 | 2.09 | 0.93 / 0.64 |
| tournament | **95%** | **99%** | **98%** | 0.00780 | 2.96 | 0.91 / 0.61 |
| cascade | 94% | 97% | 96% | 0.00113 | 2.38 | 0.94 / 0.68 |

Bold marks the best value per column. `cascade` leads no single column; it is the default because it comes within a point of `tournament`'s accuracy at a seventh of its cost.

- `recursive` fails mostly by taking a wrong branch early (mouse -> "Computer Accessories" instead of "Computer Components > Input Devices") with no way back.
- `beam` fixes most of that: the right path often ranks 2nd or 3rd during the descent (the mouse's correct path scored 0.14 against 0.78) and wins the final comparison.
- `tournament` shows every category as a full path, which is the most accurate but 24x the cost of `beam`.
- `cascade` gets close to the tournament's accuracy at a seventh of its cost, because beam's confidence separates right from wrong answers well (0.93 vs 0.64) and only about 13% of products are escalated. It is the default in `categorize.py`.
- Finals of `beam` and `tournament` use a dominance rule: a parent always "fits" too, so Jev sometimes picked "Storage Devices" (0.75) over its child "Hard Drives" (0.19). If a finalist below the winner gets at least 20% of the winner's probability, the more specific one wins. This added about 3 points to each (beam 88 -> 91%, tournament 92 -> 95%, cascade 92 -> 94%). Side effect: the combined confidence can keep a wrong descended answer from escalating (1 case in 146).
- The recursive row is from the run before the dominance rule, which does not affect it.

Cascade threshold, simulated from the saved beam and tournament results (before the dominance rule):

| Threshold | Escalated | Exact | $/product |
|---|---|---|---|
| 0 (beam only) | 0% | 88% | 0.00033 |
| 0.5 | 6% | 91% | 0.00081 |
| **0.6** | **13%** | **92%** | **0.00134** |
| 0.8 | 23% | 92% | 0.00209 |
| 1 (tournament only) | 100% | 92% | 0.00813 |

## Removed: prefilter

A `prefilter` strategy (BM25 text search picks 255 candidate categories, one Jev question decides) scored 73% exact at $0.00040 per product. `beam` is more accurate at a lower cost, so it was removed. The search was the bottleneck: the right category was among the candidates for only 54 of 73 products, since product and category names often share no word ("Samsung 990 PRO SSD" vs "Hard Drives"). With 73% exact against that 74% ceiling, Jev picked the right one almost every time it was shown.

## Review flag

`categorize.py` marks results with confidence below `--review-threshold` (default 0.8) as `needs_review`. On the benchmark:

| Auto-accept if confidence >= | Accepted automatically | Precision |
|---|---|---|
| 0 (accept everything) | 100% | 94% |
| **0.8** | **82%** | **98%** |
| 0.99 | 65% | 100% |

Metrics: *exact* means the category is one of the accepted gold categories; *top level* means the first path segment is right; *path credit* is the share of the best-matching gold path that was right from the root; *confidence right/wrong* is the mean confidence of correct vs. wrong answers. The gold labels were set by one person; ambiguous products list several accepted categories.

## Files

- `jev.py`: Jev API client (retries, request cap) and taxonomy loader.
- `strategies.py`: the strategies; add one by writing a function and registering it in `STRATEGIES`.
- `categorize.py`: CLI for categorizing a product list.
- `benchmark.py`: runs strategies against the gold set, writes per-product results to `benchmark_results.csv`.
- `assets/google_product_taxonomy.txt`: Google Product Taxonomy with ids, version 2021-09-21 (5595 categories, up to 7 levels, at most 79 children per node).
- `assets/products.txt`: 322 example product names across all top-level categories.
- `assets/gold.tsv`: 73 products with accepted categories, validated against the taxonomy on load.
