"""Strategies for putting a product into a taxonomy category with Jev.

Each strategy is a function (product, taxonomy, key) -> Result and is registered in STRATEGIES.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from jev import Node, Taxonomy, ask, choice, path_str

STAY = "STAY"


@dataclass
class Result:
    node: Node
    confidence: float  # strategy-specific; low means "worth a human look"
    cost_usd: float
    requests: int
    detail: str = ""


# --- recursive: walk down the tree one level per request -------------------------------------

# Subcategory names shown per option, so Jev knows what a branch contains. With 8, Jev often took a
# wrong branch (SSD -> "Computers", ink cartridge -> "Office Supplies") and then had to STAY there;
# with 30 most of those landed correctly.
HINTS_PER_OPTION = 30


def describe(taxonomy: Taxonomy, node: Node) -> str:
    text = path_str(node)
    hints = [child[-1] for child in taxonomy.children.get(node, [])[:HINTS_PER_OPTION]]
    if hints:
        text += f" (includes e.g. {', '.join(hints)})"
    return text


def level_question(taxonomy: Taxonomy, node: Node) -> dict:
    """Choice among the children of node; below the top level, STAY ends on node itself."""
    # Option keys are the child names; siblings never share a name.
    options = {child[-1]: describe(taxonomy, child) for child in taxonomy.children[node]}
    instructions = "Which product category does product_name belong to?"
    if node:
        instructions = f"product_name belongs to '{path_str(node)}'. Which subcategory fits it best?"
        options[STAY] = f"None of the subcategories fits; the product belongs directly in {path_str(node)}."
    return {"type": "choice", "instructions": instructions, "criteria": options}


def recursive(product: str, taxonomy: Taxonomy, key: str) -> Result:
    """Greedy descent: one choice question per level, always following the most probable child."""
    node: Node = ()
    steps, cost = [], 0.0
    while taxonomy.children.get(node):
        answers, step_cost = ask({"product_name": product}, {"level": level_question(taxonomy, node)}, key)
        probabilities = answers["level"]["probabilities"]
        cost += step_cost
        pick = max(probabilities, key=probabilities.get)
        steps.append((pick, probabilities[pick]))
        if pick == STAY:
            break
        node = node + (pick,)
    return Result(
        node=node,
        # The weakest step is the best single signal for "please review this one".
        confidence=min(p for _, p in steps),
        cost_usd=cost,
        requests=len(steps),
        detail=" | ".join(f"{c} {p:.2f}" for c, p in steps),
    )


# --- beam: like recursive, but keep the best BEAM_WIDTH paths instead of committing to one ------

BEAM_WIDTH = 3


def beam(product: str, taxonomy: Taxonomy, key: str) -> Result:
    """Beam search down the tree. A path's score is the product of its step probabilities.

    All open paths of a level are asked in one request (Jev answers independent questions in
    parallel), so it takes about as many requests as recursive, with BEAM_WIDTH times the tokens.
    Paths end at a leaf or when STAY wins a share; a final flat question compares the full paths,
    because scores of paths with different depths are not directly comparable.
    """
    open_paths: list[tuple[Node, float]] = [((), 1.0)]
    ended: dict[Node, float] = {}
    cost, requests = 0.0, 0
    while open_paths:
        questions = {f"q{i}": level_question(taxonomy, node) for i, (node, _) in enumerate(open_paths)}
        answers, step_cost = ask({"product_name": product}, questions, key)
        cost += step_cost
        requests += 1

        expanded: list[tuple[Node, float]] = []
        for i, (node, score) in enumerate(open_paths):
            for option, p in answers[f"q{i}"]["probabilities"].items():
                if option == STAY:
                    ended[node] = max(ended.get(node, 0.0), score * p)
                else:
                    expanded.append((node + (option,), score * p))
        expanded.sort(key=lambda item: item[1], reverse=True)

        open_paths = []
        for node, score in expanded[:BEAM_WIDTH]:
            if taxonomy.children.get(node):
                open_paths.append((node, score))
            else:
                ended[node] = max(ended.get(node, 0.0), score)

    # Final: the best-scoring ended paths as full paths, like the tournament's final.
    finalists = sorted(ended, key=ended.get, reverse=True)[: BEAM_WIDTH * 3]
    options = {taxonomy.ids[n]: path_str(n) for n in finalists}
    probabilities, final_cost = choice(product, FLAT_INSTRUCTIONS, options, key)
    node, confidence = pick_final(probabilities, taxonomy)
    return Result(
        node=node,
        confidence=confidence,
        cost_usd=cost + final_cost,
        requests=requests + 1,
        detail=" | ".join(f"{path_str(n)} {ended[n]:.2f}" for n in finalists),
    )


# --- tournament: every category as a full path, heats in parallel, then a final --------------

# A parent category always "fits" too, so in a final Jev sometimes prefers "Storage Devices" (0.75)
# over its child "Hard Drives" (0.19). If a finalist below the winner gets at least this share of
# the winner's probability, the more specific one wins. Clear cases give ~1.00 to one option.
DESCEND_RATIO = 0.2


def pick_final(probabilities: dict[str, float], taxonomy: Taxonomy) -> tuple[Node, float]:
    """Winner of a final keyed by category id, preferring a plausible more specific descendant."""
    winner = max(probabilities, key=probabilities.get)
    node, p = taxonomy.by_id[winner], probabilities[winner]
    descendants = [
        (taxonomy.by_id[i], q) for i, q in probabilities.items()
        if taxonomy.by_id[i][: len(node)] == node and len(taxonomy.by_id[i]) > len(node) and q >= DESCEND_RATIO * p
    ]
    if descendants:
        # The deepest plausible one; confidence is the mass on it and the parent it refines.
        deepest, q = max(descendants, key=lambda item: (len(item[0]), item[1]))
        return deepest, p + q
    return node, p


HEAT_SIZE = 250
FINALISTS_PER_HEAT = 3
FLAT_INSTRUCTIONS = "Which product category fits product_name best? Prefer the most specific category that fits."


def tournament(product: str, taxonomy: Taxonomy, key: str) -> Result:
    """All 5595 categories as full paths. No early step can lead it into a wrong branch."""
    nodes = taxonomy.nodes  # file order, so a heat holds neighbouring (similar) categories
    heats = [nodes[i : i + HEAT_SIZE] for i in range(0, len(nodes), HEAT_SIZE)]

    def run_heat(heat: list[Node]) -> tuple[list[str], float]:
        probabilities, cost = choice(product, FLAT_INSTRUCTIONS, {taxonomy.ids[n]: path_str(n) for n in heat}, key)
        return sorted(probabilities, key=probabilities.get, reverse=True)[:FINALISTS_PER_HEAT], cost

    with ThreadPoolExecutor(max_workers=len(heats)) as pool:
        results = list(pool.map(run_heat, heats))
    finalists = [category_id for ids, _ in results for category_id in ids]

    options = {category_id: path_str(taxonomy.by_id[category_id]) for category_id in finalists}
    probabilities, final_cost = choice(product, FLAT_INSTRUCTIONS, options, key)
    node, confidence = pick_final(probabilities, taxonomy)
    return Result(
        node=node,
        confidence=confidence,
        cost_usd=sum(c for _, c in results) + final_cost,
        requests=len(heats) + 1,
    )


# --- cascade: cheap strategy first, expensive one only when it is unsure ---------------------

# Beam confidence below this escalates to the tournament. Simulated on the benchmark: 0.5 escalates
# 6% of products (91% exact), 0.6 escalates 13% (92%, the tournament's level), 0.8 escalates 23%
# with no further gain.
CASCADE_THRESHOLD = 0.6


def cascade(product: str, taxonomy: Taxonomy, key: str) -> Result:
    first = beam(product, taxonomy, key)
    if first.confidence >= CASCADE_THRESHOLD:
        return first
    second = tournament(product, taxonomy, key)
    return Result(
        node=second.node,
        confidence=second.confidence,
        cost_usd=first.cost_usd + second.cost_usd,
        requests=first.requests + second.requests,
        detail=f"escalated (beam {first.confidence:.2f}: {path_str(first.node)})",
    )


STRATEGIES = {
    "recursive": recursive,
    "beam": beam,
    "tournament": tournament,
    "cascade": cascade,
}
