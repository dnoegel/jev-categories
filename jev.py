"""Jev API client and the Google Product Taxonomy, shared by the strategies."""

import json
import os
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
TAXONOMY_FILE = ROOT / "assets" / "google_product_taxonomy.txt"
API_URL = "https://openrouter.ai/api/alpha/decisions"
MODEL = "typesafe/jev-1.13"
MAX_OPTIONS = 255  # Jev's limit per choice question

# Strategies fan out requests per product and products run in parallel too; cap the total.
_in_flight = threading.BoundedSemaphore(32)

Node = tuple[str, ...]  # a category by its path, e.g. ("Electronics", "Audio")


def load_env(path: Path = ROOT / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def api_key() -> str:
    load_env()
    key = os.environ.get("OPENROUTER")
    if not key:
        raise SystemExit("Missing OPENROUTER API key in .env")
    return key


def ask(state: dict, questions: dict, key: str) -> tuple[dict, float]:
    """Send questions about a state to Jev. Returns (answers by question id, cost_usd)."""
    body = {"model": MODEL, "state": state, "questions": questions}
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    for attempt in range(3):
        try:
            with _in_flight, urllib.request.urlopen(req, timeout=90) as resp:
                data = json.load(resp)
            return data["answers"], data.get("usage", {}).get("cost", 0.0)
        except urllib.error.HTTPError as e:
            # The alpha endpoint occasionally returns transient 5xx errors (and 429 under load).
            if (e.code < 500 and e.code != 429) or attempt == 2:
                raise RuntimeError(f"API error {e.code}: {e.read().decode(errors='replace')}") from e
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def choice(product: str, instructions: str, options: dict[str, str], key: str) -> tuple[dict[str, float], float]:
    """One choice question about a product. Returns (probabilities by option key, cost_usd)."""
    assert len(options) <= MAX_OPTIONS, len(options)
    question = {"type": "choice", "instructions": instructions, "criteria": options}
    answers, cost = ask({"product_name": product}, {"category": question}, key)
    return answers["category"]["probabilities"], cost


class Taxonomy:
    def __init__(self, path: Path = TAXONOMY_FILE):
        self.ids: dict[Node, str] = {}
        self.children: dict[Node, list[Node]] = defaultdict(list)
        for line in path.read_text().splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            category_id, full_path = line.split(" - ", 1)
            node = tuple(full_path.split(" > "))
            self.ids[node] = category_id
            self.children[node[:-1]].append(node)
        self.by_id = {category_id: node for node, category_id in self.ids.items()}
        self.nodes = list(self.ids)  # file order: every parent comes right before its subtree

    def node(self, path: str) -> Node:
        node = tuple(path.split(" > "))
        if node not in self.ids:
            raise KeyError(f"not in taxonomy: {path}")
        return node


def path_str(node: Node) -> str:
    return " > ".join(node)
