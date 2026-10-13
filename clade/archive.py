"""Archive of every evaluated bundle version, used as a parent pool.

Darwin Gödel Machine: branching only from the best agent is much worse than keeping all
versions as possible parents. Huxley-Gödel Machine: pick parents by clade metaproductivity
(how good their descendants turn out), not by their own score.

Parent score = clade score + exploration bonus
  clade score  = mean held-out score of the node's descendants (own score if none yet)
  bonus        = c * sqrt(ln(N) / (1 + children))     (UCB-style; untried nodes get explored)
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class Node:
    version: str
    parent: str | None
    diagnosis: str
    per_task: dict[str, float]
    held_mean: float | None
    train_mean: float
    created: float = field(default_factory=time.time)
    children: list[str] = field(default_factory=list)
    tag: str = ""  # e.g. "incumbent", "candidate", "promoted"

    @property
    def score(self) -> float:
        return self.held_mean if self.held_mean is not None else self.train_mean


class Archive:
    def __init__(self, path: Path, *, exploration: float = 0.3):
        self.path = path
        self.exploration = exploration
        self.nodes: dict[str, Node] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    d = json.loads(line)
                    self.nodes[d["version"]] = Node(**d)

    def add(self, version: str, *, parent: str | None, diagnosis: str, per_task: dict[str, float],
            held_mean: float | None, train_mean: float, tag: str = "") -> Node:
        if version in self.nodes:
            n = self.nodes[version]
            n.per_task, n.held_mean, n.train_mean = per_task, held_mean, train_mean
            if tag:
                n.tag = tag
        else:
            n = Node(version, parent, diagnosis, per_task, held_mean, train_mean, tag=tag)
            self.nodes[version] = n
            if parent and parent in self.nodes and version not in self.nodes[parent].children:
                self.nodes[parent].children.append(version)
        self._save()
        return n

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("".join(json.dumps(asdict(n)) + "\n" for n in self.nodes.values()))

    def descendants(self, version: str) -> list[Node]:
        out, stack = [], list(self.nodes[version].children)
        while stack:
            v = stack.pop()
            if v in self.nodes:
                out.append(self.nodes[v]); stack.extend(self.nodes[v].children)
        return out

    def clade_score(self, version: str) -> float:
        desc = self.descendants(version)
        if not desc:
            return self.nodes[version].score
        return sum(d.score for d in desc) / len(desc)

    def parent_score(self, version: str) -> float:
        n_total = max(len(self.nodes), 2)
        n = self.nodes[version]
        return self.clade_score(version) + self.exploration * math.sqrt(math.log(n_total) / (1 + len(n.children)))

    def select_parent(self, *, exclude: set[str] = frozenset()) -> Node:
        cands = [n for v, n in self.nodes.items() if v not in exclude]
        return max(cands, key=lambda n: (self.parent_score(n.version), n.score, n.created))

    def best(self) -> Node:
        return max(self.nodes.values(), key=lambda n: (n.score, n.train_mean, -n.created))

    def summary(self) -> list[dict]:
        return [{"version": v, "parent": n.parent, "score": n.score, "clade": round(self.clade_score(v), 3),
                 "parent_score": round(self.parent_score(v), 3), "children": len(n.children), "tag": n.tag}
                for v, n in self.nodes.items()]
