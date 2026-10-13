"""Selector: evaluate candidates and promote one only if it wins on held-out tasks without a per-task regression.

Promotion key: mean over held-out tasks first (generalization), train mean as tie-break. The regression gate is
noise-aware: with n runs per task one failed run moves a task's mean by 1/n, so a drop of one run is tolerated.
Held-out episodes are written to the store, but the reflector never reads them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .bundle import Bundle
from .schema import Task

if TYPE_CHECKING:
    from .trainable import Trainable

GATES = ("noise-aware", "fixed", "none")


@dataclass
class Eval:
    bundle: Bundle
    per_task: dict[str, float]
    train_mean: float
    held_mean: float | None
    cost: float = 0.0

    @property
    def key(self) -> tuple[float, float]:
        return (self.held_mean if self.held_mean is not None else self.train_mean, self.train_mean)


def evaluate(tr: "Trainable", bundle: Bundle, train: list[Task], held: list[Task], n: int) -> Eval:
    """Cached per Trainable, so the same bundle is never re-run on the same tasks within one optimization."""
    cache: dict = tr.__dict__.setdefault("_eval_cache", {})
    key = (bundle.version, tuple(t.id for t in train + held), n)
    if key in cache:
        c = cache[key]
        return Eval(bundle, c.per_task, c.train_mean, c.held_mean, 0.0)
    eps = tr.evaluate(train + held, n=n, bundle=bundle, workers=int(getattr(tr, "episode_workers", 1) or 1))
    per: dict[str, list[float]] = {}
    for e in eps:
        per.setdefault(e.task_id, []).append(e.score or 0.0)
    per_task = {k: sum(v) / len(v) for k, v in per.items()}
    tm = [per_task[t.id] for t in train if t.id in per_task]
    hm = [per_task[t.id] for t in held if t.id in per_task]
    ev = Eval(bundle, per_task, sum(tm) / len(tm) if tm else 0.0, sum(hm) / len(hm) if hm else None,
              sum(e.cost_usd or 0 for e in eps))
    cache[key] = ev
    return ev


def dominates(a: Eval, b: Eval) -> bool:
    ks = set(a.per_task) | set(b.per_task)
    return all(a.per_task.get(k, 0) >= b.per_task.get(k, 0) for k in ks) and any(a.per_task.get(k, 0) > b.per_task.get(k, 0) for k in ks)


@dataclass
class Selection:
    promoted: Eval | None
    incumbent: Eval
    evals: list[Eval]
    archive: list[Eval] = field(default_factory=list)  # the Pareto set over per-task scores, for the log
    reason: str = ""


def select(incumbent: Eval, evals: list[Eval], *, max_task_drop: float = 0.34, min_gain: float = 1e-9,
           n: int | None = None, gate: str = "noise-aware") -> Selection:
    """gate: "noise-aware" (the method), "fixed" (max_task_drop regardless of n) or "none" (no per-task regression
    gate: the best candidate is promoted whenever its key beats the incumbent's). The last two are ablations."""
    if gate not in GATES:
        raise ValueError(f"unknown gate {gate!r}; choose from {GATES}")
    if n and gate == "noise-aware":
        max_task_drop = max(max_task_drop, 1.0 / n + 1e-9)
    pool = [incumbent] + evals
    archive = [e for e in pool if not any(dominates(o, e) for o in pool if o is not e)]
    best = max(evals, key=lambda e: e.key, default=None)
    if best is None or best.key <= incumbent.key or (best.key[0] - incumbent.key[0] < min_gain and best.key[1] - incumbent.key[1] < min_gain):
        return Selection(None, incumbent, evals, archive, "no candidate beat the incumbent")
    drops = {k: incumbent.per_task[k] - best.per_task.get(k, 0) for k in incumbent.per_task}
    worst = max(drops.items(), key=lambda kv: kv[1], default=(None, 0))
    if gate != "none" and worst[1] > max_task_drop:
        return Selection(None, incumbent, evals, archive, f"regression gate: {worst[0]} dropped {worst[1]:.2f}")
    return Selection(best, incumbent, evals, archive, f"improved {incumbent.key} -> {best.key}")


def log_round(path: Path, rnd: int, sel: Selection, diagnoses, extra: dict | None = None) -> None:
    rec = {
        "round": rnd,
        "incumbent": {"version": sel.incumbent.bundle.version, "key": sel.incumbent.key, "per_task": sel.incumbent.per_task},
        "candidates": [{"version": e.bundle.version, "diagnosis": e.bundle.lock.get("diagnosis"),
                        "key": e.key, "per_task": e.per_task} for e in sel.evals],
        "promoted": sel.promoted.bundle.version if sel.promoted else None,
        "reason": sel.reason,
        "archive": [e.bundle.version for e in sel.archive],
        "diagnoses": [d.__dict__ for d in diagnoses],
        **(extra or {}),
    }
    with path.open("a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
