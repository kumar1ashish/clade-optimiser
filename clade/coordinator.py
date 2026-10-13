"""Coordinator: decides, each round, how many worker lineages to run, which archive parent each branches
from, and whether the budget allows another round. It is a policy, not a model: cheap, deterministic, and
auditable. It never touches the judges or the selector's gate.

  lane 0  exploit   branch from the incumbent
  lane 1  explore   the archive's best clade (clade score + exploration bonus), if it is not the incumbent
  lane 2+ widen     the next best parents, each distinct
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .archive import Archive


LENSES = ("per-task", "contrast", "tools")


@dataclass
class Worker:
    lane: int
    parent: str
    reason: str
    lens: str = "per-task"


@dataclass
class Plan:
    workers: list[Worker] = field(default_factory=list)
    note: str = ""
    stop: bool = False


def plan_round(archive: Archive, incumbent: str, *, rnd: int, max_workers: int = 1, budget_left: float = 1.0,
               recent_promotions: list[bool] | None = None, lenses: bool = False) -> Plan:
    """lenses: when the archive cannot supply distinct parents for every lane, fill the remaining lanes from the
    same parent with a different reflector lens instead of leaving them idle."""
    recent = recent_promotions or []
    if budget_left <= 0.08:
        return Plan([], f"stopping before round {rnd}: {budget_left:.0%} of the run budget left", stop=True)
    workers = max(1, int(max_workers))
    note = []
    if budget_left < 0.25 and workers > 1:
        workers = 1; note.append(f"budget at {budget_left:.0%}, one lane only")
    if len(recent) >= 2 and not any(recent[-2:]) and workers > 1:
        note.append("two rounds without a promotion, keeping the explore lane open")
    plan = Plan([Worker(0, incumbent, "exploit the incumbent")])
    chosen = {incumbent}
    if workers > 1 and len(archive.nodes) > 1:
        ranked = sorted((n for v, n in archive.nodes.items() if v not in chosen),
                        key=lambda n: (archive.parent_score(n.version), n.score, n.created), reverse=True)
        for n in ranked[:workers - 1]:
            lane = len(plan.workers)
            why = ("explore the best clade" if lane == 1 else "widen the search") + \
                  f" (clade {archive.clade_score(n.version):.2f}, parent score {archive.parent_score(n.version):.2f})"
            plan.workers.append(Worker(lane, n.version, why)); chosen.add(n.version)
    if lenses and len(plan.workers) < workers:
        for lens in LENSES[1:]:
            if len(plan.workers) >= workers:
                break
            plan.workers.append(Worker(len(plan.workers), incumbent, f"same parent, {lens} lens", lens))
        if len(plan.workers) > 1:
            note.append("lanes share a parent with different lenses")
    elif len(plan.workers) == 1 and workers > 1:
        note.append("archive has no second parent yet")
    plan.note = f"round {rnd}: {len(plan.workers)} lane{'s' if len(plan.workers) != 1 else ''}" + (" · " + "; ".join(note) if note else "")
    return plan
