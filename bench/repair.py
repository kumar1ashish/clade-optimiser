"""Re-score test episodes that were lost to the provider.

A test episode whose every attempt failed at the provider (rate limit, no capacity) was scored 0, although the
protocol re-runs provider losses. This finds each such slot from the run's test store, runs that task again with the
run's shipped bundle, and records the change in the result file under "repaired". Agent crashes that persisted are
left at 0, as the protocol says.

    python -m bench.repair            # all results
"""

from __future__ import annotations

import json
from collections import defaultdict

from clade.reflector import is_infra_error
from clade.store import Store

from . import benchmarks
from .common import RESULTS, RUNS, Models, write_json


def provider_lost(e: dict) -> bool:
    return bool(e["error"]) and is_infra_error(e) and ("provider" in e["error"] or "HTTP" in e["error"])


def repair(path) -> list[dict]:
    r = json.loads(path.read_text())
    if r.get("status") != "done" or not r["test"].get("infra_failures_left"):
        return []
    work = RUNS / path.relative_to(RESULTS).with_suffix("")
    eps = Store(work / "test.db").episodes()
    by: dict[str, list[dict]] = defaultdict(list)
    for e in eps:
        by[e["task_id"]].append(e)
    bench = benchmarks.get(r["bench"])
    tasks = {t.id: t for t in bench.splits()[2]}
    tr = bench.trainable(work / "final", Models(**r["models"]), work / "test.db")
    tr.retries = 3
    fixed = []
    for task_id, es in by.items():
        clean = sum(1 for e in es if not (e["error"] and is_infra_error(e)))
        lost = sorted([e for e in es if e["error"] and is_infra_error(e)], key=lambda e: e["started"])[-(r["test"]["n"] - clean):] if clean < r["test"]["n"] else []
        for e in [e for e in lost if provider_lost(e)]:
            ep = tr.run(tasks[task_id])
            if ep.error and is_infra_error({"error": ep.error}):
                continue  # still lost; leave the 0
            runs = r["test"]["runs"][task_id]
            if 0.0 not in runs:
                continue
            runs[runs.index(0.0)] = float(ep.score or 0.0)
            fixed.append({"task": task_id, "lost": e["id"], "rerun": ep.id, "score": ep.score})
    if fixed:
        t = r["test"]
        t["per_task"] = {k: round(sum(v) / len(v), 4) for k, v in sorted(t["runs"].items())}
        t["mean"] = round(sum(sum(v) for v in t["runs"].values()) / t["episodes"], 4)
        t["infra_failures_left"] -= len(fixed)
        r["repaired"] = r.get("repaired", []) + fixed
        write_json(path, r)
    return fixed


def main():
    for p in sorted(RESULTS.glob("*/*/budget*/*/seed*.json")):
        fixed = repair(p)
        if fixed:
            print(f"{p.relative_to(RESULTS)}: re-scored {len(fixed)} episode(s): {[f['score'] for f in fixed]}", flush=True)


if __name__ == "__main__":
    main()
