"""Tables from bench/results: test scores, gains over the starting agent, Clade against each competitor, ablations.

    python -m bench.report            # writes bench/results/REPORT.md and prints it

Intervals are 95% bootstraps over test tasks and seeds (stats.hierarchical_bootstrap); p-values for Clade against the
competitors are Holm-adjusted within each benchmark.

Overfitting gap: the validation score of the bundle a method shipped (from its own learning episodes in runs/) minus
its test score. A method that tunes to its practice tasks shows a large positive gap.
"""

from __future__ import annotations

import json
from collections import defaultdict

from clade.reflector import is_infra_error
from clade.store import Store

from . import benchmarks
from .common import RESULTS, RUNS
from .optimizers.clade_opt import ABLATIONS
from .stats import hierarchical_bootstrap, holm, pooled_bootstrap

CORE = ["gepa", "gepa-tools", "mipro", "ace"]          # H1's Holm family
COMPETITORS = [*CORE, "mipro-demos", "simba", "textgrad"]


def load() -> dict:
    """{(bench, agent, budget): {method: [result, ...]}} for finished runs."""
    out: dict = defaultdict(lambda: defaultdict(list))
    for p in sorted(RESULTS.glob("*/*/budget*/*/seed*.json")):
        r = json.loads(p.read_text())
        if r.get("status") == "done":
            r["_learn_db"] = RUNS / p.relative_to(RESULTS).with_suffix("") / "learn.db"
            out[(r["bench"], p.parts[-4], r["budget"])][r["method"]].append(r)
    base = {(b, a): ms["baseline"] for (b, a, _), ms in out.items() if ms.get("baseline")}  # the start does not depend on the budget
    for (b, a, n), ms in out.items():
        if (b, a) in base:
            ms["baseline"] = base[(b, a)]
    return {k: ms for k, ms in out.items() if set(ms) != {"baseline"} or k[2] == max(n for (b, a, n) in out if (b, a) == k[:2])}


_VAL: dict[str, set[str]] = {}


def val_score(r: dict) -> float | None:
    """Mean validation score of the shipped bundle, from the run's learning store; None if it was never validated."""
    if r["method"] == "baseline" or not r["_learn_db"].exists():
        return None
    if r["bench"] not in _VAL:
        _VAL[r["bench"]] = {t.id for t in benchmarks.get(r["bench"]).splits()[1]}
    eps = [e for e in Store(r["_learn_db"]).episodes(bundle_version=r["bundle"]["version"])
           if e["task_id"] in _VAL[r["bench"]] and not is_infra_error(e)]
    return sum(e["score"] or 0 for e in eps) / len(eps) if eps else None


def per_task(runs: list[dict]) -> list[dict[str, float]]:
    return [r["test"]["per_task"] for r in runs]


def fmt(d: dict) -> str:
    return f"{d['delta']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]"


def table(rows: list[list], head: list[str]) -> str:
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def section(bench: str, agent: str, budget: int, methods: dict) -> str:
    base = methods.get("baseline")
    out = [f"## {bench} · agent {agent} · budget {budget} rollouts"]
    if not base:
        return out[0] + "\n\nNo baseline runs yet.\n"
    b_pt = per_task(base)
    rows = []
    order = ["baseline", "clade", *COMPETITORS, *[m for m in ABLATIONS if m != "clade"]]
    for m in [m for m in order if m in methods]:
        runs = methods[m]
        mean = sum(r["test"]["mean"] for r in runs) / len(runs)
        gain = fmt(hierarchical_bootstrap(b_pt, per_task(runs))) if m != "baseline" else "—"
        roll = round(sum(r.get("rollouts", 0) for r in runs) / len(runs))
        usd = sum(r.get("optimizer_usd", 0) + r.get("agent_learn_usd", 0) for r in runs) / len(runs)
        mins = sum(r.get("learn_seconds", 0) for r in runs) / len(runs) / 60
        write = runs[0]["test"].get("subsets", {}).get("write")
        wmean = f"{sum(sum(r['test']['per_task'][t] for t in write) / len(write) for r in runs) / len(runs):.3f}" if write else "—"
        scores = [r["test"]["mean"] for r in runs]
        spread = f"{min(scores):.2f}–{max(scores):.2f}" if len(runs) > 1 else "—"
        vals = [(v, r["test"]["mean"]) for r in runs if (v := val_score(r)) is not None]
        gap = f"{sum(v - t for v, t in vals) / len(vals):+.3f}" if vals else "—"
        rows.append([m, len(runs), f"{mean:.3f}", spread, wmean, gain, gap, roll, f"${usd:.2f}", f"{mins:.0f}"])
    out.append(table(rows, ["method", "seeds", "test score", "seed range", "write tasks", "gain over start [95% CI]",
                            "overfitting gap", "rollouts", "learning cost", "learning min"]))

    if "clade" in methods:
        c_pt = per_task(methods["clade"])
        vs = {m: hierarchical_bootstrap(per_task(methods[m]), c_pt) for m in COMPETITORS if m in methods}
        if vs:
            adj = holm({m: d["p_le_0"] for m, d in vs.items() if m in CORE})
            out.append("\nClade minus competitor (positive favours Clade; Holm over the core competitors present):\n")
            out.append(table([[m, fmt(d), adj.get(m, f"{d['p_le_0']} (unadjusted)")] for m, d in vs.items()], ["competitor", "difference [95% CI]", "p"]))
        abl = {m: hierarchical_bootstrap(c_pt, per_task(methods[m])) for m in ABLATIONS if m != "clade" and m in methods}
        if abl:
            out.append("\nAblation minus full Clade (negative means the removed part helps):\n")
            out.append(table([[m, fmt(d)] for m, d in abl.items()], ["ablation", "difference [95% CI]"]))
    return "\n".join(out) + "\n"


def pooled(data: dict, benches: list[str]) -> str:
    """Clade against each method over several benchmarks at once (same agent and budget), each weighted equally."""
    groups = {k: ms for k, ms in data.items() if k[0] in benches}
    if len({k[0] for k in groups}) < 2 or len({(k[1], k[2]) for k in groups}) != 1:
        return ""
    methods = [m for m in ["baseline", *COMPETITORS] if all(m in ms and "clade" in ms for ms in groups.values())]
    if not methods:
        return ""
    vs = {m: pooled_bootstrap([(per_task(ms[m]), per_task(ms["clade"])) for ms in groups.values()]) for m in methods}
    adj = holm({m: d["p_le_0"] for m, d in vs.items() if m in CORE})
    rows = [["starting agent" if m == "baseline" else m, fmt(d), adj.get(m, f"{d['p_le_0']} (unadjusted)")] for m, d in vs.items()]
    return (f"## Pooled: {' + '.join(sorted({k[0] for k in groups}))}, each benchmark weighted equally\n\n"
            "Clade minus method (positive favours Clade; Holm over the core competitors):\n\n"
            + table(rows, ["method", "difference [95% CI]", "p"]) + "\n")


def main():
    data = load()
    text = "# Results\n\nGenerated by `python -m bench.report` from the run files in this directory.\n\n" + \
           pooled(data, ["tau2-airline", "tau2-retail"]) + "\n" + \
           "\n".join(section(b, a, n, ms) for (b, a, n), ms in sorted(data.items()))
    (RESULTS / "REPORT.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
