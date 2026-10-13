"""Tables, figures and in-text numbers for the paper, computed from bench/results (run from the repository root):

    .venv/bin/python paper/make_assets.py

Writes paper/generated/*.tex and paper/figures/*.pdf. Every number in the paper comes from here.
"""

from __future__ import annotations

import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from bench.report import load, per_task, val_score  # noqa: E402
from bench.stats import hierarchical_bootstrap, holm, pooled_bootstrap  # noqa: E402
from clade.bundle import Bundle  # noqa: E402

GEN, FIG = ROOT / "paper" / "generated", ROOT / "paper" / "figures"
BENCH = {"tau2-airline": "Airline", "tau2-retail": "Retail", "support-desk": "Support desk"}
NAME = {"baseline": "Starting agent", "clade": "\\textsc{Clade}", "gepa": "GEPA", "gepa-tools": "GEPA+tools",
        "mipro": "MIPROv2", "ace": "ACE", "clade-rewrite": "rewrites", "clade-no-critic": "no critic",
        "clade-greedy": "greedy", "clade-no-gate": "no gate"}
CORE = ["gepa", "gepa-tools", "mipro", "ace"]
MAIN = ["baseline", "clade", *CORE]
ABL = ["clade-rewrite", "clade-no-critic", "clade-greedy", "clade-no-gate"]
_S = {"tau2": Bundle.load(ROOT / "bench/benchmarks/tau2/start"), "support": Bundle.load(ROOT / "bench/benchmarks/support_desk/start")}
START = {k: (b.prompt.strip(), (), ()) for k, b in _S.items()}  # the starting bundles have no rules and no tool overrides
NUM: dict[str, str] = {}


def num(name: str, value) -> None:
    NUM[name] = value if isinstance(value, str) else f"{value:.2f}"


def ci(d: dict, digits: int = 2) -> str:
    return f"{d['delta']:+.{digits}f} [{d['ci95'][0]:+.{digits}f}, {d['ci95'][1]:+.{digits}f}]".replace("-", "$-$")


def unchanged(r: dict) -> bool:
    """Shipped the starting agent: same prompt (up to surrounding whitespace), no rules, no tool overrides."""
    b = r["bundle"]
    return (b["prompt"].strip(), tuple(b["playbook"]), tuple(sorted(b["tools"].items()))) == START[r["bench"].split("-")[0]]


def main() -> None:
    data = {b: ms for (b, a, n), ms in load().items() if a == "openai_gpt-oss-20b" and b in BENCH}
    GEN.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)

    # --- how much does the unchanged starting agent's test score move between scorings?
    starts = {}
    for b, ms in data.items():
        scores = [r["test"]["mean"] for m, runs in ms.items() for r in runs if m == "baseline" or unchanged(r)]
        starts[b] = scores
        key = {"tau2-airline": "Air", "tau2-retail": "Ret", "support-desk": "Sup"}[b]
        num(f"start{key}Mean", st.mean(scores)); num(f"start{key}Min", min(scores)); num(f"start{key}Max", max(scores))
        num(f"start{key}N", str(len(scores)))

    # --- harmful runs: the shipped agent scored below every scoring of the unchanged starting agent
    harm = {m: [0, 0] for m in MAIN[1:]}
    for b, ms in data.items():
        for m in MAIN[1:]:
            for r in ms.get(m, []):
                harm[m][1] += 1
                harm[m][0] += r["test"]["mean"] < min(starts[b])
    for m, (h, n) in harm.items():
        num(f"harm{m.replace('-', '').replace('gepatools', 'gepaTools').capitalize()}", f"{h}/{n}")

    # --- Table 1: scores, gain over the starting agent, runs that changed the agent, cost and time
    rows = []
    for m in MAIN:
        cells = [NAME[m]]
        for b in BENCH:
            runs = data[b].get(m, [])
            if not runs:
                cells += ["--", "--"]; continue
            mean = st.mean(r["test"]["mean"] for r in runs)
            if m == "baseline":
                cells += [f"{mean:.3f}", "--"]
            else:
                d = hierarchical_bootstrap(per_task(data[b]["baseline"]), per_task(runs))
                ch = sum(not unchanged(r) for r in runs)
                cells += [f"{mean:.3f}", f"{d['delta']:+.3f}".replace("-", "$-$") + f" \\scriptsize({ch}/{len(runs)})"]
        tau = [r for b in ("tau2-airline", "tau2-retail") for r in data[b].get(m, [])]
        if m == "baseline":
            cells += ["--", "--"]
        else:
            cells += [f"{st.mean(r['optimizer_usd'] + r['agent_learn_usd'] for r in tau):.2f}",
                      f"{st.mean(r['learn_seconds'] for r in tau) / 3600:.1f}"]
        rows.append(" & ".join(cells) + " \\\\")
    (GEN / "main_table.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")

    # --- Table 2: Clade minus each competitor, per τ² benchmark and pooled (Holm within each column)
    cols = {}
    for b in ("tau2-airline", "tau2-retail"):
        c = per_task(data[b]["clade"])
        cols[b] = {m: hierarchical_bootstrap(per_task(data[b][m]), c) for m in CORE}
    cols["pooled"] = {m: pooled_bootstrap([(per_task(data[b][m]), per_task(data[b]["clade"])) for b in ("tau2-airline", "tau2-retail")]) for m in CORE}
    short, pooled_rows = [], []
    adj = {k: holm({x: d["p_le_0"] for x, d in ds.items()}) for k, ds in cols.items()}
    for m in CORE:
        short.append(f"{NAME[m]} & {ci(cols['tau2-airline'][m])} & {ci(cols['tau2-retail'][m])} \\\\")
        short.append(f" & {{\\scriptsize $p={adj['tau2-airline'][m]:.3f}$}} & {{\\scriptsize $p={adj['tau2-retail'][m]:.3f}$}} \\\\")
        pooled_rows.append(f"{NAME[m]} & {ci(cols['pooled'][m])} & {adj['pooled'][m]:.3f} \\\\")
    (GEN / "vs_table_short.tex").write_text("\n".join(short) + "\n\\bottomrule\n")
    (GEN / "vs_table_pooled.tex").write_text("\n".join(pooled_rows) + "\n\\bottomrule\n")
    sd = data["support-desk"]
    sd_vs = {m: hierarchical_bootstrap(per_task(sd[m]), per_task(sd["clade"])) for m in CORE}
    sd_adj = holm({m: d["p_le_0"] for m, d in sd_vs.items()})
    for m in CORE:
        key = m.replace("-", "").replace("gepatools", "gepaTools").capitalize()
        num(f"sdVs{key}", ci(sd_vs[m])); num(f"sdVs{key}P", f"{sd_adj[m]:.3f}")
        num(f"retVs{key}", ci(cols["tau2-retail"][m]))
    d = pooled_bootstrap([(per_task(data[b]["baseline"]), per_task(data[b]["clade"])) for b in ("tau2-airline", "tau2-retail")])
    num("cladePooledGain", ci(d)); num("cladePooledGainP", f"{d['p_le_0']:.3f}")
    d = hierarchical_bootstrap(per_task(sd["baseline"]), per_task(sd["clade"])); num("cladeSdGain", ci(d))

    # --- Table 3: ablations on airline
    air = data["tau2-airline"]
    rows = [f"full & {st.mean(r['test']['mean'] for r in air['clade']):.3f} & -- & {sum(not unchanged(r) for r in air['clade'])}/3 \\\\"]
    for m in ABL:
        d = hierarchical_bootstrap(per_task(air["clade"]), per_task(air[m]))
        rows.append(f"{NAME[m]} & {st.mean(r['test']['mean'] for r in air[m]):.3f} & {ci(d)} & {sum(not unchanged(r) for r in air[m])}/3 \\\\")
    (GEN / "ablation_table.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")

    # --- Table 4 (appendix): per-seed scores, overfitting gap, rollouts
    rows = []
    for b in BENCH:
        for m in MAIN[1:]:
            runs = sorted(data[b].get(m, []), key=lambda r: r["seed"])
            seeds = " ".join(f"{r['test']['mean']:.3f}" + ("$^\\circ$" if unchanged(r) else "") for r in runs)
            gaps = [v - r["test"]["mean"] for r in runs if (v := val_score(r)) is not None]
            gap = f"{st.mean(gaps):+.2f}".replace("-", "$-$") if gaps else "--"
            rows.append(f"{BENCH[b]} & {NAME[m]} & {seeds} & {gap} & {round(st.mean(r['rollouts'] for r in runs))} \\\\")
        rows.append("\\midrule")
    (GEN / "seed_table.tex").write_text("\n".join(rows[:-1]) + "\n\\bottomrule\n")

    # --- Figure 1: every run's test score against the spread of the unchanged starting agent
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3), sharey=False)
    order = MAIN[1:]
    labels = ["Clade", "GEPA", "GEPA\n+tools", "MIPROv2", "ACE"]
    for ax, b in zip(axes, BENCH):
        lo, hi, mu = min(starts[b]), max(starts[b]), st.mean(starts[b])
        ax.axhspan(lo, hi, color="#d9d9d9", zorder=0, lw=0)
        ax.axhline(mu, color="#7f7f7f", lw=0.8, ls="--", zorder=1)
        for i, m in enumerate(order):
            runs = data[b].get(m, [])
            for j, r in enumerate(sorted(runs, key=lambda r: r["seed"])):
                x = i + (j - 1) * 0.14
                same = unchanged(r)
                ax.scatter(x, r["test"]["mean"], s=16, zorder=3, facecolors="white" if same else ("#1f4e79" if m == "clade" else "#c55a11"),
                           edgecolors="#1f4e79" if m == "clade" else "#c55a11", linewidths=0.9)
        ax.set_xticks(range(len(order))); ax.set_xticklabels(labels, fontsize=6.5)
        ax.set_title(BENCH[b], fontsize=8); ax.tick_params(axis="y", labelsize=6.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("test score", fontsize=7)
    fig.tight_layout(pad=0.4)
    fig.savefig(FIG / "runs.pdf")

    # --- Table 6 (appendix): shipped prompt size and the airline/retail tasks that need a database write
    rows = []
    for m in MAIN[1:]:
        cells = [NAME[m]]
        for b in BENCH:
            runs = data[b].get(m, [])
            cells.append(f"{round(st.mean(r['bundle']['chars'] for r in runs)):,}".replace(",", "{,}"))
        for b in ("tau2-airline", "tau2-retail"):
            runs = data[b][m]; w = runs[0]["test"]["subsets"]["write"]
            cells.append(f"{st.mean(st.mean(r['test']['per_task'][t] for t in w) for r in runs):.3f}")
        rows.append(" & ".join(cells) + " \\\\")
    base = ["Starting agent"] + [f"{len(_S['tau2' if b != 'support-desk' else 'support'].effective_prompt())}" for b in BENCH]
    for b in ("tau2-airline", "tau2-retail"):
        runs = data[b]["baseline"]; w = runs[0]["test"]["subsets"]["write"]
        base.append(f"{st.mean(st.mean(r['test']['per_task'][t] for t in w) for r in runs):.3f}")
        num({"tau2-airline": "nWriteAir", "tau2-retail": "nWriteRet"}[b], str(len(w)))
    (GEN / "size_table.tex").write_text(" & ".join(base) + " \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n")
    for m in ("clade", "ace"):
        sizes = [r["bundle"]["chars"] for b in ("tau2-airline", "tau2-retail") for r in data[b][m]]
        num(f"size{m.capitalize()}Min", f"{min(sizes):,}".replace(",", "{,}")); num(f"size{m.capitalize()}Max", f"{max(sizes):,}".replace(",", "{,}"))

    # --- is the overfitting gap explained by how many candidates were compared on the validation tasks?
    from clade.store import Store
    from bench import benchmarks
    pts = []
    for b in BENCH:
        val = {t.id for t in benchmarks.get(b).splits()[1]}
        for m in MAIN[1:]:
            for r in data[b].get(m, []):
                v = val_score(r)
                if v is not None and r["_learn_db"].exists():
                    k = len({e["bundle_version"] for e in Store(r["_learn_db"]).episodes() if e["task_id"] in val})
                    pts.append((k, v - r["test"]["mean"]))

    def ranks(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i]); out = [0.0] * len(xs); i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            for t in range(i, j + 1):
                out[order[t]] = (i + j) / 2
            i = j + 1
        return out
    rx, ry = ranks([k for k, _ in pts]), ranks([g for _, g in pts])
    mx, my = st.mean(rx), st.mean(ry)
    rho = sum((x - mx) * (y - my) for x, y in zip(rx, ry)) / (sum((x - mx) ** 2 for x in rx) * sum((y - my) ** 2 for y in ry)) ** 0.5
    num("candGapRho", rho); num("candGapN", str(len(pts)))

    (GEN / "numbers.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(NUM.items())))
    print("\n".join(f"{k} = {v}" for k, v in sorted(NUM.items())))


if __name__ == "__main__":
    main()
