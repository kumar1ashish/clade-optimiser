"""One run: learn with one method on one benchmark and seed, then score the learned bundle on the frozen test split.

    python -m bench.run --bench tau2-airline --method clade --seed 0

Writes bench/results/<bench>/<agent>/budget<B>/<method>/seed<k>.json; scratch (stores, bundles) goes to runs/.
"""

from __future__ import annotations

import argparse
import shutil
import time
import traceback
from pathlib import Path

from clade.bundle import Bundle
from clade.cost import usd
from clade.llm import track_usage

from . import benchmarks, optimizers
from .common import RESULTS, RUNS, Ctx, Models, evaluate_test, make_llm, write_json


def result_path(bench: str, models: Models, budget: int, method: str, seed: int) -> Path:
    return RESULTS / bench / models.slug() / f"budget{budget}" / method / f"seed{seed}.json"


def run(bench_name: str, method: str, seed: int, models: Models, *, budget: int | None = None, test_n: int | None = None,
        params: dict | None = None, log=print) -> dict:
    bench = benchmarks.get(bench_name)
    train, val, test = bench.splits()
    budget = bench.budget if budget is None else budget
    test_n = bench.test_n if test_n is None else test_n
    work = RUNS / bench_name / models.slug() / f"budget{budget}" / method / f"seed{seed}"
    shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(bench.start(), work / "bundle", ignore=shutil.ignore_patterns(".versions"))
    tr = bench.trainable(work / "bundle", models, work / "learn.db")
    ctx = Ctx(bench=bench, method=method, seed=seed, models=models, budget=budget, workdir=work, start=Bundle.load(work / "bundle"),
              train=train, val=val, llm=make_llm(models.optimizer), tr=tr, log=lambda m: log(f"[{bench_name} {method} s{seed}] {m}"),
              params=params or {})
    rec = {"bench": bench_name, "method": method, "seed": seed, "models": models.__dict__, "budget": budget, "test_n": test_n,
           "params": params or {}, "started": time.time(), "status": "running"}
    t0 = time.time()
    try:
        with track_usage() as book:
            final = optimizers.get(method)(ctx)
    except Exception:  # noqa: BLE001  (a crashed run is reported, not dropped)
        rec |= {"status": "crashed", "error": traceback.format_exc()[-3000:], "rollouts": tr.rollouts}
        write_json(result_path(bench_name, models, budget, method, seed), rec)
        raise
    learn_eps = tr.store.episodes()
    rec |= {"learn_seconds": round(time.time() - t0), "rollouts": tr.rollouts, "infra_reruns": tr.infra_reruns,
            "optimizer_tokens": {m: v for m, v in book.items()},
            "optimizer_usd": round(sum(usd(m, v["prompt_tokens"], v["completion_tokens"]) for m, v in book.items()), 4),
            "agent_learn_usd": round(sum(e["cost_usd"] or 0 for e in learn_eps), 4),
            "bundle": {"version": final.version, "prompt": final.prompt, "playbook": [e.text for e in final.playbook],
                       "tools": final.tools, "chars": len(final.effective_prompt())}}
    final.save(work / "final")
    log(f"[{bench_name} {method} s{seed}] learned in {rec['learn_seconds']} s, {tr.rollouts} rollouts; scoring on {len(test)} test tasks x {test_n}")
    rec["test"] = evaluate_test(bench, work / "final", models, test, test_n, work / "test.db")
    rec["test"]["subsets"] = bench.subsets() if hasattr(bench, "subsets") else {}
    rec |= {"status": "done", "finished": time.time()}
    write_json(result_path(bench_name, models, budget, method, seed), rec)
    log(f"[{bench_name} {method} s{seed}] test {rec['test']['mean']}")
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--method", required=True, choices=optimizers.METHODS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, help="learning rollouts (default: the benchmark's)")
    ap.add_argument("--test-n", type=int, help="runs per test task (default: the benchmark's)")
    ap.add_argument("--agent", default=Models.agent)
    ap.add_argument("--optimizer", default=Models.optimizer)
    ap.add_argument("--user", default=Models.user)
    a = ap.parse_args()
    run(a.bench, a.method, a.seed, Models(a.agent, a.optimizer, a.user), budget=a.budget, test_n=a.test_n)


if __name__ == "__main__":
    main()
