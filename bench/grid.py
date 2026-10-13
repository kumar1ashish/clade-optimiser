"""Run a grid of (benchmark, method, seed) runs from a config file, several at a time, skipping finished ones.

    python -m bench.grid bench/configs/main.json --jobs 4

Each run is its own process (a crash stays isolated and is recorded); the log goes to runs/logs/. Runs are ordered
seed by seed, so a grid stopped early still has every method on the same seeds.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import benchmarks
from .common import RUNS, Models
from .run import result_path


def cells(cfg: dict) -> list[tuple[str, str, int]]:
    return [(b, m, s) for s in cfg["seeds"] for b in cfg["benchmarks"] for m in cfg["methods"]]


def budget(cfg: dict, b: str) -> int:
    v = cfg.get("budget")
    v = v.get(b) if isinstance(v, dict) else v
    return int(v) if v is not None else benchmarks.get(b).budget


def done(cfg: dict, models: Models, b: str, m: str, s: int) -> bool:
    p = result_path(b, models, budget(cfg, b), m, s)
    return p.exists() and json.loads(p.read_text()).get("status") == "done"


def launch(cfg: dict, models: Models, b: str, m: str, s: int) -> int:
    log = RUNS / "logs" / f"{b}__{models.slug()}__budget{budget(cfg, b)}__{m}__seed{s}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "bench.run", "--bench", b, "--method", m, "--seed", str(s),
           "--agent", models.agent, "--optimizer", models.optimizer, "--user", models.user]
    cmd += ["--budget", str(budget(cfg, b))]
    if cfg.get("test_n") is not None:
        cmd += ["--test-n", str(cfg["test_n"])]
    t0 = time.time()
    with log.open("w") as f:
        code = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUNBUFFERED": "1", "CLADE_SERVICE_TIER": cfg.get("service_tier", "")})
    print(f"{'ok  ' if code == 0 else 'FAIL'} {b} {m} seed{s} ({time.time() - t0:.0f} s)", flush=True)
    return code


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--dry", action="store_true", help="list the runs that would start")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    models = Models(**cfg.get("models", {}))
    todo = [c for c in cells(cfg) if not done(cfg, models, *c)]
    print(f"{len(cells(cfg))} runs in {a.config}, {len(todo)} to do", flush=True)
    if a.dry:
        for c in todo:
            print(*c)
        return
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        codes = list(ex.map(lambda c: launch(cfg, models, *c), todo))
    print(f"finished: {codes.count(0)} ok, {len(codes) - codes.count(0)} failed", flush=True)


if __name__ == "__main__":
    main()
