"""What every run shares: model endpoints, the counted rollout path, and frozen-test scoring.

Every optimizer, Clade and the competitors, runs the agent only through a StudyTrainable: one rollout is one
`run()` call, counted against the run's budget, and an episode lost to the provider (rate limit, timeout) is re-run
instead of being scored. Competitors call `Ctx.run`; Clade's own loop calls the same Trainable. All methods read the
same rendering of an episode (`Ctx.render`, which is what Clade's reflector reads) and the same reflection model.
Test episodes go to a separate store and never count as learning.
"""

from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from clade.bundle import Bundle, PlaybookEntry
from clade.cost import PRICES
from clade.llm import LLM, load_dotenv, openai_compatible
from clade.reflector import is_infra_error, render_episode
from clade.schema import Episode, Task
from clade.trainable import Trainable

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "bench" / "results"
RUNS = ROOT / "runs"

ENDPOINTS = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "ollama": ("http://localhost:11434/v1", None),
}


def endpoint(spec: str) -> tuple[str, str, str]:
    """'groq:openai/gpt-oss-20b' -> (base_url, api_key, model)."""
    load_dotenv(str(ROOT / ".env"))
    prov, _, model = spec.partition(":")
    if prov not in ENDPOINTS or not model:
        raise ValueError(f"model must be <provider>:<model>, provider one of {sorted(ENDPOINTS)}: {spec!r}")
    base, env = ENDPOINTS[prov]
    key = os.environ.get(env, "") if env else "ollama"
    if env and not key:
        raise SystemExit(f"{env} is not set (add it to .env)")
    if prov == "ollama":
        PRICES.setdefault(model, (0.0, 0.0))  # local models cost nothing
    return base, key, model


def make_llm(spec: str, temperature: float = 0.4) -> LLM:
    base, key, model = endpoint(spec)
    return openai_compatible(model, base_url=base, api_key=key, temperature=temperature, timeout=180)


@dataclass
class Models:
    agent: str = "groq:openai/gpt-oss-20b"
    optimizer: str = "groq:openai/gpt-oss-120b"   # reflection / proposal model, the same for every method
    user: str = "groq:openai/gpt-oss-120b"        # user simulator and judge, for benchmarks that have them

    def slug(self) -> str:
        return self.agent.split(":", 1)[1].replace("/", "_").replace(":", "_")


class StudyTrainable(Trainable):
    """Counts rollouts and re-runs episodes lost to the provider; only the final attempt is scored or counted."""

    retries = 2

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.rollouts = 0
        self.infra_reruns = 0
        self._count = threading.Lock()

    def run(self, task: Task, *, bundle: Bundle | None = None) -> Episode:
        with self._count:
            self.rollouts += 1
        ep = None
        for attempt in range(self.retries + 1):
            ep = super().run(task, bundle=bundle)
            if not (ep.error and is_infra_error({"error": ep.error})):
                return ep
            if attempt < self.retries:
                with self._count:
                    self.infra_reruns += 1
                time.sleep(5 * (attempt + 1))
        return ep


class BudgetExhausted(RuntimeError):
    pass


@dataclass
class Ctx:
    """What an optimizer receives. `run` is the only way to execute the agent while learning."""
    bench: object
    method: str
    seed: int
    models: Models
    budget: int
    workdir: Path
    start: Bundle
    train: list[Task]
    val: list[Task]
    llm: LLM
    tr: StudyTrainable
    log: Callable[[str], None] = print
    params: dict = field(default_factory=dict)
    _used: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def used(self) -> int:
        return self._used

    def run(self, task: Task, bundle: Bundle) -> Episode:
        """One learning rollout; raises BudgetExhausted once the budget is spent."""
        with self._lock:
            if self._used >= self.budget:
                raise BudgetExhausted(f"{self.budget} rollouts spent")
            self._used += 1
        return self.tr.run(task, bundle=bundle)

    def run_many(self, pairs: list[tuple[Task, Bundle]], workers: int = 8) -> list[Episode | None]:
        """Parallel rollouts; slots past the budget come back as None."""
        def one(p):
            try:
                return self.run(*p)
            except BudgetExhausted:
                return None
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            return list(ex.map(one, pairs))

    def render(self, ep: Episode, max_chars: int = 2500) -> str:
        """The episode as Clade's reflector reads it: trajectory, final answer and the judge's verdict."""
        return render_episode(self.tr.store, self.tr.store.episode(ep.id), max_chars)

    def feedback(self, ep: Episode) -> str:
        """The judge's verdict alone, for optimizers that take a separate feedback field."""
        return self.bench.feedback(ep)

    def candidate(self, *, prompt: str | None = None, tools: dict | None = None, playbook: list[str] | None = None,
                  note: str = "") -> Bundle:
        """A bundle derived from the starting bundle with the given components replaced; it gets its own version."""
        c = self.start.derive(f"[{self.method}] {note}".strip())
        if prompt is not None:
            c.prompt = prompt
        if tools is not None:
            c.tools = dict(tools)
        if playbook is not None:
            c.playbook = [PlaybookEntry(id=f"pb_{i}", text=t) for i, t in enumerate(playbook)]
        return c.finalize()


def evaluate_test(bench, bundle_dir: Path, models: Models, tasks: list[Task], n: int, store: Path, *, workers: int = 8) -> dict:
    """Score a bundle on the frozen test split, n runs per task."""
    tr = bench.trainable(bundle_dir, models, store)
    tr.retries = 3
    with ThreadPoolExecutor(max_workers=workers) as ex:
        eps = list(ex.map(lambda t: tr.run(t), [t for t in tasks for _ in range(n)]))
    runs: dict[str, list[float]] = {}
    for ep in eps:
        runs.setdefault(ep.task_id, []).append(float(ep.score or 0.0))
    return {"mean": round(sum(sum(v) for v in runs.values()) / len(eps), 4), "n": n, "episodes": len(eps),
            "per_task": {k: round(sum(v) / len(v), 4) for k, v in sorted(runs.items())},
            "runs": dict(sorted(runs.items())),
            "infra_failures_left": sum(bool(e.error and is_infra_error({"error": e.error})) for e in eps),
            "infra_reruns": tr.infra_reruns, "agent_usd": round(sum(e.cost_usd or 0 for e in eps), 4)}


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    tmp.replace(path)
