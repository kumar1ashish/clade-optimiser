"""Support desk: a tool-using customer-support agent for a fictional store, scored by hidden tests on the store's
side effects. 7 train, 3 validation and 20 test tasks. Clade was developed on this benchmark, so it is reported as
the development benchmark, never as evidence on its own."""

from __future__ import annotations

import re
from pathlib import Path

from clade.agent import ShellJudge
from clade.schema import load_tasks

from ...common import RUNS, Models, StudyTrainable, endpoint
from .agent import SupportAgent, SupportAgentAdapter
from .env import TOOLS

HERE = Path(__file__).resolve().parent


class SupportDesk:
    name = "support-desk"
    budget = 400   # learning rollouts per run
    test_n = 10    # runs per test task

    def splits(self):
        tasks = load_tasks(HERE / "tasks")
        by = lambda s: [t for t in tasks if t.meta["split"] == s]  # noqa: E731
        return by("train"), by("held"), by("test")

    def start(self) -> Path:
        return HERE / "start"

    def trainable(self, bundle_dir: Path, models: Models, store: Path) -> StudyTrainable:
        base, key, model = endpoint(models.agent)
        tr = StudyTrainable(SupportAgent(model=model, base_url=base, api_key=key), adapter=SupportAgentAdapter(),
                            bundle=bundle_dir, judge=ShellJudge(), store=store, work_root=RUNS / "work")
        tr.episode_workers = 8
        return tr

    def tool_descriptions(self) -> dict[str, str]:
        return {t["function"]["name"]: t["function"]["description"] for t in TOOLS}

    def feedback(self, ep) -> str:
        """The hidden test's assertion message, e.g. 'escalation was not made'."""
        if (ep.score or 0) >= 1:
            return "passed"
        msgs = re.findall(r"AssertionError: (.+)", ep.judge_rationale or "")
        return msgs[-1].strip()[:300] if msgs else (ep.judge_rationale or ep.error or "")[-300:]
