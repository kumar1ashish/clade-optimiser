"""How Clade talks to an agent and a judge. One adapter per agent framework; one judge per kind of task.

  adapter.apply(agent, bundle, cwd)   put the bundle's prompt, playbook and tool descriptions into the agent
  adapter.instrument(agent, recorder) attach hooks that record every model call and tool execution
  adapter.run(agent, task, cwd)       run one episode
  adapter.tools(agent)                current tool descriptions, which the curator may rewrite
  judge.score(task, episode, cwd)     a score in [0, 1] and a rationale the reflector reads
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .bundle import Bundle
from .schema import Call, Episode, Task, ToolExec
from .store import Store


class Recorder:
    """Writes one episode's calls and tool executions to the store as they happen."""

    def __init__(self, store: Store, episode: Episode):
        self.store, self.episode, self.call_index = store, episode, -1

    def call(self, response: dict[str, Any], *, request: dict | None = None, model: str | None = None,
             usage: dict | None = None, latency_ms: float | None = None) -> Call:
        self.call_index += 1
        c = Call(episode_id=self.episode.id, index=self.call_index, response=response, request=request,
                 model=model, usage=usage or {}, latency_ms=latency_ms)
        self.store.put_call(c)
        return c

    def tool(self, name: str, input: dict, output: Any = None, error: str | None = None) -> ToolExec:
        t = ToolExec(episode_id=self.episode.id, call_index=self.call_index if self.call_index >= 0 else None,
                     name=name, input=input, output=output, error=error)
        self.store.put_tool_exec(t)
        return t


@dataclass
class RunResult:
    final_text: str = ""
    cost_usd: float | None = None
    num_turns: int | None = None
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class Adapter(Protocol):
    def apply(self, agent: Any, bundle: Bundle, cwd: Path) -> Any: ...
    def instrument(self, agent: Any, recorder: Recorder) -> Any: ...
    def run(self, agent: Any, task: Task, cwd: Path) -> RunResult: ...
    def tools(self, agent: Any) -> list[dict[str, Any]]: ...


@dataclass
class Verdict:
    score: float
    rationale: str = ""


class Judge(Protocol):
    def score(self, task: Task, episode: Episode, cwd: Path) -> Verdict: ...


class ShellJudge:
    """Run a command in the episode's workspace; exit code 0 passes. Hidden tests in the task dir's tests_hidden/
    are copied in only now, so the agent cannot see or game them during the episode."""

    def __init__(self, default_cmd: str = "python -m pytest -q", timeout: int = 120):
        self.default_cmd, self.timeout = default_cmd, timeout

    def score(self, task: Task, episode: Episode, cwd: Path) -> Verdict:
        cmd, timeout = task.judge.get("cmd", self.default_cmd), task.judge.get("timeout", self.timeout)
        if task.workspace:
            hidden = Path(task.workspace).parent / "tests_hidden"
            if hidden.is_dir():
                shutil.copytree(hidden, cwd / "tests_hidden", dirs_exist_ok=True)
        try:
            p = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return Verdict(0.0, f"judge timeout after {timeout}s")
        return Verdict(1.0 if p.returncode == 0 else 0.0, f"exit={p.returncode}\n{(p.stdout + p.stderr)[-1500:]}")
