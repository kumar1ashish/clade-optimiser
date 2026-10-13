"""Tasks and trajectories. One Call per model response, grouped into an Episode; tool executions are separate
records joined to the call that requested them, so a reflector can read exactly what the agent did."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Task:
    id: str
    family: str
    prompt: str
    workspace: str | None = None                          # directory copied into a fresh cwd per episode
    judge: dict[str, Any] = field(default_factory=dict)   # judge config, e.g. {"cmd": "python -m pytest -q"}
    held_out: bool = False                                # held-out tasks are only used for selection
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, task_dir: Path) -> "Task":
        data = json.loads((task_dir / "task.json").read_text())
        ws = task_dir / "workspace"
        data.setdefault("workspace", str(ws) if ws.exists() else None)
        data.setdefault("id", task_dir.name)
        return cls(**data)


def load_tasks(root: str | Path) -> list[Task]:
    """A task set is a directory of task dirs, each with task.json and an optional workspace/."""
    return [Task.load(d) for d in sorted(Path(root).iterdir()) if (d / "task.json").exists()]


@dataclass
class Call:
    episode_id: str
    index: int
    response: dict[str, Any]               # {"text": [...], "tool_calls": [{"id", "name", "input"}], "thinking": [...]}
    request: dict[str, Any] | None = None
    model: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    latency_ms: float | None = None
    ts: float = field(default_factory=time.time)


@dataclass
class ToolExec:
    episode_id: str
    call_index: int | None
    name: str
    input: dict[str, Any]
    output: Any = None
    error: str | None = None
    ts: float = field(default_factory=time.time)


@dataclass
class Episode:
    id: str
    task_id: str
    bundle_version: str
    started: float = field(default_factory=time.time)
    ended: float | None = None
    score: float | None = None
    judge_rationale: str | None = None
    cwd: str | None = None
    cost_usd: float | None = None
    num_turns: int | None = None
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def new(task_id: str, bundle_version: str) -> "Episode":
        return Episode(id=f"ep_{uuid.uuid4().hex[:12]}", task_id=task_id, bundle_version=bundle_version)
