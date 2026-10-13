"""Bundle: everything about an agent that the optimizer may change, as a versioned directory.

    prompt.md       system prompt
    tools.json      {tool name: description} overrides for the agent's tools
    playbook.jsonl  learned rules, rendered into the prompt under "## Playbook"
    bundle.lock     version (content hash), parent version and the diagnosis that produced it

Every edit makes a child with derive() and gets a new version on finalize(), so each candidate has a parent and a
reason, and any promoted change can be traced back to the failures it fixed.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PlaybookEntry:
    id: str
    text: str
    helpful: int = 0
    harmful: int = 0


@dataclass
class Bundle:
    path: Path
    prompt: str = ""
    tools: dict[str, Any] = field(default_factory=dict)
    playbook: list[PlaybookEntry] = field(default_factory=list)
    lock: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "Bundle":
        p = Path(path)
        b = cls(path=p)
        if (p / "prompt.md").exists():
            b.prompt = (p / "prompt.md").read_text()
        if (p / "tools.json").exists():
            b.tools = json.loads((p / "tools.json").read_text())
        if (p / "playbook.jsonl").exists():
            b.playbook = [PlaybookEntry(**json.loads(line)) for line in (p / "playbook.jsonl").read_text().splitlines() if line.strip()]
        if (p / "bundle.lock").exists():
            b.lock = json.loads((p / "bundle.lock").read_text())
        b.lock.setdefault("version", b.content_hash())
        return b

    def save(self, path: str | Path | None = None) -> Path:
        p = Path(path) if path else self.path
        p.mkdir(parents=True, exist_ok=True)
        (p / "prompt.md").write_text(self.prompt)
        if self.tools:
            (p / "tools.json").write_text(json.dumps(self.tools, indent=2))
        elif (p / "tools.json").exists():
            (p / "tools.json").unlink()
        (p / "playbook.jsonl").write_text("".join(json.dumps(e.__dict__) + "\n" for e in self.playbook))
        (p / "bundle.lock").write_text(json.dumps(self.lock, indent=2))
        return p

    def content_hash(self) -> str:
        h = hashlib.sha256()
        h.update(self.prompt.encode())
        h.update(json.dumps(self.tools, sort_keys=True).encode())
        for e in self.playbook:  # rule texts only: helpful/harmful counters are not content
            h.update(e.text.encode())
        return "b_" + h.hexdigest()[:12]

    @property
    def version(self) -> str:
        return self.lock.get("version", self.content_hash())

    def derive(self, diagnosis: str) -> "Bundle":
        """A child for the curator to edit; it gets its version on finalize()."""
        return Bundle(path=self.path, prompt=self.prompt, tools=json.loads(json.dumps(self.tools)),
                      playbook=[PlaybookEntry(**e.__dict__) for e in self.playbook],
                      lock={"parent": self.version, "diagnosis": diagnosis})

    def finalize(self) -> "Bundle":
        self.lock["version"] = self.content_hash()
        self.lock["created"] = time.time()
        return self

    def effective_prompt(self) -> str:
        """What the agent sees: the prompt plus the live playbook rules."""
        parts = [self.prompt.strip()]
        live = [e for e in self.playbook if e.harmful <= e.helpful]
        if live:
            parts.append("## Playbook (learned from past runs)\n" + "\n".join(f"- {e.text}" for e in live))
        return "\n\n".join(p for p in parts if p)

    def snapshot(self, versions_dir: Path) -> Path:
        dst = versions_dir / self.version
        if not dst.exists():
            self.save(dst)
        return dst
