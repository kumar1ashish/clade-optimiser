"""SQLite trajectory store, safe to share between the threads that run episodes in parallel."""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .schema import Call, Episode, ToolExec

SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
  id TEXT PRIMARY KEY, task_id TEXT, bundle_version TEXT, started REAL, ended REAL, score REAL,
  judge_rationale TEXT, cwd TEXT, cost_usd REAL, num_turns INTEGER, error TEXT, meta TEXT
);
CREATE TABLE IF NOT EXISTS calls (
  episode_id TEXT, idx INTEGER, request TEXT, response TEXT, model TEXT, usage TEXT, latency_ms REAL, ts REAL,
  PRIMARY KEY (episode_id, idx)
);
CREATE TABLE IF NOT EXISTS tool_execs (
  id INTEGER PRIMARY KEY AUTOINCREMENT, episode_id TEXT, call_index INTEGER,
  name TEXT, input TEXT, output TEXT, error TEXT, ts REAL
);
CREATE TABLE IF NOT EXISTS events (
  id TEXT PRIMARY KEY, run_id TEXT, ts REAL, round INTEGER, role TEXT, kind TEXT, ref TEXT, parent TEXT, text TEXT, data TEXT
);
CREATE INDEX IF NOT EXISTS ix_ep_task ON episodes(task_id);
CREATE INDEX IF NOT EXISTS ix_ep_bundle ON episodes(bundle_version);
"""


def _j(x: Any) -> str:
    return json.dumps(x, default=str)


class Store:
    def __init__(self, path: str | Path = "clade.db"):
        self.path = str(path)
        self._db = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.RLock()  # one connection, serialized; rows are fetched inside the lock
        with self._lock:
            self._db.executescript(SCHEMA)

    def _exec(self, sql: str, params=(), *, commit: bool = False) -> list[sqlite3.Row]:
        with self._lock:
            cur = self._db.execute(sql, params)
            rows = cur.fetchall() if cur.description else []
            if commit:
                self._db.commit()
            return rows

    # --- writes ---
    def put_episode(self, ep: Episode) -> None:
        d = asdict(ep)
        d["meta"] = _j(d["meta"])
        self._exec(f"INSERT OR REPLACE INTO episodes ({', '.join(d)}) VALUES ({', '.join(':' + k for k in d)})", d, commit=True)

    def put_call(self, c: Call) -> None:
        self._exec("INSERT OR REPLACE INTO calls VALUES (?,?,?,?,?,?,?,?)",
                   (c.episode_id, c.index, _j(c.request), _j(c.response), c.model, _j(c.usage), c.latency_ms, c.ts), commit=True)

    def put_tool_exec(self, t: ToolExec) -> None:
        self._exec("INSERT INTO tool_execs (episode_id, call_index, name, input, output, error, ts) VALUES (?,?,?,?,?,?,?)",
                   (t.episode_id, t.call_index, t.name, _j(t.input), _j(t.output), t.error, t.ts), commit=True)

    def put_event(self, d: dict) -> None:
        self._exec("INSERT OR REPLACE INTO events VALUES (?,?,?,?,?,?,?,?,?,?)",
                   (d["id"], d["run_id"], d["ts"], d.get("round"), d["role"], d["kind"], d.get("ref"), d.get("parent"),
                    d.get("text", ""), _j(d.get("data") or {})), commit=True)

    # --- reads ---
    def episode(self, episode_id: str) -> dict[str, Any] | None:
        r = self._exec("SELECT * FROM episodes WHERE id=?", (episode_id,))
        return dict(r[0]) if r else None

    def episodes(self, *, task_id: str | None = None, bundle_version: str | None = None) -> list[dict[str, Any]]:
        q, args = "SELECT * FROM episodes WHERE 1=1", []
        if task_id:
            q += " AND task_id=?"; args.append(task_id)
        if bundle_version:
            q += " AND bundle_version=?"; args.append(bundle_version)
        return [dict(r) for r in self._exec(q + " ORDER BY started", args)]

    def calls(self, episode_id: str) -> list[dict[str, Any]]:
        out = []
        for r in self._exec("SELECT * FROM calls WHERE episode_id=? ORDER BY idx", (episode_id,)):
            d = dict(r)
            for k in ("request", "response", "usage"):
                d[k] = json.loads(d[k]) if d[k] else None
            out.append(d)
        return out

    def tool_execs(self, episode_id: str) -> list[dict[str, Any]]:
        out = []
        for r in self._exec("SELECT * FROM tool_execs WHERE episode_id=? ORDER BY id", (episode_id,)):
            d = dict(r)
            d["input"] = json.loads(d["input"]) if d["input"] else None
            d["output"] = json.loads(d["output"]) if d["output"] else None
            out.append(d)
        return out

    def events(self, run_id: str) -> list[dict[str, Any]]:
        out = []
        for r in self._exec("SELECT * FROM events WHERE run_id=? ORDER BY ts, rowid", (run_id,)):
            d = dict(r); d["data"] = json.loads(d["data"] or "{}"); out.append(d)
        return out

    def trace_text(self, episode_id: str, max_chars: int = 3000) -> str:
        """Compact, human-readable trace of one episode."""
        lines = []
        tools = self.tool_execs(episode_id)
        for c in self.calls(episode_id):
            r = c["response"] or {}
            for t in r.get("text", []):
                if t:
                    lines.append(f"model: {str(t)[:200]}")
            for tc in r.get("tool_calls", []):
                lines.append(f"call {tc.get('name')}({str(tc.get('input'))[:160]})")
            for t in [t for t in tools if t["call_index"] == c["idx"]]:
                lines.append(f"  -> {t['name']} error={t['error']} output={str(t['output'])[:160]}")
        return "\n".join(lines)[:max_chars]

    def usage_between(self, t0: float, t1: float) -> dict[str, dict[str, int]]:
        """Agent token usage for calls made in [t0, t1], by model (learning-cost accounting)."""
        out: dict[str, dict[str, int]] = {}
        for model, usage in self._exec("SELECT model, usage FROM calls WHERE ts >= ? AND ts <= ?", (t0, t1)):
            u = json.loads(usage) if usage else {}
            d = out.setdefault(model or "unknown", {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0})
            d["prompt_tokens"] += int(u.get("prompt_tokens", 0) or 0)
            d["completion_tokens"] += int(u.get("completion_tokens", 0) or 0)
            d["calls"] += 1
        return out
