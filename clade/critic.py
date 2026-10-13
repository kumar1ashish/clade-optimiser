"""Critic: a second opinion on the reflector's proposed fixes before any episode is spent on them.

A fix is kept when it is a general working habit, actionable, and not already covered by the playbook.
Disagreement here is cheap (one model call per round and lane); evaluating a candidate costs episodes.
The critic fails open: if its answer cannot be parsed, every fix goes through.
"""

from __future__ import annotations

from .llm import LLM, extract_json
from .reflector import Diagnosis

SYSTEM = """You review proposed fixes to an AI agent's instructions before they are tested. For each fix decide
KEEP or DROP. Keep a fix only if it is (1) a general working habit, not a hint about one task, (2) something the
agent can act on, and (3) not already said by the existing playbook. Be strict about duplicates and about
task-specific leaks (names, ids, file names, expected values from specific tasks). The agent's own tool names are fine."""

PROMPT = """Existing playbook rules:
{playbook}

Proposed fixes:
{fixes}

Return JSON: {{"verdicts": [{{"index": <int>, "keep": <true|false>, "reason": "<short>"}}]}}"""


def critique(diagnoses: list[Diagnosis], llm: LLM | None, *, playbook: list[str]) -> tuple[list[Diagnosis], list[dict]]:
    if not diagnoses or llm is None:
        return diagnoses, [{"index": i, "keep": True, "reason": "no critic"} for i, _ in enumerate(diagnoses)]
    fixes = "\n".join(f"{i}. {d.fix}" for i, d in enumerate(diagnoses))  # no cluster labels: a task id in brackets reads as a leak
    data = extract_json(llm(SYSTEM, PROMPT.format(playbook="\n".join(f"- {p}" for p in playbook) or "(none)", fixes=fixes))) or {}
    verdicts = data.get("verdicts") if isinstance(data, dict) else (data if isinstance(data, list) else None)  # extract_json may return the inner list
    if not isinstance(verdicts, list):
        return diagnoses, [{"index": i, "keep": True, "reason": "critic unparseable; kept"} for i, _ in enumerate(diagnoses)]
    keep: dict[int, dict] = {}
    for v in verdicts:
        try:
            keep[int(v["index"])] = {"index": int(v["index"]), "keep": bool(v.get("keep", True)), "reason": str(v.get("reason", ""))[:200]}
        except (KeyError, TypeError, ValueError):
            continue
    out_v = [keep.get(i, {"index": i, "keep": True, "reason": "not reviewed; kept"}) for i, _ in enumerate(diagnoses)]
    return [d for i, d in enumerate(diagnoses) if out_v[i]["keep"]], out_v
