"""Reflector: read failed (and one successful) training episodes, write one diagnosis and one general fix each.

Never sees held-out episodes. One diagnosis per failing task (up to max_clusters) plus a cross-cutting one when
several tasks fail. A fix that names task-specific identifiers is rejected: rules must transfer to unseen tasks.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .llm import LLM, extract_json
from .store import Store

SYSTEM = """You diagnose why an AI agent failed tasks. You read its trajectory: model messages, tool calls,
tool outputs, and the final verdict. Find the PROCESS mistake (what the agent did or skipped, or which rule or
condition it failed to apply), not just the symptom. Be concrete and short."""

PROMPT = """Agent system prompt in use:
---
{prompt}
---

Failed episodes:
{traces}

{successes}

The fix must be a GENERAL working habit or rule that would help on ANY task of this kind, including tasks
you have not seen. It may name the agent's tools and state a general condition the agent missed, but must NOT
mention names, ids, file names, variable names or expected values from these specific tasks.
Bad: "Implement column_mean". Good: "Replace every stub with a real implementation and run the tests before finishing."
Bad: "Rebook the passenger on flight UA123". Good: "Before changing a booking, look up the fare rules for change fees
and tell the traveller the cost before acting."

Return JSON: {{"diagnosis": "<one or two sentences: the root process mistake>",
"evidence": "<what in the traces shows it>",
"fix": "<one general imperative instruction, no task-specific identifiers>"}}"""

LENSES = {
    "per-task": "",
    "contrast": "\n\nLens: compare the failed episodes with the successful one. Name the habit the success shows and the failures lack.",
    "tools": "\n\nLens: look only at how the agent used its tools: wrong tool for the job, malformed arguments, or carrying on after a tool error.",
}

INFRA_ERRORS = ("HTTPError", "URLError", "ConnectionError", "TimeoutError", "RemoteDisconnected", "ConnectionResetError", "socket.timeout")

STATS = {"proposed": 0, "rejected_task_specific": 0, "unparseable": 0}


@dataclass
class Diagnosis:
    cluster: str
    diagnosis: str
    evidence: str
    fix: str
    episode_ids: list[str]


def render_episode(store: Store, ep: dict, max_chars: int = 2500) -> str:
    lines = [f"[task {ep['task_id']}] score={ep['score']} error={ep['error']}"]
    tools = store.tool_execs(ep["id"])
    for c in store.calls(ep["id"]):
        r = c["response"] or {}
        if r.get("text"):
            lines.append(f"  model says: {' '.join(r['text'])[:300]}")
        for tc in r.get("tool_calls", []):
            lines.append(f"  model calls {tc['name']}({str(tc['input'])[:200]})")
        for t in [t for t in tools if t["call_index"] == c["idx"]]:
            lines.append(f"    -> {t['name']} error={t['error']} output={str(t['output'])[:200]}")
    meta = json.loads(ep["meta"] or "{}")
    head, tail = lines[0], [f"  final: {meta.get('final_text', '')[:200]}", f"  judge: {(ep['judge_rationale'] or '')[-500:]}"]
    text = "\n".join([head, *lines[1:], *tail])
    if len(text) <= max_chars:
        return text
    # Too long: keep the task line and the verdict, and as much of the start and end of the trajectory as fits.
    gap = "\n  [... middle of the episode omitted ...]\n"
    room = max_chars - len(head) - len("\n".join(tail)) - 2 - len(gap)
    body = "\n".join(lines[1:])
    middle = body[:room // 2] + gap + body[len(body) - (room - room // 2):] if room > 0 else gap.strip("\n")
    return "\n".join([head, middle, *tail])[:max_chars]


def is_task_specific(fix: str, store: Store, episodes: list[dict]) -> bool:
    """A fix leaks the task when it quotes code or names an identifier seen in the episodes' tool traffic. The
    agent's own tool names are general vocabulary, not leaks, so they are allowed, quoted or not."""
    tools = {t["name"] for e in episodes for t in store.tool_execs(e["id"])}
    for name in tools:
        fix = fix.replace(f"`{name}`", name).replace(f"{name}(", name)
    if re.search(r"`[^`]+`", fix):
        return True
    words = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", fix))
    idents: set[str] = set()
    for e in episodes:
        for t in store.tool_execs(e["id"]):
            idents |= set(re.findall(r"(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", str(t["input"]) + str(t["output"])))
    generic = {"pytest", "test", "tests", "python", "assert"} | tools
    return bool((words & idents) - generic) or bool(re.search(r"\b\w+_\w+\(", fix))


def is_infra_error(ep: dict) -> bool:
    """An episode lost to the model endpoint says nothing about the agent's process."""
    return str(ep.get("error") or "").startswith(INFRA_ERRORS)


def reflect(store: Store, llm: LLM, *, bundle_version: str, prompt: str, train_task_ids: set[str],
            max_clusters: int = 4, lens: str = "per-task", task_filter: bool = True) -> list[Diagnosis]:
    """lens: "per-task" (one diagnosis per failing task plus a cross-cutting one), "contrast" (one cross-cutting
    diagnosis against successes) or "tools" (failures whose traces contain tool errors). Lanes that share a parent
    use different lenses so they do not propose the same fix. task_filter=False (ablation only) keeps fixes that
    leak task identifiers."""
    eps = [e for e in store.episodes(bundle_version=bundle_version) if e["task_id"] in train_task_ids]
    fails = [e for e in eps if (e["score"] or 0) < 1.0 and not is_infra_error(e)]
    wins = [e for e in eps if (e["score"] or 0) >= 1.0]
    by_task: dict[str, list[dict]] = {}
    for e in fails:
        by_task.setdefault(e["task_id"], []).append(e)

    if lens == "contrast":
        clusters = [("contrast", [v[0] for v in by_task.values()][:4])] if by_task else []
    elif lens == "tools":
        tool_fails = [e for e in fails if any(t.get("error") for t in store.tool_execs(e["id"]))]
        clusters = [("tools", tool_fails[:4])] if tool_fails else []
    else:
        clusters = list(by_task.items())[:max_clusters]
        if len(by_task) > 1:  # cross-cutting cluster: one failure from each task
            clusters.append(("all", [v[0] for v in by_task.values()][:4]))

    succ = "For contrast, a successful episode:\n" + render_episode(store, wins[0], 1500) if wins else ""

    out: list[Diagnosis] = []
    for name, group in clusters:
        wide = name in ("all", "contrast", "tools")
        traces = "\n\n".join(render_episode(store, e, 1200 if wide else 2500) for e in group[:4 if wide else 2])
        data = extract_json(llm(SYSTEM, PROMPT.format(prompt=prompt, traces=traces, successes=succ) + LENSES.get(lens, ""))) or {}
        STATS["proposed"] += 1
        if not isinstance(data, dict) or not data.get("fix"):
            STATS["unparseable"] += 1; continue
        if task_filter and is_task_specific(str(data["fix"]), store, group):
            STATS["rejected_task_specific"] += 1; continue
        out.append(Diagnosis(name, str(data.get("diagnosis", "")), str(data.get("evidence", "")),
                             str(data["fix"]), [e["id"] for e in group]))
    return out
