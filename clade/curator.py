"""Curator: turn diagnoses into bounded candidate bundles, one component per candidate, no wholesale rewrites.

  playbook  append the reflector's fix as one rule (semantic duplicates are skipped; past a cap the weakest goes)
  prompt    append one short section (at most max_lines lines) to the system prompt
  tool      rewrite the description of ONE tool the agent misused (bounded length)
  combined  all of this round's new rules together (cheap to test, often wins)

Ablations: `edits` limits which edit types are proposed; `rewrite=True` replaces bounded edits with one whole-prompt
rewrite per diagnosis, the mutation GEPA-style optimizers use. The defaults are the method.
"""

from __future__ import annotations

import re
import uuid

from .bundle import Bundle, PlaybookEntry
from .llm import LLM, extract_json
from .reflector import Diagnosis

SYSTEM = """You improve the instructions of an AI agent. You make ONE small, targeted edit.
Never rewrite the whole prompt. Prefer concrete, checkable steps over vague advice."""

PROMPT = """Current system prompt:
---
{prompt}
---
Current playbook rules:
{playbook}

Diagnosis of a failure: {diagnosis}
Suggested fix: {fix}

Write ONE new section to append to the system prompt that prevents this failure.
At most {max_lines} lines. Return JSON: {{"section": "<markdown text>"}}"""

TOOL_PROMPT = """The agent's tools and their current descriptions:
{tools}

Diagnosis of a failure: {diagnosis}
Suggested fix: {fix}

If the failure came from how the agent used one of these tools (wrong tool, too narrow a query, missed a constraint the
tool should have reminded it of), rewrite that ONE tool's description so the mistake is less likely. Keep what the tool
does, add the guidance, at most {max_chars} characters, no task-specific names or values. If no tool description would
help, answer {{"tool": null}}. Return JSON: {{"tool": "<tool name>", "description": "<new description>"}}"""

REWRITE_SYSTEM = """You improve the instructions of an AI agent. You rewrite its whole system prompt so the failure
described does not happen again, keeping everything in the prompt that is still correct."""

REWRITE_PROMPT = """Current system prompt:
---
{prompt}
---

Diagnosis of a failure: {diagnosis}
Suggested fix: {fix}

Write the complete new system prompt. Return JSON: {{"prompt": "<the full new system prompt>"}}"""

DEDUPE_SYSTEM = "You compare two instructions for an AI agent. Answer with exactly YES or NO."
DEDUPE_PROMPT = "Do these two rules instruct the same behavior (one would be redundant given the other)?\nA: {a}\nB: {b}"

EDIT_TYPES = ("playbook", "prompt", "tool", "combined")


def _dedupe(text: str, existing: list[str], llm: LLM | None = None) -> bool:
    """Exact match always; semantic match via a one-word judge call when an LLM is available."""
    t = text.lower().strip()
    if any(t == e.lower().strip() for e in existing):
        return True
    if llm is None:
        return False
    for e in existing:
        try:
            ans = llm(DEDUPE_SYSTEM, DEDUPE_PROMPT.format(a=e, b=text)).strip().upper()
        except Exception:  # noqa: BLE001
            return False
        if ans.startswith("YES"):
            return True
    return False


def prune_playbook(bundle: Bundle, *, cap: int = 8) -> list[str]:
    """Drop rules whose harmful count exceeds helpful, then enforce the cap by lowest (helpful - harmful).
    Returns the ids removed."""
    removed = [e.id for e in bundle.playbook if e.harmful > e.helpful]
    bundle.playbook = [e for e in bundle.playbook if e.harmful <= e.helpful]
    while len(bundle.playbook) > cap:
        worst = min(bundle.playbook, key=lambda e: (e.helpful - e.harmful, -len(e.text)))
        bundle.playbook.remove(worst); removed.append(worst.id)
    return removed


def tool_candidate(parent: Bundle, diagnoses: list[Diagnosis], llm: LLM, tool_specs: list[dict], *, max_chars: int = 400) -> Bundle | None:
    """One bounded rewrite of one tool's description, when a diagnosis is about tool use."""
    names = {t.get("name"): t.get("description", "") for t in tool_specs if t.get("name")}
    d = next((x for x in diagnoses if x.cluster == "tools" or any(n in (x.diagnosis + " " + x.fix) for n in names)), None)
    if d is None or not names:
        return None
    current = {n: parent.tools.get(n, desc) for n, desc in names.items()}
    data = extract_json(llm(SYSTEM, TOOL_PROMPT.format(tools="\n".join(f"- {n}: {desc}" for n, desc in current.items()),
                                                       diagnosis=d.diagnosis, fix=d.fix, max_chars=max_chars))) or {}
    name, desc = (data.get("tool"), str(data.get("description") or "").strip()) if isinstance(data, dict) else (None, "")
    if name not in names or not desc or desc == current[name] or len(desc) > max_chars or re.search(r"`|\b[A-Z]\d{3,}\b", desc):
        return None  # unknown tool, no change, too long, or quoting identifiers such as order ids
    c = parent.derive(f"[tool] {name}: {d.diagnosis}")
    c.tools = {**c.tools, name: desc}
    return c.finalize()


def rewrite_candidates(parent: Bundle, diagnoses: list[Diagnosis], llm: LLM, *, k: int = 3) -> list[Bundle]:
    """Ablation: one whole-prompt rewrite per diagnosis (up to k), no playbook or tool edits."""
    cands: list[Bundle] = []
    for d in diagnoses[:k]:
        data = extract_json(llm(REWRITE_SYSTEM, REWRITE_PROMPT.format(prompt=parent.prompt, diagnosis=d.diagnosis, fix=d.fix))) or {}
        text = str(data.get("prompt") or "").strip() if isinstance(data, dict) else ""
        if not text or text == parent.prompt.strip():
            continue
        c = parent.derive(f"[rewrite] {d.cluster}: {d.diagnosis}")
        c.prompt = text + "\n"
        cands.append(c.finalize())
    return cands


def curate(parent: Bundle, diagnoses: list[Diagnosis], llm: LLM | None, *, k: int = 3, max_lines: int = 6,
           playbook_cap: int = 8, tool_specs: list[dict] | None = None, edits: tuple[str, ...] = EDIT_TYPES,
           rewrite: bool = False) -> list[Bundle]:
    unknown = set(edits) - set(EDIT_TYPES)
    if unknown:
        raise ValueError(f"unknown edit types {sorted(unknown)}; choose from {EDIT_TYPES}")
    if rewrite:
        return rewrite_candidates(parent, diagnoses, llm, k=k) if llm else []
    cands: list[Bundle] = []
    rules = [e.text for e in parent.playbook]  # grows with this round's rules, so later ones are deduped against them

    # 1) playbook candidates: the reflector's fix, verbatim, as one rule
    added: list[tuple[PlaybookEntry, Bundle]] = []
    for d in (diagnoses if "playbook" in edits else []):
        if len(cands) >= k:
            break
        rule = d.fix.strip().replace("\n", " ")[:300]
        if not rule or _dedupe(rule, rules, llm):
            continue
        c = parent.derive(f"[playbook] {d.cluster}: {d.diagnosis}")
        entry = PlaybookEntry(id="pb_" + uuid.uuid4().hex[:8], text=rule)
        c.playbook.append(entry)
        prune_playbook(c, cap=playbook_cap)  # past the cap, the weakest rule makes room
        cands.append(c.finalize())
        added.append((entry, cands[-1]))
        rules.append(rule)

    # 2) one prompt-section candidate from the cross-cutting (or first) diagnosis
    if "prompt" in edits and llm and diagnoses and len(cands) < k + 1:
        d = next((x for x in diagnoses if x.cluster == "all"), diagnoses[0])
        data = extract_json(llm(SYSTEM, PROMPT.format(
            prompt=parent.prompt, playbook="\n".join(f"- {e}" for e in rules) or "(none)",
            diagnosis=d.diagnosis, fix=d.fix, max_lines=max_lines))) or {}
        section = data.get("section", "") if isinstance(data, dict) else ""
        lines = [l for l in section.strip().splitlines() if l.strip()][:max_lines]
        if lines:
            c = parent.derive(f"[prompt] {d.cluster}: {d.diagnosis}")
            c.prompt = parent.prompt.rstrip() + "\n\n" + "\n".join(lines) + "\n"
            cands.append(c.finalize())

    # 3) one tool-description candidate when a diagnosis is about tool use
    if "tool" in edits and llm and diagnoses and tool_specs:
        tc = tool_candidate(parent, diagnoses, llm, tool_specs)
        if tc:
            cands.append(tc)

    # 4) combined candidate: all new playbook rules together
    if "combined" in edits and len(added) > 1:
        c = parent.derive("[playbook] combined: " + "; ".join(x.lock["diagnosis"][11:60] for _, x in added))
        c.playbook += [e for e, _ in added]
        prune_playbook(c, cap=playbook_cap)
        cands.append(c.finalize())
    return cands
