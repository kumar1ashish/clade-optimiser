"""ACE, offline adaptation (Zhang et al., "Agentic Context Engineering", ICLR 2026), ported from the official code
(ace-agent/ace, ace/ace.py `_train_single_sample` and `_offline_train`) with its prompts read from the pinned clone.

Per training task, batch size 1: run the agent with the playbook; on failure reflect, tag the cited bullets, and
retry with the reflection (up to `ROUNDS` rounds); on success reflect once for tagging. The curator then adds bullets
(ADD only, merged deterministically, as in the official code). After each epoch the playbook is scored on the
validation tasks and the best one is kept. Official defaults: temperature 0 for every role, 3 reflection rounds,
de-duplication off. Two adaptations for a black-box agent: the agent is asked to cite bullet ids so they can be
tagged, and the environment feedback is the benchmark's verdict (the no-ground-truth prompt variant).
"""

from __future__ import annotations

import ast
import json
import random
import re
from pathlib import Path

from clade.bundle import Bundle

from ..common import ROOT, BudgetExhausted, Ctx, make_llm

ACE = ROOT / "bench" / "vendor" / "ace"
ROUNDS, EPOCHS, VAL_REPEATS, TOKEN_BUDGET = 3, 5, 2, 80000
SECTIONS = ["STRATEGIES & INSIGHTS", "FORMULAS & CALCULATIONS", "CODE SNIPPETS & TEMPLATES", "COMMON MISTAKES TO AVOID",
            "PROBLEM-SOLVING HEURISTICS", "CONTEXT CLUES & INDICATORS", "OTHERS"]
SLUGS = ["str", "calc", "code", "err", "prob", "ctx", "misc"]  # bullet id prefixes, one per section, in order
CITE = re.compile(r"\[([a-z]{3,}-\d{5})\]")
CITE_HINT = "When a playbook bullet informs an action, cite its id, e.g. [str-00001]."


def prompt(module: str, name: str) -> str:
    """A prompt constant from the official repository, read without executing it."""
    tree = ast.parse((ACE / "ace" / "prompts" / f"{module}.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(f"{name} not found in ace/prompts/{module}.py (run bench/setup.sh)")


def parse_object(text: str) -> dict:
    """The official extract_json_from_text: the whole reply, else a ```json block, else the first JSON object in it."""
    for cand in [text.strip(), *re.findall(r"```json\s*(.*?)\s*```", text, re.S | re.I)]:
        try:
            data = json.loads(cand)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            data, _ = dec.raw_decode(text, m.start())
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue
    return {}


def norm(s: str) -> str:
    return s.lower().strip().replace("&", "and").replace("-", "_").replace(" ", "_").replace("__", "_")


class Playbook:
    def __init__(self):
        self.sec = {norm(s): (s, []) for s in SECTIONS}
        self.slug = {norm(s): slug for s, slug in zip(SECTIONS, SLUGS)}
        self.next_id = 1

    def bullets(self):
        return [b for _, bs in self.sec.values() for b in bs]

    def render(self) -> str:
        out = []
        for hdr, bs in self.sec.values():
            out += [f"## {hdr}", *[f"[{b['id']}] helpful={b['h']} harmful={b['x']} :: {b['c']}" for b in bs], ""]
        return "\n".join(out).strip()

    def tag(self, tags):
        m = {t.get("id") or t.get("bullet_id"): t.get("tag") for t in tags if isinstance(t, dict)}
        for b in self.bullets():
            b["h"] += m.get(b["id"]) == "helpful"
            b["x"] += m.get(b["id"]) == "harmful"

    def add(self, section: str, content: str) -> None:
        key = norm(section) if norm(section) in self.sec else "others"
        self.sec[key][1].append({"id": f"{self.slug[key]}-{self.next_id:05d}", "h": 0, "x": 0, "c": content})
        self.next_id += 1

    def subset(self, ids) -> str:
        if not ids:
            return "(No bullets used by generator)"
        bs = [b for b in self.bullets() if b["id"] in set(ids)]
        return "\n".join(f"[{b['id']}] helpful={b['h']} harmful={b['x']} :: {b['c']}" for b in bs) or "(Generator referenced bullet IDs but none were found in playbook)"

    def stats(self) -> dict:
        bs = self.bullets()
        return {"total_bullets": len(bs), "high_performing": sum(b["h"] > 5 and b["x"] < 2 for b in bs),
                "problematic": sum(b["x"] >= b["h"] and b["x"] > 0 for b in bs), "unused": sum(b["h"] + b["x"] == 0 for b in bs)}


def optimize(ctx: Ctx) -> Bundle:
    reflector_t, curator_t = prompt("reflector", "REFLECTOR_PROMPT_NO_GT"), prompt("curator", "CURATOR_PROMPT_NO_GT")
    llm = make_llm(ctx.models.optimizer, temperature=0.0)
    rng = random.Random(ctx.seed)
    pb = Playbook()
    val = [t for t in ctx.val for _ in range(VAL_REPEATS)]

    def bundle(text: str, reflection: str | None = None) -> Bundle:
        full = f"{ctx.start.prompt.rstrip()}\n\n{CITE_HINT}\n\n# PLAYBOOK\n{text}"
        return ctx.candidate(prompt=full + (f"\n\n# REFLECTION ON YOUR PREVIOUS ATTEMPT\n{reflection}" if reflection else ""))

    def reflect(task, ep) -> str:
        traj = ctx.render(ep)
        fb = f"Reward {ep.score or 0:.2f} ({'success' if (ep.score or 0) >= 1 else 'failure'}). {ctx.feedback(ep)}"
        out = llm("", reflector_t.format(task.prompt, traj, (ep.meta or {}).get("final_text", "")[:1000], fb,
                                         pb.subset(list(dict.fromkeys(CITE.findall(traj))))))
        tags = parse_object(out).get("bullet_tags", [])
        pb.tag(tags if isinstance(tags, list) else [])
        return out  # the curator reads the reflector's raw reply, as in the official code

    remaining, best = EPOCHS, None  # one validation pass after each epoch; the best playbook is kept

    def validate(epoch: int) -> None:
        nonlocal remaining, best
        remaining -= 1
        b = bundle(pb.render())
        eps = ctx.run_many([(t, b) for t in val])
        s = sum(float(e.score or 0) for e in eps if e) / len(val)
        ctx.log(f"ace: epoch {epoch}, {len(pb.bullets())} bullets, validation {s:.3f}")
        if best is None or s > best[0]:
            best = (s, pb.render())

    epoch = 0
    try:
        for epoch in range(1, EPOCHS + 1):
            order = list(ctx.train); rng.shuffle(order)
            for step, task in enumerate(order, 1):
                if ctx.used + 1 + remaining * len(val) > ctx.budget:
                    raise BudgetExhausted
                ep = ctx.run(task, bundle(pb.render()))
                refl = None
                if (ep.score or 0) < 1:
                    for _ in range(ROUNDS):
                        refl = reflect(task, ep)
                        if ctx.used + 1 + remaining * len(val) > ctx.budget:
                            break
                        ep = ctx.run(task, bundle(pb.render(), refl))
                        if (ep.score or 0) >= 1:
                            break
                else:
                    refl = reflect(task, ep)
                ops = parse_object(llm("", curator_t.format(token_budget=TOKEN_BUDGET, current_step=step, total_samples=len(order),
                                                       playbook_stats=json.dumps(pb.stats(), indent=2), recent_reflection=refl,
                                                       current_playbook=pb.render(), question_context=task.prompt))).get("operations", [])
                for op in ops if isinstance(ops, list) else []:
                    if isinstance(op, dict) and op.get("type") == "ADD" and op.get("content"):
                        pb.add(str(op.get("section", "others")), str(op["content"]))
            validate(epoch)
    except BudgetExhausted:
        if remaining > 0:
            validate(epoch)
    return bundle(best[1]) if best else ctx.start
