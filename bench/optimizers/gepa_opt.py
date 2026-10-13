"""GEPA (Agrawal et al., ICLR 2026) through the official `gepa` library and its own reflective proposer.

gepa:        the system prompt is the only component, as in GEPA's own experiments
gepa-tools:  the system prompt and every tool description are components (round-robin), the same search space
             Clade edits, so the comparison separates the algorithm from the search space

The reflective dataset holds, per episode, exactly what Clade's reflector reads (Ctx.render) plus the verdict.
"""

from __future__ import annotations

import gepa
from gepa.core.adapter import EvaluationBatch

from clade.bundle import Bundle

from ..common import Ctx

SYSTEM = "You improve the instructions of an AI agent."
VAL_REPEATS = 2  # each validation task twice, as Clade evaluates held-out tasks with n=2


def optimize(ctx: Ctx) -> Bundle:
    tools = ctx.bench.tool_descriptions()
    seed = {"system_prompt": ctx.start.prompt}
    if ctx.method == "gepa-tools":
        seed |= {f"tool:{name}": desc for name, desc in tools.items()}

    def to_bundle(cand: dict) -> Bundle:
        changed = {k[5:]: v for k, v in cand.items() if k.startswith("tool:") and v != tools[k[5:]]}
        return ctx.candidate(prompt=cand["system_prompt"], tools={**ctx.start.tools, **changed})

    class Adapter:
        propose_new_texts = None  # GEPA's own reflective proposal

        def evaluate(self, batch, candidate, capture_traces=False):
            b = to_bundle(candidate)
            eps = ctx.run_many([(task, b) for task, _ in batch])
            scores = [float(e.score or 0.0) if e else 0.0 for e in eps]
            outs = [{"final": (e.meta or {}).get("final_text", "") if e else ""} for e in eps]
            trajs = None
            if capture_traces:
                trajs = [{"task": task.prompt, "render": ctx.render(e) if e else "(not run: budget spent)",
                          "feedback": ctx.feedback(e) if e else "", "score": s} for (task, _), e, s in zip(batch, eps, scores)]
            return EvaluationBatch(outputs=outs, scores=scores, trajectories=trajs)

        def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
            recs = [{"Inputs": t["task"], "Generated Outputs": t["render"],
                     "Feedback": "Correct." if t["score"] >= 1 else f"Incorrect: {t['feedback']}"}
                    for t in eval_batch.trajectories or []]
            return {c: recs for c in components_to_update}

    valset = [(t, r) for t in ctx.val for r in range(VAL_REPEATS)]
    ctx.budget += len(valset)  # GEPA checks its budget between iterations, so one validation pass may run over
    res = gepa.optimize(seed_candidate=seed, trainset=[(t, 0) for t in ctx.train], valset=valset, adapter=Adapter(),
                        reflection_lm=lambda prompt: ctx.llm(SYSTEM, prompt), max_metric_calls=ctx.budget - len(valset),
                        reflection_minibatch_size=3, seed=ctx.seed, display_progress_bar=False, raise_on_exception=False,
                        run_dir=str(ctx.workdir / "gepa"))
    ctx.log(f"gepa: {len(res.candidates)} candidates, best val {max(res.val_aggregate_scores):.3f}")
    return to_bundle(res.best_candidate)
