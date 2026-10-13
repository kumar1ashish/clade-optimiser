"""DSPy optimizers over the black-box agent: MIPROv2 (Opsahl-Ong et al., EMNLP 2024) and SIMBA (DSPy 3).

The DSPy program has one predictor whose instruction is the agent's system prompt; forward() runs the agent through
Ctx.run instead of calling an LM, and appends itself to the DSPy trace so bootstrapping and SIMBA see a module call.

mipro:        instruction-only (no demos), the configuration GEPA reports as MIPROv2-no-demos
mipro-demos:  up to two bootstrapped successful episodes are shown to the agent with the instruction
simba:        rules only (max_demos=0), selected on the practice tasks (SIMBA has no validation split)

Trials and steps are sized from the rollout budget with DSPy's own accounting, so every method gets the same budget.
"""

from __future__ import annotations

import math

import dspy
from dspy.clients.base_lm import GLOBAL_HISTORY
from dspy.teleprompt import MIPROv2, SIMBA

from clade.bundle import Bundle
from clade.llm import account

from ..common import BudgetExhausted, Ctx, endpoint

VAL_REPEATS = 2


class _Ref:
    """DSPy deep-copies programs; the run context must be shared, not copied."""

    def __init__(self, obj):
        self.obj = obj

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


class AgentEpisode(dspy.Signature):
    """(replaced by the starting system prompt)"""

    task: str = dspy.InputField(desc="the request the tool-using agent is given")
    trajectory: str = dspy.OutputField(desc="the agent's conversation, tool calls, results and the judge's verdict")


def render_demos(demos, max_chars: int = 1500) -> str:
    if not demos:
        return ""
    parts = [f"Task: {d.task}\nSuccessful episode:\n{str(d.trajectory)[:max_chars]}" for d in demos]
    return "\n\n# Examples of successful episodes\n\n" + "\n\n---\n\n".join(parts)


class BlackBoxAgent(dspy.Module):
    def __init__(self, ctx: Ctx, tasks: dict):
        super().__init__()
        self.ref, self.tasks = _Ref(ctx), _Ref(tasks)
        self.policy = dspy.Predict(AgentEpisode.with_instructions(ctx.start.prompt))

    def bundle(self) -> Bundle:
        return self.ref.obj.candidate(prompt=self.policy.signature.instructions + render_demos(self.policy.demos))

    def forward(self, task: str, task_id: str):
        ctx = self.ref.obj
        try:
            ep = ctx.run(self.tasks.obj[task_id], self.bundle())
            pred = dspy.Prediction(trajectory=ctx.render(ep), score=float(ep.score or 0.0), feedback=ctx.feedback(ep))
        except BudgetExhausted:
            pred = dspy.Prediction(trajectory="(not run: budget spent)", score=0.0, feedback="not run")
        if dspy.settings.trace is not None:  # make the call visible to bootstrapping and SIMBA
            dspy.settings.trace.append((self.policy, {"task": task}, pred))
        return pred


def mipro_trials(budget: int, n_train: int, n_val: int, minibatch: int, every: int) -> int:
    """Largest number of trials whose rollouts fit: bootstrap + full validation passes + minibatch trials."""
    def rollouts(t):
        full = 1 + math.ceil(t / every) if minibatch else t + 1
        return n_train + n_val * full + (t * minibatch if minibatch else 0)
    t = 1
    while rollouts(t + 1) <= budget:
        t += 1
    return t


def optimize(ctx: Ctx) -> Bundle:
    base, key, model = endpoint(ctx.models.optimizer)
    lm = dspy.LM(f"openai/{model}", api_base=base, api_key=key, temperature=1.0, max_tokens=8000, cache=False, num_retries=8)
    dspy.configure(lm=lm)
    practice = ctx.train + ctx.val
    tasks = {t.id: t for t in practice}
    ex = lambda ts: [dspy.Example(task=t.prompt, task_id=t.id).with_inputs("task", "task_id") for t in ts]  # noqa: E731
    student = BlackBoxAgent(ctx, tasks)
    start = len(GLOBAL_HISTORY)
    try:
        if ctx.method.startswith("mipro"):
            val = ex([t for t in ctx.val for _ in range(VAL_REPEATS)])
            mb = 10 if len(val) > 10 else 0
            trials = mipro_trials(ctx.budget, len(ctx.train), len(val), mb, 5)
            demos = 2 if ctx.method == "mipro-demos" else 0
            opt = MIPROv2(metric=lambda e, p, trace=None: p.score, prompt_model=lm, task_model=lm, auto=None,
                          num_candidates=min(12, max(6, trials // 3)), max_bootstrapped_demos=demos, max_labeled_demos=0,
                          metric_threshold=1.0, num_threads=8, max_errors=1000, seed=ctx.seed)
            prog = opt.compile(student, trainset=ex(ctx.train), valset=val, num_trials=trials, minibatch=bool(mb),
                               minibatch_size=mb or 35, minibatch_full_eval_steps=5, program_aware_proposer=False,
                               requires_permission_to_run=False)
            ctx.log(f"{ctx.method}: {trials} trials")
        else:  # simba
            c, bsize = 4, min(10, len(practice))
            steps = max(1, (ctx.budget - (c + 1) * len(practice)) // (bsize * (2 * c + 1)))
            opt = SIMBA(metric=lambda e, p: dspy.Prediction(score=p.score, feedback=p.feedback), bsize=bsize, num_candidates=c,
                        max_steps=steps, max_demos=0, prompt_model=lm, num_threads=8)
            prog = opt.compile(student, trainset=ex(practice), seed=ctx.seed)
            ctx.log(f"simba: {steps} steps")
    finally:
        for h in GLOBAL_HISTORY[start:]:
            account(model, h.get("usage") or {})
    return prog.bundle()
