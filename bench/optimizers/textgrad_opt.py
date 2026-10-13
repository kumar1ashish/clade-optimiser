"""TextGrad (Yuksekgonul et al., Nature 2025): textual gradient descent on the system prompt, with the
validate-and-revert step of its prompt-optimization recipe.

Each step runs a minibatch of training tasks, asks for a critique of each episode (the loss), backpropagates it to the
system prompt and applies one TGD update; the update is kept only if the validation score does not drop. The
optimizer LLM is the same reflection model every method uses.
"""

from __future__ import annotations

import os
import random

os.environ.setdefault("TEXTGRAD_LOG_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "runs", "textgrad"))
import textgrad as tg  # noqa: E402
from textgrad.autograd.string_based_ops import StringBasedFunction  # noqa: E402
from textgrad.engine.base import EngineLM  # noqa: E402

from clade.bundle import Bundle  # noqa: E402

from ..common import Ctx  # noqa: E402

BATCH, VAL_REPEATS = 4, 2


class Engine(EngineLM):
    def __init__(self, llm, name: str):
        self.llm, self.model_string = llm, name

    def generate(self, prompt, system_prompt=None, **kwargs):
        return self.llm(system_prompt or self.system_prompt, prompt if isinstance(prompt, str) else str(prompt))

    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)


def optimize(ctx: Ctx) -> Bundle:
    engine = Engine(ctx.llm, ctx.models.optimizer)
    tg.set_backward_engine(engine, override=True)
    rng = random.Random(ctx.seed)
    tasks = {t.id: t for t in ctx.train}
    val = [t for t in ctx.val for _ in range(VAL_REPEATS)]
    steps = max(1, (ctx.budget - len(val)) // (BATCH + len(val)))

    prompt = tg.Variable(ctx.start.prompt, requires_grad=True, role_description="system prompt of a tool-using agent")
    opt = tg.TGD(parameters=[prompt], engine=engine, constraints=[
        "Keep every instruction that is still correct.", "Write general rules; never mention specific task ids, names or values."])

    def episode(system_prompt, task_id):
        ep = ctx.run(tasks[task_id.value], ctx.candidate(prompt=system_prompt.value))
        return f"SCORE: {ep.score or 0}\nVERDICT: {ctx.feedback(ep)}\nEPISODE:\n{ctx.render(ep)}"

    run = StringBasedFunction(episode, function_purpose="Runs the tool-using agent configured by the system prompt on one task "
                                                        "and returns its score, the judge's verdict and the episode.")
    loss = tg.TextLoss("You review one episode of a tool-using agent. From the score, verdict and episode, state concisely "
                       "which behaviour lost reward and what general instruction in its system prompt would prevent it. "
                       "If the score is 1, say what to keep.")

    def val_score(text: str) -> float:
        b = ctx.candidate(prompt=text)
        return sum(float(e.score or 0) for e in ctx.run_many([(t, b) for t in val]) if e) / len(val)

    best = val_score(prompt.value)
    for step in range(steps):
        if ctx.used + BATCH + len(val) > ctx.budget:
            break
        opt.zero_grad()
        batch = rng.sample(list(tasks), min(BATCH, len(tasks)))
        losses = [loss(run(inputs={"system_prompt": prompt, "task_id": tg.Variable(t, requires_grad=False, role_description="task id")}))
                  for t in batch]
        tg.sum(losses).backward()
        previous = prompt.value
        opt.step()
        score = val_score(prompt.value)
        ctx.log(f"textgrad: step {step + 1}/{steps}, validation {score:.3f} (best {best:.3f})")
        if score < best:
            prompt.set_value(previous)
        else:
            best = score
    return ctx.candidate(prompt=prompt.value)
