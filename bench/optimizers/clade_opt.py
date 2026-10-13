"""Clade, in the configuration fixed before the study (bench/PROTOCOL.md), and its ablations."""

from __future__ import annotations

from clade.bundle import Bundle

from ..common import Ctx

CONFIG = dict(rounds=3, n=2, k=3, workers=3, patience=2, critic=True, lenses=True)

ABLATIONS = {
    "clade": {},
    "clade-no-critic": {"critic": False},                                  # every fix costs episodes
    "clade-no-tools": {"edits": ("playbook", "prompt", "combined")},      # no tool-description edits
    "clade-no-playbook": {"edits": ("prompt", "tool")},                    # sections and tools only
    "clade-rewrite": {"rewrite": True},                                    # whole-prompt rewrites, not bounded edits
    "clade-greedy": {"parent_policy": "incumbent", "lenses": False},       # one lane from the incumbent, no archive
    "clade-no-lenses": {"lenses": False},                                  # extra lanes only from archive parents
    "clade-fixed-gate": {"gate": "fixed"},                                 # regression gate not adjusted for noise
    "clade-no-gate": {"gate": "none"},                                     # no per-task regression gate
    "clade-no-filter": {"task_filter": False},                             # task-specific fixes allowed
    "clade-no-holdout": {},                                                # selection on training tasks (see below)
}


def optimize(ctx: Ctx) -> Bundle:
    cfg = {**CONFIG, **ABLATIONS[ctx.method], **ctx.params}
    practice = ctx.train + ctx.val
    if ctx.method == "clade-no-holdout":  # validation tasks join training: the reflector reads them, selection is on them
        practice = [type(t)(**{**t.__dict__, "held_out": False}) for t in practice]
    per_eval = len(practice) * cfg["n"]
    tr = ctx.tr
    # a ceiling, not a target: the next evaluation is started only if it fits in what is left
    budget = lambda: (ctx.budget - tr.rollouts - per_eval) / ctx.budget  # noqa: E731
    return tr.optimize(practice, llm=ctx.llm, budget=budget, log=ctx.log, **cfg)
