"""Optimizers. Each is `optimize(ctx) -> Bundle`: it may run the agent only through ctx.run, with ctx.budget rollouts,
and returns the bundle it would ship. `baseline` ships the starting bundle unchanged."""

from __future__ import annotations

from .clade_opt import ABLATIONS

COMPETITORS = {"gepa": "gepa_opt", "gepa-tools": "gepa_opt", "ace": "ace_opt", "mipro": "dspy_opt", "mipro-demos": "dspy_opt",
               "simba": "dspy_opt", "textgrad": "textgrad_opt"}
METHODS = ["baseline", *ABLATIONS, *COMPETITORS]


def get(name: str):
    if name == "baseline":
        return lambda ctx: ctx.start
    if name in ABLATIONS:
        from .clade_opt import optimize
        return optimize
    if name in COMPETITORS:
        import importlib
        return importlib.import_module(f".{COMPETITORS[name]}", __package__).optimize
    raise KeyError(f"unknown method {name!r}; choose from {METHODS}")
