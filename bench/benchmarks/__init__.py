"""Benchmarks. Each one provides:

    splits()                          -> (train, val, test) task lists; val tasks have held_out=True
    start()                           -> the starting bundle directory, shared by every method
    trainable(bundle_dir, models, db) -> a StudyTrainable that runs and judges one episode per call
    feedback(episode)                 -> the judge's verdict as text, the same for every method
    budget, test_n                    -> default learning rollouts per run, runs per test task
"""

from __future__ import annotations


def get(name: str):
    if name == "support-desk":
        from .support_desk import SupportDesk
        return SupportDesk()
    if name.startswith("tau2-"):
        from .tau2 import Tau2
        return Tau2(name.removeprefix("tau2-"))
    raise KeyError(f"unknown benchmark {name!r}; choose support-desk or tau2-<airline|retail|telecom>")
