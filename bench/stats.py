"""Statistics for the report: paired and hierarchical bootstraps over test tasks and seeds, Holm correction."""

from __future__ import annotations

import random


def paired_bootstrap(a: dict[str, float], b: dict[str, float], iters: int = 10000, seed: int = 0) -> dict:
    """mean(b) - mean(a) over the tasks both were scored on, resampling tasks."""
    keys = sorted(set(a) & set(b))
    rng = random.Random(seed)
    diffs = sorted(sum(b[k] - a[k] for k in ks) / len(ks) for ks in ([rng.choice(keys) for _ in keys] for _ in range(iters)))
    return {"delta": round(sum(b[k] - a[k] for k in keys) / len(keys), 4),
            "ci95": [round(diffs[int(0.025 * iters)], 4), round(diffs[int(0.975 * iters)], 4)],
            "p_le_0": round(sum(d <= 0 for d in diffs) / iters, 4)}


def hierarchical_bootstrap(a_seeds: list[dict[str, float]], b_seeds: list[dict[str, float]], iters: int = 10000,
                           seed: int = 0) -> dict:
    """Difference of two methods' mean test scores when both tasks and seeds vary: resample tasks (shared by both
    methods), then each method's seeds independently."""
    keys = sorted(set.intersection(*(set(s) for s in a_seeds + b_seeds)))
    rng = random.Random(seed)
    diffs = []
    for _ in range(iters):
        ks = [rng.choice(keys) for _ in keys]
        sa = [rng.choice(a_seeds) for _ in a_seeds]
        sb = [rng.choice(b_seeds) for _ in b_seeds]
        diffs.append(sum(s[k] for s in sb for k in ks) / (len(sb) * len(ks)) - sum(s[k] for s in sa for k in ks) / (len(sa) * len(ks)))
    diffs.sort()
    mean = lambda seeds: sum(s[k] for s in seeds for k in keys) / (len(seeds) * len(keys))  # noqa: E731
    return {"delta": round(mean(b_seeds) - mean(a_seeds), 4),
            "ci95": [round(diffs[int(0.025 * iters)], 4), round(diffs[int(0.975 * iters)], 4)],
            "p_le_0": round(sum(d <= 0 for d in diffs) / iters, 4)}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values."""
    out, run, m = {}, 0.0, len(pvals)
    for i, (k, p) in enumerate(sorted(pvals.items(), key=lambda kv: kv[1])):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = round(run, 4)
    return out


def pooled_bootstrap(groups: list[tuple[list[dict[str, float]], list[dict[str, float]]]], iters: int = 10000, seed: int = 0) -> dict:
    """Difference of two methods averaged over several benchmarks, each weighted equally: within each benchmark,
    resample its tasks (shared by both methods) and each method's seeds, then average the per-benchmark differences."""
    rng = random.Random(seed)
    keys = [sorted(set.intersection(*(set(s) for s in a + b))) for a, b in groups]
    mean = lambda seeds, ks: sum(s[k] for s in seeds for k in ks) / (len(seeds) * len(ks))  # noqa: E731
    diffs = []
    for _ in range(iters):
        d = 0.0
        for (a, b), ks in zip(groups, keys):
            kk = [rng.choice(ks) for _ in ks]
            d += mean([rng.choice(b) for _ in b], kk) - mean([rng.choice(a) for _ in a], kk)
        diffs.append(d / len(groups))
    diffs.sort()
    delta = sum(mean(b, ks) - mean(a, ks) for (a, b), ks in zip(groups, keys)) / len(groups)
    return {"delta": round(delta, 4), "ci95": [round(diffs[int(0.025 * iters)], 4), round(diffs[int(0.975 * iters)], 4)],
            "p_le_0": round(sum(x <= 0 for x in diffs) / iters, 4)}
