"""Learning cost: every token spent to learn, both the optimizer's own calls and the agent episodes it ran."""

from __future__ import annotations

from dataclasses import dataclass, field

from .llm import USAGE, _RUN_USAGE
from .store import Store

# Approximate list prices, USD per 1M tokens (input, output). Cache discounts are ignored, the same for every method.
PRICES = {
    "openai/gpt-oss-20b": (0.075, 0.30),
    "openai/gpt-oss-120b": (0.15, 0.60),
    "qwen/qwen3.6-27b": (0.30, 0.60),
}
DEFAULT_PRICE = (0.50, 1.50)  # unknown models are priced conservatively; register local models at (0, 0)


def usd(model: str | None, prompt_tokens: int, completion_tokens: int) -> float:
    if not model:
        return 0.0
    pi, po = PRICES.get(model, DEFAULT_PRICE)
    return (prompt_tokens * pi + completion_tokens * po) / 1e6


@dataclass
class Ledger:
    """Learning cost over a window: optimizer usage from the run's book plus agent usage from the store."""
    llm_before: dict = field(default_factory=dict)
    t0: float = 0.0

    def _book(self) -> dict:
        book = _RUN_USAGE.get()
        return book if book is not None else USAGE

    def start(self, t0: float) -> None:
        self.llm_before, self.t0 = {m: dict(v) for m, v in self._book().items()}, t0

    def close(self, store: Store, t1: float) -> dict:
        roles = {}
        for m, v in list(self._book().items()):
            d = {k: v[k] - self.llm_before.get(m, {}).get(k, 0) for k in v}
            if any(d.values()):
                roles[m] = d
        agent = store.usage_between(self.t0, t1)
        role_tokens = sum(v["prompt_tokens"] + v["completion_tokens"] for v in roles.values())
        agent_tokens = sum(v["prompt_tokens"] + v["completion_tokens"] for v in agent.values())
        spent = sum(usd(m, v["prompt_tokens"], v["completion_tokens"]) for m, v in [*roles.items(), *agent.items()])
        return {"role_tokens": role_tokens, "agent_tokens": agent_tokens, "learning_tokens": role_tokens + agent_tokens,
                "learning_usd_est": round(spent, 4), "by_model": {**roles, **{f"agent:{k}": v for k, v in agent.items()}}}
