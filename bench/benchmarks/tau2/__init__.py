"""τ²-bench (Sierra, MIT): a customer-service agent and a simulated user, scored on the final database state.

The agent is τ²'s own LLMAgent. The bundle replaces only its instruction block (prompt and playbook) and may override
tool descriptions; the domain policy is passed verbatim, so the starting bundle is τ²'s default agent. Practice tasks
come from the official train split (20 train, 10 validation, fixed by seed 0); scoring uses the official test split.

Every model call goes through `_completion`, which retries provider errors, books token usage to the episode, and
lets a Groq tool-call validation error be resampled. The retail domain's natural-language checks are judged by the
user model instead of gpt-4.1, with lenient JSON parsing.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("LOGURU_LEVEL", "CRITICAL")  # τ² logs every call at import and run time; keep the console for the harness
import litellm  # noqa: E402
from tau2.agent.llm_agent import LLMAgent  # noqa: E402
from tau2.environment.toolkit import ToolType  # noqa: E402
from tau2.evaluator import evaluator_nl_assertions as nl  # noqa: E402
from tau2.evaluator.evaluator import EvaluationType  # noqa: E402
from tau2.orchestrator.orchestrator import Orchestrator  # noqa: E402
from tau2.runner import build_environment, build_user, get_tasks, run_simulation  # noqa: E402
from tau2.utils import llm_utils  # noqa: E402

from clade.agent import RunResult, Verdict  # noqa: E402
from clade.cost import usd  # noqa: E402
from clade.llm import RETRY_CODES, extract_json  # noqa: E402
from clade.schema import Task  # noqa: E402

from ...common import RUNS, Models, StudyTrainable, endpoint  # noqa: E402

HERE = Path(__file__).resolve().parent
PRACTICE = (20, 10)   # train, validation tasks drawn from the official train split
litellm.suppress_debug_info = True

_BOOK: contextvars.ContextVar[dict | None] = contextvars.ContextVar("tau2_episode_usage", default=None)
_raw_completion = llm_utils.completion
MALFORMED = ("tool_use_failed", "Tool call validation failed", "Failed to parse tool call arguments")
_PROVIDER_ERRORS = (litellm.RateLimitError, litellm.APIConnectionError, litellm.Timeout, litellm.ServiceUnavailableError,
                    litellm.InternalServerError)


def _completion(**kw):
    """Retries provider errors; resamples (at rising temperature) a reply Groq rejected as a malformed tool call or one
    with neither text nor a tool call, two gpt-oss glitches τ² would otherwise turn into a crashed episode."""
    for attempt in range(8):
        try:
            resp = _raw_completion(**kw)
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if any(m in msg for m in MALFORMED) and attempt < 4:
                kw["temperature"] = max(kw.get("temperature") or 0.0, 0.3 * (attempt + 1))
                continue
            if getattr(e, "status_code", None) == 498 and kw.get("extra_body"):  # no flex capacity: go on demand
                kw.pop("extra_body")
                continue
            if isinstance(e, _PROVIDER_ERRORS) or getattr(e, "status_code", None) in RETRY_CODES:
                if attempt < 7:
                    time.sleep(min(60.0, 2 ** attempt + random.random()))
                    continue
                raise ConnectionError(f"provider failed after retries: {msg[:300]}") from None
            raise
        msg = resp.choices[0].message
        if not (msg.content or "").strip() and not msg.tool_calls and attempt < 4:
            kw["temperature"] = max(kw.get("temperature") or 0.0, 0.3 * (attempt + 1))
            continue
        book = _BOOK.get()
        if book is not None and getattr(resp, "usage", None):
            u = book.setdefault(kw["model"].removeprefix("openai/"), {"prompt_tokens": 0, "completion_tokens": 0})
            u["prompt_tokens"] += int(resp.usage.prompt_tokens or 0)
            u["completion_tokens"] += int(resp.usage.completion_tokens or 0)
        return resp
    return resp


llm_utils.completion = _completion


class _LenientJSON:
    """The NL-assertion judge parses its reply with json.loads; open models sometimes wrap it in prose or fences."""

    @staticmethod
    def loads(text):
        data = extract_json(text or "")
        if data is None:
            raise ConnectionError("judge returned no JSON")  # re-run the episode rather than score a judge failure
        return data if isinstance(data, dict) else {"results": data}


nl.json = _LenientJSON


def litellm_args(spec: str, temperature: float = 0.0) -> tuple[str, dict]:
    base, key, model = endpoint(spec)
    args = {"temperature": temperature, "api_base": base, "api_key": key}
    if os.environ.get("CLADE_SERVICE_TIER") and "groq.com" in base:
        args["extra_body"] = {"service_tier": os.environ["CLADE_SERVICE_TIER"]}
    return f"openai/{model}", args


class BundleAgent(LLMAgent):
    """τ²'s LLMAgent with the bundle's instructions in place of its own, and tool descriptions overridden."""

    def __init__(self, tools, domain_policy, llm, llm_args, instructions: str, tool_descriptions: dict[str, str]):
        for t in tools:
            if t.name in tool_descriptions:
                t.short_desc, t.long_desc = tool_descriptions[t.name], ""
        super().__init__(tools=tools, domain_policy=domain_policy, llm=llm, llm_args=llm_args)
        self.instructions = instructions

    @property
    def system_prompt(self) -> str:
        return f"<instructions>\n{self.instructions}\n</instructions>\n<policy>\n{self.domain_policy}\n</policy>"


def feedback_text(info) -> str:
    """The verdict an optimizer reads: what the evaluator checked and what failed (at most 500 characters)."""
    parts = [f"reward {info.reward:.0f}"]
    if info.db_check is not None and not info.db_check.db_match:
        parts.append("the final database state does not match the expected outcome")
    for a in [a for a in (info.action_checks or []) if not a.action_match][:3]:
        parts.append(f"expected action not taken: {a.action.name}({json.dumps(a.action.arguments, separators=(',', ':'))[:120]})")
    for c in [c for c in (info.communicate_checks or []) if not c.met][:2]:
        parts.append(f"did not tell the user: {c.info[:100]}")
    for c in [c for c in (info.nl_assertions or []) if not c.met][:2]:
        parts.append(f"not satisfied: {c.nl_assertion[:120]}")
    for c in [c for c in (info.env_assertions or []) if not c.met][:2]:
        parts.append(f"environment check failed: {str(c.env_assertion)[:120]}")
    return "; ".join(parts)[:500]


class Tau2Adapter:
    thread_safe = True

    def __init__(self, domain: str, models: Models, tasks: dict):
        self.domain, self.tasks = domain, tasks
        self.agent_llm, self.agent_args = litellm_args(models.agent)
        self.user_llm, self.user_args = litellm_args(models.user)
        nl.DEFAULT_LLM_NL_ASSERTIONS, nl.DEFAULT_LLM_NL_ASSERTIONS_ARGS = self.user_llm, dict(self.user_args)

    def apply(self, agent, bundle, cwd):
        return {"instructions": bundle.effective_prompt(), "tools": dict(bundle.tools)}

    def instrument(self, agent, recorder):
        return {**agent, "rec": recorder}

    def tools(self, agent):
        return [{"name": t.name, "description": t.openai_schema["function"]["description"]} for t in build_environment(self.domain).get_tools()]

    def run(self, agent, task: Task, cwd: Path) -> RunResult:
        tau_task = self.tasks[task.meta["tau2_id"]]
        env = build_environment(self.domain)
        bot = BundleAgent(env.get_tools(), env.get_policy(), self.agent_llm, dict(self.agent_args), agent["instructions"], agent["tools"])
        user = build_user("user_simulator", env, tau_task, llm=self.user_llm, llm_args=dict(self.user_args))
        orch = Orchestrator(domain=self.domain, agent=bot, user=user, environment=env, task=tau_task,
                            max_steps=200, max_errors=10, seed=random.randint(0, 2 ** 31))
        book: dict = {}
        token = _BOOK.set(book)
        try:
            sim = run_simulation(orch, evaluation_type=EvaluationType.ALL)
        except ConnectionError as e:
            return RunResult(error=f"ConnectionError: {e}")
        except Exception as e:  # noqa: BLE001  (as τ²'s own runner does, a crashed episode is re-run, then scored 0)
            return RunResult(error=f"ConnectionError: episode crashed: {type(e).__name__}: {str(e)[:300]}")
        finally:
            _BOOK.reset(token)
        if "infrastructure" in str(sim.termination_reason).lower():
            return RunResult(error=f"ConnectionError: τ² ended the episode: {sim.termination_reason}")
        final = record(agent["rec"], sim.messages, self.agent_llm.removeprefix("openai/"))
        (cwd / "tau2.json").write_text(json.dumps({"reward": sim.reward_info.reward, "feedback": feedback_text(sim.reward_info),
                                                   "termination": str(sim.termination_reason)}))
        cost = sum(usd(m, u["prompt_tokens"], u["completion_tokens"]) for m, u in book.items())
        return RunResult(final_text=final, cost_usd=round(cost, 6), num_turns=sum(m.role == "assistant" for m in sim.messages))


def record(rec, messages, model: str) -> str:
    """Write the conversation to the store: each agent turn is a call; tool results and the customer's replies are
    attached to the turn that prompted them."""
    pending: dict[str, tuple[str, dict]] = {}
    final = ""
    for m in messages:
        if m.role == "assistant":
            calls = [{"id": tc.id, "name": tc.name, "input": tc.arguments} for tc in (m.tool_calls or [])]
            rec.call({"text": [m.content] if m.content else [], "tool_calls": calls, "thinking": []}, model=model, usage=m.usage or {})
            pending |= {tc["id"]: (tc["name"], tc["input"]) for tc in calls}
            final = m.content or final
        elif m.role == "tool":
            name, args = pending.pop(m.id, ("tool", {}))
            if getattr(m, "requestor", "assistant") == "user":
                name = f"customer device: {name}"
            rec.tool(name, args, output=m.content, error="error" if getattr(m, "error", False) else None)
        elif m.role == "user":
            for tc in m.tool_calls or []:
                pending[tc.id] = (tc.name, tc.arguments)
            if m.content:
                rec.tool("customer", {}, output=m.content)
    return final


class Tau2Judge:
    def score(self, task, episode, cwd: Path) -> Verdict:
        f = cwd / "tau2.json"
        if not f.exists():
            return Verdict(0.0, f"episode did not finish: {episode.error}")
        r = json.loads(f.read_text())
        return Verdict(float(r["reward"]), r["feedback"])


class Tau2:
    budget = 600   # learning rollouts per run
    test_n = 4     # runs per test task

    def __init__(self, domain: str):
        self.domain, self.name = domain, f"tau2-{domain}"
        self._tasks = {t.id: t for split in ("train", "test") for t in get_tasks(domain, task_split_name=split)}

    def splits(self):
        train_ids = [t.id for t in get_tasks(self.domain, task_split_name="train")]
        random.Random(0).shuffle(train_ids)
        n_train, n_val = PRACTICE
        def mk(tid: str, held: bool) -> Task:
            short = tid if len(tid) <= 16 else hashlib.sha1(tid.encode()).hexdigest()[:10]  # telecom ids are long
            return Task(id=f"{self.domain}-{short}", family=self.name, prompt=f"τ² {self.domain} task {short}",
                        held_out=held, meta={"tau2_id": tid})
        test = [mk(t.id, False) for t in get_tasks(self.domain, task_split_name="test")]
        return [mk(t, False) for t in train_ids[:n_train]], [mk(t, True) for t in train_ids[n_train:n_train + n_val]], test

    def start(self) -> Path:
        return HERE / "start"

    def trainable(self, bundle_dir: Path, models: Models, store: Path) -> StudyTrainable:
        tr = StudyTrainable({}, adapter=Tau2Adapter(self.domain, models, self._tasks), bundle=bundle_dir, judge=Tau2Judge(),
                            store=store, work_root=RUNS / "work")
        tr.episode_workers = 8
        return tr

    def tool_descriptions(self) -> dict[str, str]:
        return {t.name: t.openai_schema["function"]["description"] for t in build_environment(self.domain).get_tools()}

    def subsets(self) -> dict[str, list[str]]:
        """Test tasks whose expected outcome needs a database write; an agent that does nothing passes the others."""
        tk = build_environment(self.domain).tools
        writes = {n for n in tk.tools if tk.tool_type(n) == ToolType.WRITE}
        _, _, test = self.splits()
        return {"write": [t.id for t in test if any(a.name in writes for a in (self._tasks[t.meta["tau2_id"]].evaluation_criteria.actions or []))]}

    def feedback(self, ep) -> str:
        return ep.judge_rationale or ep.error or ""
