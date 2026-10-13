"""A plain tool-calling support agent on any OpenAI-compatible endpoint, and its Clade adapter.

The bundle's prompt and playbook become the system prompt; its tools.json overrides tool descriptions.
"""

from __future__ import annotations

import copy
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from clade.agent import RunResult
from clade.llm import RETRY_CODES, service_tier
from .env import TOOLS, Store


class SupportAgent:
    def __init__(self, model: str, base_url: str, api_key: str, max_steps: int = 14, deadline_s: float = 300.0):
        self.model, self.base_url, self.api_key, self.max_steps = model, base_url.rstrip("/"), api_key, max_steps
        self.deadline_s, self._deadline = deadline_s, None  # one stalled call must not hold an evaluation batch for minutes
        self.system_prompt, self.tools = "", TOOLS
        self.on_call = self.on_tool = None
        self.last_usage: dict = {}

    def _left(self) -> float:
        if self._deadline is None:
            return 120.0
        left = self._deadline - time.time()
        if left <= 0:
            raise TimeoutError(f"episode exceeded its {self.deadline_s:.0f} s deadline")
        return left

    def _chat(self, messages):
        tier = service_tier(self.base_url)
        for attempt in range(8):
            body = json.dumps({"model": self.model, "messages": messages, "tools": self.tools, "temperature": 0.3, "stream": False, **tier}).encode()
            left = self._left()
            req = urllib.request.Request(f"{self.base_url}/chat/completions", data=body, headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}", "User-Agent": "clade/0.1"})
            try:
                t0 = time.time()
                with urllib.request.urlopen(req, timeout=min(120.0, left)) as r:
                    data = json.load(r)
                return data, (time.time() - t0) * 1000
            except urllib.error.HTTPError as e:
                detail = e.read()[:300].decode(errors="replace")
                if e.code == 498 and tier:  # no flex capacity: this request goes at the on-demand tier
                    tier = {}
                    continue
                if e.code in RETRY_CODES and attempt < 7:
                    time.sleep(min(60.0, left, float(e.headers.get("retry-after") or 2 ** attempt + 1))); continue
                if e.code == 400 and "tool_use_failed" in detail and attempt < 2:
                    continue
                if e.code in RETRY_CODES:
                    raise ConnectionError(f"HTTP {e.code} after retries: {detail}") from None  # the provider, not the agent
                raise RuntimeError(f"HTTP {e.code}: {detail}") from None
            except (TimeoutError, urllib.error.URLError, ConnectionError) as e:
                if attempt < 2 and self._left() > 5:
                    time.sleep(2 ** attempt); continue
                raise TimeoutError(f"model did not answer: {e}") from None
        raise ConnectionError("model unavailable")

    def run(self, prompt: str, cwd: Path) -> str:
        self._deadline = time.time() + self.deadline_s
        store = Store(cwd)
        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt}]
        final = ""
        for _ in range(self.max_steps):
            data, ms = self._chat(messages)
            self.last_usage = data.get("usage") or {}
            msg = data["choices"][0]["message"]
            if self.on_call:
                self.on_call(messages[-1], msg, ms)
            calls = msg.get("tool_calls") or []
            messages.append({"role": "assistant", "content": msg.get("content") or "", **({"tool_calls": calls} if calls else {})})
            if not calls:
                final = msg.get("content") or ""
                break
            done = False
            for tc in calls:
                name = tc["function"]["name"]
                t0 = time.time()
                try:
                    args = json.loads(tc["function"].get("arguments") or "{}")
                    out, err = store.call(name, args), None
                except Exception as e:  # noqa: BLE001  (tool errors go back to the model, as in production)
                    args, out, err = {}, None, f"{type(e).__name__}: {e}"
                if self.on_tool:
                    self.on_tool(name, args, out, err, (time.time() - t0) * 1000)
                messages.append({"role": "tool", "tool_call_id": tc.get("id"), "content": out if err is None else f"ERROR {err}"})
                if name == "reply_to_customer" and err is None:
                    final, done = args.get("message", ""), True
            if done:
                break
        return final


class SupportAgentAdapter:
    name = "support_desk"
    thread_safe = True

    def apply(self, agent: SupportAgent, bundle, cwd: Path) -> SupportAgent:
        a = copy.copy(agent)
        a.system_prompt = bundle.effective_prompt()
        a.tools = [{"type": "function", "function": {**t["function"], "description": bundle.tools.get(t["function"]["name"], t["function"]["description"])}} for t in TOOLS]
        return a

    def instrument(self, agent: SupportAgent, recorder) -> SupportAgent:
        def on_call(request, msg, ms):
            recorder.call({"text": [msg["content"]] if msg.get("content") else [],
                           "tool_calls": [{"id": tc.get("id"), "name": tc["function"]["name"], "input": tc["function"].get("arguments")} for tc in (msg.get("tool_calls") or [])],
                           "thinking": []}, request=request, model=agent.model, latency_ms=ms, usage=agent.last_usage)

        def on_tool(name, args, out, err, ms):
            recorder.tool(name, args, output=out, error=err)

        agent.on_call, agent.on_tool = on_call, on_tool
        return agent

    def run(self, agent: SupportAgent, task, cwd: Path) -> RunResult:
        try:
            return RunResult(final_text=agent.run(task.prompt, cwd))
        except Exception as e:  # noqa: BLE001  (TimeoutError and transport errors are recognised as infrastructure, not agent mistakes)
            return RunResult(error=f"{type(e).__name__}: {e}")

    def tools(self, agent):
        return [t["function"] for t in TOOLS]
