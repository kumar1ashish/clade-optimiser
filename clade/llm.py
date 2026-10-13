"""Minimal client for the optimizer's own model calls (reflector, critic, curator). Any OpenAI-compatible endpoint.

An LLM is just `Callable[[system, user], str]`; tests pass a fake. Token usage is booked per model so a run can
report what learning cost.
"""

from __future__ import annotations

import contextlib
import contextvars
import http.client
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from typing import Callable

LLM = Callable[[str, str], str]

USAGE: dict[str, dict[str, int]] = {}  # process-wide usage of optimizer calls, by model
_RUN_USAGE: contextvars.ContextVar[dict | None] = contextvars.ContextVar("clade_run_usage", default=None)
_LOCK = threading.Lock()

RETRY_CODES = (429, 498, 500, 502, 503, 504)  # 498: Groq's flex tier had no capacity for the request


def _add(book: dict, model: str, usage: dict) -> None:
    u = book.setdefault(model, {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0})
    u["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
    u["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
    u["calls"] += 1


def account(model: str, usage: dict | None) -> None:
    """Book one call's usage to the process counter and to the current run's book, if one is open."""
    if not usage:
        return
    with _LOCK:
        _add(USAGE, model, usage)
        book = _RUN_USAGE.get()
        if book is not None:
            _add(book, model, usage)


@contextlib.contextmanager
def track_usage():
    """Attribute optimizer usage to the current run only; threads started with contextvars.copy_context() count too."""
    book: dict[str, dict[str, int]] = {}
    token = _RUN_USAGE.set(book)
    try:
        yield book
    finally:
        _RUN_USAGE.reset(token)


def load_dotenv(path: str = ".env") -> None:
    try:
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except FileNotFoundError:
        pass


def service_tier(base_url: str) -> dict:
    """CLADE_SERVICE_TIER=flex asks Groq for its flex tier (ten times the rate limit, may answer 498 when busy)."""
    tier = os.environ.get("CLADE_SERVICE_TIER", "").strip()
    return {"service_tier": tier} if tier and "groq.com" in base_url else {}


def openai_compatible(model: str, base_url: str, api_key: str | None = None, temperature: float = 0.7,
                      timeout: int | None = None) -> LLM:
    """Timeouts and dropped connections are retried; rate-limit waits are capped so one stalled call cannot hang a run."""
    timeout = int(timeout or os.environ.get("CLADE_LLM_TIMEOUT", "120"))
    key = api_key or os.environ.get("OPENAI_API_KEY", "none")

    def call(system: str, user: str) -> str:
        temp, tier = temperature, service_tier(base_url)
        for attempt in range(10):
            body = json.dumps({"model": model, "temperature": temp, "stream": False, **tier,
                               "messages": ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]}).encode()
            req = urllib.request.Request(f"{base_url}/chat/completions", data=body,
                                         headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}",
                                                  "User-Agent": "clade/0.1"})
            try:
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    data = json.load(r)
                account(model, data.get("usage"))
                return strip_think(data["choices"][0]["message"].get("content") or "")
            except urllib.error.HTTPError as e:
                if e.code == 498 and tier:  # no flex capacity: send this request at the on-demand tier instead
                    tier = {}
                    continue
                if e.code in RETRY_CODES and attempt < 9:
                    time.sleep(min(60.0, float(e.headers.get("retry-after") or 2 ** attempt + 1)) + random.random())
                    continue
                detail = e.read()[:400].decode(errors="replace")
                if e.code == 400 and "tool_use_failed" in detail and attempt < 2:  # gpt-oss sometimes invents a tool call;
                    temp = max(temp, 0.3)                                            # at temperature 0 a retry would repeat it
                    continue
                raise urllib.error.HTTPError(e.url, e.code, f"{e.msg}: {detail}", e.headers, None) from None
            except (TimeoutError, urllib.error.URLError, ConnectionError, http.client.RemoteDisconnected) as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                raise TimeoutError(f"{model} did not answer after {attempt + 1} attempts: {e}") from None
        return ""

    return call


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def extract_json(text: str):
    """The first JSON object or array in an LLM reply."""
    text = strip_think(text)
    m = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
    if m:
        text = m.group(1)
    for opener, closer in (("[", "]"), ("{", "}")):
        i, j = text.find(opener), text.rfind(closer)
        if i != -1 and j > i:
            for strict in (True, False):  # strict=False tolerates raw newlines inside strings
                try:
                    return json.loads(text[i:j + 1], strict=strict)
                except json.JSONDecodeError:
                    continue
    return None
