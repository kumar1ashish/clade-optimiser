"""The optimizer with a fake agent and a fake LLM: no model calls. The agent passes a task only once its prompt tells
it to run the tests, so a working optimizer must find and promote that instruction."""

import json

import pytest

from clade import Bundle, PlaybookEntry, RunResult, ShellJudge, Task, Trainable, Verdict
from clade.archive import Archive
from clade.coordinator import plan_round
from clade.critic import critique
from clade.curator import _dedupe, curate, prune_playbook
from clade.events import EventLog
from clade.reflector import Diagnosis, is_task_specific, reflect
from clade.selector import Eval, select
from clade.store import Store


class PromptSensitiveAdapter:
    def instrument(self, agent, recorder):
        agent["recorder"] = recorder
        return agent

    def apply(self, agent, bundle, cwd):
        return {"prompt": bundle.effective_prompt()}

    def run(self, agent, task, cwd):
        follows = "run the tests" in agent["prompt"].lower()
        agent["recorder"].call({"text": ["running tests" if follows else "done"], "tool_calls": [], "thinking": []})
        (cwd / "ok").write_text("1" if follows else "0")
        return RunResult(final_text="done")

    def tools(self, agent):
        return []


class FileJudge:
    def score(self, task, episode, cwd):
        ok = (cwd / "ok").read_text() == "1"
        return Verdict(1.0 if ok else 0.0, "pass" if ok else "agent never ran tests")


def fake_llm(system, user):
    if "diagnose" in system:
        return json.dumps({"diagnosis": "claims done without verifying", "evidence": "no test run",
                           "fix": "Always run the tests before finishing."})
    if "review proposed fixes" in system:
        return json.dumps({"verdicts": [{"index": 0, "keep": True, "reason": "general habit"}]})
    return json.dumps({"section": "## Verify\nRun the tests and read the output before finishing."})


def make(tmp_path):
    bdir = tmp_path / "bundle"
    bdir.mkdir()
    (bdir / "prompt.md").write_text("You are a coding agent.\n")
    tr = Trainable({}, adapter=PromptSensitiveAdapter(), bundle=bdir, judge=FileJudge(), store=tmp_path / "t.db", work_root=tmp_path / "work")
    return tr, bdir


def tasks():
    return [Task(id=f"t{i}", family="coding", prompt="x", held_out=(i == 3)) for i in range(4)]


# ---- bundle, store, episodes ----

def test_bundle_versions_and_playbook(tmp_path):
    _, bdir = make(tmp_path)
    b = Bundle.load(bdir)
    child = b.derive("agent forgot to run tests")
    child.playbook.append(PlaybookEntry(id="p1", text="Always run pytest before finishing."))
    child.finalize()
    assert child.version != b.version and child.lock["parent"] == b.version
    assert "Always run pytest" in child.effective_prompt()
    again = Bundle.load(child.snapshot(tmp_path / ".versions"))
    assert again.version == child.version and again.playbook[0].text.startswith("Always")
    again.playbook[0].harmful = 3  # harmful rules drop out of what the agent sees
    assert "Always run pytest" not in again.effective_prompt()


def test_shell_judge_runs_hidden_tests(tmp_path):
    task_dir = tmp_path / "task"
    (task_dir / "workspace").mkdir(parents=True)
    (task_dir / "tests_hidden").mkdir()
    (task_dir / "tests_hidden" / "check.py").write_text("import pathlib, sys\nsys.exit(0 if pathlib.Path('answer.txt').read_text() == '42' else 1)\n")
    (task_dir / "task.json").write_text(json.dumps({"family": "x", "prompt": "answer", "judge": {"cmd": "python tests_hidden/check.py"}}))
    task = Task.load(task_dir)

    class Writer:
        def __init__(self, text): self.text = text
        def instrument(self, agent, rec): agent["rec"] = rec; return agent
        def apply(self, agent, bundle, cwd): return {}
        def run(self, agent, task, cwd):
            agent["rec"].call({"text": ["writing"], "tool_calls": [{"id": "1", "name": "write", "input": {}}]})
            agent["rec"].tool("write", {"path": "answer.txt"}, output="ok")
            (cwd / "answer.txt").write_text(self.text)
            return RunResult(final_text="done")
        def tools(self, agent): return []

    _, bdir = make(tmp_path)
    for text, score in (("42", 1.0), ("41", 0.0)):
        tr = Trainable({}, adapter=Writer(text), bundle=bdir, judge=ShellJudge(), store=tmp_path / "s.db", work_root=tmp_path / "w")
        ep = tr.run(task)
        assert ep.score == score and "exit=" in ep.judge_rationale
        assert [t["name"] for t in tr.store.tool_execs(ep.id)] == ["write"] and len(tr.store.calls(ep.id)) == 1


# ---- the loop ----

def test_optimize_promotes_fix(tmp_path):
    tr, bdir = make(tmp_path)
    v0 = tr.bundle.version
    out = tr.optimize(tasks(), llm=fake_llm, rounds=3, n=1, log=lambda *_: None)
    assert out.version != v0 and "run the tests" in out.effective_prompt().lower()
    on_disk = Bundle.load(bdir)
    assert on_disk.version == out.version and on_disk.lock["parent"] == v0
    log = [json.loads(l) for l in (bdir / ".versions" / "optimize_log.jsonl").read_text().splitlines()]
    assert log[0]["promoted"] == out.version and log[0]["incumbent"]["key"][0] == 0.0
    assert len(log) == 1  # round 2: everything passes, nothing to fix, the loop stops


def test_reflector_never_sees_held_out(tmp_path):
    tr, _ = make(tmp_path)
    seen = []
    tr.optimize(tasks(), llm=lambda s, u: (seen.append(u), fake_llm(s, u))[1], rounds=1, n=1, log=lambda *_: None)
    assert seen and not any("[task t3]" in u for u in seen)


def test_events_record_every_role(tmp_path):
    tr, _ = make(tmp_path)
    ev = EventLog(tr.store, "run")
    tr.optimize(tasks(), llm=fake_llm, rounds=2, n=1, k=2, workers=2, critic=True, lenses=True, events=ev, log=lambda *_: None)
    assert {"coordinator", "reflector", "critic", "curator", "selector"} <= {e.role for e in ev.events}
    assert {"root", "plan", "candidate", "evaluate", "promote"} <= {e.kind for e in ev.events}
    assert len(Store(tr.store.path).events("run")) == len(ev.events)
    plan = next(e for e in ev.events if e.kind == "plan")
    assert [l["lens"] for l in plan.data["lanes"]] == ["per-task", "contrast"]  # second lane rereads the same parent


def test_regression_gate(tmp_path):
    b = Bundle(path=tmp_path)
    inc = Eval(b, {"a": 1.0, "b": 0.0}, 0.5, None)
    assert "regression gate" in select(inc, [Eval(b, {"a": 0.0, "b": 1.0, "c": 1.0}, 0.67, None)]).reason
    inc = Eval(b, {"a": 1.0, "h": 0.5}, 1.0, 0.5)
    noisy = Eval(b, {"a": 0.5, "h": 1.0}, 0.5, 1.0)  # one of two runs failed on "a": noise
    assert select(inc, [noisy], n=2).promoted is noisy
    assert select(inc, [noisy], n=4).promoted is None  # two of four runs failed: a real drop


def test_task_specific_fixes_rejected(tmp_path):
    s = Store(tmp_path / "s.db")
    assert is_task_specific("Implement `column_mean` properly", s, [])
    assert is_task_specific("Call sliding_max(xs, k) with the right bound", s, [])
    assert not is_task_specific("Replace every stub with a real implementation, then run pytest and fix failures.", s, [])


def test_critic_drops_and_fails_open():
    diags = [Diagnosis("t0", "d", "e", "Always run the tests before finishing.", []), Diagnosis("t1", "d", "e", "Implement column_mean first.", [])]
    strict = lambda s, u: json.dumps({"verdicts": [{"index": 0, "keep": True}, {"index": 1, "keep": False, "reason": "task specific"}]})  # noqa: E731
    kept, verdicts = critique(diags, strict, playbook=[])
    assert [d.cluster for d in kept] == ["t0"] and verdicts[1]["keep"] is False
    kept, _ = critique(diags, lambda s, u: "not json", playbook=[])
    assert len(kept) == 2


def test_dedupe_and_prune(tmp_path):
    judge = lambda s, u: "YES" if "run the tests" in u.lower() and "run pytest" in u.lower() else "NO"  # noqa: E731
    assert _dedupe("Run pytest before finishing.", ["Always run the tests before you finish."], judge)
    assert not _dedupe("Read the file first.", ["Always run the tests before you finish."], judge)
    b = Bundle(path=tmp_path, playbook=[PlaybookEntry("a", "x", 1, 3), PlaybookEntry("b", "y", 2, 0), PlaybookEntry("c", "z", 0, 0)])
    assert prune_playbook(b, cap=1) == ["a", "c"] and [e.id for e in b.playbook] == ["b"]


def test_tool_description_edit_is_bounded(tmp_path):
    parent = Bundle(path=tmp_path, prompt="p")
    specs = [{"name": "search_policy", "description": "Search the store policy."}, {"name": "refund_items", "description": "Refund items."}]
    d = [Diagnosis("tools", "the agent searched search_policy with a narrow query and missed the amount limit", "e", "Search the policy once per concern.", [])]
    good = lambda s, u: json.dumps({"tool": "search_policy", "description": "Search the store policy once per concern: refund window, condition, final sale, amount limits."}) if "tools and their" in u else "{}"  # noqa: E731
    tool = [c for c in curate(parent, d, good, k=1, tool_specs=specs) if c.lock["diagnosis"].startswith("[tool]")]
    assert len(tool) == 1 and "amount limits" in tool[0].tools["search_policy"] and "refund_items" not in tool[0].tools
    leaky = lambda s, u: json.dumps({"tool": "search_policy", "description": "Always refund order O1006 items"}) if "tools and" in u else "{}"  # noqa: E731
    assert not [c for c in curate(parent, d, leaky, k=1, tool_specs=specs) if c.lock["diagnosis"].startswith("[tool]")]


def test_archive_prefers_productive_clades(tmp_path):
    a = Archive(tmp_path / "archive.jsonl", exploration=0.0)
    for v, parent, score in (("root", None, 0.5), ("hi", "root", 0.9), ("hi-kid", "hi", 0.2), ("mid", "root", 0.6), ("mid-kid", "mid", 0.95)):
        a.add(v, parent=parent, diagnosis="", per_task={}, held_mean=score, train_mean=score)
    assert a.clade_score("hi") == 0.2 and a.clade_score("mid") == 0.95 and a.select_parent().version == "mid-kid"
    b = Archive(tmp_path / "b.jsonl", exploration=1.0)
    b.add("p", parent=None, diagnosis="", per_task={}, held_mean=0.5, train_mean=0.5)
    b.add("c1", parent="p", diagnosis="", per_task={}, held_mean=0.5, train_mean=0.5)
    assert b.parent_score("c1") > b.parent_score("p")  # untried nodes get explored
    assert Archive(tmp_path / "archive.jsonl").nodes["root"].children == ["hi", "mid"]


def test_coordinator_plans_lanes_and_respects_budget(tmp_path):
    a = Archive(tmp_path / "archive.jsonl")
    a.add("v0", parent=None, diagnosis="", per_task={"t0": 0.5}, held_mean=0.5, train_mean=0.5)
    assert len(plan_round(a, "v0", rnd=1, max_workers=3).workers) == 1
    assert [w.lens for w in plan_round(a, "v0", rnd=1, max_workers=3, lenses=True).workers] == ["per-task", "contrast", "tools"]
    a.add("v1", parent="v0", diagnosis="", per_task={"t0": 0.75}, held_mean=0.75, train_mean=0.75)
    a.add("v2", parent="v0", diagnosis="", per_task={"t0": 0.6}, held_mean=0.6, train_mean=0.6)
    plan = plan_round(a, "v1", rnd=2, max_workers=3)
    assert [w.parent for w in plan.workers][0] == "v1" and len({w.parent for w in plan.workers}) == 3
    assert len(plan_round(a, "v1", rnd=3, max_workers=3, budget_left=0.2).workers) == 1
    assert plan_round(a, "v1", rnd=3, max_workers=3, budget_left=0.05).stop


# ---- ablation switches: each changes only what it names ----

def _diags():
    return [Diagnosis("t0", "claims done without verifying", "e", "Always run the tests before finishing.", []),
            Diagnosis("t1", "stops reading bash output at the first error", "e", "Read the whole error output before editing.", [])]


def _curator_llm(system, user):
    if "rewrite its whole system prompt" in system:
        return json.dumps({"prompt": "You are a careful coding agent. Run the tests before finishing."})
    if "ONE tool's description" in user:
        return json.dumps({"tool": "bash", "description": "Run a shell command. Read all of its output."})
    if "exactly YES or NO" in system:
        return "NO"
    return json.dumps({"section": "## Verify\nRun the tests before finishing."})


def _kinds(cands):
    return sorted(c.lock["diagnosis"].split("]")[0].strip("[") for c in cands)


SPECS = [{"name": "bash", "description": "Run a command."}]


@pytest.mark.parametrize("edits,expect", [(None, ["playbook", "playbook", "playbook", "prompt", "tool"]),
                                          (("playbook",), ["playbook", "playbook"]), (("prompt",), ["prompt"]), (("tool",), ["tool"])])
def test_edit_types(tmp_path, edits, expect):
    kw = {} if edits is None else {"edits": edits}
    assert _kinds(curate(Bundle(path=tmp_path, prompt="p"), _diags(), _curator_llm, k=3, tool_specs=SPECS, **kw)) == expect


def test_rewrite_replaces_the_whole_prompt_only(tmp_path):
    parent = Bundle(path=tmp_path, prompt="p", playbook=[PlaybookEntry("a", "keep me")])
    cands = curate(parent, _diags(), _curator_llm, k=3, rewrite=True, tool_specs=SPECS)
    assert _kinds(cands) == ["rewrite", "rewrite"]
    assert all(c.prompt.startswith("You are a careful") and [e.text for e in c.playbook] == ["keep me"] for c in cands)
    with pytest.raises(ValueError):
        curate(parent, _diags(), _curator_llm, edits=("weights",))


def test_task_filter_switch(tmp_path):
    tr, _ = make(tmp_path)
    tr.evaluate(tasks()[:2], n=1)
    leaky = lambda s, u: json.dumps({"diagnosis": "d", "evidence": "e", "fix": "Call `column_mean` before returning."})  # noqa: E731
    kw = dict(bundle_version=tr.bundle.version, prompt="p", train_task_ids={"t0", "t1"})
    assert reflect(tr.store, leaky, **kw) == []
    assert "column_mean" in reflect(tr.store, leaky, task_filter=False, **kw)[0].fix


def test_gate_variants(tmp_path):
    b = Bundle(path=tmp_path)
    inc = Eval(b, {"a": 1.0, "b": 0.0, "c": 0.0}, 1 / 3, None)
    cand = Eval(b, {"a": 0.5, "b": 1.0, "c": 1.0}, 5 / 6, None)
    assert select(inc, [cand], n=2).promoted is not None and select(inc, [cand], n=2, gate="fixed").promoted is None
    worse = Eval(b, {"a": 0.0, "b": 1.0, "c": 1.0}, 2 / 3, None)
    assert select(inc, [worse], n=2).promoted is None and select(inc, [worse], n=2, gate="none").promoted is not None
    with pytest.raises(ValueError):
        select(inc, [cand], gate="lenient")


def test_budget_is_a_ceiling_within_a_round(tmp_path):
    tr, _ = make(tmp_path)
    used = lambda: len(tr.store.episodes())  # noqa: E731
    tr.optimize(tasks(), llm=fake_llm, rounds=3, n=1, k=3, log=lambda *_: None, budget=lambda: 1 - used() / 8)
    assert used() <= 8  # 4 for the incumbent, room for one candidate


def test_long_episodes_keep_their_verdict(tmp_path):
    from clade.reflector import render_episode
    from clade.schema import Episode
    s = Store(tmp_path / "s.db")
    ep = Episode.new("t0", "v"); ep.score, ep.judge_rationale = 0.0, "AssertionError: refund was not issued"
    s.put_episode(ep)
    from clade.agent import Recorder
    rec = Recorder(s, ep)
    for i in range(80):
        rec.call({"text": [f"step {i} " + "x" * 100], "tool_calls": []})
    text = render_episode(s, s.episode(ep.id), 2500)
    assert len(text) <= 2500 and text.startswith("[task t0]") and "refund was not issued" in text
    assert "step 0 " in text and "step 79 " in text and "omitted" in text


def test_no_flex_capacity_falls_back_to_on_demand(monkeypatch):
    import io
    import urllib.error
    import urllib.request
    from clade import llm
    monkeypatch.setenv("CLADE_SERVICE_TIER", "flex")
    sent = []

    def fake_urlopen(req, timeout=None):
        body = json.loads(req.data)
        sent.append(body.get("service_tier"))
        if body.get("service_tier") == "flex":
            raise urllib.error.HTTPError(req.full_url, 498, "capacity", {}, io.BytesIO(b"capacity_exceeded"))
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": "ok"}}], "usage": {}}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    call = llm.openai_compatible("m", base_url="https://api.groq.com/openai/v1", api_key="k")
    assert call("s", "u") == "ok" and sent == ["flex", None]
