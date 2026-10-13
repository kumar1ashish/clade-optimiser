"""The benchmark harness, offline: rollout accounting, the run pipeline on a fake benchmark, and the statistics."""

import json

import pytest

from bench import common, run as run_mod
from bench.common import BudgetExhausted, Ctx, Models, StudyTrainable
from bench.stats import hierarchical_bootstrap, holm, paired_bootstrap
from clade import Bundle, RunResult, Task, Verdict


class FakeAdapter:
    """Passes a task when the prompt mentions the task's keyword; the first attempt at task 'flaky' hits a rate limit."""
    thread_safe = True
    calls: dict = {}

    def instrument(self, agent, rec):
        return {**agent, "rec": rec}

    def apply(self, agent, bundle, cwd):
        return {"prompt": bundle.effective_prompt()}

    def run(self, agent, task, cwd):
        n = FakeAdapter.calls[task.id] = FakeAdapter.calls.get(task.id, 0) + 1
        if task.id == "flaky" and n == 1:
            return RunResult(error="ConnectionError: HTTP 429")
        agent["rec"].call({"text": ["working"], "tool_calls": []})
        (cwd / "ok").write_text("1" if task.meta["key"] in agent["prompt"] else "0")
        return RunResult(final_text="done", cost_usd=0.001)

    def tools(self, agent):
        return [{"name": "lookup", "description": "Look something up."}]


class Judge:
    def score(self, task, ep, cwd):
        ok = (cwd / "ok").exists() and (cwd / "ok").read_text() == "1"
        return Verdict(1.0 if ok else 0.0, "passed" if ok else f"AssertionError: missed {task.meta['key']}")


class FakeBench:
    name, budget, test_n = "fake", 40, 2

    def __init__(self, tmp):
        self.tmp = tmp

    def splits(self):
        mk = lambda i, held=False: Task(id=f"t{i}", family="f", prompt=f"task {i}", held_out=held, meta={"key": "verify"})  # noqa: E731
        return [mk(0), mk(1), mk(2)], [mk(3, True)], [mk(4), mk(5)]

    def start(self):
        d = self.tmp / "start"
        d.mkdir(exist_ok=True)
        (d / "prompt.md").write_text("You are an agent.\n")
        return d

    def trainable(self, bundle_dir, models, store):
        return StudyTrainable({}, adapter=FakeAdapter(), bundle=bundle_dir, judge=Judge(), store=store, work_root=self.tmp / "work")

    def tool_descriptions(self):
        return {"lookup": "Look something up."}

    def feedback(self, ep):
        return ep.judge_rationale or ""


def fake_llm(system, user):
    if "diagnose" in system:
        return json.dumps({"diagnosis": "skips verification", "evidence": "e", "fix": "Always verify the result before replying."})
    if "review proposed fixes" in system:
        return json.dumps({"verdicts": [{"index": 0, "keep": True}]})
    return json.dumps({"section": "## Check\nverify every answer."})


@pytest.fixture
def fake(tmp_path, monkeypatch):
    bench = FakeBench(tmp_path)
    monkeypatch.setattr(run_mod.benchmarks, "get", lambda name: bench)
    monkeypatch.setattr(run_mod, "make_llm", lambda spec: fake_llm)
    monkeypatch.setattr(run_mod, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(run_mod, "RUNS", tmp_path / "runs")
    FakeAdapter.calls = {}
    return bench


def test_infra_failures_are_rerun_and_counted_once(fake, tmp_path, monkeypatch):
    monkeypatch.setattr(common.time, "sleep", lambda s: None)
    tr = fake.trainable(fake.start(), Models(), tmp_path / "s.db")
    tr.retries = 1
    ep = tr.run(Task(id="flaky", family="f", prompt="x", meta={"key": "k"}))
    assert ep.error is None and tr.rollouts == 1 and tr.infra_reruns == 1


def test_ctx_enforces_the_budget(fake, tmp_path):
    tr = fake.trainable(fake.start(), Models(), tmp_path / "s.db")
    train, val, _ = fake.splits()
    ctx = Ctx(bench=fake, method="m", seed=0, models=Models(), budget=3, workdir=tmp_path, start=Bundle.load(fake.start()),
              train=train, val=val, llm=fake_llm, tr=tr)
    eps = ctx.run_many([(t, ctx.start) for t in train + val])
    assert sum(e is not None for e in eps) == 3 and ctx.used == 3 and tr.rollouts == 3
    with pytest.raises(BudgetExhausted):
        ctx.run(train[0], ctx.start)
    assert "AssertionError: missed verify" in ctx.render(eps[0])  # what every optimizer reads, verdict included


def test_run_learns_scores_and_records(fake, tmp_path):
    base = run_mod.run("fake", "baseline", 0, Models(), log=lambda m: None)
    assert base["status"] == "done" and base["rollouts"] == 0 and base["test"]["mean"] == 0.0 and base["test"]["episodes"] == 4
    rec = run_mod.run("fake", "clade", 0, Models(), log=lambda m: None)
    assert rec["status"] == "done" and rec["test"]["mean"] == 1.0 and "verify" in rec["bundle"]["prompt"] + " ".join(rec["bundle"]["playbook"])
    assert 0 < rec["rollouts"] <= 40
    saved = json.loads(run_mod.result_path("fake", Models(), 40, "clade", 0).read_text())
    assert saved["test"]["per_task"] == {"t4": 1.0, "t5": 1.0}


def test_clade_stays_under_a_tight_budget(fake):
    rec = run_mod.run("fake", "clade", 1, Models(), budget=12, log=lambda m: None)  # one evaluation is 8 rollouts
    assert rec["rollouts"] <= 12


def test_statistics():
    a, b = {"x": 0.0, "y": 1.0, "z": 0.5}, {"x": 1.0, "y": 1.0, "z": 0.5}
    d = paired_bootstrap(a, b, iters=2000)
    assert d["delta"] == pytest.approx(1 / 3, abs=1e-3) and d["ci95"][0] >= 0 and d["p_le_0"] < 0.5
    same = hierarchical_bootstrap([a, a], [a, a], iters=500)
    assert same["delta"] == 0 and same["ci95"] == [0.0, 0.0]
    assert holm({"p": 0.01, "q": 0.04, "r": 0.03}) == {"p": 0.03, "r": 0.06, "q": 0.06}


def test_ace_reads_the_whole_curator_reply():
    from bench.optimizers.ace_opt import Playbook, parse_object
    reply = 'Sure.\n{"reasoning": "add two", "operations": [{"type": "ADD", "section": "common_mistakes_to_avoid", "content": "Check first."}]}'
    ops = parse_object(reply)["operations"]
    assert ops[0]["content"] == "Check first." and parse_object('```json\n{"a": [1]}\n```') == {"a": [1]} and parse_object("no json") == {}
    pb = Playbook()
    pb.add(ops[0]["section"], ops[0]["content"])
    pb.tag([{"id": "err-00001", "tag": "helpful"}])
    assert "[err-00001] helpful=1 harmful=0 :: Check first." in pb.render()


def test_report_from_runs(fake, tmp_path, monkeypatch):
    from bench import report
    for seed in (0, 1):
        run_mod.run("fake", "baseline", seed, Models(), log=lambda m: None)
        run_mod.run("fake", "clade", seed, Models(), log=lambda m: None)
    run_mod.run("fake", "clade", 0, Models(), budget=24, log=lambda m: None)  # a budget sweep shares the baseline
    monkeypatch.setattr(report, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(report, "RUNS", tmp_path / "runs")
    data = report.load()
    assert set(data) == {("fake", "openai_gpt-oss-20b", 40), ("fake", "openai_gpt-oss-20b", 24)}
    text = report.section("fake", "openai_gpt-oss-20b", 40, data[("fake", "openai_gpt-oss-20b", 40)])
    assert "| clade | 2 | 1.000 | 1.00–1.00 |" in text and "+1.000" in text
    clade_row = next(l for l in text.splitlines() if l.startswith("| clade |"))
    assert "| +0.000 |" in clade_row  # validated at 1.0 and tested at 1.0: no overfitting gap
    assert "| baseline |" in report.section("fake", "openai_gpt-oss-20b", 24, data[("fake", "openai_gpt-oss-20b", 24)])


def test_ace_accepts_every_section_name_the_curator_uses():
    from bench.optimizers.ace_opt import SECTIONS, Playbook
    pb = Playbook()
    for name in [*SECTIONS, "problem-solving heuristics", "Problem_Solving_Heuristics", "strategies and insights", "unknown"]:
        pb.add(name, f"rule for {name}")
    ids = [b["id"] for b in pb.bullets()]
    assert len(ids) == len(set(ids)) == len(SECTIONS) + 4
    assert sum(b["id"].startswith("prob-") for b in pb.bullets()) == 3 and sum(b["id"].startswith("misc-") for b in pb.bullets()) == 2


def test_pooled_bootstrap_weights_benchmarks_equally():
    from bench.stats import pooled_bootstrap
    a1, b1 = [{"x": 0.0, "y": 0.0}], [{"x": 1.0, "y": 1.0}]       # +1 on benchmark 1
    a2, b2 = [{"u": 0.5, "v": 0.5, "w": 0.5}], [{"u": 0.5, "v": 0.5, "w": 0.5}]  # 0 on benchmark 2
    d = pooled_bootstrap([(a1, b1), (a2, b2)], iters=500)
    assert d["delta"] == 0.5 and d["ci95"] == [0.5, 0.5] and d["p_le_0"] == 0.0
