"""Trainable: wrap an agent, run and judge episodes, and optimize its bundle. The model's weights never change.

One round of optimize():
  coordinator  plans lanes: lane 0 exploits the incumbent, further lanes branch from the archive's best clades,
               or reread the incumbent through another lens when the archive has no second parent yet
  reflector    diagnoses failed training episodes of each lane's parent (held-out episodes are never read)
  critic       drops fixes that are task hints, not actionable, or already in the playbook
  curator      turns each surviving fix into one bounded candidate (a rule, a prompt section, or a tool description)
  selector     evaluates every candidate on train + held-out tasks and promotes the best only if it beats the
               incumbent on held-out mean without a per-task regression larger than one run of noise
The loop stops at a plateau (`patience` rounds without a promotion), when nothing fails, or when the budget is spent.
"""

from __future__ import annotations

import contextvars
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

from .agent import Adapter, Judge, Recorder
from .archive import Archive
from .bundle import Bundle
from .coordinator import Worker, plan_round
from .cost import Ledger, usd
from .critic import critique
from .curator import EDIT_TYPES, curate
from .events import EventLog
from .reflector import reflect
from .schema import Episode, Task
from .selector import evaluate, log_round, select
from .store import Store


class Trainable:
    def __init__(self, agent: Any, *, adapter: Adapter, bundle: str | Path | Bundle, judge: Judge,
                 store: Store | str | Path = "clade.db", work_root: str | Path | None = None):
        self.agent, self.adapter, self.judge = agent, adapter, judge
        self.bundle = bundle if isinstance(bundle, Bundle) else Bundle.load(bundle)
        self.store = store if isinstance(store, Store) else Store(store)
        self.work_root = Path(work_root) if work_root else Path(tempfile.gettempdir()) / "clade"
        self.versions_dir = self.bundle.path / ".versions"
        self.bundle.snapshot(self.versions_dir)
        self.episode_workers = 1  # parallel episodes per evaluation (only for adapters marked thread_safe)

    # ---- one episode ----
    def run(self, task: Task, *, bundle: Bundle | None = None) -> Episode:
        b = bundle or self.bundle
        ep = Episode.new(task.id, b.version)
        cwd = self.work_root / ep.id
        if task.workspace:
            shutil.copytree(task.workspace, cwd, ignore=shutil.ignore_patterns(".git", "__pycache__", ".venv"))
        else:
            cwd.mkdir(parents=True)
        ep.cwd = str(cwd)
        self.store.put_episode(ep)

        agent = self.adapter.instrument(self.adapter.apply(self.agent, b, cwd), Recorder(self.store, ep))
        res = self.adapter.run(agent, task, cwd)

        ep.ended = time.time()
        ep.cost_usd, ep.num_turns, ep.error = res.cost_usd, res.num_turns, res.error
        if ep.num_turns is None or ep.cost_usd is None:  # derive from the recorded calls
            calls = self.store.calls(ep.id)
            if ep.num_turns is None:
                ep.num_turns = len(calls)
            if ep.cost_usd is None:
                ep.cost_usd = round(sum(usd(c.get("model"), int((c.get("usage") or {}).get("prompt_tokens", 0) or 0),
                                            int((c.get("usage") or {}).get("completion_tokens", 0) or 0)) for c in calls), 6)
        ep.meta["final_text"] = res.final_text[:4000]
        verdict = self.judge.score(task, ep, cwd)
        ep.score, ep.judge_rationale = verdict.score, verdict.rationale
        self.store.put_episode(ep)
        shutil.rmtree(cwd, ignore_errors=True)
        return ep

    # ---- many episodes ----
    def evaluate(self, tasks: Iterable[Task], *, n: int = 1, bundle: Bundle | None = None, workers: int = 1) -> list[Episode]:
        """Each task n times. Parallel only when the adapter says it is thread-safe; each worker runs in a copy of
        the caller's context so usage attribution follows it."""
        jobs = [t for t in tasks for _ in range(n)]
        if workers <= 1 or not getattr(self.adapter, "thread_safe", False) or len(jobs) <= 1:
            return [self.run(t, bundle=bundle) for t in jobs]
        with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
            futs = [ex.submit(contextvars.copy_context().run, self.run, t, bundle=bundle) for t in jobs]
            return [f.result() for f in as_completed(futs)]

    # ---- the optimizer ----
    def optimize(self, tasks: Iterable[Task], *, llm, rounds: int = 3, n: int = 1, k: int = 3, patience: int = 2,
                 workers: int = 1, critic: bool = True, lenses: bool = False, parent_policy: str = "archive",
                 budget=None, promote_to_disk: bool = True, log=print, events: EventLog | None = None,
                 edits: tuple[str, ...] | None = None, rewrite: bool = False, task_filter: bool = True,
                 gate: str = "noise-aware") -> Bundle:
        """tasks:   all practice tasks; those with held_out=True are used only by the selector.
        llm:     Callable[[system, user], str] for the reflector, critic and curator.
        n, k:    runs per task in each evaluation; rule candidates per lane and round.
        workers: lanes per round. Reflection, critique and curation run in parallel per lane; candidates are
                 evaluated one after another.
        parent_policy: "archive" (lanes branch from the archive by clade score plus exploration) or "incumbent".
        budget:  callable returning the fraction of the rollout budget left. The coordinator narrows to one lane
                 below 25% and stops below 8%; within a round no further parent or candidate is evaluated once it
                 reaches 0.
        edits, rewrite, task_filter, gate: ablation switches (curator.curate, reflector.reflect, selector.select).
                 The defaults are the method.
        """
        edits = tuple(edits) if edits is not None else EDIT_TYPES
        ev = events or EventLog()

        def exhausted() -> bool:
            if budget is None:
                return False
            try:
                return float(budget()) <= 0.0
            except Exception:  # noqa: BLE001
                return False

        try:
            tool_specs = list(self.adapter.tools(self.agent) or [])  # descriptions the curator may rewrite, one at a time
        except Exception:  # noqa: BLE001
            tool_specs = []
        tasks = list(tasks)
        train = [t for t in tasks if not t.held_out]
        held = [t for t in tasks if t.held_out]
        train_ids = {t.id for t in train}
        log_path = self.versions_dir / "optimize_log.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        archive = Archive(self.versions_dir / "archive.jsonl")
        ev.round = 0
        log(f"[round 0] evaluating incumbent {self.bundle.version} on {len(train)} train + {len(held)} held-out, n={n}")
        incumbent = evaluate(self, self.bundle, train, held, n)
        archive.add(incumbent.bundle.version, parent=incumbent.bundle.lock.get("parent"), diagnosis=incumbent.bundle.lock.get("diagnosis", ""),
                    per_task=incumbent.per_task, held_mean=incumbent.held_mean, train_mean=incumbent.train_mean, tag="incumbent")
        log(f"          held={incumbent.held_mean} train={incumbent.train_mean:.2f} {incumbent.per_task}")
        ev.emit("selector", "root", f"incumbent {incumbent.bundle.version}: held {incumbent.held_mean} train {incumbent.train_mean:.2f}",
                ref=incumbent.bundle.version, held=incumbent.held_mean, train=incumbent.train_mean, per_task=incumbent.per_task)
        stale = 0
        tried: set[str] = {incumbent.bundle.version}
        promotions: list[bool] = []

        def lane_think(w, pe):
            """Reflect, critique and curate for one lane. Runs in a thread; touches only the store and the LLM."""
            diags = reflect(self.store, llm, bundle_version=pe.bundle.version, prompt=pe.bundle.effective_prompt(),
                            train_task_ids=train_ids, lens=w.lens, task_filter=task_filter)
            proposed = len(diags)
            for d in diags:
                ev.emit("reflector", "propose", f"lane {w.lane} ({w.lens}) · {d.cluster}: {d.fix[:140]}",
                        ref=pe.bundle.version, lane=w.lane, lens=w.lens, cluster=d.cluster, diagnosis=d.diagnosis, fix=d.fix)
            if critic and diags:
                kept, verdicts = critique(diags, llm, playbook=[e.text for e in pe.bundle.playbook])
                for v, d in zip(verdicts, diags):
                    ev.emit("critic", "critique", f"lane {w.lane} · {'kept' if v['keep'] else 'dropped'}: {d.fix[:90]}" + (f" ({v['reason']})" if v.get("reason") else ""),
                            ref=pe.bundle.version, lane=w.lane, keep=v["keep"], reason=v.get("reason", ""), fix=d.fix)
                diags = kept
            cands = curate(pe.bundle, diags, llm, k=k, tool_specs=tool_specs, edits=edits, rewrite=rewrite) if diags else []
            return w, pe, diags, cands, proposed

        for rnd in range(1, rounds + 1):
            ev.round = rnd
            if incumbent.train_mean >= 1.0 and (incumbent.held_mean is None or incumbent.held_mean >= 1.0):
                log(f"[round {rnd}] incumbent passes every task; no headroom left, stopping")
                ev.emit("coordinator", "stop", "the incumbent passes every training and held-out task; nothing left to gain on this task set")
                break
            ledger = Ledger(); ledger.start(time.time())
            left = 1.0
            if budget is not None:
                try:
                    left = max(0.0, min(1.0, float(budget())))
                except Exception:  # noqa: BLE001
                    left = 1.0
            lanes = max(1, int(workers)) if parent_policy == "archive" else 1
            plan = plan_round(archive, incumbent.bundle.version, rnd=rnd, max_workers=lanes, budget_left=left, recent_promotions=promotions, lenses=lenses)
            if lanes == 1 and parent_policy == "archive" and len(archive.nodes) > 1:
                node = archive.select_parent()  # single-lane behaviour: the archive may still pick a non-incumbent parent
                if node.version != incumbent.bundle.version and (self.versions_dir / node.version).exists():
                    plan.workers[0] = Worker(0, node.version, f"parent from archive (clade {archive.clade_score(node.version):.2f}, "
                                                              f"parent score {archive.parent_score(node.version):.2f}) instead of incumbent {incumbent.bundle.version}")
            ev.emit("coordinator", "stop" if plan.stop else "plan", plan.note, budget_left=round(left, 3),
                    lanes=[{"lane": w.lane, "parent": w.parent, "reason": w.reason, "lens": w.lens} for w in plan.workers])
            log(f"[round {rnd}] coordinator: {plan.note}")
            if plan.stop:
                break
            staged = []  # parents are evaluated here, in order, so no two lanes run episodes at once
            for w in plan.workers:
                if w.parent == incumbent.bundle.version:
                    pe = incumbent
                elif exhausted():
                    continue
                else:
                    snap = self.versions_dir / w.parent
                    pe = evaluate(self, Bundle.load(snap), train, held, n) if snap.exists() else incumbent
                    log(f"          lane {w.lane} from {w.parent}: {w.reason}")
                staged.append((w, pe))
            if len(staged) > 1:
                with ThreadPoolExecutor(max_workers=len(staged)) as ex:  # each lane keeps the run's usage book
                    results = [f.result() for f in [ex.submit(contextvars.copy_context().run, lane_think, *wp) for wp in staged]]
            else:
                results = [lane_think(*staged[0])]
            all_diags, evals, any_new, proposed_total = [], [], False, 0
            for w, pe, diags, cands, proposed in results:
                proposed_total += proposed
                all_diags += diags
                log(f"[round {rnd}] lane {w.lane}: {len(diags)} diagnoses, {len(cands)} candidates")
                for d in diags:
                    log(f"          {d.cluster}: {d.diagnosis[:120]} | fix: {d.fix[:120]}")
                cands = [c for c in cands if c.version not in tried]
                tried |= {c.version for c in cands}
                for c in cands:
                    if exhausted():
                        log(f"          budget exhausted; {c.version} not evaluated")
                        ev.emit("coordinator", "note", f"budget exhausted before {c.version} was evaluated", ref=c.version)
                        continue
                    any_new = True
                    c.snapshot(self.versions_dir)
                    ev.emit("curator", "candidate", c.lock["diagnosis"][:240], ref=c.version, parent=pe.bundle.version, lane=w.lane, lens=w.lens)
                    e = evaluate(self, c, train, held, n)
                    log(f"          cand {c.version} {c.lock['diagnosis'][:70]!r} held={e.held_mean} train={e.train_mean:.2f}")
                    ev.emit("selector", "evaluate", f"{c.version}: held {e.held_mean} train {e.train_mean:.2f}", ref=c.version,
                            parent=pe.bundle.version, lane=w.lane, held=e.held_mean, train=e.train_mean, per_task=e.per_task)
                    evals.append(e)
                    archive.add(c.version, parent=pe.bundle.version, diagnosis=c.lock.get("diagnosis", ""),
                                per_task=e.per_task, held_mean=e.held_mean, train_mean=e.train_mean, tag="candidate")
            if not all_diags and not proposed_total:
                log("          nothing to fix on train tasks; stopping")
                ev.emit("coordinator", "stop", "nothing left to fix on the training tasks")
                break
            if not all_diags or not any_new:
                why = f"critic rejected all {proposed_total} proposed fixes" if not all_diags else "no new candidates this round"
                log(f"          {why}")
                stale += 1; promotions.append(False)
                ev.emit("selector", "reject", why, versions=[])
                if stale >= patience:
                    log("          plateau; stopping"); ev.emit("coordinator", "stop", f"plateau: no promotion for {stale} rounds")
                    break
                continue
            sel = select(incumbent, evals, n=n, gate=gate)
            cost = ledger.close(self.store, time.time())
            log_round(log_path, rnd, sel, all_diags, extra={"parent": results[0][1].bundle.version, "cost": cost, "archive": archive.summary(),
                                                            "lanes": [{"lane": w.lane, "parent": pe.bundle.version} for w, pe, _, _, _ in results]})
            log(f"          -> {sel.reason}  [learning tokens {cost['learning_tokens']:,}, ~${cost['learning_usd_est']}]")
            if sel.promoted:
                parent_ids = {p.id for p in incumbent.bundle.playbook}
                for entry in sel.promoted.bundle.playbook:  # new rules earn a helpful count; texts, and so the version, are unchanged
                    if entry.id not in parent_ids:
                        entry.helpful += 1
                incumbent, stale = sel.promoted, 0
                promotions.append(True)
                archive.add(incumbent.bundle.version, parent=incumbent.bundle.lock.get("parent"), diagnosis=incumbent.bundle.lock.get("diagnosis", ""),
                            per_task=incumbent.per_task, held_mean=incumbent.held_mean, train_mean=incumbent.train_mean, tag="promoted")
                ev.emit("selector", "promote", f"promoted {incumbent.bundle.version}: {sel.reason}", ref=incumbent.bundle.version,
                        parent=incumbent.bundle.lock.get("parent"), held=incumbent.held_mean, train=incumbent.train_mean, cost=cost)
                if promote_to_disk:
                    incumbent.bundle.save(self.bundle.path)
            else:
                promotions.append(False)
                best = max(evals, key=lambda e: e.key)
                blocked = sel.reason.startswith("regression gate")
                ev.emit("selector", "blocked" if blocked else "reject", sel.reason, ref=best.bundle.version if blocked else None,
                        versions=[e.bundle.version for e in evals], cost=cost)
                stale += 1
                if stale >= patience:
                    log("          plateau; stopping"); ev.emit("coordinator", "stop", f"plateau: no promotion for {stale} rounds")
                    break
        self.bundle = incumbent.bundle
        ev.emit("coordinator", "note", f"finished at {incumbent.bundle.version}", ref=incumbent.bundle.version)
        return self.bundle
