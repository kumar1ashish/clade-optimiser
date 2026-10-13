# Clade

Clade improves an LLM agent without touching its weights. It runs the agent on practice tasks, reads the failures,
and edits the agent's **bundle** (system prompt, playbook rules, tool descriptions) one bounded change at a time. A
change is kept only if it wins on held-out tasks without a per-task regression larger than run-to-run noise.

## Method

One round of `Trainable.optimize` ([clade/trainable.py](clade/trainable.py)):

| Role | Module | |
|---|---|---|
| Coordinator | [coordinator.py](clade/coordinator.py) | plans lanes: exploit the incumbent, explore the archive's best clade, or reread the incumbent through another lens |
| Archive | [archive.py](clade/archive.py) | every evaluated version; parents chosen by clade score (how good their descendants are) plus an exploration bonus |
| Reflector | [reflector.py](clade/reflector.py) | one diagnosis per failing training task plus a cross-cutting one; fixes naming task-specific identifiers are rejected |
| Critic | [critic.py](clade/critic.py) | drops fixes that are task hints, not actionable, or already in the playbook, before any episode is spent |
| Curator | [curator.py](clade/curator.py) | turns each fix into one bounded edit: a playbook rule, a short prompt section, or one tool description |
| Selector | [selector.py](clade/selector.py) | promotes on held-out mean behind a per-task regression gate that tolerates one run of noise |

The optimizer is standard library only. An agent plugs in through a four-method adapter and a judge
([clade/agent.py](clade/agent.py)); an LLM is any `Callable[[system, user], str]`.

## Install

```bash
uv venv --python 3.12 .venv && uv pip install -e ".[dev]"
.venv/bin/python -m pytest -q
```

## Results

gpt-oss-20b agent, gpt-oss-120b reflection model, 600 learning rollouts per run, 3 seeds; full tables with
intervals in [bench/results/REPORT.md](bench/results/REPORT.md).

| Method | τ² airline | τ² retail | Support desk (dev) | Learning cost per τ² run |
|---|---|---|---|---|
| Starting agent | 0.433 | 0.475 | 0.748 | — |
| **Clade** | 0.467 | **0.533** | **0.882** | **$2.69 – $3.92** |
| GEPA | 0.467 | 0.438 | 0.640 | $3.57 – $5.41 |
| GEPA, with tool descriptions | 0.425 | 0.410 | 0.587 | $4.13 – $6.11 |
| MIPROv2 | 0.479 | 0.454 | 0.702 | $2.95 – $4.77 |
| ACE | 0.442 | 0.475 | 0.723 | $3.90 – $5.46 |

Test score is the mean reward over the official test split (4 runs per task). On airline no difference between
methods is significant; on retail Clade is ahead of every competitor (significant against GEPA with tool descriptions and marginal
against MIPROv2 after Holm correction) and is the only method above the starting agent.

## Paper

[paper/](paper/) holds the paper (ACL format). `make -C paper` regenerates every table, figure and number from
`bench/results` and builds `paper.pdf`.

## Reproduce the benchmarks

```bash
bash bench/setup.sh                                   # venv, competitors, pinned τ²-bench and ACE prompts
echo "GROQ_API_KEY=..." > .env                        # agent, reflection and user-simulator models run on Groq
.venv/bin/python -m bench.grid bench/configs/main.json --jobs 16
.venv/bin/python -m bench.report                      # writes bench/results/REPORT.md
```

| Config | What it runs | Runs |
|---|---|---|
| [main](bench/configs/main.json) | Clade, GEPA, GEPA with tool descriptions, MIPROv2 and ACE on τ²-bench airline and retail, 3 seeds | 36 |
| [ablations](bench/configs/ablations.json) | four Clade ablations on airline, 3 seeds | 12 |
| [support-desk](bench/configs/support-desk.json) | the development benchmark, same methods, 3 seeds | 18 |

Optional: [extended](bench/configs/extended.json) (SIMBA, TextGrad, MIPROv2 with demos),
[ablations-extended](bench/configs/ablations-extended.json), [telecom](bench/configs/telecom.json),
[qwen](bench/configs/qwen.json) (qwen3.6-27b as the agent), and the budget sweep
([300](bench/configs/budget-300.json), [1200](bench/configs/budget-1200.json)).

One run is `python -m bench.run --bench tau2-airline --method clade --seed 0`. Each writes a JSON file under
[bench/results](bench/results) with the learned bundle, rollouts, cost and per-task test scores. The design,
hypotheses and analysis are fixed in [bench/PROTOCOL.md](bench/PROTOCOL.md).

Competitors run through their official libraries where one exists: `gepa`, DSPy (MIPROv2, SIMBA), TextGrad, and
ACE's official prompts. Every method gets the same agent, reflection model, practice tasks, episode rendering and
rollout budget ([bench/common.py](bench/common.py)).

## License

MIT, see [LICENSE](LICENSE). τ²-bench and the ACE prompts are fetched by `bench/setup.sh` under their own licenses (MIT and
Apache-2.0).
