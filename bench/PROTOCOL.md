# Protocol

Fixed before the main runs; changes after that point are listed under [Deviations](#deviations) with their reason.
Before it was fixed, the harness was checked end to end with short runs (60 to 240 rollouts, one run per test task)
on the support desk and on τ² airline and retail. Those checks found only infrastructure faults: malformed gpt-oss
tool calls and empty replies that crashed τ² episodes, and a JSON-parsing bug in the ACE port that kept ACE's curator
from adding bullets. Their scores were not used for any decision and are not reported.

## Question

At the same rollout budget and with the same reflection model, does Clade improve a tool-using agent more than the
published prompt and context optimizers?

## Hypotheses

- **H1 (primary).** On τ²-bench airline and retail with a gpt-oss-20b agent, Clade's test score is higher than each
  core competitor's (GEPA, GEPA with tool descriptions, MIPROv2, ACE). Tested per benchmark, Holm-adjusted over the
  four, α = 0.05.
- **H2.** Clade's learning cost (dollars, optimizer calls plus agent episodes) is lower than GEPA's and MIPROv2's.
- **H3 (exploratory).** Each ablated component contributes: removing it lowers the test score. Reported with
  intervals, no correction.

## Benchmarks

| Benchmark | Practice tasks | Test tasks | Runs per test task | Budget (rollouts) |
|---|---|---|---|---|
| τ²-bench airline (v1.0.1, commit 4ce7c03) | 20 train + 10 validation, from the official train split (seed-0 shuffle) | official test split, 20 | 4 | 600 |
| τ²-bench retail | 20 + 10, as above | official test split, 40 | 4 | 600 |
| τ²-bench telecom (optional) | 20 + 10, as above | official test split, 40 | 4 | 600 |
| Support desk (development) | 7 + 3 | 20 | 10 | 400 |

- No optimizer ever runs, reads or selects on a test task.
- The τ² agent is τ²'s own LLMAgent. The bundle replaces its instruction block and may override tool descriptions;
  the domain policy is passed verbatim. The user simulator and retail's natural-language judge are gpt-oss-120b at
  temperature 0, since the default (gpt-4.1) is not available to every reader.
- Clade was designed on the support desk, so that benchmark is reported separately and is never evidence for H1.

## Models

- **Agent:** gpt-oss-20b (Groq). Optional: qwen3.6-27b (Groq). A local qwen3:8b through Ollama was considered and
  dropped: on the study machine it generates about 6 tokens per second, about 60 hours per run.
- **Reflection and proposal model:** gpt-oss-120b for every method.

## Methods

| Method | Implementation | Searches |
|---|---|---|
| `clade` | this repository: rounds 3, n 2, k 3, lanes 3, patience 2, critic and lenses on | prompt sections, playbook rules, tool descriptions |
| `gepa` | `gepa` 0.1.4 with its reflective proposer, minibatch 3, Pareto selection | system prompt |
| `gepa-tools` | as above, with every tool description as a component | system prompt, tool descriptions |
| `ace` | official prompts from ace-agent/ace at 82709de; official defaults (temperature 0, 3 reflection rounds, no de-duplication), up to 5 epochs, best epoch on validation | playbook |
| `mipro` | DSPy 3.4.0 MIPROv2, instruction only, trials sized to the budget | system prompt |
| `mipro-demos` (optional) | as above, with up to two bootstrapped successful episodes | system prompt, examples |
| `simba` (optional) | DSPy 3.4.0 SIMBA, rules only, steps sized to the budget | system prompt rules |
| `textgrad` (optional) | TextGrad 0.1.8 TGD, minibatch 4, validate and revert | system prompt |

Every method:

- reads the same rendering of an episode: Clade's reflector view, with the trajectory and the judge's verdict;
- evaluates validation tasks twice;
- runs the agent only through the harness's counted path.

A rollout is one agent episode. An episode lost to the provider is re-run and counted once. On τ², a model call that
Groq rejects as a malformed tool call, or that returns neither text nor a tool call, is resampled up to four times at
rising temperature; an episode that still crashes is re-run up to twice, as τ²'s own runner does, and scores 0 if the
crash persists. This holds for every method. Methods must stay within
the budget, except that GEPA may finish one validation pass beyond it. Actual usage is reported.

## Ablations

Each removes one component from the full Clade configuration. The first four run on airline (`ablations` config);
the rest are optional (`ablations-extended`):

| Ablation | What is removed or changed |
|---|---|
| `clade-rewrite` | bounded edits, replaced by whole-prompt rewrites |
| `clade-no-critic` | the critic |
| `clade-greedy` | the archive and extra lanes: one lane from the incumbent |
| `clade-no-gate` | the per-task regression gate |
| `clade-no-tools` | tool-description edits |
| `clade-no-playbook` | playbook rules |
| `clade-no-lenses` | lens lanes |
| `clade-fixed-gate` | the noise adjustment of the regression gate |
| `clade-no-filter` | the task-specific filter |
| `clade-no-holdout` | the held-out split: selection on training tasks |

## Seeds and runs

- **Main comparison and ablations:** 3 seeds per (benchmark, method).
- **Baseline:** 3 independent test scorings of the starting bundle.
- Every run is reported, including crashed ones.

## Analysis

- **Metric:** mean reward over test episodes (pass^1).
- **Intervals:** hierarchical bootstrap, resampling test tasks and then seeds, 10,000 draws.
- **H1:** Clade minus competitor, one-sided p, Holm-adjusted across the four core competitors within each benchmark.
- **Secondary:**
  - learning cost and time;
  - pass^k;
  - the airline tasks that require a database write;
  - optional: a budget sweep (300, 600, 1200) for Clade, GEPA and MIPROv2 on airline; telecom; a qwen3.6-27b agent.

## Method version

The `clade/` package in this repository is the code every run used; it did not change once the protocol was fixed.
The one change made while preparing the study, before any τ²
optimization run: the reflector keeps an episode's task line and verdict when the trajectory exceeds its character
budget, trimming the middle rather than the end. τ² conversations are far longer than the support-desk episodes the
method was developed on, and without this change the reflector would not see why a long episode failed.

## Deviations

1. **τ² learning budget lowered from 900 to 600 rollouts** (12 Oct 2026, protocol v1.1). The main grid was started at
   900, and Groq's capacity put it at about 1.5 days. The first runs were stopped about 90 minutes in, before any τ²
   learning run had finished or been scored, and their data was discarded; the grid restarted at 600 for every
   method. The budget sweep moved from 300/900/1800 to 300/600/1200 accordingly. The method and every other setting
   are unchanged.
2. **Scope reduced** (12 Oct 2026, protocol v1.2), before any τ² result existed: four core competitors instead of
   seven (SIMBA, TextGrad and MIPROv2 with demos become optional), three seeds instead of five, four ablations on
   airline instead of ten on both domains; telecom, the second agent model and the budget sweep become optional. The
   full design needed about 2.5 days of compute at our provider capacity; this one about 10 hours. H1's Holm family
   shrinks from seven to four accordingly. The support-desk runs already finished are kept.
3. **ACE port fix** (12 Oct 2026, v1.2.1). The port stored section names with hyphens and looked their bullet-id prefix
   up without them, so the first bullet the curator filed under "problem-solving heuristics" crashed the run. Both
   seed-0 τ² ACE runs, the only ones started before the fix, were restarted from scratch. No other method is affected.
