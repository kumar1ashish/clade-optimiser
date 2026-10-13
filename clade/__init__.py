"""clade: optimize an LLM agent's prompt, playbook and tool descriptions from its own failures, weights frozen.

The loop (clade.core.trainable.Trainable.optimize): a coordinator plans lanes from an archive of bundle versions,
a reflector diagnoses failed training episodes, a critic filters the proposed fixes, a curator turns each into one
bounded edit, and a selector promotes a candidate only if it beats the incumbent on held-out tasks without a
per-task regression beyond run-to-run noise.
"""

from .agent import Recorder, RunResult, ShellJudge, Verdict
from .bundle import Bundle, PlaybookEntry
from .schema import Episode, Task, load_tasks
from .trainable import Trainable

__all__ = ["Trainable", "Bundle", "PlaybookEntry", "Task", "Episode", "load_tasks", "Recorder", "RunResult", "Verdict", "ShellJudge"]
