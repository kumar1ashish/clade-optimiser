"""Events: a structured record of what each role did in a run (coordinator plans, reflector proposals, critic
verdicts, candidates, evaluations, promotions), kept separate from the prose log so a run can be analysed after."""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Event:
    id: str
    run_id: str
    ts: float
    round: int | None
    role: str
    kind: str                      # plan | root | propose | critique | candidate | evaluate | promote | reject | blocked | note | stop
    ref: str | None = None         # the bundle version the event is about
    parent: str | None = None
    text: str = ""
    data: dict[str, Any] = field(default_factory=dict)


class EventLog:
    """Keeps events in memory and, when given a store, writes them to it."""

    def __init__(self, store: Any = None, run_id: str = "local"):
        self.store, self.run_id = store, run_id
        self.events: list[Event] = []
        self.round: int | None = None

    def emit(self, role: str, kind: str, text: str = "", *, ref: str | None = None, parent: str | None = None, **data: Any) -> Event:
        ev = Event(id="ev_" + uuid.uuid4().hex[:10], run_id=self.run_id, ts=time.time(), round=self.round, role=role,
                   kind=kind, ref=ref, parent=parent, text=text, data=data)
        self.events.append(ev)
        if self.store is not None:
            try:
                self.store.put_event(asdict(ev))
            except Exception:  # noqa: BLE001  (events are observability, never the reason a run fails)
                pass
        return ev
