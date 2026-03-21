from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Literal


class State(Enum):
    LOOK = auto()
    MOVE = auto()
    LOOK_AGAIN = auto()
    ACT = auto()
    DONE = auto()
    FAIL = auto()


@dataclass
class StateResult:
    """Outcome of a single state handler invocation."""

    status: Literal["ok", "retry", "fail"]
    next_state: State
    payload: dict[str, object] = field(default_factory=dict)
    message: str = ""
