"""Capability policy: default-deny tools, risk tiers, and resource budgets."""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum


class Risk(IntEnum):
    LOW = 0        # run automatically (e.g. calculator)
    HIGH = 1       # requires M-of-N operator approval per call
    FORBIDDEN = 2  # never runs, even with approval


class PolicyViolation(Exception):
    pass


@dataclass
class Budget:
    max_calls: int = 1_000
    max_output_tokens: int = 200_000
    max_wall_seconds: float = 3_600.0
    _calls: int = 0
    _tokens: int = 0
    _started: float = field(default_factory=time.monotonic)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def charge(self, calls: int = 0, tokens: int = 0) -> None:
        with self._lock:
            if time.monotonic() - self._started > self.max_wall_seconds:
                raise PolicyViolation("wall-clock budget exhausted")
            if self._calls + calls > self.max_calls:
                raise PolicyViolation("call budget exhausted")
            if self._tokens + tokens > self.max_output_tokens:
                raise PolicyViolation("token budget exhausted")
            self._calls += calls
            self._tokens += tokens


@dataclass
class Policy:
    tools: dict[str, Risk] = field(default_factory=dict)
    budget: Budget = field(default_factory=Budget)
    max_tokens_per_call: int = 4_096
    # Patterns that must never appear in model output (e.g. leaked secrets).
    # Hitting one trips the kill switch, not just redacts.
    output_tripwires: list[str] = field(default_factory=lambda: [
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
        r"\bAKIA[0-9A-Z]{16}\b",
    ])

    def tool_risk(self, name: str) -> Risk:
        # Default deny: an unlisted tool is forbidden.
        return self.tools.get(name, Risk.FORBIDDEN)

    def tripwire(self, text: str) -> str | None:
        for pat in self.output_tripwires:
            if re.search(pat, text):
                return pat
        return None
