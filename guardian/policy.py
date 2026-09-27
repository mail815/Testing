"""Capability policy: default-deny tools, risk tiers, budgets, screening."""

from __future__ import annotations

import base64
import binascii
import re
import threading
import time
import unicodedata
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable

# A monitor inspects text and returns a reason string if it should be blocked.
# Plug in model-based classifiers here; regexes alone are easy to evade.
Monitor = Callable[[str], "str | None"]


class Risk(IntEnum):
    LOW = 0        # run automatically (e.g. calculator)
    HIGH = 1       # requires M-of-N operator approval per call
    FORBIDDEN = 2  # never runs, even with approval


class PolicyViolation(Exception):
    pass


@dataclass
class Budget:
    """Lifetime caps plus a sliding-window rate limit."""
    max_calls: int = 1_000
    max_output_tokens: int = 200_000
    max_wall_seconds: float = 3_600.0
    max_calls_per_minute: int = 120
    _calls: int = 0
    _tokens: int = 0
    _started: float = field(default_factory=time.monotonic)
    _recent: deque = field(default_factory=deque, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def charge(self, calls: int = 0, tokens: int = 0) -> None:
        with self._lock:
            now = time.monotonic()
            if now - self._started > self.max_wall_seconds:
                raise PolicyViolation("wall-clock budget exhausted")
            if self._calls + calls > self.max_calls:
                raise PolicyViolation("call budget exhausted")
            if self._tokens + tokens > self.max_output_tokens:
                raise PolicyViolation("token budget exhausted")
            while self._recent and now - self._recent[0] > 60:
                self._recent.popleft()
            if calls and len(self._recent) + calls > self.max_calls_per_minute:
                raise PolicyViolation("rate limit exceeded")
            self._calls += calls
            self._tokens += tokens
            self._recent.extend([now] * calls)


_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)
_B64 = re.compile(r"[A-Za-z0-9+/_-]{24,}={0,2}")
_HEX = re.compile(r"\b(?:[0-9a-fA-F]{2}){16,}\b")


def normalized_views(text: str, max_decodes: int = 64) -> list[str]:
    """Return text plus normalized and decoded variants for screening.

    Defeats cheap evasions: Unicode look-alikes (NFKC), zero-width characters,
    and secrets smuggled as base64 or hex.
    """
    base = unicodedata.normalize("NFKC", text).translate(_ZERO_WIDTH)
    views = [text, base]
    for pat, dec in ((_B64, lambda s: base64.b64decode(s + "=" * (-len(s) % 4),
                                                      altchars=b"-_" if "-" in s or "_" in s else None)),
                     (_HEX, bytes.fromhex)):
        for m in pat.findall(base)[:max_decodes]:
            try:
                views.append(dec(m).decode("utf-8", "ignore"))
            except (binascii.Error, ValueError):
                pass
    return views


DEFAULT_TRIPWIRES = [
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    r"\bAKIA[0-9A-Z]{16}\b",                     # AWS access key id
    r"\bgh[pousr]_[A-Za-z0-9]{36,}\b",           # GitHub token
    r"\bsk-[A-Za-z0-9_-]{20,}\b",                # common API key shape
    r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b",         # Slack token
]


@dataclass
class Policy:
    tools: dict[str, Risk] = field(default_factory=dict)
    budget: Budget = field(default_factory=Budget)
    max_tokens_per_call: int = 4_096
    max_prompt_chars: int = 200_000
    # Hitting a tripwire *halts the system*, not just redacts.
    output_tripwires: list[str] = field(default_factory=lambda: list(DEFAULT_TRIPWIRES))
    output_monitors: list[Monitor] = field(default_factory=list)
    input_monitors: list[Monitor] = field(default_factory=list)
    # Repeated denials look like probing for a weak spot: halt.
    max_denials: int = 5
    denial_window_s: float = 300.0

    def tool_risk(self, name: str) -> Risk:
        return self.tools.get(name, Risk.FORBIDDEN)  # default deny

    def tripwire(self, text: str) -> str | None:
        for view in normalized_views(text):
            for pat in self.output_tripwires:
                if re.search(pat, view):
                    return pat
        for mon in self.output_monitors:
            if hit := _safe_monitor(mon, text):
                return hit
        return None

    def screen_input(self, text: str) -> str | None:
        if len(text) > self.max_prompt_chars:
            return "prompt too long"
        for mon in self.input_monitors:
            if hit := _safe_monitor(mon, text):
                return hit
        return None


def _safe_monitor(mon: Monitor, text: str) -> str | None:
    try:
        return mon(text)
    except Exception as e:  # a crashing monitor blocks (fail closed)
        return f"monitor error: {e!r}"


class DenialTracker:
    def __init__(self, limit: int, window: float):
        self.limit, self.window = limit, window
        self._events: deque[float] = deque()
        self._lock = threading.Lock()

    def record(self) -> bool:
        """Record a denial; return True if the probing threshold is reached."""
        with self._lock:
            now = time.monotonic()
            self._events.append(now)
            while now - self._events[0] > self.window:
                self._events.popleft()
            return len(self._events) >= self.limit
