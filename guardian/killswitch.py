"""Latched kill switch with dead-man's switch.

Three independent ways to halt the system:

1. ``trip()`` in-process (any code path, any operator, no approval needed).
2. An external *halt file*: any process or person with filesystem access to the
   host can ``touch`` it. This works even if the Python process is wedged
   inside model code, because every guarded operation re-checks it.
3. Dead-man's switch: operators must send a heartbeat every
   ``heartbeat_timeout`` seconds. If oversight goes silent, the system halts.

Stopping is always cheap and unilateral. Resuming is deliberately expensive:
the switch is *latched* and can only be reset with an M-of-N operator quorum.
"""

from __future__ import annotations

import os
import threading
import time

from .audit import AuditLog
from .quorum import ApprovalRequest, Quorum


class Halted(RuntimeError):
    """Raised by any guarded operation after the kill switch has tripped."""


class KillSwitch:
    def __init__(
        self,
        audit: AuditLog,
        quorum: Quorum,
        halt_file: str | os.PathLike[str] | None = None,
        heartbeat_timeout: float | None = None,
    ):
        self._audit = audit
        self._quorum = quorum
        self._halt_file = os.fspath(halt_file) if halt_file else None
        self._timeout = heartbeat_timeout
        self._last_beat = time.monotonic()
        self._tripped = threading.Event()
        self._reason: str | None = None
        self._lock = threading.Lock()

    # -- stopping ------------------------------------------------------------
    def trip(self, reason: str, by: str = "unknown") -> None:
        with self._lock:
            if self._tripped.is_set():
                return
            self._reason = reason
            self._tripped.set()
        self._audit.append("killswitch.tripped", reason=reason, by=by)

    def heartbeat(self, operator: str) -> None:
        self._last_beat = time.monotonic()
        self._audit.append("oversight.heartbeat", operator=operator)

    @property
    def tripped(self) -> bool:
        self._poll()
        return self._tripped.is_set()

    @property
    def reason(self) -> str | None:
        return self._reason

    def check(self) -> None:
        """Call before every guarded operation. Raises Halted if stopped."""
        self._poll()
        if self._tripped.is_set():
            raise Halted(f"system halted: {self._reason}")

    def _poll(self) -> None:
        if self._tripped.is_set():
            return
        if self._halt_file and os.path.exists(self._halt_file):
            self.trip("external halt file present", by="halt_file")
        elif self._timeout is not None and (
            time.monotonic() - self._last_beat > self._timeout
        ):
            self.trip("oversight heartbeat missed (dead-man's switch)", by="deadman")

    # -- resuming (hard on purpose) -----------------------------------------
    def reset_request(self) -> ApprovalRequest:
        return ApprovalRequest(action="killswitch.reset",
                               params={"reason": self._reason})

    def reset(self, request: ApprovalRequest, signatures: dict[str, str]) -> None:
        if request.action != "killswitch.reset":
            raise ValueError("not a reset request")
        if self._halt_file and os.path.exists(self._halt_file):
            raise Halted("remove the external halt file before resetting")
        approvers = self._quorum.verify(request, signatures)
        with self._lock:
            self._tripped.clear()
            self._reason = None
            self._last_beat = time.monotonic()
        self._audit.append("killswitch.reset", approvers=approvers)
