"""Latched, persistent, fail-closed kill switch with dead-man's switch.

Ways to halt the system (any one is enough):

1. ``trip()`` in-process (any code path, any operator, no approval needed).
2. An external *halt file*: anyone with filesystem access can ``touch`` it.
3. Dead-man's switch: operators must send a *signed* heartbeat every
   ``heartbeat_timeout`` seconds. If oversight goes silent, the system halts.
4. Fail-closed: if the switch cannot determine its own state (e.g. the halt
   path is unreadable), it treats that as a halt.
5. ``Watchdog``: a background thread that polls the switch and, on trip, runs
   hard-stop callbacks (kill child processes, ``os._exit``) even if the main
   thread is stuck inside model code.

A trip is written to a *latch file* before anything else, so restarting the
process does not un-press the button. Resuming requires an M-of-N quorum.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Callable

from .audit import AuditLog
from .quorum import ApprovalRequest, Quorum, QuorumError


class Halted(RuntimeError):
    """Raised by any guarded operation after the kill switch has tripped."""


class KillSwitch:
    def __init__(
        self,
        audit: AuditLog,
        quorum: Quorum,
        halt_file: str | os.PathLike[str] | None = None,
        latch_file: str | os.PathLike[str] | None = None,
        heartbeat_timeout: float | None = None,
        heartbeat_max_skew: float = 30.0,
    ):
        self._audit = audit
        self._quorum = quorum
        self._halt_file = os.fspath(halt_file) if halt_file else None
        self._latch_file = os.fspath(latch_file) if latch_file else None
        self._timeout = heartbeat_timeout
        self._skew = heartbeat_max_skew
        self._last_beat = time.monotonic()
        self._last_beat_ts = 0.0
        self._tripped = threading.Event()
        self._reason: str | None = None
        self._lock = threading.RLock()
        self._callbacks: list[Callable[[str], None]] = []
        if self._latch_file and os.path.exists(self._latch_file):
            with open(self._latch_file, encoding="utf-8") as f:
                self._reason = f.read().strip() or "latched from previous run"
            self._tripped.set()
            self._audit.append("killswitch.latched_on_start", reason=self._reason)

    # -- stopping ------------------------------------------------------------
    def on_trip(self, callback: Callable[[str], None]) -> None:
        """Register a hard-stop action (kill workers, cut network, exit...)."""
        self._callbacks.append(callback)

    def trip(self, reason: str, by: str = "unknown") -> None:
        with self._lock:
            if self._tripped.is_set():
                return
            self._reason = reason
            self._tripped.set()  # in-memory first: never depends on I/O succeeding
            try:
                if self._latch_file:
                    with open(self._latch_file, "w", encoding="utf-8") as f:
                        f.write(reason)
                        f.flush()
                        os.fsync(f.fileno())
            finally:
                try:
                    self._audit.append("killswitch.tripped", reason=reason, by=by)
                finally:
                    for cb in list(self._callbacks):
                        try:
                            cb(reason)
                        except Exception:  # one failing callback must not stop others
                            pass

    def heartbeat(self, request: ApprovalRequest, signatures: dict[str, str]) -> None:
        """Signed proof that a human operator is still watching.

        ``request`` must be ``heartbeat_request()`` signed by >= 1 operator.
        """
        if request.action != "oversight.heartbeat":
            raise ValueError("not a heartbeat")
        ts = float(request.params.get("ts", 0))
        if abs(time.time() - ts) > self._skew or ts <= self._last_beat_ts:
            raise QuorumError("stale or replayed heartbeat")
        approvers = self._quorum.verify(request, signatures, threshold=1)
        with self._lock:
            self._last_beat, self._last_beat_ts = time.monotonic(), ts
        self._audit.append("oversight.heartbeat", operators=approvers)

    @staticmethod
    def heartbeat_request() -> ApprovalRequest:
        return ApprovalRequest("oversight.heartbeat", {"ts": time.time()},
                               expires_at=time.time() + 60)

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
        try:
            if self._halt_file and os.path.lexists(self._halt_file):
                self.trip("external halt file present", by="halt_file")
            elif self._timeout is not None and (
                time.monotonic() - self._last_beat > self._timeout
            ):
                self.trip("oversight heartbeat missed (dead-man's switch)", by="deadman")
        except Exception as e:  # fail closed
            if not self._tripped.is_set():
                try:
                    self.trip(f"kill switch self-check failed: {e!r}", by="selfcheck")
                except Exception:
                    pass
            self._tripped.set()

    # -- resuming (hard on purpose) -----------------------------------------
    def reset_request(self) -> ApprovalRequest:
        return ApprovalRequest(action="killswitch.reset",
                               params={"reason": self._reason})

    def reset(self, request: ApprovalRequest, signatures: dict[str, str]) -> None:
        if request.action != "killswitch.reset":
            raise ValueError("not a reset request")
        if request.params.get("reason") != self._reason:
            raise QuorumError("reset was signed for a different halt")
        if self._halt_file and os.path.lexists(self._halt_file):
            raise Halted("remove the external halt file before resetting")
        approvers = self._quorum.verify(request, signatures)
        with self._lock:
            if self._latch_file and os.path.exists(self._latch_file):
                os.remove(self._latch_file)
            self._tripped.clear()
            self._reason = None
            self._last_beat = time.monotonic()
        self._audit.append("killswitch.reset", approvers=approvers)


class Watchdog:
    """Background poller so a halt takes effect even if the main thread hangs."""

    def __init__(self, kill: KillSwitch, interval: float = 0.5):
        self._kill, self._interval = kill, interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="guardian-watchdog",
                                        daemon=True)

    def start(self) -> "Watchdog":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            if self._kill.tripped:
                return


def hard_exit(code: int = 70) -> Callable[[str], None]:
    """on_trip callback that terminates the process immediately."""
    return lambda _reason: os._exit(code)
