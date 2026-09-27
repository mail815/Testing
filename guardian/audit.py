"""Tamper-evident, append-only audit log.

Every record contains the SHA-256 hash of the previous record, forming a hash
chain. Editing, deleting or reordering any past record breaks the chain and is
detected by :meth:`AuditLog.verify`. For stronger guarantees, periodically
ship the latest head hash to an external system the model cannot write to
(e.g. a WORM bucket or a separate operator machine).
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterator

GENESIS = "0" * 64


class AuditIntegrityError(Exception):
    """Raised when the audit chain has been tampered with."""


def _digest(record: dict[str, Any]) -> str:
    body = json.dumps(record, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode()).hexdigest()


@dataclass(frozen=True)
class AuditRecord:
    seq: int
    ts: float
    event: str
    data: dict[str, Any]
    prev: str
    hash: str


class AuditLog:
    """Hash-chained JSONL log. Thread-safe; fsyncs every write."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = os.fspath(path)
        self._lock = threading.Lock()
        self._seq = 0
        self._head = GENESIS
        if os.path.exists(self.path):
            for rec in self.verify():
                self._seq, self._head = rec.seq + 1, rec.hash

    @property
    def head(self) -> str:
        return self._head

    def append(self, event: str, **data: Any) -> AuditRecord:
        with self._lock:
            unsigned = {
                "seq": self._seq,
                "ts": time.time(),
                "event": event,
                "data": data,
                "prev": self._head,
            }
            h = _digest(unsigned)
            line = json.dumps({**unsigned, "hash": h}, sort_keys=True, default=str)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())
            self._seq += 1
            self._head = h
            return AuditRecord(hash=h, **unsigned)

    def records(self) -> Iterator[dict[str, Any]]:
        if not os.path.exists(self.path):
            return
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)

    def verify(self) -> list[AuditRecord]:
        """Re-walk the whole chain. Raises AuditIntegrityError on any tampering."""
        prev, out = GENESIS, []
        for i, raw in enumerate(self.records()):
            claimed = raw.pop("hash", None)
            if raw.get("seq") != i:
                raise AuditIntegrityError(f"record {i}: sequence gap or reorder")
            if raw.get("prev") != prev:
                raise AuditIntegrityError(f"record {i}: broken chain link")
            if _digest(raw) != claimed:
                raise AuditIntegrityError(f"record {i}: content hash mismatch")
            out.append(AuditRecord(hash=claimed, **raw))
            prev = claimed
        return out
