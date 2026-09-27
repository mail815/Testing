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
from typing import Any, Callable, Iterator

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

    def __init__(self, path: str | os.PathLike[str],
                 anchor_sink: Callable[[int, str], None] | None = None,
                 anchor_every: int = 100):
        self.path = os.fspath(path)
        self._sink = anchor_sink
        self._anchor_every = anchor_every
        self._lock = threading.RLock()
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
            if self._sink and self._seq % self._anchor_every == 0:
                self.anchor()
            return AuditRecord(hash=h, **unsigned)

    def anchor(self) -> tuple[int, str]:
        """Publish (record count, head hash) to the external sink.

        A hash chain alone cannot detect deletion of the *newest* records.
        Comparing against an anchor held elsewhere (write-once storage, a
        separate operator machine) can.
        """
        with self._lock:
            point = (self._seq, self._head)
        if self._sink:
            self._sink(*point)
        return point

    def records(self) -> Iterator[dict[str, Any]]:
        if not os.path.exists(self.path):
            return
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as e:
                        raise AuditIntegrityError(f"corrupt record: {e}") from e

    def verify(self, anchor: tuple[int, str] | None = None) -> list[AuditRecord]:
        """Re-walk the whole chain. Raises AuditIntegrityError on any tampering.

        With ``anchor=(count, head)`` from :meth:`anchor`, also detects
        truncation of records written before that anchor.
        """
        prev, out = GENESIS, []
        for i, raw in enumerate(self.records()):
            claimed = raw.pop("hash", None)
            if raw.get("seq") != i:
                raise AuditIntegrityError(f"record {i}: sequence gap or reorder")
            if raw.get("prev") != prev:
                raise AuditIntegrityError(f"record {i}: broken chain link")
            if _digest(raw) != claimed:
                raise AuditIntegrityError(f"record {i}: content hash mismatch")
            if set(raw) != {"seq", "ts", "event", "data", "prev"}:
                raise AuditIntegrityError(f"record {i}: unexpected fields")
            out.append(AuditRecord(hash=claimed, **raw))
            prev = claimed
        if anchor is not None:
            count, head = anchor
            if len(out) < count:
                raise AuditIntegrityError(
                    f"log truncated: {len(out)} records, anchor says {count}")
            if count and out[count - 1].hash != head:
                raise AuditIntegrityError("log diverges from anchor")
        return out
