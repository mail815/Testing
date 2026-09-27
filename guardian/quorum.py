"""M-of-N human operator approval.

Each operator holds a secret key (kept off the model host in production, e.g.
on a hardware token). A privileged action is approved only when at least
``threshold`` distinct operators sign the exact same request. Signatures are
bound to the action, its parameters, a single-use nonce and an expiry, so they
cannot be replayed or reused for a different action.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any


class QuorumError(Exception):
    pass


@dataclass(frozen=True)
class ApprovalRequest:
    action: str
    params: dict[str, Any]
    nonce: str = field(default_factory=lambda: secrets.token_hex(16))
    expires_at: float = field(default_factory=lambda: time.time() + 300)

    def canonical(self) -> bytes:
        return json.dumps(
            {"action": self.action, "params": self.params,
             "nonce": self.nonce, "expires_at": self.expires_at},
            sort_keys=True, separators=(",", ":"), default=str,
        ).encode()


def sign(key: bytes, request: ApprovalRequest) -> str:
    """Operator-side: produce a signature for a request."""
    return hmac.new(key, request.canonical(), hashlib.sha256).hexdigest()


class Quorum:
    def __init__(self, operator_keys: dict[str, bytes], threshold: int):
        if threshold < 1 or threshold > len(operator_keys):
            raise ValueError("threshold must be between 1 and number of operators")
        self._keys = dict(operator_keys)
        self.threshold = threshold
        self._used_nonces: set[str] = set()
        self._lock = threading.Lock()

    @property
    def operators(self) -> list[str]:
        return sorted(self._keys)

    def verify(self, request: ApprovalRequest, signatures: dict[str, str]) -> list[str]:
        """Return the approving operators, or raise QuorumError.

        Consumes the nonce on success so the approval is single-use.
        """
        if time.time() > request.expires_at:
            raise QuorumError("approval request expired")
        msg = request.canonical()
        valid = [
            op for op, sig in signatures.items()
            if op in self._keys and hmac.compare_digest(
                hmac.new(self._keys[op], msg, hashlib.sha256).hexdigest(), sig)
        ]
        if len(valid) < self.threshold:
            raise QuorumError(
                f"need {self.threshold} valid operator signatures, got {len(valid)}")
        with self._lock:
            if request.nonce in self._used_nonces:
                raise QuorumError("approval already used (replay)")
            self._used_nonces.add(request.nonce)
        return sorted(valid)
