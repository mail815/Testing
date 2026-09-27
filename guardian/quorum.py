"""M-of-N human operator approval with Ed25519 signatures.

Each operator holds an Ed25519 *private* key off the model host (ideally on a
hardware token). The host only holds *public* keys, so compromising the host
does not let an attacker (or the model) forge approvals.

A privileged action is approved only when at least ``threshold`` distinct
operators sign the exact same request. Signatures are bound to the action, its
parameters, a single-use nonce and an expiry. Used nonces can be persisted so
approvals cannot be replayed across restarts.
"""

from __future__ import annotations

import json
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey, Ed25519PublicKey)

MAX_APPROVAL_TTL = 3600.0


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
            {"v": 1, "action": self.action, "params": self.params,
             "nonce": self.nonce, "expires_at": self.expires_at},
            sort_keys=True, separators=(",", ":"), default=str,
        ).encode()


def generate_operator_key() -> tuple[Ed25519PrivateKey, Ed25519PublicKey]:
    priv = Ed25519PrivateKey.generate()
    return priv, priv.public_key()


def sign(private_key: Ed25519PrivateKey, request: ApprovalRequest) -> str:
    """Operator-side: produce a signature for a request."""
    return private_key.sign(request.canonical()).hex()


class NonceStore:
    """Append-only record of consumed approval nonces (optionally on disk)."""

    def __init__(self, path: str | os.PathLike[str] | None = None):
        self.path = os.fspath(path) if path else None
        self._used: set[str] = set()
        self._lock = threading.Lock()
        if self.path and os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                self._used = {line.strip() for line in f if line.strip()}

    def consume(self, nonce: str) -> None:
        with self._lock:
            if nonce in self._used:
                raise QuorumError("approval already used (replay)")
            if self.path:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(nonce + "\n")
                    f.flush()
                    os.fsync(f.fileno())
            self._used.add(nonce)


class Quorum:
    def __init__(self, operator_keys: dict[str, Ed25519PublicKey], threshold: int,
                 nonces: NonceStore | None = None):
        if threshold < 1 or threshold > len(operator_keys):
            raise ValueError("threshold must be between 1 and number of operators")
        for k in operator_keys.values():
            if not isinstance(k, Ed25519PublicKey):
                raise TypeError("Quorum takes Ed25519 *public* keys only")
        self._keys = dict(operator_keys)
        self.threshold = threshold
        self._nonces = nonces or NonceStore()

    @property
    def operators(self) -> list[str]:
        return sorted(self._keys)

    def _valid(self, request: ApprovalRequest, signatures: dict[str, str]) -> list[str]:
        now = time.time()
        if now > request.expires_at:
            raise QuorumError("approval request expired")
        if request.expires_at - now > MAX_APPROVAL_TTL:
            raise QuorumError("approval lifetime too long")
        return self.valid_signers(request, signatures)

    def valid_signers(self, request: ApprovalRequest,
                      signatures: dict[str, str]) -> list[str]:
        """Check signatures only (no expiry, no nonce). For re-verifying
        stored, already-consumed approvals such as deployment records."""
        msg, valid = request.canonical(), []
        for op, sig in signatures.items():
            if op not in self._keys:
                continue
            try:
                self._keys[op].verify(bytes.fromhex(sig), msg)
            except (InvalidSignature, ValueError):
                continue
            valid.append(op)
        return sorted(valid)

    def verify(self, request: ApprovalRequest, signatures: dict[str, str],
               threshold: int | None = None) -> list[str]:
        """Return the approving operators, or raise QuorumError.

        Consumes the nonce on success so the approval is single-use.
        """
        need = self.threshold if threshold is None else threshold
        valid = self._valid(request, signatures)
        if len(valid) < need:
            raise QuorumError(f"need {need} valid operator signatures, got {len(valid)}")
        self._nonces.consume(request.nonce)
        return valid
