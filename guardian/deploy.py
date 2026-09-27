"""Deployment gate: only quorum-approved, eval-passing weights may be served.

A deployment approval binds the SHA-256 of a checkpoint to the SHA-256 of the
evaluation report it passed. Approvals are stored *with their signatures* in a
registry file, and every lookup re-verifies those signatures against operator
public keys, so hand-editing the registry cannot approve anything.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from typing import Any

from .audit import AuditLog
from .quorum import ApprovalRequest, Quorum, QuorumError


class DeploymentError(Exception):
    pass


def sha256_file(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def report_digest(report: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()


class DeploymentGate:
    def __init__(self, quorum: Quorum, audit: AuditLog,
                 registry: str | os.PathLike[str]):
        self._quorum, self._audit = quorum, audit
        self._registry = os.fspath(registry)
        self._lock = threading.Lock()

    @staticmethod
    def approval_request(weights_sha256: str, report: dict[str, Any]) -> ApprovalRequest:
        return ApprovalRequest("deploy.weights", {
            "weights_sha256": weights_sha256, "report_sha256": report_digest(report)})

    def approve(self, weights_sha256: str, report: dict[str, Any],
                request: ApprovalRequest, signatures: dict[str, str]) -> None:
        if report.get("passed") is not True:
            raise DeploymentError("evaluation report did not pass; refusing to deploy")
        if report.get("weights_sha256") != weights_sha256:
            raise DeploymentError("evaluation report is for different weights")
        expected = self.approval_request(weights_sha256, report).params
        if request.action != "deploy.weights" or request.params != expected:
            raise DeploymentError("approval does not match these weights and report")
        approvers = self._quorum.verify(request, signatures)
        entry = {"request": {"action": request.action, "params": request.params,
                             "nonce": request.nonce, "expires_at": request.expires_at},
                 "signatures": signatures}
        with self._lock:
            with open(self._registry, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, sort_keys=True) + "\n")
                f.flush()
                os.fsync(f.fileno())
        self._audit.append("deploy.approved", weights_sha256=weights_sha256,
                           report_sha256=expected["report_sha256"], approvers=approvers)

    def is_approved(self, weights_sha256: str) -> bool:
        if not os.path.exists(self._registry):
            return False
        with open(self._registry, encoding="utf-8") as f:
            for line in f:
                try:
                    e = json.loads(line)
                    req = ApprovalRequest(**e["request"])
                    if (req.action == "deploy.weights"
                            and req.params.get("weights_sha256") == weights_sha256
                            and len(self._quorum.valid_signers(req, e["signatures"]))
                            >= self._quorum.threshold):
                        return True
                except (ValueError, KeyError, TypeError, QuorumError):
                    continue
        return False

    def require(self, path: str | os.PathLike[str]) -> bytes:
        """Read the file once and return its bytes only if they are approved.

        Callers must deserialize the *returned bytes*, never re-open the path:
        re-reading would let the file be swapped between check and use.
        """
        with open(path, "rb") as f:
            data = f.read()
        digest = hashlib.sha256(data).hexdigest()
        if not self.is_approved(digest):
            self._audit.append("deploy.rejected", path=os.fspath(path), sha256=digest)
            raise DeploymentError(f"weights {digest[:16]}… are not approved for serving")
        self._audit.append("deploy.loaded", sha256=digest)
        return data
