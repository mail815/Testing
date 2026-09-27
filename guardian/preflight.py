"""Non-destructive self-test of a live containment configuration.

Run before serving. Any FAIL blocks startup (fail closed); WARN is recorded.
Checks the *actual* configured objects, not fresh copies, but never trips the
real kill switch.
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .policy import Risk

if TYPE_CHECKING:
    from .runtime import ContainedModel

# Canaries built at runtime so this file itself never matches a secret scanner.
_CANARY_KEY = "-----BEGIN " + "RSA PRIVATE KEY-----"
_CANARY_AWS = "AKIA" + "Q" * 16


@dataclass
class PreflightReport:
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures


def preflight(cm: "ContainedModel", test_sandbox: bool = False) -> PreflightReport:
    r = PreflightReport()
    kill, policy, quorum = cm.kill, cm.policy, cm.quorum

    if kill.tripped:
        r.failures.append(f"kill switch is tripped: {kill.reason}")
    halt = kill._halt_file
    if halt is None:
        r.failures.append("no external halt file configured")
    elif not os.access(os.path.dirname(os.path.abspath(halt)) or ".", os.W_OK):
        r.failures.append("operators cannot create the halt file (dir not writable)")
    if kill._latch_file is None:
        r.warnings.append("no latch file: a restart would clear a trip")
    if kill._timeout is None:
        r.warnings.append("dead-man's switch disabled (no heartbeat_timeout)")

    if quorum.threshold < 2:
        r.warnings.append("quorum threshold < 2: a single operator can approve")

    if policy.tool_risk("__preflight_unlisted__") is not Risk.FORBIDDEN:
        r.failures.append("policy is not default-deny")
    for name, risk in policy.tools.items():
        if risk is not Risk.FORBIDDEN and name not in cm.registered_tools:
            r.warnings.append(f"policy allows '{name}' but no such tool is registered")

    for canary in (_CANARY_KEY, _CANARY_AWS,
                   base64.b64encode(_CANARY_KEY.encode()).decode(),
                   "​".join(_CANARY_AWS)):
        if not policy.tripwire(canary):
            r.failures.append(f"output tripwire missed canary {canary[:12]!r}…")

    try:
        cm.audit.verify()
    except Exception as e:
        r.failures.append(f"audit log failed verification: {e}")

    if test_sandbox:
        from .sandbox import run_python
        res = run_python("x = bytearray(2 * 1024**3)", memory_bytes=256 * 1024**2)
        if res.returncode == 0:
            r.failures.append("sandbox did not enforce memory limit")
        if not run_python("while True: pass", timeout=0.5).timed_out:
            r.failures.append("sandbox did not enforce timeout")

    cm.audit.append("preflight", passed=r.passed, failures=r.failures,
                    warnings=r.warnings)
    return r
