"""ContainedModel: the only interface through which a model may act.

The model never touches tools, the network or the filesystem directly. Every
generation and tool call goes through this wrapper, which enforces, in order:

    preflight passed -> kill switch -> input screening -> budget
    -> capability policy -> (quorum approval) -> execute
    -> output screening -> audit log

Repeated denials (probing for a weak spot) trip the kill switch.
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Protocol

from .audit import AuditLog
from .killswitch import Halted, KillSwitch
from .policy import DenialTracker, Policy, PolicyViolation, Risk
from .quorum import ApprovalRequest, Quorum, QuorumError


class GenerateFn(Protocol):
    def __call__(self, prompt: str, max_tokens: int,
                 should_stop: Callable[[], bool]) -> str: ...


def _fingerprint(text: str) -> str:
    # Log hashes + lengths, not raw content, so the audit log itself can't leak data.
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class ContainedModel:
    def __init__(
        self,
        generate: GenerateFn,
        policy: Policy,
        kill_switch: KillSwitch,
        quorum: Quorum,
        audit: AuditLog,
        tools: dict[str, Callable[..., Any]] | None = None,
        require_preflight: bool = True,
    ):
        self._generate = generate
        self.policy = policy
        self.kill = kill_switch
        self.quorum = quorum
        self.audit = audit
        self._tools = dict(tools or {})
        self._denials = DenialTracker(policy.max_denials, policy.denial_window_s)
        self._ready = not require_preflight

    @property
    def registered_tools(self) -> list[str]:
        return sorted(self._tools)

    def start(self, test_sandbox: bool = False):
        """Run preflight; serving is refused until it passes."""
        from .preflight import preflight
        report = preflight(self, test_sandbox=test_sandbox)
        if not report.passed:
            raise Halted("preflight failed: " + "; ".join(report.failures))
        self._ready = True
        return report

    def _guard(self) -> None:
        if not self._ready:
            raise Halted("preflight has not passed; call start() first")
        self.kill.check()

    def _deny(self, what: str, reason: str) -> PolicyViolation:
        self.audit.append("denied", what=what, reason=reason)
        if self._denials.record():
            self.kill.trip(f"repeated policy denials (possible probing), last: {what}",
                           by="denial_tracker")
        return PolicyViolation(reason)

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        self._guard()
        if hit := self.policy.screen_input(prompt):
            raise self._deny("generate", f"input blocked: {hit}")
        max_tokens = max(1, min(max_tokens, self.policy.max_tokens_per_call))
        self.policy.budget.charge(calls=1, tokens=max_tokens)
        self.audit.append("generate.start", prompt=_fingerprint(prompt),
                          prompt_len=len(prompt), max_tokens=max_tokens)
        # should_stop lets the kill switch interrupt mid-generation, per token.
        out = self._generate(prompt, max_tokens, should_stop=lambda: self.kill.tripped)
        if not isinstance(out, str):
            raise self._deny("generate", "backend returned non-text output")
        self.kill.check()
        self._screen(out, "generate")
        self.audit.append("generate.end", output=_fingerprint(out), output_len=len(out))
        return out

    def request_tool(self, name: str, **kwargs: Any) -> ApprovalRequest:
        """Create an approval request that operators sign out-of-band."""
        req = ApprovalRequest(action=f"tool:{name}", params=kwargs)
        self.audit.append("tool.approval_requested", tool=name, nonce=req.nonce,
                          params=_fingerprint(repr(sorted(kwargs.items()))))
        return req

    def call_tool(self, name: str, approval: tuple[ApprovalRequest, dict[str, str]]
                  | None = None, **kwargs: Any) -> Any:
        self._guard()
        risk = self.policy.tool_risk(name)
        if risk is Risk.FORBIDDEN or name not in self._tools:
            raise self._deny(f"tool:{name}", f"tool '{name}' is not permitted")
        approvers: list[str] = []
        if risk is Risk.HIGH:
            if approval is None:
                raise self._deny(f"tool:{name}",
                                 f"tool '{name}' requires operator approval")
            req, sigs = approval
            if req.action != f"tool:{name}" or req.params != kwargs:
                raise self._deny(f"tool:{name}", "approval does not match this exact call")
            try:
                approvers = self.quorum.verify(req, sigs)
            except QuorumError as e:
                raise self._deny(f"tool:{name}", str(e)) from e
        self.policy.budget.charge(calls=1)
        self.audit.append("tool.call", tool=name, approvers=approvers)
        try:
            result = self._tools[name](**kwargs)
        except Exception as e:
            self.audit.append("tool.error", tool=name, error=type(e).__name__)
            raise
        self._screen(str(result), f"tool:{name}")
        self.kill.check()
        return result

    def _screen(self, text: str, source: str) -> None:
        hit = self.policy.tripwire(text)
        if hit:
            self.kill.trip(f"output tripwire in {source}: {hit}", by="tripwire")
            raise Halted("output tripwire fired; system halted pending review")


__all__ = ["ContainedModel", "GenerateFn"]
