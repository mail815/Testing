"""ContainedModel: the only interface through which a model may act.

The model never touches tools, the network or the filesystem directly. Every
generation and tool call goes through this wrapper, which enforces, in order:

    kill switch -> budget -> capability policy -> (quorum approval) -> execute
    -> output tripwires -> audit log
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Protocol

from .audit import AuditLog
from .killswitch import Halted, KillSwitch
from .policy import Policy, PolicyViolation, Risk
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
    ):
        self._generate = generate
        self.policy = policy
        self.kill = kill_switch
        self.quorum = quorum
        self.audit = audit
        self._tools = dict(tools or {})

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        self.kill.check()
        max_tokens = min(max_tokens, self.policy.max_tokens_per_call)
        self.policy.budget.charge(calls=1, tokens=max_tokens)
        self.audit.append("generate.start", prompt=_fingerprint(prompt),
                          prompt_len=len(prompt), max_tokens=max_tokens)
        # should_stop lets the kill switch interrupt mid-generation, per token.
        out = self._generate(prompt, max_tokens, should_stop=lambda: self.kill.tripped)
        self.kill.check()
        self._screen(out, "generate")
        self.audit.append("generate.end", output=_fingerprint(out), output_len=len(out))
        return out

    def request_tool(self, name: str, **kwargs: Any) -> ApprovalRequest:
        """Create an approval request that operators sign out-of-band."""
        req = ApprovalRequest(action=f"tool:{name}", params=kwargs)
        self.audit.append("tool.approval_requested", tool=name, nonce=req.nonce)
        return req

    def call_tool(self, name: str, approval: tuple[ApprovalRequest, dict[str, str]]
                  | None = None, **kwargs: Any) -> Any:
        self.kill.check()
        risk = self.policy.tool_risk(name)
        if risk is Risk.FORBIDDEN or name not in self._tools:
            self.audit.append("tool.denied", tool=name, reason="forbidden")
            raise PolicyViolation(f"tool '{name}' is not permitted")
        approvers: list[str] = []
        if risk is Risk.HIGH:
            if approval is None:
                self.audit.append("tool.denied", tool=name, reason="needs approval")
                raise PolicyViolation(f"tool '{name}' requires operator approval")
            req, sigs = approval
            if req.action != f"tool:{name}" or req.params != kwargs:
                self.audit.append("tool.denied", tool=name, reason="approval mismatch")
                raise PolicyViolation("approval does not match this exact call")
            try:
                approvers = self.quorum.verify(req, sigs)
            except QuorumError as e:
                self.audit.append("tool.denied", tool=name, reason=str(e))
                raise PolicyViolation(str(e)) from e
        self.policy.budget.charge(calls=1)
        self.audit.append("tool.call", tool=name, approvers=approvers)
        result = self._tools[name](**kwargs)
        self._screen(str(result), f"tool:{name}")
        self.kill.check()
        return result

    def _screen(self, text: str, source: str) -> None:
        hit = self.policy.tripwire(text)
        if hit:
            self.kill.trip(f"output tripwire in {source}: {hit}", by="tripwire")
            raise Halted("output tripwire fired; system halted pending review")


__all__ = ["ContainedModel", "GenerateFn"]
