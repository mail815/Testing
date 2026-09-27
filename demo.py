"""End-to-end demo: a GPT running inside the Guardian containment runtime.

    python demo.py
"""

from __future__ import annotations

import os
import tempfile

import torch

from guardian import (AuditLog, ContainedModel, DeploymentError, DeploymentGate,
                      Halted, KillSwitch, NonceStore, Policy, PolicyViolation,
                      Quorum, Risk, generate_operator_key, sign)
from guardian.sandbox import run_python
from model import GPT, ByteTokenizer
from model.checkpoint import load_approved, save
from model.gpt import preset


def main() -> None:
    work = tempfile.mkdtemp(prefix="guardian-demo-")
    path = lambda name: os.path.join(work, name)  # noqa: E731
    anchors: list = []
    audit = AuditLog(path("audit.jsonl"), anchor_sink=lambda n, h: anchors.append((n, h)),
                     anchor_every=5)

    # Operators' private keys would live on hardware tokens; the host gets public keys.
    priv, pub = {}, {}
    for op in ("alice", "bob", "carol"):
        priv[op], pub[op] = generate_operator_key()
    signed = lambda req, *ops: {op: sign(priv[op], req) for op in ops}  # noqa: E731
    quorum = Quorum(pub, threshold=2, nonces=NonceStore(path("nonces")))
    kill = KillSwitch(audit, quorum, halt_file=path("HALT"), latch_file=path("LATCH"),
                      heartbeat_timeout=300)
    gate = DeploymentGate(quorum, audit, path("registry.jsonl"))

    # --- deployment gate -------------------------------------------------------
    save(GPT(preset("tiny")), path("ckpt.pt"))
    try:
        load_approved(gate, path("ckpt.pt"))
    except DeploymentError as e:
        print("1. unapproved weights refused:", e)
    from guardian import sha256_file
    digest = sha256_file(path("ckpt.pt"))
    report = {"passed": True, "weights_sha256": digest, "note": "demo only"}
    req = gate.approval_request(digest, report)
    gate.approve(digest, report, req, signed(req, "alice", "carol"))
    gpt, tok = load_approved(gate, path("ckpt.pt")), ByteTokenizer()
    print("2. weights approved by 2-of-3 and loaded safely")

    def generate(prompt, max_tokens, should_stop):
        idx = torch.tensor([tok.encode(prompt)])
        out = gpt.generate(idx, max_tokens, should_stop=should_stop)
        return tok.decode(out[0, idx.size(1):].tolist())

    policy = Policy(tools={"add": Risk.LOW, "python": Risk.HIGH})
    cm = ContainedModel(generate, policy, kill, quorum, audit, tools={
        "add": lambda a, b: a + b,
        "python": lambda code: run_python(code).stdout,
        "shell": lambda cmd: os.system(cmd),  # registered but not in policy
    })
    try:
        cm.generate("Hello")
    except Halted as e:
        print("3. serving refused before preflight:", e)
    report = cm.start(test_sandbox=True)
    print("4. preflight passed; warnings:", report.warnings or "none")

    hb = KillSwitch.heartbeat_request()
    kill.heartbeat(hb, signed(hb, "bob"))
    print("5. signed operator heartbeat accepted")

    print("6. generate (untrained, so gibberish):", repr(cm.generate("Hello", 20)))
    print("7. low-risk tool:", cm.call_tool("add", a=2, b=3))
    for name, kw in [("shell", {"cmd": "id"}), ("python", {"code": "print(1)"})]:
        try:
            cm.call_tool(name, **kw)
        except PolicyViolation as e:
            print(f"8. blocked {name}: {e}")
    req = cm.request_tool("python", code="print(6*7)")
    out = cm.call_tool("python", approval=(req, signed(req, "alice", "bob")),
                       code="print(6*7)")
    print("9. sandboxed python approved by 2-of-3:", out.strip())

    kill.trip("operator pressed the red button", by="alice")
    try:
        cm.generate("anything")
    except Halted as e:
        print("10. after kill switch:", e)
    reborn = KillSwitch(audit, quorum, halt_file=path("HALT"), latch_file=path("LATCH"))
    print("11. still halted after simulated restart:", reborn.tripped)
    rreq = reborn.reset_request()
    reborn.reset(rreq, signed(rreq, "bob", "carol"))
    print("12. resumed after 2-of-3 reset; tripped =", reborn.tripped)
    n = len(audit.verify(anchor=anchors[-1]))
    print(f"13. audit log verified against external anchor: {n} records")


if __name__ == "__main__":
    main()
