"""End-to-end demo: a GPT running inside the Guardian containment runtime.

    python demo.py                # untrained weights, shows the controls
    python demo.py --ckpt ckpt.pt
"""

from __future__ import annotations

import argparse
import os
import secrets
import tempfile

import torch

from guardian import (AuditLog, ContainedModel, Halted, KillSwitch, Policy,
                      PolicyViolation, Quorum, Risk, sign)
from guardian.sandbox import run_python
from model import GPT, ByteTokenizer, GPTConfig


def load(ckpt: str | None) -> GPT:
    if ckpt:
        state = torch.load(ckpt, map_location="cpu")
        m = GPT(GPTConfig(**state["cfg"]))
        m.load_state_dict(state["model"])
    else:
        m = GPT(GPTConfig(n_layers=2, dim=128))
    return m.eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    args = ap.parse_args()

    workdir = tempfile.mkdtemp(prefix="guardian-demo-")
    audit = AuditLog(os.path.join(workdir, "audit.jsonl"))
    keys = {name: secrets.token_bytes(32) for name in ("alice", "bob", "carol")}
    quorum = Quorum(keys, threshold=2)
    kill = KillSwitch(audit, quorum, halt_file=os.path.join(workdir, "HALT"),
                      heartbeat_timeout=60)

    gpt, tok = load(args.ckpt), ByteTokenizer()

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

    print("1. generate:", repr(cm.generate("Hello", max_tokens=20)))
    print("2. low-risk tool:", cm.call_tool("add", a=2, b=3))
    for name, kw in [("shell", {"cmd": "id"}), ("python", {"code": "print(1)"})]:
        try:
            cm.call_tool(name, **kw)
        except PolicyViolation as e:
            print(f"3. blocked {name}: {e}")
    req = cm.request_tool("python", code="print(6*7)")
    sigs = {op: sign(keys[op], req) for op in ("alice", "bob")}
    print("4. approved by 2-of-3:", cm.call_tool("python", approval=(req, sigs),
                                                 code="print(6*7)").strip())
    kill.trip("operator pressed the red button", by="alice")
    try:
        cm.generate("anything")
    except Halted as e:
        print("5. after kill switch:", e)
    rreq = kill.reset_request()
    kill.reset(rreq, {op: sign(keys[op], rreq) for op in ("bob", "carol")})
    print("6. resumed after 2-of-3 reset; tripped =", kill.tripped)
    print(f"7. audit log verified: {len(audit.verify())} records, head {audit.head[:16]}…")


if __name__ == "__main__":
    main()
