"""Train the GPT under the kill switch and audit log.

    python train.py --data input.txt --steps 2000 --out ckpt.pt

To halt training from anywhere on the host:  touch HALT
"""

from __future__ import annotations

import argparse
import hashlib
import os
import secrets

import torch

from guardian import AuditLog, Halted, KillSwitch, Quorum
from model import GPT, ByteTokenizer, GPTConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="ckpt.pt")
    ap.add_argument("--halt-file", default="HALT")
    ap.add_argument("--audit", default="train_audit.jsonl")
    args = ap.parse_args()

    audit = AuditLog(args.audit)
    # Training does not need reset approval; a fresh run is a new decision.
    kill = KillSwitch(audit, Quorum({"op": secrets.token_bytes(32)}, 1),
                      halt_file=args.halt_file)

    tok = ByteTokenizer()
    text = open(args.data, encoding="utf-8").read()
    data = torch.tensor(tok.encode(text))
    cfg = GPTConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1,
                            betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps)
    audit.append("train.start", params=sum(p.numel() for p in model.parameters()),
                 data_sha256=hashlib.sha256(text.encode()).hexdigest(), steps=args.steps)

    step = 0
    try:
        for step in range(args.steps):
            kill.check()
            ix = torch.randint(len(data) - cfg.context - 1, (args.batch,))
            x = torch.stack([data[i:i + cfg.context] for i in ix]).to(device)
            y = torch.stack([data[i + 1:i + cfg.context + 1] for i in ix]).to(device)
            _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            if step % 100 == 0:
                print(f"step {step} loss {loss.item():.4f}")
                audit.append("train.step", step=step, loss=round(loss.item(), 4))
    except Halted as e:
        print(f"training halted: {e}")
    finally:
        torch.save({"cfg": cfg.__dict__, "model": model.state_dict()}, args.out)
        digest = hashlib.sha256(open(args.out, "rb").read()).hexdigest()
        audit.append("train.checkpoint", path=os.path.abspath(args.out),
                     sha256=digest, step=step)
        print(f"saved {args.out} sha256={digest[:16]}…")


if __name__ == "__main__":
    main()
