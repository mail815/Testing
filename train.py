"""Train the GPT under the kill switch and audit log, then evaluate it.

    python train.py --data input.txt --preset tiny --steps 2000

To halt training from anywhere on the host:  touch HALT
The run ends with an evaluation report (report.json) that operators review
and sign before the weights can be served (see docs/OPERATIONS.md).
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os

import torch

from guardian import AuditLog, Halted, KillSwitch, Quorum, generate_operator_key
from model import GPT, ByteTokenizer
from model.checkpoint import save
from model.evals import evaluate
from model.gpt import PRESETS, preset


def lr_at(step: int, total: int, peak: float, warmup: int) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--preset", default="tiny", choices=sorted(PRESETS))
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation steps")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--max-val-loss", type=float, default=2.5)
    ap.add_argument("--out", default="ckpt.pt")
    ap.add_argument("--report", default="report.json")
    ap.add_argument("--halt-file", default="HALT")
    ap.add_argument("--audit", default="train_audit.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    audit = AuditLog(args.audit)
    # Training halts but never needs to resume in-process: a new run is a new decision.
    kill = KillSwitch(audit, Quorum({"n/a": generate_operator_key()[1]}, 1),
                      halt_file=args.halt_file)

    text = open(args.data, encoding="utf-8").read()
    data = torch.tensor(ByteTokenizer().encode(text))
    split = int(0.9 * len(data))
    train, val = data[:split], data[split:]
    cfg = preset(args.preset)
    if len(val) <= cfg.context + 1:
        raise SystemExit(f"need more data: validation split must exceed {cfg.context} bytes")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT(cfg).to(device)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), fused=device == "cuda")
    amp = (torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda"
           else contextlib.nullcontext())
    audit.append("train.start", preset=args.preset, seed=args.seed,
                 params=sum(p.numel() for p in model.parameters()),
                 data_sha256=hashlib.sha256(text.encode()).hexdigest(), steps=args.steps)

    step, halted = 0, False
    try:
        for step in range(args.steps):
            kill.check()
            for g in opt.param_groups:
                g["lr"] = lr_at(step, args.steps, args.lr, args.warmup)
            for _ in range(args.accum):
                ix = torch.randint(len(train) - cfg.context - 1, (args.batch,))
                x = torch.stack([train[i:i + cfg.context] for i in ix]).to(device)
                y = torch.stack([train[i + 1:i + cfg.context + 1] for i in ix]).to(device)
                with amp:
                    _, loss = model(x, y)
                (loss / args.accum).backward()
            if not torch.isfinite(loss):
                kill.trip(f"non-finite loss at step {step}", by="trainer")
                kill.check()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            if step % 100 == 0:
                print(f"step {step} loss {loss.item():.4f}")
                audit.append("train.step", step=step, loss=round(loss.item(), 4))
    except Halted as e:
        halted = True
        print(f"training halted: {e}")
    finally:
        save(model.cpu(), args.out)

    report = evaluate(args.out, val, args.max_val_loss)
    report.update(steps_completed=step + 1, halted_early=halted)
    if halted:
        report["passed"] = False  # a halted run is never deployable as-is
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    audit.append("train.evaluated", report=report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
