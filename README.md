# Contained LM

A modern transformer language model wrapped in a containment and human-oversight
runtime: a latched kill switch, a dead-man's switch, M-of-N operator approval,
default-deny tools, sandboxed execution and a tamper-evident audit log.

## Scope, honestly

This repo gives you the **architecture and the safety controls**. It does not
give you a model that beats today's frontier LLMs. That takes thousands of
accelerators, trillions of tokens of curated data, months of training,
post-training (instruction tuning and RLHF-style methods) and a large evaluation
effort. The model code scales by changing `GPTConfig`. The containment layer
doesn't depend on the model, so you can wrap any `generate` function, including
a hosted API, in `ContainedModel`.

## Layout

```
model/
  gpt.py          Transformer: RMSNorm, RoPE, GQA, SwiGLU, KV cache, size presets
  checkpoint.py   Safe save/load (no pickle code execution), gated serving
  evals.py        Pre-deployment evaluation report
guardian/         Containment runtime (depends only on `cryptography`)
  killswitch.py   Persistent latched kill switch, halt file, signed dead-man's switch, watchdog
  quorum.py       M-of-N Ed25519 approvals: single-use, expiring, replay-proof
  policy.py       Default-deny tools, budgets, rate limits, obfuscation-aware tripwires, monitors
  preflight.py    Self-test of the live configuration; serving refused until it passes
  deploy.py       Deployment gate: only approved, eval-passing weights are served
  sandbox.py      Resource-limited, output-capped subprocess execution
  audit.py        Hash-chained audit log with external anchoring
  runtime.py      ContainedModel: the only path from model to world
  cli.py          Operator tool: keygen, sign, halt, verify-audit
train.py          Training under the kill switch, ending in an eval report
demo.py           End-to-end walkthrough of every control
docs/             SAFETY, RISK_REGISTER, OPERATIONS, UPDATES
```

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q               # 48 tests
python demo.py                    # see each control in action
python train.py --data input.txt  # touch HALT to stop training at any point
```

## Explained simply

- **The brain** (`model/`) learns to guess the next letter, over and over,
  until it gets good at language.
- **The cage** (`sandbox.py`, `policy.py`) means the brain can only use the toys
  it's given, and only in a padded room.
- **The big red button** (`killswitch.py`): anyone in charge can press it, and it
  stays pressed. If the grown-ups stop checking in, it presses itself.
- **Grown-ups watching** (`quorum.py`): risky actions and un-pressing the button
  need two or more grown-ups to sign off.
- **The diary** (`audit.py`): everything goes in a diary where ripping out or
  changing a page is always noticed.
- **The report card and sign-off** (`evals.py`, `deploy.py`): a new brain has
  to pass its tests, and grown-ups have to sign for that exact brain, before it
  can be used.
- **The pre-flight check** (`preflight.py`): before starting, the system
  checks that its own safety features work, and refuses to start if not.

Read `docs/SAFETY.md` and `docs/RISK_REGISTER.md` before deploying anything.
