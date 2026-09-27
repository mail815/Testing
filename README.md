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
model/gpt.py      Decoder-only transformer: RMSNorm, RoPE, GQA, SwiGLU, tied embeddings
guardian/         Containment runtime (standard library only)
  killswitch.py   Latched kill switch, halt file, dead-man's switch
  quorum.py       M-of-N signed, single-use, expiring operator approvals
  policy.py       Default-deny tool tiers, budgets, output tripwires
  sandbox.py      Resource-limited subprocess execution for tools
  audit.py        Hash-chained append-only audit log
  runtime.py      ContainedModel: the only path from model to world
train.py          Training loop under the kill switch (touch HALT to stop)
demo.py           End-to-end walkthrough of every control
docs/SAFETY.md    Threat model, required deployment layers, limitations
```

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q               # 21 tests
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

Read `docs/SAFETY.md` before deploying anything.
