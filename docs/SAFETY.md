# Safety design: containment, kill switch, oversight

## Principles

1. **Stopping is cheap and unilateral; resuming is expensive and collective.**
   Any operator, a file on disk, a missed heartbeat, or an output tripwire can
   halt the system. Only an M-of-N operator quorum can resume it.
2. **Default deny.** A tool the policy doesn't list is forbidden, even if code
   registered it.
3. **The model is the adversary in the threat model.** Controls live outside
   the model, in code paths the model's outputs can't change. The model gets
   text in and text out, and nothing else.
4. **Everything is recorded, and edits are detectable.** Every decision goes
   into a hash-chained log. Prompts and outputs are stored as fingerprints, so
   the log can't leak data.
5. **Defence in depth.** This package is the *inner* layer. It has to run inside
   an outer isolation boundary.

## Controls implemented (`guardian/`)

| Control | Where | What it guarantees |
|---|---|---|
| Latched kill switch | `killswitch.py` | Once tripped, every guarded call raises `Halted` until a quorum resets it. |
| External halt file | `killswitch.py` | `touch HALT` from any process stops generation and training. |
| Dead-man's switch | `killswitch.py` | If operators stop sending heartbeats, the system halts itself. |
| Per-token interrupt | `runtime.py`, `model/gpt.py` | The kill switch is checked between every generated token. |
| M-of-N approval | `quorum.py` | High-risk actions need N distinct operator Ed25519 signatures over the exact action and arguments. Approvals are single-use (even across restarts), expire, and last at most 1 hour. The host holds only public keys. |
| Capability policy | `policy.py` | Tools are LOW, HIGH or FORBIDDEN. Unlisted tools are FORBIDDEN. |
| Budgets | `policy.py` | Hard caps on calls, tokens and wall-clock time. |
| Output tripwires | `policy.py`, `runtime.py` | A dangerous pattern in output (e.g. private keys) *halts* the system rather than just redacting it. |
| Tamper-evident audit | `audit.py` | Edits, deletions and reordering of past records are detected. External anchors also catch deletion of the newest records. |
| Persistent latch | `killswitch.py` | A trip is written to disk, so restarting the process doesn't un-press the button. |
| Signed heartbeats | `killswitch.py` | Only an operator's signature keeps the dead-man's switch alive. |
| Watchdog + hard exit | `killswitch.py` | A background thread enforces halts even if the main thread hangs. |
| Probing detection | `runtime.py` | Repeated policy denials trip the kill switch. |
| Preflight | `preflight.py` | Serving is refused until a self-test of the live configuration passes. |
| Deployment gate | `deploy.py`, `model/checkpoint.py` | Only quorum-approved weights that passed evals are served, loaded without running pickled code. |
| Sandboxed tools | `sandbox.py` | Tools run in a separate process with CPU, memory, file and process limits, a scrubbed environment and a temp working dir. A timeout kills the whole process group. |

## Required outer layers (deployment, not code)

The in-process controls assume the host itself is trustworthy. In production:

- **Isolation:** run the model and its tools in a microVM (Firecracker, or gVisor)
  with a read-only root filesystem, seccomp, no host mounts and no cloud credentials.
- **Network:** deny all egress except through an allow-listed, logging proxy.
- **Key custody:** operator keys stay on hardware tokens held by separate people.
  They never go on the model host, which holds only public keys.
- **Out-of-band kill:** keep a power or network cutoff that doesn't depend on
  this software at all (hypervisor stop, network ACL, power cut).
- **Audit anchoring:** regularly copy `AuditLog.head` to write-once storage
  controlled by a different team.
- **Weights security:** encrypt checkpoints at rest, record the SHA-256 digests
  that `train.py` writes to the audit log, and restrict and log access.
- **Evaluations before deployment:** run dangerous-capability evals, red-teaming
  and staged rollouts before widening access.

See `RISK_REGISTER.md` for the full list of risks, `OPERATIONS.md` for
procedures and `UPDATES.md` for how changes are made.

## Known limitations (read these)

- Output tripwires handle common encodings (base64, hex, Unicode tricks), but
  paraphrase or a custom cipher gets past them. Pair them with model-based
  `output_monitors` and human review.
- `resource` limits don't block network access or filesystem reads. That's the
  outer sandbox's job.
- Nothing here addresses *alignment* of the model itself (deceptive behaviour,
  sandbagging, and so on). Containment limits what a model can *do*, but it
  doesn't make the model trustworthy.
