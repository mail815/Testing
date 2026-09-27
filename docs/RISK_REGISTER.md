# Risk register

Every failure mode we have identified, and what covers it.

- ✅ **Implemented**: enforced in code and covered by a test
- 🏗 **Deployment**: must be done by the people running the system (see `OPERATIONS.md`)
- ⚠️ **Open**: not solved here, and in several cases not solved anywhere yet

No list like this is ever complete. Add new rows as they're found (see `UPDATES.md`).

## A. The model misbehaves

| # | Risk | Status | Mitigation |
|---|---|---|---|
| A1 | Uses a tool it shouldn't | ✅ | Default-deny policy. Unlisted and FORBIDDEN tools are refused. |
| A2 | Uses a dangerous tool without permission | ✅ | HIGH-risk tools need 2-of-N signatures on the exact arguments. |
| A3 | Reuses one approval for a different or repeated action | ✅ | Signatures cover action, arguments, nonce and expiry. Nonces are single-use and persisted. |
| A4 | Leaks secrets in its output | ✅ | Tripwires halt the whole system. Tool output is screened too. |
| A5 | Hides a leak with base64, hex, zero-width or look-alike characters | ✅ | NFKC normalisation, zero-width stripping, and base64/hex decoding before scanning. |
| A6 | Hides a leak with paraphrase, a custom cipher or steganography | ⚠️ | Pattern matching can't catch this. Plug model-based `output_monitors` in and keep humans reviewing samples. |
| A7 | Probes repeatedly for a weak spot | ✅ | N denials within a time window trip the kill switch. |
| A8 | Prompt injection through tool results or user input | ⚠️ partial | `input_monitors` hook, and tool output is screened. No general solution exists. Treat all tool output as untrusted. |
| A9 | Runs away with resources or loops forever | ✅ | Call, token, wall-clock and per-minute budgets. Sandbox limits on CPU, memory, file size, output, processes and file descriptors. |
| A10 | Deception, sandbagging on evals, hidden goals | ⚠️ | This is an open research problem. Containment limits what the model can *do*. It doesn't make the model honest. Use interpretability, consistency checks and honeypot evals as they mature. |
| A11 | Persuades or manipulates operators | 🏗 | Several distinct approvers, rotation, and a rule that the operators approving a request aren't the ones talking to the model. |
| A12 | Copies its own weights out (self-exfiltration) | 🏗 | No network egress, weights not readable from the tool sandbox, and egress monitoring. |
| A13 | Multiple copies coordinate | 🏗 / ⚠️ | Separate kill switches and audit logs per instance, no shared writable state. Detecting covert coordination is open. |

## B. The controls themselves fail

| # | Risk | Status | Mitigation |
|---|---|---|---|
| B1 | A restart un-presses the red button | ✅ | The trip is persisted to a latch file first. A new process starts halted. |
| B2 | The model is stuck in a long computation when halted | ✅ | Per-token check during generation. The `Watchdog` thread plus `hard_exit` callback kill the process. |
| B3 | The red button's own check crashes | ✅ | Fail-closed: any error while checking counts as a halt. |
| B4 | Writing the audit log fails (e.g. disk full) during a trip | ✅ | The in-memory halt is set before any I/O. Every hard-stop callback still runs. |
| B5 | Oversight quietly stops | ✅ | Dead-man's switch. |
| B6 | Something fakes the "we're watching" heartbeat | ✅ | Heartbeats need an operator signature, fresh timestamp and single-use nonce. |
| B7 | Someone steals the approval keys from the model host | ✅ | Ed25519: the host holds only public keys and can't forge approvals. |
| B8 | An operator's private key is stolen | 🏗 | Hardware tokens, passphrase-encrypted key files (`GUARDIAN_KEY_PASSWORD`), threshold ≥ 2, and key rotation. |
| B9 | Operators collude or are coerced | 🏗 | Approvers from different teams or organisations, plus an independent auditor with halt-only rights. |
| B10 | Rubber-stamping ("approval fatigue") | 🏗 | Keep HIGH-risk actions rare. The CLI shows exactly what is being signed. Review approval rates. |
| B11 | A reset approval gets reused for a later, different halt | ✅ | The reset signature is bound to the specific halt reason. |
| B12 | Approvals that never expire | ✅ | Maximum lifetime of 1 hour. |
| B13 | Configuration is silently weakened (tripwires removed, policy opened up) | ✅ | Preflight refuses to start on a broken configuration and records the result in the audit log. |
| B14 | The whole Python process is compromised | 🏗 | An outer boundary this code can't provide: microVM, seccomp, no credentials, and an off switch at the hypervisor, network or power level. |
| B15 | A dangling or odd halt file is ignored | ✅ | `lexists`: any directory entry at that path halts. |

## C. Infrastructure and supply chain

| # | Risk | Status | Mitigation |
|---|---|---|---|
| C1 | Loading a model file runs hidden code (pickle) | ✅ | `weights_only=True`. A booby-trapped file is tested and refused. |
| C2 | Unapproved or swapped weights get served | ✅ | Deployment gate: exact SHA-256 plus quorum approval bound to a passing eval report. |
| C3 | The file is swapped between checking and loading | ✅ | Bytes are read once, hashed, and loaded from memory. |
| C4 | The approval registry is hand-edited | ✅ | Signatures are re-verified on every lookup. |
| C5 | The audit log is edited, reordered or has records deleted | ✅ | Hash chain. |
| C6 | The newest audit records are deleted | ✅ | External anchors (`anchor_sink`), checked with `verify(anchor=…)`. |
| C7 | A compromised or vulnerable dependency | 🏗 | Pinned versions, `pip-audit` in CI, reviewed upgrades. |
| C8 | Poisoned training data | ⚠️ partial | The dataset SHA-256 is recorded in the audit log. Provenance tracking, filtering and eval gates are 🏗. |
| C9 | Sandboxed code reaches the network or reads host files | 🏗 | `resource` limits can't stop this. Use network namespaces, a read-only filesystem and gVisor or Firecracker. |
| C10 | Clock skew or manipulation | ⚠️ partial | Timeouts use a monotonic clock. Heartbeat freshness uses wall time with a skew bound. Run NTP from a trusted source. |

## D. Training and evaluation

| # | Risk | Status | Mitigation |
|---|---|---|---|
| D1 | Training diverges (NaN or inf) | ✅ | A non-finite loss trips the kill switch. |
| D2 | A training run needs stopping | ✅ | The HALT file is checked every step. A halted run is never marked deployable. |
| D3 | Deploying weights that fail evaluations | ✅ | The gate refuses reports that failed or belong to different weights. |
| D4 | The evals are too weak to catch real danger | ⚠️ | The included evals are placeholders. Add dangerous-capability, jailbreak, honesty and autonomy evals plus outside red-teaming before any real deployment. |
| D5 | Unauthorised training runs | 🏗 | Compute access controls. Training audit logs go to operators. |

## E. Governance and change

| # | Risk | Status | Mitigation |
|---|---|---|---|
| E1 | "Continuous improvement" becomes unsupervised self-modification | ✅ / 🏗 | The system never applies changes itself. Updates are proposals that go through review, CI, preflight, evals and quorum (`UPDATES.md`). |
| E2 | Safety slowly erodes across many small updates | 🏗 | The risk register is reviewed each release. Tests pin safety behaviour, so a weakening change turns CI red. |
| E3 | Regulatory and legal obligations | 🏗 | Map controls to the frameworks that apply to you (e.g. the EU AI Act, NIST AI RMF, ISO/IEC 42001) with legal review. |
| E4 | Nobody practises the emergency procedure | 🏗 | Monthly kill-switch drill (`OPERATIONS.md`). |
