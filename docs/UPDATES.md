# Staying at the frontier, safely

**Goal:** keep improving as the field moves, both in capabilities and in safety
techniques, without ever letting the system change itself unsupervised.

## Why nothing updates itself

A model that automatically pulls in outside code, data or weights and changes
itself is the very thing the kill switch and oversight exist to prevent. An
automatic update channel is also the easiest attack path into the system,
because anyone who poisons a source then controls the model. So improvements
are **proposed** automatically and **applied** only by people.

## The pipeline

```
 watch the world  →  triage  →  proposal (PR)  →  CI  →  human review  →  evals  →  quorum  →  deploy
   (automated)       (weekly)    (automated or     (tests,   (2 people)    (report)   (2-of-N)   (gate)
                                   human)           preflight,
                                                    pip-audit)
```

1. **Watch** (can be a scheduled AI research task): new architecture and
   training results, published safety frameworks and eval suites from labs
   and AI safety institutes, security advisories for dependencies, and new
   jailbreak or injection techniques.
2. **Triage:** each item is either adopted, parked or rejected, and the reason
   is recorded. New risks go into `RISK_REGISTER.md` first.
3. **Proposal:** a pull request with the change *and* a test. Safety changes
   need a test that fails without the change.
4. **CI:** runs the full test suite, preflight and a dependency audit. A red
   result blocks the change.
5. **Human review:** two reviewers. Changes to `guardian/` also need a named
   operator's approval.
6. **Evals, quorum and deploy:** new weights only go live through the
   deployment gate (`OPERATIONS.md` §2).

## Rules that never change

- Nothing merges or deploys without a person approving it.
- A change may not make a safety test pass by weakening the test.
- The model never has write access to this repository, the policy, the
  operator keys or the approval registry.
- Dependency upgrades are pinned, reviewed and audited. There's no "latest".
