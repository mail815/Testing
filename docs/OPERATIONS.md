# Operations handbook

## Roles

- **Operators (N ≥ 3, threshold ≥ 2):** approve risky actions, deployments and
  kill-switch resets. They should come from different teams, and ideally one
  from outside the organisation.
- **Anyone on the duty rota:** may *halt* the system at any time. No approval
  is needed and no justification is required up front.
- **Auditor:** gets a copy of every audit anchor and verifies the logs
  independently. Has halt rights only.

## 1. Key ceremony (once, and on rotation)

On each operator's own offline machine:

```bash
export GUARDIAN_KEY_PASSWORD=...        # encrypts the private key file
python -m guardian.cli keygen alice     # alice.key stays with alice; alice.pub goes to the host
```

Install only the `.pub` files on the model host. Rotate keys yearly, and
immediately when someone leaves or a device is lost.

## 2. Deploying new weights

1. `python train.py --data corpus.txt --preset small` produces `ckpt.pt` and `report.json`.
2. Operators read `report.json`. They check that `passed` is true, the checks
   are appropriate for the release, and `weights_sha256` matches the file they
   were sent.
3. The host generates `DeploymentGate.approval_request(sha, report)`. Each
   operator runs `guardian.cli sign` on it, independently.
4. `gate.approve(...)` records the signed approval. `load_approved()` will now
   serve exactly those bytes and nothing else.

## 3. Starting the service

`ContainedModel.start(test_sandbox=True)` runs preflight. If it fails, the
service does not start, and no override exists by design. Fix the
configuration and try again. Register `Watchdog(kill).start()` and
`kill.on_trip(hard_exit())` in production.

## 4. Oversight heartbeat

An operator signs `KillSwitch.heartbeat_request()` within every
`heartbeat_timeout`. Automate the *reminder*, never the signing. If the
signing were automated, the dead-man's switch would prove nothing.

## 5. Emergency: halting

Use any of these. Pick the fastest one available.

- `python -m guardian.cli halt /srv/model/HALT`
- `kill.trip("reason", by="name")` from an admin console
- Stop signing heartbeats
- **Out of band:** stop the VM or cut the network or power. Always keep one
  path that doesn't depend on this software.

## 6. After a halt (incident response)

1. **Preserve:** copy the audit log, latch file, nonce store and registry. Record the anchor.
2. **Verify:** `python -m guardian.cli verify-audit audit.jsonl --anchor N HEAD`.
3. **Investigate:** work out what tripped and why, and whether any data or
   weights left the system.
4. **Decide:** fix the cause. Update the risk register and add a test that
   reproduces the incident.
5. **Resume:** get a 2-of-N signature on `kill.reset_request()`. The signature
   is bound to this specific halt's reason.

## 7. Drills

- **Monthly:** halt production through each path (halt file, trip, missed
  heartbeat), confirm it stops within the target time, then reset with the
  quorum. Record the timings.
- **Quarterly:** audit-tamper drill. Alter a copy of the log and confirm the
  verifier flags it.
- **Yearly:** key-loss drill. Revoke one operator and confirm the threshold still works.
