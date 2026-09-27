import json
import secrets
import time

import pytest

from guardian import (ApprovalRequest, AuditIntegrityError, AuditLog, Budget,
                      ContainedModel, Halted, KillSwitch, Policy,
                      PolicyViolation, Quorum, QuorumError, Risk, sign)
from guardian.sandbox import run_python

KEYS = {n: secrets.token_bytes(32) for n in ("alice", "bob", "carol")}


@pytest.fixture
def env(tmp_path):
    audit = AuditLog(tmp_path / "audit.jsonl")
    quorum = Quorum(KEYS, threshold=2)
    kill = KillSwitch(audit, quorum, halt_file=tmp_path / "HALT")
    return tmp_path, audit, quorum, kill


def sigs(req, *ops):
    return {op: sign(KEYS[op], req) for op in ops}


def make_model(env, gen=None, **policy_kw):
    _, audit, quorum, kill = env
    gen = gen or (lambda p, n, should_stop: "ok")
    policy = Policy(tools={"add": Risk.LOW, "danger": Risk.HIGH,
                           "banned": Risk.FORBIDDEN}, **policy_kw)
    tools = {"add": lambda a, b: a + b, "danger": lambda x: f"did {x}",
             "banned": lambda: "never", "unlisted": lambda: "never"}
    return ContainedModel(gen, policy, kill, quorum, audit, tools)


# -- audit log ---------------------------------------------------------------
def test_audit_chain_verifies_and_survives_reopen(tmp_path):
    log = AuditLog(tmp_path / "a.jsonl")
    for i in range(5):
        log.append("e", i=i)
    head = log.head
    reopened = AuditLog(tmp_path / "a.jsonl")
    assert reopened.head == head
    reopened.append("e", i=5)
    assert len(reopened.verify()) == 6


@pytest.mark.parametrize("tamper", ["edit", "delete", "reorder"])
def test_audit_detects_tampering(tmp_path, tamper):
    p = tmp_path / "a.jsonl"
    log = AuditLog(p)
    for i in range(4):
        log.append("e", i=i)
    lines = p.read_text().splitlines()
    if tamper == "edit":
        rec = json.loads(lines[1]); rec["data"]["i"] = 99; lines[1] = json.dumps(rec)
    elif tamper == "delete":
        del lines[1]
    else:
        lines[1], lines[2] = lines[2], lines[1]
    p.write_text("\n".join(lines) + "\n")
    with pytest.raises(AuditIntegrityError):
        log.verify()


# -- quorum ------------------------------------------------------------------
def test_quorum_requires_threshold_distinct_valid_signatures():
    q = Quorum(KEYS, 2)
    req = ApprovalRequest("x", {})
    with pytest.raises(QuorumError):
        q.verify(req, sigs(req, "alice"))
    with pytest.raises(QuorumError):  # forged second signature
        q.verify(req, {**sigs(req, "alice"), "bob": "00" * 32})
    with pytest.raises(QuorumError):  # unknown operator
        q.verify(req, {**sigs(req, "alice"), "mallory": sign(b"k", req)})
    assert q.verify(req, sigs(req, "alice", "bob")) == ["alice", "bob"]


def test_quorum_rejects_replay_expiry_and_tampered_params():
    q = Quorum(KEYS, 2)
    req = ApprovalRequest("x", {"a": 1})
    s = sigs(req, "alice", "bob")
    q.verify(req, s)
    with pytest.raises(QuorumError, match="replay"):
        q.verify(req, s)
    altered = ApprovalRequest("x", {"a": 2}, nonce="n", expires_at=req.expires_at)
    with pytest.raises(QuorumError):
        q.verify(altered, sigs(ApprovalRequest("x", {"a": 1}, "n", req.expires_at),
                               "alice", "bob"))
    old = ApprovalRequest("x", {}, expires_at=time.time() - 1)
    with pytest.raises(QuorumError, match="expired"):
        q.verify(old, sigs(old, "alice", "bob"))


# -- kill switch -------------------------------------------------------------
def test_kill_switch_latches_and_needs_quorum_to_reset(env):
    _, audit, _, kill = env
    kill.check()
    kill.trip("test", by="alice")
    with pytest.raises(Halted):
        kill.check()
    req = kill.reset_request()
    with pytest.raises(QuorumError):
        kill.reset(req, sigs(req, "alice"))
    assert kill.tripped
    kill.reset(req, sigs(req, "alice", "carol"))
    kill.check()
    events = [r.event for r in audit.verify()]
    assert events == ["killswitch.tripped", "killswitch.reset"]


def test_external_halt_file_stops_system_and_blocks_reset(env):
    tmp, _, _, kill = env
    (tmp / "HALT").touch()
    with pytest.raises(Halted, match="halt file"):
        kill.check()
    req = kill.reset_request()
    with pytest.raises(Halted):
        kill.reset(req, sigs(req, "alice", "bob"))
    (tmp / "HALT").unlink()
    kill.reset(req, sigs(req, "alice", "bob"))
    kill.check()


def test_dead_mans_switch(tmp_path):
    audit = AuditLog(tmp_path / "a.jsonl")
    kill = KillSwitch(audit, Quorum(KEYS, 2), heartbeat_timeout=0.05)
    kill.heartbeat("alice")
    kill.check()
    time.sleep(0.1)
    with pytest.raises(Halted, match="dead-man"):
        kill.check()


# -- contained runtime -------------------------------------------------------
def test_tools_default_deny_and_high_risk_needs_approval(env):
    cm = make_model(env)
    assert cm.call_tool("add", a=1, b=2) == 3
    for name in ("banned", "unlisted", "not_registered"):
        with pytest.raises(PolicyViolation):
            cm.call_tool(name)
    with pytest.raises(PolicyViolation, match="approval"):
        cm.call_tool("danger", x="y")
    req = cm.request_tool("danger", x="y")
    with pytest.raises(PolicyViolation):  # only one signer
        cm.call_tool("danger", approval=(req, sigs(req, "bob")), x="y")
    assert cm.call_tool("danger", approval=(req, sigs(req, "bob", "carol")),
                        x="y") == "did y"


def test_approval_cannot_be_reused_for_different_args(env):
    cm = make_model(env)
    req = cm.request_tool("danger", x="harmless")
    with pytest.raises(PolicyViolation, match="match"):
        cm.call_tool("danger", approval=(req, sigs(req, "alice", "bob")), x="evil")


def test_kill_switch_interrupts_generation_mid_stream(env):
    _, _, _, kill = env
    produced = []

    def gen(prompt, n, should_stop):
        for i in range(n):
            if should_stop():
                break
            produced.append(i)
            if i == 3:
                kill.trip("mid-stream", by="test")
        return "x" * len(produced)

    cm = make_model(env, gen=gen)
    with pytest.raises(Halted):
        cm.generate("p", max_tokens=100)
    assert len(produced) == 4


def test_output_tripwire_trips_kill_switch(env):
    _, _, _, kill = env
    leak = "-----BEGIN RSA PRIVATE KEY-----"
    cm = make_model(env, gen=lambda p, n, should_stop: leak)
    with pytest.raises(Halted):
        cm.generate("p")
    assert kill.tripped and "tripwire" in kill.reason


def test_budget_enforced(env):
    cm = make_model(env, budget=Budget(max_calls=2))
    cm.generate("a")
    cm.generate("b")
    with pytest.raises(PolicyViolation, match="budget"):
        cm.generate("c")


def test_audit_does_not_store_raw_prompts(env):
    _, audit, _, _ = env
    cm = make_model(env)
    cm.generate("my secret prompt")
    assert "my secret prompt" not in audit.path and \
        "my secret prompt" not in open(audit.path).read()


# -- sandbox -----------------------------------------------------------------
def test_sandbox_runs_code_with_scrubbed_env(monkeypatch):
    monkeypatch.setenv("SUPER_SECRET", "hunter2")
    r = run_python("import os; print(os.environ.get('SUPER_SECRET'))")
    assert r.stdout.strip() == "None"


def test_sandbox_timeout_and_memory_limit():
    r = run_python("while True: pass", timeout=0.5, cpu_seconds=5)
    assert r.timed_out
    r = run_python("x = bytearray(2 * 1024**3)", memory_bytes=256 * 1024**2)
    assert r.returncode != 0 and "MemoryError" in r.stderr
