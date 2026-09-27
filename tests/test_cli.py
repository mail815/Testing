import os

from guardian import ApprovalRequest, AuditLog, Quorum
from guardian.cli import load_public_key, main, request_to_json


def test_keygen_sign_verify_roundtrip(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["keygen", "alice"]) == 0
    assert oct(os.stat("alice.key").st_mode & 0o777) == "0o600"
    req = ApprovalRequest("killswitch.reset", {"reason": "x"})
    (tmp_path / "req.json").write_text(request_to_json(req))
    capsys.readouterr()
    main(["sign", "alice.key", "req.json"])
    sig = capsys.readouterr().out.strip()
    q = Quorum({"alice": load_public_key("alice.pub")}, 1)
    assert q.verify(req, {"alice": sig}) == ["alice"]


def test_halt_and_verify_audit(tmp_path, capsys):
    main(["halt", str(tmp_path / "HALT")])
    assert (tmp_path / "HALT").exists()
    log = AuditLog(tmp_path / "a.jsonl")
    log.append("e")
    assert main(["verify-audit", str(tmp_path / "a.jsonl")]) == 0
    assert main(["verify-audit", str(tmp_path / "a.jsonl"), "--anchor", "5", "x"]) == 2
