"""Operator command line. Run on an operator machine, not the model host.

    python -m guardian.cli keygen alice            # alice.key (private), alice.pub
    python -m guardian.cli sign alice.key req.json # prints signature
    python -m guardian.cli halt /path/to/HALT      # press the red button
    python -m guardian.cli verify-audit audit.jsonl [--anchor N HEAD]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey, Ed25519PublicKey)

from .audit import AuditIntegrityError, AuditLog
from .quorum import ApprovalRequest, generate_operator_key, sign


def load_public_key(path: str) -> Ed25519PublicKey:
    with open(path, "rb") as f:
        key = serialization.load_pem_public_key(f.read())
    if not isinstance(key, Ed25519PublicKey):
        raise TypeError(f"{path} is not an Ed25519 public key")
    return key


def load_private_key(path: str, password: bytes | None = None) -> Ed25519PrivateKey:
    with open(path, "rb") as f:
        key = serialization.load_pem_private_key(f.read(), password=password)
    if not isinstance(key, Ed25519PrivateKey):
        raise TypeError(f"{path} is not an Ed25519 private key")
    return key


def request_to_json(req: ApprovalRequest) -> str:
    return json.dumps(req.__dict__, sort_keys=True, indent=2)


def request_from_json(text: str) -> ApprovalRequest:
    return ApprovalRequest(**json.loads(text))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="guardian")
    sub = ap.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("keygen"); k.add_argument("name")
    s = sub.add_parser("sign"); s.add_argument("key"); s.add_argument("request")
    h = sub.add_parser("halt"); h.add_argument("path")
    v = sub.add_parser("verify-audit"); v.add_argument("log")
    v.add_argument("--anchor", nargs=2, metavar=("COUNT", "HEAD"))
    a = ap.parse_args(argv)
    pw = os.environ.get("GUARDIAN_KEY_PASSWORD", "").encode() or None

    if a.cmd == "keygen":
        priv, pub = generate_operator_key()
        enc = (serialization.BestAvailableEncryption(pw) if pw
               else serialization.NoEncryption())
        fd = os.open(f"{a.name}.key", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(priv.private_bytes(serialization.Encoding.PEM,
                                       serialization.PrivateFormat.PKCS8, enc))
        with open(f"{a.name}.pub", "wb") as f:
            f.write(pub.public_bytes(serialization.Encoding.PEM,
                                     serialization.PublicFormat.SubjectPublicKeyInfo))
        print(f"wrote {a.name}.key (keep offline) and {a.name}.pub (install on host)")
    elif a.cmd == "sign":
        with open(a.request, encoding="utf-8") as f:
            req = request_from_json(f.read())
        print(f"About to sign: {req.action} {json.dumps(req.params, sort_keys=True)}",
              file=sys.stderr)
        print(sign(load_private_key(a.key, pw), req))
    elif a.cmd == "halt":
        with open(a.path, "a"):
            pass
        print(f"halt file created: {a.path}")
    elif a.cmd == "verify-audit":
        anchor = (int(a.anchor[0]), a.anchor[1]) if a.anchor else None
        try:
            n = len(AuditLog(a.log).verify(anchor=anchor))
        except AuditIntegrityError as e:
            print(f"TAMPERING DETECTED: {e}", file=sys.stderr)
            return 2
        print(f"ok: {n} records intact")
    return 0


if __name__ == "__main__":
    sys.exit(main())
