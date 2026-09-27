"""Minimal, fast git plumbing: one long-lived ``cat-file --batch`` process."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

EMPTY = "0" * 40


class GitError(Exception):
    pass


@dataclass(frozen=True)
class Change:
    status: str        # A, M, D, R, C, T
    old_path: str | None
    new_path: str | None
    old_blob: str      # EMPTY when added
    new_blob: str      # EMPTY when deleted
    old_mode: str
    new_mode: str


@dataclass(frozen=True)
class Commit:
    sha: str
    parents: tuple[str, ...]
    subject: str
    message: str


class Repo:
    def __init__(self, path: str = "."):
        self.path = path
        top = self._git("rev-parse", "--show-toplevel").strip()
        self.path = top
        self._cat: subprocess.Popen | None = None

    def _git(self, *args: str, input: bytes | None = None) -> str:
        r = subprocess.run(["git", "-C", self.path, *args], input=input,
                           capture_output=True)
        if r.returncode:
            raise GitError(r.stderr.decode(errors="replace").strip() or f"git {args[0]} failed")
        return r.stdout.decode("utf-8", errors="replace")

    def commits(self, rev_range: str, first_parent: bool = False) -> list[Commit]:
        args = ["log", "--reverse", "--format=%H%x1f%P%x1f%B%x1e"]
        if first_parent:
            args.append("--first-parent")
        out = self._git(*args, rev_range, "--")
        result = []
        for rec in out.split("\x1e"):
            rec = rec.strip("\n")
            if not rec:
                continue
            sha, parents, msg = rec.split("\x1f", 2)
            msg = msg.strip()
            result.append(Commit(sha, tuple(parents.split()), msg.splitlines()[0] if msg else "", msg))
        return result

    def resolve(self, rev: str) -> str:
        return self._git("rev-parse", "--verify", f"{rev}^{{commit}}").strip()

    def changes(self, old: str | None, new: str) -> list[Change]:
        """Changed paths between two commits (``old=None``: new is a root commit)."""
        return self.changes_many([(new, old)]).get(new, [])

    def changes_many(self, pairs: list[tuple[str, str | None]]) -> dict[str, list[Change]]:
        """Changes for many (commit, parent) pairs using a single git process."""
        if not pairs:
            return {}
        stdin = "".join(f"{c} {p}\n" if p else f"{c}\n" for c, p in pairs).encode()
        fields = self._git("diff-tree", "--stdin", "--root", "-r", "-z", "--raw", "-M",
                           input=stdin).split("\0")
        out: dict[str, list[Change]] = {c: [] for c, _ in pairs}
        current, i = None, 0
        while i < len(fields):
            f = fields[i]
            if not f.startswith(":"):
                if f:
                    current = f.strip()  # commit header
                i += 1
                continue
            om, nm, ob, nb, st = f[1:].split(" ")
            kind = st[0]
            if kind in "RC":
                op, np_, i = fields[i + 1], fields[i + 2], i + 3
            else:
                op = np_ = fields[i + 1]
                i += 2
            if current is not None:
                out.setdefault(current, []).append(Change(
                    kind, None if kind == "A" else op, None if kind == "D" else np_,
                    ob, nb, om, nm))
        return out

    def blob(self, sha: str) -> bytes:
        if sha == EMPTY:
            return b""
        if self._cat is None:
            self._cat = subprocess.Popen(["git", "-C", self.path, "cat-file", "--batch"],
                                         stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        assert self._cat.stdin and self._cat.stdout
        self._cat.stdin.write(sha.encode() + b"\n")
        self._cat.stdin.flush()
        header = self._cat.stdout.readline().split()
        if len(header) < 3 or header[1] == b"missing":
            raise GitError(f"missing object {sha}")
        size = int(header[2])
        data = self._cat.stdout.read(size)
        self._cat.stdout.read(1)  # trailing newline
        return data

    def close(self) -> None:
        if self._cat:
            self._cat.stdin.close()  # type: ignore[union-attr]
            self._cat.wait()
            self._cat = None

    def __enter__(self) -> "Repo":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
