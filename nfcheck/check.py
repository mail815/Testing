"""Decide whether a commit kept its "no behaviour change" promise."""

from __future__ import annotations

import difflib
import os
import posixpath
import re
from dataclasses import dataclass, field
from enum import Enum

from concurrent.futures import ProcessPoolExecutor

from .fingerprint import (Fingerprint, Quick, Unparseable, code_units, fingerprint,
                          listing, quick)
from .git import EMPTY, Change, Commit, Repo

# EXPLICIT claims may appear anywhere in the message: they are unambiguous.
# A bare "NFC" counts only in the subject, as a tag ("[NFC]", "(NFC)") or on
# a line of its own: in prose it is usually a mention ("claims like NFC, ...").
NFC_SUBJECT = re.compile(r"\bNFCI?\b")
EXPLICIT = re.compile(
    r"""
    \[NFCI?\] | \(NFCI?\) | ^\s*NFCI?[.!]?\s*$
    | \bno[ -]functional[ -]changes?\b
    | \bno[ -](?:behaviou?ral|behaviou?r|logic|semantic)[ -]changes?\b
    | \bbehaviou?r[ -]preserving\b
    | \bpure[ -]refactor(?:ing)?\b
    """, re.IGNORECASE | re.VERBOSE | re.MULTILINE)
# FORMATTING claims only count in the subject line, in unambiguous forms.
# ("Format X" is excluded: in a formatter's own repo it describes a feature.)
FORMATTING = re.compile(
    r"""
    ^(?:style|fmt)(?:\([^)]*\))?!?:                   # conventional commits
    | ^re-?format(?:ted|ting)?\b
    | ^(?:run|apply|applied|ran)\s+(?:black|ruff[ -]format|isort|autopep8|yapf)\b
    | \b(?:formatting|whitespace)[ -]only\s*$
    """, re.IGNORECASE | re.VERBOSE)
# Both of the above are STRONG: formatting and "no functional change" both
# mean the compiled program must be identical, so any difference breaks them.
# WEAK claims ("refactor") promise preserved behaviour but expect the code to
# change. A compiled difference is not proof of a break: it marks exactly
# which functions a reviewer must check.
WEAK = re.compile(r"^refactor(?:ing|ed|s)?\b", re.IGNORECASE)

DOC_EXT = {".md", ".rst", ".txt", ".adoc", ".markdown"}
DOC_NAMES = {"license", "licence", "authors", "contributors", "changelog",
             "changes", "news", "readme", "notice", "codeowners", ".gitignore",
             ".mailmap", ".git-blame-ignore-revs"}
PY_EXT = {".py", ".pyw"}
STUB_EXT = {".pyi"}


@dataclass(frozen=True)
class Claim:
    text: str
    strong: bool


_OPEN_SINGLE = re.compile(r"(?:^|(?<=\s))['\u2018](?=\S)")
_CLOSE_SINGLE = re.compile(r"(?<=\S)['\u2019](?=\s|$|[.,;:!?)])")


def _quoted(text: str, start: int) -> bool:
    """True if position ``start`` is inside quotes on its line: a mention
    (e.g. 'adds a checker for "NFC" claims') rather than a claim."""
    line_start = text.rfind("\n", 0, start) + 1
    before = text[line_start:start]
    if before.count('"') % 2 or before.count("`") % 2:
        return True
    if before.count("\u201c") > before.count("\u201d"):
        return True
    # Single quotes double as apostrophes ("don't"), so only count quotes
    # that open before a word and close after one.
    opened = len(_OPEN_SINGLE.findall(text[line_start:start + 1]))
    return opened > len(_CLOSE_SINGLE.findall(before))


def claims_no_change(message: str) -> Claim | None:
    subject = message.strip().splitlines()[0] if message.strip() else ""
    for m in EXPLICIT.finditer(message):
        if not _quoted(message, m.start()):
            return Claim(m.group(0).strip(), True)
    for m in NFC_SUBJECT.finditer(subject):
        if not _quoted(subject, m.start()):
            return Claim(m.group(0), True)
    if m := FORMATTING.search(subject):
        return Claim(m.group(0).strip(), True)
    if m := WEAK.search(subject):
        return Claim(m.group(0).strip(), False)
    return None


class Verdict(str, Enum):
    PROVEN = "proven"              # compiled behaviour provably identical
    REORDERED = "imports-reordered"  # identical except order/grouping of imports
    CHANGED = "changed"            # compiled code differs (see which functions)
    UNVERIFIABLE = "unverifiable"  # something changed that we cannot judge


@dataclass
class FileResult:
    path: str
    verdict: Verdict
    reason: str
    detail: list[str] = field(default_factory=list)


@dataclass
class CommitResult:
    commit: Commit
    claim: Claim | None
    verdict: Verdict
    files: list[FileResult]
    notes: list[str] = field(default_factory=list)

    @property
    def promise_broken(self) -> bool:
        """Only a strong claim is broken by a compiled difference."""
        return bool(self.claim and self.claim.strong and self.verdict is Verdict.CHANGED)

    @property
    def needs_review(self) -> bool:
        return bool(self.claim and not self.claim.strong
                    and self.verdict is Verdict.CHANGED)


def _kind(path: str) -> str:
    base = posixpath.basename(path).lower()
    stem, ext = posixpath.splitext(base)
    parts = path.lower().split("/")
    if parts[0] in ("docs", "doc") and (ext in PY_EXT or ext in DOC_EXT):
        return "docs"  # documentation build config, e.g. docs/conf.py
    if ext in PY_EXT:
        return "python"
    if ext in STUB_EXT or base == "py.typed":
        return "typing"
    if ext in DOC_EXT or stem in DOC_NAMES or base in DOC_NAMES or "docs" in parts[:-1]:
        return "docs"
    return "other"


def _is_test(path: str) -> bool:
    p = path.lower()
    base = posixpath.basename(p)
    return (base.startswith("test_") or base.endswith("_test.py") or base == "conftest.py"
            or "/tests/" in f"/{p}" or "/test/" in f"/{p}")


_IMPORT_OPS = {"IMPORT_NAME", "IMPORT_FROM", "IMPORT_STAR", "STORE_NAME",
               "STORE_GLOBAL", "LOAD_CONST", "POP_TOP"}


def explain(old: Fingerprint, new: Fingerprint, max_lines: int = 12) -> list[str]:
    """Name the functions whose behaviour changed, with an instruction diff."""
    out: list[str] = []
    a, b = code_units(old.code), code_units(new.code)  # type: ignore[arg-type]
    for name in sorted(set(old.units) | set(new.units), key=lambda n: (n != "<module>", n)):
        if old.units.get(name) == new.units.get(name):
            continue
        if name not in old.units:
            out.append(f"+ added: {name}")
            continue
        if name not in new.units:
            out.append(f"- removed: {name}")
            continue
        out.append(f"~ changed: {name}")
        la, lb = listing(a[name]), listing(b[name])
        if la != lb and sorted(la) == sorted(lb):
            moved = {l for tag, i1, i2, j1, j2 in
                     difflib.SequenceMatcher(None, la, lb, autojunk=False).get_opcodes()
                     if tag != "equal" for l in la[i1:i2] + lb[j1:j2]}
            imports = all(l.split()[0] in _IMPORT_OPS for l in moved)
            out.append("    same instructions, different order"
                       + (" (import order only: matters only if imports have side effects)"
                          if imports else ""))
            continue
        diff = [l for l in difflib.unified_diff(la, lb, lineterm="", n=1)
                if not l.startswith(("---", "+++"))]
        if not diff:  # difference is in metadata (e.g. a parameter was renamed)
            diff = ["  (same instructions; signature, flags or constants differ)"]
        out += ["    " + l for l in diff[:max_lines]]
        if len(diff) > max_lines:
            out.append(f"    … {len(diff) - max_lines} more lines")
    return out


def _quick_worker(job: tuple[bytes, bool]):
    try:
        return quick(*job)
    except Unparseable as e:
        return e


class Checker:
    """Checks commits. Results for each file version (git blob) are computed
    once and cached; big batches are fingerprinted on all CPU cores."""

    def __init__(self, repo: Repo, docstrings: bool = False, include_tests: bool = False,
                 jobs: int | None = None, explain_all: bool = False):
        self.repo = repo
        self.docstrings = docstrings
        self.include_tests = include_tests
        self.jobs = jobs if jobs is not None else (os.cpu_count() or 1)
        self.explain_all = explain_all
        self._quick: dict[str, Quick | Unparseable] = {}
        self._full: dict[str, Fingerprint] = {}
        self.stats = {"blobs_compiled": 0, "cache_hits": 0}

    # -- fingerprint cache ---------------------------------------------------
    def prefetch(self, blobs: set[str]) -> None:
        todo = [b for b in blobs if b not in self._quick and b != EMPTY]
        if not todo:
            return
        sources = [self.repo.blob(b) for b in todo]
        jobs = [(src, self.docstrings) for src in sources]
        if self.jobs > 1 and len(todo) >= 32:
            with ProcessPoolExecutor(self.jobs) as pool:
                results = list(pool.map(_quick_worker, jobs, chunksize=8))
        else:
            results = [_quick_worker(j) for j in jobs]
        self._quick.update(zip(todo, results))
        self.stats["blobs_compiled"] += len(todo)

    def _get(self, blob: str) -> Quick:
        if blob in self._quick:
            self.stats["cache_hits"] += 1
        else:
            self.prefetch({blob})
        hit = self._quick[blob]
        if isinstance(hit, Unparseable):
            raise hit
        return hit

    def _fingerprint(self, blob: str) -> Fingerprint:
        """Full fingerprint (units, code, imports split) - only when needed."""
        if blob not in self._full:
            self._full[blob] = fingerprint(self.repo.blob(blob), self.docstrings)
        return self._full[blob]

    # -- judging -------------------------------------------------------------
    @staticmethod
    def _needs_fingerprint(ch: Change) -> bool:
        kinds = {_kind(p) for p in (ch.old_path, ch.new_path) if p}
        return (kinds == {"python"} and ch.status in "MRCT"
                and ch.old_path == ch.new_path and ch.old_blob != ch.new_blob)

    def check_change(self, ch: Change, explain_diff: bool = True) -> FileResult | None:
        path = ch.new_path or ch.old_path or "?"
        kinds = {_kind(p) for p in (ch.old_path, ch.new_path) if p}
        if ch.old_mode != ch.new_mode and ch.old_blob == ch.new_blob:
            return FileResult(path, Verdict.UNVERIFIABLE,
                              f"file mode changed {ch.old_mode}→{ch.new_mode}")
        if kinds <= {"docs"} or kinds <= {"typing"}:
            return None  # documentation and stub files are never executed
        if "python" not in kinds:
            return FileResult(path, Verdict.UNVERIFIABLE, "non-Python file changed")
        if ch.status == "A":
            return FileResult(path, Verdict.CHANGED, "new module added")
        if ch.status == "D":
            return FileResult(path, Verdict.CHANGED, "module deleted")
        if ch.old_path != ch.new_path:
            return FileResult(path, Verdict.CHANGED,
                              f"moved {ch.old_path} → {ch.new_path} (import path changed)")
        try:
            a, b = self._get(ch.old_blob), self._get(ch.new_blob)
        except Unparseable as e:
            return FileResult(path, Verdict.UNVERIFIABLE, f"cannot compile: {e}")
        if a.digest == b.digest:
            return FileResult(path, Verdict.PROVEN, "identical compiled behaviour")
        if a.imports and a.imports == b.imports and a.nested == b.nested:
            fa, fb = self._fingerprint(ch.old_blob), self._fingerprint(ch.new_blob)
            if fa.sans_imports == fb.sans_imports:
                return FileResult(path, Verdict.REORDERED,
                                  "identical except import order/grouping "
                                  "(differs only if imports have side effects)")
        detail = (explain(self._fingerprint(ch.old_blob), self._fingerprint(ch.new_blob))
                  if explain_diff else [])
        return FileResult(path, Verdict.CHANGED, "compiled behaviour differs", detail)

    def check_commit(self, commit: Commit, claim: Claim | None = None,
                     changes: list[Change] | None = None) -> CommitResult:
        notes: list[str] = []
        if len(commit.parents) > 1:
            notes.append("merge commit: compared against first parent")
        if changes is None:
            parent = commit.parents[0] if commit.parents else None
            changes = self.repo.changes(parent, commit.sha)
        explain_diff = claim is not None or self.explain_all
        files = [r for ch in changes if (r := self.check_change(ch, explain_diff)) is not None]
        judged = files
        if not self.include_tests:
            # Tests are not the product: their changes are reported, not judged.
            tests = [f for f in files if _is_test(f.path) and f.verdict is not Verdict.PROVEN]
            if tests:
                notes.append("test files changed (not counted): "
                             + ", ".join(f.path for f in tests))
            judged = [f for f in files if f not in tests]
        verdicts = {f.verdict for f in judged}
        verdict = next((v for v in (Verdict.CHANGED, Verdict.UNVERIFIABLE, Verdict.REORDERED)
                        if v in verdicts), Verdict.PROVEN)
        return CommitResult(commit, claim, verdict, files, notes)

    def check_range(self, rev_range: str, all_commits: bool = False,
                    first_parent: bool = False):
        selected = [(c, claims_no_change(c.message))
                    for c in self.repo.commits(rev_range, first_parent)]
        selected = [(c, cl) for c, cl in selected if cl or all_commits]
        changes = self.repo.changes_many(
            [(c.sha, c.parents[0] if c.parents else None) for c, _ in selected])
        self.prefetch({blob for chs in changes.values() for ch in chs
                       if self._needs_fingerprint(ch) for blob in (ch.old_blob, ch.new_blob)})
        for c, cl in selected:
            yield self.check_commit(c, cl, changes.get(c.sha, []))
