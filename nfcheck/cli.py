"""nfcheck: prove or disprove "no functional change" claims in git history.

    nfcheck                          # commits on this branch not yet on main/master
    nfcheck HEAD~50..HEAD            # a range
    nfcheck --all v1.0..v2.0         # every commit, claimed or not
    nfcheck --commit abc123          # one commit, even without a claim
    nfcheck --files old.py new.py    # compare two files directly
    nfcheck --json ...               # machine-readable

Exit status: 0 no promise broken, 1 a promise was broken,
             2 usage or git error, 3 unverifiable (only with --strict).
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from . import __version__
from .check import Checker, CommitResult, Verdict, claims_no_change, explain
from .fingerprint import Unparseable, fingerprint
from .git import GitError, Repo

ICON = {Verdict.PROVEN: "✓", Verdict.REORDERED: "≈", Verdict.CHANGED: "✗",
        Verdict.UNVERIFIABLE: "?"}


def _default_range(repo: Repo) -> str:
    for base in ("origin/HEAD", "origin/main", "origin/master", "main", "master"):
        try:
            b, head = repo.resolve(base), repo.resolve("HEAD")
        except GitError:
            continue
        if b != head:
            return f"{base}..HEAD"
    return "HEAD~1..HEAD"


def _color(enabled: bool):
    codes = {Verdict.PROVEN: "32", Verdict.REORDERED: "36", Verdict.CHANGED: "31",
             Verdict.UNVERIFIABLE: "33"}
    return (lambda v, s: f"\033[{codes[v]}m{s}\033[0m") if enabled else (lambda v, s: s)


def _print(r: CommitResult, paint, verbose: bool) -> None:
    c = r.commit
    tag = ("PROMISE BROKEN" if r.promise_broken
           else "REVIEW" if r.needs_review else r.verdict.value)
    claim = (f'  [{"promises" if r.claim.strong else "refactor"}: "{r.claim.text}"]'
             if r.claim else "")
    print(paint(r.verdict, f"{ICON[r.verdict]} {c.sha[:10]} {tag:<14}") + f" {c.subject[:72]}{claim}")
    for f in r.files:
        if f.verdict is Verdict.PROVEN and not verbose:
            continue
        print(f"    {paint(f.verdict, ICON[f.verdict])} {f.path}: {f.reason}")
        for line in f.detail:
            print(f"        {line}")
    for n in r.notes:
        print(f"    note: {n}")


def _as_json(r: CommitResult) -> dict:
    return {"sha": r.commit.sha, "subject": r.commit.subject,
            "claim": r.claim and {"text": r.claim.text, "strong": r.claim.strong},
            "verdict": r.verdict.value, "promise_broken": r.promise_broken,
            "needs_review": r.needs_review,
            "notes": r.notes,
            "files": [{"path": f.path, "verdict": f.verdict.value, "reason": f.reason,
                       "detail": f.detail} for f in r.files]}


def _compare_files(a: str, b: str, docstrings: bool) -> int:
    try:
        fa = fingerprint(open(a, "rb").read(), docstrings)
        fb = fingerprint(open(b, "rb").read(), docstrings)
    except (OSError, Unparseable) as e:
        print(f"nfcheck: {e}", file=sys.stderr)
        return 2
    if fa.digest == fb.digest:
        print("✓ identical compiled behaviour")
        return 0
    print("✗ behaviour differs")
    for line in explain(fa, fb):
        print(f"    {line}")
    return 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="nfcheck", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("range", nargs="?", help="git revision range (default: branch vs main)")
    ap.add_argument("-C", "--repo", default=".", help="repository path")
    ap.add_argument("--all", action="store_true", help="check every commit, not only claims")
    ap.add_argument("--commit", help="check one commit regardless of its message")
    ap.add_argument("--files", nargs=2, metavar=("OLD", "NEW"), help="compare two files")
    ap.add_argument("--first-parent", action="store_true", help="follow first parents only")
    ap.add_argument("--docstrings", action="store_true",
                    help="treat docstring edits as behaviour changes")
    ap.add_argument("--include-tests", action="store_true",
                    help="judge changes to test files too")
    ap.add_argument("--strict", action="store_true",
                    help="exit 3 when a claimed commit is unverifiable")
    ap.add_argument("-j", "--jobs", type=int, default=None,
                    help="parallel workers (default: CPU count)")
    ap.add_argument("--json", action="store_true", help="JSON output")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="list proven files and explain unclaimed changes too")
    ap.add_argument("--version", action="version", version=f"nfcheck {__version__}")
    a = ap.parse_args(argv)

    if a.files:
        return _compare_files(*a.files, a.docstrings)

    paint = _color(sys.stdout.isatty() and not a.json)
    started = time.perf_counter()
    try:
        with Repo(a.repo) as repo:
            checker = Checker(repo, docstrings=a.docstrings, include_tests=a.include_tests,
                              jobs=a.jobs, explain_all=a.verbose)
            if a.commit:
                c = repo.commits(f"{repo.resolve(a.commit)}^!")
                results = [checker.check_commit(c[0], claims_no_change(c[0].message))]
            else:
                rng = a.range or _default_range(repo)
                results = []
                for r in checker.check_range(rng, a.all, a.first_parent):
                    results.append(r)
                    if not a.json:
                        _print(r, paint, a.verbose)
            if a.commit and not a.json:
                _print(results[0], paint, a.verbose)
            stats = checker.stats
    except GitError as e:
        print(f"nfcheck: {e}", file=sys.stderr)
        return 2

    strong = [r for r in results if r.claim and r.claim.strong]
    weak = [r for r in results if r.claim and not r.claim.strong]
    broken = [r for r in strong if r.promise_broken]
    count = lambda rs, v: sum(r.verdict is v for r in rs)  # noqa: E731
    unver = [r for r in strong + weak if r.verdict is Verdict.UNVERIFIABLE]
    elapsed = time.perf_counter() - started
    summary = {
        "promises": len(strong), "promises_kept": count(strong, Verdict.PROVEN),
        "promises_kept_imports_reordered": count(strong, Verdict.REORDERED),
        "promises_broken": len(broken), "refactors": len(weak),
        "refactors_proven": count(weak, Verdict.PROVEN),
        "refactors_to_review": count(weak, Verdict.CHANGED),
        "unverifiable": len(unver), "seconds": round(elapsed, 3), **stats}
    if a.json:
        print(json.dumps({"results": [_as_json(r) for r in results],
                          "summary": summary}, indent=2))
    else:
        silent = sum(1 for r in results if not r.claim and r.verdict is Verdict.PROVEN)
        reord = summary["promises_kept_imports_reordered"]
        print(f"\nno-change promises: {summary['promises_kept']} proven kept, "
              + (f"{reord} kept apart from import order, " if reord else "")
              + 
              f"{len(broken)} broken | refactors: {summary['refactors_proven']} proven "
              f"behaviour-preserving, {summary['refactors_to_review']} to review | "
              f"{len(unver)} unverifiable"
              + (f" | {silent} unlabelled commits proven behaviour-preserving" if a.all else "")
              + f"  ({elapsed:.2f}s, {stats['blobs_compiled']} files compiled)")
    if broken:
        return 1
    if unver and a.strict:
        return 3
    return 0


def run() -> int:
    try:
        return main()
    except BrokenPipeError:  # output piped to e.g. `head`
        sys.stderr.close()
        return 0


if __name__ == "__main__":
    sys.exit(run())
