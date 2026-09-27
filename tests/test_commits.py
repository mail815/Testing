import json
import subprocess

import pytest

from nfcheck.check import claims_no_change
from nfcheck.cli import main

CODE = "def price(qty, unit=2):\n    if qty < 10:\n        return qty * unit\n    return qty * unit * 0.9\n"


def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def commit(repo, msg, files):
    for path, text in files.items():
        p = repo / path
        if text is None:
            git(repo, "rm", "-q", path)
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        git(repo, "add", path)
    git(repo, "commit", "-q", "--allow-empty", "-m", msg)


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q", "-b", "main")
    git(tmp_path, "config", "user.email", "t@example.com")
    git(tmp_path, "config", "user.name", "t")
    commit(tmp_path, "initial", {"shop/pricing.py": CODE, "README.md": "# shop\n"})
    return tmp_path


def run(repo, *args, capsys=None):
    code = main(["-C", str(repo), "--json", *args])
    out = json.loads(capsys.readouterr().out) if capsys else None
    return code, out


def by_subject(out):
    return {r["subject"]: r for r in out["results"]}


# --- claim detection --------------------------------------------------------
@pytest.mark.parametrize("msg,strong", [
    ("[NFC] tidy parser", True),
    ("Simplify loop (NFCI)", True),
    ("parser: extract helper\n\nNo functional change.", True),
    ("No behavior changes intended", True),
    ("style: run black", True),
    ("fmt(core): wrap long lines", True),
    ("Reformat with ruff", True),
    ("apply black", True),
    ("Run isort", True),
    ("refactor: split module", False),
    ("Refactored the cache", False),
])
def test_claims_detected(msg, strong):
    c = claims_no_change(msg)
    assert c is not None and c.strong is strong


@pytest.mark.parametrize("msg", [
    "Format subscriptions per PEP 8",           # a formatter's feature, not a claim
    "Fix bug where whitespace-only files crash",
    "Add NFC reader support".replace("NFC", "nfcpy"),
    "Improve tests\n\nRefactored assoc helper",  # "refactor" only counts in subject
    "Merge pull request #1 from x/refactor-y",
])
def test_non_claims_ignored(msg):
    assert claims_no_change(msg) is None


# --- end to end on a real git repo ------------------------------------------
def test_kept_and_broken_promises(repo, capsys):
    commit(repo, "style: reformat pricing", {"shop/pricing.py": CODE.replace(
        "return qty * unit\n", "return (\n            qty * unit\n        )  # small order\n")})
    commit(repo, "Tidy discount [NFC]", {"shop/pricing.py": CODE.replace("qty < 10", "qty <= 10")})
    commit(repo, "refactor: rename local", {"shop/pricing.py": CODE.replace("unit", "u")})
    code, out = run(repo, "main~3..main", capsys=capsys)
    r = by_subject(out)
    assert r["style: reformat pricing"]["verdict"] == "proven"
    broken = r["Tidy discount [NFC]"]
    assert broken["promise_broken"] and broken["verdict"] == "changed"
    assert any("price" in line for line in broken["files"][0]["detail"])
    assert r["refactor: rename local"]["needs_review"]  # 'unit' is a parameter
    assert code == 1
    assert out["summary"]["promises_kept"] == 1 and out["summary"]["promises_broken"] == 1


def test_docs_and_tests_only_are_not_judged(repo, capsys):
    commit(repo, "NFC: docs and tests", {"README.md": "# shop!\n",
                                         "docs/conf.py": "project = 'x'\n",
                                         "tests/test_p.py": "def test_x():\n    assert 1\n"})
    code, out = run(repo, "main~1..main", capsys=capsys)
    res = out["results"][0]
    assert res["verdict"] == "proven" and code == 0
    assert any("test files changed" in n for n in res["notes"])


def test_non_python_change_is_unverifiable(repo, capsys):
    commit(repo, "NFC: bump config", {"setup.cfg": "[metadata]\nname = shop\n"})
    code, out = run(repo, "main~1..main", capsys=capsys)
    assert out["results"][0]["verdict"] == "unverifiable" and code == 0
    assert main(["-C", str(repo), "--strict", "main~1..main"]) == 3


def test_module_move_is_a_change(repo, capsys):
    git(repo, "mv", "shop/pricing.py", "shop/prices.py")
    git(repo, "commit", "-q", "-m", "NFC: rename module")
    _, out = run(repo, "main~1..main", capsys=capsys)
    assert out["results"][0]["promise_broken"]


def test_import_reorder_keeps_promise_with_caveat(repo, capsys):
    commit(repo, "add imports", {"shop/app.py": "import sys\nimport os\nprint(os, sys)\n"})
    commit(repo, "Run isort", {"shop/app.py": "import os\nimport sys\nprint(os, sys)\n"})
    code, out = run(repo, "main~1..main", capsys=capsys)
    res = out["results"][0]
    assert res["verdict"] == "imports-reordered" and not res["promise_broken"] and code == 0


def test_all_mode_finds_unlabelled_safe_commits(repo, capsys):
    commit(repo, "tweak", {"shop/pricing.py": CODE + "\n\n# trailing comment\n"})
    _, out = run(repo, "--all", "main~1..main", capsys=capsys)
    assert out["results"][0]["claim"] is None and out["results"][0]["verdict"] == "proven"


def test_cache_reuses_compiled_blobs(repo, capsys):
    for i in range(3):  # flip back and forth between two versions
        commit(repo, f"style: pass {i}", {"shop/pricing.py": CODE if i % 2 else CODE + "\n"})
    _, out = run(repo, "main~3..main", capsys=capsys)
    assert out["summary"]["blobs_compiled"] == 2 and out["summary"]["cache_hits"] >= 4


def test_compare_files(tmp_path, capsys):
    a, b = tmp_path / "a.py", tmp_path / "b.py"
    a.write_text(CODE)
    b.write_text(CODE.replace("0.9", "0.8"))
    assert main(["--files", str(a), str(a)]) == 0
    assert main(["--files", str(a), str(b)]) == 1
    assert "price" in capsys.readouterr().out


def test_bad_range_is_a_clean_error(repo):
    assert main(["-C", str(repo), "nope..main"]) == 2
