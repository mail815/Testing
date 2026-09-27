# nfcheck

**Prove or disprove "no functional change" claims in git history.**

Commits say "NFC", "reformat", "style:" or "refactor" all the time. They are
promises that behaviour did not change, and nobody checks them. nfcheck
compiles each changed Python file before and after the commit with Python's
own compiler, erases everything that cannot affect behaviour, and compares
what is left.

```
$ nfcheck HEAD~200..HEAD
✓ 02756dd417 proven         reformat changelog, update docs config  [promises: "reformat"]
✗ 93ba3ba112 PROMISE BROKEN apply black  [promises: "apply black"]
    ✗ setup.py: compiled behaviour differs
        ~ changed: <module>
             LOAD_ATTR search
            -LOAD_CONST "__version__ = '(.*?)'"
            +LOAD_CONST '__version__ = "(.*?)"'
✗ 44d5da00b5 PROMISE BROKEN Reformat codebase with isort  [promises: "Reformat"]
    ≈ fuzz.py: identical except import order/grouping (differs only if imports have side effects)
✗ 4e85c776ec REVIEW         refactor open_stream and text stream utilities  [refactor: "refactor"]
    ✗ click/_compat.py: compiled behaviour differs
        ~ changed: <module>
            +LOAD_CONST <code _is_compat_stream_attr>
```

(Excerpts of real output from the histories of [click](https://github.com/pallets/click)
and [black](https://github.com/psf/black), shortened.)

## Explained simply

Someone says "I only tidied my room. I didn't move anything important."
nfcheck takes an X-ray of the room before and after. The X-ray can't see
dust, cushions or where the posters hang (formatting, comments, blank lines).
It *can* see the furniture (the program's actual logic). If the X-rays match,
the promise is **proven**. If they don't, nfcheck circles the exact piece of
furniture that moved.

## Install and use

```bash
pip install .                    # no dependencies; Python 3.9+
nfcheck                          # commits on your branch that aren't on main yet
nfcheck HEAD~50..HEAD            # a range
nfcheck --all v1.0..v2.0         # every commit, labelled or not
nfcheck --commit abc123          # one commit, even without a claim
nfcheck --files old.py new.py    # two files, no git needed
nfcheck --json ...               # for tools
```

Exit status: `0` no promise broken · `1` a promise was broken · `2` usage/git error ·
`3` unverifiable claim (only with `--strict`).

### In CI (GitHub Actions)

```yaml
- uses: actions/checkout@v4
  with: { fetch-depth: 0 }
- run: pip install nfcheck && nfcheck origin/${{ github.base_ref }}..HEAD
```

### As a git hook

```bash
printf '#!/bin/sh\nexec nfcheck @{upstream}..HEAD\n' > .git/hooks/pre-push && chmod +x .git/hooks/pre-push
```

## What the verdicts mean

| Verdict | Meaning |
|---|---|
| **✓ proven** | Every changed Python file compiles to identical behaviour. This is a proof, not a heuristic (see below). |
| **≈ imports-reordered** | Identical except the order or grouping of top-level imports. Harmless unless an import has side effects (e.g. `gevent.monkey`). |
| **✗ PROMISE BROKEN** | A *strong* claim ("NFC", "no functional change", "reformat", `style:`, "apply black") but compiled code differs. |
| **✗ REVIEW** | A "refactor" changed compiled code. Refactors are *expected* to change code, so this isn't a verdict of guilt: it lists exactly which functions to review. |
| **? unverifiable** | Something changed that nfcheck can't judge (config, C code, data files, syntax this Python can't parse). |

Docs (`*.md`, `*.rst`, `docs/`) and type stubs (`*.pyi`) are ignored. Test
files are reported but not judged (use `--include-tests` to judge them).

## Why "proven" is a proof

Two files get the same fingerprint only when CPython compiles them to the same
instructions, constants (type-exact: `3` ≠ `3.0`, `0.0` ≠ `-0.0`), global and
attribute names, parameter names and defaults, flags, closures and exception
handling. nfcheck removes only things that cannot change what the program computes:

- **Layout.** Every syntax node is moved to line 1, column 0 before compiling.
  This matters: CPython's code generator looks at line numbers and emits
  different bytecode for the same expression split across lines. Without this
  step, formatters like Black trigger false alarms.
- **Names of local variables** (not parameters: those are public API).
- **Docstrings** (use `--docstrings` to count them).

Because it compares what Python actually runs, nfcheck also proves things a
reviewer can miss. On black's history it proved that a commit changing
`return -1` to `return -2` was harmless: the line comes after a `try` block
that always returns, so the compiler deletes it as unreachable.

**What "proven" does not cover:** code that inspects itself (line numbers in
tracebacks, `inspect.getsource`, `locals()` keys, `__doc__` when docstrings
are ignored), and non-Python files. Verdicts come from the Python version
running nfcheck.

## Validation

Run over the complete histories of click, black and attrs (7,564 commits):

- **Soundness audit:** all 1,394 file changes nfcheck marked *proven* were
  re-checked against an independent syntax-tree comparison. 1,325 were
  formatting, comments or docstrings only. The other 69 were inspected by hand
  (local renames, local annotations, `u''` prefixes, `f''` without
  placeholders, `...`/`pass`, `import X as X`, unreachable code). **0 false proofs.**
- **Real findings:**
  - click's "apply black" also changed a version-parsing regex in `setup.py`
    and the escaping of a regex in `parser.py`.
  - black's "Reformat codebase with isort" changed import order in 4 files,
    plus real code in `src/black/__init__.py`.
- **Speed:** full histories in 12s (click, 3,377 commits), 20s (black) and 5s
  (attrs) on 4 cores. Each file version is compiled once. Git is read through
  two long-lived processes, and compilation runs on all cores.

## Prior art

Before building this, I searched for existing tools that check "no functional
change" claims mechanically. I found discussions of the gap (LLVM's "NFC"
became "NFCI" after "NFC" commits broke things; an open issue titled
"Nothing mechanical checks that a refactor preserved behaviour"), research
that uses language models to *guess* whether refactors changed behaviour, and
general techniques that compare build outputs (reproducible builds, compiler
translation validation). I found no tool that holds commit-message promises
to a compiled-code proof. That is not proof that none exists. If you know of
one, please open an issue.
