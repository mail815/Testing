import textwrap

import pytest

from nfcheck.fingerprint import Unparseable, fingerprint

BASE = textwrap.dedent('''
    """Module docstring."""
    import os

    LIMIT = 3

    def total(xs, limit=LIMIT, *, scale=1):
        """Sum the first few values."""
        acc = 0
        for i, x in enumerate(xs):
            if i < limit:
                acc += x * scale
        return acc

    class Box:
        size = 2
        def area(self):
            return self.size ** 2
''')


def same(a: str, b: str, **kw) -> bool:
    return fingerprint(a, **kw).digest == fingerprint(b, **kw).digest


# --- things that must NOT count as behaviour changes ------------------------
@pytest.mark.parametrize("name,edit", [
    ("comments", lambda s: s.replace("acc = 0", "acc = 0  # running total")),
    ("blank lines", lambda s: s.replace("\n\n", "\n\n\n\n")),
    ("shifted down", lambda s: "\n" * 40 + s),
    ("local renamed", lambda s: s.replace("acc", "running")),
    ("loop var renamed", lambda s: s.replace("i, x", "k, x").replace("if i <", "if k <")),
    ("docstring edited", lambda s: s.replace("Sum the first few values.", "Add up.")),
    ("docstring removed", lambda s: s.replace('    """Sum the first few values."""\n', "")),
    ("constant folded", lambda s: s.replace("LIMIT = 3", "LIMIT = 1 + 2")),
    ("parenthesised", lambda s: s.replace("acc += x * scale", "acc += (x * scale)")),
    ("line split", lambda s: s.replace("if i < limit:", "if (\n            i < limit\n        ):")),
    ("quote style", lambda s: s.replace('"""Module docstring."""', "'''Module docstring.'''")),
    ("trailing comma", lambda s: s.replace("*, scale=1)", "*, scale=1,)")),
])
def test_behaviour_preserving_edits_are_proven(name, edit):
    assert same(BASE, edit(BASE)), name


def test_black_style_boolean_chain_split_is_proven():
    one = "def f(a, b, c):\n    return a or b and g(b) or ''\n"
    many = ("def f(a, b, c):\n    return (\n        a\n        or b\n"
            "        and g(b)\n        or \"\"\n    )\n")
    assert same(one, many)


# --- things that MUST count as behaviour changes ----------------------------
@pytest.mark.parametrize("name,edit", [
    ("comparison", lambda s: s.replace("i < limit", "i <= limit")),
    ("parameter renamed", lambda s: s.replace("limit", "cap")),
    ("keyword-only renamed", lambda s: s.replace("scale", "factor")),
    ("default changed", lambda s: s.replace("limit=LIMIT", "limit=4")),
    ("int to float", lambda s: s.replace("LIMIT = 3", "LIMIT = 3.0")),
    ("int to bool", lambda s: s.replace("acc = 0", "acc = False")),
    ("global renamed", lambda s: s.replace("LIMIT", "MAX")),
    ("attribute", lambda s: s.replace("self.size", "self.width")),
    ("method renamed", lambda s: s.replace("def area", "def surface")),
    ("statement order", lambda s: s.replace("    size = 2\n", "").replace(
        "            return self.size ** 2", "            return self.size ** 2\n    size = 2")),
    ("operator", lambda s: s.replace("x * scale", "x + scale")),
    ("import", lambda s: s.replace("import os", "import sys")),
    ("keyword-only made positional", lambda s: s.replace("*, scale", "scale")),
])
def test_behaviour_changes_are_detected(name, edit):
    assert not same(BASE, edit(BASE)), name


def test_docstrings_count_when_requested():
    edited = BASE.replace("Sum the first few values.", "Add up.")
    assert not same(BASE, edited, docstrings=True)


def test_signed_zero_and_nan_constants():
    assert not same("x = 0.0\n", "x = -0.0\n")
    assert same("x = float('nan')\n", "x = float('nan')\n")
    assert same("x = 1e1000 - 1e1000\n", "x = 1e1000 - 1e1000\n")  # folded nan


def test_frozenset_constant_order_is_stable():
    a = "def f(x):\n    return x in {'a', 'b', 'c', 'd', 'e'}\n"
    b = "def f(x):\n    return x in {'e', 'd', 'c', 'b', 'a'}\n"
    assert same(a, b)


def test_units_attribute_change_to_innermost_function():
    a = "def outer():\n    def inner():\n        return 1\n    return inner\n"
    b = a.replace("return 1", "return 2")
    fa, fb = fingerprint(a), fingerprint(b)
    changed = {k for k in fa.units if fa.units[k] != fb.units.get(k)}
    assert changed == {"outer.<locals>.inner"}


def test_import_reorder_is_recognised_but_not_proven():
    a = "import os\nimport sys\nfrom a import (x, y)\nprint(os, sys)\n"
    b = "from a import y, x\nimport sys\nimport os\nprint(os, sys)\n"
    fa, fb = fingerprint(a), fingerprint(b)
    assert fa.digest != fb.digest
    assert fa.imports == fb.imports and fa.sans_imports == fb.sans_imports


def test_future_imports_are_not_treated_as_movable():
    a = "from __future__ import annotations\nimport os\n"
    assert fingerprint(a).imports == ((0, None, "os", None),)


def test_unparseable_source():
    with pytest.raises(Unparseable):
        fingerprint("def broken(:\n")
    with pytest.raises(Unparseable):
        fingerprint(b"print 'python 2'\n")


def test_encoding_cookie_bytes():
    src = "# -*- coding: latin-1 -*-\nname = 'caf\xe9'\n".encode("latin-1")
    assert same(src, "name = 'café'\n")


def test_docstring_only_bodies():
    a = 'class E(Exception):\n    """Oops."""\n\ndef f():\n    """Nothing."""\n'
    b = "class E(Exception):\n    pass\n\ndef f():\n    pass\n"
    assert same(a, b)
    assert not same(a, b, docstrings=True)
