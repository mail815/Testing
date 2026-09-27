"""Behavioural fingerprints of Python source.

Two sources get the same fingerprint when CPython compiles them to the same
executable code, after removing things that cannot affect what the program
computes:

- comments, whitespace, formatting, blank lines   (not in bytecode at all)
- line and column numbers                          (co_firstlineno, co_linetable, ...)
- names of non-parameter local variables           (only their slot index matters)
- docstrings, unless ``docstrings=True``           (documentation, not logic)
- constant expressions CPython folds (``1 + 2`` == ``3``)

Everything observable by callers is kept: parameter names and defaults,
global/attribute names, constants (with exact type: ``3`` != ``3.0``,
``0.0`` != ``-0.0``), control flow, exception handling, closures, flags.
"""

from __future__ import annotations

import ast
import dis
import hashlib
import types
import warnings
from dataclasses import dataclass, field

_CO_VARARGS, _CO_VARKEYWORDS = 0x04, 0x08


class Unparseable(Exception):
    """The source cannot be compiled by this interpreter."""


@dataclass
class Fingerprint:
    digest: str
    # qualified name -> digest of that code object alone (for explanations)
    units: dict[str, str] = field(default_factory=dict)
    code: types.CodeType | None = field(default=None, repr=False, compare=False)
    # Top-level import bindings (sorted) and the digest of everything else:
    # equal on both sides => only import order or grouping changed.
    imports: tuple = ()
    sans_imports: str = ""


def _n_params(co: types.CodeType) -> int:
    return (co.co_argcount + co.co_kwonlyargcount
            + bool(co.co_flags & _CO_VARARGS) + bool(co.co_flags & _CO_VARKEYWORDS))


def _const(c) -> tuple:
    """Canonical, hash-stable, type-exact form of a constant."""
    if isinstance(c, types.CodeType):
        return ("code", _code_key(c))
    if isinstance(c, (float, complex)):
        return (type(c).__name__, repr(c))  # -0.0 != 0.0, nan == nan
    if isinstance(c, tuple):
        return ("tuple", tuple(_const(x) for x in c))
    if isinstance(c, frozenset):  # iteration order varies with hash seed
        return ("frozenset", tuple(sorted((_const(x) for x in c), key=repr)))
    return (type(c).__name__, c)


def _code_key(co: types.CodeType, deep: bool = True) -> tuple:
    """Everything behavioural about a code object.

    With ``deep=False`` nested code objects are referenced by name only, so a
    change inside an inner function is attributed to that function alone.
    """
    consts = tuple(
        ("code", getattr(c, "co_qualname", c.co_name))
        if isinstance(c, types.CodeType) and not deep else _const(c)
        for c in co.co_consts)
    return (
        co.co_code, getattr(co, "co_exceptiontable", b""),
        consts,
        co.co_names,
        co.co_varnames[:_n_params(co)],  # parameter names are public API
        co.co_nlocals,
        len(co.co_cellvars), len(co.co_freevars),
        co.co_argcount, co.co_posonlyargcount, co.co_kwonlyargcount,
        co.co_flags, co.co_stacksize,
        co.co_name, getattr(co, "co_qualname", co.co_name),
    )


def _digest(key: tuple) -> str:
    return hashlib.blake2b(repr(key).encode(), digest_size=16).hexdigest()


_DOC_HOLDERS = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def _normalize(tree: ast.Module, docstrings: bool) -> None:
    """One pass that (a) drops docstrings unless requested and (b) puts every
    node at line 1, column 0.

    (b) matters because CPython's code generator consults positions: it keeps
    NOPs to hold line numbers and picks different jump forms for multi-line
    expressions, so reformatting alone can change bytecode. Compiling a
    position-free tree makes the output depend on program structure only.
    """
    stack: list = [tree]
    pop, push = stack.pop, stack.append
    while stack:
        node = pop()
        if not docstrings and isinstance(node, _DOC_HOLDERS):
            body = node.body
            if (body and type(body[0]) is ast.Expr and type(body[0].value) is ast.Constant
                    and type(body[0].value.value) is str):
                node.body = body[1:] or [ast.Pass(lineno=1, col_offset=0,
                                                  end_lineno=1, end_col_offset=0)]
        if hasattr(node, "lineno"):
            node.lineno = node.end_lineno = 1
            node.col_offset = node.end_col_offset = 0
        for name in node._fields:
            value = getattr(node, name, None)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        push(item)
            elif isinstance(value, ast.AST):
                push(value)


def _parse(source: bytes | str, docstrings: bool) -> ast.Module:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError) as e:
        raise Unparseable(f"{type(e).__name__}: {e}") from e
    _normalize(tree, docstrings)
    return tree


def _compile(tree: ast.Module) -> types.CodeType:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # e.g. SyntaxWarning from old code
            return compile(tree, "<nfcheck>", "exec", dont_inherit=True)
    except (SyntaxError, ValueError, RecursionError) as e:
        raise Unparseable(f"{type(e).__name__}: {e}") from e


def _split_imports(tree: ast.Module) -> tuple[tuple, ast.Module]:
    bindings, rest = [], []
    for st in tree.body:
        if isinstance(st, ast.Import):
            bindings += [(0, None, a.name, a.asname) for a in st.names]
        elif isinstance(st, ast.ImportFrom) and st.module != "__future__":
            bindings += [(st.level, st.module, a.name, a.asname) for a in st.names]
        else:
            rest.append(st)
    return tuple(sorted(bindings, key=repr)), ast.Module(body=rest, type_ignores=[])


def _walk(co: types.CodeType):
    yield getattr(co, "co_qualname", co.co_name), co
    for c in co.co_consts:
        if isinstance(c, types.CodeType):
            yield from _walk(c)


def code_units(co: types.CodeType) -> dict[str, types.CodeType]:
    """Every code object in a module by qualified name (lambdas etc. numbered)."""
    units: dict[str, types.CodeType] = {}
    for name, sub in _walk(co):
        key, i = name, 1
        while key in units:
            i += 1
            key = f"{name}#{i}"
        units[key] = sub
    return units


@dataclass(frozen=True)
class Quick:
    """The cheap, picklable part of a fingerprint: enough to decide equality."""
    digest: str
    imports: tuple
    # Digest of the functions/classes defined at module level (order-free).
    # If this differs, a change cannot be "import order only".
    nested: str = ""


def _top_level_imports(tree: ast.Module) -> tuple:
    return _split_imports(tree)[0]


def quick(source: bytes | str, docstrings: bool = False) -> Quick:
    tree = _parse(source, docstrings)
    co = _compile(tree)
    imports = _top_level_imports(tree)
    nested = ""
    if imports:
        nested = _digest(tuple(sorted(repr(_const(c)) for c in co.co_consts
                                      if isinstance(c, types.CodeType))))
    return Quick(_digest(_code_key(co)), imports, nested)


def fingerprint(source: bytes | str, docstrings: bool = False) -> Fingerprint:
    tree = _parse(source, docstrings)
    co = _compile(tree)
    units = {k: _digest(_code_key(c, deep=False)) for k, c in code_units(co).items()}
    imports, rest = _split_imports(tree)
    sans = _digest(_code_key(_compile(rest))) if imports else ""
    return Fingerprint(_digest(_code_key(co)), units, co, imports, sans)


def listing(co: types.CodeType) -> list[str]:
    """Readable instruction listing with line numbers and local names erased."""
    n = _n_params(co)
    out = []
    for ins in dis.get_instructions(co):
        if ins.opname in ("RESUME", "CACHE", "NOP"):
            continue
        arg = ins.argrepr
        if ins.opname.startswith(("LOAD_FAST", "STORE_FAST", "DELETE_FAST")) \
                and isinstance(ins.arg, int) and ins.arg >= n:
            arg = f"<local {ins.arg - n}>"
        elif isinstance(ins.argval, types.CodeType):
            arg = f"<code {getattr(ins.argval, 'co_qualname', ins.argval.co_name)}>"
        elif ins.opname.startswith(("JUMP", "POP_JUMP", "FOR_ITER", "SEND")):
            arg = "->"  # absolute offsets shift with any edit; show shape only
        out.append(f"{ins.opname} {arg}".rstrip())
    return out

