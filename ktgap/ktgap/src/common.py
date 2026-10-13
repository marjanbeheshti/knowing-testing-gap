"""Shared helpers for the knowing-testing gap pipeline."""
import ast
import json
import os
import re
import subprocess
import sys

# ---------------------------------------------------------------- datasets

def load_problems(dataset):
    """Return {task_id: problem} for 'humaneval' or 'mbpp' (EvalPlus versions)."""
    from evalplus.data import get_human_eval_plus, get_mbpp_plus
    if dataset == "humaneval":
        return get_human_eval_plus()
    if dataset == "mbpp":
        return get_mbpp_plus()
    raise ValueError(f"unknown dataset: {dataset}")


def canonical_code(dataset, problem):
    """Full reference implementation (used to check whether a test is valid)."""
    if dataset == "humaneval":
        return problem["prompt"] + problem["canonical_solution"]
    return problem["canonical_solution"]  # MBPP+ stores the full function


def select_tasks(problems, limit=None, task_ids=None):
    ids = list(problems)
    if task_ids:
        ids = [t for t in ids if t in set(task_ids)]
    if limit:
        ids = ids[:limit]
    return ids

# ---------------------------------------------------------------- jsonl io

def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

# ---------------------------------------------------------------- code extraction

_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text):
    """Return the longest fenced code block, or the raw text if there is none."""
    blocks = _FENCE.findall(text)
    if not blocks:
        # model may have stopped before closing the fence
        m = re.search(r"```(?:python|py)?\s*\n(.*)", text, re.DOTALL)
        return (m.group(1) if m else text).strip()
    return max(blocks, key=len).strip()


_UNITTEST_BINARY = {  # self.assertX(a, b) -> assert a <op> b
    "assertEqual": ast.Eq, "assertEquals": ast.Eq, "assertNotEqual": ast.NotEq,
    "assertIs": ast.Is, "assertIsNot": ast.IsNot, "assertIn": ast.In, "assertNotIn": ast.NotIn,
    "assertGreater": ast.Gt, "assertGreaterEqual": ast.GtE, "assertLess": ast.Lt,
    "assertLessEqual": ast.LtE,
}


def _is_self_assert(call):
    return (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name) and call.func.value.id == "self"
            and call.func.attr.startswith("assert"))


def _kw(call, name):
    for k in call.keywords:
        if k.arg == name:
            return k.value
    return None


def _convert_self_assert(call):
    """Turn one unittest assertion call into an `assert` statement (or None if unsupported)."""
    name, a = call.func.attr, call.args
    if name in _UNITTEST_BINARY and len(a) >= 2:
        return ast.Assert(test=ast.Compare(left=a[0], ops=[_UNITTEST_BINARY[name]()], comparators=[a[1]]))
    if name in ("assertTrue", "assert_") and a:
        return ast.Assert(test=a[0])
    if name == "assertFalse" and a:
        return ast.Assert(test=ast.UnaryOp(op=ast.Not(), operand=a[0]))
    if name == "assertIsNone" and a:
        return ast.Assert(test=ast.Compare(left=a[0], ops=[ast.Is()], comparators=[ast.Constant(None)]))
    if name == "assertIsNotNone" and a:
        return ast.Assert(test=ast.Compare(left=a[0], ops=[ast.IsNot()], comparators=[ast.Constant(None)]))
    if name == "assertIsInstance" and len(a) >= 2:
        return ast.Assert(test=ast.Call(func=ast.Name("isinstance"), args=[a[0], a[1]], keywords=[]))
    if name in ("assertAlmostEqual", "assertAlmostEquals") and len(a) >= 2:
        delta = _kw(call, "delta") or (a[3] if len(a) > 3 else None)
        if delta is not None:
            return ast.parse(f"assert abs(({ast.unparse(a[0])}) - ({ast.unparse(a[1])})) <= ({ast.unparse(delta)})").body[0]
        places = _kw(call, "places") or (a[2] if len(a) > 2 else ast.Constant(7))
        return ast.parse(f"assert round(abs(({ast.unparse(a[0])}) - ({ast.unparse(a[1])})), {ast.unparse(places)}) == 0").body[0]
    if name == "assertCountEqual" and len(a) >= 2:
        return ast.parse(f"assert sorted({ast.unparse(a[0])}, key=repr) == sorted({ast.unparse(a[1])}, key=repr)").body[0]
    if name in ("assertRaises", "assertRaisesRegex") and len(a) >= 2:  # call form
        fn, args = a[1], a[2:] if name == "assertRaises" else a[3:]
        call_src = f"{ast.unparse(fn)}({', '.join(ast.unparse(x) for x in args)})"
        return ast.parse(f"try:\n    {call_src}\nexcept {ast.unparse(a[0])}:\n    pass\nelse:\n"
                         f"    raise AssertionError('expected exception')").body[0]
    return None


class _UnittestToAssert(ast.NodeTransformer):
    """Rewrite unittest/pytest-style checks inside a test body into plain statements."""

    def visit_Expr(self, node):
        if _is_self_assert(node.value):
            new = _convert_self_assert(node.value)
            return ast.copy_location(new, node) if new else node
        return node

    def visit_With(self, node):  # with self.assertRaises(E): body / with pytest.raises(E): body
        self.generic_visit(node)
        if len(node.items) == 1 and isinstance(node.items[0].context_expr, ast.Call):
            cm = node.items[0].context_expr
            is_unittest = _is_self_assert(cm) and cm.func.attr in ("assertRaises", "assertRaisesRegex")
            is_pytest = (isinstance(cm.func, ast.Attribute) and isinstance(cm.func.value, ast.Name)
                         and cm.func.value.id == "pytest" and cm.func.attr == "raises")
            if (is_unittest or is_pytest) and cm.args:
                body = "\n".join("    " + line for s in node.body for line in ast.unparse(s).splitlines())
                return ast.parse(f"try:\n{body}\nexcept {ast.unparse(cm.args[0])}:\n    pass\nelse:\n"
                                 f"    raise AssertionError('expected exception')").body[0]
        return node


def _has_check(stmts):
    return any(isinstance(n, ast.Assert) or (isinstance(n, ast.Raise)) for s in stmts for n in ast.walk(s))


def _unit_from_body(body, uses_self, setup_src):
    stmts = [s for s in body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)
                                     and isinstance(s.value.value, str))]  # drop docstrings
    stmts = [_UnittestToAssert().visit(s) for s in stmts]
    if not _has_check(stmts):
        return None
    src = "\n".join(ast.unparse(ast.fix_missing_locations(s)) for s in stmts)
    if uses_self:
        src = "class _Self:\n    pass\nself = _Self()\n" + setup_src + src
    return src


def extract_asserts(text, max_asserts=None, entry_point=None):
    """Extract test units from a model response.

    Handles plain `assert` lines, pytest-style `def test_*()` functions, and unittest
    classes (`self.assertEqual(...)`, `with self.assertRaises(...)`). Each returned unit is a
    self-contained code snippet that raises AssertionError when the check fails. Imports and
    helper definitions in the response are prepended to every unit; any redefinition or import
    of the function under test (`entry_point`) is dropped so tests always run against the
    candidate solution.
    """
    code = extract_code(text)
    try:
        tree = ast.parse(code)
    except SyntaxError:
        units = []
        for line in code.splitlines():  # salvage single-line asserts
            line = line.strip()
            if line.startswith("assert"):
                try:
                    ast.parse(line)
                    units.append(line)
                except SyntaxError:
                    pass
        return _dedupe(units, max_asserts)

    prelude, units = [], []

    def imports_entry(node):
        return isinstance(node, (ast.Import, ast.ImportFrom)) and entry_point and any(
            (al.asname or al.name).split(".")[-1] == entry_point for al in node.names)

    for node in tree.body:
        if isinstance(node, ast.If) and "__name__" in ast.unparse(node.test):
            continue  # if __name__ == "__main__": unittest.main()
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if not imports_entry(node) and not (isinstance(node, ast.ImportFrom) and node.level):
                prelude.append(ast.unparse(node))
        elif isinstance(node, ast.Assert):
            units.append(ast.unparse(node))
        elif isinstance(node, ast.Expr) and _is_self_assert(node.value):
            continue
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == entry_point:
                continue  # never let the test redefine the function under test
            if node.name.startswith("test"):
                u = _unit_from_body(node.body, False, "")
                if u:
                    units.append(u)
            else:
                prelude.append(ast.unparse(node))  # helper function
        elif isinstance(node, ast.ClassDef):
            methods = {m.name: m for m in node.body if isinstance(m, ast.FunctionDef)}
            setup_src = ""
            if "setUp" in methods:
                setup_src = "\n".join(ast.unparse(s) for s in methods["setUp"].body) + "\n"
            for name, m in methods.items():
                if name.startswith("test"):
                    u = _unit_from_body(m.body, True, setup_src)
                    if u:
                        units.append(u)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            prelude.append(ast.unparse(node))  # shared test data

    head = "\n".join(prelude)
    units = [(head + "\n" + u) if head else u for u in units]
    return _dedupe(units, max_asserts)


def _dedupe(units, max_units):
    seen, out = set(), []
    for u in units:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out[:max_units] if max_units else out

# ---------------------------------------------------------------- sandboxed execution

_RESULT_MARK = "__KTGAP_RESULT__"

_HARNESS = r'''
import json, signal, sys
try:
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
except Exception:
    pass
payload = json.loads(sys.stdin.read())
T = payload["per_assert_timeout"]

class _Timeout(BaseException):
    pass

def _handler(signum, frame):
    raise _Timeout()

signal.signal(signal.SIGALRM, _handler)
ns = {"__name__": "__ktgap__"}
load = "ok"
try:
    signal.alarm(T)
    exec(payload["code"], ns)
    signal.alarm(0)
except BaseException as e:
    signal.alarm(0)
    load = "error:" + type(e).__name__
results = []
for a in payload["asserts"]:
    if load != "ok":
        results.append("not_run")
        continue
    try:
        signal.alarm(T)
        exec(a, ns)
        signal.alarm(0)
        results.append("pass")
    except AssertionError:
        signal.alarm(0)
        results.append("fail")
    except _Timeout:
        results.append("timeout")
    except BaseException as e:
        signal.alarm(0)
        results.append("error:" + type(e).__name__)
sys.stdout.write("\n''' + _RESULT_MARK + r'''" + json.dumps({"load": load, "results": results}) + "\n")
'''


def run_asserts(code, asserts, per_assert_timeout=3, total_timeout=60):
    """Run each assert against `code` in a separate Python process.

    Returns {"load": "ok" | "error:...", "results": ["pass"|"fail"|"timeout"|"error:X"|...]}.
    Untrusted model code runs here: use only on a VM / Colab, never on a machine you care about.
    """
    if not asserts:
        return {"load": "ok", "results": []}
    payload = json.dumps({"code": code, "asserts": asserts,
                          "per_assert_timeout": per_assert_timeout})
    try:
        proc = subprocess.run([sys.executable, "-c", _HARNESS], input=payload,
                              capture_output=True, text=True, timeout=total_timeout)
        for line in proc.stdout.splitlines()[::-1]:
            if line.startswith(_RESULT_MARK):
                return json.loads(line[len(_RESULT_MARK):])
        return {"load": "crash", "results": ["crash"] * len(asserts)}
    except subprocess.TimeoutExpired:
        return {"load": "timeout", "results": ["timeout"] * len(asserts)}
