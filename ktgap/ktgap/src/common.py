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


def extract_asserts(text, max_asserts=None):
    """Pull assert statements out of a model response (handles multi-line asserts)."""
    code = extract_code(text)
    asserts = []
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                seg = ast.get_source_segment(code, node)
                if seg:
                    asserts.append(seg)
    except SyntaxError:
        # fall back to single-line asserts that parse on their own
        for line in code.splitlines():
            line = line.strip()
            if line.startswith("assert"):
                try:
                    ast.parse(line)
                    asserts.append(line)
                except SyntaxError:
                    pass
    seen, unique = set(), []
    for a in asserts:  # drop duplicates, keep order
        if a not in seen:
            seen.add(a)
            unique.append(a)
    return unique[:max_asserts] if max_asserts else unique

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
