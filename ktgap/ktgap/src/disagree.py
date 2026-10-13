"""Sibling disagreement: does a solution behave differently from the model's other solutions?

For each task, the model wrote several solutions (siblings). We run every sibling on the same
inputs and score each solution by how often its output differs from its siblings' outputs
(cf. CodeT / execution consensus). No ground truth is used: the inputs come from the model's
OWN description-only tests (spec mode), never from the hidden EvalPlus tests.

score = mean over inputs of the share of other siblings whose output differs (0 = always agrees)

Example
-------
python disagree.py --dataset mbpp --solutions ../results/qwen3b_mbpp/solutions_labeled.jsonl \
    --tests ../results/qwen3b_mbpp/selftests_spec.jsonl --out ../results/qwen3b_mbpp/disagree.jsonl
"""
import argparse
import ast
import json
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from common import load_problems, read_jsonl, write_jsonl

_MARK = "__KTGAP_CALLS__"
_HARNESS = r'''
import json, signal, sys, itertools
try:
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
except Exception:
    pass
p = json.loads(sys.stdin.read())
class _T(BaseException):
    pass
def _h(s, f):
    raise _T()
signal.signal(signal.SIGALRM, _h)
def _norm(v, d=0):
    if d > 20:
        return repr(v)
    if isinstance(v, float):
        return round(v, 6)
    if isinstance(v, (list, tuple)):
        return (type(v).__name__, [_norm(x, d + 1) for x in v])
    if isinstance(v, (set, frozenset)):
        return ("set", sorted(repr(_norm(x, d + 1)) for x in v))
    if isinstance(v, dict):
        return ("dict", sorted((repr(_norm(k, d + 1)), repr(_norm(x, d + 1))) for k, x in v.items()))
    if hasattr(v, "__next__"):
        return ("iter", [_norm(x, d + 1) for x in itertools.islice(v, 1000)])
    return v
ns = {"__name__": "__ktgap__"}
out = []
try:
    signal.alarm(5); exec(p["code"], ns); signal.alarm(0); ok = True
except BaseException:
    signal.alarm(0); ok = False
for setup, call in p["inputs"]:
    if not ok:
        out.append("LOADERR"); continue
    local = dict(ns)
    try:
        signal.alarm(2)
        if setup:
            exec(setup, local)
        r = repr(_norm(eval(call, local)))
        signal.alarm(0)
        out.append(r[:2000])
    except _T:
        out.append("TIMEOUT")
    except BaseException as e:
        signal.alarm(0)
        out.append("EXC:" + type(e).__name__)
sys.stdout.write("\n''' + _MARK + r'''" + json.dumps(out) + "\n")
'''


def inputs_from_tests(units, entry_point, max_inputs):
    """(setup_code, call_expression) pairs: every call to the function under test inside the asserts."""
    seen, pairs = set(), []
    for u in units:
        try:
            tree = ast.parse(u)
        except SyntaxError:
            continue
        setup = "\n".join(ast.unparse(s) for s in tree.body if not isinstance(s, ast.Assert))
        def calls_entry(node):
            return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == entry_point
                       for n in ast.walk(node))

        for s in tree.body:
            if not isinstance(s, ast.Assert):
                continue
            exprs = []
            if isinstance(s.test, ast.Compare):
                # compare the whole side the test compares, e.g. set(f(x)), so the test's own
                # normalization (set, sorted, round, ...) is respected
                exprs = [side for side in [s.test.left] + s.test.comparators if calls_entry(side)]
            if not exprs:
                exprs = [n for n in ast.walk(s.test) if isinstance(n, ast.Call)
                         and isinstance(n.func, ast.Name) and n.func.id == entry_point]
            for node in exprs:
                call = ast.unparse(node)
                if call not in seen:
                    seen.add(call)
                    pairs.append((setup, call))
    return pairs[:max_inputs]


def run_calls(code, inputs, timeout=60):
    payload = json.dumps({"code": code, "inputs": inputs})
    try:
        proc = subprocess.run([sys.executable, "-c", _HARNESS], input=payload, capture_output=True,
                              text=True, timeout=timeout)
        for line in proc.stdout.splitlines()[::-1]:
            if line.startswith(_MARK):
                return json.loads(line[len(_MARK):])
    except subprocess.TimeoutExpired:
        pass
    return ["CRASH"] * len(inputs)


def disagreement_scores(outputs):
    """outputs: list (one per sibling) of equal-length output lists -> one score per sibling."""
    n = len(outputs)
    if n < 2 or not outputs[0]:
        return [None] * n
    scores = []
    for i in range(n):
        per_input = [sum(outputs[i][k] != outputs[j][k] for j in range(n) if j != i) / (n - 1)
                     for k in range(len(outputs[i]))]
        scores.append(sum(per_input) / len(per_input))
    return scores


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["humaneval", "mbpp"], required=True)
    ap.add_argument("--solutions", required=True)
    ap.add_argument("--tests", required=True, help="spec-mode tests (one test set per task)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-inputs", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    problems = load_problems(args.dataset)
    sols = read_jsonl(args.solutions)
    tests = {t["task_id"]: t for t in read_jsonl(args.tests)}
    if any(t.get("mode") != "spec" for t in tests.values()):
        raise SystemExit("use description-only (spec) tests as the input source")
    by_task = defaultdict(list)
    for s in sols:
        by_task[s["task_id"]].append(s)
    inputs = {t: inputs_from_tests(tests[t]["asserts"], problems[t]["entry_point"], args.max_inputs)
              if t in tests else [] for t in by_task}
    jobs = [(t, i) for t, ss in by_task.items() for i in range(len(ss)) if inputs[t]]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(lambda ti: run_calls(by_task[ti[0]][ti[1]]["solution"], inputs[ti[0]]), jobs))
    outs = defaultdict(dict)
    for (t, i), r in zip(jobs, results):
        outs[t][i] = r
    rows = []
    for t, ss in by_task.items():
        scores = disagreement_scores([outs[t][i] for i in range(len(ss))]) if inputs[t] else [None] * len(ss)
        for s, sc in zip(ss, scores):
            rows.append({"dataset": args.dataset, "task_id": t, "sample_id": s["sample_id"],
                         "disagree": sc, "n_inputs": len(inputs[t]), "correct": s.get("correct")})
    write_jsonl(args.out, rows)
    scored = [r for r in rows if r["disagree"] is not None]
    print(f"scored {len(scored)}/{len(rows)} solutions ({sum(1 for t in by_task if inputs[t])} tasks with inputs)")
    if scored and scored[0]["correct"] is not None:
        from sklearn.metrics import roc_auc_score
        y = [0 if r["correct"] else 1 for r in scored]
        print(f"sanity: AUROC of disagreement for buggy vs correct (all solutions) = "
              f"{roc_auc_score(y, [r['disagree'] for r in scored]):.3f}")


if __name__ == "__main__":
    main()
