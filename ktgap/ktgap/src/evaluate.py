"""Stage 2 and 4 of the pipeline: label solutions, then run the generated tests.

Examples
--------
# 2) label each solution correct/buggy with the EvalPlus hidden tests
python src/evaluate.py label --dataset humaneval \
    --solutions results/qwen0.5b/solutions.jsonl --out results/qwen0.5b/solutions_labeled.jsonl

# 4) run each test set against its solution; report how often own tests miss own bugs
python src/evaluate.py run-tests --dataset humaneval \
    --solutions results/qwen0.5b/solutions_labeled.jsonl \
    --tests results/qwen0.5b/selftests.jsonl --out results/qwen0.5b/selftest_results.jsonl
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from common import canonical_code, load_problems, read_jsonl, run_asserts, write_jsonl

# ---------------------------------------------------------------- label (EvalPlus)

def _expected_outputs(dataset, problems):
    from evalplus.data import get_human_eval_plus_hash, get_mbpp_plus_hash
    from evalplus.evaluate import MBPP_OUTPUT_NOT_NONE_TASKS, get_groundtruth
    if dataset == "humaneval":
        return get_groundtruth(problems, get_human_eval_plus_hash(), [])
    return get_groundtruth(problems, get_mbpp_plus_hash(), MBPP_OUTPUT_NOT_NONE_TASKS)


def _check(args):
    from evalplus.evaluate import check_correctness
    dataset, i, problem, solution, expected = args
    r = check_correctness(dataset, i, problem, solution, expected, False, True, i)
    return i, r["base"][0], r["plus"][0]


def run_label(args):
    problems = load_problems(args.dataset)
    expected = _expected_outputs(args.dataset, problems)  # cached by EvalPlus after first run
    rows = read_jsonl(args.solutions)
    jobs = [(args.dataset, i, problems[r["task_id"]], r["solution"], expected[r["task_id"]])
            for i, r in enumerate(rows)]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, base, plus in ex.map(_check, jobs):
            rows[i]["base_status"] = base
            rows[i]["plus_status"] = plus
            rows[i]["correct"] = base == "pass" and plus == "pass"
    write_jsonl(args.out, rows)
    n_ok = sum(r["correct"] for r in rows)
    n_base_only = sum(r["base_status"] == "pass" and not r["correct"] for r in rows)
    print(f"labeled {len(rows)} solutions: {n_ok} correct, {len(rows) - n_ok} buggy "
          f"({n_base_only} buggy ones pass the original tests but fail EvalPlus extra tests)")

# ---------------------------------------------------------------- run-tests

def run_tests(args):
    problems = load_problems(args.dataset)
    sols = read_jsonl(args.solutions)
    if "correct" not in sols[0]:
        raise SystemExit("run-tests needs a labeled solutions file (run the label stage first)")
    tests = read_jsonl(args.tests)
    by_id = {(r["task_id"], r["sample_id"]): r for r in sols}

    # pair every test set with the solution(s) it should be run against
    pairs = []
    for t in tests:
        if t["mode"] == "spec":  # one test set per task -> every solution of that task
            pairs += [(t, s) for s in sols if s["task_id"] == t["task_id"]]
        elif (t["task_id"], t["sample_id"]) in by_id:
            pairs.append((t, by_id[(t["task_id"], t["sample_id"])]))

    # a test is "valid" if it passes on the reference solution (otherwise its expected value is wrong)
    def validity(task_id, asserts):
        code = canonical_code(args.dataset, problems[task_id])
        res = run_asserts(code, asserts, args.per_assert_timeout)
        return [x == "pass" for x in res["results"]]

    def on_solution(pair):
        t, s = pair
        return run_asserts(s["solution"], t["asserts"], args.per_assert_timeout)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        valid_list = list(ex.map(lambda t: validity(t["task_id"], t["asserts"]), tests))
        sol_results = list(ex.map(on_solution, pairs))
    valid_by_test = {id(t): v for t, v in zip(tests, valid_list)}

    out = []
    for (t, s), res in zip(pairs, sol_results):
        valid = valid_by_test[id(t)]
        passed = [x == "pass" for x in res["results"]]
        n = len(passed)
        all_pass = n > 0 and all(passed)  # what a developer would see: "tests are green"
        valid_catch = any(v and not p for v, p in zip(valid, passed))
        out.append({
            "task_id": s["task_id"], "sample_id": s["sample_id"],
            "solution_model": s["solution_model"], "test_model": t["test_model"],
            "mode": t["mode"], "self_test": t["test_model"] == s["solution_model"],
            "correct": s["correct"], "n_asserts": n, "n_valid": sum(valid),
            "assert_results": res["results"], "assert_valid": valid,
            "tests_all_pass": all_pass,
            "valid_test_catches_bug": (not s["correct"]) and valid_catch,
            "missed_bug": (not s["correct"]) and all_pass,     # key outcome for RQ1/RQ2
            "false_alarm": s["correct"] and n > 0 and not all_pass,
        })
    write_jsonl(args.out, out)
    summarize(out)


def summarize(rows):
    def pct(a, b):
        return f"{a}/{b} ({100 * a / b:.1f}%)" if b else f"{a}/0"
    usable = [r for r in rows if r["n_asserts"] > 0]
    buggy = [r for r in usable if not r["correct"]]
    correct = [r for r in usable if r["correct"]]
    n_asserts = sum(r["n_asserts"] for r in usable)
    print("\n=== summary ===")
    print(f"pairs with at least one parsable assert: {pct(len(usable), len(rows))}")
    print(f"assert validity (passes on reference solution): {pct(sum(r['n_valid'] for r in usable), n_asserts)}")
    print(f"buggy solutions whose tests are all green (MISSED BUGS): "
          f"{pct(sum(r['missed_bug'] for r in buggy), len(buggy))}")
    print(f"buggy solutions caught by at least one valid assert: "
          f"{pct(sum(r['valid_test_catches_bug'] for r in buggy), len(buggy))}")
    print(f"correct solutions with a failing test (false alarms): "
          f"{pct(sum(r['false_alarm'] for r in correct), len(correct))}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["label", "run-tests"])
    ap.add_argument("--dataset", choices=["humaneval", "mbpp"], default="humaneval")
    ap.add_argument("--solutions", required=True)
    ap.add_argument("--tests", help="(run-tests) test sets from generate.py tests")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--per-assert-timeout", type=int, default=3)
    args = ap.parse_args()
    if args.stage == "label":
        run_label(args)
    else:
        if not args.tests:
            raise SystemExit("run-tests needs --tests")
        run_tests(args)


if __name__ == "__main__":
    main()
