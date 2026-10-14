"""Find real motivating examples for Fig. 1 (CPU, read-only).

A good example is a MISSED BUG (all of the model's own code-mode tests pass) where
  * one of the model's own assertions is wrong in exactly the buggy way: it fails on the reference
    solution but passes on the buggy code (it "copies the bug"), and
  * an assertion written from the task description alone (spec mode) is valid and fails on the
    buggy code (it "catches the bug").
For each candidate we report the probe's rank among the green (test-passing) solutions and, if
available, the model's own review verdict. Short candidates are listed first (they fit a figure).

python find_example.py --run ../results/qwen3b_mbpp --dataset mbpp --probe ../results/probe_qwen3b_v4.json
"""
import argparse
import json
import os

from common import canonical_code, load_problems


def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--dataset", required=True, choices=["humaneval", "mbpp"])
    ap.add_argument("--probe", required=True)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--max-lines", type=int, default=12, help="skip solutions longer than this")
    args = ap.parse_args()

    problems = load_problems(args.dataset)
    sols = {(r["task_id"], r["sample_id"]): r for r in load(os.path.join(args.run, "solutions_labeled.jsonl"))}
    own_tests = {(r["task_id"], r["sample_id"]): r for r in load(os.path.join(args.run, "selftests.jsonl"))}
    own_res = {(r["task_id"], r["sample_id"]): r for r in load(os.path.join(args.run, "selftest_results.jsonl"))}
    spec_tests = {r["task_id"]: r for r in load(os.path.join(args.run, "selftests_spec.jsonl"))}
    spec_res = {(r["task_id"], r["sample_id"]): r for r in load(os.path.join(args.run, "selftest_results_spec.jsonl"))}
    verdicts = {}
    vpath = os.path.join(args.run, "selftests_judge_first.jsonl")
    if os.path.exists(vpath):
        verdicts = {(r["task_id"], r["sample_id"]): (r.get("judgment", ""), r.get("verdict")) for r in load(vpath)}

    oof = {(o["task_id"], o["sample_id"]): o for o in json.load(open(args.probe))["oof"] if o["dataset"] == args.dataset}
    green = [k for k, r in own_res.items() if r["n_asserts"] > 0 and r["tests_all_pass"] and k in oof
             and oof[k].get("probe: logistic") is not None]
    ranked = sorted(green, key=lambda k: -oof[k]["probe: logistic"])
    pct = {k: (i + 1) / len(ranked) for i, k in enumerate(ranked)}  # 0.05 = top 5% most suspicious

    cands = []
    for k in green:
        if sols[k]["correct"] or k not in spec_res or k[0] not in spec_tests:
            continue
        r, t = own_res[k], own_tests[k]
        copies = [a for a, v, x in zip(t["asserts"], r["assert_valid"], r["assert_results"]) if not v and x == "pass"]
        s_res, s_tests = spec_res[k], spec_tests[k[0]]
        catches = [a for a, v, x in zip(s_tests["asserts"], s_res["assert_valid"], s_res["assert_results"])
                   if v and x != "pass"]
        n_lines = sols[k]["solution"].count("\n") + 1
        if copies and catches and n_lines <= args.max_lines:
            cands.append((n_lines + min(len(a) for a in copies) / 40, k, copies, catches))
    print(f"{len(green)} green solutions, {sum(not sols[k]['correct'] for k in green)} missed bugs; "
          f"{len(cands)} candidates with a bug-copying own test AND a catching spec test\n")
    for _, k, copies, catches in sorted(cands)[:args.top]:
        task = problems[k[0]]
        print("=" * 90)
        print(f"{k[0]} sample {k[1]} | probe rank: top {pct[k]:.0%} of green code "
              f"| base={sols[k].get('base_status')} plus={sols[k].get('plus_status')}")
        print("--- task:\n" + task["prompt"].strip()[:600])
        print("--- reference solution:\n" + canonical_code(args.dataset, task).strip()[:600])
        print("--- model's buggy solution:\n" + sols[k]["solution"].strip())
        print("--- model's own test that copies the bug (fails on reference, passes on buggy code):")
        print("\n".join(copies[:2]))
        print("--- test from the task only that catches the bug:")
        print("\n".join(catches[:2]))
        if k in verdicts:
            judgment, verdict = verdicts[k]
            print(f"--- model's self-review verdict: {verdict} | ...{judgment.strip()[-160:]}")
    if not cands:
        print("no candidate found; try --max-lines 20 or another run")


if __name__ == "__main__":
    main()
