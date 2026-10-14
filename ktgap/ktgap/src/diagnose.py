"""Why is the within-task knowing-testing gap weak? Diagnostics on existing results (CPU, exploratory).

A1  bug type: missed bugs that pass the ORIGINAL benchmark tests and fail only EvalPlus's extra
    tests ("edge-case-only") vs bugs that also fail the original tests ("plain failures").
A2  power: true same-task comparisons (missed bug vs test-passing correct solution of the SAME task).
A3  shared vs unique mistakes: missed bugs whose siblings behave the same (low disagreement:
    a shared misreading) vs differently (high disagreement).

For each subset, the task-centered probe score is compared with test-passing correct solutions
(AUROC, 95% CI from a bootstrap over tasks). All results here are exploratory.

python diagnose.py --runs ../results/qwen3b_full:humaneval ../results/qwen3b_mbpp:mbpp \
    --probe ../results/probe_qwen3b_v4.json
"""
import argparse
import json
import os

import numpy as np

from probe import boot_auc

SCORE = "probe: logistic [task-centered]"


def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="RUN_DIR:dataset")
    ap.add_argument("--probe", required=True)
    ap.add_argument("--score", default=SCORE)
    ap.add_argument("--extra", default=None,
                    help="extra own-test results in each run dir (e.g. selftest_results_morecorrect.jsonl): "
                         "own tests for correct solutions that had none, adding same-task comparisons")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    info = {}
    for spec in args.runs:
        run_dir, ds = spec.rsplit(":", 1)
        for r in load(os.path.join(run_dir, "solutions_labeled.jsonl")):
            info[(ds, r["task_id"], r["sample_id"])] = {"base": r.get("base_status"), "plus": r.get("plus_status")}
        dpath = os.path.join(run_dir, "disagree.jsonl")
        if os.path.exists(dpath):
            for r in load(dpath):
                info.setdefault((ds, r["task_id"], r["sample_id"]), {})["disagree"] = r.get("disagree")
    extra_green = {}
    if args.extra:
        for spec in args.runs:
            run_dir, ds = spec.rsplit(":", 1)
            path = os.path.join(run_dir, args.extra)
            if os.path.exists(path):
                for r in load(path):
                    if r["n_asserts"] > 0:
                        extra_green[(ds, r["task_id"], r["sample_id"])] = r["tests_all_pass"]
            else:
                print(f"note: {path} missing")
        print(f"extra own-test results: {len(extra_green)} solutions, "
              f"{sum(extra_green.values())} with all own tests passing")
    rows = []
    for o in json.load(open(args.probe))["oof"]:
        k = (o["dataset"], o["task_id"], o["sample_id"])
        green = o.get("tests_green")
        if green is None and k in extra_green:
            green = extra_green[k]
        if green is not True or k not in info or o.get(args.score) is None:
            continue
        rows.append({**info[k], "key": k, "group": f"{k[0]}::{k[1]}", "buggy": not o["correct"],
                     "score": o[args.score]})
    y = np.array([r["buggy"] for r in rows])
    s = np.array([r["score"] for r in rows], float)
    g = np.array([r["group"] for r in rows])
    correct = ~y
    print(f"test-passing set: {y.sum()} missed bugs, {correct.sum()} correct (score: {args.score})")

    def report(name, mask):
        m = mask | correct
        n_b = int((mask & y).sum())
        if n_b < 5:
            print(f"  {name:44s} n={n_b:4d}  (too few)")
            return
        est, lo, hi, p = boot_auc(y[m], s[m], g[m], args.n_boot, args.seed)
        print(f"  {name:44s} n={n_b:4d}  AUROC {est:.3f} [{lo:.3f}, {hi:.3f}]  p(<=0.5)={p:.3f}")

    print("\nA1  bug type (each subset vs all test-passing correct solutions)")
    base_pass = np.array([r.get("base") == "pass" for r in rows])
    timeout = np.array(["timeout" in str(r.get("plus")) or "timeout" in str(r.get("base")) for r in rows])
    report("all missed bugs", y)
    report("edge-case-only (pass original, fail extra)", y & base_pass)
    report("plain failures (fail original tests)", y & ~base_pass)
    if (y & timeout).sum():
        report("involving a timeout", y & timeout)
    print(f"  share of missed bugs that are edge-case-only: {(y & base_pass).sum() / max(y.sum(), 1):.1%}")

    print("\nA2  true same-task comparisons (missed bug vs correct solution of the same task)")
    wins, n_pairs, per_task = 0.0, 0, {}
    for t in np.unique(g):
        mt = g == t
        b, c = s[mt & y], s[mt & correct]
        if len(b) and len(c):
            w = sum((sb > c).sum() + 0.5 * (sb == c).sum() for sb in b)
            per_task[t] = (w, len(b) * len(c))
            wins += w
            n_pairs += len(b) * len(c)
    if n_pairs:
        rng = np.random.default_rng(args.seed)
        ts = list(per_task)
        draws = []
        for _ in range(args.n_boot):
            pick = rng.choice(len(ts), len(ts))
            w = sum(per_task[ts[i]][0] for i in pick)
            n = sum(per_task[ts[i]][1] for i in pick)
            draws.append(w / n)
        lo, hi = np.percentile(draws, [2.5, 97.5])
        print(f"  {n_pairs} pairs in {len(per_task)} tasks; probe ranks the bug higher in {wins / n_pairs:.3f} "
              f"[{lo:.3f}, {hi:.3f}]")
        print(f"  missed bugs with NO test-passing correct sibling: "
              f"{sum(1 for t in np.unique(g[y]) if t not in per_task)} of {len(np.unique(g[y]))} tasks")

    print("\nA3  shared vs unique mistakes (sibling disagreement of the missed bug)")
    d = np.array([np.nan if r.get("disagree") is None else r["disagree"] for r in rows], float)
    if np.isnan(d[y]).all():
        print("  no disagreement scores (run disagree.py)")
    else:
        med = np.nanmedian(d[y])
        report(f"shared mistake (disagreement <= {med:.2f})", y & (d <= med))
        report(f"unique mistake (disagreement > {med:.2f})", y & (d > med))


if __name__ == "__main__":
    main()
