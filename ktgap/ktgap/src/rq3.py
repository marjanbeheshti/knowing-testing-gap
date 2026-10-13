"""RQ3: can the model's internal signal decide WHICH code gets extra testing?

Deployment scenario: the model writes code and its own tests. If its tests all pass ("green"),
the code would ship. A trigger policy picks a budget of green solutions for ESCALATION: extra
tests written in another condition (e.g. from the task description only). A bug is caught if
the extra tests fail; a correct solution whose extra tests fail is a false alarm.

Because each escalation condition was run on all solutions in advance, every policy and budget
is evaluated offline, on exactly the same tests.

Policies (higher score = escalate first):
  probe            out-of-fold hidden-state probe score (from probe.py --out)
  probe-centered   same, minus the mean over the task's other solutions
  confidence       low mean token log-probability of the response
  verdict          the model's own spoken verdict from judge_first ("Verdict: buggy" first)
  random           expected value of a random choice at the same budget
  all              escalate every green solution (budget 100%)

Example
-------
python rq3.py --runs ../results/qwen3b_full:humaneval ../results/qwen3b_mbpp:mbpp \
    --probe ../results/probe_qwen3b.json --escalate spec=selftest_results_spec.jsonl
"""
import argparse
import json
import os

import numpy as np


def load_results(path):
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out[(r["task_id"], r["sample_id"])] = r
    return out


def load_verdicts(path):
    v = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            v[(r["task_id"], r["sample_id"])] = r.get("verdict")
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="RUN_DIR:dataset")
    ap.add_argument("--probe", required=True, help="JSON written by probe.py --out")
    ap.add_argument("--base", default="selftest_results.jsonl", help="own-test (code mode) results file in each run")
    ap.add_argument("--escalate", nargs="+", default=["spec=selftest_results_spec.jsonl"],
                    help="NAME=results file in each run dir, one per escalation condition")
    ap.add_argument("--verdicts", default=None, help="judge_first TESTS file in each run dir (adds the verdict policy)")
    ap.add_argument("--budgets", default="0.1,0.2,0.3,0.5,1.0")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    budgets = [float(b) for b in args.budgets.split(",")]
    rng = np.random.default_rng(args.seed)

    probe = json.load(open(args.probe))
    oof = {(o["dataset"], o["task_id"], o["sample_id"]): o for o in probe["oof"]}

    # ---- the green set: solutions whose own tests all pass -----------------------------------
    rows = []
    escal = {name: {} for name, _ in (e.split("=", 1) for e in args.escalate)}
    verdicts = {}
    for spec in args.runs:
        run_dir, dataset = spec.rsplit(":", 1)
        base = load_results(os.path.join(run_dir, args.base))
        for e in args.escalate:
            name, fname = e.split("=", 1)
            path = os.path.join(run_dir, fname)
            if os.path.exists(path):
                for k, r in load_results(path).items():
                    escal[name][(dataset, *k)] = r
            else:
                print(f"note: {path} missing; condition '{name}' has no data for {dataset}")
        if args.verdicts and os.path.exists(os.path.join(run_dir, args.verdicts)):
            for k, v in load_verdicts(os.path.join(run_dir, args.verdicts)).items():
                verdicts[(dataset, *k)] = v
        for (task, sample), r in base.items():
            if r["n_asserts"] == 0 or not r["tests_all_pass"]:
                continue  # not green: own tests already raised an alarm
            key = (dataset, task, sample)
            o = oof.get(key)
            if o is None:
                continue
            rows.append({"key": key, "group": f"{dataset}::{task}", "buggy": not r["correct"],
                         "probe": o.get("probe: logistic"),
                         "probe-centered": o.get("probe: logistic [task-centered]"),
                         "confidence": o.get("baseline: low mean log-prob"),
                         "verdict": (1.0 if verdicts.get(key) == "buggy" else 0.0) if key in verdicts else None})
    y = np.array([r["buggy"] for r in rows])
    groups = np.array([r["group"] for r in rows])
    print(f"green solutions (own tests all pass): {len(rows)} = {y.sum()} missed bugs + {(~y).sum()} correct")

    policies = ["probe", "probe-centered", "confidence"] + (["verdict"] if verdicts else [])
    scores = {}
    for p in policies:
        s = np.array([np.nan if r[p] is None else r[p] for r in rows], dtype=float)
        if np.isnan(s).any():
            print(f"note: policy '{p}' has no score for {int(np.isnan(s).sum())} solutions (ranked last)")
            s = np.where(np.isnan(s), -np.inf, s)
        scores[p] = s + 1e-9 * rng.random(len(s))  # random tie-break (matters for the 0/1 verdict)

    for name, res in escal.items():
        flag = np.array([(r["key"] in res and res[r["key"]]["n_asserts"] > 0
                          and not res[r["key"]]["tests_all_pass"]) for r in rows])
        covered = np.array([r["key"] in res for r in rows])
        print(f"\n=== escalation: '{name}' (extra tests fail on: {flag[y].sum()}/{y.sum()} missed bugs, "
              f"{flag[~y].sum()}/{(~y).sum()} correct; coverage {covered.mean():.0%}) ===")
        print("caught = share of the missed bugs caught by escalation; false alarm = share of green correct "
              "solutions flagged;\n'vs random' = caught minus the random-policy expectation, 95% CI from a "
              "bootstrap over tasks")
        header = f"{'budget':>7s} {'policy':16s}{'escalated':>10s}{'caught':>9s}{'false alarm':>13s}" \
                 f"{'precision':>11s}   vs random"
        print(header)
        tasks = np.unique(groups)
        members = {g: np.where(groups == g)[0] for g in tasks}

        def evaluate(idx, s, b):
            n = max(1, int(round(b * len(idx))))
            top = idx[np.argsort(-s[idx])[:n]]
            chosen = np.zeros(len(y), bool)
            chosen[top] = True
            yb, fb = y[idx], flag[idx]
            caught = (chosen[idx] & yb & fb).sum() / max(yb.sum(), 1)
            false_alarm = (chosen[idx] & ~yb & fb).sum() / max((~yb).sum(), 1)
            precision = yb[chosen[idx]].mean()
            random_caught = b * (yb & fb).sum() / max(yb.sum(), 1)
            return n, caught, false_alarm, precision, caught - random_caught

        all_idx = np.arange(len(y))
        for b in budgets:
            rand_caught = b * (y & flag).sum() / max(y.sum(), 1)
            rand_fa = b * (~y & flag).sum() / max((~y).sum(), 1)
            print(f"{b:>6.0%}  {'random (expected)':16s}{int(round(b * len(y))):>10d}{rand_caught:>9.3f}"
                  f"{rand_fa:>13.3f}{y.mean():>11.3f}")
            if b >= 1.0:
                continue
            for p in policies:
                n, caught, fa, prec, gain = evaluate(all_idx, scores[p], b)
                draws = []
                for _ in range(args.n_boot):
                    pick = np.concatenate([members[g] for g in rng.choice(tasks, len(tasks))])
                    draws.append(evaluate(pick, scores[p], b)[4])
                lo, hi = np.percentile(draws, [2.5, 97.5])
                print(f"{'':7s} {p:16s}{n:>10d}{caught:>9.3f}{fa:>13.3f}{prec:>11.3f}   "
                      f"{gain:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        n_missed = int(y.sum())
        print(f"\nshipped bugs without escalation: {n_missed}. With escalation of everything ('all'): "
              f"{n_missed - int((y & flag).sum())} shipped, {int((~y & flag).sum())} false alarms, "
              f"{len(y)} extra test runs.")


if __name__ == "__main__":
    main()
