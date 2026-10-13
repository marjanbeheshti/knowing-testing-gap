"""RQ3: can the model's internal signal decide WHICH code gets extra testing?

Deployment scenario: the model writes code and its own tests. If its tests all pass ("green"),
the code would ship. A trigger policy picks a budget of green solutions for ESCALATION: extra
tests from another condition. A bug is caught if the extra tests fail; a correct solution whose
extra tests fail is a false alarm. Every condition was run on all solutions in advance, so each
policy and budget is evaluated offline on exactly the same tests.

Escalations: NAME=FILE, or NAME=FILE1+FILE2 (flagged if ANY of the extra test sets fails).

Policies (higher score = escalate first):
  probe           out-of-fold hidden-state probe score (probe.py --out)
  probe-centered  same, minus the mean over the task's solutions
  confidence      low mean token log-probability of the response
  verbal-prob     the model's graded answer: P(no) to "Is this solution correct?"
  disagreement    how often the solution's outputs differ from its siblings' (disagree.py)
  combined        logistic regression on the signals above, trained with task-grouped CV on
                  ALL solutions (never on the evaluated task)
  verdict         the model's spoken verdict from judge_first (optional)
  random          expected value of random selection

All confidence intervals come from ONE bootstrap over tasks shared by all policies, so
"vs probe" differences are paired.

Example
-------
python rq3.py --runs ../results/qwen3b_mbpp:mbpp --probe ../results/probe_qwen3b_v3.json \
    --escalate spec=selftest_results_spec.jsonl both=selftest_results_spec.jsonl+selftest_results_spec_first.jsonl
"""
import argparse
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

FEATURES = {"probe": "probe: logistic", "probe-centered": "probe: logistic [task-centered]",
            "confidence": "baseline: low mean log-prob", "verbal-prob": "baseline: verbal P(incorrect)"}
COMBINED_INPUTS = ["probe", "verbal-prob", "confidence", "disagreement"]


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def combined_scores(table, inputs, seed):
    """Out-of-fold stacked score for every solution in `table` (dict key -> features, label, group)."""
    keys = list(table)
    cols = [c for c in inputs if any(table[k].get(c) is not None for k in keys)]
    X = np.array([[np.nan if table[k].get(c) is None else table[k][c] for c in cols] for k in keys], float)
    y = np.array([table[k]["buggy"] for k in keys], int)
    g = np.array([table[k]["group"] for k in keys])
    out = np.full(len(keys), np.nan)
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        mu = np.nanmean(X[tr], 0)
        xtr = np.where(np.isnan(X[tr]), mu, X[tr])
        xte = np.where(np.isnan(X[te]), mu, X[te])
        sd = xtr.std(0) + 1e-9
        clf = LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
        clf.fit((xtr - mu) / sd, y[tr])
        out[te] = clf.decision_function((xte - mu) / sd)
    return dict(zip(keys, out)), cols


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="RUN_DIR:dataset")
    ap.add_argument("--probe", required=True, help="JSON written by probe.py --out")
    ap.add_argument("--base", default="selftest_results.jsonl", help="own-test (code mode) results in each run")
    ap.add_argument("--escalate", nargs="+", default=["spec=selftest_results_spec.jsonl"])
    ap.add_argument("--disagree", default="disagree.jsonl", help="disagree.py output in each run (optional)")
    ap.add_argument("--verdicts", default=None, help="judge_first TESTS file in each run (optional)")
    ap.add_argument("--budgets", default="0.1,0.2,0.3,0.5,1.0")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    budgets = [float(b) for b in args.budgets.split(",")]
    rng = np.random.default_rng(args.seed)
    datasets = {spec.rsplit(":", 1)[1] for spec in args.runs}

    # ---- features for every solution (for the combined policy) ----------------------------------
    table = {}
    for o in json.load(open(args.probe))["oof"]:
        if o["dataset"] not in datasets:
            continue
        key = (o["dataset"], o["task_id"], o["sample_id"])
        table[key] = {"buggy": not o["correct"], "group": f'{o["dataset"]}::{o["task_id"]}',
                      **{p: o.get(f) for p, f in FEATURES.items()}}
    escal, verdicts = {}, {}
    for spec in args.runs:
        run_dir, dataset = spec.rsplit(":", 1)
        dpath = os.path.join(run_dir, args.disagree)
        if os.path.exists(dpath):
            for r in load_jsonl(dpath):
                k = (dataset, r["task_id"], r["sample_id"])
                if k in table:
                    table[k]["disagreement"] = r["disagree"]
        else:
            print(f"note: {dpath} missing (no disagreement scores for {dataset})")
        for e in args.escalate:
            name, files = e.split("=", 1)
            for fname in files.split("+"):
                path = os.path.join(run_dir, fname)
                if not os.path.exists(path):
                    print(f"note: {path} missing; '{name}' incomplete for {dataset}")
                    continue
                for r in load_jsonl(path):
                    k = (dataset, r["task_id"], r["sample_id"])
                    fails = r["n_asserts"] > 0 and not r["tests_all_pass"]
                    escal.setdefault(name, {})[k] = escal.get(name, {}).get(k, False) or fails
        if args.verdicts and os.path.exists(os.path.join(run_dir, args.verdicts)):
            for r in load_jsonl(os.path.join(run_dir, args.verdicts)):
                verdicts[(dataset, r["task_id"], r["sample_id"])] = r.get("verdict")
    probe_json = json.load(open(args.probe))
    if "train" in probe_json:  # transferred scores: a combined policy would train on the test benchmark
        comb = {k: None for k in table}
        print(f"transfer scores (probe trained on {probe_json['train']}): combined policy disabled")
    else:
        comb, comb_cols = combined_scores(table, COMBINED_INPUTS, args.seed)
        print(f"combined policy uses: {comb_cols}")

    # ---- the green set ---------------------------------------------------------------------------
    rows = []
    for spec in args.runs:
        run_dir, dataset = spec.rsplit(":", 1)
        for r in load_jsonl(os.path.join(run_dir, args.base)):
            k = (dataset, r["task_id"], r["sample_id"])
            if r["n_asserts"] == 0 or not r["tests_all_pass"] or k not in table:
                continue
            row = {"key": k, "group": table[k]["group"], "buggy": table[k]["buggy"], "combined": comb[k],
                   **{p: table[k].get(p) for p in list(FEATURES) + ["disagreement"]}}
            if verdicts:
                row["verdict"] = (1.0 if verdicts.get(k) == "buggy" else 0.0) if k in verdicts else None
            rows.append(row)
    y = np.array([r["buggy"] for r in rows])
    groups = np.array([r["group"] for r in rows])
    print(f"green solutions (own tests all pass): {len(rows)} = {y.sum()} missed bugs + {(~y).sum()} correct")

    policies = [p for p in ["probe", "probe-centered", "confidence", "verbal-prob", "disagreement", "combined",
                            "verdict"] if any(r.get(p) is not None for r in rows)]
    scores = {}
    for p in policies:
        s = np.array([np.nan if r.get(p) is None else r[p] for r in rows], float)
        if np.isnan(s).any():
            print(f"note: '{p}' missing for {int(np.isnan(s).sum())} solutions (ranked last)")
        scores[p] = np.where(np.isnan(s), -np.inf, s) + 1e-9 * rng.random(len(s))  # random tie-break

    tasks = np.unique(groups)
    members = {g: np.where(groups == g)[0] for g in tasks}
    draws = [np.concatenate([members[g] for g in rng.choice(tasks, len(tasks))]) for _ in range(args.n_boot)]

    def auc(idx, s):
        yy = y[idx]
        return roc_auc_score(yy, s[idx]) if 0 < yy.sum() < len(yy) else np.nan

    def ci(vals):
        vals = np.array([v for v in vals if not np.isnan(v)])
        return np.percentile(vals, [2.5, 97.5]) if len(vals) else (np.nan, np.nan)

    # ---- budget-free ranking quality on the green set -------------------------------------------
    print("\nranking the green set (missed bugs above correct), AUROC; paired difference vs probe")
    base_auc = {p: auc(np.arange(len(y)), scores[p]) for p in policies}
    boot_auc = {p: [auc(d, scores[p]) for d in draws] for p in policies}
    for p in policies:
        lo, hi = ci(boot_auc[p])
        line = f"  {p:15s} {base_auc[p]:.3f} [{lo:.3f}, {hi:.3f}]"
        if p != "probe":
            dlo, dhi = ci([a - b for a, b in zip(boot_auc[p], boot_auc["probe"])])
            line += f"   vs probe {base_auc[p] - base_auc['probe']:+.3f} [{dlo:+.3f}, {dhi:+.3f}]"
        print(line)

    # ---- escalation at fixed budgets ---------------------------------------------------------------
    for name, res in escal.items():
        flag = np.array([res.get(r["key"], False) for r in rows])
        cover = np.mean([r["key"] in res for r in rows])
        print(f"\n=== escalation '{name}': extra tests fail on {flag[y].sum()}/{y.sum()} missed bugs, "
              f"{flag[~y].sum()}/{(~y).sum()} correct (coverage {cover:.0%}) ===")
        print(f"{'budget':>6s} {'policy':15s}{'n':>5s}{'caught':>8s}{'false al.':>10s}{'prec.':>7s}"
              f"   {'vs random (95% CI)':24s} vs probe (95% CI)")

        def caught_at(idx, s, b):
            n = max(1, int(round(b * len(idx))))
            top = idx[np.argsort(-s[idx])[:n]]
            yb, fb = y[idx], flag[idx]
            chosen = np.isin(np.arange(len(y)), top)
            c = (chosen[idx] & yb & fb).sum() / max(yb.sum(), 1)
            fa = (chosen[idx] & ~yb & fb).sum() / max((~yb).sum(), 1)
            return n, c, fa, yb[chosen[idx]].mean(), c - b * (yb & fb).sum() / max(yb.sum(), 1)

        all_idx = np.arange(len(y))
        for b in budgets:
            rc = b * (y & flag).sum() / max(y.sum(), 1)
            rf = b * (~y & flag).sum() / max((~y).sum(), 1)
            print(f"{b:>6.0%} {'random (exp.)':15s}{int(round(b * len(y))):>5d}{rc:>8.3f}{rf:>10.3f}{y.mean():>7.3f}")
            if b >= 1.0:
                continue
            gains = {p: [caught_at(d, scores[p], b)[4] for d in draws] for p in policies}
            probe_gain = caught_at(all_idx, scores["probe"], b)[4]
            for p in policies:
                n, c, fa, prec, gain = caught_at(all_idx, scores[p], b)
                lo, hi = ci(gains[p])
                line = f"{'':6s} {p:15s}{n:>5d}{c:>8.3f}{fa:>10.3f}{prec:>7.3f}   {gain:+.3f} [{lo:+.3f}, {hi:+.3f}]  "
                if p != "probe":
                    dlo, dhi = ci([a - b2 for a, b2 in zip(gains[p], gains["probe"])])
                    line += f"  {gain - probe_gain:+.3f} [{dlo:+.3f}, {dhi:+.3f}]"
                print(line)
        print(f"  escalating everything: {int(y.sum() - (y & flag).sum())}/{int(y.sum())} bugs still shipped, "
              f"{int((~y & flag).sum())} false alarms, {len(y)} extra test runs")


if __name__ == "__main__":
    main()
