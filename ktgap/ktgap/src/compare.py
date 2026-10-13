"""Paired comparison of two probe.py runs on the same solutions (e.g. before vs after reasoning).

python compare.py --a ../results/probe_judge.json --b ../results/probe_think.json \
    --score "probe: logistic [task-centered]"
Positive difference = run B ranks bugs above correct better than run A. CI: bootstrap over tasks.
"""
import argparse
import json

import numpy as np

from probe import boot_auc, boot_auc_diff


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--score", nargs="+", default=["probe: logistic [task-centered]", "probe: logistic"])
    ap.add_argument("--b-score", nargs="+", default=None,
                    help="score names to take from B (same length as --score); default: same names. "
                         "E.g. compare verbal answers before vs after reasoning within one file.")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    A = {(o["dataset"], o["task_id"], o["sample_id"]): o for o in json.load(open(args.a))["oof"]}
    B = {(o["dataset"], o["task_id"], o["sample_id"]): o for o in json.load(open(args.b))["oof"]}
    keys = sorted(set(A) & set(B))
    y = np.array([0 if A[k]["correct"] else 1 for k in keys])
    groups = np.array([f"{k[0]}::{k[1]}" for k in keys])
    sets = {"ALL": np.ones(len(keys), bool), "GAP": np.array([A[k].get("tests_green") is True for k in keys])}
    print(f"{len(keys)} solutions in both runs; GAP: {(sets['GAP'] & (y == 1)).sum()} missed bugs vs "
          f"{(sets['GAP'] & (y == 0)).sum()} test-passing correct")
    b_names = args.b_score or args.score
    if len(b_names) != len(args.score):
        raise SystemExit("--b-score needs as many names as --score")
    for name, bname in zip(args.score, b_names):
        sa = np.array([np.nan if A[k].get(name) is None else A[k][name] for k in keys], float)
        sb = np.array([np.nan if B[k].get(bname) is None else B[k][bname] for k in keys], float)
        if bname != name:
            print(f"A = '{name}'   B = '{bname}'")
        for set_name, m in sets.items():
            ea = boot_auc(y[m], sa[m], groups[m], args.n_boot, args.seed)[:3]
            eb = boot_auc(y[m], sb[m], groups[m], args.n_boot, args.seed)[:3]
            d, lo, hi = boot_auc_diff(y[m], sb[m], sa[m], groups[m], args.n_boot, args.seed)
            print(f"{name:34s} {set_name:4s} A {ea[0]:.3f} [{ea[1]:.3f},{ea[2]:.3f}]  "
                  f"B {eb[0]:.3f} [{eb[1]:.3f},{eb[2]:.3f}]  B-A {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    main()
