"""Cross-benchmark transfer: train the probe on one benchmark, evaluate it on the other. CPU only.

Practical question: a team trains the probe once on labeled data, then uses it on NEW tasks.
Does it still work? (Orgad et al. 2025 found such probes often fail across datasets.)

The probe (position + layer selected on an inner split of the TRAINING benchmark only, then
logistic regression and mean-difference) is fit on all solutions of the training benchmark and
scores every solution of the test benchmark. Baselines need no training (verbal probability,
token confidence). Scores are saved in probe.py's JSON format, so rq3.py can use them directly:
  python rq3.py --runs ../results/qwen3b_mbpp:mbpp --probe ../results/transfer_he_to_mbpp.json ...

Example
-------
python transfer.py --runs ../results/qwen3b_full:humaneval ../results/qwen3b_mbpp:mbpp \
    --train humaneval --test mbpp --out ../results/transfer_he_to_mbpp.json
"""
import argparse
import json

import numpy as np

from probe import (BASELINE_POSITION, boot_auc, boot_auc_diff, center_features, load_runs, logistic_scores,
                   meandiff_scores, mixed_task_mask, select_config)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="RUN_DIR:dataset (both benchmarks)")
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--c", type=float, default=0.01)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--center-features", action="store_true",
                    help="within-task probe (see probe.py): centered features, trained on mixed tasks only")
    args = ap.parse_args()

    feats, meta = load_runs(args.runs)
    ds = np.array([m["dataset"] for m in meta])
    y = np.array([0 if m["correct"] else 1 for m in meta])
    groups = np.array([f'{m["dataset"]}::{m["task_id"]}' for m in meta])
    green = np.array([m.get("tests_green") is True for m in meta])
    tr, te = np.where(ds == args.train)[0], np.where(ds == args.test)[0]
    if args.center_features:
        center_features(feats, groups)
        tr = tr[mixed_task_mask(y, groups)[tr]]
        print(f"within-task probe: centered features; training on {len(tr)} solutions from mixed tasks")
    print(f"train: {args.train} ({len(tr)} solutions, {y[tr].sum()} buggy) -> "
          f"test: {args.test} ({len(te)} solutions, {y[te].sum()} buggy; "
          f"gap set {(green[te] & (y[te] == 1)).sum()} missed bugs vs {(green[te] & (y[te] == 0)).sum()} correct)")

    positions = [k for k in feats if k != BASELINE_POSITION]
    val_auc, pos, layer = select_config(feats, positions, y, groups, tr, args.seed)
    print(f"selected on {args.train} only: {pos} @ layer {layer} (validation AUROC {val_auc:.3f})")
    x = feats[pos][:, layer]
    scores = {"probe: logistic": logistic_scores(x[tr], y[tr], x[te], args.c),
              "probe: mean-diff": meandiff_scores(x[tr], y[tr], x[te])}
    _, ppos, player = select_config(feats, [BASELINE_POSITION], y, groups, tr, args.seed)
    scores["baseline: prompt-only probe"] = meandiff_scores(feats[ppos][tr, player], y[tr], feats[ppos][te, player])
    for key, sign, name in [("judge_pyes", -1, "baseline: verbal P(incorrect)"),
                            ("mean_logprob", -1, "baseline: low mean log-prob")]:
        v = np.array([meta[i].get(key, np.nan) for i in te], float)
        if not np.isnan(v).all():
            scores[name] = sign * v

    g_te, y_te, green_te = groups[te], y[te], green[te]
    for name in list(scores):  # task-centered versions (difficulty removed)
        s, c = scores[name], scores[name].copy()
        for g in np.unique(g_te):
            m = g_te == g
            c[m] = s[m] - np.nanmean(s[m])
        scores[name + " [task-centered]"] = c

    sets = {"ALL": np.ones(len(te), bool), "GAP": green_te}
    print(f"\nAUROC on {args.test} (never seen in training), 95% CI from a bootstrap over tasks")
    print(f"{'method':48s}{'ALL':>26s}{'GAP (missed vs test-passing correct)':>40s}")
    summary = {"train": args.train, "test": args.test, "selected": {"position": pos, "layer": int(layer)},
               "results": {}}
    for name, s in scores.items():
        row = f"{name:48s}"
        summary["results"][name] = {}
        for k, m in sets.items():
            ok = m & ~np.isnan(s)
            est, lo, hi, p = boot_auc(y_te[ok], s[ok], g_te[ok], args.n_boot, args.seed)
            row += f"{est:.3f} [{lo:.3f},{hi:.3f}]".rjust(26 if k == "ALL" else 40)
            summary["results"][name][k] = {"auroc": est, "ci": [lo, hi], "p_auc_le_0.5": p}
        print(row)
    if "baseline: verbal P(incorrect)" in scores:
        print("\npaired: probe (logistic) minus verbal P(incorrect)")
        for suffix in ["", " [task-centered]"]:
            for k, m in sets.items():
                est, lo, hi = boot_auc_diff(y_te[m], scores["probe: logistic" + suffix][m],
                                            scores["baseline: verbal P(incorrect)" + suffix][m], g_te[m],
                                            args.n_boot, args.seed)
                print(f"  {k:4s}{suffix or ' [raw]':18s} {est:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    if args.out:
        summary["oof"] = [{**{k: meta[i][k] for k in ("dataset", "task_id", "sample_id", "correct")},
                           "tests_green": meta[i].get("tests_green"),
                           **{n: (None if np.isnan(s[j]) else float(s[j])) for n, s in scores.items()}}
                          for j, i in enumerate(te)]
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=1, default=float)
        print(f"\nsaved {args.out} (usable as --probe for rq3.py on {args.test})")


if __name__ == "__main__":
    main()
