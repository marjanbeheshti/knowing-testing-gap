"""E2, step 2: train correctness probes and measure the knowing-testing gap. CPU only.

Protocol (following Ribeiro et al. 2026, "On the Robustness of LLMs' Internal Representation of
Code Correctness"):
  * probes are trained on the model's REAL mistakes (all labeled solutions), never on mutants
  * 5-fold cross-validation grouped by task, so no task appears in both train and test
  * inside each training fold, the read-out position and layer are selected on a held-out
    validation split (validation-selected, not best-layer, accuracy is reported)
  * mean-difference probe is primary; L2 logistic regression is secondary

Evaluation sets (scores are out-of-fold, so every solution is scored by a probe that never saw
its task):
  ALL     buggy vs correct, all solutions                        (replicates prior work)
  GAP     missed bugs vs correct solutions, both with ALL of the model's own tests passing
          -> the knowing-testing gap: AUROC > 0.5 means the model internally separates bugs
             its own tests could not
  CAUGHT  caught bugs vs the same test-passing correct solutions (for contrast)

Baselines: a probe on the prompt_end position (before any code is written: task difficulty
only), mean and min token log-probability of the response, and code length.

Task difficulty is a confound: hard tasks produce more bugs AND look different internally, so a
probe can "detect" bugs just by recognizing hard tasks. The PRIMARY metric therefore uses
task-centered scores: each solution's score minus the mean score of all solutions of the same
task (all 5 samples have hidden states). This removes everything shared by the task and keeps
only what distinguishes this solution from its siblings. The prompt-only probe is constant
within a task, so its task-centered AUROC is exactly 0.5 (a built-in sanity check).

Example
-------
python probe.py --runs ../results/qwen3b_full:humaneval ../results/qwen3b_mbpp:mbpp \
    --out ../results/probe_qwen3b.json
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

BASELINE_POSITION = "prompt_end"


# ---------------------------------------------------------------- data

def load_runs(specs):
    feats, meta = defaultdict(list), []
    for spec in specs:
        run_dir, dataset = spec.rsplit(":", 1)
        npzs = sorted(glob.glob(os.path.join(run_dir, "hidden", "shard_*.npz")))
        if not npzs:
            raise SystemExit(f"no hidden-state shards in {run_dir}/hidden (run extract.py first)")
        for npz in npzs:
            data = dict(np.load(npz))
            extra = npz.replace(os.sep + "hidden" + os.sep, os.sep + "hidden_extra" + os.sep)
            if os.path.exists(extra):  # extra judge framings extracted later, same shard layout
                more = np.load(extra)
                for k in more.files:
                    assert len(more[k]) == len(next(iter(data.values()))), f"row mismatch in {extra}"
                    data[k] = more[k]
            for k in data:
                feats[k].append(data[k])
            with open(npz.replace("shard_", "meta_").replace(".npz", ".jsonl")) as f:
                meta += [json.loads(l) for l in f]
        tests = {}
        res_path = os.path.join(run_dir, "selftest_results.jsonl")
        if os.path.exists(res_path):
            with open(res_path) as f:
                for line in f:
                    r = json.loads(line)
                    if r["n_asserts"] > 0:
                        tests[(r["task_id"], r["sample_id"])] = r["tests_all_pass"]
        for m in meta:
            if m["dataset"] == dataset and "tests_green" not in m:
                m["tests_green"] = tests.get((m["task_id"], m["sample_id"]))  # None = no own tests
    n_parts = max(len(v) for v in feats.values())
    missing = [k for k, v in feats.items() if len(v) != n_parts]
    if missing:
        print(f"note: positions {missing} are missing for some runs and will be ignored")
    feats = {k: np.concatenate(v) for k, v in feats.items() if len(v) == n_parts}
    return feats, meta


# ---------------------------------------------------------------- probes

def standardize(train, *others):
    mu = train.mean(0)
    sd = train.std(0) + 1e-6
    return [(x - mu) / sd for x in (train, *others)]


def meandiff_scores(xtr, ytr, xte):
    xtr, xte = standardize(xtr.astype(np.float32), xte.astype(np.float32))
    w = xtr[ytr == 1].mean(0) - xtr[ytr == 0].mean(0)
    return xte @ w


def logistic_scores(xtr, ytr, xte, c):
    xtr, xte = standardize(xtr.astype(np.float32), xte.astype(np.float32))
    clf = LogisticRegression(C=c, max_iter=3000)
    clf.fit(xtr, ytr)
    return clf.decision_function(xte)


def safe_auc(y, s):
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan


def select_config(feats, positions, y, groups, idx_train, seed):
    """Pick (position, layer) by mean-difference AUROC on a grouped validation split of the training fold."""
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    itr, iva = next(gss.split(idx_train, y[idx_train], groups[idx_train]))
    itr, iva = idx_train[itr], idx_train[iva]
    best = (-1.0, None, None)
    for pos in positions:
        for layer in range(feats[pos].shape[1]):
            s = meandiff_scores(feats[pos][itr, layer], y[itr], feats[pos][iva, layer])
            auc = safe_auc(y[iva], s)
            if auc > best[0]:
                best = (auc, pos, layer)
    return best


def cross_validated_scores(feats, positions, y, groups, n_folds, c, seed):
    oof = {"meandiff": np.full(len(y), np.nan), "logistic": np.full(len(y), np.nan)}
    chosen = []
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=n_folds).split(y, y, groups)):
        val_auc, pos, layer = select_config(feats, positions, y, groups, tr, seed + fold)
        x = feats[pos][:, layer]
        oof["meandiff"][te] = meandiff_scores(x[tr], y[tr], x[te])
        oof["logistic"][te] = logistic_scores(x[tr], y[tr], x[te], c)
        chosen.append({"fold": fold, "position": pos, "layer": int(layer), "val_auc": round(float(val_auc), 3)})
    return oof, chosen

# ---------------------------------------------------------------- evaluation

def boot_auc(y, s, groups, n_boot, seed):
    """AUROC with a 95% CI from a bootstrap over tasks."""
    est = safe_auc(y, s)
    uniq = np.unique(groups)
    members = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pick = np.concatenate([members[g] for g in rng.choice(uniq, len(uniq))])
        a = safe_auc(y[pick], s[pick])
        if not np.isnan(a):
            draws.append(a)
    lo, hi = np.percentile(draws, [2.5, 97.5]) if draws else (np.nan, np.nan)
    p_le_half = float(np.mean(np.array(draws) <= 0.5)) if draws else np.nan  # one-sided: AUROC > 0.5
    return est, lo, hi, p_le_half


def within_task_pairs(y, s, groups):
    """Share of (bug, correct) pairs from the SAME task where the bug gets the higher score."""
    wins, n = 0.0, 0
    for g in np.unique(groups):
        m = groups == g
        b, c = s[m & (y == 1)], s[m & (y == 0)]
        for sb in b:
            wins += (sb > c).sum() + 0.5 * (sb == c).sum()
            n += len(c)
    return (wins / n if n else np.nan), n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="RUN_DIR:dataset, e.g. ../results/qwen3b_mbpp:mbpp")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--c", type=float, default=0.01, help="logistic regression inverse regularization")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="save out-of-fold scores and summary as JSON")
    ap.add_argument("--positions", nargs="*", default=None,
                    help="probe read-out positions to select from (default: all except prompt_end); "
                         "e.g. --positions judgebug_end to evaluate one framing alone")
    args = ap.parse_args()

    feats, meta = load_runs(args.runs)
    y = np.array([0 if m["correct"] else 1 for m in meta])          # 1 = buggy
    groups = np.array([f'{m["dataset"]}::{m["task_id"]}' for m in meta])
    green = np.array([m.get("tests_green") is True for m in meta])
    has_tests = np.array([m.get("tests_green") is not None for m in meta])
    print(f"solutions: {len(y)} ({y.sum()} buggy) in {len(np.unique(groups))} tasks; "
          f"with own tests: {has_tests.sum()}")
    print(f"GAP set: {(green & (y == 1)).sum()} missed bugs vs {(green & (y == 0)).sum()} test-passing correct")

    positions = args.positions or [k for k in feats if k != BASELINE_POSITION]
    print(f"probe positions: {positions}")
    scores, chosen = {}, {}
    oof, chosen["probe"] = cross_validated_scores(feats, positions, y, groups, args.folds, args.c, args.seed)
    scores["probe: mean-diff"], scores["probe: logistic"] = oof["meandiff"], oof["logistic"]
    oof_b, chosen["prompt-only"] = cross_validated_scores(feats, [BASELINE_POSITION], y, groups, args.folds,
                                                          args.c, args.seed)
    scores["baseline: prompt-only probe"] = oof_b["meandiff"]
    for key, sign, name in [("mean_logprob", -1, "baseline: low mean log-prob"),
                            ("min_logprob", -1, "baseline: low min log-prob"),
                            ("code_chars", 1, "baseline: code length")]:
        scores[name] = sign * np.array([m.get(key, np.nan) for m in meta], dtype=float)

    # task-centered versions (primary for the gap): remove the task-level component of each score
    for name in ["probe: mean-diff", "probe: logistic", "baseline: prompt-only probe",
                 "baseline: low mean log-prob"]:
        s = scores[name]
        centered = s.copy()
        for g in np.unique(groups):
            m = groups == g
            centered[m] = s[m] - np.nanmean(s[m])
        scores[name + " [task-centered]"] = centered

    sets = {"ALL": np.ones(len(y), bool),
            "GAP": green,                                            # all own tests pass
            "CAUGHT": (has_tests & ~green & (y == 1)) | (green & (y == 0))}
    summary = {"chosen_configs": chosen, "results": {}}
    print("\nAUROC (higher = probe ranks buggy above correct), 95% CI from a bootstrap over tasks")
    print(f"{'method':48s}" + "".join(f"{k:>30s}" for k in sets))
    for name, s in scores.items():
        row = f"{name:48s}"
        summary["results"][name] = {}
        for k, m in sets.items():
            ok = m & ~np.isnan(s)
            est, lo, hi, p = boot_auc(y[ok], s[ok], groups[ok], args.n_boot, args.seed)
            row += f"{est:>10.3f} [{lo:.3f},{hi:.3f}]".rjust(30)
            summary["results"][name][k] = {"auroc": est, "ci": [lo, hi], "p_auc_le_0.5": p, "n": int(ok.sum())}
        print(row)
    pr, n_pairs = within_task_pairs(y[green], scores["probe: mean-diff"][green], groups[green])
    print(f"\nGAP within-task pairs (missed bug vs test-passing correct, same task): "
          f"probe ranks the bug higher in {pr:.3f} of {n_pairs} pairs")
    for name in ["probe: mean-diff [task-centered]", "probe: logistic [task-centered]"]:
        gap = summary["results"][name]["GAP"]
        print(f"KNOWING-TESTING GAP ({name}): AUROC {gap['auroc']:.3f} "
              f"[{gap['ci'][0]:.3f}, {gap['ci'][1]:.3f}], one-sided p(AUROC <= 0.5) = {gap['p_auc_le_0.5']:.4f}")
    print("\nselected (position, layer) per fold:")
    for k, v in chosen.items():
        print(f"  {k}: " + ", ".join(f"{c['position']}@{c['layer']} (val {c['val_auc']})" for c in v))
    if args.out:
        summary["within_task_pairs"] = {"share": pr, "n_pairs": n_pairs}
        summary["oof"] = [{**{k: m[k] for k in ("dataset", "task_id", "sample_id", "correct")},
                           "tests_green": m.get("tests_green"),
                           **{n: (None if np.isnan(s[i]) else float(s[i])) for n, s in scores.items()}}
                          for i, m in enumerate(meta)]
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=1, default=float)
        print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
