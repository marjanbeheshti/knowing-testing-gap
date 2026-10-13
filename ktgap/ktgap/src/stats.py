"""Cluster-robust statistics for the self-blind-spot experiments.

Solutions of the same task are not independent (spec-mode tests are shared by all solutions of a
task), so every test here resamples or permutes whole TASKS, never individual solutions.

Metrics (computed over buggy solutions only):
  caught      share of buggy solutions caught by at least one valid test
  confirm     share of wrong (invalid) assertions that pass on the buggy code ("bug-confirming")

Usage
-----
# E3: code vs spec for one tester (paired, same bugs)
python stats.py modes --code R/selftest_results.jsonl --spec R/selftest_results_spec.jsonl

# E1: 2x2 cross-model design on the same bugs (tester x mode)
python stats.py cross --self-code A --self-spec B --other-code C --other-spec D
"""
import argparse
import json
from collections import defaultdict

import numpy as np


def load_buggy(path):
    """{(task_id, sample_id): (caught, n_confirm, n_invalid)} for buggy solutions with tests."""
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r["correct"] or r["n_asserts"] == 0:
                continue
            v, x = r["assert_valid"], r["assert_results"]
            caught = any(a and b != "pass" for a, b in zip(v, x))
            confirm = sum((not a) and b == "pass" for a, b in zip(v, x))
            invalid = sum(not a for a in v)
            out[(r["task_id"], r["sample_id"])] = (int(caught), confirm, invalid)
    return out


def task_arrays(conds, keys):
    """Per-task sums for each condition: caught, n_solutions, confirm, invalid. Shape (T, C, 4)."""
    by_task = defaultdict(list)
    for k in keys:
        by_task[k[0]].append(k)
    tasks = sorted(by_task)
    arr = np.zeros((len(tasks), len(conds), 4))
    for i, t in enumerate(tasks):
        for j, d in enumerate(conds):
            for k in by_task[t]:
                c, conf, inv = d[k]
                arr[i, j] += (c, 1, conf, inv)
    return tasks, arr


def rates(arr):
    """Pooled rates per condition from (T, C, 4) sums -> caught (C,), confirm (C,)."""
    s = arr.sum(axis=0)
    caught = s[:, 0] / np.maximum(s[:, 1], 1)
    confirm = s[:, 2] / np.maximum(s[:, 3], 1)
    return caught, confirm


def bootstrap(arr, stat, n_boot, seed):
    rng = np.random.default_rng(seed)
    T = arr.shape[0]
    est = stat(arr)
    draws = np.array([stat(arr[rng.integers(0, T, T)]) for _ in range(n_boot)])
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    p = 2 * np.minimum((draws <= 0).mean(axis=0), (draws >= 0).mean(axis=0))
    return est, lo, hi, np.minimum(p, 1.0)


def signflip_p(per_task_diff, n_perm, seed):
    """Task-level paired permutation test: randomly flip the sign of each task's difference."""
    rng = np.random.default_rng(seed)
    obs = abs(per_task_diff.sum())
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(per_task_diff)))
    null = np.abs((signs * per_task_diff).sum(axis=1))
    return (1 + (null >= obs).sum()) / (n_perm + 1)


def fmt(name, est, lo, hi, p):
    return f"  {name:<44s} {est:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  bootstrap p = {p:.4f}"


def run_modes(args):
    code, spec = load_buggy(args.code), load_buggy(args.spec)
    keys = sorted(set(code) & set(spec))
    tasks, arr = task_arrays([code, spec], keys)
    caught, confirm = rates(arr)
    print(f"buggy solutions: {len(keys)} in {len(tasks)} tasks")
    print(f"  caught by a valid test: code {caught[0]:.3f} | spec {caught[1]:.3f}")
    print(f"  bug-confirming share of wrong asserts: code {confirm[0]:.3f} | spec {confirm[1]:.3f}")

    def stat(a):
        c, f = rates(a)
        return np.array([c[1] - c[0], f[0] - f[1]])

    est, lo, hi, p = bootstrap(arr, stat, args.n_boot, args.seed)
    print("\ntask-level bootstrap (positive = worse when the model sees its own code):")
    print(fmt("caught drop (spec - code)", est[0], lo[0], hi[0], p[0]))
    print(fmt("bug-confirming increase (code - spec)", est[1], lo[1], hi[1], p[1]))
    diff = arr[:, 1, 0] - arr[:, 0, 0]  # per-task: caught in spec minus caught in code
    print(f"task-level sign-flip permutation test, caught: p = {signflip_p(diff, args.n_perm, args.seed):.4f}")


def run_cross(args):
    conds = [load_buggy(p) for p in (args.self_code, args.self_spec, args.other_code, args.other_spec)]
    keys = sorted(set.intersection(*[set(d) for d in conds]))
    tasks, arr = task_arrays(conds, keys)
    caught, confirm = rates(arr)
    print(f"buggy solutions (written by the SELF model): {len(keys)} in {len(tasks)} tasks\n")
    print(f"{'':16s}{'code':>10s}{'spec':>10s}{'drop':>10s}   (caught by a valid test)")
    print(f"{'self tester':16s}{caught[0]:10.3f}{caught[1]:10.3f}{caught[1] - caught[0]:10.3f}")
    print(f"{'other tester':16s}{caught[2]:10.3f}{caught[3]:10.3f}{caught[3] - caught[2]:10.3f}")
    print(f"\n{'':16s}{'code':>10s}{'spec':>10s}{'diff':>10s}   (bug-confirming share of wrong asserts)")
    print(f"{'self tester':16s}{confirm[0]:10.3f}{confirm[1]:10.3f}{confirm[0] - confirm[1]:10.3f}")
    print(f"{'other tester':16s}{confirm[2]:10.3f}{confirm[3]:10.3f}{confirm[2] - confirm[3]:10.3f}")

    def stat(a):
        c, f = rates(a)
        drop_self, drop_other = c[1] - c[0], c[3] - c[2]
        conf_self, conf_other = f[0] - f[1], f[2] - f[3]
        return np.array([drop_self, drop_other, drop_self - drop_other,
                         conf_self, conf_other, conf_self - conf_other,
                         c[2] - c[0]])

    est, lo, hi, p = bootstrap(arr, stat, args.n_boot, args.seed)
    names = ["caught drop, self tester", "caught drop, other tester",
             "SELF-SPECIFIC caught drop (interaction)",
             "bug-confirming increase, self tester", "bug-confirming increase, other tester",
             "SELF-SPECIFIC confirming (interaction)",
             "other minus self, caught in code mode"]
    print("\ntask-level bootstrap:")
    for i, n in enumerate(names):
        print(fmt(n, est[i], lo[i], hi[i], p[i]))
    # permutation test for the interaction: per-task (self drop - other drop), in counts
    inter = (arr[:, 1, 0] - arr[:, 0, 0]) - (arr[:, 3, 0] - arr[:, 2, 0])
    print(f"task-level sign-flip permutation test, interaction (caught): "
          f"p = {signflip_p(inter, args.n_perm, args.seed):.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("modes")
    m.add_argument("--code", required=True)
    m.add_argument("--spec", required=True)
    c = sub.add_parser("cross")
    for a in ("--self-code", "--self-spec", "--other-code", "--other-spec"):
        c.add_argument(a, required=True)
    for p in (m, c):
        p.add_argument("--n-boot", type=int, default=10000)
        p.add_argument("--n-perm", type=int, default=10000)
        p.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run_modes(args) if args.cmd == "modes" else run_cross(args)


if __name__ == "__main__":
    main()
