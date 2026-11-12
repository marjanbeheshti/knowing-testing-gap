"""Table II rows marked dagger: probe vs prompt-only probe, output-level classifier, verbal confidence.
Paired task bootstrap (2000 draws) on ALL solutions and on the green set (GAP). Run: python paired_test.py"""
import numpy as np
from sklearn.metrics import roc_auc_score
from _common import MODELS, probe_rows, col, output_level_scores
for m in MODELS:
    rng = np.random.default_rng(0)
    rows, src = probe_rows(m)
    y = np.array([not r['correct'] for r in rows], int)
    g = np.array([f"{r['dataset']}|{r['task_id']}" for r in rows])
    green = np.array([bool(r['tests_green']) for r in rows])
    S = {'probe': col(rows, 'probe: logistic'),
         'prompt-only': col(rows, 'baseline: prompt-only probe (logistic)'),
         'verbal': col(rows, 'baseline: verbal P(incorrect)'),
         'output-level': output_level_scores(rows, y, g)}
    print(f"\n======== {m} ({src}) ========")
    for name, mask in [('ALL', np.ones(len(y), bool)), ('GAP', green)]:
        idx = np.where(mask)[0]; tasks = np.unique(g[idx]); mem = {t: idx[g[idx] == t] for t in tasks}
        draws = [np.concatenate([mem[t] for t in rng.choice(tasks, len(tasks))]) for _ in range(2000)]
        auc = lambda s, ii: roc_auc_score(y[ii], s[ii])
        print(f"  {name} (n={len(idx)}): " + "  ".join(f"{k}={auc(v, idx):.3f}" for k, v in S.items() if not np.isnan(v).all()))
        for a_, b_ in [('probe', 'prompt-only'), ('probe', 'output-level'), ('probe', 'verbal'), ('output-level', 'verbal')]:
            if np.isnan(S[b_]).all(): continue
            d = [auc(S[a_], ii) - auc(S[b_], ii) for ii in draws if 0 < y[ii].sum() < len(ii)]
            lo, hi = np.percentile(d, [2.5, 97.5]); pt = auc(S[a_], idx) - auc(S[b_], idx)
            print(f"      {a_} - {b_:13s} {pt:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  {'SIGNIFICANT' if lo > 0 or hi < 0 else 'n.s.'}")
