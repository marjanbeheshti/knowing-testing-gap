"""Shared paths/helpers for the post-review analyses. Results live in ktgap/ktgap/results."""
import os, json
import numpy as np
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results')
MODELS = ['dscoder1b', 'qwen1_5b', 'qwen3b', 'qwen7b']
def L(p): return [json.loads(l) for l in open(p)]
def ids(p): return {(r['task_id'], r['sample_id']) for r in L(p)}
def col(rows, k): return np.array([np.nan if r.get(k) is None else r[k] for r in rows], float)
def probe_rows(m):
    d = os.path.join(R, f'{m}_analysis')
    f = next(p for p in [os.path.join(d, 'probe_v2.json'), os.path.join(d, 'probe.json')] if os.path.exists(p))
    return json.load(open(f))['oof'], os.path.basename(f)
BB = ['baseline: low mean log-prob', 'baseline: low min log-prob', 'baseline: code length',
      'baseline: verbal P(incorrect)', 'baseline: verbal P(bug)']
def output_level_scores(rows, y, g):
    """Out-of-fold logistic regression on output-level signals, grouped by task (same CV as the probe)."""
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import LogisticRegression
    X = np.column_stack([col(rows, c) for c in BB]); out = np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(X, y, g):
        mu = np.nanmean(X[tr], 0); xtr = np.where(np.isnan(X[tr]), mu, X[tr]); xte = np.where(np.isnan(X[te]), mu, X[te])
        a, s = xtr.mean(0), xtr.std(0) + 1e-9
        out[te] = LogisticRegression(max_iter=3000).fit((xtr - a) / s, y[tr]).decision_function((xte - a) / s)
    return out
