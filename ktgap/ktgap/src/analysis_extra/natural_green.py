"""Table V(c) natural green set, likelihood ratios (RQ1), and code+spec union (RQ1/G1). Run: python natural_green.py"""
import os, numpy as np
from _common import R, MODELS, L, ids, col, probe_rows, output_level_scores
DSN = {'he': 'humaneval', 'mbpp': 'mbpp'}
def vcaught(r): return any(a and b != 'pass' for a, b in zip(r['assert_valid'], r['assert_results']))
rng = np.random.default_rng(0)
for m in MODELS:
    oof, _ = probe_rows(m)
    pos = {(r['dataset'], r['task_id'], r['sample_id']): i for i, r in enumerate(oof)}
    yall = np.array([not r['correct'] for r in oof], int); gall = np.array([f"{r['dataset']}|{r['task_id']}" for r in oof])
    POL = {'probe': col(oof, 'probe: logistic'), 'verbal': col(oof, 'baseline: verbal P(incorrect)'),
           'prompt-only': col(oof, 'baseline: prompt-only probe (logistic)'), 'output-level': output_level_scores(oof, yall, gall)}
    sets = {'base (paper)': [], 'natural': []}; flag = np.zeros(len(oof), bool); lr = {}; un = {}
    for ds in ['mbpp', 'he']:
        d = os.path.join(R, f'{m}_{ds}'); D = DSN[ds]
        lab = {(r['task_id'], r['sample_id']): r['correct'] for r in L(d + '/solutions_labeled.jsonl')}
        base = ids(d + '/green_ids.jsonl'); mc = L(d + '/selftest_results_morecorrect.jsonl')
        nat = base | {(r['task_id'], r['sample_id']) for r in mc if r['tests_all_pass']}
        sets['base (paper)'] += [pos[(D,) + k] for k in base]; sets['natural'] += [pos[(D,) + k] for k in nat]
        for f in ['selftest_results_spec.jsonl', 'selftest_results_spec_first.jsonl']:
            for r in L(f'{d}/{f}'):
                if r['n_asserts'] > 0 and not r['tests_all_pass']: flag[pos[(D, r['task_id'], r['sample_id'])]] = True
        code = {(r['task_id'], r['sample_id']): r for r in L(d + '/selftest_results.jsonl') + mc if r['n_asserts'] > 0}
        spec = {(r['task_id'], r['sample_id']): r for r in L(d + '/selftest_results_spec.jsonl') if r['n_asserts'] > 0}
        cb = [k for k in code if not lab[k]]; cc = [k for k in code if lab[k]]
        pgc = np.mean([code[k]['tests_all_pass'] for k in cc]); pgb = np.mean([code[k]['tests_all_pass'] for k in cb])
        lr[ds] = (pgc, pgb, pgc / pgb, np.mean([lab[k] for k in nat]))
        kb = [k for k in cb if k in spec]; kc = [k for k in cc if k in spec]
        un[ds] = (100*np.mean([vcaught(code[k]) for k in kb]), 100*np.mean([vcaught(spec[k]) for k in kb]),
                  100*np.mean([vcaught(code[k]) or vcaught(spec[k]) for k in kb]),
                  100*np.mean([not code[k]['tests_all_pass'] for k in kc]), 100*np.mean([not spec[k]['tests_all_pass'] for k in kc]),
                  100*np.mean([(not code[k]['tests_all_pass']) or (not spec[k]['tests_all_pass']) for k in kc]))
    print(f"\n================ {m} ================")
    for sname, idx in sets.items():
        idx = np.array(sorted(set(idx))); y = yall[idx].astype(bool); fl = flag[idx]; g = gall[idx]
        tasks = np.unique(g); mem = {t: np.where(g == t)[0] for t in tasks}
        draws = [np.concatenate([mem[t] for t in rng.choice(tasks, len(tasks))]) for _ in range(1000)]
        def esc(sub, sc, b):
            n = max(1, int(round(b * len(sub)))); top = sub[np.argsort(-sc[idx][sub])[:n]]; yb = y[sub]
            c = (np.isin(sub, top) & yb & fl[sub]).sum() / max(yb.sum(), 1)
            fa = (np.isin(sub, top) & ~yb & fl[sub]).sum() / max((~yb).sum(), 1)
            return c, fa, c - b * (yb & fl[sub]).sum() / max(yb.sum(), 1)
        allsub = np.arange(len(idx))
        print(f"  [{sname}] green={len(idx)} ({y.sum()} missed, {(~y).sum()} correct; buggy share {y.mean():.0%})")
        for b in [0.1, 0.2, 0.3]:
            line = f"    {int(b*100)}%: random {100*b*(y & fl).sum()/y.sum():5.1f}"
            for p, sc in POL.items():
                if np.isnan(sc).all(): continue
                c, fa, _ = esc(allsub, sc, b); gs = [esc(ii, sc, b)[2] for ii in draws]; lo, _ = np.percentile(gs, [2.5, 97.5])
                line += f" | {p} {100*c:5.1f}{'*' if lo > 0 else ' '} (fa {100*fa:4.1f})"
            print(line)
    print("  LR P(green|correct)/P(green|buggy), all correct solutions with own tests; P(correct|green) on natural set:")
    for ds, (a, b_, r, post) in lr.items(): print(f"    {ds:4s}: {a:.3f} / {b_:.3f} = LR {r:5.2f}   P(correct|green) = {post:.3f}")
    print("  bugs caught (valid) code/spec/union || false alarms on correct code/spec/union:")
    for ds, v in un.items(): print(f"    {ds:4s}: {v[0]:5.1f} / {v[1]:5.1f} / {v[2]:5.1f}   ||   {v[3]:5.1f} / {v[4]:5.1f} / {v[5]:5.1f}")
