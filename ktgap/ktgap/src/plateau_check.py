
import os, json, numpy as np

R = '../results'

def L(p): return [json.loads(l) for l in open(p)]

for m in ['dscoder1b', 'qwen1_5b', 'qwen3b', 'qwen7b']:

    oof = json.load(open(f'{R}/{m}_analysis/probe_v2.json'))['oof']

    pos = {(r['dataset'], r['task_id'], r['sample_id']): i for i, r in enumerate(oof)}

    sc = np.array([r['probe: logistic'] for r in oof]); y = np.array([not r['correct'] for r in oof]); flag = np.zeros(len(oof), bool)

    sets = {'base': set(), 'natural': set()}

    for ds, D in [('mbpp', 'mbpp'), ('he', 'humaneval')]:

        d = f'{R}/{m}_{ds}'

        base = {(r['task_id'], r['sample_id']) for r in L(d + '/green_ids.jsonl')}

        nat = base | {(r['task_id'], r['sample_id']) for r in L(d + '/selftest_results_morecorrect.jsonl') if r['tests_all_pass']}

        sets['base'] |= {pos[(D,) + k] for k in base}; sets['natural'] |= {pos[(D,) + k] for k in nat}

        for f in ['selftest_results_spec.jsonl', 'selftest_results_spec_first.jsonl']:

            for r in L(f'{d}/{f}'):

                if r['n_asserts'] > 0 and not r['tests_all_pass']: flag[pos[(D, r['task_id'], r['sample_id'])]] = True

    for name, S in sets.items():

        idx = np.array(sorted(S)); order = idx[np.argsort(-sc[idx])]

        counts = [int((y[order[:int(round(b*len(idx)))]] & flag[order[:int(round(b*len(idx)))]]).sum()) for b in [.1, .15, .2, .25, .3]]

        print(f"{m:10s} {name:8s} n={len(idx):5d}  missed bugs caught at 10/15/20/25/30%: {counts}")

