"""RQ1 'Running the task's own example': (1) share of missed bugs failing the prompt's example (MBPP+ asserts,
HumanEval+ doctests); (2) RQ1 code-vs-spec drop restricted to bugs that pass the example; (3) Fig. 1 candidates.
Executes model-generated code in subprocesses with a timeout (run in a sandbox/VM). Requires evalplus.
Run: python prompt_example.py"""
import os, sys, json, subprocess, doctest, tempfile, collections
import numpy as np
from evalplus.data import get_human_eval_plus, get_mbpp_plus
from _common import R, MODELS, L, ids
PROMPTS = {'he': {k: v['prompt'] for k, v in get_human_eval_plus().items()},
           'mbpp': {k: v['prompt'] for k, v in get_mbpp_plus().items()}}
def examples(prompt, ds):
    if ds == 'mbpp':
        return [l.strip() for l in prompt.splitlines() if l.strip().startswith('assert ')]
    try:   # the last doctest's "want" would otherwise swallow the closing triple quotes
        return [(x.source.strip(), w) for x in doctest.DocTestParser().get_examples(prompt)
                for w in [x.want.split('"""')[0].split("'''")[0].strip()] if w]
    except ValueError:
        return []
RUNNER = r'''
import sys, json
code, tests, kind = json.loads(sys.stdin.read())
g = {}
try:
    exec(code, g)
except Exception:
    print("ERR"); sys.exit()
fails = 0
for t in tests:
    try:
        if kind == "mbpp":
            exec(t, g)
        else:
            src, want = t
            got = eval(src, g)
            try: ok = (got == eval(want, g))
            except Exception: ok = (repr(got).strip() == want)
            fails += (not ok)
    except Exception:
        fails += 1
print("FAIL" if fails else "PASS")
'''
WORK = tempfile.mkdtemp(prefix='ktgap_ex_')
def run(code, tests, kind):
    try:
        p = subprocess.run([sys.executable, '-c', RUNNER], input=json.dumps([code, tests, kind]),
                           capture_output=True, text=True, timeout=10, cwd=WORK)
        return p.stdout.strip().splitlines()[-1] if p.stdout.strip() else 'ERR'
    except subprocess.TimeoutExpired:
        return 'TIMEOUT'
def caught(r): return any(a and b != 'pass' for a, b in zip(r['assert_valid'], r['assert_results']))
CODEKEYS = ['solution', 'code', 'completion', 'program', 'response']
rng = np.random.default_rng(0)
for m in MODELS:
    tot = collections.Counter()
    for ds in ['mbpp', 'he']:
        d = os.path.join(R, f'{m}_{ds}')
        sols = {(r['task_id'], r['sample_id']): r for r in L(d + '/solutions_labeled.jsonl')}
        ck = next(k for k in CODEKEYS if k in next(iter(sols.values())))
        green = ids(d + '/green_ids.jsonl')
        code = {(r['task_id'], r['sample_id']): r for r in L(d + '/selftest_results.jsonl') if not r['correct'] and r['n_asserts'] > 0}
        spec = {(r['task_id'], r['sample_id']): r for r in L(d + '/selftest_results_spec.jsonl') if not r['correct'] and r['n_asserts'] > 0}
        keys = sorted(set(code) & set(spec))
        status = {}
        for k in set(keys) | green:
            ex = examples(PROMPTS[ds].get(k[0], ''), ds)
            status[k] = 'none' if not ex else run(sols[k][ck], ex, ds)
        # (1) missed bugs failing the example; sanity: green correct must pass (FAIL ~ 0)
        for grp, sel in [('missed', [k for k in green if not sols[k]['correct']]), ('green_correct', [k for k in green if sols[k]['correct']])]:
            c = collections.Counter(status[k] for k in sel)
            print(f"{m:10s} {ds:4s} {grp:13s} n={len(sel):4d} FAIL={c['FAIL']:4d} PASS={c['PASS']:4d} ERR={c['ERR']} TIMEOUT={c['TIMEOUT']} no_example={c['none']}")
            if grp == 'missed': tot['n'] += len(sel); tot['fail'] += c['FAIL']
        # (2) code-vs-spec drop on bugs that pass the example
        def diff(ks): return 100 * (np.mean([caught(spec[k]) for k in ks]) - np.mean([caught(code[k]) for k in ks]))
        keep = [k for k in keys if status[k] != 'FAIL']
        tasks = sorted({k[0] for k in keep}); by = {t: [k for k in keep if k[0] == t] for t in tasks}
        boots = [diff(sum((by[t] for t in rng.choice(tasks, len(tasks))), [])) for _ in range(1000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f"           drop all buggy: {diff(keys):5.1f} pts (n={len(keys)}) | passing the example: {diff(keep):5.1f} pts (n={len(keep)}) 95% CI [{lo:.1f}, {hi:.1f}]")
        # (3) Fig. 1 candidates
        if m == 'qwen3b' and ds == 'mbpp':
            cand = [k for k in keep if k in green and status[k] == 'PASS' and caught(spec[k])
                    and any((not a) and b == 'pass' for a, b in zip(code[k]['assert_valid'], code[k]['assert_results']))]
            print(f"           Fig. 1 candidates: {cand}")
    print(f"{m:10s} ALL missed bugs failing the prompt's example: {tot['fail']}/{tot['n']} ({100*tot['fail']/max(tot['n'],1):.1f}%)\n")
