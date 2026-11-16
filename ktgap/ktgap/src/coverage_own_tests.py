"""Exploratory: do missed bugs come from own tests whose inputs explore too little?

For each buggy solution with code-visible own tests, take its VALID assertions and measure
  (1) line/branch coverage of the benchmark's reference solution, and
  (2) sibling-bug score: the share of the task's OTHER buggy behaviors these assertions catch.
Siblings that behave exactly like the solution on all EvalPlus inputs are excluded, otherwise
the comparison would be circular. Missed bugs are compared with caught bugs on the same task
(task bootstrap). "Pure" missed bugs = all own assertions valid (the 76% case).

Usage:  python3 coverage_own_tests.py qwen3b        (one model, to check first)
        python3 coverage_own_tests.py               (all four models)
"""
import json, os, sys, pickle, subprocess, tempfile
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from evalplus.data import get_human_eval_plus, get_mbpp_plus

MODELS = sys.argv[1:] or ['dscoder1b', 'qwen1_5b', 'qwen3b', 'qwen7b']
P = {'he': get_human_eval_plus(), 'mbpp': get_mbpp_plus()}
WORKERS = max(1, (os.cpu_count() or 2) - 1)
OUT = '../results/coverage_analysis'
os.makedirs(OUT, exist_ok=True)
CHILD = os.path.abspath(os.path.join(OUT, '_child.py'))

open(CHILD, 'w').write(r'''
import sys, pickle, json, signal, resource, hashlib, copy
resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
def _alarm(*a): raise TimeoutError()
signal.signal(signal.SIGALRM, _alarm)
job = pickle.loads(sys.stdin.buffer.read())
kind = job['kind']
if kind == 'cov':
    import coverage
    open('ref.py', 'w').write(job['code'])
    cov = coverage.Coverage(branch=True, data_file=None, include=['ref.py'])
    cov.start()
    g = {}
    try:
        exec(compile(job['code'], 'ref.py', 'exec'), g)
        for a in job['asserts']:
            try:
                signal.alarm(3); exec(a, g)
            except BaseException:
                pass
            finally:
                signal.alarm(0)
    finally:
        cov.stop()
    cov.json_report(outfile='cov.json', include=['ref.py'])
    t = json.load(open('cov.json'))['totals']
    print(json.dumps([t['covered_lines'] / max(t['num_statements'], 1),
                      t['covered_branches'] / t['num_branches'] if t['num_branches'] else None]))
elif kind == 'fp':
    g = {}
    exec(job['code'], g)
    f = g[job['entry']]
    outs = []
    for inp in job['inputs']:
        try:
            signal.alarm(2); outs.append(repr(f(*copy.deepcopy(inp))))
        except BaseException as e:
            outs.append('EXC:' + type(e).__name__)
        finally:
            signal.alarm(0)
    print(json.dumps(hashlib.sha1('\n'.join(outs).encode()).hexdigest()))
elif kind == 'run':
    g = {}
    try:
        exec(job['code'], g); ok = True
    except BaseException:
        ok = False
    res = []
    for a in job['asserts']:
        if not ok:
            res.append(False); continue
        try:
            signal.alarm(3); exec(a, g); res.append(True)
        except BaseException:
            res.append(False)
        finally:
            signal.alarm(0)
    print(json.dumps(res))
''')


def child(job, timeout):
    """Run one job in a separate process; return its last printed JSON line, or None."""
    with tempfile.TemporaryDirectory() as tmp:
        try:
            p = subprocess.run([sys.executable, CHILD], input=pickle.dumps(job),
                               capture_output=True, timeout=timeout, cwd=tmp)
            lines = p.stdout.decode(errors='replace').strip().splitlines()
            return json.loads(lines[-1]) if lines else None
        except Exception:
            return None


def L(path):
    return [json.loads(l) for l in open(path)]


def within_task(recs, idx, pure_only):
    """Per task: mean(missed) - mean(caught) of field idx; returns list of task differences."""
    by = {}
    for r in recs:
        if r[idx] is not None:
            by.setdefault(r[0], []).append(r)
    diffs = []
    for xs in by.values():
        a = [x[idx] for x in xs if x[1] and (x[2] or not pure_only)]
        b = [x[idx] for x in xs if not x[1]]
        if a and b:
            diffs.append(np.mean(a) - np.mean(b))
    return diffs


def ci(d, rng):
    if not d:
        return float('nan'), float('nan')
    boots = [np.mean(rng.choice(d, len(d))) for _ in range(2000)]
    return tuple(np.percentile(boots, [2.5, 97.5]))


for m in MODELS:
    recs, stats = [], dict(missed=0, missed_with_bug_assert=0, no_valid=0, mismatch=0)
    for ds in ['mbpp', 'he']:
        d = f'../results/{m}_{ds}'
        sols = {(r['task_id'], r['sample_id']): r for r in L(d + '/solutions_labeled.jsonl')}
        tests = {(r['task_id'], r['sample_id']): r['asserts']
                 for r in L(d + '/selftests.jsonl') if r.get('mode') == 'code'}
        res = [r for r in L(d + '/selftest_results.jsonl')
               if r.get('mode') == 'code' and not r['correct']]
        buggy = [k for k, s in sols.items() if not s['correct']]

        # Behavior fingerprint of every buggy solution on all EvalPlus inputs (cached).
        cache = f'{OUT}/{m}_{ds}_fingerprints.json'
        fp = json.load(open(cache)) if os.path.exists(cache) else {}
        todo = [k for k in buggy if f'{k[0]}|{k[1]}' not in fp]

        def do_fp(k):
            pr = P[ds][k[0]]
            return f'{k[0]}|{k[1]}', child({'kind': 'fp', 'code': sols[k]['solution'],
                                            'entry': pr['entry_point'],
                                            'inputs': pr['base_input'] + pr['plus_input']}, 300)
        with ThreadPoolExecutor(WORKERS) as ex:
            for key, h in ex.map(do_fp, todo):
                fp[key] = h
        json.dump(fp, open(cache, 'w'))

        jobs = []
        for r in res:
            k = (r['task_id'], r['sample_id'])
            if r['missed_bug']:
                stats['missed'] += 1
                stats['missed_with_bug_assert'] += not all(r['assert_valid'])
            if not (r['missed_bug'] or r['valid_test_catches_bug']):
                continue
            if k not in tests or len(tests[k]) != len(r['assert_valid']):
                stats['mismatch'] += 1; continue
            valid = [a for a, v in zip(tests[k], r['assert_valid']) if v]
            if not valid:
                stats['no_valid'] += 1; continue
            pure = r['missed_bug'] and all(r['assert_valid'])
            mine = fp.get(f'{k[0]}|{k[1]}')
            sibs = {}
            for k2 in buggy:
                if k2[0] != k[0] or k2 == k:
                    continue
                h = fp.get(f'{k2[0]}|{k2[1]}')
                if mine is None or h is None or h == mine:
                    continue
                sibs.setdefault(h, k2)          # one representative per distinct behavior
            jobs.append((k, r['missed_bug'], pure, valid, list(sibs.values())))

        def do_job(j):
            k, missed, pure, valid, sibs = j
            pr = P[ds][k[0]]
            c = child({'kind': 'cov', 'code': pr['prompt'] + pr['canonical_solution'],
                       'asserts': valid}, 60)
            caught = []
            for k2 in sibs:
                out = child({'kind': 'run', 'code': sols[k2]['solution'], 'asserts': valid}, 60)
                if out is not None:
                    caught.append(not all(out))
            return (ds + '|' + k[0], missed, pure,
                    c[0] if c else None, c[1] if c else None,
                    float(np.mean(caught)) if caught else None)
        with ThreadPoolExecutor(WORKERS) as ex:
            recs += list(ex.map(do_job, jobs))

    rng = np.random.default_rng(0)
    n_m = sum(r[1] for r in recs); n_p = sum(r[2] for r in recs); n_c = sum(not r[1] for r in recs)
    print(f"\n=== {m} ===")
    print(f"sanity: {stats['missed_with_bug_assert']} of {stats['missed']} missed bugs have an "
          f"own assertion encoding the bug (Qwen-3B in the paper: 33 of 135)")
    print(f"analysed: {n_m} missed ({n_p} pure) vs {n_c} caught; skipped: "
          f"{stats['no_valid']} no valid assertion, {stats['mismatch']} test/result mismatch")
    for name, idx in [('line coverage', 3), ('branch coverage', 4), ('sibling-bug score', 5)]:
        for label, pure_only in [('all missed', False), ('pure missed', True)]:
            dd = within_task(recs, idx, pure_only)
            lo, hi = ci(dd, rng)
            print(f"  {name:17s} {label:11s}: within-task missed - caught = "
                  f"{100*np.mean(dd) if dd else float('nan'):+.1f} pts "
                  f"[{100*lo:+.1f}, {100*hi:+.1f}]  ({len(dd)} tasks)")
