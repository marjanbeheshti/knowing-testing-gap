
import json

for tag in ['dscoder1b', 'qwen1_5b', 'qwen3b', 'qwen7b']:

    v = [r.get('verdict') for r in map(json.loads, open(f'../results/{tag}_mbpp/selftests_judge_first_caught.jsonl'))]

    n, b, c, u = len(v), v.count('buggy'), v.count('correct'), v.count(None)

    print(f"{tag:10s} CAUGHT bugs n={n}  'correct' {c}  'buggy' {b}  no verdict {u}  flagged buggy: {b/max(n-u,1):.1%} of parsed")

