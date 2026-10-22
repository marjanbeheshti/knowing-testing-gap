#!/usr/bin/env bash
# RQ2 for more models: the model reviews its own green (test-passing) MBPP+ solutions and ends with
# "Verdict: correct" or "Verdict: buggy" (same judge_first prompt as for Qwen2.5-Coder-3B).
#   bash selfreview.sh MODEL TAG [BATCH]      e.g. bash selfreview.sh Qwen/Qwen2.5-Coder-1.5B-Instruct qwen1_5b 8
set -o pipefail
MODEL=${1:?model}; TAG=${2:?tag}; B=${3:-8}
R=../results/${TAG}_mbpp; OUT=$R/selftests_judge_first.jsonl
[ -e "$R/green_ids.jsonl" ] || { echo "missing $R/green_ids.jsonl (run run_model.sh first)"; exit 1; }
if [ ! -e "$OUT" ]; then
  python generate.py tests --model "$MODEL" --dataset mbpp --solutions $R/solutions_labeled.jsonl --mode judge_first \
    --only-ids $R/green_ids.jsonl --batch-size $B --out $OUT || exit 1
fi
python - "$R" "$TAG" <<'EOF' | tee ../results/${TAG}_analysis/selfreview.txt
import json, sys
run, tag = sys.argv[1], sys.argv[2]
lab = {(r["task_id"], r["sample_id"]): r["correct"] for r in map(json.loads, open(run + "/solutions_labeled.jsonl"))}
v = [(lab[(r["task_id"], r["sample_id"])], r.get("verdict")) for r in map(json.loads, open(run + "/selftests_judge_first.jsonl"))]
mb = [x for c, x in v if not c]; ok = [x for c, x in v if c]
def row(name, xs):
    n = len(xs); b = xs.count("buggy"); c = xs.count("correct"); u = xs.count(None)
    return f"{name:22s} n={n:4d}  'correct' {c:4d}  'buggy' {b:4d}  no verdict {u:4d}  flagged buggy: {b / max(n - u, 1):.1%} of parsed"
print(f"##### self-review, {tag}, green MBPP+ solutions")
print(row("missed bugs", mb))
print(row("green correct", ok))
EOF
