
#!/bin/bash

# RQ2 control: same self-review prompt (judge_first) on 200 sampled MBPP+ buggy solutions that the model's own tests CAUGHT.

set -o pipefail

for spec in "deepseek-ai/deepseek-coder-1.3b-instruct dscoder1b 8" "Qwen/Qwen2.5-Coder-1.5B-Instruct qwen1_5b 8" "Qwen/Qwen2.5-Coder-3B-Instruct qwen3b 8" "Qwen/Qwen2.5-Coder-7B-Instruct qwen7b 4"; do

  set -- $spec; MODEL=$1; TAG=$2; B=$3; R=../results/${TAG}_mbpp

  python - "$R" <<'PY'

import json, random, sys

R = sys.argv[1]

green = {(r["task_id"], r["sample_id"]) for r in map(json.loads, open(R + "/green_ids.jsonl"))}

rows = [r for r in map(json.loads, open(R + "/selftest_results.jsonl")) if not r["correct"] and r["n_asserts"] > 0 and (r["task_id"], r["sample_id"]) not in green]

random.Random(0).shuffle(rows)

with open(R + "/caught_ids_sample.jsonl", "w") as f:

    for r in rows[:200]: f.write(json.dumps({"task_id": r["task_id"], "sample_id": r["sample_id"]}) + "\n")

print(R, "caught bugs available:", len(rows), "-> sampled", min(200, len(rows)))

PY

  OUT=$R/selftests_judge_first_caught.jsonl

  [ -e "$OUT" ] || python generate.py tests --model "$MODEL" --dataset mbpp --solutions $R/solutions_labeled.jsonl --mode judge_first --only-ids $R/caught_ids_sample.jsonl --batch-size $B --out $OUT || echo "FAILED $TAG"

  echo "DONE $TAG"

done

