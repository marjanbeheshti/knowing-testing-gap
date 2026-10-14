#!/usr/bin/env bash
# Run the whole frozen protocol (progress report, Section 6.1) for one model.
#
#   bash run_model.sh MODEL TAG [GEN_BATCH] [EXTRACT_BATCH]
#   e.g.  bash run_model.sh Qwen/Qwen2.5-Coder-7B-Instruct qwen7b 4 2
#
# Safe to re-run: finished steps are skipped and every generation step resumes where it stopped.
# Everything is logged to ../results/TAG_log.txt.
# Optional add-ons (off by default; costly on large models):
#   THINK=1       E9 think-then-probe          ALLCORRECT=1  own tests for ALL correct solutions
#   DRY=1         print the commands without running them
set -o pipefail
MODEL=${1:?model name}; TAG=${2:?tag}; GB=${3:-8}; EB=${4:-4}
RES=../results; LOG=$RES/${TAG}_log.txt; AN=$RES/${TAG}_analysis
mkdir -p "$AN"

say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
run() {  # run a command, append output to the log, stop the pipeline on failure
  say "RUN: $*"
  if [ "${DRY:-0}" = "1" ]; then return 0; fi
  "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" != "0" ]; then say "FAILED (exit $rc): $*"; say "fix the problem, then re-run the same command; finished steps are skipped"; exit "$rc"; fi
}
step() {  # step NAME OUTPUT_FILE CMD... : skip if OUTPUT_FILE already exists
  local name=$1 out=$2; shift 2
  if [ -e "$out" ]; then say "skip  $name ($out exists)"; else say "START $name"; run "$@"; fi
}

say "===== $TAG: $MODEL (generation batch $GB, extraction batch $EB) ====="
for PAIR in humaneval:he mbpp:mbpp; do
  DS=${PAIR%%:*}; R=$RES/${TAG}_${PAIR##*:}; mkdir -p "$R"
  step "$DS solutions" $R/solutions.jsonl \
    python generate.py solutions --model "$MODEL" --dataset $DS --n 5 --batch-size $GB --out $R/solutions.jsonl
  step "$DS labels" $R/solutions_labeled.jsonl \
    python evaluate.py label --dataset $DS --solutions $R/solutions.jsonl --out $R/solutions_labeled.jsonl
  step "$DS own tests (code mode)" $R/selftests.jsonl \
    python generate.py tests --model "$MODEL" --dataset $DS --solutions $R/solutions_labeled.jsonl --mode code \
      --only-buggy-and-n-correct 1 --batch-size $GB --out $R/selftests.jsonl
  step "$DS run own tests" $R/selftest_results.jsonl \
    python evaluate.py run-tests --dataset $DS --solutions $R/solutions_labeled.jsonl --tests $R/selftests.jsonl \
      --out $R/selftest_results.jsonl
  step "$DS spec tests" $R/selftests_spec.jsonl \
    python generate.py tests --model "$MODEL" --dataset $DS --solutions $R/solutions_labeled.jsonl --mode spec \
      --batch-size $GB --out $R/selftests_spec.jsonl
  step "$DS run spec tests" $R/selftest_results_spec.jsonl \
    python evaluate.py run-tests --dataset $DS --solutions $R/solutions_labeled.jsonl --tests $R/selftests_spec.jsonl \
      --out $R/selftest_results_spec.jsonl
  step "$DS test-passing ids" $R/green_ids.jsonl \
    python -c "
import json
rows=[json.loads(l) for l in open('$R/selftest_results.jsonl')]
g=[{'task_id':r['task_id'],'sample_id':r['sample_id']} for r in rows if r['n_asserts']>0 and r['tests_all_pass']]
open('$R/green_ids.jsonl','w').write('\n'.join(json.dumps(x) for x in g)+'\n'); print(len(g), 'test-passing solutions')"
  step "$DS spec_first tests (test-passing only)" $R/selftests_spec_first.jsonl \
    python generate.py tests --model "$MODEL" --dataset $DS --solutions $R/solutions_labeled.jsonl --mode spec_first \
      --only-ids $R/green_ids.jsonl --batch-size $GB --out $R/selftests_spec_first.jsonl
  step "$DS run spec_first tests" $R/selftest_results_spec_first.jsonl \
    python evaluate.py run-tests --dataset $DS --solutions $R/solutions_labeled.jsonl \
      --tests $R/selftests_spec_first.jsonl --out $R/selftest_results_spec_first.jsonl
  step "$DS hidden states" $R/hidden/DONE \
    bash -c "python extract.py --model '$MODEL' --dataset $DS --solutions $R/solutions_labeled.jsonl --framings gen,judge --batch-size $EB --out-dir $R/hidden && touch $R/hidden/DONE"
  step "$DS verbal yes/no" $R/hidden_yesno/DONE \
    bash -c "python extract.py --model '$MODEL' --dataset $DS --solutions $R/solutions_labeled.jsonl --framings judge,judgebug --yesno-only --batch-size $EB --out-dir $R/hidden_yesno && touch $R/hidden_yesno/DONE"
  step "$DS sibling disagreement" $R/disagree.jsonl \
    python disagree.py --dataset $DS --solutions $R/solutions_labeled.jsonl --tests $R/selftests_spec.jsonl --out $R/disagree.jsonl
  if [ "${THINK:-0}" = "1" ]; then
    step "$DS think-then-probe" $R/hidden_think/DONE \
      bash -c "python think.py --model '$MODEL' --dataset $DS --solutions $R/solutions_labeled.jsonl --batch-size $EB --out-dir $R/hidden_think && touch $R/hidden_think/DONE"
  fi
  if [ "${ALLCORRECT:-0}" = "1" ]; then
    step "$DS own tests, all correct" $R/selftests_all.jsonl \
      python generate.py tests --model "$MODEL" --dataset $DS --solutions $R/solutions_labeled.jsonl --mode code \
        --only-buggy-and-n-correct 99 --batch-size $GB --out $R/selftests_all.jsonl
    step "$DS run own tests, all correct" $R/selftest_results_all.jsonl \
      python evaluate.py run-tests --dataset $DS --solutions $R/solutions_labeled.jsonl --tests $R/selftests_all.jsonl \
        --out $R/selftest_results_all.jsonl
  fi
done

# ---- analysis (CPU only, always re-run; a few minutes) -------------------------------------------
HE=$RES/${TAG}_he; MB=$RES/${TAG}_mbpp
say "START analysis -> $AN"
run python stats.py modes --code $HE/selftest_results.jsonl --spec $HE/selftest_results_spec.jsonl
run python stats.py modes --code $MB/selftest_results.jsonl --spec $MB/selftest_results_spec.jsonl
run python probe.py --runs $HE:humaneval $MB:mbpp --out $AN/probe.json
run python transfer.py --runs $HE:humaneval $MB:mbpp --train humaneval --test mbpp --out $AN/transfer_he_to_mbpp.json
run python transfer.py --runs $HE:humaneval $MB:mbpp --train mbpp --test humaneval --out $AN/transfer_mbpp_to_he.json
run python rq3.py --runs $HE:humaneval $MB:mbpp --probe $AN/probe.json --budgets 0.1,0.2,0.3,1.0 \
  --escalate spec=selftest_results_spec.jsonl both=selftest_results_spec.jsonl+selftest_results_spec_first.jsonl
run python rq3.py --runs $MB:mbpp --probe $AN/transfer_he_to_mbpp.json --budgets 0.1,0.2,0.3,1.0 \
  --escalate both=selftest_results_spec.jsonl+selftest_results_spec_first.jsonl
if [ "${THINK:-0}" = "1" ]; then
  run python probe.py --runs $HE:humaneval $MB:mbpp --positions judge_end --out $AN/probe_before.json
  run python probe.py --runs $HE:humaneval $MB:mbpp --positions think_end --out $AN/probe_after.json
  run python compare.py --a $AN/probe_before.json --b $AN/probe_after.json
fi
say "===== DONE: $TAG. Full log: $LOG ====="
