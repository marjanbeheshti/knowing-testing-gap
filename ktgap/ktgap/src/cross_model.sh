#!/usr/bin/env bash
# E10: cross-model monitor. A READER model reads the solutions written by an AUTHOR model and answers
# "Is this solution correct?"; we probe the reader's hidden states and compare with the author's own
# probe at the same read-out (judge_end). Pre-specified primary test: same-task pair accuracy.
#
#   bash cross_model.sh READER_MODEL READER_TAG AUTHOR_TAG [EXTRACT_BATCH]
#   e.g. bash cross_model.sh Qwen/Qwen2.5-Coder-3B-Instruct qwen3b qwen7b 4
#
# Resumable: finished extraction shards are skipped. Results go to ../results/AUTHOR_TAG_analysis/.
set -o pipefail
READER=${1:?reader model}; RT=${2:?reader tag}; AT=${3:?author tag}; EB=${4:-4}
RES=../results; HE=$RES/${AT}_he; MB=$RES/${AT}_mbpp; AN=$RES/${AT}_analysis; HD=hidden_reader_$RT
LOG=$RES/${AT}_reader_${RT}_log.txt
mkdir -p "$AN"
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
save() {  # save FILE CMD...
  local out=$1; shift
  say "RUN: $* > $out"
  "$@" 2>&1 | tee "$out" | tee -a "$LOG" > /dev/null
  local rc=${PIPESTATUS[0]}
  if [ "$rc" != "0" ]; then say "FAILED (exit $rc): see $out"; exit "$rc"; fi
}

say "===== cross-model monitor: $READER reads $AT ====="
for PAIR in humaneval:$HE mbpp:$MB; do
  DS=${PAIR%%:*}; R=${PAIR#*:}
  if [ -e "$R/$HD/DONE" ]; then say "skip  $DS extraction (done)"; continue; fi
  save "$R/${HD}_extract.txt" python extract.py --model "$READER" --dataset $DS --solutions $R/solutions_labeled.jsonl \
    --framings judge,judgebug --allow-other-model --batch-size $EB --out-dir $R/$HD
  touch "$R/$HD/DONE"
done

save $AN/probe_reader_$RT.txt python probe.py --runs $HE:humaneval $MB:mbpp --hidden-dir $HD --out $AN/probe_reader_$RT.json
save $AN/probe_own_judge.txt python probe.py --runs $HE:humaneval $MB:mbpp --positions judge_end --out $AN/probe_own_judge.json
save $AN/compare_reader_$RT.txt python compare.py --a $AN/probe_own_judge.json --b $AN/probe_reader_$RT.json
save $AN/diagnose_reader_$RT.txt python diagnose.py --runs $HE:humaneval $MB:mbpp --probe $AN/probe_reader_$RT.json \
  --extra selftest_results_morecorrect.jsonl
save $AN/diagnose_own_judge.txt python diagnose.py --runs $HE:humaneval $MB:mbpp --probe $AN/probe_own_judge.json \
  --extra selftest_results_morecorrect.jsonl
save $AN/rq3_reader_$RT.txt python rq3.py --runs $HE:humaneval $MB:mbpp --probe $AN/probe_reader_$RT.json \
  --budgets 0.1,0.2,0.3,1.0 --escalate both=selftest_results_spec.jsonl+selftest_results_spec_first.jsonl

{ echo "##### $AT read by $RT (B) vs its own judge_end probe (A)"
  echo "== same-task pairs (PRIMARY), own probe:";    sed -n '/A2/,/NO test/p' $AN/diagnose_own_judge.txt | sed -n 2p
  echo "== same-task pairs (PRIMARY), reader probe:"; sed -n '/A2/,/NO test/p' $AN/diagnose_reader_$RT.txt | sed -n 2p
  echo "== paired AUROC, own (A) vs reader (B):";     grep -E "probe: logistic" $AN/compare_reader_$RT.txt
  echo "== reader probe, ranking the green set:";     sed -n '/ranking the green/,/^$/p' $AN/rq3_reader_$RT.txt
  echo "== reader probe, 30% budget:";                sed -n '/30% random/,/100% random/p' $AN/rq3_reader_$RT.txt
} > $AN/SUMMARY_reader_$RT.txt
cat $AN/SUMMARY_reader_$RT.txt | tee -a "$LOG"
say "===== DONE: $AT read by $RT ====="
