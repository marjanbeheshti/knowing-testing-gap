# Knowing–Testing Gap

Do LLMs internally "know" their code is wrong when their own unit tests pass it?

## Pipeline

| Stage | Command | Output |
|---|---|---|
| 1. Generate solutions | `python src/generate.py solutions ...` | `solutions.jsonl` |
| 2. Label correct / buggy (EvalPlus) | `python src/evaluate.py label ...` | `solutions_labeled.jsonl` |
| 3. Model writes tests for its own solutions | `python src/generate.py tests --mode code ...` | `selftests.jsonl` |
| 4. Run tests, measure missed bugs | `python src/evaluate.py run-tests ...` | `selftest_results.jsonl` + summary |
| 5. Hidden-state extraction | `src/extract.py` (next) | |
| 6. Probes + gap analysis | `src/probe.py`, `src/analyze.py` (next) | |

## Quick start (10 problems, 0.5B model)

```bash
pip install -r requirements.txt
M=Qwen/Qwen2.5-Coder-0.5B-Instruct
D=results/qwen0.5b
python src/generate.py solutions --model $M --dataset humaneval --limit 10 --n 5 --out $D/solutions.jsonl
python src/evaluate.py label --dataset humaneval --solutions $D/solutions.jsonl --out $D/solutions_labeled.jsonl
python src/generate.py tests --model $M --dataset humaneval --solutions $D/solutions_labeled.jsonl --mode code --out $D/selftests.jsonl
python src/evaluate.py run-tests --dataset humaneval --solutions $D/solutions_labeled.jsonl --tests $D/selftests.jsonl --out $D/selftest_results.jsonl
```

On a GPU with vLLM installed, add `--backend vllm` to the generate commands.

## Key definitions

- **correct**: passes both the original HumanEval/MBPP tests and the EvalPlus extra tests.
- **valid assert**: passes on the reference solution (its expected value is right).
- **missed bug**: a buggy solution whose generated tests all pass ("tests are green").
- **false alarm**: a correct solution with at least one failing generated test.
- **mode `code`**: the tester sees the solution (white-box, tests for its own code).
  **mode `spec`**: the tester sees only the task description (black-box baseline).
- **cross-model control**: run `generate.py tests` with a different `--model` on the same solutions file.

## Safety

Model-generated code is executed. Run only on Colab or a Jetstream2 VM, never on a personal machine.
