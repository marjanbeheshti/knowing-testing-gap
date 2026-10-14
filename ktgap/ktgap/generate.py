"""Stage 1 and 3 of the pipeline: generate solutions, then generate unit tests.

Examples
--------
# 1) sample 5 solutions per problem for 10 HumanEval+ problems
python src/generate.py solutions --model Qwen/Qwen2.5-Coder-0.5B-Instruct \
    --dataset humaneval --limit 10 --n 5 --out results/qwen0.5b/solutions.jsonl

# 3) have the same model write tests for each of its own solutions (white-box)
python src/generate.py tests --model Qwen/Qwen2.5-Coder-0.5B-Instruct \
    --dataset humaneval --solutions results/qwen0.5b/solutions_labeled.jsonl \
    --mode code --out results/qwen0.5b/selftests.jsonl
"""
import argparse
import json
import os
import time

from common import extract_asserts, extract_code, load_problems, read_jsonl, select_tasks, write_jsonl

SYSTEM = "You are a careful Python programmer."

SOLUTION_PROMPT = """Complete the following Python task. The function must be named `{entry}`.
Respond with the complete function (including imports and the signature) in a single ```python code block.

```python
{prompt}
```"""

TEST_PROMPT_CODE = """Below is a programming task and an implementation of it.

Task:
```python
{prompt}
```

Implementation:
```python
{solution}
```

Write {k} unit tests for `{entry}` as Python assert statements. Each test must check that a specific input produces the output the task description requires. Write each test as a plain top-level `assert` statement that calls `{entry}` directly (do not use unittest or pytest, and do not redefine `{entry}`). Respond with a single ```python code block that contains only the assert statements."""

TEST_PROMPT_SPEC = """Below is a programming task.

Task:
```python
{prompt}
```

Write {k} unit tests for `{entry}` as Python assert statements. Each test must check that a specific input produces the output the task description requires. Write each test as a plain top-level `assert` statement that calls `{entry}` directly (do not use unittest or pytest, and do not redefine `{entry}`). Respond with a single ```python code block that contains only the assert statements."""


_ASSERT_RULES = ("Write each test as a plain top-level `assert` statement that calls `{entry}` directly "
                 "(do not use unittest or pytest, and do not redefine `{entry}`). "
                 "Respond with a single ```python code block that contains only the assert statements.")

# RQ3 test-writing conditions. All show the implementation except `spec`.
TEST_PROMPT_WARN = """Below is a programming task and an implementation of it. Note: this implementation may contain a bug.

Task:
```python
{prompt}
```

Implementation:
```python
{solution}
```

Write {k} unit tests for `{entry}` that would reveal a bug if there is one. Each test must check that a specific input produces the output the task description requires. """ + _ASSERT_RULES

TEST_PROMPT_SPEC_FIRST = """Below is a programming task and an implementation of it.

Task:
```python
{prompt}
```

Implementation:
```python
{solution}
```

Write {k} unit tests for `{entry}`. For each test, first work out the expected output from the task description ALONE, ignoring what the implementation would return, and write that reasoning as a short comment; then write the assert. Use plain top-level `assert` statements that call `{entry}` directly (no unittest or pytest, do not redefine `{entry}`). Respond with a single ```python code block."""

FEWSHOT_EXAMPLES = """Here are two examples of good tests: each expected output comes from the task description, so the test catches the bug in the implementation.

Example 1
Task: Return the second-largest number in a list of at least two distinct numbers.
Implementation:
```python
def second_largest(nums):
    return max(nums)
```
Tests:
```python
assert second_largest([1, 5, 3]) == 3
assert second_largest([10, 2]) == 2
```

Example 2
Task: Count the vowels (a, e, i, o, u) in a lowercase string.
Implementation:
```python
def count_vowels(s):
    return sum(1 for c in s[1:] if c in "aeiou")
```
Tests:
```python
assert count_vowels("apple") == 2
assert count_vowels("echo") == 2
```

Now do the same for the following task.

"""

JUDGE_TURN = """Below is a programming task and an implementation of it.

Task:
```python
{prompt}
```

Implementation:
```python
{solution}
```

Review the implementation carefully against the task description, including edge cases. Is it correct? Explain briefly, then end with exactly one line: "Verdict: correct" or "Verdict: buggy"."""

JUDGE_FOLLOWUP = ("Now write {k} unit tests for `{entry}`. Each test must check that a specific input produces the "
                  "output the task description requires. " + _ASSERT_RULES)

PER_SOLUTION_MODES = ["code", "warn", "spec_first", "fewshot", "judge_first"]
ALL_MODES = PER_SOLUTION_MODES + ["spec"]


def build_test_prompt(mode, problem, solution, k):
    f = dict(prompt=problem["prompt"].strip(), solution=solution, entry=problem["entry_point"], k=k)
    if mode == "code":
        return TEST_PROMPT_CODE.format(**f)
    if mode == "warn":
        return TEST_PROMPT_WARN.format(**f)
    if mode == "spec_first":
        return TEST_PROMPT_SPEC_FIRST.format(**f)
    if mode == "fewshot":
        return FEWSHOT_EXAMPLES + TEST_PROMPT_CODE.format(**f)
    if mode == "spec":
        return TEST_PROMPT_SPEC.format(**f)
    raise ValueError(mode)


def parse_verdict(text):
    t = text.lower()
    i_b, i_c = t.rfind("verdict: buggy"), t.rfind("verdict: correct")
    if i_b < 0 and i_c < 0:
        return None
    return "buggy" if i_b > i_c else "correct"

# ---------------------------------------------------------------- backends

class HFBackend:
    """Hugging Face transformers. Works on CPU and on a single GPU (Colab, Jetstream2)."""

    def __init__(self, model_name, seed, batch_size):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        torch.manual_seed(seed)
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        if self.tok.chat_template is None:
            raise SystemExit("This model has no chat template; use an *-Instruct model.")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else (
            torch.float16 if device == "cuda" else torch.float32)
        # full precision weights (no quantization): hidden states must match the real model
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device)
        self.model.eval()
        self.device = device
        self.batch_size = batch_size

    def chat_text(self, user_msg, history=None):
        msgs = [{"role": "system", "content": SYSTEM}] + (history or []) + [{"role": "user", "content": user_msg}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def generate(self, prompts, n, temperature, top_p, max_new_tokens):
        outs = []
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i:i + self.batch_size]
            enc = self.tok(batch, return_tensors="pt", padding=True).to(self.device)
            kwargs = dict(max_new_tokens=max_new_tokens, num_return_sequences=n,
                          pad_token_id=self.tok.pad_token_id)
            if temperature > 0:
                kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
            else:
                kwargs.update(do_sample=False)
            with self.torch.no_grad():
                gen = self.model.generate(**enc, **kwargs)
            gen = gen[:, enc["input_ids"].shape[1]:]
            texts = self.tok.batch_decode(gen, skip_special_tokens=True)
            for j in range(len(batch)):
                outs.append(texts[j * n:(j + 1) * n])
            print(f"  generated {min(i + self.batch_size, len(prompts))}/{len(prompts)} prompts", flush=True)
        return outs


class VLLMBackend:
    """vLLM: much faster on a GPU (use on Jetstream2). Does not return hidden states."""

    def __init__(self, model_name, seed, batch_size):
        from vllm import LLM
        self.llm = LLM(model=model_name, dtype="bfloat16", seed=seed)
        self.tok = self.llm.get_tokenizer()
        self.seed = seed

    def chat_text(self, user_msg, history=None):
        msgs = [{"role": "system", "content": SYSTEM}] + (history or []) + [{"role": "user", "content": user_msg}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def generate(self, prompts, n, temperature, top_p, max_new_tokens):
        from vllm import SamplingParams
        params = SamplingParams(n=n, temperature=temperature, top_p=top_p if temperature > 0 else 1.0,
                                max_tokens=max_new_tokens, seed=self.seed)
        results = self.llm.generate(prompts, params)
        return [[o.text for o in r.outputs] for r in results]


def make_backend(args):
    cls = VLLMBackend if args.backend == "vllm" else HFBackend
    return cls(args.model, args.seed, args.batch_size)

# ---------------------------------------------------------------- stages

# ---------------------------------------------------------------- save-and-resume

def _partial(out):
    """Rows already saved by an interrupted run (OUT.partial), so a restart only does what is missing."""
    path = out + ".partial"
    return path, (read_jsonl(path) if os.path.exists(path) else [])


def _append(path, rows):
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _finish(out, path, rows, order_key):
    write_jsonl(out, sorted(rows, key=order_key))  # same order as an uninterrupted run
    os.remove(path)


# ---------------------------------------------------------------- stages

def run_solutions(args):
    problems = load_problems(args.dataset)
    task_ids = select_tasks(problems, args.limit, args.task_ids)
    part, rows = _partial(args.out)
    done = {r["task_id"] for r in rows}
    todo = [t for t in task_ids if t not in done]
    if done:
        print(f"resuming: {len(done)} tasks already saved in {part}, {len(todo)} to go")
    t0 = time.time()
    if todo:
        backend = make_backend(args)
        for i in range(0, len(todo), args.chunk):
            chunk = todo[i:i + args.chunk]
            prompts = [backend.chat_text(SOLUTION_PROMPT.format(prompt=problems[t]["prompt"].strip(),
                                                                entry=problems[t]["entry_point"])) for t in chunk]
            outs = backend.generate(prompts, args.n, args.temperature, args.top_p, args.max_new_tokens)
            new_rows = [{"task_id": t, "sample_id": k, "dataset": args.dataset, "solution_model": args.model,
                         "prompt_text": p, "raw": raw, "solution": extract_code(raw),
                         "temperature": args.temperature, "seed": args.seed}
                        for t, p, samples in zip(chunk, prompts, outs) for k, raw in enumerate(samples)]
            _append(part, new_rows)
            rows += new_rows
            print(f"saved {len(done) + i + len(chunk)}/{len(task_ids)} tasks", flush=True)
    order = {t: i for i, t in enumerate(task_ids)}
    _finish(args.out, part, rows, lambda r: (order[r["task_id"]], r["sample_id"]))
    print(f"wrote {len(rows)} solutions to {args.out} in {time.time() - t0:.0f}s")


def _tests_for(chunk, args, problems, backend):
    """Prompts, generation and rows for one chunk of test targets (both turns for judge_first)."""
    extra = {}
    if args.mode == "judge_first":
        # turn 1: the model reviews the code; turn 2: it writes tests in the same conversation
        judge_msgs = [JUDGE_TURN.format(prompt=problems[r["task_id"]]["prompt"].strip(), solution=r["solution"])
                      for r in chunk]
        judgments = [o[0] for o in backend.generate([backend.chat_text(m) for m in judge_msgs], 1,
                                                     args.temperature, args.top_p, 256)]
        prompts = [backend.chat_text(
            JUDGE_FOLLOWUP.format(entry=problems[r["task_id"]]["entry_point"], k=args.k),
            history=[{"role": "user", "content": m}, {"role": "assistant", "content": j}])
            for r, m, j in zip(chunk, judge_msgs, judgments)]
        extra = {i: {"judgment": j, "verdict": parse_verdict(j)} for i, j in enumerate(judgments)}
    elif args.mode == "spec":
        prompts = [backend.chat_text(build_test_prompt("spec", problems[r["task_id"]], None, args.k)) for r in chunk]
    else:
        prompts = [backend.chat_text(build_test_prompt(args.mode, problems[r["task_id"]], r["solution"], args.k))
                   for r in chunk]
    outs = backend.generate(prompts, 1, args.temperature, args.top_p, args.max_new_tokens)
    return [{"task_id": r["task_id"], "sample_id": r["sample_id"], "dataset": args.dataset,
             "solution_model": r.get("solution_model"), "test_model": args.model,
             "mode": args.mode, "prompt_text": p, "raw": o[0],
             "asserts": extract_asserts(o[0], max_asserts=args.k, entry_point=problems[r["task_id"]]["entry_point"]),
             **extra.get(i, {})}
            for i, (r, p, o) in enumerate(zip(chunk, prompts, outs))]


def run_tests(args):
    problems = load_problems(args.dataset)
    if args.mode in PER_SOLUTION_MODES:
        targets = read_jsonl(args.solutions)  # one test set per solution
        if args.only_buggy_and_n_correct is not None:
            targets = _subsample(targets, args.only_buggy_and_n_correct)
        if args.only_ids:  # e.g. only the solutions a probe flagged
            keep = {(d["task_id"], d["sample_id"]) for d in read_jsonl(args.only_ids)}
            targets = [r for r in targets if (r["task_id"], r["sample_id"]) in keep]
    else:  # spec: one test set per task (black-box), shared by all solutions of that task
        sols = read_jsonl(args.solutions)
        task_ids = sorted({r["task_id"] for r in sols}, key=list(problems).index)
        targets = [{"task_id": t, "sample_id": None, "solution_model": None} for t in task_ids]
    key = lambda r: (r["task_id"], r["sample_id"])
    part, rows = _partial(args.out)
    done = {key(r) for r in rows}
    todo = [r for r in targets if key(r) not in done]
    if done:
        print(f"resuming: {len(done)} test sets already saved in {part}, {len(todo)} to go")
    t0 = time.time()
    if todo:
        backend = make_backend(args)
        for i in range(0, len(todo), args.chunk):
            new_rows = _tests_for(todo[i:i + args.chunk], args, problems, backend)
            _append(part, new_rows)
            rows += new_rows
            print(f"saved {len(done) + i + len(new_rows)}/{len(targets)} test sets", flush=True)
    order = {key(r): i for i, r in enumerate(targets)}
    _finish(args.out, part, rows, lambda r: order[key(r)])
    empty = sum(1 for r in rows if not r["asserts"])
    print(f"wrote {len(rows)} test sets to {args.out} in {time.time() - t0:.0f}s "
          f"({empty} produced no parsable asserts)")
    if args.mode == "judge_first":
        v = [r.get("verdict") for r in rows]
        print(f"verdicts: {v.count('buggy')} buggy, {v.count('correct')} correct, {v.count(None)} unparsed")


def _subsample(rows, n_correct):
    """Keep every buggy solution plus up to n_correct correct ones per task (needs labels)."""
    if "correct" not in rows[0]:
        raise SystemExit("--only-buggy-and-n-correct needs a labeled solutions file (run evaluate.py label)")
    kept, seen = [], {}
    for r in rows:
        if not r["correct"]:
            kept.append(r)
        elif seen.get(r["task_id"], 0) < n_correct:
            kept.append(r)
            seen[r["task_id"]] = seen.get(r["task_id"], 0) + 1
    return kept


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["solutions", "tests"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", choices=["humaneval", "mbpp"], default="humaneval")
    ap.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="first N problems only")
    ap.add_argument("--task-ids", nargs="*", default=None)
    ap.add_argument("--n", type=int, default=5, help="solutions per problem")
    ap.add_argument("--k", type=int, default=5, help="asserts requested per test set")
    ap.add_argument("--solutions", help="(tests stage) solutions file, labeled or not")
    ap.add_argument("--mode", choices=ALL_MODES, default="code",
                    help="code = sees the solution; spec = task only; warn = told it may be buggy; "
                         "spec_first = derive outputs from the task; fewshot = examples of bug-catching tests; "
                         "judge_first = reviews the code, then writes tests")
    ap.add_argument("--only-ids", default=None,
                    help="(tests stage) jsonl with task_id/sample_id: only these solutions (e.g. probe-flagged)")
    ap.add_argument("--only-buggy-and-n-correct", type=int, default=None,
                    help="(tests stage) keep all buggy solutions + N correct ones per task")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=64,
                    help="prompts per saved chunk; an interrupted run resumes from the last saved chunk")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.temperature is None:  # diverse solutions (we need bugs); near-deterministic tests
        args.temperature = 0.8 if args.stage == "solutions" else 0.2
    if args.stage == "solutions":
        run_solutions(args)
    else:
        if not args.solutions:
            raise SystemExit("tests stage needs --solutions")
        run_tests(args)


if __name__ == "__main__":
    main()
