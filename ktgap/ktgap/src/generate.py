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

Write {k} unit tests for `{entry}` as Python assert statements. Each test must check that a specific input produces the output the task description requires. Respond with a single ```python code block that contains only the assert statements."""

TEST_PROMPT_SPEC = """Below is a programming task.

Task:
```python
{prompt}
```

Write {k} unit tests for `{entry}` as Python assert statements. Each test must check that a specific input produces the output the task description requires. Respond with a single ```python code block that contains only the assert statements."""


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
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
        self.model.eval()
        self.device = device
        self.batch_size = batch_size

    def chat_text(self, user_msg):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_msg}]
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

    def chat_text(self, user_msg):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_msg}]
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

def run_solutions(args):
    problems = load_problems(args.dataset)
    task_ids = select_tasks(problems, args.limit, args.task_ids)
    backend = make_backend(args)
    prompts = [backend.chat_text(SOLUTION_PROMPT.format(prompt=problems[t]["prompt"].strip(),
                                                        entry=problems[t]["entry_point"]))
               for t in task_ids]
    t0 = time.time()
    outs = backend.generate(prompts, args.n, args.temperature, args.top_p, args.max_new_tokens)
    rows = []
    for t, prompt_text, samples in zip(task_ids, prompts, outs):
        for k, raw in enumerate(samples):
            rows.append({"task_id": t, "sample_id": k, "dataset": args.dataset,
                         "solution_model": args.model, "prompt_text": prompt_text,
                         "raw": raw, "solution": extract_code(raw),
                         "temperature": args.temperature, "seed": args.seed})
    write_jsonl(args.out, rows)
    print(f"wrote {len(rows)} solutions to {args.out} in {time.time() - t0:.0f}s")


def run_tests(args):
    problems = load_problems(args.dataset)
    backend = make_backend(args)
    if args.mode == "code":
        targets = read_jsonl(args.solutions)  # one test set per solution (white-box)
        if args.only_buggy_and_n_correct is not None:
            targets = _subsample(targets, args.only_buggy_and_n_correct)
        prompts = [backend.chat_text(TEST_PROMPT_CODE.format(
            prompt=problems[r["task_id"]]["prompt"].strip(), solution=r["solution"],
            entry=problems[r["task_id"]]["entry_point"], k=args.k)) for r in targets]
    else:  # spec: one test set per task (black-box), shared by all solutions of that task
        sols = read_jsonl(args.solutions)
        task_ids = sorted({r["task_id"] for r in sols}, key=list(problems).index)
        targets = [{"task_id": t, "sample_id": None, "solution_model": None} for t in task_ids]
        prompts = [backend.chat_text(TEST_PROMPT_SPEC.format(
            prompt=problems[t]["prompt"].strip(), entry=problems[t]["entry_point"], k=args.k))
            for t in task_ids]
    t0 = time.time()
    outs = backend.generate(prompts, 1, args.temperature, args.top_p, args.max_new_tokens)
    rows = []
    for r, prompt_text, samples in zip(targets, prompts, outs):
        raw = samples[0]
        rows.append({"task_id": r["task_id"], "sample_id": r["sample_id"], "dataset": args.dataset,
                     "solution_model": r.get("solution_model"), "test_model": args.model,
                     "mode": args.mode, "prompt_text": prompt_text, "raw": raw,
                     "asserts": extract_asserts(raw, max_asserts=args.k)})
    write_jsonl(args.out, rows)
    empty = sum(1 for r in rows if not r["asserts"])
    print(f"wrote {len(rows)} test sets to {args.out} in {time.time() - t0:.0f}s "
          f"({empty} produced no parsable asserts)")


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
    ap.add_argument("--mode", choices=["code", "spec"], default="code",
                    help="code = tester sees the solution (own-code tests); spec = docstring only")
    ap.add_argument("--only-buggy-and-n-correct", type=int, default=None,
                    help="(tests stage) keep all buggy solutions + N correct ones per task")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=4)
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
