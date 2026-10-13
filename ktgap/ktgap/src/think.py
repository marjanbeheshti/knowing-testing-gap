"""E9: think, then probe. Does reasoning first make the model's doubt more visible?

For every labeled solution the model first TRACES its code (task example + one edge case), then
is asked for a final yes/no answer. We record, at the "Final answer:" token:
  * the hidden state of every layer      -> position think_end   (RUN/hidden_think/shard_*.npz)
  * P(yes) / (P(yes) + P(no))            -> think_pyes           (RUN/hidden_think/meta_think_*.jsonl)
plus the trace text. Shards have the same layout as RUN/hidden, so probe.py merges them.
Resumable: finished shards are skipped, so an interruption loses at most one shard.

Example
-------
python think.py --model Qwen/Qwen2.5-Coder-3B-Instruct --dataset mbpp \
    --solutions ../results/qwen3b_mbpp/solutions_labeled.jsonl --out-dir ../results/qwen3b_mbpp/hidden_think
"""
import argparse
import json
import os
import re
import time

import numpy as np

from common import load_problems, read_jsonl

SYSTEM = "You are a careful Python programmer."
THINK_PROMPT = """Below is a programming task and a proposed solution.

Task:
```python
{prompt}
```

Solution:
```python
{solution}
```

Check the solution by tracing it by hand on the example from the task and on one tricky edge case. Reply in exactly this format, with no other text:
Example: input = <input>; required = <output the task requires>; returned = <output the solution returns>
Edge case: input = <input>; required = <output the task requires>; returned = <output the solution returns>
Final answer: <yes if the solution is correct, no if it is not>"""

# The reply is started for the model, which keeps every trace short and in the same format.
TRACE_PREFIX = "Example: input ="

_FINAL = re.compile(r"final answer\b\s*:?", re.IGNORECASE)  # generation may stop before the colon


def build_prompt(tok, task_prompt, solution):
    """Chat prompt with the assistant reply already started with TRACE_PREFIX."""
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": THINK_PROMPT.format(prompt=task_prompt.strip(), solution=solution)}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + TRACE_PREFIX


def cut_trace(trace):
    """Reasoning text up to (not including) the model's own final answer, and that answer if given."""
    m = _FINAL.search(trace)
    if not m:
        return trace.rstrip(), None
    after = trace[m.end():].strip().lower()
    answer = "yes" if after.startswith("yes") else "no" if after.startswith("no") else None
    return trace[:m.start()].rstrip(), answer


def readout_text(prompt_text, trace):
    """The text whose last token is read: prompt (incl. TRACE_PREFIX) + reasoning + 'Final answer:'."""
    reasoning, _ = cut_trace(trace)
    return prompt_text + reasoning + "\nFinal answer:"


def generate_traces(model, tok, enc, max_new_tokens):
    """Greedy generation that stops as soon as the model writes 'Final answer'."""
    kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
    try:
        return model.generate(**enc, stop_strings=["Final answer"], tokenizer=tok, **kwargs)
    except (TypeError, ValueError):  # older transformers without stop_strings
        return model.generate(**enc, **kwargs)


def answer_token_ids(tok):
    """Single-token ids for ' yes' / ' no' style answers after 'Final answer:'."""
    def ids(words):
        out = sorted({e[0] for e in (tok.encode(w, add_special_tokens=False) for w in words) if len(e) == 1})
        if not out:
            raise SystemExit(f"none of {words} is a single token")
        return out
    return ids([" yes", " Yes", "yes", "Yes"]), ids([" no", " No", "no", "No"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", choices=["humaneval", "mbpp"], required=True)
    ap.add_argument("--solutions", required=True, help="labeled solutions written by THIS model")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shard-size", type=int, default=64, help="must match the hidden/ extraction (default 64)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None, help="first N solutions only (smoke test)")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = read_jsonl(args.solutions)
    if {r["solution_model"] for r in rows} != {args.model}:
        raise SystemExit("hidden states must come from the model that wrote the code")
    if args.limit:
        rows = rows[:args.limit]
    problems = load_problems(args.dataset)
    os.makedirs(args.out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device).eval()
    yes_ids, no_ids = answer_token_ids(tok)
    print(f"answer tokens: yes {yes_ids}, no {no_ids}")

    shards = [rows[i:i + args.shard_size] for i in range(0, len(rows), args.shard_size)]
    t0, done_now = time.time(), 0
    for si, shard in enumerate(shards):
        npz_path = os.path.join(args.out_dir, f"shard_{si:05d}.npz")
        meta_path = os.path.join(args.out_dir, f"meta_think_{si:05d}.jsonl")
        if os.path.exists(npz_path) and os.path.exists(meta_path):
            continue
        states, metas = [], []
        for j in range(0, len(shard), args.batch_size):
            batch = shard[j:j + args.batch_size]
            prompts = [build_prompt(tok, problems[r["task_id"]]["prompt"], r["solution"]) for r in batch]
            # 1) the model reasons (greedy, deterministic)
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            with torch.no_grad():
                gen = generate_traces(model, tok, enc, args.max_new_tokens)
            traces = [TRACE_PREFIX + t for t in
                      tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]
            # 2) read hidden states and P(yes) at "Final answer:"
            base = [p[:-len(TRACE_PREFIX)] for p in prompts]  # prompt without the prefix (the trace has it)
            seqs = [tok(readout_text(b_, tr), add_special_tokens=False)["input_ids"] for b_, tr in zip(base, traces)]
            maxlen = max(len(s) for s in seqs)
            ids = torch.full((len(seqs), maxlen), tok.pad_token_id, dtype=torch.long)
            mask = torch.zeros((len(seqs), maxlen), dtype=torch.long)
            for b, s in enumerate(seqs):  # right padding for the read-out pass
                ids[b, :len(s)] = torch.tensor(s)
                mask[b, :len(s)] = 1
            with torch.no_grad():
                out = model.model(input_ids=ids.to(device), attention_mask=mask.to(device), output_hidden_states=True)
                for b, s in enumerate(seqs):
                    last = len(s) - 1
                    states.append(torch.stack([h[b, last] for h in out.hidden_states]).float().cpu().numpy()
                                  .astype(np.float16))
                    probs = torch.softmax(model.lm_head(out.last_hidden_state[b, last]).float(), dim=-1)
                    py, pn = float(probs[yes_ids].sum()), float(probs[no_ids].sum())
                    _, own = cut_trace(traces[b])
                    complete = bool(_FINAL.search(traces[b]))
                    metas.append({"task_id": batch[b]["task_id"], "sample_id": batch[b]["sample_id"],
                                  "dataset": args.dataset, "think_pyes": py / max(py + pn, 1e-12),
                                  "think_answer_mass": py + pn,
                                  "think_answer": own or ("yes" if py >= pn else "no"),
                                  "trace_complete": complete, "trace": traces[b]})
        tmp = npz_path + ".tmp.npz"
        np.savez(tmp, think_end=np.stack(states))
        os.replace(tmp, npz_path)
        with open(meta_path + ".tmp", "w") as f:
            for m in metas:
                f.write(json.dumps(m) + "\n")
        os.replace(meta_path + ".tmp", meta_path)
        done_now += 1
        rate = (time.time() - t0) / done_now
        remaining = sum(1 for k in range(si + 1, len(shards))
                        if not os.path.exists(os.path.join(args.out_dir, f"shard_{k:05d}.npz")))
        n_complete = sum(m["trace_complete"] for m in metas)
        n_no = sum(m["think_answer"] == "no" for m in metas)
        print(f"shard {si + 1}/{len(shards)} saved ({rate:.0f}s/shard, ~{rate * remaining / 60:.0f} min left; "
              f"{n_complete}/{len(metas)} traces complete, {n_no} answered 'no')", flush=True)
    print(f"done: {len(shards)} shards in {args.out_dir}")


if __name__ == "__main__":
    main()
