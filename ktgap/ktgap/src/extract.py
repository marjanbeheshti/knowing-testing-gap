"""E2, step 1: extract hidden states for saved solutions (no regeneration).

For every labeled solution, the model reads text again in one forward pass and we keep the hidden
state of every layer at a few token positions. Two framings (cf. Ribeiro et al. 2026):

  gen    the exact generation context: chat prompt + the model's own response
         positions: prompt_end (before any code; task-difficulty baseline),
                    code_end   (last token of the code), resp_end (last token of the response)
  judge  a new chat asking "Is this solution correct?"; position: judge_end (last prompt token)

We also store cheap baselines from the gen pass: mean and min token log-probability of the
response, and response length.

Output (resumable): <out-dir>/shard_00000.npz ... each with float16 arrays of shape
(n, n_layers + 1, hidden) per position, plus <out-dir>/meta_00000.jsonl. Finished shards are
skipped on restart, so an interrupted run loses at most one shard.

Example
-------
python extract.py --model Qwen/Qwen2.5-Coder-3B-Instruct --dataset mbpp \
    --solutions ../results/qwen3b_mbpp/solutions_labeled.jsonl --out-dir ../results/qwen3b_mbpp/hidden
"""
import argparse
import json
import os
import time

import numpy as np

from common import load_problems, read_jsonl

SYSTEM = "You are a careful Python programmer."
JUDGE_PROMPT = """Below is a programming task and a proposed solution.

Task:
```python
{prompt}
```

Solution:
```python
{solution}
```

Is this solution correct? Answer yes or no."""

POSITIONS = ["prompt_end", "code_end", "resp_end", "judge_end"]


def gen_inputs(tok, row):
    """Tokens of prompt+response and the token indices of prompt_end, code_end, resp_end."""
    prompt, raw = row["prompt_text"], row["raw"].rstrip()
    full = prompt + raw
    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    n_prompt_chars = len(prompt)

    def token_at(char_idx):  # index of the token that contains character char_idx
        for i, (s, e) in enumerate(offsets):
            if s <= char_idx < e:
                return i
        # fall back to the last token that starts at or before char_idx
        return max(i for i, (s, _) in enumerate(offsets) if s <= char_idx)

    prompt_end = token_at(n_prompt_chars - 1)
    resp_end = len(enc["input_ids"]) - 1
    code = row["solution"]
    pos = raw.find(code) if code else -1
    code_end = token_at(n_prompt_chars + pos + len(code) - 1) if pos >= 0 else resp_end
    return enc["input_ids"], {"prompt_end": prompt_end, "code_end": code_end, "resp_end": resp_end}


def judge_inputs(tok, row, task_prompt):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": JUDGE_PROMPT.format(prompt=task_prompt.strip(), solution=row["solution"])}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return ids, {"judge_end": len(ids) - 1}


def plan_shards(rows, shard_size):
    return [rows[i:i + shard_size] for i in range(0, len(rows), shard_size)]


def shard_paths(out_dir, i):
    return os.path.join(out_dir, f"shard_{i:05d}.npz"), os.path.join(out_dir, f"meta_{i:05d}.jsonl")


def forward_states(torch, model, seqs, positions_list, device, want_logprobs, n_prompt_list=None):
    """Run one padded batch; return per-sequence {pos: (L+1, H) float16} and optional logprob stats."""
    maxlen = max(len(s) for s in seqs)
    pad = model.config.pad_token_id if model.config.pad_token_id is not None else 0
    ids = torch.full((len(seqs), maxlen), pad, dtype=torch.long)
    mask = torch.zeros((len(seqs), maxlen), dtype=torch.long)
    for b, s in enumerate(seqs):  # right padding: positions stay valid
        ids[b, :len(s)] = torch.tensor(s)
        mask[b, :len(s)] = 1
    ids, mask = ids.to(device), mask.to(device)
    with torch.no_grad():
        out = model.model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        hs = torch.stack(out.hidden_states, dim=1)  # (B, L+1, T, H)
        results = []
        for b, positions in enumerate(positions_list):
            states = {p: hs[b, :, idx, :].float().cpu().numpy().astype(np.float16) for p, idx in positions.items()}
            stats = {}
            if want_logprobs:
                start, end = n_prompt_list[b], len(seqs[b])  # response tokens are [start, end)
                if end - start >= 1:
                    h = out.last_hidden_state[b, start - 1:end - 1]  # predicts tokens start..end-1
                    logp = torch.log_softmax(model.lm_head(h).float(), dim=-1)
                    tgt = ids[b, start:end]
                    tok_lp = logp.gather(1, tgt[:, None]).squeeze(1)
                    stats = {"mean_logprob": float(tok_lp.mean()), "min_logprob": float(tok_lp.min()),
                             "resp_tokens": int(end - start)}
            results.append((states, stats))
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", choices=["humaneval", "mbpp"], required=True)
    ap.add_argument("--solutions", required=True, help="labeled solutions written by THIS model")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shard-size", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="first N solutions only (smoke test)")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = read_jsonl(args.solutions)
    if "correct" not in rows[0]:
        raise SystemExit("needs a labeled solutions file")
    models = {r["solution_model"] for r in rows}
    if models != {args.model}:
        raise SystemExit(f"solutions were written by {models}, not {args.model}: hidden states must "
                         "come from the model that wrote the code")
    if args.limit:
        rows = rows[:args.limit]
    problems = load_problems(args.dataset)
    os.makedirs(args.out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device).eval()

    shards = plan_shards(rows, args.shard_size)
    t0, done_before = time.time(), 0
    for i, shard in enumerate(shards):
        npz_path, meta_path = shard_paths(args.out_dir, i)
        if os.path.exists(npz_path) and os.path.exists(meta_path):
            done_before += 1
            continue
        feats = {p: [] for p in POSITIONS}
        meta = []
        for j in range(0, len(shard), args.batch_size):
            batch = shard[j:j + args.batch_size]
            g = [gen_inputs(tok, r) for r in batch]
            n_prompt = [pos["prompt_end"] + 1 for _, pos in g]
            gen_out = forward_states(torch, model, [s for s, _ in g], [p for _, p in g], device, True, n_prompt)
            jd = [judge_inputs(tok, r, problems[r["task_id"]]["prompt"]) for r in batch]
            judge_out = forward_states(torch, model, [s for s, _ in jd], [p for _, p in jd], device, False)
            for r, (gs, stats), (js, _) in zip(batch, gen_out, judge_out):
                for p in ("prompt_end", "code_end", "resp_end"):
                    feats[p].append(gs[p])
                feats["judge_end"].append(js["judge_end"])
                meta.append({"task_id": r["task_id"], "sample_id": r["sample_id"], "dataset": args.dataset,
                             "correct": r["correct"], "code_chars": len(r["solution"]), **stats})
        tmp = npz_path + ".tmp.npz"
        np.savez(tmp, **{p: np.stack(v) for p, v in feats.items()})
        os.replace(tmp, npz_path)  # atomic: a shard is either complete or absent
        with open(meta_path, "w") as f:
            for m in meta:
                f.write(json.dumps(m) + "\n")
        done = i + 1 - done_before
        rate = (time.time() - t0) / max(done, 1)
        print(f"shard {i + 1}/{len(shards)} saved ({rate:.0f}s/shard, "
              f"~{rate * (len(shards) - i - 1) / 60:.0f} min left)", flush=True)
    print(f"done: {len(shards)} shards in {args.out_dir}")


if __name__ == "__main__":
    main()
