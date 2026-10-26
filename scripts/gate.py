#!/usr/bin/env python3
"""Numerics gate: teacher-forced logit comparison, ferroseek vs serving stack.

For each prompt: the vLLM endpoint greedy-completes it; the full sequence
(prompt + completion) is then teacher-forced through BOTH engines and the
per-position logprobs/ranks/top-1 are compared. This is robust to the
documented f32-vs-bf16 intermediate differences, which can flip a free-
running argmax at a near-tie but barely move logprobs.

Usage (on a Spark with the checkpoint):

    python3 scripts/gate.py --model-dir /path/to/snapshot \
        --endpoint http://192.168.1.232:8000

Exit code 0 = every position within tolerance.
"""
import argparse
import json
import subprocess
import sys
import urllib.request

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "In 1969, the first",
]
import os
if os.environ.get("GATE_PROMPTS"):
    PROMPTS = os.environ["GATE_PROMPTS"].split("|||")
# Tolerances from the bisection (docs: agent-hjg0mqf7 report): rank-aware,
# because tail tokens are hypersensitive to f32<->bf16 transport while
# rank-1 tokens are robust. Near-ties (top1-top2 gap < 0.05) are flagged,
# not failed.
TOL_RANK1 = 0.1        # |dlogprob| for rank-1 consumed tokens
TOL_RANK100 = 0.5      # rank 2-100
TOL_TAIL = 3.0         # rank > 100
TOL_TOP1_LOGPROB = 0.25
NEAR_TIE = 0.05        # top1-top2 gap below this -> indeterminate, not a fail


def post(endpoint, path, body, timeout=300):
    req = urllib.request.Request(
        endpoint + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def vllm_completion(endpoint, model, prompt, max_tokens):
    r = post(endpoint, "/v1/completions", {
        "model": model, "prompt": prompt,
        "max_tokens": max_tokens, "temperature": 0})
    return r["choices"][0]["text"]


def vllm_prompt_logprobs(endpoint, model, text):
    r = post(endpoint, "/v1/completions", {
        "model": model, "prompt": text,
        "max_tokens": 1, "temperature": 0, "prompt_logprobs": 5})
    return r["choices"][0]["prompt_logprobs"]


def ferroseek_logprobs(ferroseek, model_dir, text):
    out = subprocess.run(
        [ferroseek, "decode", model_dir, text, "1", "--logits-out", "/tmp/fs-logits.json"],
        capture_output=True, text=True, timeout=7200)
    if out.returncode != 0:
        raise RuntimeError(f"ferroseek failed: {out.stderr[-400:]}")
    with open("/tmp/fs-logits.json") as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", default="deepseek-v4.1-flash")
    ap.add_argument("--ferroseek", default="./target/release/ferroseek")
    ap.add_argument("--max-tokens", type=int, default=6)
    ap.add_argument("--tokenizer", required=True, help="model dir for ferroseek tokens")
    args = ap.parse_args()

    failed = 0
    for prompt in PROMPTS:
        completion = vllm_completion(args.endpoint, args.model, prompt, args.max_tokens)
        full = prompt + completion

        # vLLM token ids for the full sequence, via ferroseek's tokenizer
        ids_out = subprocess.run(
            [args.ferroseek, "tokens", args.tokenizer, full],
            capture_output=True, text=True, check=True).stdout
        ids = json.loads(ids_out.splitlines()[0])

        ref = vllm_prompt_logprobs(args.endpoint, args.model, full)
        fs = {r["pos"]: r for r in ferroseek_logprobs(args.ferroseek, args.model_dir, full)}

        worst = 0.0
        top1_diff = 0
        near_ties = 0
        ok = True
        for pos in range(1, len(ids)):
            consumed = ids[pos]
            entry = (ref[pos] or {}).get(str(consumed))
            mine = fs.get(pos)
            if entry is None or mine is None:
                continue
            d = abs(entry["logprob"] - mine["logprob"])
            worst = max(worst, d)
            rank = entry.get("rank", 0)
            tol = TOL_RANK1 if rank <= 1 else (TOL_RANK100 if rank <= 100 else TOL_TAIL)
            ref_top1 = int(max(ref[pos], key=lambda k: ref[pos][k]["logprob"]))
            top1_lp_gap = abs(entry["logprob"] - mine["top1_logprob"]) if ref_top1 == mine["top1"] else 0
            top2 = sorted(ref[pos].values(), key=lambda v: -v["logprob"])
            gap = (top2[0]["logprob"] - top2[1]["logprob"]) if len(top2) > 1 else 1.0
            tied = gap < NEAR_TIE
            near_ties += tied
            if ref_top1 != mine["top1"] and not tied:
                top1_diff += 1
                ok = False
                print(f"  top1@{pos}: vllm={ref_top1} fs={mine['top1']} "
                      f"(consumed {consumed} rank {rank}, dlogprob {d:.4f})")
            elif d > tol and not tied:
                ok = False
                print(f"  dlogprob@{pos}: {d:.4f} > tol {tol} (rank {rank})")
            if ref_top1 == mine["top1"] and top1_lp_gap > TOL_TOP1_LOGPROB and not tied:
                ok = False
                print(f"  top1logprob@{pos}: gap {top1_lp_gap:.4f} > {TOL_TOP1_LOGPROB}")
        failed += 0 if ok else 1
        print(f"{'PASS' if ok else 'FAIL'} {prompt!r}: max|dlogprob|={worst:.4f}, "
              f"top1 mismatches={top1_diff}, near-ties={near_ties}, completion={completion!r}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
