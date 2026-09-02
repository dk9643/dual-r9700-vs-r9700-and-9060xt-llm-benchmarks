#!/usr/bin/env python3
"""
Ollama LLM benchmark for before/after GPU upgrade comparisons.

Usage:
    python3 ollama_bench.py <model_name> [--runs N] [--label LABEL]

Examples:
    python3 ollama_bench.py llama3.1:8b
    python3 ollama_bench.py qwen2.5:32b --runs 5 --label "r9700+9060xt"
    python3 ollama_bench.py qwen2.5:32b --runs 5 --label "dual-r9700"

Results are appended to ollama_bench_results.json in the current directory
so you can diff runs across hardware configs.

Requires: Ollama running locally (default http://localhost:11434).
No external Python packages needed (stdlib only).
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.request
import uuid
from datetime import datetime, timezone
from statistics import mean, stdev

OLLAMA_URL = "http://localhost:11434"
RESULTS_FILE = "ollama_bench_results.json"

# Prompt sizes: short chat, medium reasoning, then progressively larger
# long-context ingestion tests (~2.5k, ~8k, ~16k tokens). Long prompts stress
# prompt-processing (compute bound, where dual GPUs and PCIe x8 links matter
# most); generation speed is mostly VRAM-bandwidth bound. The KV cache for the
# 8k/16k tests also eats real VRAM, which is exactly what the upgrade adds.
_FILLER = (
    "The history of computing spans mechanical calculators, vacuum tubes, "
    "transistors, integrated circuits, and modern multicore processors. "
    "Each generation traded cost, power, density, and reliability against "
    "raw performance. "
)  # roughly 40 tokens per repetition (varies slightly by tokenizer)

_LONG_PREFIX = (
    "Summarize the following passage and then answer: what are the key "
    "tradeoffs discussed?\n\n"
)

PROMPTS = {
    "short": "Explain why the sky is blue in two sentences.",
    "medium": (
        "Write a detailed step-by-step explanation of how a modern CPU "
        "pipeline works, covering fetch, decode, execute, memory access, "
        "and writeback stages, including hazards and branch prediction. "
        "Aim for thoroughness."
    ),
    "long": _LONG_PREFIX + _FILLER * 60,            # ~2.5k tokens
    "extra_long": _LONG_PREFIX + _FILLER * 200,     # ~8k tokens
    "extremely_long": _LONG_PREFIX + _FILLER * 400, # ~16k tokens
    "colossal_32k": _LONG_PREFIX + _FILLER * 800,   # ~32k tokens
    "colossal_64k": _LONG_PREFIX + _FILLER * 1600,  # ~64k tokens
}

# No num_ctx is set here: context window is inherited from each model's
# Modelfile. The per-run "prompt tok" count printed below confirms the full
# prompt was processed (if it reads far lower than expected, the model's
# context is smaller than the prompt and Ollama truncated it).

GEN_TOKENS = 256  # fixed generation length for comparable numbers


def http_json(path, payload=None, timeout=600):
    url = OLLAMA_URL + path
    if payload is not None:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
    else:
        req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def sh(cmd):
    try:
        return subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=30
        ).stdout.strip()
    except Exception as e:
        return f"unavailable ({e})"


def hardware_snapshot():
    return {
        "gpus_lspci": sh("lspci | grep -i 'vga\\|display'"),
        "rocm_smi": sh("rocm-smi --showproductname --showmeminfo vram --csv 2>/dev/null"),
        "kernel": sh("uname -r"),
        "ollama_version": sh("ollama --version"),
    }


def gpu_vram_used():
    """Grab current VRAM usage per GPU via rocm-smi (best effort)."""
    return sh("rocm-smi --showmeminfo vram --csv 2>/dev/null")


def bench_once(model, prompt):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": GEN_TOKENS, "temperature": 0},
    }
    t0 = time.time()
    resp = http_json("/api/generate", payload)
    wall = time.time() - t0

    # Ollama reports durations in nanoseconds
    pe_count = resp.get("prompt_eval_count", 0)
    pe_dur = resp.get("prompt_eval_duration", 0) / 1e9
    ev_count = resp.get("eval_count", 0)
    ev_dur = resp.get("eval_duration", 0) / 1e9

    return {
        "wall_seconds": round(wall, 2),
        "prompt_tokens": pe_count,
        "prompt_eval_tps": round(pe_count / pe_dur, 1) if pe_dur > 0 else None,
        "gen_tokens": ev_count,
        "gen_tps": round(ev_count / ev_dur, 1) if ev_dur > 0 else None,
        "total_duration_s": round(resp.get("total_duration", 0) / 1e9, 2),
        "load_duration_s": round(resp.get("load_duration", 0) / 1e9, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="Ollama model name, e.g. llama3.1:8b")
    ap.add_argument("--runs", type=int, default=3, help="runs per prompt size (default 3)")
    ap.add_argument("--label", default="", help="tag this run, e.g. 'dual-r9700'")
    args = ap.parse_args()

    # Sanity check: is Ollama up? Is the model available?
    try:
        tags = http_json("/api/tags")
    except Exception as e:
        sys.exit(f"Cannot reach Ollama at {OLLAMA_URL}: {e}")
    models = [m["name"] for m in tags.get("models", [])]
    if args.model not in models and not any(m.startswith(args.model) for m in models):
        print(f"Note: '{args.model}' not in local model list; Ollama may pull it first.")

    print(f"\n=== Benchmarking {args.model} ({args.runs} runs/prompt) ===")
    print("Warming up (loads model into VRAM)...")
    bench_once(args.model, "Say hello.")  # warmup, excluded from results

    # Where did the model land? (100% GPU vs partial CPU offload matters a lot)
    ps = sh("ollama ps")
    print(f"\nollama ps after load:\n{ps}\n")
    print(f"VRAM usage:\n{gpu_vram_used()}\n")

    results = {}
    for size, prompt in PROMPTS.items():
        print(f"\n-- {size} --")
        runs = []
        for i in range(args.runs):
            # Ollama caches the KV state of a repeated prompt prefix, which
            # makes repeat runs report absurdly high prompt-eval speeds
            # (cache hit, not real prefill). A unique nonce at the very START
            # of the prompt breaks the prefix match so every run measures a
            # true full prefill. Adds only ~10 tokens.
            nonce = f"[bench session {uuid.uuid4().hex[:12]}]\n"
            r = bench_once(args.model, nonce + prompt)
            runs.append(r)
            print(f"  [{size} run {i+1}/{args.runs}] "
                  f"{r['prompt_tokens']} prompt tok | "
                  f"prompt: {r['prompt_eval_tps']} tok/s | "
                  f"gen: {r['gen_tps']} tok/s | "
                  f"wall: {r['wall_seconds']}s")
        pe = [r["prompt_eval_tps"] for r in runs if r["prompt_eval_tps"]]
        gv = [r["gen_tps"] for r in runs if r["gen_tps"]]
        results[size] = {
            "runs": runs,
            "prompt_eval_tps_mean": round(mean(pe), 1) if pe else None,
            "prompt_eval_tps_stdev": round(stdev(pe), 1) if len(pe) > 1 else 0,
            "gen_tps_mean": round(mean(gv), 1) if gv else None,
            "gen_tps_stdev": round(stdev(gv), 1) if len(gv) > 1 else 0,
        }

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "model": args.model,
        "gen_tokens_requested": GEN_TOKENS,
        "hardware": hardware_snapshot(),
        "ollama_ps": ps,
        "results": results,
    }

    # Append to results file
    try:
        with open(RESULTS_FILE) as f:
            history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        history = []
    history.append(record)
    with open(RESULTS_FILE, "w") as f:
        json.dump(history, f, indent=2)

    # Summary table
    print(f"\n=== Summary: {args.model} {f'[{args.label}]' if args.label else ''} ===")
    print(f"{'prompt':<8} {'prompt eval tok/s':>18} {'generation tok/s':>18}")
    for size, r in results.items():
        print(f"{size:<8} "
              f"{str(r['prompt_eval_tps_mean']) + ' ± ' + str(r['prompt_eval_tps_stdev']):>18} "
              f"{str(r['gen_tps_mean']) + ' ± ' + str(r['gen_tps_stdev']):>18}")
    print(f"\nSaved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
