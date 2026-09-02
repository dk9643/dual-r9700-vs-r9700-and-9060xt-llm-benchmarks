#!/usr/bin/env python3
"""
llama.cpp (Vulkan) benchmark via the llama-server /completion API.

Mirrors ollama_bench.py methodology exactly: same five prompt tiers, unique
nonce per run to defeat prompt caching (plus cache_prompt=false), 256
generation tokens, temperature 0, warmup excluded.

Usage: python3 llamacpp_bench.py <model_name> [--runs N] [--label L] [--url U]
Results append to vulkan_bench_results.json in the current directory,
same schema as ollama_bench_results.json so compare.py works on it.
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

RESULTS_FILE = "vulkan_bench_results.json"

_FILLER = (
    "The history of computing spans mechanical calculators, vacuum tubes, "
    "transistors, integrated circuits, and modern multicore processors. "
    "Each generation traded cost, power, density, and reliability against "
    "raw performance. "
)

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
    "long": _LONG_PREFIX + _FILLER * 60,
    "extra_long": _LONG_PREFIX + _FILLER * 200,
    "extremely_long": _LONG_PREFIX + _FILLER * 400,
    "colossal_32k": _LONG_PREFIX + _FILLER * 800,
    "colossal_64k": _LONG_PREFIX + _FILLER * 1600,
}

GEN_TOKENS = 256


def http_json(url, payload=None, timeout=600):
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
        "llamacpp_version": sh("llama-server --version 2>&1 | grep -E '^version' | head -1"),
    }


def bench_once(url, prompt):
    payload = {
        "prompt": prompt,
        "n_predict": GEN_TOKENS,
        "temperature": 0,
        "cache_prompt": False,
    }
    t0 = time.time()
    resp = http_json(url + "/completion", payload)
    wall = time.time() - t0

    t = resp.get("timings", {})
    pps = t.get("prompt_per_second")
    gps = t.get("predicted_per_second")
    return {
        "wall_seconds": round(wall, 2),
        "prompt_tokens": t.get("prompt_n", 0),
        "prompt_eval_tps": round(pps, 1) if pps else None,
        "gen_tokens": t.get("predicted_n", 0),
        "gen_tps": round(gps, 1) if gps else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_name", help="name to record for this model")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--label", default="")
    ap.add_argument("--url", default="http://127.0.0.1:8089")
    args = ap.parse_args()

    print(f"\n=== Benchmarking {args.model_name} via llama-server ({args.runs} runs/prompt) ===")
    print("Warming up...")
    bench_once(args.url, "Say hello.")

    results = {}
    for size, prompt in PROMPTS.items():
        print(f"\n-- {size} --")
        runs = []
        for i in range(args.runs):
            nonce = f"[bench session {uuid.uuid4().hex[:12]}]\n"
            r = bench_once(args.url, nonce + prompt)
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
        "model": args.model_name,
        "backend": "llama.cpp-vulkan",
        "gen_tokens_requested": GEN_TOKENS,
        "hardware": hardware_snapshot(),
        "results": results,
    }

    try:
        with open(RESULTS_FILE) as f:
            history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        history = []
    history.append(record)
    with open(RESULTS_FILE, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n=== Summary: {args.model_name} {f'[{args.label}]' if args.label else ''} ===")
    print(f"{'prompt':<8} {'prompt eval tok/s':>18} {'generation tok/s':>18}")
    for size, r in results.items():
        print(f"{size:<8} "
              f"{str(r['prompt_eval_tps_mean']) + ' ± ' + str(r['prompt_eval_tps_stdev']):>18} "
              f"{str(r['gen_tps_mean']) + ' ± ' + str(r['gen_tps_stdev']):>18}")
    print(f"\nSaved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
