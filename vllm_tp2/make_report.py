#!/usr/bin/env python3
"""make_report.py - markdown comparison tables: vLLM TP=2 results vs the published layer-split numbers.

Reads ../vllm_tp2_bench_results.json (this experiment; one record per model per fresh
server launch) and the published ollama/vulkan JSONs from the repo root (read-only).
Published labels: "dual-r9700" (2x R9700) and "r9700+9060xt" (R9700 + RX 9060 XT).

compare.py cannot cross-match these records because the model keys differ
(qwen3.5-27b-fp8 vs qwen3.5-27b:131072); the MODELS table below does the mapping.
"""
import json
import os
import statistics

TIERS = ["short", "medium", "long", "extra_long", "extremely_long", "colossal_32k", "colossal_64k"]
HERE = os.path.dirname(os.path.abspath(__file__))
GPU_REPO = os.path.join(HERE, "..")
TP2_RESULTS = os.path.join(GPU_REPO, "vllm_tp2_bench_results.json")
MODELS = {  # served name -> (ollama model key, vulkan model key, display)
    "qwen3.5-27b-fp8": ("qwen3.5-27b:131072", "qwen3.5-27b", "Qwen3.5-27B"),
    "gemma-4-31b-it-fp8": ("gemma-4-31b-it:latest", "gemma-4-31b-it", "Gemma-4-31B-it"),
}
PUBLISHED = [  # (column label, file, label in file, key index into MODELS tuple)
    ("2xR9700 Ollama", "ollama_bench_results.json", "dual-r9700", 0),
    ("2xR9700 llama.cpp", "vulkan_bench_results.json", "dual-r9700", 1),
    ("R9700+9060XT Ollama", "ollama_bench_results.json", "r9700+9060xt", 0),
    ("R9700+9060XT llama.cpp", "vulkan_bench_results.json", "r9700+9060xt", 1),
]


def load_published(fn, label, model_key):
    found = {}
    for e in json.load(open(os.path.join(GPU_REPO, fn))):
        if e.get("label") == label and e.get("model") == model_key:
            found = e["results"]  # latest record wins, same rule as compare.py
    return found


def fmt(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def best(xs):
    xs = [x for x in xs if x is not None]
    return max(xs) if xs else None


def table(header_cells, rows):
    print("| " + " | ".join(header_cells) + " |")
    print("|" + "---|" * len(header_cells))
    for r in rows:
        print("| " + " | ".join(r) + " |")


def main():
    runs = {}
    for d in json.load(open(TP2_RESULTS)):
        runs.setdefault(d["model"], {})[d.get("launch", "A")] = d

    for served, keys in MODELS.items():
        if served not in runs:
            continue
        disp = keys[2]
        launches = sorted(runs[served])
        pub = [(col, load_published(fn, label, keys[ki])) for col, fn, label, ki in PUBLISHED]

        print(f"\n### {disp} - decode tok/s (single request, 256 gen tokens, temp 0)\n")
        hdr = ["Tier", "Prompt tok"] + [c for c, _ in pub] + [f"TP=2 {L}" for L in launches] + \
              ["TP=2 runs (min-max)", "TP=2 vs best 2xR9700 split", "TP=2 vs best R9700+9060XT split"]
        rows = []
        for t in TIERS:
            allruns = [r["gen_tps"] for L in launches for r in runs[served][L]["results"][t]["runs"] if r["gen_tps"]]
            ptok = runs[served][launches[0]]["results"][t]["runs"][0]["prompt_tokens"]
            m = statistics.mean(allruns) if allruns else None
            vals = [res.get(t, {}).get("gen_tps_mean") for _, res in pub]
            b_dual = best(vals[0:2]); b_mix = best(vals[2:4])
            rows.append([t, str(ptok)] + [fmt(v) for v in vals]
                        + [fmt(runs[served][L]["results"][t]["gen_tps_mean"]) for L in launches]
                        + [f"{fmt(min(allruns))}-{fmt(max(allruns))}" if allruns else "-",
                           f"{m / b_dual:.2f}x" if m and b_dual else "-",
                           f"{m / b_mix:.2f}x" if m and b_mix else "-"])
        table(hdr, rows)

        print(f"\n### {disp} - prefill tok/s\n")
        hdr = ["Tier"] + [c for c, _ in pub] + [f"TP=2 {L}" for L in launches]
        rows = []
        for t in TIERS:
            vals = [res.get(t, {}).get("prompt_eval_tps_mean") for _, res in pub]
            rows.append([t] + [fmt(v, 0) for v in vals]
                        + [fmt(runs[served][L]["results"][t]["prompt_eval_tps_mean"], 0) for L in launches])
        table(hdr, rows)

        notes = []
        for L in launches:
            for t in TIERS:
                cts = [r["completion_tokens"] for r in runs[served][L]["results"][t]["runs"]]
                if min(cts) < 128:
                    notes.append(f"launch {L} {t}: only {min(cts)}-{max(cts)} tokens generated (model hit EOS)")
        if notes:
            print("\nShort generations: " + "; ".join(notes))


if __name__ == "__main__":
    main()
