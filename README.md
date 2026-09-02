# Dual-GPU LLM benchmarks: AMD R9700 + RX 9060 XT → dual R9700

This repo benchmarks local LLM speeds (Ollama) before and after a GPU
upgrade: swapping an RX 9060 XT (16 GB) for a second Radeon AI PRO R9700 —
going from 48 GB of mismatched VRAM to 64 GB of matching gfx1201. The
benchmark tooling is included.

**Status: the "before" numbers are done. "After" numbers coming once the
second card is in.**

Along the way, benchmarking surfaced a reproducible ROCm crash that
affects mixture-of-experts models split across mismatched AMD GPUs. For
anyone thinking about mixing different AMD cards in one box,
[that section](#the-moe-crash) might be the most useful thing here.

## Hardware

| | Before | After |
|---|---|---|
| GPU 0 | Radeon AI PRO R9700 32 GB (gfx1201) | Radeon AI PRO R9700 32 GB (gfx1201) |
| GPU 1 | Radeon RX 9060 XT 16 GB (gfx1200) | Radeon AI PRO R9700 32 GB (gfx1201) |
| Total VRAM | 48 GB (mixed arch) | 64 GB (uniform arch) |

Both GPUs run at PCIe x8. Full platform details are in
[environment.md](environment.md).

## Models

All four are vanilla instruct releases, Q8-class GGUF quants, created in
Ollama with `num_ctx 131072`. Modelfiles with exact provenance are in
[modelfiles/](modelfiles/).

| Model | Type | Quant | VRAM footprint (loaded, 131k ctx) |
|---|---|---|---|
| qwen3.5-27b | dense | Q8_0 (28 GB) | 34 GB, 100% GPU |
| gemma-4-31b-it | dense | Q8_0 (33 GB) | 36 GB, 100% GPU |
| gemma-4-26b-a4b | MoE, ~4B active | UD-Q8_K_XL (27.6 GB) | 31 GB, 100% GPU |
| qwen3.5-35b-a3b | MoE, ~3B active | Q8_0 (36.9 GB) | 39 GB, 100% GPU |

## Results — before (R9700 + RX 9060 XT)

Ollama 0.33.2, 5 runs per tier, mean ± stdev. Prefill is how fast the prompt
gets processed (compute-bound); decode is how fast tokens come out
(memory-bandwidth-bound).

### qwen3.5-27b (dense)

| tier | prompt tokens | prefill tok/s | decode tok/s |
|---|---|---|---|
| short | ~36 | 205.7 ± 5.6 | 14.9 ± 0.0 |
| medium | ~67 | 351.7 ± 6.1 | 14.9 ± 0.0 |
| long | ~2.4k | 1263.4 ± 6.9 | 14.7 ± 0.0 |
| extra_long | ~8.0k | 1444.6 ± 1.7 | 14.3 ± 0.1 |
| extremely_long | ~16.0k | 1331.8 ± 1.1 | 13.9 ± 0.0 |

### gemma-4-31b-it (dense)

| tier | prompt tokens | prefill tok/s | decode tok/s |
|---|---|---|---|
| short | ~42 | 224.6 ± 3.1 | 12.8 ± 0.1 |
| medium | ~77 | 330.6 ± 38.6 | 12.8 ± 0.0 |
| long | ~2.3k | 942.3 ± 10.3 | 12.4 ± 0.0 |
| extra_long | ~7.5k | 1056.2 ± 21.1 | 12.0 ± 0.0 |
| extremely_long | ~14.8k | 1003.7 ± 4.9 | 11.5 ± 0.0 |

### gemma-4-26b-a4b (MoE) and qwen3.5-35b-a3b (MoE)

**These couldn't be benchmarked on this config** — both crash within a few
requests (see [The MoE crash](#the-moe-crash)). Here's the partial data from
the runs that completed before the crash. Indicative only:

| Model | short-prompt prefill | decode |
|---|---|---|
| gemma-4-26b-a4b | ~480–520 tok/s | ~56.5 tok/s |
| qwen3.5-35b-a3b | ~400–430 tok/s | ~60.3 tok/s |

Both MoEs loaded fully onto GPU (split across both cards) and produced
coherent output — roughly 4× the decode speed of the dense models, which is
what ~3–4B active params should deliver. Neither survived a full benchmark
run.

## Results — after (dual R9700)

Doesn't exist yet — the second R9700 isn't in the machine. Once it is, this
gets generated with:

```
python3 compare.py --before "r9700+9060xt" --after "dual-r9700" --markdown
```

## The MoE crash

Both MoE models crash after 2–4 successful requests with:

```
ggml_cuda_compute_forward: MUL_MAT failed
ROCm error: no kernel image is available for execution on the device
```

The llama.cpp runner inside Ollama aborts (core dump), Ollama hands the
benchmark an HTTP 500 and reloads the model. Meanwhile the two dense
models did 100+ requests each on the identical stack without a single error.

Things ruled out, one variable at a time:

| Variable | Tested | Result |
|---|---|---|
| KV cache quantization | q8_0 → f16 | still crashes |
| Ollama / engine version | 0.32.14 → 0.33.2 | still crashes |
| Quant style | UD-Q8_K_XL (gemma) and plain Q8_0 (qwen) | both crash |
| Model family | Gemma 4 MoE and Qwen3.5 MoE | both crash |

**What it actually is:** this matches
[llama.cpp #19893](https://github.com/ggml-org/llama.cpp/issues/19893) —
when a model is split across GPUs with *different* architectures (here,
gfx1201 + gfx1200), the engine can dispatch a kernel variant that was never
compiled for the device it lands on. The reported workaround is single-GPU
execution. MoE expert routing exercises way more of the multi-device dispatch
paths than dense inference does, which is why only the MoEs trigger it. See
also [llama.cpp #20024](https://github.com/ggml-org/llama.cpp/issues/20024)
for a related Qwen3.5-35B-A3B ROCm crash. Both issues were still open at the
time of writing — no fixed release exists.

**The implication:** the GPU upgrade itself should fix this. Two R9700s
means the split becomes homogeneous (gfx1201 + gfx1201) — exactly the
configuration that doesn't crash. The "after" half of this repo will test
that prediction.

## How the measuring works

- [ollama_bench.py](ollama_bench.py) is stdlib-only Python talking to the
  local Ollama API. Five prompt tiers (~40 tokens up to ~16k tokens), 256
  generation tokens, temperature 0, 5 runs per tier, one warmup request
  that doesn't count.
- **Cache busting:** Ollama caches KV state for repeated prompt prefixes,
  which makes repeat runs report inflated prefill speeds. Every
  run prepends a unique nonce so it's always a real, full prefill.
- **Truncation detection:** the script logs the server-reported prompt token
  count on every run — silent context truncation would show up there. All
  published runs processed their full prompts (`num_ctx 131072` in every
  Modelfile).
- **Isolation:** one model loaded at a time (`ollama stop` between models).
  Placement (`ollama ps`, per-GPU VRAM) gets recorded into the results JSON
  with every record.
- **Why only 256 decode tokens:** decode stdev across runs was ≤ 0.2 tok/s
  at every tier, and deep-context effects get captured by the prompt tiers
  rather than by generating longer.
- **Version stability:** the dense suite was run on both Ollama 0.32.14 and
  0.33.2 — everything agreed within noise (decode within ±0.1 tok/s). The
  tables above are 0.33.2.
- **Caveats:** these numbers are Ollama-specific (its scheduler, its bundled
  llama.cpp/ROCm build) — don't expect them to transfer exactly to raw
  llama.cpp or vLLM. Short-tier prefill tok/s mostly measures fixed
  per-request overhead, not real throughput. Both PCIe slots run x8.

## Running it yourself

```
python3 ollama_bench.py <model>:<tag> --runs 5 --label "your-config-name"
python3 compare.py --before "config-a" --after "config-b"
```

Results append to `ollama_bench_results.json` with a hardware snapshot per
record.

## License

MIT — see [LICENSE](LICENSE).
