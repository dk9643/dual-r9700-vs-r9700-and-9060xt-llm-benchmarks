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
[that section](#the-moe-crash) might be the most useful thing here. The MoE
models did get benchmarked in the end — through llama.cpp's Vulkan backend,
which sidesteps the bug entirely.

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
| short | ~39 | 213.3 ± 4.0 | 14.9 ± 0.0 |
| medium | ~68 | 352.3 ± 4.0 | 14.9 ± 0.1 |
| long | ~2.4k | 1264.6 ± 1.8 | 14.7 ± 0.1 |
| extra_long | ~8.0k | 1443.8 ± 2.2 | 14.3 ± 0.1 |
| extremely_long | ~16.0k | 1332.4 ± 0.4 | 13.9 ± 0.0 |
| colossal_32k | ~32.0k | 1108.8 ± 4.0 | 13.0 ± 0.0 |
| colossal_64k | ~64.0k | 806.8 ± 0.1 | 11.6 ± 0.0 |

### gemma-4-31b-it (dense)

| tier | prompt tokens | prefill tok/s | decode tok/s |
|---|---|---|---|
| short | ~42 | 216.2 ± 20.9 | 12.8 ± 0.0 |
| medium | ~77 | 344.6 ± 7.3 | 12.8 ± 0.0 |
| long | ~2.3k | 924.9 ± 17.1 | 12.4 ± 0.0 |
| extra_long | ~7.4k | 1061.5 ± 4.4 | 12.0 ± 0.0 |
| extremely_long | ~14.9k | 1003.3 ± 5.5 | 11.5 ± 0.0 |
| colossal_32k | ~29.7k | 825.4 ± 1.1 | 10.6 ± 0.0 |
| colossal_64k | ~59.3k | 577.0 ± 0.1 | 9.2 ± 0.0 |

### gemma-4-26b-a4b (MoE) and qwen3.5-35b-a3b (MoE)

**These couldn't be benchmarked on Ollama/ROCm** — both crash within a few
requests (see [The MoE crash](#the-moe-crash)). Both load fully onto GPU
(split across the cards) and generate coherently until the crash; the runs
that completed before it showed ~56.5 and ~60.3 tok/s decode respectively.
Full numbers for both came from llama.cpp's Vulkan backend instead — next
section.

## Results — before (llama.cpp Vulkan)

All four models were also run through `llama-server` (llama.cpp build
10745) on the Vulkan backend, which compiles GPU code per-device at runtime
and is structurally immune to the rocBLAS mixed-architecture bug. For the
MoEs this is the only full benchmark data on this config; for the dense
models it doubles as a cross-stack check. Same methodology
([llamacpp_bench.py](llamacpp_bench.py) mirrors the Ollama script: same seven
tiers, nonce cache-busting plus `cache_prompt: false`, 256 generation
tokens, temperature 0, 5 runs, warmup excluded) and the same settings as the
Ollama runs: ctx 131072, flash attention on, q8_0 KV cache, split across
both discrete GPUs. Raw data in
[vulkan_bench_results.json](vulkan_bench_results.json).

### qwen3.5-27b (dense)

| tier | prompt tokens | prefill tok/s | decode tok/s |
|---|---|---|---|
| short | ~28 | 109.8 ± 8.2 | 14.3 ± 0.1 |
| medium | ~55 | 207.4 ± 6.9 | 14.4 ± 0.1 |
| long | ~2.4k | 869.4 ± 17.2 | 14.4 ± 0.0 |
| extra_long | ~8.0k | 1164.6 ± 3.7 | 14.2 ± 0.1 |
| extremely_long | ~16.0k | 1174.8 ± 6.2 | 13.9 ± 0.1 |
| colossal_32k | ~32.0k | 1075.7 ± 4.8 | 13.5 ± 0.1 |
| colossal_64k | ~64.0k | 867.5 ± 0.7 | 12.7 ± 0.0 |

### gemma-4-31b-it (dense)

| tier | prompt tokens | prefill tok/s | decode tok/s |
|---|---|---|---|
| short | ~26 | 109.7 ± 10.5 | — |
| medium | ~60 | 212.4 ± 23.2 | 12.5 ± 0.1 |
| long | ~2.3k | 677.9 ± 3.6 | 12.0 ± 0.1 |
| extra_long | ~7.4k | 902.8 ± 4.9 | 11.5 ± 0.1 |
| extremely_long | ~14.8k | 894.0 ± 4.0 | 11.1 ± 0.0 |
| colossal_32k | ~29.6k | 733.5 ± 0.9 | 10.7 ± 0.0 |
| colossal_64k | ~59.2k | 508.1 ± 0.2 | 9.9 ± 0.0 |

*(The empty short-tier decode cell: on the raw short prompt this model emitted
an immediate end-of-sequence token in all five runs, so there was nothing to
time. A raw-completion quirk at temperature 0, not a performance issue.)*

### gemma-4-26b-a4b (MoE, ~4B active)

| tier | prompt tokens | prefill tok/s | decode tok/s |
|---|---|---|---|
| short | ~27 | 259.5 ± 20.9 | 59.7 ± 0.2 |
| medium | ~61 | 442.0 ± 16.3 | 59.5 ± 0.1 |
| long | ~2.3k | 1983.7 ± 42.4 | 55.7 ± 0.3 |
| extra_long | ~7.4k | 3197.8 ± 55.3 | 53.1 ± 0.2 |
| extremely_long | ~14.8k | 3373.0 ± 14.3 | 51.3 ± 0.1 |
| colossal_32k | ~29.6k | 2831.8 ± 5.2 | 48.4 ± 0.1 |
| colossal_64k | ~59.2k | 1933.9 ± 1.4 | 43.6 ± 0.2 |

### qwen3.5-35b-a3b (MoE, ~3B active)

| tier | prompt tokens | prefill tok/s | decode tok/s |
|---|---|---|---|
| short | ~27 | 219.6 ± 23.3 | 68.7 ± 1.3 |
| medium | ~58 | 369.4 ± 27.6 | 69.1 ± 0.1 |
| long | ~2.4k | 2101.1 ± 50.3 | 68.4 ± 0.1 |
| extra_long | ~8.0k | 3382.9 ± 28.0 | 67.2 ± 0.1 |
| extremely_long | ~16.0k | 3483.9 ± 13.6 | 65.7 ± 0.1 |
| colossal_32k | ~32.0k | 3119.0 ± 10.4 | 62.4 ± 0.0 |
| colossal_64k | ~64.0k | 2394.1 ± 6.0 | 56.3 ± 0.2 |

Worth noticing:

- qwen3.5-35b-a3b's decode barely decays with context depth in the normal
  range — 68.7 → 65.7 tok/s (~4%) from short to ~16k, the flattest curve of
  the four. That's its Gated DeltaNet linear-attention blocks doing exactly
  what they promise. The gemma MoE drops ~14% over the same range.
- The colossal tiers are where deep context bites everyone: prefill peaks
  around the 8–16k tiers and falls off hard by ~64k (gemma MoE 3373 → 1934
  tok/s), and decode keeps sliding too — though the qwen MoE still holds up
  best (68.7 → 56.3 by ~64k, vs 59.7 → 43.6 for the gemma MoE).
- **Cross-stack sanity check:** comparing the dense tables here against the
  Ollama ones above, Vulkan decode lands within ~5% of ROCm (13.9 tok/s on
  both stacks for qwen at ~16k — and at 64k the Vulkan runs actually decay
  *less*: 12.7 vs 11.6), while Vulkan prefill runs ~15–20% lower through the
  mid tiers. So the MoE numbers are, if anything, slightly conservative
  relative to what ROCm should do once its bug is fixed.

## Results — after (dual R9700)

Doesn't exist yet — the second R9700 isn't in the machine. Once it is, this
gets generated with:

```
python3 compare.py --before "r9700+9060xt" --after "dual-r9700" --markdown
python3 compare.py --file vulkan_bench_results.json --before "r9700+9060xt" --after "dual-r9700" --markdown
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
| Backend | llama.cpp Vulkan, same models, same GPU split | no crash — full suite passes |

The last row is the clincher: the same two models on the same mixed-GPU
split completed the entire benchmark suite on Vulkan (~125 requests across
all four models) without a single error. The bug lives in the ROCm stack,
not in the models or the hardware.

**What it actually is:** this matches
[llama.cpp #19893](https://github.com/ggml-org/llama.cpp/issues/19893) —
the same error when a model is split across GPUs with *different*
architectures (here, gfx1201 + gfx1200). That issue was traced to a rocBLAS
bug ([ROCm/rocm-libraries#3413](https://github.com/ROCm/rocm-libraries/issues/3413):
rocBLAS can launch a kernel object compiled for the wrong GPU in
mixed-architecture systems) and closed once a fix
([ROCm/rocm-libraries#4781](https://github.com/ROCm/rocm-libraries/pull/4781),
"solution library per gfx", merged 2026-03-02) landed — confirmed fixed by
building rocBLAS from source with that commit.

The catch: that fix is not in any ROCm 7.2.x release (verified by commit
containment against the 7.2.1–7.2.4 tags), and Ollama bundles the 7.2 line —
so even the latest Ollama (0.33.2 at the time of writing) still ships the
bug. The crashes above are the proof. On stock Ollama the workaround remains
single-GPU execution. MoE expert routing exercises far more of the
multi-device dispatch paths than dense inference does, which is why only the
MoEs trigger it here.
([llama.cpp #20024](https://github.com/ggml-org/llama.cpp/issues/20024), a
Qwen3.5-35B-A3B ROCm crash with a different signature, was a separate bug —
fixed in llama.cpp in March 2026 and already absent from current builds.)

**The implication:** the GPU upgrade itself should fix this — two R9700s
make the split homogeneous (gfx1201 + gfx1201), which sidesteps the rocBLAS
bug entirely. The "after" half of this repo will test that prediction. For
anyone stuck on mixed cards there are two escape hatches: wait for Ollama's
bundle to move past ROCm 7.2.x (the fix first ships in newer ROCm
releases), or use a Vulkan backend — that worked here, and it's where the
MoE numbers above came from.

## How the measuring works

- [ollama_bench.py](ollama_bench.py) is stdlib-only Python talking to the
  local Ollama API. Seven prompt tiers (~40 tokens up to ~64k tokens), 256
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
- **Vulkan side-suite:** the MoE numbers come from
  [llamacpp_bench.py](llamacpp_bench.py) hitting `llama-server` — identical
  tiers, nonce, and decode length, plus `cache_prompt: false` on every
  request. Server settings matched the Ollama config:
  `-c 131072 -fa on -ctk q8_0 -ctv q8_0 -ngl 999 --device Vulkan1,Vulkan2`.
  That last flag matters: Vulkan enumerates the iGPU as a device, and
  without it the iGPU takes layers backed by slow system RAM.
- **Caveats:** the dense numbers are Ollama-specific (its scheduler, its
  bundled llama.cpp/ROCm build) and the MoE numbers are
  llama.cpp-Vulkan-specific — don't expect either to transfer exactly to
  other stacks, and don't compare across the two tables without the
  cross-stack note above. Short-tier prefill tok/s mostly measures fixed
  per-request overhead, not real throughput. Both PCIe slots run x8.

## Running it yourself

Ollama:

```
python3 ollama_bench.py <model>:<tag> --runs 5 --label "your-config-name"
python3 compare.py --before "config-a" --after "config-b"
```

llama.cpp Vulkan (one model at a time):

```
llama-server -m <model>.gguf -c 131072 -ngl 999 -fa on -ctk q8_0 -ctv q8_0 --device Vulkan1,Vulkan2 --port 8089
python3 llamacpp_bench.py <model-name> --runs 5 --label "your-config-name" --url http://127.0.0.1:8089
python3 compare.py --file vulkan_bench_results.json --before "config-a" --after "config-b"
```

Results append to `ollama_bench_results.json` / `vulkan_bench_results.json`
with a hardware snapshot per record.

## License

MIT — see [LICENSE](LICENSE).
