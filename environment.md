# Environment

## Platform

- Motherboard: Gigabyte AI TOP B850
- CPU: AMD Ryzen 9 9900X (12 cores)
- RAM: 32 GB
- OS: Ubuntu Server, kernel 7.0.0-30-generic (in-kernel amdgpu driver)
- Storage: NVMe SSD

## GPUs (before configuration)

| Device | VRAM | Arch | PCIe |
|---|---|---|---|
| Radeon AI PRO R9700 | 32 GB | gfx1201 | x8 |
| Radeon RX 9060 XT | 16 GB | gfx1200 | x8 |
| Ryzen 9900X iGPU | 512 MB | gfx1036 | — (unused; takes no layers) |

## Software

- Ollama 0.33.2 (official installer, systemd service), bundling its own
  llama.cpp + ROCm 7.2 runtime (`rocm_v7_2`) — dense-model benchmarks
  - a 0.32.14 run of the dense suite agreed within noise
- llama.cpp build 10745 (commit `c845263f8`), Vulkan backend via RADV —
  MoE-model benchmarks (see the README for why)
- Host ROCm tooling: rocm-smi 7.1.1
- Python 3.14.4 (benchmark scripts, stdlib only)

## Ollama service configuration

Set via systemd override (`/etc/systemd/system/ollama.service.d/override.conf`):

```
OLLAMA_FLASH_ATTENTION=1
OLLAMA_KV_CACHE_TYPE=q8_0
OLLAMA_KEEP_ALIVE=-1
OLLAMA_MODELS=<local model store path>
```

Notes:

- `OLLAMA_NUM_PARALLEL` unset (defaults to 1) — a single request slot, so the
  full `num_ctx` belongs to one sequence.
- KV cache quantization (`q8_0`) + flash attention roughly halve KV memory at
  131k context vs f16; all published numbers use this configuration.
- `OLLAMA_KEEP_ALIVE=-1` keeps models resident indefinitely; the benchmark
  runner explicitly runs `ollama stop <model>` between models so each is
  measured alone.
