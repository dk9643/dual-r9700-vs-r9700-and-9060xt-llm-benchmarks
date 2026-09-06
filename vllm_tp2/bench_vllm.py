#!/usr/bin/env python3
"""bench_vllm.py - single-request decode/prefill benchmark against a vLLM /v1/completions endpoint.

Methodology matches the published dual-R9700 Ollama / llama.cpp numbers:
seven prompt tiers, one excluded warmup, 3 runs per tier, 256 gen tokens, temperature 0,
raw prompt (no chat template), unique nonce per request, no prefix caching.

vLLM returns no per-request timings, so they are derived from the SSE stream:
  t0 = request sent, t_first = first non-empty content chunk, t_last = last content chunk.
  prefill tok/s = prompt_tokens / (t_first - t0)
  decode  tok/s = (completion_tokens - 1) / (t_last - t_first)
Stdlib only (http.client) so nothing needs installing on the server.
"""
import argparse
import http.client
import json
import statistics
import time
import uuid
from datetime import datetime, timezone

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
    "medium": ("Write a detailed step-by-step explanation of how a modern CPU "
               "pipeline works, covering fetch, decode, execute, memory access, "
               "and writeback stages, including hazards and branch prediction. "
               "Aim for thoroughness."),
    "long": _LONG_PREFIX + _FILLER * 60,
    "extra_long": _LONG_PREFIX + _FILLER * 200,
    "extremely_long": _LONG_PREFIX + _FILLER * 400,
    "colossal_32k": _LONG_PREFIX + _FILLER * 800,
    "colossal_64k": _LONG_PREFIX + _FILLER * 1600,
}
GEN_TOKENS = 256


def one_request(host, port, model, prompt, timeout):
    body = json.dumps({
        "model": model, "prompt": prompt, "max_tokens": GEN_TOKENS,
        "temperature": 0, "stream": True,
        "stream_options": {"include_usage": True},
    })
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    t0 = time.perf_counter()
    conn.request("POST", "/v1/completions", body=body,
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    if resp.status != 200:
        raise RuntimeError("HTTP %d: %r" % (resp.status, resp.read()[:500]))
    t_first = None
    t_last = None
    n_chunks = 0
    usage = None
    finish = None
    text = []
    buf = b""
    while True:
        data = resp.read1(65536) if hasattr(resp, "read1") else resp.read(65536)
        if not data:
            break
        buf += data
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                continue
            now = time.perf_counter()
            obj = json.loads(payload)
            if obj.get("usage"):
                usage = obj["usage"]
            ch = obj.get("choices") or []
            if ch:
                piece = ch[0].get("text") or ""
                if ch[0].get("finish_reason"):
                    finish = ch[0]["finish_reason"]
                if piece:
                    if t_first is None:
                        t_first = now
                    t_last = now
                    n_chunks += 1
                    text.append(piece)
    conn.close()
    t_end = time.perf_counter()
    if usage is None:
        raise RuntimeError("no usage in stream (server needs stream_options.include_usage support)")
    p_tok = usage["prompt_tokens"]
    c_tok = usage["completion_tokens"]
    if t_first is None:
        t_first = t_end
    prefill_tps = p_tok / (t_first - t0) if t_first > t0 else None
    gen_tps = None
    if c_tok >= 2 and t_last is not None and t_last > t_first:
        gen_tps = (c_tok - 1) / (t_last - t_first)
    return {
        "prompt_tokens": p_tok, "completion_tokens": c_tok, "chunks": n_chunks,
        "ttft_s": round(t_first - t0, 4), "total_s": round(t_end - t0, 4),
        "prompt_eval_tps": prefill_tps, "gen_tps": gen_tps,
        "finish_reason": finish, "output_head": "".join(text)[:80],
    }


def r1(x):
    return None if x is None else round(x, 1)


def r2(x):
    return None if x is None else round(x, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="served model name")
    ap.add_argument("--hf-repo", default="")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--tiers", default=",".join(PROMPTS))
    ap.add_argument("--launch", default="A", help="tag for which fresh server launch this is (bimodal check)")
    ap.add_argument("--kv-dtype", default="bf16")
    ap.add_argument("--image", default="")
    ap.add_argument("--max-model-len", type=int, default=131072)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    tiers = [t for t in a.tiers.split(",") if t]
    nonce = "[bench session %s]\n" % uuid.uuid4().hex[:12]
    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "label": "dual-r9700-vllm-tp2", "model": a.model, "hf_repo": a.hf_repo,
        "backend": "vllm-rocm-tp2", "gen_tokens_requested": GEN_TOKENS,
        "launch": a.launch, "kv_cache_dtype": a.kv_dtype, "image": a.image,
        "max_model_len": a.max_model_len,
        "timing_method": "client-side SSE stream timing (t0=send, t_first=first content chunk, t_last=last content chunk)",
        "hardware": {
            "gpus": "2x Radeon AI PRO R9700 32GB (gfx1201), PCIe x8/x8",
            "cpu": "AMD Ryzen 9 9900X", "ram_gb": 32, "board": "Gigabyte AI TOP B850",
            "os": "Ubuntu Server, kernel 7.0.0-30-generic, in-kernel amdgpu",
            "topology": "tensor-parallel 2, NCCL_PROTO=Simple, TRITON_ATTN, AITER off",
        },
        "results": {},
    }
    print("warmup (%s) ..." % a.model, flush=True)
    w = one_request(a.host, a.port, a.model, nonce + PROMPTS["short"], a.timeout)
    print("  warmup: %d tok, gen %s tok/s" % (w["completion_tokens"], r1(w["gen_tps"])), flush=True)
    for tier in tiers:
        runs = []
        for i in range(a.runs):
            r = one_request(a.host, a.port, a.model, nonce + PROMPTS[tier], a.timeout)
            runs.append(r)
            print("  %-15s run%d: prompt=%6d comp=%3d prefill=%s decode=%s finish=%s" % (
                tier, i + 1, r["prompt_tokens"], r["completion_tokens"],
                r1(r["prompt_eval_tps"]), r2(r["gen_tps"]), r["finish_reason"]), flush=True)
        pe = [r["prompt_eval_tps"] for r in runs if r["prompt_eval_tps"]]
        ge = [r["gen_tps"] for r in runs if r["gen_tps"]]

        def st(xs):
            return statistics.stdev(xs) if len(xs) > 1 else 0.0

        res = {
            "runs": runs,
            "prompt_eval_tps_mean": statistics.mean(pe) if pe else None,
            "prompt_eval_tps_stdev": st(pe) if pe else None,
            "gen_tps_mean": statistics.mean(ge) if ge else None,
            "gen_tps_stdev": st(ge) if ge else None,
        }
        out["results"][tier] = res
        print("  %-15s MEAN prefill=%s decode=%s" % (
            tier, r1(res["prompt_eval_tps_mean"]), r2(res["gen_tps_mean"])), flush=True)
    path = a.out or "results_%s_%s_launch%s.json" % (a.model, a.kv_dtype, a.launch)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", path)


if __name__ == "__main__":
    main()
