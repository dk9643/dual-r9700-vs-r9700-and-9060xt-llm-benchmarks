#!/usr/bin/env python3
"""Compare before/after GPU-upgrade benchmark runs.

Usage:
    python3 compare.py --before "r9700+9060xt" --after "dual-r9700"
    python3 compare.py --before "r9700+9060xt" --after "dual-r9700" --markdown

Reads ollama_bench_results.json (override with --file), picks the most
recent record per (model, label), and prints per-tier prefill/decode
comparisons with speedup factors. --markdown emits README-ready tables.
"""

import argparse
import json


def latest_per_model(records, label):
    out = {}
    for r in records:
        if r.get("label") == label:
            out[r["model"]] = r  # file is append-ordered; last record wins
    return out


def fmt(v):
    return "-" if v is None else f"{v:g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="ollama_bench_results.json")
    ap.add_argument("--before", required=True, help="label of the baseline run")
    ap.add_argument("--after", required=True, help="label of the comparison run")
    ap.add_argument("--markdown", action="store_true", help="emit markdown tables")
    args = ap.parse_args()

    with open(args.file) as f:
        history = json.load(f)

    before = latest_per_model(history, args.before)
    after = latest_per_model(history, args.after)

    models = [m for m in before if m in after]
    for m in sorted(set(before) | set(after)):
        if m not in models:
            missing = args.after if m in before else args.before
            print(f"note: {m} has no '{missing}' record; skipped")

    for m in models:
        print(f"\n## {m}")
        if args.markdown:
            print("| tier | prefill before | prefill after | speedup | decode before | decode after | speedup |")
            print("|---|---|---|---|---|---|---|")
        else:
            print(f"{'tier':<15} {'prefill before':>14} {'prefill after':>14} {'x':>6} "
                  f"{'decode before':>14} {'decode after':>13} {'x':>6}")
        for tier, b in before[m]["results"].items():
            a = after[m]["results"].get(tier)
            if not a:
                continue
            pb, pa = b["prompt_eval_tps_mean"], a["prompt_eval_tps_mean"]
            gb, ga = b["gen_tps_mean"], a["gen_tps_mean"]
            ps = f"{pa / pb:.2f}" if pb and pa else "-"
            gs = f"{ga / gb:.2f}" if gb and ga else "-"
            if args.markdown:
                print(f"| {tier} | {fmt(pb)} | {fmt(pa)} | {ps} | {fmt(gb)} | {fmt(ga)} | {gs} |")
            else:
                print(f"{tier:<15} {fmt(pb):>14} {fmt(pa):>14} {ps:>6} "
                      f"{fmt(gb):>14} {fmt(ga):>13} {gs:>6}")

    if not models:
        print("no models present under both labels — nothing to compare")


if __name__ == "__main__":
    main()
