#!/usr/bin/env bash
# run_cycle.sh <hf_repo> <served_name> <launch_tag> <log_prefix> [extra vllm args...]
# Fresh container -> wait ready -> benchmark -> docker rm. Writes <log_prefix>_<tag>.log and <log_prefix>_<tag>.done
cd "$(dirname "$0")"
REPO="$1"; NAME="$2"; TAG="$3"; PFX="$4"; shift 4
LOG="${PFX}_${TAG}.log"; DONE="${PFX}_${TAG}.done"
rm -f "$DONE"
{
  echo "=== cycle $TAG start $(date +%T)"
  HIP_DEVS="${HIP_DEVS:-0,1}" ./launch_vllm.sh "$REPO" "$NAME" "$@" || { echo "LAUNCH_FAILED"; echo EXIT=1 > "$DONE"; exit 1; }
  if ! ./wait_ready.sh 3600; then
    echo "NOT_READY"; docker logs vllm > "${PFX}_${TAG}_container.log" 2>&1
    echo EXIT=2 > "$DONE"; exit 2
  fi
  docker logs vllm 2>&1 | grep -E "GPU KV cache size|Model loading took|init engine" | cut -c1-200
  echo "=== bench start $(date +%T)"
  python3 bench_vllm.py --model "$NAME" --hf-repo "$REPO" --launch "$TAG" --image "${IMAGE:-vllm/vllm-openai-rocm:v0.28.0}" ${BENCH_ARGS:-}
  rc=$?
  docker logs vllm > "${PFX}_${TAG}_container.log" 2>&1
  docker rm -f vllm >/dev/null 2>&1
  echo "=== cycle $TAG end $(date +%T) rc=$rc"
  echo EXIT=$rc > "$DONE"
} > "$LOG" 2>&1
