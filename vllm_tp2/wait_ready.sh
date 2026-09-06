#!/usr/bin/env bash
# wait_ready.sh [timeout_s] - poll until vLLM answers /v1/models or the container dies.
T="${1:-3600}"; start=$(date +%s)
while true; do
  if curl -sf http://localhost:8000/v1/models >/dev/null 2>&1; then echo READY; exit 0; fi
  if ! docker ps --format '{{.Names}}' | grep -qx vllm; then echo "CONTAINER_EXITED"; docker logs --tail 60 vllm; exit 1; fi
  if [ $(( $(date +%s) - start )) -gt "$T" ]; then echo TIMEOUT; exit 2; fi
  sleep 5
done
