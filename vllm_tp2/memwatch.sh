#!/usr/bin/env bash
# memwatch.sh - kill vLLM if MemAvailable drops below 6 GiB. Log next to this script.
cd "$(dirname "$0")"
while true; do
  avail=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
  echo "$(date +%T) MemAvailable_kB=$avail" >> memwatch.log
  if [ "$avail" -lt 6291456 ]; then
    echo "$(date +%T) LOW MEMORY - killing vllm" >> memwatch.log
    docker kill vllm 2>/dev/null; pkill -9 -f 'vllm' 2>/dev/null
  fi
  sleep 2
done
