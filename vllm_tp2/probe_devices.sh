#!/usr/bin/env bash
# probe_devices.sh - list ROCm devices as seen inside the vLLM image, to pick HIP_VISIBLE_DEVICES.
IMAGE="${IMAGE:-vllm/vllm-openai-rocm:latest}"
docker run --rm --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render --security-opt seccomp=unconfined \
  --entrypoint bash "$IMAGE" -c 'rocminfo | grep -E "^\s+Name:\s+gfx" ; echo ---; rocm-smi --showproductname 2>/dev/null | grep -iE "^GPU|card series|gfx" '
