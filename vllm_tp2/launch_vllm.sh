#!/usr/bin/env bash
# launch_vllm.sh <hf_repo> <served_name> [extra vllm args...]
# Starts container "vllm" with the TP=2 recipe. Set HIP_DEVS (e.g. "0,1") and optionally IMAGE, MAX_LEN.
set -u
cd "$(dirname "$0")"
REPO="$1"; NAME="$2"; shift 2
IMAGE="${IMAGE:-vllm/vllm-openai-rocm:latest}"
HIP_DEVS="${HIP_DEVS:?set HIP_DEVS to the two gfx1201 indices, e.g. HIP_DEVS=0,1}"
MAX_LEN="${MAX_LEN:-131072}"
MODELS_DIR="${MODELS_DIR:?set MODELS_DIR to the directory holding your Hugging Face cache (mounted as /models)}"
[ -d "$MODELS_DIR" ] || { echo "MODELS_DIR $MODELS_DIR does not exist"; exit 1; }
docker rm -f vllm >/dev/null 2>&1
docker run -d --name vllm \
  --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render \
  --ipc=host --shm-size 16g \
  --security-opt seccomp=unconfined --cap-add=SYS_PTRACE \
  -e HIP_VISIBLE_DEVICES="$HIP_DEVS" \
  -e NCCL_PROTO=Simple \
  -e NCCL_P2P_DISABLE=1 \
  -e VLLM_ROCM_USE_AITER=0 \
  -e HF_HOME=/models \
  ${HF_TOKEN:+-e HF_TOKEN=$HF_TOKEN} \
  ${EXTRA_ENV:-} \
  -v "$MODELS_DIR":/models \
  -p 8000:8000 \
  "$IMAGE" \
  "$REPO" \
  --tensor-parallel-size 2 \
  --distributed-executor-backend mp \
  --attention-backend TRITON_ATTN \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization 0.90 \
  --no-enable-prefix-caching \
  --served-model-name "$NAME" \
  --port 8000 "$@"
echo "launched: image=$IMAGE model=$REPO name=$NAME devs=$HIP_DEVS max_len=$MAX_LEN extra=$*"
