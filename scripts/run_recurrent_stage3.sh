#!/bin/bash
# Stage 3: QwenVL correction-token injection
#
# Architecture:
#   correction_tokens (Δz injected) inserted between action_tokens and
#   embodied_action_tokens in QwenVL sequence → embodied_action_tokens attend
#   to world-model prediction error → better action generation
#
# Pipeline:
#   [1/3] Token extraction (with Stage 3 metadata)
#   [2/3] Train CorrectionProjector
#   [3/3] Evaluate
#
# Usage:
#   bash scripts/run_recurrent_stage3.sh [task_suite] [num_trials]

set -e

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$(pwd):${PYTHONPATH}

_NVIDIA_LIBS=/home/choi/miniconda3/envs/vjepa2/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH=$(find $_NVIDIA_LIBS -name "lib" -type d | tr '\n' ':')$LD_LIBRARY_PATH

export LIBERO_HOME=/home/choi/LGHA/LIBERO
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME}

SIM_PYTHON=/home/choi/miniconda3/envs/vla_jepa/bin/python
BASE_CKPT=/media/choi/8AA890DCA890C859/vjepa2_baseline/checkpoints/VLA-JEPA/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt
STAGE3_CKPT=checkpoints/recurrent_stage3/best.pt
TASK_SUITE=${1:-libero_spatial}
NUM_TRIALS=${2:-50}
PORT=15089

echo "============================================================"
echo " Stage 3 Pipeline: QwenVL Correction-Token Injection"
echo "  Task suite : ${TASK_SUITE}"
echo "  Trials     : ${NUM_TRIALS}"
echo "============================================================"

# ── Step 1: Token extraction (skip if metadata already present) ──
echo ""
TOKEN_ROOT=/media/choi/8AA890DCA890C859/vjepa2_baseline/datasets/recurrent_jepa_tokens
SAMPLE_PT=$(find ${TOKEN_ROOT} -name "*.pt" 2>/dev/null | head -1)

if [ -n "${SAMPLE_PT}" ]; then
    HAS_META=$(${SIM_PYTHON} -c "
import torch
d = torch.load('${SAMPLE_PT}', weights_only=False)
print('yes' if 'hdf5_path' in d else 'no')
" 2>/dev/null)
else
    HAS_META="no"
fi

if [ "${HAS_META}" = "yes" ]; then
    echo "[1/3] Stage 3 metadata found. Skipping extraction."
else
    echo "[1/3] Extracting tokens with Stage 3 metadata..."
    ${SIM_PYTHON} scripts/extract_recurrent_tokens.py --dataset all
    echo "Extraction complete."
fi

# ── Step 2: Train CorrectionProjector ───────────────────────────
echo ""
echo "[2/3] Training CorrectionProjector + QwenVL LoRA..."
${SIM_PYTHON} -u scripts/train_recurrent_stage3.py \
    --epochs 30 \
    --max_steps 10000 \
    --batch_size 4 \
    --lr 1e-4 \
    --lora_r 16 \
    --lora_alpha 32
echo "Training complete. Best checkpoint: ${STAGE3_CKPT}"

# ── Step 3: Evaluation ───────────────────────────────────────────
echo ""
echo "[3/3] Evaluating on ${TASK_SUITE} (${NUM_TRIALS} trials/task)..."

VIDEO_OUT=results/${TASK_SUITE}/recurrent_stage3
mkdir -p ${VIDEO_OUT}

if [ ! -f "${STAGE3_CKPT}" ]; then
    echo "ERROR: Stage 3 checkpoint not found: ${STAGE3_CKPT}"
    exit 1
fi

fuser -k ${PORT}/tcp 2>/dev/null || true; sleep 1

rm -f /tmp/vla_server_stage3.log
${SIM_PYTHON} ./deployment/model_server/server_policy.py \
    --ckpt_path    ${BASE_CKPT} \
    --stage3_ckpt  ${STAGE3_CKPT} \
    --port         ${PORT} \
    --use_bf16 \
    --cuda 0 > /tmp/vla_server_stage3.log 2>&1 &
SERVER_PID=$!
echo "  Policy server PID: ${SERVER_PID}"

echo "  Waiting for server..."
elapsed=0
until grep -q "server running" /tmp/vla_server_stage3.log 2>/dev/null; do
    sleep 2; elapsed=$((elapsed + 2))
    if [ $elapsed -ge 120 ]; then
        echo "ERROR: Server failed to start. Log:"
        cat /tmp/vla_server_stage3.log
        kill $SERVER_PID 2>/dev/null
        exit 1
    fi
done
echo "  Server ready."

${SIM_PYTHON} ./examples/LIBERO/eval_libero.py \
    --args.pretrained-path ${BASE_CKPT} \
    --args.host "127.0.0.1" \
    --args.port ${PORT} \
    --args.task-suite-name "${TASK_SUITE}" \
    --args.num-trials-per-task ${NUM_TRIALS} \
    --args.video-out-path "${VIDEO_OUT}" \
    --args.with_state "true" \
    2>&1 | tee "${VIDEO_OUT}/eval.log"

kill ${SERVER_PID} 2>/dev/null

echo ""
echo "============================================================"
echo " Stage 3 Pipeline Complete"
echo "  Checkpoint : ${STAGE3_CKPT}"
echo "  Results    : ${VIDEO_OUT}/eval.log"
echo "============================================================"
