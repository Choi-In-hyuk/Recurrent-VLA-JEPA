#!/bin/bash
# Recurrent-JEPA Phase 2: Extract → Train → Eval (all suites)
#
# Usage:
#   bash scripts/run_recurrent_phase2.sh [task_suite] [num_trials]
#
# Examples:
#   bash scripts/run_recurrent_phase2.sh                      # all datasets, libero_spatial eval
#   bash scripts/run_recurrent_phase2.sh libero_10 50
#   bash scripts/run_recurrent_phase2.sh all libero_spatial   # extract all, eval spatial

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
PHASE2_CKPT=checkpoints/recurrent_jepa_ft/best.pt
TASK_SUITE=${1:-libero_spatial}
NUM_TRIALS=${2:-50}
PORT=15088

echo "============================================================"
echo " Recurrent-JEPA Phase 2 Pipeline"
echo "  Task suite : ${TASK_SUITE}"
echo "  Trials     : ${NUM_TRIALS}"
echo "============================================================"

# ── Step 1: Feature extraction ────────────────────────────────
echo ""
TOKEN_ROOT=/media/choi/8AA890DCA890C859/vjepa2_baseline/datasets/recurrent_jepa_tokens
N_EXISTING=$(find ${TOKEN_ROOT} -name "*.pt" 2>/dev/null | wc -l)

if [ "${N_EXISTING}" -gt 100 ]; then
    echo "[1/3] Token files already exist (${N_EXISTING} files). Skipping extraction."
else
    echo "[1/3] Extracting offline features (all LIBERO datasets)..."
    ${SIM_PYTHON} scripts/extract_recurrent_tokens.py --dataset all
    echo "Extraction complete."
fi

# ── Step 2: Train fusion module ───────────────────────────────
echo ""
echo "[2/3] Training fusion + vj_to_dit + DiT..."
${SIM_PYTHON} scripts/train_recurrent_fusion.py \
    --epochs 50 \
    --max_steps 20000 \
    --batch_size 16 \
    --lr_fusion 1e-4 \
    --lr_vj_to_dit 1e-4 \
    --lr_action 1e-4
echo "Training complete. Best checkpoint: ${PHASE2_CKPT}"

# ── Step 3: Evaluation ────────────────────────────────────────
echo ""
echo "[3/3] Evaluating on ${TASK_SUITE} (${NUM_TRIALS} trials/task)..."

VIDEO_OUT=results/${TASK_SUITE}/recurrent_phase2
mkdir -p ${VIDEO_OUT}

if [ ! -f "${PHASE2_CKPT}" ]; then
    echo "ERROR: Phase 2 checkpoint not found: ${PHASE2_CKPT}"
    exit 1
fi

fuser -k ${PORT}/tcp 2>/dev/null || true; sleep 1

rm -f /tmp/vla_server_phase2.log
${SIM_PYTHON} ./deployment/model_server/server_policy.py \
    --ckpt_path   ${BASE_CKPT} \
    --recurrent_ckpt ${PHASE2_CKPT} \
    --port        ${PORT} \
    --use_bf16 \
    --cuda        0 > /tmp/vla_server_phase2.log 2>&1 &
SERVER_PID=$!
echo "  Policy server PID: ${SERVER_PID}"

echo "  Waiting for server..."
elapsed=0
until grep -q "server listening\|server running" /tmp/vla_server_phase2.log 2>/dev/null; do
    sleep 2; elapsed=$((elapsed + 2))
    if [ $elapsed -ge 120 ]; then
        echo "ERROR: Server failed to start. Log:"
        cat /tmp/vla_server_phase2.log
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
echo " Phase 2 Pipeline Complete"
echo "  Checkpoint : ${PHASE2_CKPT}"
echo "  Results    : ${VIDEO_OUT}/eval.log"
echo "============================================================"
