#!/bin/bash
# Recurrent-JEPA eval script (Phase 1: sanity check)
# Logs pred/obs cosine_sim each step to verify the world model predictor
# is making meaningful predictions at inference time.
#
# Usage: bash eval_libero_recurrent.sh [task_suite] [num_trials]
# Example: bash eval_libero_recurrent.sh libero_spatial 50

export PYTHONDONTWRITEBYTECODE=1

_NVIDIA_LIBS=/home/choi/miniconda3/envs/vjepa2/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH=$(find $_NVIDIA_LIBS -name "lib" -type d | tr '\n' ':')$LD_LIBRARY_PATH

export LIBERO_HOME=/home/choi/LGHA/LIBERO
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME}
export PYTHONPATH=$(pwd):${PYTHONPATH}
export sim_python=/home/choi/miniconda3/envs/vla_jepa/bin/python

your_ckpt=/media/choi/8AA890DCA890C859/vjepa2_baseline/checkpoints/VLA-JEPA/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt
task_suite_name=${1:-libero_spatial}
num_trials_per_task=${2:-50}
port=15087
with_state="true"

folder_name="recurrent_phase1"
video_out_path="results/${task_suite_name}/${folder_name}"
mkdir -p ${video_out_path}

echo "Task suite : ${task_suite_name}"
echo "Checkpoint : ${your_ckpt}"
echo "Output     : ${video_out_path}"
echo "Mode       : Recurrent-JEPA Phase 1 (cos_sim logging)"

if [ ! -f "${your_ckpt}" ]; then
    echo "ERROR: Checkpoint not found: ${your_ckpt}"
    exit 1
fi

fuser -k ${port}/tcp 2>/dev/null || true; sleep 1

rm -f /tmp/vla_server_recurrent.log
${sim_python} ./deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16 \
    --cuda 0 \
    --recurrent > /tmp/vla_server_recurrent.log 2>&1 &
SERVER_PID=$!
echo "Policy server PID: ${SERVER_PID}"

echo "Waiting for server to be ready..."
elapsed=0
until grep -q "server listening" /tmp/vla_server_recurrent.log 2>/dev/null; do
    sleep 2; elapsed=$((elapsed + 2))
    if [ $elapsed -ge 120 ]; then
        echo "ERROR: Server failed to start within 120s. Log:"
        cat /tmp/vla_server_recurrent.log
        kill $SERVER_PID 2>/dev/null
        exit 1
    fi
done
echo "Server is up (Recurrent-JEPA enabled)."

${sim_python} ./examples/LIBERO/eval_libero.py \
    --args.pretrained-path ${your_ckpt} \
    --args.host "127.0.0.1" \
    --args.port ${port} \
    --args.task-suite-name "${task_suite_name}" \
    --args.num-trials-per-task ${num_trials_per_task} \
    --args.video-out-path "${video_out_path}" \
    --args.with_state "${with_state}" \
    2>&1 | tee "${video_out_path}/eval.log"

kill ${SERVER_PID} 2>/dev/null
echo ""
echo "=== Phase 1 cos_sim summary ==="
grep "Recurrent-JEPA" /tmp/vla_server_recurrent.log | \
    awk -F'= ' '{print $2}' | \
    awk 'BEGIN{s=0;n=0} {s+=$1; n++} END{printf "steps=%d  mean_cos_sim=%.4f\n", n, s/n}'
echo "Full server log: /tmp/vla_server_recurrent.log"
echo "Done. Results in ${video_out_path}/eval.log"
