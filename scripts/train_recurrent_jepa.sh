#!/bin/bash
# Phase 2: Recurrent JEPA VLA fine-tuning
# Trains: LearnedGatingFusion + VJtoDiTProjection + DiT (action_model)
# Frozen: QwenVL + vj_encoder + vj_predictor
#
# Usage: bash scripts/train_recurrent_jepa.sh [num_gpus]

export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000
export TMPDIR=/home/dataset-local/tmp
export FFMPEG_THREADS=1
export OMP_NUM_THREADS=1
export WANDB_MODE=disabled

NUM_GPUS=${1:-1}

echo "Starting Recurrent-JEPA Phase 2 training"
echo "  GPUs     : ${NUM_GPUS}"
echo "  Config   : scripts/config/recurrent_jepa_ft.yaml"
echo "  Trains   : fusion + vj_to_dit + action_model (DiT)"
echo "  Frozen   : qwen_vl_interface + vj_encoder + vj_predictor"

accelerate launch \
  --config_file ./starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${NUM_GPUS} \
  ./starVLA/training/train_starvla.py \
  --config_yaml ./scripts/config/recurrent_jepa_ft.yaml
