"""
Offline feature extraction for Recurrent-JEPA fusion training.

For each LIBERO demo, saves one .pt file per sampled timestep:
  {
    "vj_obs":     (256, 2816) float32  -- vj_encoder spatial tokens
    "vj_pred":    (256, 2816) float32  -- predictor output (= obs at cold start t=0)
    "emb_tokens": (32, 2048)  float32  -- QwenVL embodied_action_tokens
    "actions":    (action_horizon, 7)  float32  -- delta_qpos target
    "states":     (8,)        float32  -- joint_states(7) + gripper(1)
  }

Usage:
  cd /home/choi/Recurrent-VLA-JEPA
  python scripts/extract_recurrent_tokens.py [--dataset all|libero_spatial|...]
"""

import argparse
import glob
import os
import sys

import h5py
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, "/home/choi/Recurrent-VLA-JEPA")

CKPT_PATH = "/media/choi/8AA890DCA890C859/vjepa2_baseline/checkpoints/VLA-JEPA/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt"
OUT_ROOT  = "/media/choi/8AA890DCA890C859/vjepa2_baseline/datasets/recurrent_jepa_tokens"

# (dataset_name, hdf5_dir, num_train_demos_per_task)
DATASETS = [
    ("libero_spatial", "/media/choi/8AA890DCA890C859/vjepa2_baseline/datasets/libero_spatial/libero_spatial", 40),
    ("libero_10",      "/home/choi/LGHA/LIBERO/libero/datasets/libero_10",     40),
    ("libero_goal",    "/home/choi/LGHA/LIBERO/libero/datasets/libero_goal",   40),
    ("libero_object",  "/home/choi/LGHA/LIBERO/libero/datasets/libero_object", 40),
]

IMAGE_SIZE = (224, 224)


def load_model(ckpt_path, device):
    from starVLA.model.framework.base_framework import baseframework
    model = baseframework.from_pretrained(ckpt_path)
    model = model.to(torch.bfloat16).to(device).eval()
    model.load_recurrent()
    print(f"  Loaded model on {device}")
    return model


@torch.no_grad()
def extract_demo(model, hdf5_path, demo_key, task_str, device, chunk_size, action_horizon):
    with h5py.File(hdf5_path, "r") as f:
        demo      = f["data"][demo_key]
        agentview = demo["obs"]["agentview_rgb"][:]    # (T, H, W, 3)
        wrist     = demo["obs"]["eye_in_hand_rgb"][:]  # (T, H, W, 3)
        actions   = demo["actions"][:]                 # (T, 7)
        joints    = demo["obs"]["joint_states"][:]     # (T, 7)
        gripper   = demo["obs"]["gripper_states"][:]   # (T, 2)

    T = len(actions)
    states_full = np.concatenate([joints, gripper[:, :1]], axis=-1).astype(np.float32)  # (T, 8)
    # pad actions so every timestep has a full horizon window
    padded_actions = np.concatenate(
        [actions, np.zeros((action_horizon - 1, 7), dtype=np.float32)], axis=0
    )

    model.reset_recurrent()
    samples = []

    for t in range(0, T - action_horizon + 1, chunk_size):
        agent_img = Image.fromarray(agentview[t]).resize(IMAGE_SIZE, Image.BILINEAR)
        wrist_img = Image.fromarray(wrist[t]).resize(IMAGE_SIZE, Image.BILINEAR)

        result = model.predict_action(
            batch_images=[[agent_img, wrist_img]],
            instructions=[task_str],
            state=[states_full[t].tolist()],
        )

        samples.append({
            "vj_obs":     model._vj_prev_obs[0].float().cpu().numpy(),    # (256, 2816)
            "vj_pred":    model._last_vj_pred[0].float().cpu().numpy(),   # (256, 2816)
            "emb_tokens": result["embodied_action_tokens"][0],             # (32, 2048)
            "actions":    padded_actions[t : t + action_horizon],          # (action_horizon, 7)
            "states":     states_full[t],                                  # (8,)
        })

    return samples


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(args.ckpt_path, device)

    action_horizon = model.config.framework.action_model.action_horizon
    chunk_size     = model.config.framework.action_model.future_action_window_size + 1
    print(f"  action_horizon={action_horizon}  chunk_size={chunk_size}")

    target_datasets = [d for d in DATASETS if args.dataset in ("all", d[0])]
    if not target_datasets:
        sys.exit(f"Unknown dataset '{args.dataset}'. Choices: all, " +
                 ", ".join(d[0] for d in DATASETS))

    for ds_name, hdf5_dir, train_demos in target_datasets:
        if not os.path.isdir(hdf5_dir):
            print(f"  SKIP {ds_name}: {hdf5_dir} not found")
            continue

        out_dir = os.path.join(args.out_dir, ds_name)
        for split in ("train", "val"):
            os.makedirs(os.path.join(out_dir, split), exist_ok=True)

        hdf5_files = sorted(glob.glob(os.path.join(hdf5_dir, "*.hdf5")))
        print(f"\n[{ds_name}]  {len(hdf5_files)} task files  →  {out_dir}")

        counters = {"train": 0, "val": 0}

        for hdf5_path in tqdm(hdf5_files, desc=ds_name):
            task_name = os.path.splitext(os.path.basename(hdf5_path))[0]
            task_str  = task_name.replace("_demo", "").replace("_", " ")

            with h5py.File(hdf5_path, "r") as f:
                demo_keys = sorted(f["data"].keys(),
                                   key=lambda x: int(x.replace("demo_", "")))

            for demo_key in demo_keys:
                demo_idx = int(demo_key.replace("demo_", ""))
                split    = "train" if demo_idx < train_demos else "val"

                samples = extract_demo(model, hdf5_path, demo_key, task_str,
                                       device, chunk_size, action_horizon)

                for sample in samples:
                    idx      = counters[split]
                    out_path = os.path.join(out_dir, split, f"{idx:06d}.pt")
                    torch.save(sample, out_path)
                    counters[split] += 1

        print(f"  [{ds_name}]  train={counters['train']}  val={counters['val']}")

    print("\nExtraction complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", default=CKPT_PATH)
    parser.add_argument("--out_dir",   default=OUT_ROOT)
    parser.add_argument("--dataset",   default="all",
                        help="all | libero_spatial | libero_10 | libero_goal | libero_object")
    args = parser.parse_args()
    main(args)
