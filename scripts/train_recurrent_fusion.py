"""
Standalone training script for Recurrent-JEPA Phase 2.

Trains only: LearnedGatingFusion + VJtoDiTProjection + action_model (DiT)
Frozen:      vj_encoder, vj_predictor, qwen_vl_interface

Reads pre-extracted .pt files from scripts/extract_recurrent_tokens.py.

Saved checkpoint contains full model state_dict (base + new modules),
ready to be loaded by server_policy.py --recurrent_ckpt.

Usage:
  cd /home/choi/Recurrent-VLA-JEPA
  python scripts/train_recurrent_fusion.py
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, "/home/choi/Recurrent-VLA-JEPA")

from starVLA.dataloader.recurrent_token_dataset import RecurrentTokenDataset
from starVLA.model.modules.fusion.gating import LearnedGatingFusion
from starVLA.model.modules.projector.vj_to_dit import VJtoDiTProjection

CKPT_PATH  = "/media/choi/8AA890DCA890C859/vjepa2_baseline/checkpoints/VLA-JEPA/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt"
TOKEN_ROOT = "/media/choi/8AA890DCA890C859/vjepa2_baseline/datasets/recurrent_jepa_tokens"
SAVE_DIR   = "checkpoints/recurrent_jepa_ft"

DATA_DIRS = [
    os.path.join(TOKEN_ROOT, "libero_spatial"),
    os.path.join(TOKEN_ROOT, "libero_10"),
    os.path.join(TOKEN_ROOT, "libero_goal"),
    os.path.join(TOKEN_ROOT, "libero_object"),
]


def load_model(ckpt_path, device):
    from starVLA.model.framework.base_framework import baseframework
    model = baseframework.from_pretrained(ckpt_path)

    # Base checkpoint has no recurrent modules — inject them manually
    vj_dim  = model.vj_encoder.config.hidden_size * 2        # 1408 * 2 = 2816
    dit_dim = model.qwen_vl_interface.model.config.hidden_size  # 2048
    model.fusion    = LearnedGatingFusion(vj_dim, use_cosine=True)
    model.vj_to_dit = VJtoDiTProjection(vj_dim, dit_dim, n_cond_tokens=8)
    model._recurrent_training = True

    model = model.to(device)

    # Freeze everything except fusion, vj_to_dit, action_model
    for name, param in model.named_parameters():
        trainable = any(name.startswith(p)
                        for p in ("fusion.", "vj_to_dit.", "action_model."))
        param.requires_grad = trainable

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {n_train:,} / {n_total:,} params")
    return model


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(args.ckpt_path, device)
    model.train()

    data_dirs = [d for d in DATA_DIRS if os.path.isdir(d)]
    if not data_dirs:
        sys.exit(f"No token directories found under {TOKEN_ROOT}. "
                 "Run scripts/extract_recurrent_tokens.py first.")

    train_ds = RecurrentTokenDataset(data_dirs, split="train", seed=42)
    val_ds   = RecurrentTokenDataset(data_dirs, split="val",   seed=42)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    optimizer = torch.optim.AdamW([
        {"params": list(model.fusion.parameters()),      "lr": args.lr_fusion},
        {"params": list(model.vj_to_dit.parameters()),  "lr": args.lr_vj_to_dit},
        {"params": list(model.action_model.parameters()),"lr": args.lr_action},
    ], lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-8)

    os.makedirs(args.save_dir, exist_ok=True)
    repeated   = model.config.framework.action_model.repeated_diffusion_steps
    global_step = 0
    best_val    = float("inf")

    for epoch in range(args.epochs):
        model.train()
        total_loss = n_batches = 0

        for batch in train_loader:
            vj_obs     = batch["vj_obs"].to(device)      # (B, 256, 2816)
            vj_pred    = batch["vj_pred"].to(device)     # (B, 256, 2816)
            emb_tokens = batch["emb_tokens"].to(device)  # (B, 32, 2048)
            actions    = batch["actions"].to(device)     # (B, action_horizon, 7)
            states     = batch["states"].to(device)      # (B, 8)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                fused  = model.fusion(vj_obs.float(), vj_pred.float())   # (B, 256, 2816)
                cond_t = model.vj_to_dit(fused)                          # (B, 8, 2048)
                dit_in = torch.cat([emb_tokens, cond_t], dim=1)          # (B, 40, 2048)

                dit_in_rep  = dit_in.repeat(repeated, 1, 1)
                actions_rep = actions.repeat(repeated, 1, 1)
                states_rep  = states.repeat(repeated, 1)

                loss = model.action_model(dit_in_rep, actions_rep, states_rep)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()

            total_loss  += loss.item()
            n_batches   += 1
            global_step += 1

            if global_step % args.log_freq == 0:
                print(f"  step={global_step:5d}  loss={loss.item():.4f}")

            if global_step >= args.max_steps:
                break

        avg_loss = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch+1:3d}  train_loss={avg_loss:.4f}")

        # --- validation ---
        model.eval()
        val_loss = val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                vj_obs     = batch["vj_obs"].to(device)
                vj_pred    = batch["vj_pred"].to(device)
                emb_tokens = batch["emb_tokens"].to(device)
                actions    = batch["actions"].to(device)
                states     = batch["states"].to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    fused  = model.fusion(vj_obs.float(), vj_pred.float())
                    cond_t = model.vj_to_dit(fused)
                    dit_in = torch.cat([emb_tokens, cond_t], dim=1)
                    loss   = model.action_model(dit_in, actions, states)
                val_loss += loss.item()
                val_n    += 1

        avg_val = val_loss / max(val_n, 1)
        print(f"           val_loss={avg_val:.4f}")

        # save checkpoint (full state_dict so server_policy.py can load it directly)
        ckpt_path = os.path.join(args.save_dir, f"epoch{epoch+1:03d}.pt")
        torch.save(model.state_dict(), ckpt_path)
        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best.pt"))
            print(f"           *** best val={best_val:.4f} → best.pt")

        if global_step >= args.max_steps:
            break

    print(f"\nDone. Checkpoints → {args.save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path",    default=CKPT_PATH)
    parser.add_argument("--save_dir",     default=SAVE_DIR)
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--max_steps",    type=int,   default=20000)
    parser.add_argument("--batch_size",   type=int,   default=16)
    parser.add_argument("--lr",           type=float, default=1e-5)
    parser.add_argument("--lr_fusion",    type=float, default=1e-4)
    parser.add_argument("--lr_vj_to_dit", type=float, default=1e-4)
    parser.add_argument("--lr_action",    type=float, default=1e-4)
    parser.add_argument("--log_freq",     type=int,   default=10)
    args = parser.parse_args()
    train(args)
