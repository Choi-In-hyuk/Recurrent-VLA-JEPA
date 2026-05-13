# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# Implemented by [Jinhui YE / HKUST University] in [2025].

import logging
import socket
import argparse
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from starVLA.model.framework.base_framework import baseframework
import torch, os


def main(args) -> None:
    # Example usage:
    # policy = YourPolicyClass()  # Replace with your actual policy class
    # server = WebsocketPolicyServer(policy, host="localhost", port=10091)
    # server.serve_forever()

    vla = baseframework.from_pretrained( # TODO should auto detect framework from model path
        args.ckpt_path,
    )

    device = torch.device(f"cuda:{str(args.cuda)}")

    if args.use_bf16: # False
        vla = vla.to(torch.bfloat16)
    vla = vla.to(device).eval()

    if args.lds_path:
        vla.load_kf(args.lds_path, q_noise=args.kf_q, r_noise=args.kf_r)
        logging.info("KF enabled: %s  q=%.3f  r=%.3f", args.lds_path, args.kf_q, args.kf_r)
    elif args.ema_alpha is not None:
        vla.load_ema(args.ema_alpha)
        logging.info("EMA enabled: alpha=%.3f", args.ema_alpha)
    elif args.recurrent_ckpt:
        # Phase 2: load fine-tuned fusion + vj_to_dit + action_model weights
        from starVLA.model.modules.fusion.gating import LearnedGatingFusion
        from starVLA.model.modules.projector.vj_to_dit import VJtoDiTProjection
        vj_dim  = vla.vj_encoder.config.hidden_size * 2
        dit_dim = vla.qwen_vl_interface.model.config.hidden_size
        vla.fusion    = LearnedGatingFusion(vj_dim, use_cosine=True).to(device)
        vla.vj_to_dit = VJtoDiTProjection(vj_dim, dit_dim, n_cond_tokens=8).to(device)
        vla._recurrent_training = True
        vla._recurrent_enabled  = True
        vla._vj_prev_obs   = None
        vla._prev_action_t = None
        vla._last_vj_pred  = None
        phase2_sd = torch.load(args.recurrent_ckpt, map_location=device, weights_only=False)
        missing, unexpected = vla.load_state_dict(phase2_sd, strict=False)
        logging.info("Recurrent-JEPA Phase 2 loaded: %s  missing=%d  unexpected=%d",
                     args.recurrent_ckpt, len(missing), len(unexpected))
    elif args.recurrent:
        vla.load_recurrent()
        logging.info("Recurrent-JEPA enabled (Phase 1 sanity check)")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    # start websocket server
    server = WebsocketPolicyServer(
        policy=vla,
        host="0.0.0.0",
        port=args.port,
        metadata={"env": "simpler_env"},
    )
    logging.info("server running ...")
    server.serve_forever()


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--cuda", default=0)
    parser.add_argument("--lds_path", type=str, default=None, help="Path to LDS .npz for KF filtering")
    parser.add_argument("--kf_q", type=float, default=0.1)
    parser.add_argument("--kf_r", type=float, default=5.0)
    parser.add_argument("--ema_alpha", type=float, default=None, help="EMA smoothing alpha (0~1); overridden by --lds_path")
    parser.add_argument("--recurrent", action="store_true", help="Enable Recurrent-JEPA inference (Phase 1: cos_sim logging)")
    parser.add_argument("--recurrent_ckpt", type=str, default=None, help="Phase 2 fine-tuned checkpoint (fusion+vj_to_dit+action_model)")
    return parser


def start_debugpy_once():
    """start debugpy once"""
    import debugpy
    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10091))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10091 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parser = build_argparser()
    args = parser.parse_args()
    if os.getenv("DEBUG", False):
        print("🔍 DEBUGPY is enabled")
        start_debugpy_once()
    main(args)
