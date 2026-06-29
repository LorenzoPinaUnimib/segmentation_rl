#!/usr/bin/env python3
"""train_rl.py — Train only the RL refinement agent (requires baseline checkpoint)."""
import sys, os, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
from src.utils.io_utils import load_config, set_seed, get_device, ensure_dirs
from src.utils.visualization import plot_rl_curves
from src.data.dataset import get_datasets, make_dataloaders
from src.models.unet import build_unet
from src.rl.environment import MaskRefinementEnv
from src.rl.agent import DQNAgent
from src.train.rl_trainer import RLTrainer


def main(cfg_path: str):
    cfg = load_config(cfg_path)
    set_seed(cfg.get("seed", 42))
    device = get_device(cfg.get("device", "auto"))
    ensure_dirs(cfg)

    train_ds, val_ds, _ = get_datasets(cfg)
    train_loader, val_loader, _ = make_dataloaders(cfg, train_ds, val_ds, val_ds)

    model = build_unet(cfg)
    ckpt_path = os.path.join(cfg["output"]["checkpoint_dir"], "best_model.pth")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[rl] Loaded baseline from {ckpt_path}")
    else:
        print("[rl] WARNING: No baseline checkpoint found. Using random model.")

    env   = MaskRefinementEnv(cfg)
    agent = DQNAgent(cfg, device)

    trainer = RLTrainer(model, agent, env, train_loader, val_loader, device, cfg)
    history = trainer.train(eval_every=max(1, cfg["rl"]["episodes"] // 15))

    plot_rl_curves(history, cfg["output"]["figures_dir"])
    print("RL training done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
