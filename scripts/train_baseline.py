#!/usr/bin/env python3
"""train_baseline.py — Train only the U-Net baseline."""
import sys, os, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.utils.io_utils import load_config, set_seed, get_device, ensure_dirs
from src.utils.visualization import plot_training_curves
from src.data.dataset import get_datasets, make_dataloaders
from src.models.unet import build_unet, build_loss
from src.train.trainer import BaselineTrainer, build_optimizer_scheduler


def main(cfg_path: str):
    cfg = load_config(cfg_path)
    set_seed(cfg.get("seed", 42))
    device = get_device(cfg.get("device", "auto"))
    ensure_dirs(cfg)

    train_ds, val_ds, _ = get_datasets(cfg)
    train_loader, val_loader, _ = make_dataloaders(cfg, train_ds, val_ds, val_ds)

    model     = build_unet(cfg)
    criterion = build_loss(cfg)
    optimizer, scheduler = build_optimizer_scheduler(model, cfg)

    trainer = BaselineTrainer(model, criterion, optimizer, scheduler, device, cfg)
    history = trainer.train(train_loader, val_loader, epochs=cfg["training"]["epochs"])

    plot_training_curves(history, cfg["output"]["figures_dir"])
    print("Baseline training done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
