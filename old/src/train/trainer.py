"""
trainer.py — Supervised training loop for U-Net baseline.
Features: early stopping, grad clipping, best model checkpoint, history logging.
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
from tqdm import tqdm

from ..eval.metrics import batch_dice, batch_iou


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best      = None
        self.stop      = False

    def __call__(self, val_loss: float) -> bool:
        if self.best is None or val_loss < self.best - self.min_delta:
            self.best    = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


class BaselineTrainer:
    """Trains a segmentation model with supervised learning."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        device: torch.device,
        cfg: dict,
    ):
        self.model     = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device    = device
        self.cfg       = cfg

        self.checkpoint_dir = cfg["output"]["checkpoint_dir"]
        self.grad_clip      = cfg["training"].get("grad_clip", 1.0)
        self.patience       = cfg["training"].get("early_stopping_patience", 10)

        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [],
            "train_dice": [], "val_dice": [],
            "train_iou":  [], "val_iou":  [],
        }
        self.best_val_loss = float("inf")
        self.best_val_dice = 0.0
        self.early_stop    = EarlyStopping(patience=self.patience)

    # ── Single epoch ──────────────────────────────────────────────────────────

    def _run_epoch(self, loader: DataLoader, train: bool) -> Dict[str, float]:
        self.model.train(train)
        total_loss = dice_sum = iou_sum = 0.0
        n = 0

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch in tqdm(loader, leave=False,
                              desc="  train" if train else "  val"):
                images  = batch["image"].to(self.device)
                targets = batch["mask"].to(self.device)

                logits = self.model(images)

                # Handle shape mismatch (should not happen but be safe)
                if logits.shape != targets.shape:
                    targets = targets[:, :logits.size(1)]

                loss = self.criterion(logits, targets)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.grad_clip:
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.grad_clip
                        )
                    self.optimizer.step()

                bs = images.size(0)
                total_loss += loss.item() * bs
                dice_sum   += batch_dice(logits.detach(), targets).item() * bs
                iou_sum    += batch_iou(logits.detach(), targets).item() * bs
                n += bs

        return {
            "loss": total_loss / n,
            "dice": dice_sum   / n,
            "iou":  iou_sum    / n,
        }

    # ── Full training ─────────────────────────────────────────────────────────

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
    ) -> Dict[str, List[float]]:

        print(f"\n{'='*60}")
        print(f" Baseline training for {epochs} epochs")
        print(f"{'='*60}")

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            train_stats = self._run_epoch(train_loader, train=True)
            val_stats   = self._run_epoch(val_loader,   train=False)

            if self.scheduler is not None:
                if isinstance(self.scheduler,
                              torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_stats["loss"])
                else:
                    self.scheduler.step()

            # Record history
            self.history["train_loss"].append(train_stats["loss"])
            self.history["val_loss"].append(val_stats["loss"])
            self.history["train_dice"].append(train_stats["dice"])
            self.history["val_dice"].append(val_stats["dice"])
            self.history["train_iou"].append(train_stats["iou"])
            self.history["val_iou"].append(val_stats["iou"])

            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:03d}/{epochs:03d} | "
                f"TrainLoss={train_stats['loss']:.4f} "
                f"Dice={train_stats['dice']:.4f} | "
                f"ValLoss={val_stats['loss']:.4f} "
                f"Dice={val_stats['dice']:.4f} | "
                f"{elapsed:.1f}s"
            )

            # Save best checkpoint
            if val_stats["dice"] > self.best_val_dice:
                self.best_val_dice = val_stats["dice"]
                self._save_checkpoint("best_model.pth", epoch, val_stats)

            # Early stopping on val loss
            if self.early_stop(val_stats["loss"]):
                print(f"[trainer] Early stopping at epoch {epoch}.")
                break

        # Save final
        self._save_checkpoint("final_model.pth", epoch, val_stats)
        return self.history

    # ── Checkpoint ───────────────────────────────────────────────────────────

    def _save_checkpoint(self, name: str, epoch: int, stats: dict) -> None:
        path = os.path.join(self.checkpoint_dir, name)
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "stats": stats,
        }, path)

    def load_best(self) -> None:
        path = os.path.join(self.checkpoint_dir, "best_model.pth")
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            print(f"[trainer] Loaded best model (epoch {ckpt['epoch']}, "
                  f"dice={ckpt['stats']['dice']:.4f})")
        else:
            print("[trainer] No best checkpoint found.")


def build_optimizer_scheduler(model: nn.Module, cfg: dict):
    """Build optimizer and scheduler from config."""
    t = cfg["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=t.get("lr", 1e-4),
        weight_decay=t.get("weight_decay", 1e-5),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    return optimizer, scheduler
