"""
fast_trainer.py — Training loop ottimizzato per DirectML su Windows.

FIX rispetto alla versione precedente:
  - Ripristinata tqdm progress bar per ogni batch (come nell'originale)
  - Loss usa solo DiceLoss/TverskyLoss (BCE non gira su DirectML)
  - zero_grad(set_to_none=True) più veloce
  - PrefetchLoader riduce tempo morto GPU tra batch
"""
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Prefetch loader
# ─────────────────────────────────────────────────────────────────────────────

class PrefetchLoader:
    """Manda il prossimo batch su device mentre la GPU elabora quello corrente."""
    def __init__(self, loader: DataLoader, device: torch.device):
        self.loader = loader
        self.device = device

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        it = iter(self.loader)
        try:
            next_batch = next(it)
        except StopIteration:
            return

        while True:
            batch = next_batch
            # Sposta su device con non_blocking
            batch_on_device = {
                k: v.to(self.device, non_blocking=True)
                   if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            try:
                next_batch = next(it)
            except StopIteration:
                yield batch_on_device
                return
            yield batch_on_device


# ─────────────────────────────────────────────────────────────────────────────
# Early stopping
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# FastTrainer
# ─────────────────────────────────────────────────────────────────────────────

class FastTrainer:
    """
    Drop-in replacement di BaselineTrainer, ottimizzato per DirectML/Windows.
    Mantiene la tqdm progress bar per monitorare ogni batch.
    """

    def __init__(self, model, criterion, optimizer, scheduler, device, cfg):
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
        self.best_val_dice = 0.0
        self.early_stop = EarlyStopping(patience=self.patience)

    @staticmethod
    def _align_targets(logits, targets):
        """[B,H,W] → [B,1,H,W] se necessario."""
        targets = targets.float()
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)
        return targets

    @staticmethod
    def _batch_dice(logits, targets, smooth=1e-6):
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            p = probs.view(probs.size(0), -1)
            t = targets.view(targets.size(0), -1)
            inter = (p * t).sum(dim=1)
            return ((2 * inter + smooth) /
                    (p.sum(dim=1) + t.sum(dim=1) + smooth)).mean().item()

    @staticmethod
    def _batch_iou(logits, targets, smooth=1e-6):
        with torch.no_grad():
            probs = (torch.sigmoid(logits) > 0.5).float()
            p = probs.view(probs.size(0), -1)
            t = targets.view(targets.size(0), -1)
            inter = (p * t).sum(dim=1)
            union = p.sum(dim=1) + t.sum(dim=1) - inter
            return ((inter + smooth) / (union + smooth)).mean().item()

    def _run_epoch(self, loader: DataLoader, train: bool) -> Dict[str, float]:
        self.model.train(train)
        total_loss = dice_sum = iou_sum = 0.0
        n = 0

        pf = PrefetchLoader(loader, self.device)
        label = "train" if train else "val  "

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            # tqdm con postfix aggiornato in tempo reale
            pbar = tqdm(pf, total=len(loader), desc=f"  {label}",
                        leave=False, dynamic_ncols=True)
            for batch in pbar:
                images  = batch["image"]
                targets = batch["mask"].to(self.device, non_blocking=True)

                logits  = self.model(images)
                targets = self._align_targets(logits, targets)
                loss    = self.criterion(logits, targets)

                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if self.grad_clip:
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.grad_clip)
                    self.optimizer.step()

                bs = images.size(0)
                total_loss += loss.item() * bs
                dice_val    = self._batch_dice(logits.detach(), targets)
                iou_val     = self._batch_iou(logits.detach(),  targets)
                dice_sum   += dice_val * bs
                iou_sum    += iou_val  * bs
                n += bs

                # Aggiorna la barra con le metriche correnti
                pbar.set_postfix({
                    "loss": f"{total_loss/n:.4f}",
                    "dice": f"{dice_sum/n:.4f}",
                })
            pbar.close()

        return {"loss": total_loss / n, "dice": dice_sum / n, "iou": iou_sum / n}

    def train(self, train_loader, val_loader, epochs) -> Dict[str, List[float]]:
        print(f"\n{'='*60}")
        print(f" FastTrainer: {epochs} epochs | device={self.device}")
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

            if val_stats["dice"] > self.best_val_dice:
                self.best_val_dice = val_stats["dice"]
                self._save_checkpoint("best_model.pth", epoch, val_stats)

            if self.early_stop(val_stats["loss"]):
                print(f"[trainer] Early stopping at epoch {epoch}.")
                break

        self._save_checkpoint("final_model.pth", epoch, val_stats)
        return self.history

    def _save_checkpoint(self, name, epoch, stats):
        path = os.path.join(self.checkpoint_dir, name)
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "stats": stats,
        }, path)

    def load_best(self):
        path = os.path.join(self.checkpoint_dir, "best_model.pth")
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(ckpt["model_state_dict"])
            print(f"[trainer] Loaded best model (epoch {ckpt['epoch']}, "
                  f"dice={ckpt['stats']['dice']:.4f})")
        else:
            print("[trainer] No best checkpoint found.")


def build_optimizer_scheduler(model, cfg):
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