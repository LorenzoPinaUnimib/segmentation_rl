"""
metrics.py — Segmentation evaluation metrics.
All functions accept binary numpy arrays or torch tensors (B,1,H,W) or (H,W).
"""
import numpy as np
import torch
from typing import Dict, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Core pixel-wise stats
# ─────────────────────────────────────────────────────────────────────────────

def _to_binary_numpy(t) -> np.ndarray:
    """Convert tensor or numpy to flat binary (0/1) float array."""
    if isinstance(t, torch.Tensor):
        t = t.cpu().numpy()
    t = np.asarray(t, dtype=np.float32).flatten()
    return (t > 0.5).astype(np.float32)


def confusion_stats(pred: np.ndarray, gt: np.ndarray):
    """Return (TP, FP, FN, TN) from flattened binary arrays."""
    tp = ((pred == 1) & (gt == 1)).sum()
    fp = ((pred == 1) & (gt == 0)).sum()
    fn = ((pred == 0) & (gt == 1)).sum()
    tn = ((pred == 0) & (gt == 0)).sum()
    return float(tp), float(fp), float(fn), float(tn)


# ─────────────────────────────────────────────────────────────────────────────
# Individual metrics
# ─────────────────────────────────────────────────────────────────────────────

def dice_coefficient(pred, gt, smooth: float = 1e-6) -> float:
    p = _to_binary_numpy(pred)
    g = _to_binary_numpy(gt)
    intersection = (p * g).sum()
    return (2 * intersection + smooth) / (p.sum() + g.sum() + smooth)


def iou_score(pred, gt, smooth: float = 1e-6) -> float:
    p = _to_binary_numpy(pred)
    g = _to_binary_numpy(gt)
    intersection = (p * g).sum()
    union = p.sum() + g.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def precision_score(pred, gt, smooth: float = 1e-6) -> float:
    p = _to_binary_numpy(pred)
    g = _to_binary_numpy(gt)
    tp, fp, fn, tn = confusion_stats(p, g)
    return (tp + smooth) / (tp + fp + smooth)


def recall_score(pred, gt, smooth: float = 1e-6) -> float:
    p = _to_binary_numpy(pred)
    g = _to_binary_numpy(gt)
    tp, fp, fn, tn = confusion_stats(p, g)
    return (tp + smooth) / (tp + fn + smooth)


def specificity_score(pred, gt, smooth: float = 1e-6) -> float:
    p = _to_binary_numpy(pred)
    g = _to_binary_numpy(gt)
    tp, fp, fn, tn = confusion_stats(p, g)
    return (tn + smooth) / (tn + fp + smooth)


def pixel_accuracy(pred, gt) -> float:
    p = _to_binary_numpy(pred)
    g = _to_binary_numpy(gt)
    return float((p == g).mean())


def hausdorff_distance(pred, gt) -> float:
    """
    Approximated Hausdorff distance using scipy.
    Returns 0.0 if either mask is empty.
    """
    try:
        from scipy.spatial.distance import directed_hausdorff
    except ImportError:
        return float("nan")

    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.cpu().numpy()

    pred = np.squeeze(np.asarray(pred) > 0.5)
    gt   = np.squeeze(np.asarray(gt) > 0.5)

    coords_pred = np.argwhere(pred)
    coords_gt   = np.argwhere(gt)

    if len(coords_pred) == 0 or len(coords_gt) == 0:
        return 0.0

    d1 = directed_hausdorff(coords_pred, coords_gt)[0]
    d2 = directed_hausdorff(coords_gt, coords_pred)[0]
    return float(max(d1, d2))


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate compute_all
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(pred, gt, compute_hausdorff: bool = False) -> Dict[str, float]:
    """
    Compute all metrics for a single (pred, gt) pair.
    pred, gt: binary tensors or numpy arrays.
    Includes raw tp/fp/fn/tn for aggregated confusion matrices.
    """
    p = _to_binary_numpy(pred)
    g = _to_binary_numpy(gt)
    tp, fp, fn, tn = confusion_stats(p, g)
    sm = 1e-6
    metrics = {
        "dice":        float(2*tp + sm) / float(2*tp + fp + fn + sm),
        "iou":         float(tp + sm)   / float(tp + fp + fn + sm),
        "precision":   float(tp + sm)   / float(tp + fp + sm),
        "recall":      float(tp + sm)   / float(tp + fn + sm),
        "specificity": float(tn + sm)   / float(tn + fp + sm),
        "pixel_acc":   float(tp + tn)   / float(tp + fp + fn + tn + sm),
        # raw counts for aggregated confusion matrix
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
    if compute_hausdorff:
        metrics["hausdorff"] = hausdorff_distance(pred, gt)
    return metrics


def average_metrics(metrics_list) -> Dict[str, float]:
    """Average a list of metric dicts. Raw counts (tp/fp/fn/tn) are summed."""
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    count_keys = {"tp", "fp", "fn", "tn"}
    result = {}
    for k in keys:
        vals = [m[k] for m in metrics_list if k in m]
        if not vals:
            continue
        if k in count_keys:
            result[k] = float(sum(vals))
        else:
            result[k] = float(np.nanmean(vals))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Batch-level (used in training loop)
# ─────────────────────────────────────────────────────────────────────────────

def batch_dice(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """Differentiable soft dice on batched logits for monitoring."""
    probs = torch.sigmoid(logits)
    probs_f   = probs.view(probs.size(0), -1)
    targets_f = targets.view(targets.size(0), -1)
    intersection = (probs_f * targets_f).sum(dim=1)
    dice = (2 * intersection + smooth) / (probs_f.sum(dim=1) + targets_f.sum(dim=1) + smooth)
    return dice.mean()


def batch_iou(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs_f   = (probs > 0.5).float().view(probs.size(0), -1)
    targets_f = targets.view(targets.size(0), -1)
    inter = (probs_f * targets_f).sum(dim=1)
    union = probs_f.sum(dim=1) + targets_f.sum(dim=1) - inter
    return ((inter + smooth) / (union + smooth)).mean()