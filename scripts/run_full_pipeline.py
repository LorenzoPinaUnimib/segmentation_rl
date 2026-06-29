#!/usr/bin/env python3
"""
run_full_pipeline.py
Pipeline DOPPIO RL senza U-Net:
  1) Carica dataset
  2) Addestra Agent 1 (ROIFinderAgent) → localizza il tumore con bounding box
  3) Addestra Agent 2 (ROIRefinementAgent) → raffina la maschera nel crop ROI
  4) Valuta il pipeline combinato sul test set
  5) Salva metriche e report

ARCHITETTURA:
  Input Image
      │
      ▼
  [Agent 1: ROIFinder] ──► Bounding Box ROI
      │
      ▼  (crop + resize)
  [Prob Map Adattiva]  ──► Maschera iniziale grezza
      │
      ▼
  [Agent 2: ROIRefiner] ──► Maschera raffinata finale
      │
      ▼
  Metriche (Dice, IoU, ...)

Nessun modello U-Net viene usato in questo pipeline.
I file originali NON vengono modificati.

NOVITÀ:
  - Training: ogni iterazione usa TUTTE le immagini del train set.
  - Validazione: ad ogni eval_every si valuta su TUTTO il val set (IoU/Dice).
  - Test: solo alla fine, sul test set, per metriche e visualizzazioni finali.
  - Checkpoint parziali salvati sul val set durante il training.
"""
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
from skimage.transform import resize as sk_resize

from src.utils.io_utils import load_config, set_seed, get_device, ensure_dirs
from src.data.dataset import get_datasets, make_dataloaders
from src.eval.metrics import compute_metrics, average_metrics, dice_coefficient

# Nuovi moduli RL (non sovrascrivono nulla di esistente)
from src.rl.environment_roi import ROIFinderEnv
from src.rl.environment_refine import ROIRefinementEnv
from src.rl.agent_roi import ROIFinderAgent
from src.rl.agent_refine import ROIRefinementAgent
from src.train.rl_trainer import (
    ROIFinderTrainer,
    ROIRefinerTrainer,
    make_prob_map_adaptive,
    make_prob_map_otsu,
    _tensor_to_gray,
    _tensor_to_mask,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_img_size(cfg: dict) -> int:
    sz = cfg["dataset"].get("image_size", 256)
    return sz[0] if isinstance(sz, (list, tuple)) else int(sz)


def _gray_to_rgb(img_gray: np.ndarray) -> np.ndarray:
    """[H,W] float grayscale → uint8 RGB."""
    img = img_gray.astype(np.float32)
    if img.max() <= 1.01:
        img = img * 255.0
    img_u8 = np.clip(img, 0, 255).astype(np.uint8)
    return np.stack([img_u8, img_u8, img_u8], axis=-1)


def _draw_box(
    img_rgb: np.ndarray,
    box: list,
    color: tuple,
    thickness: int = 2,
    label: str = "",
) -> np.ndarray:
    """Disegna un bounding box su immagine RGB."""
    out = img_rgb.copy()
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    if label:
        cv2.putText(
            out, label, (x1, max(y1 - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
        )
    return out


def run_roi_finder_inference(
    image_t: torch.Tensor,
    mask_t: torch.Tensor,
    finder_agent: ROIFinderAgent,
    finder_env: ROIFinderEnv,
) -> dict:
    """Inferenza ROI Finder: restituisce box predetta, GT e metriche."""
    img_gray = _tensor_to_gray(image_t)
    gt_mask  = _tensor_to_mask(mask_t)

    state = finder_env.reset(img_gray, gt_mask)
    done = False
    while not done:
        action = finder_agent.act(state, greedy=True)
        state, _, done, info = finder_env.step(action)

    return {
        "img_gray": img_gray,
        "gt_mask":  gt_mask,
        "gt_box":   list(finder_env.gt_box),
        "pred_box": finder_env.get_box(),
        "iou":      info["iou"],
    }


def _overlay_mask_contour(
    img_rgb: np.ndarray,
    mask: np.ndarray,
    color: tuple = (50, 255, 50),
    thickness: int = 2,
) -> np.ndarray:
    """Disegna il contorno di una maschera su immagine RGB."""
    out = img_rgb.copy()
    mask_u8 = (mask > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, thickness)
    return out


def _iter_subdir(episode: int, is_final: bool = False) -> str:
    return "final" if is_final else f"iter_{episode:04d}"


def save_test_roi_predictions(
    data_loader,
    finder_agent: ROIFinderAgent,
    finder_env: ROIFinderEnv,
    cfg: dict,
    tag: str = "roi_finder",
    subdir: str = "final",
    split_name: str = "Test",
    verbose: bool = True,
) -> str:
    """
    Salva ogni immagine di test con GT ROI e predizione ROI.
    Output in outputs/predictions/<tag>/<subdir>/.
    """
    save_dir = os.path.join(cfg["output"]["predictions_dir"], tag, subdir)
    os.makedirs(save_dir, exist_ok=True)

    if verbose:
        print(f"\n[pipeline] Salvataggio ROI ({split_name}) → {save_dir}")

    sample_idx = 0
    grid_rows = []

    for batch in data_loader:
        imgs = batch["image"]
        masks = batch["mask"]
        for i in range(imgs.size(0)):
            result = run_roi_finder_inference(
                imgs[i], masks[i], finder_agent, finder_env,
            )
            img_rgb = _gray_to_rgb(result["img_gray"])
            gt_box = result["gt_box"]
            pred_box = result["pred_box"]
            iou = result["iou"]

            panel_gt = _draw_box(img_rgb, gt_box, (50, 220, 50), 2, "GT")
            panel_pred = _draw_box(img_rgb, pred_box, (255, 90, 50), 2, f"Pred IoU={iou:.3f}")
            panel_both = _draw_box(
                _draw_box(img_rgb, gt_box, (50, 220, 50), 2, "GT"),
                pred_box, (255, 90, 50), 2, "Pred",
            )

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            fig.patch.set_facecolor("#0F172A")
            fig.suptitle(
                f"{split_name} #{sample_idx:04d} — ROI Finder (IoU={iou:.4f})",
                color="white", fontsize=11, fontweight="bold",
            )
            titles = ["Ground Truth ROI", "Predicted ROI", "GT + Prediction"]
            panels = [panel_gt, panel_pred, panel_both]
            for ax, title, panel in zip(axes, titles, panels):
                ax.imshow(panel)
                ax.set_title(title, color="white", fontsize=9)
                ax.axis("off")

            out_path = os.path.join(save_dir, f"test_{sample_idx:04d}.png")
            fig.tight_layout(pad=0.4)
            fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)

            grid_rows.append((panel_both, sample_idx, iou))
            sample_idx += 1

    # Griglia riassuntiva (max 16 campioni)
    if grid_rows:
        n_show = min(16, len(grid_rows))
        cols = 4
        rows = int(np.ceil(n_show / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
        fig.patch.set_facecolor("#0F172A")
        fig.suptitle(
            f"ROI Finder — {split_name} set (GT verde, Pred arancio)",
            color="white", fontsize=13, fontweight="bold",
        )
        axes_flat = np.array(axes).flatten()
        for j, ax in enumerate(axes_flat):
            if j < n_show:
                panel, idx, iou = grid_rows[j]
                ax.imshow(panel)
                ax.set_title(f"#{idx:04d} IoU={iou:.3f}", color="white", fontsize=8)
            ax.axis("off")
        grid_path = os.path.join(save_dir, "roi_summary_grid.png")
        fig.tight_layout(pad=0.3)
        fig.savefig(grid_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        if verbose:
            print(f"  Salvate {sample_idx} immagini + griglia → {grid_path}")

    return save_dir


def evaluate_roi_finder_only(
    data_loader,
    finder_agent: ROIFinderAgent,
    finder_env: ROIFinderEnv,
) -> dict:
    """Valuta solo il ROI Finder (senza refinement) su un loader."""
    ious = []
    per_sample = []
    sample_idx = 0

    for batch in data_loader:
        imgs = batch["image"]
        masks = batch["mask"]
        for i in range(imgs.size(0)):
            result = run_roi_finder_inference(
                imgs[i], masks[i], finder_agent, finder_env,
            )
            ious.append(result["iou"])
            per_sample.append({
                "sample_idx": sample_idx,
                "iou": result["iou"],
                "gt_box": result["gt_box"],
                "pred_box": result["pred_box"],
            })
            sample_idx += 1

    return {
        "mean_iou":   float(np.mean(ious)) if ious else 0.0,
        "std_iou":    float(np.std(ious)) if ious else 0.0,
        "median_iou": float(np.median(ious)) if ious else 0.0,
        "n_samples":  len(ious),
        "per_sample": per_sample,
    }


def save_test_dual_rl_predictions(
    data_loader,
    finder_agent: ROIFinderAgent,
    finder_env: ROIFinderEnv,
    refiner_agent: ROIRefinementAgent,
    refiner_env: ROIRefinementEnv,
    img_size: int,
    cfg: dict,
    prob_method: str = "adaptive",
    tag: str = "dual_rl",
    subdir: str = "final",
    split_name: str = "Test",
    verbose: bool = True,
) -> str:
    """
    Salva immagini di test con pipeline completa (con refinement).
    Pannelli: immagine+ROI | GT mask | baseline (no refiner) | RL refined.
    """
    save_dir = os.path.join(cfg["output"]["predictions_dir"], tag, subdir)
    os.makedirs(save_dir, exist_ok=True)

    if verbose:
        print(f"\n[pipeline] Salvataggio dual RL ({split_name}) → {save_dir}")

    sample_idx = 0
    grid_rows = []

    for batch in data_loader:
        imgs = batch["image"]
        masks = batch["mask"]
        for i in range(imgs.size(0)):
            result = run_full_inference(
                imgs[i], masks[i],
                finder_agent, finder_env,
                refiner_agent, refiner_env,
                img_size, prob_method,
            )
            img_gray = _tensor_to_gray(imgs[i])
            img_rgb = _gray_to_rgb(img_gray)
            x1, y1, x2, y2 = [int(v) for v in result["box"]]
            crop_gray = img_gray[y1:y2, x1:x2]
            if crop_gray.size == 0:
                crop_gray = img_gray
            crop_r = sk_resize(
                crop_gray, (img_size, img_size),
                anti_aliasing=True, preserve_range=True,
            ).astype(np.float32)
            crop_rgb = _gray_to_rgb(crop_r)

            gt_mask = result["gt_mask"]
            bl_mask = result["baseline_mask"]
            rl_mask = result["refined_mask"]
            iou_roi = result["iou_finder"]
            dice_bl = dice_coefficient(bl_mask, gt_mask)
            dice_rl = dice_coefficient(rl_mask, gt_mask)

            panel_roi = _draw_box(
                _draw_box(img_rgb, list(finder_env.gt_box), (50, 220, 50), 2, "GT"),
                result["box"], (255, 90, 50), 2, f"Pred IoU={iou_roi:.3f}",
            )
            panel_gt  = _overlay_mask_contour(crop_rgb, gt_mask, (50, 220, 50))
            panel_bl  = _overlay_mask_contour(crop_rgb, bl_mask, (255, 140, 50))
            panel_rl  = _overlay_mask_contour(crop_rgb, rl_mask, (80, 160, 255))

            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            fig.patch.set_facecolor("#0F172A")
            fig.suptitle(
                f"{split_name} #{sample_idx:04d} — Con refinement "
                f"(Dice BL={dice_bl:.3f} RL={dice_rl:.3f})",
                color="white", fontsize=11, fontweight="bold",
            )
            titles = ["ROI sul crop", "GT (crop)", "Baseline (no refiner)", "RL refined"]
            panels = [panel_roi, panel_gt, panel_bl, panel_rl]
            for ax, title, panel in zip(axes, titles, panels):
                ax.imshow(panel)
                ax.set_title(title, color="white", fontsize=9)
                ax.axis("off")

            out_path = os.path.join(save_dir, f"test_{sample_idx:04d}.png")
            fig.tight_layout(pad=0.4)
            fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)

            grid_rows.append((panel_rl, sample_idx, dice_rl))
            sample_idx += 1

    if grid_rows:
        n_show = min(16, len(grid_rows))
        cols = 4
        rows = int(np.ceil(n_show / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
        fig.patch.set_facecolor("#0F172A")
        fig.suptitle(
            f"Dual RL con refinement — {split_name} set",
            color="white", fontsize=13, fontweight="bold",
        )
        axes_flat = np.array(axes).flatten()
        for j, ax in enumerate(axes_flat):
            if j < n_show:
                panel, idx, dice = grid_rows[j]
                ax.imshow(panel)
                ax.set_title(f"#{idx:04d} Dice={dice:.3f}", color="white", fontsize=8)
            ax.axis("off")
        grid_path = os.path.join(save_dir, "dual_rl_summary_grid.png")
        fig.tight_layout(pad=0.3)
        fig.savefig(grid_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        if verbose:
            print(f"  Salvate {sample_idx} immagini + griglia → {grid_path}")

    return save_dir


def _save_partial_metrics(
    cfg: dict,
    prefix: str,
    episode: int,
    metrics: dict,
    is_final: bool = False,
) -> str:
    """Salva CSV metriche parziali."""
    partial_dir = os.path.join(cfg["output"]["metrics_dir"], "partial")
    os.makedirs(partial_dir, exist_ok=True)
    suffix = "final" if is_final else f"iter_{episode:04d}"
    path = os.path.join(partial_dir, f"{prefix}_{suffix}.csv")
    row = {"iteration": episode, **metrics}
    pd.DataFrame([row]).to_csv(path, index=False)
    return path


def make_finder_partial_callback(
    val_loader,
    finder_agent: ROIFinderAgent,
    finder_env: ROIFinderEnv,
    cfg: dict,
):
    """Callback: salva risultati parziali sul val set ad ogni eval (senza refinement)."""
    def callback(episode: int, stats: dict) -> None:
        subdir = _iter_subdir(episode)
        print(f"\n[pipeline] Checkpoint ROI Finder @ iter {episode:04d} (val, senza refinement)")
        save_test_roi_predictions(
            val_loader, finder_agent, finder_env, cfg,
            tag="roi_finder", subdir=subdir, split_name="Val", verbose=True,
        )
        _save_partial_metrics(cfg, "roi_finder_val", episode, {
            "val_iou_mean":   stats.get("mean", 0.0),
            "val_iou_std":    stats.get("std", 0.0),
            "val_iou_median": stats.get("median", 0.0),
            "n_samples":      stats.get("n", 0),
        })
    return callback


def make_refiner_partial_callback(
    val_loader,
    finder_agent: ROIFinderAgent,
    finder_env: ROIFinderEnv,
    refiner_agent: ROIRefinementAgent,
    refiner_env: ROIRefinementEnv,
    img_size: int,
    cfg: dict,
    prob_method: str,
):
    """Callback: salva risultati parziali sul val set ad ogni eval (con refinement)."""
    def callback(episode: int, stats: dict) -> None:
        subdir = _iter_subdir(episode)
        print(f"\n[pipeline] Checkpoint Refiner @ iter {episode:04d} (val, con refinement)")
        save_test_dual_rl_predictions(
            val_loader, finder_agent, finder_env,
            refiner_agent, refiner_env,
            img_size, cfg, prob_method,
            tag="dual_rl", subdir=subdir, split_name="Val", verbose=True,
        )
        _save_partial_metrics(cfg, "dual_rl_val", episode, {
            "val_dice_bl_mean":   stats.get("dice_bl", {}).get("mean", 0.0),
            "val_dice_rl_mean":   stats.get("dice_rl", {}).get("mean", 0.0),
            "val_dice_rl_std":    stats.get("dice_rl", {}).get("std", 0.0),
            "val_dice_rl_median": stats.get("dice_rl", {}).get("median", 0.0),
            "val_iou_rl_mean":    stats.get("iou_rl", {}).get("mean", 0.0),
            "n_samples":          stats.get("n", 0),
        })
    return callback


def run_full_inference(
    image_t: torch.Tensor,
    mask_t: torch.Tensor,
    finder_agent: ROIFinderAgent,
    finder_env: ROIFinderEnv,
    refiner_agent: ROIRefinementAgent,
    refiner_env: ROIRefinementEnv,
    img_size: int,
    prob_method: str = "adaptive",
) -> dict:
    """
    Inferenza completa sul singolo campione con i due agenti RL.

    Ritorna:
        - refined_mask    [H,W] maschera finale raffinata (su img_size)
        - baseline_mask   [H,W] maschera pre-raffinamento
        - gt_mask         [H,W] ground truth
        - box             [x1,y1,x2,y2] ROI trovata
        - iou_finder      IoU del bounding box
    """
    img_gray = _tensor_to_gray(image_t)
    gt_mask  = _tensor_to_mask(mask_t)

    # ── Agente 1: ROI Finder ──────────────────────────────────────────────────
    state = finder_env.reset(img_gray, gt_mask)
    done = False
    while not done:
        action = finder_agent.act(state, greedy=True)
        state, _, done, info = finder_env.step(action)

    box = finder_env.get_box()
    iou_finder = info["iou"]
    x1, y1, x2, y2 = [int(v) for v in box]

    # Crop
    crop_gray = img_gray[y1:y2, x1:x2]
    gt_crop   = gt_mask[y1:y2, x1:x2]

    if crop_gray.size == 0:
        crop_gray = img_gray
        gt_crop   = gt_mask
        box = [0, 0, img_size, img_size]

    # Resize a img_size
    crop_r = sk_resize(crop_gray, (img_size, img_size),
                       anti_aliasing=True, preserve_range=True).astype(np.float32)
    gt_r   = (sk_resize(gt_crop, (img_size, img_size),
                        anti_aliasing=False, preserve_range=True) > 0.5).astype(np.float32)

    # Prob map adattiva (senza U-Net)
    if prob_method == "otsu":
        prob_map = make_prob_map_otsu(crop_r)
    else:
        prob_map = make_prob_map_adaptive(crop_r)

    baseline_mask_crop = (prob_map > 0.5).astype(np.float32)

    # ── Agente 2: ROI Refiner ─────────────────────────────────────────────────
    state = refiner_env.reset(crop_r, prob_map, gt_r)
    done = False
    while not done:
        action = refiner_agent.act(state, greedy=True)
        state, _, done, _ = refiner_env.step(action)

    refined_crop = refiner_env.get_refined_mask()

    return {
        "refined_mask":  refined_crop,
        "baseline_mask": baseline_mask_crop,
        "gt_mask":       gt_r,
        "gt_full":       gt_mask,
        "box":           box,
        "iou_finder":    iou_finder,
        "prob_map":      prob_map,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Valutazione sul test set
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_dual_rl(
    test_loader,
    finder_agent: ROIFinderAgent,
    finder_env: ROIFinderEnv,
    refiner_agent: ROIRefinementAgent,
    refiner_env: ROIRefinementEnv,
    img_size: int,
    cfg: dict,
    prob_method: str = "adaptive",
) -> dict:
    """Valuta il pipeline doppio RL su TUTTO il test set."""
    print("\n[pipeline] Valutazione su TUTTO il test set...")

    all_metrics_bl = []
    all_metrics_rl = []
    all_iou_finder = []
    do_hausdorff   = cfg["evaluation"].get("hausdorff", False)
    total_samples  = 0

    for batch_idx, batch in enumerate(test_loader):
        imgs    = batch["image"]
        targets = batch["mask"]

        for i in range(imgs.size(0)):
            result = run_full_inference(
                imgs[i], targets[i],
                finder_agent, finder_env,
                refiner_agent, refiner_env,
                img_size, prob_method,
            )

            m_bl = compute_metrics(
                result["baseline_mask"], result["gt_mask"],
                compute_hausdorff=do_hausdorff,
            )
            m_rl = compute_metrics(
                result["refined_mask"], result["gt_mask"],
                compute_hausdorff=do_hausdorff,
            )

            all_metrics_bl.append(m_bl)
            all_metrics_rl.append(m_rl)
            all_iou_finder.append(result["iou_finder"])
            total_samples += 1

        if batch_idx % 5 == 0:
            print(f"  Campioni elaborati: {total_samples}")

    print(f"  Totale campioni valutati: {total_samples}")

    avg_bl = average_metrics(all_metrics_bl)
    avg_rl = average_metrics(all_metrics_rl)
    avg_iou_finder = float(np.mean(all_iou_finder))
    std_iou_finder = float(np.std(all_iou_finder))

    # Statistiche dettagliate per ogni metrica
    dice_vals_rl = [m.get("dice", 0) for m in all_metrics_rl]
    dice_vals_bl = [m.get("dice", 0) for m in all_metrics_bl]

    print(f"\n  ROI Finder IoU: μ={avg_iou_finder:.4f} σ={std_iou_finder:.4f}")
    print(f"\n  Baseline (sogliatura adattiva) [n={total_samples}]:")
    for k, v in avg_bl.items():
        if k not in ("tp", "fp", "fn", "tn"):
            print(f"    {k:12s}: {v:.4f}")
    print(f"\n  Doppio RL raffinato [n={total_samples}]:")
    for k, v in avg_rl.items():
        if k not in ("tp", "fp", "fn", "tn"):
            print(f"    {k:12s}: {v:.4f}")

    print(f"\n  Dice RL: μ={np.mean(dice_vals_rl):.4f} "
          f"σ={np.std(dice_vals_rl):.4f} "
          f"med={np.median(dice_vals_rl):.4f}")
    print(f"  Dice BL: μ={np.mean(dice_vals_bl):.4f} "
          f"σ={np.std(dice_vals_bl):.4f} "
          f"med={np.median(dice_vals_bl):.4f}")

    return {
        "baseline": avg_bl,
        "dual_rl":  avg_rl,
        "finder_iou": avg_iou_finder,
        "finder_iou_std": std_iou_finder,
        "per_sample_bl": all_metrics_bl,
        "per_sample_rl": all_metrics_rl,
        "n_samples": total_samples,
        "dice_rl_std":    float(np.std(dice_vals_rl)),
        "dice_rl_median": float(np.median(dice_vals_rl)),
        "dice_bl_std":    float(np.std(dice_vals_bl)),
        "dice_bl_median": float(np.median(dice_vals_bl)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def write_report(results: dict, cfg: dict) -> None:
    bl = results["baseline"]
    rl = results["dual_rl"]
    finder_iou = results["finder_iou"]
    finder_iou_std = results.get("finder_iou_std", 0.0)
    n_samples = results.get("n_samples", "?")
    dice_rl_std = results.get("dice_rl_std", 0.0)
    dice_rl_median = results.get("dice_rl_median", 0.0)

    skip = {"tp", "fp", "fn", "tn"}
    lines = [
        "# Brain Tumor Segmentation — Dual RL Pipeline (No U-Net)\n\n",
        f"Dataset: `{cfg['dataset']['source']}`  \n",
        f"Image size: `{cfg['dataset']['image_size']}`  \n",
        f"ROI Finder episodes: `{cfg['rl'].get('roi_episodes', cfg['rl'].get('episodes', 300))}`  \n",
        f"ROI Refiner episodes: `{cfg['rl'].get('refine_episodes', cfg['rl'].get('episodes', 300))}`  \n",
        f"Campioni di test valutati: `{n_samples}` (full test set)  \n\n",
        f"**ROI Finder IoU sul test set: μ={finder_iou:.4f} σ={finder_iou_std:.4f}**\n\n",
        "## Metriche di Segmentazione\n\n",
        "| Metrica | Baseline (sogliatura) | Doppio RL | Delta |\n",
        "|---------|----------------------|-----------|-------|\n",
    ]
    for k in bl:
        if k in skip:
            continue
        bv = bl[k]
        rv = rl.get(k, float("nan"))
        delta = rv - bv if not np.isnan(rv) else float("nan")
        lines.append(f"| {k} | {bv:.4f} | {rv:.4f} | {delta:+.4f} |\n")

    dice_delta = rl.get("dice", 0) - bl.get("dice", 0)
    lines.append("\n## Stabilità dei risultati\n\n")
    lines.append(f"- Dice RL: μ={rl.get('dice',0):.4f} σ={dice_rl_std:.4f} "
                 f"mediana={dice_rl_median:.4f} (n={n_samples})\n")
    lines.append(f"- Baseline Dice: μ={bl.get('dice',0):.4f} "
                 f"σ={results.get('dice_bl_std',0):.4f} "
                 f"mediana={results.get('dice_bl_median',0):.4f}\n")
    lines.append(f"- ROI Finder: IoU μ={finder_iou:.4f} σ={finder_iou_std:.4f}\n")

    lines.append("\n## Analisi\n\n")
    if dice_delta > 0.005:
        lines.append(f"- Agent 2 (Refiner) ha migliorato il Dice di **{dice_delta:+.4f}**\n")
    elif dice_delta < -0.005:
        lines.append(f"- Agent 2 (Refiner) ha ridotto il Dice di {dice_delta:+.4f} "
                     "— considerare più episodi di training.\n")
    else:
        lines.append(f"- Agent 2 (Refiner) effetto minimo sul Dice ({dice_delta:+.4f}).\n")

    lines.append("\n## File Output\n\n")
    lines.append("| Path | Contenuto |\n")
    lines.append("|------|-----------|\n")
    lines.append(f"| `{cfg['output']['metrics_dir']}/final_results.csv` | Metriche finali |\n")
    lines.append(f"| `{cfg['output']['metrics_dir']}/roi_finder_history.csv` | Curva training Finder |\n")
    lines.append(f"| `{cfg['output']['metrics_dir']}/roi_refiner_history.csv` | Curva training Refiner |\n")
    lines.append(f"| `{cfg['output']['predictions_dir']}/roi_finder/iter_XXXX/` | Checkpoint val (senza refinement) |\n")
    lines.append(f"| `{cfg['output']['predictions_dir']}/roi_finder/test_final/` | Test finale ROI Finder |\n")
    lines.append(f"| `{cfg['output']['predictions_dir']}/dual_rl/iter_XXXX/` | Checkpoint val (con refinement) |\n")
    lines.append(f"| `{cfg['output']['predictions_dir']}/dual_rl/test_final/` | Test finale pipeline completa |\n")
    lines.append(f"| `{cfg['output']['metrics_dir']}/partial/` | Metriche val (iter) e test (final) |\n")
    lines.append(f"| `{cfg['output']['checkpoint_dir']}/best_roi_finder_agent.pth` | Pesi Agent 1 |\n")
    lines.append(f"| `{cfg['output']['checkpoint_dir']}/best_roi_refiner_agent.pth` | Pesi Agent 2 |\n")

    report_path = os.path.join(cfg["output"]["reports_dir"], "report.md")
    os.makedirs(cfg["output"]["reports_dir"], exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"[pipeline] Report salvato → {report_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cfg_path = os.path.join(ROOT, "configs", "config.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config non trovata: {cfg_path}")

    cfg = load_config(cfg_path)
    set_seed(cfg.get("seed", 42))
    device = get_device(cfg.get("device", "auto"))
    ensure_dirs(cfg)

    img_size = get_img_size(cfg)
    rl_cfg   = cfg.get("rl", {})

    # ── 1) Dataset ────────────────────────────────────────────────────────────
    print("\n[pipeline] Step 1: Caricamento dataset...")
    train_ds, val_ds, test_ds = get_datasets(cfg)
    train_loader, val_loader, test_loader = make_dataloaders(
        cfg, train_ds, val_ds, test_ds
    )
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # ── 2) Setup Agent 1: ROI Finder ──────────────────────────────────────────
    print("\n[pipeline] Step 2: Inizializzazione Agent 1 (ROI Finder)...")
    finder_env   = ROIFinderEnv(cfg)
    finder_agent = ROIFinderAgent(cfg, device)

    roi_episodes = rl_cfg.get("roi_episodes", rl_cfg.get("episodes", 300))
    eval_every_1 = max(1, roi_episodes // 15)

    print(f"  State dim: {finder_env.STATE_DIM} | Actions: {finder_env.NUM_ACTIONS}")
    print(f"  Episodi: {roi_episodes} | Step max/episodio: {finder_env.max_steps}")
    print(f"  Ogni iterazione = tutte le immagini train")
    print(f"  Validazione su val set ogni {eval_every_1} iterazioni")

    finder_trainer = ROIFinderTrainer(
        agent        = finder_agent,
        env          = finder_env,
        train_loader = train_loader,
        val_loader   = val_loader,
        device       = device,
        cfg          = cfg,
        on_eval_callback = make_finder_partial_callback(
            val_loader, finder_agent, finder_env, cfg,
        ),
    )
    finder_trainer.episodes = roi_episodes

    # ── 3) Training Agent 1 ───────────────────────────────────────────────────
    print("\n[pipeline] Step 3: Training Agent 1 (ROI Finder)...")
    history_finder = finder_trainer.train(eval_every=eval_every_1)
    finder_agent.load("best_roi_finder_agent.pth")

    # Valutazione e salvataggio finale sul TEST set (solo ROI Finder)
    print("\n[pipeline] Valutazione finale ROI Finder sul test set...")
    roi_preds_dir = save_test_roi_predictions(
        data_loader=test_loader,
        finder_agent=finder_agent,
        finder_env=finder_env,
        cfg=cfg,
        tag="roi_finder",
        subdir="test_final",
        split_name="Test",
    )
    roi_test = evaluate_roi_finder_only(test_loader, finder_agent, finder_env)
    _save_partial_metrics(cfg, "roi_finder_test", roi_episodes, {
        "test_iou_mean":   roi_test["mean_iou"],
        "test_iou_std":    roi_test["std_iou"],
        "test_iou_median": roi_test["median_iou"],
        "n_samples":       roi_test["n_samples"],
    }, is_final=True)
    print(f"  Test IoU: μ={roi_test['mean_iou']:.4f} σ={roi_test['std_iou']:.4f}")

    # Salva history finder (estesa con medie mobili e statistiche full-test)
    n_eps = len(history_finder["rewards"])
    n_val = len(history_finder["val_iou_mean"])
    finder_hist_df = pd.DataFrame({
        "episode":        range(1, n_eps + 1),
        "reward":         history_finder["rewards"],
        "reward_smooth":  history_finder["rewards_smooth"],
        "iou":            history_finder["iou"],
        "iou_smooth":     history_finder["iou_smooth"],
        "epsilon":        history_finder["epsilon"],
    })
    finder_hist_df.to_csv(
        os.path.join(cfg["output"]["metrics_dir"], "roi_finder_history.csv"),
        index=False
    )
    # Salva anche la history degli eval (full-test)
    if n_val > 0:
        pd.DataFrame({
            "eval_episode":   history_finder["val_ep"],
            "val_iou_mean":   history_finder["val_iou_mean"],
            "val_iou_std":    history_finder["val_iou_std"],
            "val_iou_median": history_finder["val_iou_median"],
        }).to_csv(
            os.path.join(cfg["output"]["metrics_dir"], "roi_finder_eval_history.csv"),
            index=False
        )

    # ── 4) Setup Agent 2: ROI Refiner ─────────────────────────────────────────
    print("\n[pipeline] Step 4: Inizializzazione Agent 2 (ROI Refiner)...")
    refiner_env   = ROIRefinementEnv(cfg)
    refiner_agent = ROIRefinementAgent(cfg, device)

    base_agent_path = os.path.join(cfg["output"]["checkpoint_dir"], "best_rl_agent.pth")
    refiner_agent.load_from_base_agent(base_agent_path)

    refine_episodes = rl_cfg.get("refine_episodes", rl_cfg.get("episodes", 300))
    eval_every_2    = max(1, refine_episodes // 15)
    prob_method     = rl_cfg.get("prob_method", "adaptive")

    print(f"  State dim: {refiner_env.STATE_DIM} | Actions: {refiner_env.NUM_ACTIONS}")
    print(f"  Episodi: {refine_episodes} | Metodo prob map: {prob_method}")
    print(f"  Ogni iterazione = tutte le immagini train")
    print(f"  Validazione su val set ogni {eval_every_2} iterazioni")

    refiner_trainer = ROIRefinerTrainer(
        finder_agent  = finder_agent,
        finder_env    = finder_env,
        refiner_agent = refiner_agent,
        refiner_env   = refiner_env,
        train_loader  = train_loader,
        val_loader    = val_loader,
        device        = device,
        cfg           = cfg,
        prob_method   = prob_method,
        on_eval_callback = make_refiner_partial_callback(
            val_loader, finder_agent, finder_env,
            refiner_agent, refiner_env,
            img_size, cfg, prob_method,
        ),
    )
    refiner_trainer.episodes = refine_episodes

    # ── 5) Training Agent 2 ───────────────────────────────────────────────────
    print("\n[pipeline] Step 5: Training Agent 2 (ROI Refiner)...")
    history_refiner = refiner_trainer.train(eval_every=eval_every_2)
    refiner_agent.load("best_roi_refiner_agent.pth")

    # Salva risultati finali con refinement sul TEST set
    print("\n[pipeline] Salvataggio predizioni finali sul test set (con refinement)...")
    dual_preds_dir = save_test_dual_rl_predictions(
        test_loader, finder_agent, finder_env,
        refiner_agent, refiner_env,
        img_size, cfg, prob_method,
        tag="dual_rl", subdir="test_final", split_name="Test",
    )

    # Salva history refiner (estesa)
    n_eps_r = len(history_refiner["rewards"])
    n_val_r = len(history_refiner["val_dice_rl_mean"])
    pd.DataFrame({
        "episode":        range(1, n_eps_r + 1),
        "reward":         history_refiner["rewards"],
        "reward_smooth":  history_refiner["rewards_smooth"],
        "dice":           history_refiner["dice"],
        "dice_smooth":    history_refiner["dice_smooth"],
        "epsilon":        history_refiner["epsilon"],
    }).to_csv(
        os.path.join(cfg["output"]["metrics_dir"], "roi_refiner_history.csv"),
        index=False
    )
    if n_val_r > 0:
        pd.DataFrame({
            "eval_episode":      history_refiner["val_ep"],
            "val_dice_bl_mean":  history_refiner["val_dice_bl_mean"],
            "val_dice_rl_mean":  history_refiner["val_dice_rl_mean"],
            "val_dice_rl_std":   history_refiner["val_dice_rl_std"],
            "val_dice_rl_median":history_refiner["val_dice_rl_median"],
            "val_iou_rl_mean":   history_refiner["val_iou_rl_mean"],
        }).to_csv(
            os.path.join(cfg["output"]["metrics_dir"], "roi_refiner_eval_history.csv"),
            index=False
        )

    # ── 6) Valutazione pipeline completo su TUTTO il test set ────────────────
    print("\n[pipeline] Step 6: Valutazione finale pipeline doppio RL su TUTTO il test set...")
    results = evaluate_dual_rl(
        test_loader   = test_loader,
        finder_agent  = finder_agent,
        finder_env    = finder_env,
        refiner_agent = refiner_agent,
        refiner_env   = refiner_env,
        img_size      = img_size,
        cfg           = cfg,
        prob_method   = prob_method,
    )

    # ── 7) Salvataggio risultati ───────────────────────────────────────────────
    print("\n[pipeline] Step 7: Salvataggio risultati...")

    skip = {"tp", "fp", "fn", "tn"}
    bl = results["baseline"]
    rl = results["dual_rl"]

    rows = []
    for k in bl:
        if k in skip:
            continue
        rows.append({
            "metric":      k,
            "baseline":    bl[k],
            "dual_rl":     rl.get(k, float("nan")),
            "delta":       rl.get(k, bl[k]) - bl[k],
        })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(cfg["output"]["metrics_dir"], "final_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Metriche → {csv_path}")
    print(df.to_string(index=False))

    # Salva anche metriche per singolo campione (utile per analisi statistica)
    per_sample_rows = []
    for idx, (m_bl, m_rl) in enumerate(zip(results["per_sample_bl"], results["per_sample_rl"])):
        row = {"sample_idx": idx}
        for k in m_bl:
            if k not in skip:
                row[f"bl_{k}"] = m_bl[k]
                row[f"rl_{k}"] = m_rl.get(k, float("nan"))
        per_sample_rows.append(row)
    pd.DataFrame(per_sample_rows).to_csv(
        os.path.join(cfg["output"]["metrics_dir"], "per_sample_results.csv"),
        index=False
    )

    _save_partial_metrics(cfg, "dual_rl_test", refine_episodes, {
        "test_dice_bl_mean":   results["baseline"].get("dice", 0),
        "test_dice_rl_mean":   results["dual_rl"].get("dice", 0),
        "test_dice_rl_std":    results["dice_rl_std"],
        "test_dice_rl_median": results["dice_rl_median"],
        "test_iou_rl_mean":    results["dual_rl"].get("iou", 0),
        "finder_iou_mean":     results["finder_iou"],
        "n_samples":           results["n_samples"],
    }, is_final=True)

    write_report(results, cfg)

    partial_dir = os.path.join(cfg["output"]["metrics_dir"], "partial")

    # ── Fine ──────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(" Pipeline (Doppio RL, No U-Net) completato!")
    print("="*60)
    print(f"  Checkpoint Agent 1 → {cfg['output']['checkpoint_dir']}/best_roi_finder_agent.pth")
    print(f"  Checkpoint Agent 2 → {cfg['output']['checkpoint_dir']}/best_roi_refiner_agent.pth")
    print(f"  Metriche CSV       → {cfg['output']['metrics_dir']}/final_results.csv")
    print(f"  Per-sample CSV     → {cfg['output']['metrics_dir']}/per_sample_results.csv")
    print(f"  Metriche parziali  → {partial_dir}/")
    print(f"  ROI val checkpoints→ {cfg['output']['predictions_dir']}/roi_finder/iter_XXXX/")
    print(f"  ROI test finale    → {roi_preds_dir}/")
    print(f"  Dual RL test finale→ {dual_preds_dir}/")
    print(f"  Report             → {cfg['output']['reports_dir']}/report.md")
    print(f"\n  Campioni di test valutati : {results['n_samples']}")
    print(f"  ROI Finder IoU μ          : {results['finder_iou']:.4f} ± {results['finder_iou_std']:.4f}")
    print(f"  Baseline Dice μ           : {bl.get('dice', 0):.4f} ± {results['dice_bl_std']:.4f}")
    print(f"  Dual RL Dice μ            : {rl.get('dice', 0):.4f} ± {results['dice_rl_std']:.4f}")
    print(f"  Dual RL Dice mediana      : {results['dice_rl_median']:.4f}")
    print(f"  Miglioramento Dice        : {rl.get('dice', 0) - bl.get('dice', 0):+.4f}")


if __name__ == "__main__":
    main()
