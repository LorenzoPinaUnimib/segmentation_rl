"""
visualization.py — All plotting helpers for the full pipeline.

Sections:
  A) Dataset EDA  — after data load
       · grid of images with GT overlaid (contour overlay, not opacity)
       · clean image grid (no annotations)
       · tumor size distribution (px count histogram + box)
       · tumor centroid position scatter (heatmap)
       · pixel intensity distribution (fg vs bg)
       · per-image tumor coverage %
       · aspect-ratio distribution of bounding boxes
       · class balance bar

  B) U-Net evaluation — after baseline training
       · training curves (loss / Dice / IoU)
       · test grid: image | GT | prediction | error map
       · per-sample metric scatter (Dice, IoU, Hausdorff)
       · metric distribution violin plots
       · precision-recall scatter
       · confusion matrix (aggregated pixels)
       · metric summary bar

  C) RL evaluation — after RL training
       · RL training curves (reward, delta, epsilon)
       · test grid: image | GT | baseline | RL | diff map
       · per-sample improvement scatter (RL dice - baseline dice)
       · metric delta violin plots
       · final baseline vs RL bar comparison
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from scipy import ndimage
from typing import List, Dict, Optional, Tuple

# ── colour palette ─────────────────────────────────────────────────────────
C_BLUE   = "#2563EB"
C_RED    = "#DC2626"
C_GREEN  = "#16A34A"
C_ORANGE = "#EA580C"
C_PURPLE = "#7C3AED"
C_GRAY   = "#6B7280"
C_LBLUE  = "#93C5FD"
C_LGREEN = "#86EFAC"


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _to_hw_uint8(t) -> np.ndarray:
    """Tensor/array [C,H,W] or [H,W] → uint8 [H,W,C] or [H,W]."""
    if hasattr(t, "cpu"):
        t = t.cpu().numpy()
    t = np.squeeze(t)
    if t.ndim == 3:
        t = np.transpose(t, (1, 2, 0))
    t = t.astype(np.float32)
    if t.max() <= 1.01:
        t = t * 255.0
    return np.clip(t, 0, 255).astype(np.uint8)


def _overlay_contour(img_rgb: np.ndarray, mask_hw: np.ndarray,
                     color=(255, 50, 50), thickness: int = 2) -> np.ndarray:
    """Draw mask boundary contour on RGB image."""
    import cv2
    result = img_rgb.copy()
    mask_u8 = (mask_hw > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, color, thickness)
    return result


def _save(fig, path: str, dpi: int = 150) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[viz] Saved {path}")


def _styled_fig(nrows, ncols, **kw):
    fig, axes = plt.subplots(nrows, ncols, **kw)
    fig.patch.set_facecolor("#F8FAFC")
    return fig, axes


def _ax_style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor("#F1F5F9")
    ax.grid(True, color="white", linewidth=0.8, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    if title:  ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    if xlabel: ax.set_xlabel(xlabel, fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)


# ═══════════════════════════════════════════════════════════════════════════════
# A) DATASET EDA
# ═══════════════════════════════════════════════════════════════════════════════

def plot_dataset_eda(dataset, save_dir: str, n_samples: int = 16,
                     tag: str = "dataset") -> None:
    """
    Full EDA report for the loaded dataset.
    dataset: a torch Dataset returning {"image": tensor, "mask": tensor}
    """
    os.makedirs(save_dir, exist_ok=True)

    # ── collect samples ──────────────────────────────────────────────────────
    indices = np.linspace(0, len(dataset) - 1, min(n_samples * 4, len(dataset)),
                          dtype=int)
    images, masks = [], []
    for idx in indices:
        sample = dataset[int(idx)]
        images.append(_to_hw_uint8(sample["image"]))
        masks.append(sample["mask"].numpy().squeeze())

    # ── A1: image grid with GT contour overlay ───────────────────────────────
    _plot_image_grid_overlay(images[:n_samples], masks[:n_samples],
                             os.path.join(save_dir, f"{tag}_images_with_gt.png"))

    # ── A2: clean image grid ─────────────────────────────────────────────────
    _plot_image_grid_clean(images[:n_samples],
                           os.path.join(save_dir, f"{tag}_images_clean.png"))

    # ── A3-A8: statistics on all collected samples ───────────────────────────
    _plot_eda_statistics(images, masks, save_dir, tag)


def _plot_image_grid_overlay(images, masks, path, cols=4):
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    fig.patch.set_facecolor("#0F172A")
    fig.suptitle("Dataset samples — GT contour overlay (red)",
                 color="white", fontsize=13, fontweight="bold", y=1.01)
    axes = np.array(axes).flatten()
    for i, ax in enumerate(axes):
        if i < len(images):
            img = images[i]
            if img.ndim == 2:
                img = np.stack([img]*3, axis=-1)
            vis = _overlay_contour(img, masks[i])
            ax.imshow(vis)
            has_tumor = masks[i].sum() > 0
            ax.set_title(f"#{i+1} {'[tumor]' if has_tumor else '[no tumor]'}",
                         color="white", fontsize=7)
        ax.axis("off")
    fig.tight_layout(pad=0.3)
    _save(fig, path, dpi=130)


def _plot_image_grid_clean(images, path, cols=4):
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    fig.patch.set_facecolor("#0F172A")
    fig.suptitle("Dataset samples — images only",
                 color="white", fontsize=13, fontweight="bold", y=1.01)
    axes = np.array(axes).flatten()
    for i, ax in enumerate(axes):
        if i < len(images):
            img = images[i]
            ax.imshow(img if img.ndim == 3 else img, cmap="gray" if images[i].ndim==2 else None)
            ax.set_title(f"#{i+1}", color="white", fontsize=7)
        ax.axis("off")
    fig.tight_layout(pad=0.3)
    _save(fig, path, dpi=130)


def _plot_eda_statistics(images, masks, save_dir, tag):
    # ── compute per-sample stats ─────────────────────────────────────────────
    tumor_px, coverages, centroids_xy, bbox_aspects = [], [], [], []
    fg_intensities, bg_intensities = [], []

    for img, mask in zip(images, masks):
        bin_mask = (mask > 0.5)
        px_count = int(bin_mask.sum())
        tumor_px.append(px_count)
        total = mask.size
        coverages.append(100.0 * px_count / total)

        if px_count > 0:
            coords = np.argwhere(bin_mask)
            cy, cx = coords.mean(axis=0)
            centroids_xy.append((cx / mask.shape[1], cy / mask.shape[0]))

            # bounding box aspect ratio
            r_min, c_min = coords.min(axis=0)
            r_max, c_max = coords.max(axis=0)
            h = max(r_max - r_min, 1)
            w = max(c_max - c_min, 1)
            bbox_aspects.append(w / h)

        # pixel intensities
        gray = img.mean(axis=-1) if img.ndim == 3 else img.astype(float)
        gray_norm = gray / 255.0
        if px_count > 0:
            fg_intensities.extend(gray_norm[bin_mask].tolist())
            bg_intensities.extend(gray_norm[~bin_mask].tolist())

    n_with_tumor = sum(1 for p in tumor_px if p > 0)
    n_total = len(tumor_px)

    # ── fig: 6-panel statistics ─────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 12), facecolor="#F8FAFC")
    fig.suptitle("Dataset EDA — Statistics", fontsize=15, fontweight="bold",
                 color="#1E293B", y=0.98)
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Panel 1: tumor size distribution
    ax1 = fig.add_subplot(gs[0, 0])
    pos_px = [p for p in tumor_px if p > 0]
    if pos_px:
        ax1.hist(pos_px, bins=25, color=C_BLUE, edgecolor="white", linewidth=0.5)
    _ax_style(ax1, "Tumor Size Distribution", "Pixels", "Count")
    ax1.axvline(np.mean(pos_px) if pos_px else 0, color=C_RED,
                linestyle="--", linewidth=1.5, label=f"Mean={np.mean(pos_px):.0f}" if pos_px else "")
    if pos_px:
        ax1.legend(fontsize=7)

    # Panel 2: centroid scatter / heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    if centroids_xy:
        xs = [c[0] for c in centroids_xy]
        ys = [c[1] for c in centroids_xy]
        h2d, xedge, yedge = np.histogram2d(xs, ys, bins=20,
                                             range=[[0,1],[0,1]])
        im = ax2.imshow(h2d.T, origin="lower", extent=[0,1,0,1],
                        cmap="YlOrRd", aspect="auto")
        plt.colorbar(im, ax=ax2, shrink=0.8, label="count")
        ax2.scatter(xs, ys, s=6, color="white", alpha=0.4, linewidth=0)
    _ax_style(ax2, "Tumor Centroid Heatmap", "X (normalised)", "Y (normalised)")

    # Panel 3: coverage % histogram
    ax3 = fig.add_subplot(gs[0, 2])
    pos_cov = [c for c in coverages if c > 0]
    if pos_cov:
        ax3.hist(pos_cov, bins=25, color=C_GREEN, edgecolor="white", linewidth=0.5)
    _ax_style(ax3, "Tumor Coverage per Image", "Coverage (%)", "Count")
    ax3.axvline(np.mean(pos_cov) if pos_cov else 0, color=C_RED,
                linestyle="--", linewidth=1.5,
                label=f"Mean={np.mean(pos_cov):.2f}%" if pos_cov else "")
    if pos_cov:
        ax3.legend(fontsize=7)

    # Panel 4: pixel intensity distribution fg vs bg
    ax4 = fig.add_subplot(gs[1, 0])
    sample_n = min(50000, len(fg_intensities), len(bg_intensities))
    rng = np.random.default_rng(0)
    if fg_intensities and bg_intensities:
        fg_s = rng.choice(fg_intensities, size=sample_n, replace=True)
        bg_s = rng.choice(bg_intensities, size=sample_n, replace=True)
        ax4.hist(bg_s, bins=60, color=C_GRAY, alpha=0.6, density=True,
                 label="Background", edgecolor="none")
        ax4.hist(fg_s, bins=60, color=C_RED, alpha=0.7, density=True,
                 label="Tumor", edgecolor="none")
    _ax_style(ax4, "Pixel Intensity: Tumor vs Background",
              "Normalised Intensity", "Density")
    ax4.legend(fontsize=7)

    # Panel 5: bbox aspect ratio
    ax5 = fig.add_subplot(gs[1, 1])
    if bbox_aspects:
        ax5.hist(bbox_aspects, bins=25, color=C_PURPLE,
                 edgecolor="white", linewidth=0.5)
    _ax_style(ax5, "Tumor Bounding-Box Aspect Ratio (W/H)",
              "Width / Height", "Count")
    ax5.axvline(1.0, color=C_GRAY, linestyle="--", linewidth=1, label="Square")
    ax5.legend(fontsize=7)

    # Panel 6: class balance
    ax6 = fig.add_subplot(gs[1, 2])
    n_no_tumor = n_total - n_with_tumor
    bars = ax6.bar(["With Tumor", "No Tumor / Empty"],
                   [n_with_tumor, n_no_tumor],
                   color=[C_RED, C_GRAY], edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, [n_with_tumor, n_no_tumor]):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{val}\n({100*val/n_total:.1f}%)", ha="center",
                 va="bottom", fontsize=8, fontweight="bold")
    _ax_style(ax6, "Class Balance (Tumor Present?)", "", "Images")

    _save(fig, os.path.join(save_dir, f"{tag}_eda_statistics.png"), dpi=130)


# ═══════════════════════════════════════════════════════════════════════════════
# B) U-NET EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def plot_training_curves(history: dict, save_dir: str) -> None:
    """Loss, Dice, IoU training curves."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = _styled_fig(1, 3, figsize=(15, 4))
    fig.suptitle("U-Net Training History", fontsize=13, fontweight="bold",
                 color="#1E293B")

    pairs = [
        ("train_loss",  "val_loss",  "Loss",            None),
        ("train_dice",  "val_dice",  "Dice Coefficient", (0, 1)),
        ("train_iou",   "val_iou",   "IoU (Jaccard)",    (0, 1)),
    ]
    for ax, (tr_key, va_key, title, ylim) in zip(axes, pairs):
        ax.plot(epochs, history[tr_key], color=C_BLUE, linewidth=1.8,
                label="Train")
        ax.plot(epochs, history[va_key], color=C_RED, linewidth=1.8,
                label="Val", linestyle="--")
        _ax_style(ax, title, "Epoch")
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(fontsize=8)
        # mark best val
        best_ep = int(np.argmin(history[va_key]) if "loss" in va_key
                      else np.argmax(history[va_key])) + 1
        best_v  = (min(history[va_key]) if "loss" in va_key
                   else max(history[va_key]))
        ax.axvline(best_ep, color=C_ORANGE, linewidth=1, linestyle=":",
                   label=f"Best ep={best_ep}")
        ax.legend(fontsize=7)

    fig.tight_layout()
    _save(fig, os.path.join(save_dir, "training_curves.png"))


def plot_unet_test_results(
    images: list, gt_masks: list, predictions: list,
    per_sample_metrics: list, save_dir: str, n_show: int = 12
) -> None:
    """
    Full U-Net test evaluation report:
      · qualitative grid (image | GT overlay | pred overlay | error map)
      · per-sample metric scatter
      · metric violin distributions
      · confusion matrix (pixel-level)
      · metric summary bar
    """
    os.makedirs(save_dir, exist_ok=True)
    n = min(n_show, len(images))

    # ── B1: qualitative grid ─────────────────────────────────────────────────
    _plot_unet_qual_grid(images[:n], gt_masks[:n], predictions[:n],
                         os.path.join(save_dir, "unet_qualitative.png"))

    # ── B2-B5: metric plots ──────────────────────────────────────────────────
    _plot_unet_metrics(per_sample_metrics, save_dir)


def _plot_unet_qual_grid(images, gts, preds, path):
    n = len(images)
    cols = 4   # image | gt_overlay | pred_overlay | error
    fig, axes = plt.subplots(n, cols, figsize=(cols * 3.2, n * 3.2))
    fig.patch.set_facecolor("#0F172A")
    fig.suptitle("U-Net Test Predictions", color="white",
                 fontsize=13, fontweight="bold", y=1.005)
    col_titles = ["Image", "Ground Truth", "Prediction", "Error Map"]
    if n == 1:
        axes = [axes]
    for i in range(n):
        img  = _to_hw_uint8(images[i])
        gt   = gts[i].numpy().squeeze() if hasattr(gts[i], "numpy") else np.squeeze(gts[i])
        pred = preds[i].numpy().squeeze() if hasattr(preds[i], "numpy") else np.squeeze(preds[i])

        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)

        gt_vis   = _overlay_contour(img, gt,   color=(50, 255, 50))
        pred_vis = _overlay_contour(img, pred, color=(255, 100, 50))

        # error map: FP = orange, FN = blue
        gt_b   = (gt > 0.5)
        pred_b = (pred > 0.5)
        err_rgb = np.zeros((*gt.shape, 3), dtype=np.uint8)
        err_rgb[gt_b   & ~pred_b] = [50, 100, 255]   # FN blue
        err_rgb[~gt_b  &  pred_b] = [255, 120, 50]   # FP orange
        err_rgb[gt_b   &  pred_b] = [80, 200, 80]    # TP green

        row_imgs = [img, gt_vis, pred_vis, err_rgb]
        for j, (ax, im) in enumerate(zip(axes[i], row_imgs)):
            ax.imshow(im)
            ax.axis("off")
            if i == 0:
                ax.set_title(col_titles[j], color="white",
                             fontsize=9, fontweight="bold")
        # row label
        axes[i][0].set_ylabel(f"#{i+1}", color="white", fontsize=8,
                               rotation=0, labelpad=25)
    # legend
    legend_els = [
        mpatches.Patch(color="#50C850", label="True Positive"),
        mpatches.Patch(color="#3264FF", label="False Negative"),
        mpatches.Patch(color="#FF7832", label="False Positive"),
    ]
    fig.legend(handles=legend_els, loc="lower center", ncol=3,
               facecolor="#1E293B", labelcolor="white", fontsize=8,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(pad=0.3)
    _save(fig, path, dpi=120)


def _plot_unet_metrics(per_sample_metrics: list, save_dir: str):
    if not per_sample_metrics:
        return

    keys_main = ["dice", "iou", "precision", "recall", "specificity", "pixel_acc"]
    keys_show = [k for k in keys_main if k in per_sample_metrics[0]]

    values = {k: [m[k] for m in per_sample_metrics] for k in keys_show}
    hausdorff = [m["hausdorff"] for m in per_sample_metrics
                 if "hausdorff" in m and not np.isnan(m.get("hausdorff", np.nan))]

    # ── B2: violin distributions ─────────────────────────────────────────────
    fig, axes = _styled_fig(1, 2, figsize=(14, 5))
    fig.suptitle("U-Net Test — Metric Distributions", fontsize=12,
                 fontweight="bold", color="#1E293B")

    vparts = axes[0].violinplot([values[k] for k in keys_show],
                                positions=range(len(keys_show)),
                                showmedians=True, showextrema=True)
    for pc in vparts["bodies"]:
        pc.set_facecolor(C_LBLUE); pc.set_alpha(0.7)
    vparts["cmedians"].set_color(C_BLUE); vparts["cmedians"].set_linewidth(2)
    axes[0].set_xticks(range(len(keys_show)))
    axes[0].set_xticklabels([k.replace("_"," ").title() for k in keys_show],
                             fontsize=8)
    axes[0].set_ylim(0, 1.05)
    # overlay individual points
    for j, k in enumerate(keys_show):
        jitter = np.random.default_rng(j).uniform(-0.08, 0.08, len(values[k]))
        axes[0].scatter(j + jitter, values[k], s=8, color=C_BLUE,
                        alpha=0.4, linewidth=0)
    _ax_style(axes[0], "Metric Distribution (all test samples)", "", "Score")

    # Hausdorff
    if hausdorff:
        axes[1].hist(hausdorff, bins=30, color=C_ORANGE,
                     edgecolor="white", linewidth=0.5)
        axes[1].axvline(np.mean(hausdorff), color=C_RED, linewidth=1.5,
                        linestyle="--",
                        label=f"Mean={np.mean(hausdorff):.1f}")
        axes[1].legend(fontsize=8)
    _ax_style(axes[1], "Hausdorff Distance Distribution", "Pixels", "Count")

    fig.tight_layout()
    _save(fig, os.path.join(save_dir, "unet_metric_distributions.png"))

    # ── B3: precision-recall scatter ─────────────────────────────────────────
    if "precision" in values and "recall" in values:
        fig, ax = plt.subplots(figsize=(6, 6), facecolor="#F8FAFC")
        sc = ax.scatter(values["recall"], values["precision"],
                        c=values["dice"], cmap="RdYlGn",
                        s=30, alpha=0.7, edgecolors="white", linewidth=0.3,
                        vmin=0, vmax=1)
        plt.colorbar(sc, ax=ax, label="Dice")
        ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.05)
        # iso-F1 curves
        for f1 in [0.3, 0.5, 0.7, 0.9]:
            r_arr = np.linspace(0.01, 1, 200)
            p_arr = f1 * r_arr / (2 * r_arr - f1)
            p_arr = np.clip(p_arr, 0, 1)
            valid = p_arr > 0
            ax.plot(r_arr[valid], p_arr[valid], color=C_GRAY,
                    linewidth=0.8, linestyle="--", alpha=0.5)
            idx = len(r_arr) // 2
            ax.text(r_arr[valid][idx], p_arr[valid][idx],
                    f"F1={f1}", fontsize=6, color=C_GRAY)
        _ax_style(ax, "Precision-Recall (coloured by Dice)", "Recall", "Precision")
        _save(fig, os.path.join(save_dir, "unet_precision_recall.png"))

    # ── B4: aggregated pixel confusion matrix ─────────────────────────────────
    # recompute from predictions stored in per_sample_metrics via TP/FP/FN/TN
    # We stored them; if not available, skip gracefully
    tp_all = sum(m.get("tp", 0) for m in per_sample_metrics)
    fp_all = sum(m.get("fp", 0) for m in per_sample_metrics)
    fn_all = sum(m.get("fn", 0) for m in per_sample_metrics)
    tn_all = sum(m.get("tn", 0) for m in per_sample_metrics)
    if tp_all + fp_all + fn_all + tn_all > 0:
        _plot_confusion_matrix(tp_all, fp_all, fn_all, tn_all,
                               os.path.join(save_dir, "unet_confusion_matrix.png"),
                               title="U-Net Pixel Confusion Matrix")

    # ── B5: mean metric bar ───────────────────────────────────────────────────
    means = {k: float(np.mean(values[k])) for k in keys_show}
    _plot_metric_bar(means, os.path.join(save_dir, "unet_metric_summary.png"),
                     title="U-Net Test Set — Mean Metrics", color=C_BLUE)


def _plot_confusion_matrix(tp, fp, fn, tn, path, title="Confusion Matrix"):
    cm = np.array([[tn, fp], [fn, tp]], dtype=np.float64)
    total = cm.sum()
    cm_pct = cm / (total + 1e-8) * 100

    fig, ax = plt.subplots(figsize=(5, 4), facecolor="#F8FAFC")
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, label="%")
    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels[i][j]}\n{cm[i,j]:.2e}\n({cm_pct[i,j]:.1f}%)",
                    ha="center", va="center", fontsize=9,
                    color="white" if cm_pct[i,j] > 50 else "#1E293B",
                    fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred Neg", "Pred Pos"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual Neg", "Actual Pos"])
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    fig.tight_layout()
    _save(fig, path)


def _plot_metric_bar(means: dict, path: str, title: str, color=C_BLUE):
    keys = list(means.keys())
    vals = [means[k] for k in keys]
    fig, ax = plt.subplots(figsize=(9, 4), facecolor="#F8FAFC")
    bars = ax.bar(range(len(keys)), vals, color=color, edgecolor="white",
                  linewidth=0.5, width=0.6)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.4f}", ha="center", va="bottom", fontsize=8,
                fontweight="bold")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([k.replace("_"," ").title() for k in keys], fontsize=9)
    ax.set_ylim(0, 1.1)
    _ax_style(ax, title, "", "Score")
    fig.tight_layout()
    _save(fig, path)


# ═══════════════════════════════════════════════════════════════════════════════
# C) RL EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def plot_rl_curves(rl_history: dict, save_dir: str) -> None:
    """RL training curves: reward, delta, epsilon, val improvement."""
    n = len(rl_history["rewards"])
    episodes = np.arange(1, n + 1)
    window = max(1, n // 20)

    def smooth(arr):
        arr = np.array(arr, dtype=float)
        return np.convolve(arr, np.ones(window)/window, mode="valid")

    has_val = len(rl_history.get("val_improvement", [])) > 0

    ncols = 4 if has_val else 3
    fig, axes = _styled_fig(1, ncols, figsize=(ncols * 4.5, 4.5))
    fig.suptitle("RL Agent Training", fontsize=13, fontweight="bold",
                 color="#1E293B")

    # reward
    rewards = np.array(rl_history["rewards"], dtype=float)
    axes[0].fill_between(episodes, rewards, alpha=0.2, color=C_BLUE)
    axes[0].plot(episodes, rewards, color=C_LBLUE, linewidth=0.6, alpha=0.6)
    sm = smooth(rewards)
    axes[0].plot(np.arange(window, n+1), sm, color=C_BLUE, linewidth=2)
    _ax_style(axes[0], "Episode Reward", "Episode", "Reward")

    # delta dice
    di = np.array(rl_history.get("dice_improvement", [0]*n), dtype=float)
    axes[1].fill_between(episodes, di, where=di>=0, alpha=0.4, color=C_GREEN)
    axes[1].fill_between(episodes, di, where=di<0,  alpha=0.4, color=C_RED)
    axes[1].plot(episodes, di, color=C_GRAY, linewidth=0.5, alpha=0.5)
    sm2 = smooth(di)
    axes[1].plot(np.arange(window, n+1), sm2, color=C_GREEN, linewidth=2)
    axes[1].axhline(0, color="#1E293B", linewidth=1)
    _ax_style(axes[1], "Dice Improvement per Episode\n(RL - Baseline)", "Episode", "Delta Dice")

    # epsilon
    eps = np.array(rl_history.get("epsilon", [1.0]*n), dtype=float)
    axes[2].plot(episodes, eps, color=C_ORANGE, linewidth=2)
    axes[2].fill_between(episodes, eps, alpha=0.15, color=C_ORANGE)
    axes[2].set_ylim(0, 1.05)
    _ax_style(axes[2], "Epsilon (Exploration Decay)", "Episode", "Epsilon")

    # val improvement
    if has_val:
        vi = rl_history["val_improvement"]
        x_val = np.linspace(1, n, len(vi))
        axes[3].plot(x_val, vi, color=C_PURPLE, linewidth=2, marker="o",
                     markersize=5, markerfacecolor="white")
        axes[3].axhline(0, color="#1E293B", linewidth=1, linestyle="--")
        best_idx = int(np.argmax(vi))
        axes[3].scatter(x_val[best_idx], vi[best_idx], s=80, color=C_RED,
                        zorder=5, label=f"Best={vi[best_idx]:.4f}")
        axes[3].legend(fontsize=8)
        _ax_style(axes[3], "Validation Dice Improvement", "Episode", "Delta Dice")

    fig.tight_layout()
    _save(fig, os.path.join(save_dir, "rl_curves.png"))


def plot_rl_test_results(
    images: list, gt_masks: list,
    baseline_preds: list, rl_preds: list,
    bl_metrics_per_sample: list, rl_metrics_per_sample: list,
    save_dir: str, n_show: int = 12
) -> None:
    """
    Full RL test evaluation report:
      · qualitative grid (image | GT | baseline | RL | diff)
      · per-sample improvement scatter
      · delta violin distributions
    """
    os.makedirs(save_dir, exist_ok=True)
    n = min(n_show, len(images))

    _plot_rl_qual_grid(images[:n], gt_masks[:n],
                       baseline_preds[:n], rl_preds[:n],
                       bl_metrics_per_sample[:n], rl_metrics_per_sample[:n],
                       os.path.join(save_dir, "rl_qualitative.png"))

    _plot_rl_improvement(bl_metrics_per_sample, rl_metrics_per_sample, save_dir)


def _plot_rl_qual_grid(images, gts, bl_preds, rl_preds,
                       bl_ms, rl_ms, path):
    n = len(images)
    cols = 5  # image | GT | baseline | RL | diff
    fig, axes = plt.subplots(n, cols, figsize=(cols * 3.2, n * 3.2))
    fig.patch.set_facecolor("#0F172A")
    fig.suptitle("RL Refinement — Test Predictions", color="white",
                 fontsize=13, fontweight="bold", y=1.005)
    col_titles = ["Image", "Ground Truth", "Baseline U-Net",
                  "RL Refined", "Diff (RL - Baseline)"]
    if n == 1:
        axes = [axes]
    for i in range(n):
        img = _to_hw_uint8(images[i])
        gt  = gts[i].numpy().squeeze() if hasattr(gts[i], "numpy") else np.squeeze(gts[i])
        bl  = bl_preds[i].numpy().squeeze() if hasattr(bl_preds[i], "numpy") else np.squeeze(bl_preds[i])
        rl  = rl_preds[i].numpy().squeeze() if hasattr(rl_preds[i], "numpy") else np.squeeze(rl_preds[i])

        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)

        gt_vis = _overlay_contour(img, gt,  color=(50, 255, 50))
        bl_vis = _overlay_contour(img, bl,  color=(255, 100, 50))
        rl_vis = _overlay_contour(img, rl,  color=(100, 150, 255))

        # diff: pixels gained (green) / lost (red) by RL vs baseline
        bl_b = (bl > 0.5); rl_b = (rl > 0.5); gt_b = (gt > 0.5)
        diff_rgb = np.zeros((*gt.shape, 3), dtype=np.uint8)
        diff_rgb[rl_b  & ~bl_b] = [80, 220, 80]    # gained
        diff_rgb[~rl_b &  bl_b] = [220, 80, 80]    # lost
        diff_rgb[rl_b  &  bl_b] = [180, 180, 180]  # unchanged positive

        row_imgs = [img, gt_vis, bl_vis, rl_vis, diff_rgb]
        for j, (ax, im) in enumerate(zip(axes[i], row_imgs)):
            ax.imshow(im)
            ax.axis("off")
            if i == 0:
                ax.set_title(col_titles[j], color="white",
                             fontsize=9, fontweight="bold")

        # annotate dice on baseline and RL
        bl_d = bl_ms[i].get("dice", 0) if i < len(bl_ms) else 0
        rl_d = rl_ms[i].get("dice", 0) if i < len(rl_ms) else 0
        delta = rl_d - bl_d
        col = C_GREEN if delta >= 0 else C_RED
        axes[i][2].set_xlabel(f"Dice={bl_d:.3f}", color="wheat", fontsize=7)
        axes[i][3].set_xlabel(
            f"Dice={rl_d:.3f}  ({delta:+.3f})",
            color=col, fontsize=7, fontweight="bold"
        )

    legend_els = [
        mpatches.Patch(color="#50DC50", label="RL gained"),
        mpatches.Patch(color="#DC5050", label="RL lost"),
        mpatches.Patch(color="#B4B4B4", label="Unchanged positive"),
    ]
    fig.legend(handles=legend_els, loc="lower center", ncol=3,
               facecolor="#1E293B", labelcolor="white", fontsize=8,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(pad=0.3)
    _save(fig, path, dpi=120)


def _plot_rl_improvement(bl_ms: list, rl_ms: list, save_dir: str):
    if not bl_ms or not rl_ms:
        return

    keys = ["dice", "iou", "precision", "recall", "specificity"]
    keys = [k for k in keys if k in bl_ms[0] and k in rl_ms[0]]

    bl_vals = {k: np.array([m[k] for m in bl_ms]) for k in keys}
    rl_vals = {k: np.array([m[k] for m in rl_ms]) for k in keys}
    deltas  = {k: rl_vals[k] - bl_vals[k] for k in keys}

    # ── per-sample dice improvement scatter ──────────────────────────────────
    fig, axes = _styled_fig(1, 2, figsize=(13, 5))
    fig.suptitle("RL Test — Improvement over Baseline", fontsize=12,
                 fontweight="bold", color="#1E293B")

    n = len(bl_vals["dice"])
    ax = axes[0]
    colors = [C_GREEN if d >= 0 else C_RED for d in deltas["dice"]]
    ax.bar(range(n), deltas["dice"], color=colors, edgecolor="none", width=1.0)
    ax.axhline(0, color="#1E293B", linewidth=1)
    ax.axhline(np.mean(deltas["dice"]), color=C_BLUE, linewidth=1.5,
               linestyle="--",
               label=f"Mean delta={np.mean(deltas['dice']):+.4f}")
    ax.legend(fontsize=8)
    _ax_style(ax, "Per-Sample Dice Improvement (RL - Baseline)",
              "Sample Index", "Delta Dice")

    # ── delta violin per metric ───────────────────────────────────────────────
    ax2 = axes[1]
    data = [deltas[k] for k in keys]
    vp = ax2.violinplot(data, positions=range(len(keys)),
                        showmedians=True, showextrema=True)
    for pc in vp["bodies"]:
        pc.set_facecolor(C_LBLUE); pc.set_alpha(0.6)
    vp["cmedians"].set_color(C_BLUE); vp["cmedians"].set_linewidth(2)
    ax2.axhline(0, color=C_RED, linewidth=1, linestyle="--")
    ax2.set_xticks(range(len(keys)))
    ax2.set_xticklabels([k.replace("_"," ").title() for k in keys], fontsize=8)
    _ax_style(ax2, "Metric Deltas (RL - Baseline)\nDistribution", "", "Delta")

    fig.tight_layout()
    _save(fig, os.path.join(save_dir, "rl_improvement.png"))

    # ── side-by-side confusion matrices ──────────────────────────────────────
    tp_bl = sum(m.get("tp",0) for m in bl_ms)
    fp_bl = sum(m.get("fp",0) for m in bl_ms)
    fn_bl = sum(m.get("fn",0) for m in bl_ms)
    tn_bl = sum(m.get("tn",0) for m in bl_ms)
    tp_rl = sum(m.get("tp",0) for m in rl_ms)
    fp_rl = sum(m.get("fp",0) for m in rl_ms)
    fn_rl = sum(m.get("fn",0) for m in rl_ms)
    tn_rl = sum(m.get("tn",0) for m in rl_ms)
    if tp_bl + fp_bl + fn_bl + tn_bl > 0:
        _plot_confusion_matrix(tp_bl, fp_bl, fn_bl, tn_bl,
                               os.path.join(save_dir, "rl_confusion_baseline.png"),
                               title="Baseline Pixel Confusion Matrix")
    if tp_rl + fp_rl + fn_rl + tn_rl > 0:
        _plot_confusion_matrix(tp_rl, fp_rl, fn_rl, tn_rl,
                               os.path.join(save_dir, "rl_confusion_rl.png"),
                               title="RL Refined Pixel Confusion Matrix")


# ═══════════════════════════════════════════════════════════════════════════════
# D) FINAL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

def plot_comparison(baseline_metrics: dict, rl_metrics: dict,
                    save_dir: str) -> None:
    """Grouped bar chart + delta bars comparing baseline vs RL."""
    keys   = ["dice", "iou", "precision", "recall", "specificity"]
    labels = ["Dice", "IoU", "Precision", "Recall", "Specificity"]
    bl_v   = [baseline_metrics.get(k, 0) for k in keys]
    rl_v   = [rl_metrics.get(k, 0) for k in keys]
    deltas = [r - b for r, b in zip(rl_v, bl_v)]

    fig, axes = _styled_fig(1, 2, figsize=(14, 5))
    fig.suptitle("Final Comparison: Baseline U-Net vs RL Refined",
                 fontsize=13, fontweight="bold", color="#1E293B")

    x = np.arange(len(labels))
    w = 0.36
    bars_bl = axes[0].bar(x - w/2, bl_v, w, label="Baseline U-Net",
                          color=C_BLUE, edgecolor="white", linewidth=0.5)
    bars_rl = axes[0].bar(x + w/2, rl_v, w, label="RL Refined",
                          color=C_GREEN, edgecolor="white", linewidth=0.5)
    for bar, v in zip(list(bars_bl) + list(bars_rl), bl_v + rl_v):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f"{v:.3f}", ha="center", va="bottom", fontsize=7,
                     fontweight="bold")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_ylim(0, 1.12)
    axes[0].legend(fontsize=9)
    _ax_style(axes[0], "Absolute Scores", "", "Score")

    # delta bars
    colors_d = [C_GREEN if d >= 0 else C_RED for d in deltas]
    bars_d = axes[1].bar(x, deltas, w * 1.5, color=colors_d,
                         edgecolor="white", linewidth=0.5)
    axes[1].axhline(0, color="#1E293B", linewidth=1)
    for bar, d in zip(bars_d, deltas):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + (0.001 if d >= 0 else -0.003),
                     f"{d:+.4f}", ha="center",
                     va="bottom" if d >= 0 else "top",
                     fontsize=8, fontweight="bold")
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=9)
    _ax_style(axes[1], "Delta (RL - Baseline)", "", "Delta Score")

    fig.tight_layout()
    _save(fig, os.path.join(save_dir, "baseline_vs_rl.png"))


# keep old name as alias for backward compat
def save_qualitative_grid(*args, **kwargs):
    """Legacy alias — use plot_unet_test_results or plot_rl_test_results instead."""
    pass