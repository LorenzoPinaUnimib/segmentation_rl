#!/usr/bin/env python3
"""
run_full_pipeline.py
Full end-to-end pipeline with comprehensive visualisations at every stage:
  1) Load dataset  → EDA figures (grid+overlay, stats, distributions)
  2) Train U-Net   → training curves
  3) Eval U-Net    → qualitative grid, metric distributions, PR scatter,
                     confusion matrix, summary bar
  4) Train RL      → RL training curves
  5) Eval RL       → qualitative grid, improvement scatter, delta violins,
                     confusion matrices, final comparison
  6) Reports & CSVs
"""
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import torch

from src.utils.io_utils import load_config, set_seed, get_device, ensure_dirs
from src.utils.visualization import (
    plot_dataset_eda,
    plot_training_curves,
    plot_unet_test_results,
    plot_rl_curves,
    plot_rl_test_results,
    plot_comparison,
)
from src.data.dataset import get_datasets, make_dataloaders
from src.models.unet import build_unet, build_loss
from src.train.trainer import BaselineTrainer, build_optimizer_scheduler
from src.train.rl_trainer import RLTrainer
from src.rl.environment import MaskRefinementEnv
from src.rl.agent import DQNAgent
from src.eval.metrics import compute_metrics, average_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Step helpers
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_baseline(model, test_loader, device, cfg):
    """
    Run baseline U-Net on the test set.
    Returns:
        avg_metrics   dict of averaged metrics
        per_sample    list of per-sample metric dicts (incl. tp/fp/fn/tn)
        sample_imgs   list of image tensors (for viz)
        sample_gts    list of gt mask tensors
        sample_preds  list of prediction tensors
    """
    model.eval()
    all_metrics = []
    sample_imgs, sample_gts, sample_preds = [], [], []
    do_hausdorff = cfg["evaluation"].get("hausdorff", True)
    n_qual = cfg["evaluation"].get("num_qualitative", 12)

    with torch.no_grad():
        for batch in test_loader:
            imgs    = batch["image"].to(device)
            targets = batch["mask"]
            logits  = model(imgs)
            probs   = torch.sigmoid(logits).cpu()

            for i in range(imgs.size(0)):
                pred = (probs[i] > 0.5).float()
                gt   = targets[i]
                m = compute_metrics(pred.numpy(), gt.numpy(),
                                    compute_hausdorff=do_hausdorff)
                all_metrics.append(m)
                if len(sample_imgs) < n_qual:
                    sample_imgs.append(imgs[i].cpu())
                    sample_gts.append(gt)
                    sample_preds.append(pred)

    avg = average_metrics(all_metrics)
    print("\n[eval] Baseline test metrics:")
    for k, v in avg.items():
        if k not in ("tp", "fp", "fn", "tn"):
            print(f"  {k:12s}: {v:.4f}")
    return avg, all_metrics, sample_imgs, sample_gts, sample_preds


def evaluate_rl(model, agent, env, test_loader, device, cfg):
    """
    Run RL agent on test set.
    Returns avg_metrics, per_sample_bl, per_sample_rl,
            sample_imgs, sample_gts, sample_bl_preds, sample_rl_preds
    """
    model.eval()
    agent.online.eval()
    all_bl_metrics, all_rl_metrics = [], []
    sample_imgs, sample_gts = [], []
    sample_bl_preds, sample_rl_preds = [], []
    n_qual = cfg["evaluation"].get("num_qualitative", 12)
    do_hausdorff = cfg["evaluation"].get("hausdorff", True)

    with torch.no_grad():
        for batch in test_loader:
            imgs    = batch["image"].to(device)
            targets = batch["mask"]
            logits  = model(imgs)
            probs   = torch.sigmoid(logits).cpu()

            for i in range(imgs.size(0)):
                img_np  = imgs[i].cpu().numpy()
                if img_np.ndim == 3:
                    img_np = img_np.transpose(1, 2, 0)
                prob_np = probs[i].numpy().squeeze()
                gt_np   = targets[i].numpy().squeeze()

                # baseline mask
                bl_mask = (prob_np > 0.5).astype(np.float32)
                m_bl = compute_metrics(bl_mask, gt_np,
                                       compute_hausdorff=do_hausdorff)
                all_bl_metrics.append(m_bl)

                # RL refined mask
                state = env.reset(img_np, prob_np, gt_np)
                done = False
                while not done:
                    action = agent.act(state, greedy=True)
                    state, _, done, _ = env.step(action)
                refined = env.get_refined_mask()
                m_rl = compute_metrics(refined, gt_np,
                                       compute_hausdorff=do_hausdorff)
                all_rl_metrics.append(m_rl)

                if len(sample_imgs) < n_qual:
                    sample_imgs.append(imgs[i].cpu())
                    sample_gts.append(targets[i])
                    sample_bl_preds.append(
                        torch.from_numpy(bl_mask).unsqueeze(0))
                    sample_rl_preds.append(
                        torch.from_numpy(refined).unsqueeze(0))

    avg_rl = average_metrics(all_rl_metrics)
    print("\n[eval] RL refined test metrics:")
    for k, v in avg_rl.items():
        if k not in ("tp", "fp", "fn", "tn"):
            print(f"  {k:12s}: {v:.4f}")
    return (avg_rl, all_bl_metrics, all_rl_metrics,
            sample_imgs, sample_gts, sample_bl_preds, sample_rl_preds)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cfg_path = os.path.join(ROOT, "configs", "config.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    cfg = load_config(cfg_path)
    set_seed(cfg.get("seed", 42))
    device = get_device(cfg.get("device", "auto"))
    ensure_dirs(cfg)

    figs_dir    = cfg["output"]["figures_dir"]
    metrics_dir = cfg["output"]["metrics_dir"]
    preds_dir   = cfg["output"]["predictions_dir"]

    # ── 1) Dataset ───────────────────────────────────────────────────────────
    print("\n[pipeline] Step 1: Loading dataset...")
    train_ds, val_ds, test_ds = get_datasets(cfg)
    train_loader, val_loader, test_loader = make_dataloaders(
        cfg, train_ds, val_ds, test_ds
    )

    print("\n[pipeline] Step 1b: Saving dataset EDA figures...")
    eda_dir = os.path.join(figs_dir, "eda")
    os.makedirs(eda_dir, exist_ok=True)
    # EDA on train split (largest, most representative)
    plot_dataset_eda(train_ds, eda_dir, n_samples=16, tag="train")
    # Brief EDA on test split too
    plot_dataset_eda(test_ds, eda_dir, n_samples=12, tag="test")

    # ── 2) Build & train U-Net ───────────────────────────────────────────────
    print("\n[pipeline] Step 2: Building U-Net...")
    model = build_unet(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    criterion = build_loss(cfg)
    optimizer, scheduler = build_optimizer_scheduler(model, cfg)

    print("\n[pipeline] Step 3: Training baseline...")
    trainer = BaselineTrainer(model, criterion, optimizer, scheduler, device, cfg)
    history = trainer.train(train_loader, val_loader,
                            epochs=cfg["training"]["epochs"])
    trainer.load_best()

    print("\n[pipeline] Step 3b: Saving training curves...")
    plot_training_curves(history, figs_dir)

    # Save training history CSV
    pd.DataFrame(history).to_csv(
        os.path.join(metrics_dir, "training_history.csv"), index=False)

    # ── 3) Evaluate U-Net ────────────────────────────────────────────────────
    print("\n[pipeline] Step 4: Evaluating baseline on test set...")
    (bl_metrics, bl_per_sample,
     sample_imgs, sample_gts, sample_preds) = evaluate_baseline(
        model, test_loader, device, cfg)

    print("\n[pipeline] Step 4b: Saving U-Net evaluation figures...")
    unet_preds_dir = os.path.join(preds_dir, "unet")
    os.makedirs(unet_preds_dir, exist_ok=True)
    plot_unet_test_results(
        images           = sample_imgs,
        gt_masks         = sample_gts,
        predictions      = sample_preds,
        per_sample_metrics = bl_per_sample,
        save_dir         = unet_preds_dir,
        n_show           = cfg["evaluation"].get("num_qualitative", 12),
    )

    # ── 4) Train RL ───────────────────────────────────────────────────────────
    print("\n[pipeline] Step 5: Training RL agent...")
    env   = MaskRefinementEnv(cfg)
    agent = DQNAgent(cfg, device)
    rl_trainer = RLTrainer(
        model, agent, env, train_loader, val_loader, device, cfg)
    rl_history = rl_trainer.train(
        eval_every=max(1, cfg["rl"]["episodes"] // 15))

    agent.load("best_rl_agent.pth")

    print("\n[pipeline] Step 5b: Saving RL training curves...")
    plot_rl_curves(rl_history, figs_dir)

    # Save RL history CSV
    n_rl = len(rl_history["rewards"])
    pd.DataFrame({
        "episode":          range(1, n_rl + 1),
        "reward":           rl_history["rewards"],
        "epsilon":          rl_history["epsilon"],
        "dice_improvement": rl_history["dice_improvement"],
    }).to_csv(os.path.join(metrics_dir, "rl_history.csv"), index=False)

    # ── 5) Evaluate RL ────────────────────────────────────────────────────────
    print("\n[pipeline] Step 6: Evaluating RL on test set...")
    (rl_metrics, rl_bl_per_sample, rl_per_sample,
     rl_imgs, rl_gts, rl_bl_preds, rl_preds) = evaluate_rl(
        model, agent, env, test_loader, device, cfg)

    print("\n[pipeline] Step 6b: Saving RL evaluation figures...")
    rl_preds_dir = os.path.join(preds_dir, "rl")
    os.makedirs(rl_preds_dir, exist_ok=True)
    plot_rl_test_results(
        images               = rl_imgs,
        gt_masks             = rl_gts,
        baseline_preds       = rl_bl_preds,
        rl_preds             = rl_preds,
        bl_metrics_per_sample= rl_bl_per_sample,
        rl_metrics_per_sample= rl_per_sample,
        save_dir             = rl_preds_dir,
        n_show               = cfg["evaluation"].get("num_qualitative", 12),
    )

    # ── 6) Final comparison & reports ─────────────────────────────────────────
    print("\n[pipeline] Step 7: Final comparison and reports...")
    plot_comparison(bl_metrics, rl_metrics, figs_dir)

    # Metrics CSV (exclude raw counts from report)
    _scalar_keys = [k for k in bl_metrics if k not in ("tp","fp","fn","tn")]
    rows = []
    for k in _scalar_keys:
        rows.append({
            "metric":   k,
            "baseline": bl_metrics[k],
            "rl":       rl_metrics.get(k, float("nan")),
            "delta":    rl_metrics.get(k, bl_metrics[k]) - bl_metrics[k],
        })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(metrics_dir, "test_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[pipeline] Metrics saved → {csv_path}")
    print(df.to_string(index=False))

    _write_report(bl_metrics, rl_metrics, cfg)

    print("\n" + "="*60)
    print(" Pipeline complete!")
    print("="*60)
    print(f"  EDA figures    → {figs_dir}/eda/")
    print(f"  Training curves→ {figs_dir}/")
    print(f"  U-Net results  → {preds_dir}/unet/")
    print(f"  RL results     → {preds_dir}/rl/")
    print(f"  Comparison     → {figs_dir}/baseline_vs_rl.png")
    print(f"  Metrics CSV    → {metrics_dir}/")
    print(f"  Report         → {cfg['output']['reports_dir']}/report.md")


# ─────────────────────────────────────────────────────────────────────────────
# Report writer
# ─────────────────────────────────────────────────────────────────────────────

def _write_report(bl: dict, rl: dict, cfg: dict) -> None:
    skip = {"tp", "fp", "fn", "tn"}
    lines = [
        "# Brain Tumor Segmentation -- Results Report\n\n",
        f"Dataset source: `{cfg['dataset']['source']}`  \n",
        f"Image size: `{cfg['dataset']['image_size']}`  \n",
        f"Epochs: `{cfg['training']['epochs']}`  \n",
        f"RL episodes: `{cfg['rl']['episodes']}`  \n\n",
        "## Test Set Metrics\n\n",
        "| Metric | Baseline | RL Refined | Delta |\n",
        "|--------|----------|------------|-------|\n",
    ]
    for k in bl:
        if k in skip:
            continue
        bl_v  = bl[k]
        rl_v  = rl.get(k, float("nan"))
        delta = rl_v - bl_v
        lines.append(f"| {k} | {bl_v:.4f} | {rl_v:.4f} | {delta:+.4f} |\n")

    lines.append("\n## Key Takeaways\n\n")
    dice_delta = rl.get("dice", bl.get("dice", 0)) - bl.get("dice", 0)
    if dice_delta > 0.005:
        lines.append(f"- RL refinement improved Dice by **{dice_delta:+.4f}** [OK]\n")
    elif dice_delta < -0.005:
        lines.append(f"- RL refinement reduced Dice by {dice_delta:+.4f} "
                     "-- consider more episodes or hyperparameter tuning.\n")
    else:
        lines.append(f"- RL refinement had minimal effect on Dice ({dice_delta:+.4f}).\n")

    lines.append("\n## Output Files\n\n")
    lines.append("| Location | Contents |\n")
    lines.append("|----------|----------|\n")
    lines.append(f"| `{cfg['output']['figures_dir']}/eda/` | Dataset EDA: image grids, size/position/intensity distributions |\n")
    lines.append(f"| `{cfg['output']['figures_dir']}/training_curves.png` | U-Net loss, Dice, IoU curves |\n")
    lines.append(f"| `{cfg['output']['predictions_dir']}/unet/` | Qualitative grid, metric violin, PR scatter, confusion matrix |\n")
    lines.append(f"| `{cfg['output']['figures_dir']}/rl_curves.png` | RL reward, delta, epsilon, val improvement |\n")
    lines.append(f"| `{cfg['output']['predictions_dir']}/rl/` | RL qualitative grid, improvement scatter, delta violins |\n")
    lines.append(f"| `{cfg['output']['figures_dir']}/baseline_vs_rl.png` | Side-by-side bar + delta comparison |\n")

    report_path = os.path.join(cfg["output"]["reports_dir"], "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"[pipeline] Report saved → {report_path}")


if __name__ == "__main__":
    main()