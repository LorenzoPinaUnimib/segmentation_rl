# Brain Tumor Segmentation + RL Refinement

End-to-end pipeline for semantic segmentation of brain tumors from MRI images,
with a DQN-based reinforcement learning agent that refines the baseline mask output.

---

## Architecture Overview

```
Dataset (Kaggle / local / synthetic)
        ↓
  Data Pipeline  →  BrainTumorDataset / SyntheticBrainDataset
        ↓
  U-Net Baseline  →  encoder-decoder with skip connections
        ↓
  RL Refinement Agent (DQN)
    State:   [image stats | probability map | mask shape | uncertainty]
    Actions: stop | accept | expand | shrink | remove_small | smooth | thresh±
    Reward:  ΔDice + ΔIoU − fragmentation − step_cost + terminal_dice
        ↓
  Evaluation & Reports  →  CSV + PNG figures + qualitative grids
```

---

## Setup

```bash
pip install -r requirements.txt
```

**Optional Kaggle setup** (only needed if `source: kaggle` in config):
```bash
pip install kaggle
# Place ~/.kaggle/kaggle.json with your credentials
```

---

## Running

### Full pipeline (recommended)
```bash
python scripts/run_full_pipeline.py
```

### Separate stages
```bash
# Train baseline only
python scripts/train_baseline.py --config configs/config.yaml

# Train RL only (requires baseline checkpoint in outputs/checkpoints/)
python scripts/train_rl.py --config configs/config.yaml
```

---

## Configuration (`configs/config.yaml`)

| Key | Default | Description |
|-----|---------|-------------|
| `dataset.source` | `synthetic` | `synthetic` / `kaggle` / `local` |
| `dataset.local_path` | `null` | Path to local dataset root |
| `dataset.image_size` | `[256,256]` | Resize all images to this |
| `training.epochs` | `30` | Baseline training epochs |
| `training.batch_size` | `8` | Batch size |
| `rl.episodes` | `300` | RL training episodes |
| `rl.max_steps_per_episode` | `8` | Max mask refinement steps |
| `rl.epsilon_decay` | `200` | Epsilon decay (episodes) |

---

## Dataset Support

The pipeline auto-detects dataset structure via 3 heuristics (in order):
1. `root/train/images/*.png` + `root/train/masks/*.png`
2. `root/images/*.png` + `root/masks/*.png` (sibling dirs)
3. Name-matching: `tumor_001.png` ↔ `tumor_mask_001.png`

Falls back to synthetic data if no pairs are found.

---

## Outputs

```
outputs/
├── checkpoints/
│   ├── best_model.pth          # best baseline U-Net
│   └── best_rl_agent.pth       # best RL policy
├── figures/
│   ├── training_curves.png
│   ├── rl_curves.png
│   └── baseline_vs_rl.png
├── metrics/
│   ├── test_metrics.csv
│   ├── training_history.csv
│   └── rl_history.csv
├── predictions/
│   └── qualitative_results.png
└── reports/
    └── report.md
```

---

## Metrics

| Metric | Description |
|--------|-------------|
| Dice | Overlap measure (primary) |
| IoU | Jaccard index |
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| Specificity | TN / (TN + FP) |
| Pixel Accuracy | Fraction correct pixels |
| Hausdorff Distance | Boundary error (optional) |

---

## RL Action Space

| Action | Effect |
|--------|--------|
| 0 | Stop episode |
| 1 | Accept (no-op) |
| 2 | Expand mask (dilation) |
| 3 | Shrink mask (erosion) |
| 4 | Remove small components |
| 5 | Smooth mask (Gaussian) |
| 6 | Threshold up (fewer positives) |
| 7 | Threshold down (more positives) |
