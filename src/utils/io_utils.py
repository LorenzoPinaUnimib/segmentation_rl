"""
io_utils.py — Config loading, seed setting, device selection, dir creation.
"""
import os
import random
import numpy as np
import torch
import yaml
from pathlib import Path


def load_config(path: str) -> dict:
    """Load YAML config and return as nested dict."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def set_seed(seed: int) -> None:
    """Fix all RNG sources for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(cfg_device: str = "auto") -> torch.device:
    """Return the best available device."""
    if cfg_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg_device)
    print(f"[device] Using: {device}")
    return device


def ensure_dirs(cfg: dict) -> None:
    """Create all output directories from config."""
    out = cfg.get("output", {})
    dirs = [
        out.get("root", "outputs"),
        out.get("checkpoint_dir", "outputs/checkpoints"),
        out.get("log_dir", "outputs/logs"),
        out.get("metrics_dir", "outputs/metrics"),
        out.get("figures_dir", "outputs/figures"),
        out.get("predictions_dir", "outputs/predictions"),
        out.get("reports_dir", "outputs/reports"),
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
