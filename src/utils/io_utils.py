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
    """
    Restituisce il device migliore disponibile.

    Ordine di preferenza in modalità "auto":
      1) CUDA (NVIDIA), se disponibile
      2) DirectML (torch-directml — GPU AMD/Intel/NVIDIA su Windows senza
         bisogno di CUDA), se il pacchetto è installato
      3) CPU come fallback

    cfg_device può anche essere impostato esplicitamente a "dml" /
    "directml" per forzare DirectML.
    """
    cfg_device = (cfg_device or "auto").lower()

    if cfg_device in ("dml", "directml"):
        device = _get_directml_device(required=True)
    elif cfg_device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = _get_directml_device(required=False)
            if device is None:
                device = torch.device("cpu")
    else:
        device = torch.device(cfg_device)

    print(f"[device] Using: {device}")
    return device


def _get_directml_device(required: bool = False):
    """Prova a ottenere un device DirectML (pacchetto opzionale torch-directml)."""
    try:
        import torch_directml
        device = torch_directml.device()
        print("[device] DirectML disponibile e selezionato.")
        return device
    except ImportError:
        if required:
            raise RuntimeError(
                "DirectML richiesto (device: dml) ma 'torch-directml' non è "
                "installato. Esegui: pip install torch-directml"
            )
        return None


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
