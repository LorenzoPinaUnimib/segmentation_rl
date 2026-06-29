"""
cached_dataset.py — Dataset con cache in-memory dei tensori preprocessati.

La prima epoch carica e preprocessa tutto, le successive leggono dalla RAM.
Con 2145 immagini a 256x256x3 float32 = ~1.3 GB RAM (accettabile).
A 128x128 = ~330 MB. Abbatte I/O e CPU di preprocessing del 90%+ dalla ep2.
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Optional
import threading


class CachedDataset(Dataset):
    """
    Wrapper che mette in cache tutti i tensori in RAM dopo il primo accesso.
    Thread-safe. Stampa progresso ogni 10%.
    """
    def __init__(self, base_dataset: Dataset, verbose: bool = True):
        self.ds = base_dataset
        self.verbose = verbose
        self._cache: Dict[int, Dict[str, torch.Tensor]] = {}
        self._lock = threading.Lock()
        self._total = len(base_dataset)

    def __len__(self):
        return self._total

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx not in self._cache:
            item = self.ds[idx]
            with self._lock:
                self._cache[idx] = item
            if self.verbose and len(self._cache) % max(1, self._total // 10) == 0:
                pct = 100 * len(self._cache) / self._total
                print(f"  [cache] {len(self._cache)}/{self._total} ({pct:.0f}%)")
        return self._cache[idx]

    def preload_all(self):
        """Precarica tutto in un unico passaggio (opzionale, da chiamare esplicitamente)."""
        print(f"[cache] Preloading {self._total} samples into RAM...")
        for i in range(self._total):
            _ = self[i]
        print(f"[cache] Done. RAM cache ready.")
