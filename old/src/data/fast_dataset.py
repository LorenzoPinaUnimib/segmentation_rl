"""
fast_dataset.py — DataLoader factory ottimizzata per Windows + DirectML.

Problemi del DataLoader originale su Windows:
  - num_workers > 0 su Windows usa spawn (non fork) → ogni worker
    reimporta tutto il codice Python → overhead enorme (~2-5s di avvio)
  - pin_memory=True con DirectML non porta benefici (non è CUDA)
  - persistent_workers=True aiuta ma non basta

Soluzione: num_workers=0 + CachedDataset in RAM.
  - Epoch 1: carica da disco (lenta una volta sola)
  - Epoch 2+: tutto da RAM → ~10-50x più veloce sul data loading

Uso:
    from fast_dataset import make_fast_dataloaders
    train_loader, val_loader, test_loader = make_fast_dataloaders(
        cfg, train_ds, val_ds, test_ds, preload=True
    )
"""
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
from .cached_dataset import CachedDataset


def make_fast_dataloaders(
    cfg: dict,
    train_ds: Dataset,
    val_ds: Dataset,
    test_ds: Dataset,
    preload: bool = True,     # True = precarica tutto in RAM prima del training
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Crea DataLoader ottimizzati per Windows + DirectML.

    Args:
        preload: se True, precarica tutti i dataset in RAM prima di restituire
                 i loader. Aggiunge ~30-60s una tantum ma elimina I/O da disco
                 per tutte le epoche successive.
    """
    bs = cfg["training"]["batch_size"]

    # Wrappa con cache in-memory
    print("[fast_data] Wrapping datasets with in-memory cache...")
    cached_train = CachedDataset(train_ds, verbose=True)
    cached_val   = CachedDataset(val_ds,   verbose=False)
    cached_test  = CachedDataset(test_ds,  verbose=False)

    if preload:
        print("[fast_data] Preloading train set...")
        cached_train.preload_all()
        print("[fast_data] Preloading val set...")
        cached_val.preload_all()
        # test set: lazy (lo carichiamo solo durante eval)

    # num_workers=0: nessun subprocess spawn su Windows
    # pin_memory=False: DirectML non è CUDA, pin_memory non aiuta
    train_loader = DataLoader(
        cached_train,
        batch_size=bs,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )
    val_loader = DataLoader(
        cached_val,
        batch_size=bs,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    test_loader = DataLoader(
        cached_test,
        batch_size=bs,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    print(f"[fast_data] Train={len(train_ds)} Val={len(val_ds)} Test={len(test_ds)}")
    print(f"[fast_data] num_workers=0, pin_memory=False (ottimizzato per Windows/DirectML)")
    return train_loader, val_loader, test_loader
