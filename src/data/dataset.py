"""
dataset.py
  - COCOAnnotationDataset: loads images + renders polygon masks from COCO JSON
  - BrainTumorDataset: loads (image, mask-file) pairs from disk
  - SyntheticBrainDataset: generates medical-like synthetic data on-the-fly
  - make_dataloaders: factory that builds train/val/test DataLoaders
  - load_kaggle_dataset / load_local_dataset: top-level entry points
"""
import os
import json
import math
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

from .dataset_inspector import DatasetInspector
from .transforms import (
    normalize_image, get_train_transform, get_val_transform,
    FallbackTransform, HAS_ALBUMENTATIONS
)

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Polygon → mask rasterisation helper
# ─────────────────────────────────────────────────────────────────────────────

def polygons_to_mask(annotations: list, height: int, width: int) -> np.ndarray:
    """
    Render COCO polygon segmentations for one image into a single binary mask.
    Each annotation may have multiple polygons (segmentation is a list of lists).
    Returns float32 [H, W] in {0, 1}.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    for ann in annotations:
        segs = ann.get("segmentation", [])
        if isinstance(segs, dict):
            # RLE format — skip (rare in this dataset)
            continue
        for poly in segs:
            if len(poly) < 6:
                continue  # need at least 3 points
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
            pts = pts.astype(np.int32)
            cv2.fillPoly(mask, [pts], color=1)
    return mask.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# COCO annotation dataset
# ─────────────────────────────────────────────────────────────────────────────

class COCOAnnotationDataset(Dataset):
    """
    Loads images from disk and renders binary masks from COCO polygon
    segmentations on the fly. No separate mask files required.

    pairs: list of (image_path, ann_dict) where ann_dict has:
        - annotations: list of COCO annotation dicts for this image
        - height, width: original image dimensions
    """

    def __init__(
        self,
        pairs: List[Tuple[Path, dict]],
        cfg: dict,
        split: str = "train",
    ):
        self.pairs = pairs
        self.cfg = cfg
        self.split = split
        self.image_size = tuple(cfg["dataset"]["image_size"])
        self.in_channels = cfg["dataset"].get("in_channels", 3)
        self.norm_mode = cfg["preprocessing"]["normalization"]
        self.binarize = cfg["preprocessing"].get("binarize_mask", True)
        self.thresh = cfg["preprocessing"].get("mask_threshold", 0.5)
        self.augment = (split == "train")
        self._build_transform()

    def _build_transform(self):
        if HAS_ALBUMENTATIONS:
            if self.augment:
                self.transform = get_train_transform(self.cfg, self.image_size)
            else:
                self.transform = get_val_transform(self.image_size)
            self.use_alb = True
        else:
            self.transform = FallbackTransform(self.image_size, augment=self.augment)
            self.use_alb = False

    def __len__(self):
        return len(self.pairs)

    def _load_image(self, path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.in_channels == 1:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path, ann_dict = self.pairs[idx]

        image = self._load_image(img_path)
        h = ann_dict["height"]
        w = ann_dict["width"]
        mask = polygons_to_mask(ann_dict["annotations"], h, w)

        # Normalize image to [0, 1]
        image = normalize_image(image, mode=self.norm_mode)

        if self.binarize:
            mask = (mask > self.thresh).astype(np.float32)

        if self.use_alb:
            if image.ndim == 2:
                image = image[..., np.newaxis]
                image = np.repeat(image, 3, axis=-1)
            aug = self.transform(image=image, mask=mask)
            image_t = aug["image"].float()
            mask_t  = aug["mask"].float()
        else:
            out = self.transform(image, mask)
            image_t, mask_t = out["image"], out["mask"]

        if image_t.ndim == 2:
            image_t = image_t.unsqueeze(0)
        if mask_t.ndim == 2:
            mask_t = mask_t.unsqueeze(0)

        return {"image": image_t, "mask": mask_t, "idx": idx}


# ─────────────────────────────────────────────────────────────────────────────
# File-based dataset (image file + mask file pairs)
# ─────────────────────────────────────────────────────────────────────────────

class BrainTumorDataset(Dataset):
    """
    Loads (image, mask) pairs from a list of (Path, Path) tuples.
    Applies normalization and optional augmentation.
    """

    def __init__(
        self,
        pairs: List[Tuple[Path, Path]],
        cfg: dict,
        split: str = "train",
    ):
        self.pairs = pairs
        self.cfg = cfg
        self.split = split
        self.image_size = tuple(cfg["dataset"]["image_size"])
        self.in_channels = cfg["dataset"].get("in_channels", 3)
        self.norm_mode = cfg["preprocessing"]["normalization"]
        self.binarize = cfg["preprocessing"].get("binarize_mask", True)
        self.thresh = cfg["preprocessing"].get("mask_threshold", 0.5)
        self.augment = (split == "train")
        self._build_transform()

    def _build_transform(self):
        if HAS_ALBUMENTATIONS:
            if self.augment:
                self.transform = get_train_transform(self.cfg, self.image_size)
            else:
                self.transform = get_val_transform(self.image_size)
            self.use_alb = True
        else:
            self.transform = FallbackTransform(self.image_size, augment=self.augment)
            self.use_alb = False

    def __len__(self):
        return len(self.pairs)

    def _load_image(self, path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.in_channels == 1:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img

    def _load_mask(self, path: Path) -> np.ndarray:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask: {path}")
        return mask

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path, mask_path = self.pairs[idx]
        image = self._load_image(img_path)
        mask  = self._load_mask(mask_path)

        image = normalize_image(image, mode=self.norm_mode)

        if mask.max() > 1:
            mask = mask.astype(np.float32) / 255.0
        else:
            mask = mask.astype(np.float32)

        if self.binarize:
            mask = (mask > self.thresh).astype(np.float32)

        if self.use_alb:
            if image.ndim == 2:
                image = image[..., np.newaxis]
                image = np.repeat(image, 3, axis=-1)
            aug = self.transform(image=image, mask=mask)
            image_t = aug["image"].float()
            mask_t  = aug["mask"].float()
        else:
            out = self.transform(image, mask)
            image_t, mask_t = out["image"], out["mask"]

        if image_t.ndim == 2:
            image_t = image_t.unsqueeze(0)
        if mask_t.ndim == 2:
            mask_t = mask_t.unsqueeze(0)

        return {"image": image_t, "mask": mask_t, "idx": idx}


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic dataset (fallback / demo)
# ─────────────────────────────────────────────────────────────────────────────

class SyntheticBrainDataset(Dataset):
    """
    Generates synthetic MRI-like brain images with tumor blobs as masks.
    """

    def __init__(self, cfg: dict, n_samples: int = 300, split: str = "train"):
        self.n = n_samples
        self.split = split
        self.image_size = tuple(cfg["dataset"]["image_size"])
        self.in_channels = cfg["dataset"].get("in_channels", 3)
        self.norm_mode = cfg["preprocessing"]["normalization"]
        self.augment = (split == "train")
        self.cfg = cfg
        self._build_transform()

    def _build_transform(self):
        if HAS_ALBUMENTATIONS:
            if self.augment:
                self.transform = get_train_transform(self.cfg, self.image_size)
            else:
                self.transform = get_val_transform(self.image_size)
            self.use_alb = True
        else:
            self.transform = FallbackTransform(self.image_size, augment=self.augment)
            self.use_alb = False

    def __len__(self):
        return self.n

    def _generate_sample(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.RandomState(idx)
        H, W = self.image_size
        xx, yy = np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H))
        bg = 0.4 + 0.2 * np.sin(3 * xx) * np.cos(3 * yy)
        noise = rng.randn(H, W) * 0.05
        brain = np.clip(bg + noise, 0, 1).astype(np.float32)
        cx, cy = W // 2, H // 2
        dist = np.sqrt((xx - 0.5) ** 2 + (yy - 0.5) ** 2)
        skull = (dist > 0.42) & (dist < 0.50)
        brain[skull] = 0.15
        tx = rng.uniform(0.25, 0.75)
        ty = rng.uniform(0.25, 0.75)
        rx = rng.uniform(0.05, 0.15)
        ry = rng.uniform(0.04, 0.12)
        angle = rng.uniform(0, np.pi)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        dx = xx - tx
        dy = yy - ty
        rot_x =  cos_a * dx + sin_a * dy
        rot_y = -sin_a * dx + cos_a * dy
        tumor_mask = ((rot_x / rx) ** 2 + (rot_y / ry) ** 2) < 1.0
        n_sats = rng.randint(1, 4)
        for _ in range(n_sats):
            sx = tx + rng.uniform(-rx, rx) * 2
            sy = ty + rng.uniform(-ry, ry) * 2
            sr = rng.uniform(0.01, 0.04)
            sat = ((xx - sx) ** 2 + (yy - sy) ** 2) < sr ** 2
            tumor_mask = tumor_mask | sat
        brain[tumor_mask] = np.clip(brain[tumor_mask] + 0.3, 0, 1)
        image = np.stack([brain, brain, brain], axis=-1)
        mask  = tumor_mask.astype(np.float32)
        return image, mask

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image, mask = self._generate_sample(idx)
        if self.use_alb:
            aug = self.transform(image=image, mask=mask)
            image_t = aug["image"].float()
            mask_t  = aug["mask"].float()
        else:
            out = self.transform(image, mask)
            image_t, mask_t = out["image"], out["mask"]
        if image_t.ndim == 2:
            image_t = image_t.unsqueeze(0)
        if mask_t.ndim == 2:
            mask_t = mask_t.unsqueeze(0)
        return {"image": image_t, "mask": mask_t, "idx": idx}


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def make_dataloaders(
    cfg: dict,
    train_ds: Dataset,
    val_ds: Dataset,
    test_ds: Dataset,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    bs        = cfg["training"]["batch_size"]
    nw        = cfg["training"].get("num_workers", 2)
    persistent = (nw > 0)

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=nw, pin_memory=True, drop_last=True,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=nw, pin_memory=True,
        persistent_workers=persistent,
    )
    test_loader = DataLoader(
        test_ds, batch_size=bs, shuffle=False,
        num_workers=nw, pin_memory=True,
        persistent_workers=persistent,
    )
    print(f"[data] Train={len(train_ds)} Val={len(val_ds)} Test={len(test_ds)}")
    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading entry points
# ─────────────────────────────────────────────────────────────────────────────

def _split_pairs(pairs, train_r, val_r, seed):
    """Split list of pairs into train/val/test."""
    rng = random.Random(seed)
    rng.shuffle(pairs)
    n = len(pairs)
    n_train = int(n * train_r)
    n_val   = int(n * val_r)
    return pairs[:n_train], pairs[n_train:n_train+n_val], pairs[n_train+n_val:]


def _build_datasets_from_pairs(pairs, inspector, cfg, seed):
    """
    Given a flat list of pairs (COCO or file-pair), split and wrap in Dataset.
    Handles both inspector.coco_mode=True and False.
    """
    train_r = cfg["dataset"]["train_ratio"]
    val_r   = cfg["dataset"]["val_ratio"]
    train_p, val_p, test_p = _split_pairs(pairs, train_r, val_r, seed)

    if inspector.coco_mode:
        return (
            COCOAnnotationDataset(train_p, cfg, split="train"),
            COCOAnnotationDataset(val_p,   cfg, split="val"),
            COCOAnnotationDataset(test_p,  cfg, split="test"),
        )
    else:
        return (
            BrainTumorDataset(train_p, cfg, split="train"),
            BrainTumorDataset(val_p,   cfg, split="val"),
            BrainTumorDataset(test_p,  cfg, split="test"),
        )


def load_local_dataset(cfg: dict) -> Tuple[Dataset, Dataset, Dataset]:
    """Load dataset from local filesystem (auto-detects COCO JSON or file pairs)."""
    root = cfg["dataset"]["local_path"]
    assert root and os.path.exists(root), f"local_path not found: {root}"

    inspector = DatasetInspector(root)
    pairs = inspector.inspect()

    if cfg["dataset"].get("cache_pairs") and pairs:
        cache = os.path.join(cfg["output"]["root"], "pairs_cache.csv")
        inspector.save_pairs_csv(pairs, cache)

    if not pairs:
        print("[data] No pairs found — falling back to synthetic dataset.")
        return _make_synthetic(cfg)

    seed = cfg.get("seed", 42)
    return _build_datasets_from_pairs(pairs, inspector, cfg, seed)


def load_kaggle_dataset(cfg: dict) -> Tuple[Dataset, Dataset, Dataset]:
    """Download dataset via kagglehub and load."""
    kaggle_id = cfg["dataset"]["kaggle_id"]
    print(f"[data] Downloading Kaggle dataset: {kaggle_id}")
    try:
        import kagglehub
        path = kagglehub.dataset_download(kaggle_id)
        print(f"[data] Downloaded to: {path}")
        cfg["dataset"]["local_path"] = path
        return load_local_dataset(cfg)
    except Exception as e:
        print(f"[data] Kaggle download failed ({e}). Falling back to synthetic.")
        return _make_synthetic(cfg)


def _make_synthetic(cfg: dict) -> Tuple[Dataset, Dataset, Dataset]:
    print("[data] Using SYNTHETIC brain tumor dataset.")
    return (
        SyntheticBrainDataset(cfg, n_samples=400, split="train"),
        SyntheticBrainDataset(cfg, n_samples=80,  split="val"),
        SyntheticBrainDataset(cfg, n_samples=80,  split="test"),
    )


def get_datasets(cfg: dict) -> Tuple[Dataset, Dataset, Dataset]:
    """Top-level dispatcher."""
    source = cfg["dataset"]["source"]
    if source == "kaggle":
        return load_kaggle_dataset(cfg)
    elif source == "local":
        return load_local_dataset(cfg)
    else:
        return _make_synthetic(cfg)