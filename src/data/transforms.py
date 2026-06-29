"""
transforms.py — Albumentations-based augmentation and preprocessing.
"""
import numpy as np
import cv2

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False
    print("[transforms] albumentations not found; using fallback transforms.")

import torch
from typing import Tuple, Dict


# ─────────────────────────────────────────────────────────────────────────────
# Normalization helpers
# ─────────────────────────────────────────────────────────────────────────────

def normalize_image(img: np.ndarray, mode: str = "per_image") -> np.ndarray:
    """
    Normalize float32 image [H,W,C] or [H,W].
    mode: per_image | min_max | z_score
    """
    img = img.astype(np.float32)
    if mode == "per_image":
        mn, mx = img.min(), img.max()
        if mx - mn > 1e-6:
            img = (img - mn) / (mx - mn)
        else:
            img = img * 0.0
    elif mode == "min_max":
        img = img / 255.0
    elif mode == "z_score":
        mean, std = img.mean(), img.std()
        if std > 1e-6:
            img = (img - mean) / std
        else:
            img = img - mean
    return img


# ─────────────────────────────────────────────────────────────────────────────
# Albumentations pipelines
# ─────────────────────────────────────────────────────────────────────────────

def get_train_transform(cfg: dict, image_size: Tuple[int, int]) -> "A.Compose":
    aug = cfg.get("augmentation", {})
    size_h, size_w = image_size

    if not HAS_ALBUMENTATIONS:
        raise RuntimeError("albumentations required for augmentation.")

    transforms = [
        A.Resize(size_h, size_w),
    ]

    if aug.get("enabled", True):
        if aug.get("horizontal_flip", 0) > 0:
            transforms.append(A.HorizontalFlip(p=aug["horizontal_flip"]))
        if aug.get("vertical_flip", 0) > 0:
            transforms.append(A.VerticalFlip(p=aug["vertical_flip"]))
        rot = aug.get("rotate_limit", 15)
        if rot > 0:
            transforms.append(A.Rotate(limit=rot, p=0.5,
                                        border_mode=cv2.BORDER_REFLECT_101))
        bl = aug.get("brightness_limit", 0.15)
        cl = aug.get("contrast_limit", 0.15)
        if bl > 0 or cl > 0:
            transforms.append(
                A.RandomBrightnessContrast(
                    brightness_limit=bl, contrast_limit=cl, p=0.5
                )
            )
        nl = aug.get("noise_var_limit", [5.0, 20.0])
        transforms.append(A.GaussNoise(p=0.3))

    transforms.append(ToTensorV2())
    return A.Compose(transforms)


def get_val_transform(image_size: Tuple[int, int]) -> "A.Compose":
    size_h, size_w = image_size
    if not HAS_ALBUMENTATIONS:
        raise RuntimeError("albumentations required.")
    return A.Compose([
        A.Resize(size_h, size_w),
        ToTensorV2(),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Fallback (no albumentations)
# ─────────────────────────────────────────────────────────────────────────────

class FallbackTransform:
    """Minimal numpy-only transform: resize + to tensor."""
    def __init__(self, image_size: Tuple[int, int], augment: bool = False):
        self.h, self.w = image_size
        self.augment = augment

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Dict:
        # resize
        image = cv2.resize(image, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        mask  = cv2.resize(mask,  (self.w, self.h), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            if np.random.rand() > 0.5:
                image = np.fliplr(image).copy()
                mask  = np.fliplr(mask).copy()
            if np.random.rand() > 0.7:
                image = np.flipud(image).copy()
                mask  = np.flipud(mask).copy()

        # [H,W,C] → [C,H,W] tensor
        if image.ndim == 2:
            image = image[..., np.newaxis]
        image_t = torch.from_numpy(image.transpose(2, 0, 1)).float()

        if mask.ndim == 3:
            mask = mask[..., 0]
        mask_t = torch.from_numpy(mask).float()
        return {"image": image_t, "mask": mask_t}
