"""
environment.py — RL Environment for mask refinement.

State:  [img_features | mask_stats | prob_stats | uncertainty]
Actions: 0=stop, 1=accept, 2=expand, 3=shrink, 4=remove_small,
         5=smooth, 6=threshold_up, 7=threshold_down
Reward: ΔDICE + ΔIoU − fragmentation_penalty − step_penalty
"""
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from scipy import ndimage


# ─────────────────────────────────────────────────────────────────────────────
# Mask operations (deterministic, numpy-based)
# ─────────────────────────────────────────────────────────────────────────────

def _to_np(t) -> np.ndarray:
    if isinstance(t, torch.Tensor):
        return t.cpu().numpy().squeeze()
    return np.asarray(t).squeeze()


def op_expand(mask: np.ndarray, iters: int = 3) -> np.ndarray:
    """Binary dilation."""
    struct = ndimage.generate_binary_structure(2, 2)
    return ndimage.binary_dilation(mask > 0.5, structure=struct,
                                   iterations=iters).astype(np.float32)


def op_shrink(mask: np.ndarray, iters: int = 3) -> np.ndarray:
    """Binary erosion."""
    struct = ndimage.generate_binary_structure(2, 2)
    eroded = ndimage.binary_erosion(mask > 0.5, structure=struct,
                                    iterations=iters)
    return eroded.astype(np.float32)


def op_remove_small(mask: np.ndarray, min_size: int = 50) -> np.ndarray:
    """Remove connected components smaller than min_size pixels."""
    labeled, n = ndimage.label(mask > 0.5)
    result = np.zeros_like(mask)
    for i in range(1, n + 1):
        comp = labeled == i
        if comp.sum() >= min_size:
            result[comp] = 1.0
    return result


def op_smooth(mask: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Gaussian smooth then re-threshold."""
    smoothed = ndimage.gaussian_filter(mask.astype(np.float32), sigma=sigma)
    return (smoothed > 0.5).astype(np.float32)


def op_threshold_up(prob_map: np.ndarray, delta: float = 0.05,
                     current_thresh: float = 0.5) -> Tuple[np.ndarray, float]:
    new_t = min(current_thresh + delta, 0.95)
    return (prob_map > new_t).astype(np.float32), new_t


def op_threshold_down(prob_map: np.ndarray, delta: float = 0.05,
                       current_thresh: float = 0.5) -> Tuple[np.ndarray, float]:
    new_t = max(current_thresh - delta, 0.05)
    return (prob_map > new_t).astype(np.float32), new_t


# ─────────────────────────────────────────────────────────────────────────────
# State feature extractor
# ─────────────────────────────────────────────────────────────────────────────

def extract_state_features(
    image_np: np.ndarray,   # [H, W] or [H, W, C] in [0,1]
    mask_np: np.ndarray,    # [H, W] binary float
    prob_np: np.ndarray,    # [H, W] probability map in [0,1]
) -> np.ndarray:
    """
    Extract a compact state vector from image, current mask, and probability map.
    Returns a 1D float32 numpy array of fixed dimension.
    """
    # Flatten to 2D
    if image_np.ndim == 3:
        img_gray = image_np.mean(axis=-1)
    else:
        img_gray = image_np

    H, W = img_gray.shape
    total_pixels = H * W

    # --- Image features (6)
    img_feats = np.array([
        img_gray.mean(),
        img_gray.std(),
        img_gray.max(),
        img_gray.min(),
        np.percentile(img_gray, 25),
        np.percentile(img_gray, 75),
    ], dtype=np.float32)

    # --- Probability map features (6)
    prob_feats = np.array([
        prob_np.mean(),
        prob_np.std(),
        (prob_np > 0.5).mean(),   # fraction positive
        (prob_np > 0.7).mean(),   # high confidence positive
        (prob_np < 0.3).mean(),   # high confidence negative
        float(np.sum((prob_np > 0.3) & (prob_np < 0.7))) / total_pixels,  # uncertainty
    ], dtype=np.float32)

    # --- Entropy map (1 scalar)
    eps = 1e-8
    entropy = -prob_np * np.log(prob_np + eps) - (1 - prob_np) * np.log(1 - prob_np + eps)
    entropy_feats = np.array([entropy.mean()], dtype=np.float32)

    # --- Current mask shape features (8)
    bin_mask = (mask_np > 0.5).astype(np.float32)
    labeled, n_comp = ndimage.label(bin_mask)
    mask_area = bin_mask.mean()

    # Boundary length proxy: perimeter via difference
    eroded = ndimage.binary_erosion(bin_mask > 0).astype(np.float32)
    boundary = bin_mask - eroded
    boundary_len = boundary.sum() / (total_pixels + 1e-8)

    # Compactness: area / perimeter^2
    perim = max(boundary.sum(), 1)
    compactness = (4 * np.pi * bin_mask.sum()) / (perim ** 2 + 1e-8)
    compactness = np.clip(compactness, 0, 1)

    # Solidity: area / convex hull area
    try:
        from skimage.measure import regionprops, label as sk_label
        lbl = sk_label(bin_mask > 0)
        props_list = regionprops(lbl)
        if props_list:
            p = props_list[0]
            solidity = p.solidity
            eccentricity = p.eccentricity
        else:
            solidity = eccentricity = 0.0
    except Exception:
        solidity = eccentricity = 0.0

    mask_feats = np.array([
        mask_area,
        float(n_comp) / 10.0,   # normalised component count
        boundary_len,
        compactness,
        float(solidity),
        float(eccentricity),
        # Mean prob inside mask
        float(prob_np[bin_mask > 0].mean()) if bin_mask.sum() > 0 else 0.0,
        # Mean prob outside mask
        float(prob_np[bin_mask < 1].mean()) if (1 - bin_mask).sum() > 0 else 0.0,
    ], dtype=np.float32)

    state = np.concatenate([img_feats, prob_feats, entropy_feats, mask_feats])
    return state.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# RL Environment
# ─────────────────────────────────────────────────────────────────────────────

class MaskRefinementEnv:
    """
    Single-image RL environment.
    Each episode: agent refines the baseline mask for one image.

    Action space (discrete, 8 actions):
      0 = stop
      1 = accept (no-op, collect terminal reward)
      2 = expand
      3 = shrink
      4 = remove_small_components
      5 = smooth
      6 = threshold_up
      7 = threshold_down

    Reward = w_dice * ΔDICE + w_iou * ΔIoU
             − w_frag * fragmentation_penalty
             − step_penalty per step
             + w_terminal * final_dice at episode end
    """

    NUM_ACTIONS = 8
    STATE_DIM   = 21   # 6 img + 6 prob + 1 entropy + 8 mask

    def __init__(self, cfg: dict):
        rl = cfg.get("rl", {})
        self.max_steps       = rl.get("max_steps_per_episode", 8)
        self.w_dice          = rl.get("reward_dice_weight", 1.0)
        self.w_iou           = rl.get("reward_iou_weight", 0.5)
        self.w_frag          = rl.get("reward_fragment_penalty", 0.3)
        self.step_pen        = rl.get("reward_step_penalty", 0.02)
        self.w_terminal      = rl.get("reward_terminal_weight", 2.0)
        self.thresh_delta    = 0.05

        # Will be set by reset()
        self.image_np   : Optional[np.ndarray] = None
        self.prob_np    : Optional[np.ndarray] = None
        self.gt_np      : Optional[np.ndarray] = None
        self.mask_np    : Optional[np.ndarray] = None
        self.threshold  : float = 0.5
        self.step_count : int   = 0
        self._prev_dice : float = 0.0
        self._prev_iou  : float = 0.0

    def reset(
        self,
        image: np.ndarray,
        prob_map: np.ndarray,
        gt_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Reset environment for a new image.
        image:    [H,W] or [H,W,C] float32 in [0,1]
        prob_map: [H,W] float32 sigmoid output from baseline model
        gt_mask:  [H,W] binary float32 (used for reward during training)
        Returns initial state vector.
        """
        self.image_np  = image
        self.prob_np   = prob_map.astype(np.float32)
        self.gt_np     = gt_mask.astype(np.float32)
        self.threshold = 0.5
        self.mask_np   = (prob_map > 0.5).astype(np.float32)
        self.step_count = 0

        self._prev_dice = self._dice(self.mask_np)
        self._prev_iou  = self._iou(self.mask_np)

        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Apply action, return (next_state, reward, done, info)."""
        done = False
        info: Dict = {"action": action}

        # --- Apply action
        if action == 0:   # stop
            done = True
        elif action == 1: # accept (no-op)
            pass
        elif action == 2: # expand
            self.mask_np = op_expand(self.mask_np)
        elif action == 3: # shrink
            self.mask_np = op_shrink(self.mask_np)
        elif action == 4: # remove small
            self.mask_np = op_remove_small(self.mask_np)
        elif action == 5: # smooth
            self.mask_np = op_smooth(self.mask_np)
        elif action == 6: # threshold up
            self.mask_np, self.threshold = op_threshold_up(
                self.prob_np, self.thresh_delta, self.threshold)
        elif action == 7: # threshold down
            self.mask_np, self.threshold = op_threshold_down(
                self.prob_np, self.thresh_delta, self.threshold)

        # --- Compute reward
        new_dice = self._dice(self.mask_np)
        new_iou  = self._iou(self.mask_np)

        delta_dice = new_dice - self._prev_dice
        delta_iou  = new_iou  - self._prev_iou

        # Fragmentation penalty
        _, n_comp = ndimage.label(self.mask_np > 0.5)
        frag_penalty = max(0, n_comp - 3) * self.w_frag

        reward = (
            self.w_dice * delta_dice
            + self.w_iou * delta_iou
            - frag_penalty
            - self.step_pen
        )

        self._prev_dice = new_dice
        self._prev_iou  = new_iou

        self.step_count += 1

        # Terminal conditions
        if self.step_count >= self.max_steps:
            done = True

        # Terminal bonus: final dice
        if done:
            reward += self.w_terminal * new_dice

        info["dice"] = new_dice
        info["iou"]  = new_iou
        info["n_components"] = n_comp

        return self._get_state(), reward, done, info

    def get_refined_mask(self) -> np.ndarray:
        return self.mask_np

    # ── Internals ─────────────────────────────────────────────────────────────

    def _get_state(self) -> np.ndarray:
        return extract_state_features(self.image_np, self.mask_np, self.prob_np)

    def _dice(self, mask: np.ndarray) -> float:
        p = (mask > 0.5).flatten().astype(np.float32)
        g = (self.gt_np > 0.5).flatten().astype(np.float32)
        inter = (p * g).sum()
        return float(2 * inter / (p.sum() + g.sum() + 1e-6))

    def _iou(self, mask: np.ndarray) -> float:
        p = (mask > 0.5).flatten().astype(np.float32)
        g = (self.gt_np > 0.5).flatten().astype(np.float32)
        inter = (p * g).sum()
        union = p.sum() + g.sum() - inter
        return float(inter / (union + 1e-6))
