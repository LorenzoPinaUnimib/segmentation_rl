"""
environment_refine.py — RL Environment per ROI Refinement (Agent 2).

Dato un crop della ROI trovata dall'Agent 1, questo agente
raffina la maschera di segmentazione binaria sul crop.

Azioni (8):
  0 = stop
  1 = espandi maschera (dilation)
  2 = restringi maschera (erosion)
  3 = rimuovi componenti piccole
  4 = smoothing gaussiano
  5 = threshold su
  6 = threshold giù
  7 = accetta (no-op con reward terminale)

Stato: vettore feature della maschera corrente sul crop [21 dim]
       identico a MaskRefinementEnv per compatibilità.
"""
import numpy as np
from typing import Optional, Tuple, Dict
from scipy import ndimage


# ─────────────────────────────────────────────────────────────────────────────
# Operatori morfologici (riutilizzati da environment.py)
# ─────────────────────────────────────────────────────────────────────────────

def _expand(mask: np.ndarray, iters: int = 2) -> np.ndarray:
    struct = ndimage.generate_binary_structure(2, 2)
    return ndimage.binary_dilation(mask > 0.5, structure=struct,
                                   iterations=iters).astype(np.float32)


def _shrink(mask: np.ndarray, iters: int = 2) -> np.ndarray:
    struct = ndimage.generate_binary_structure(2, 2)
    eroded = ndimage.binary_erosion(mask > 0.5, structure=struct,
                                    iterations=iters)
    return eroded.astype(np.float32)


def _remove_small(mask: np.ndarray, min_size: int = 30) -> np.ndarray:
    labeled, n = ndimage.label(mask > 0.5)
    result = np.zeros_like(mask)
    for i in range(1, n + 1):
        comp = labeled == i
        if comp.sum() >= min_size:
            result[comp] = 1.0
    return result


def _smooth(mask: np.ndarray, sigma: float = 1.2) -> np.ndarray:
    smoothed = ndimage.gaussian_filter(mask.astype(np.float32), sigma=sigma)
    return (smoothed > 0.5).astype(np.float32)


def _threshold_up(prob: np.ndarray, t: float, delta: float = 0.05) -> Tuple[np.ndarray, float]:
    new_t = min(t + delta, 0.95)
    return (prob > new_t).astype(np.float32), new_t


def _threshold_down(prob: np.ndarray, t: float, delta: float = 0.05) -> Tuple[np.ndarray, float]:
    new_t = max(t - delta, 0.05)
    return (prob > new_t).astype(np.float32), new_t


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction (stessa struttura di environment.py per consistenza)
# ─────────────────────────────────────────────────────────────────────────────

def extract_refine_state(
    image_np: np.ndarray,
    mask_np: np.ndarray,
    prob_np: np.ndarray,
) -> np.ndarray:
    """
    Estrae vettore di stato 21-dim per il refinement agent.
    Uguale a MaskRefinementEnv.STATE_DIM per compatibilità con DQNAgent.
    """
    if image_np.ndim == 3:
        img_gray = image_np.mean(axis=-1)
    else:
        img_gray = image_np.astype(np.float32)

    H, W = img_gray.shape
    total = H * W

    # --- Image features (6)
    img_feats = np.array([
        img_gray.mean(), img_gray.std(), img_gray.max(),
        img_gray.min(), np.percentile(img_gray, 25), np.percentile(img_gray, 75),
    ], dtype=np.float32)

    # --- Prob map features (6)
    eps = 1e-8
    prob_feats = np.array([
        prob_np.mean(), prob_np.std(),
        (prob_np > 0.5).mean(),
        (prob_np > 0.7).mean(),
        (prob_np < 0.3).mean(),
        float(np.sum((prob_np > 0.3) & (prob_np < 0.7))) / total,
    ], dtype=np.float32)

    # --- Entropy (1)
    entropy = -prob_np * np.log(prob_np + eps) - (1 - prob_np) * np.log(1 - prob_np + eps)
    entropy_feats = np.array([entropy.mean()], dtype=np.float32)

    # --- Mask shape features (8)
    bin_mask = (mask_np > 0.5).astype(np.float32)
    labeled, n_comp = ndimage.label(bin_mask)
    eroded = ndimage.binary_erosion(bin_mask > 0).astype(np.float32)
    boundary = bin_mask - eroded
    perim = max(boundary.sum(), 1)
    compactness = np.clip((4 * np.pi * bin_mask.sum()) / (perim ** 2 + eps), 0, 1)

    try:
        from skimage.measure import regionprops, label as sk_label
        lbl = sk_label(bin_mask > 0)
        props = regionprops(lbl)
        if props:
            solidity = props[0].solidity
            eccentricity = props[0].eccentricity
        else:
            solidity = eccentricity = 0.0
    except Exception:
        solidity = eccentricity = 0.0

    mask_inside_prob = float(prob_np[bin_mask > 0].mean()) if bin_mask.sum() > 0 else 0.0
    mask_outside_prob = float(prob_np[bin_mask < 1].mean()) if (1 - bin_mask).sum() > 0 else 0.0

    mask_feats = np.array([
        bin_mask.mean(),
        float(n_comp) / 10.0,
        boundary.sum() / (total + eps),
        compactness,
        float(solidity),
        float(eccentricity),
        mask_inside_prob,
        mask_outside_prob,
    ], dtype=np.float32)

    return np.concatenate([img_feats, prob_feats, entropy_feats, mask_feats]).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Ambiente ROI Refiner
# ─────────────────────────────────────────────────────────────────────────────

class ROIRefinementEnv:
    """
    Ambiente RL per il raffinamento della segmentazione sul crop della ROI.

    Usato come Agent 2 nel pipeline doppio RL.
    Riceve un crop dell'immagine e una mappa di probabilità iniziale
    (generata da una semplice sogliatura o threshold adattivo),
    e raffina la maschera binaria step-by-step.
    """

    NUM_ACTIONS = 8
    STATE_DIM   = 21   # identico a MaskRefinementEnv per compatibilità agente

    def __init__(self, cfg: dict):
        rl = cfg.get("rl", {})
        self.max_steps   = rl.get("refine_max_steps", 10)
        self.w_dice      = rl.get("reward_dice_weight", 1.0)
        self.w_iou       = rl.get("reward_iou_weight", 0.5)
        self.w_frag      = rl.get("reward_fragment_penalty", 0.3)
        self.step_pen    = rl.get("reward_step_penalty", 0.02)
        self.w_terminal  = rl.get("reward_terminal_weight", 2.0)
        self.thresh_delta = 0.05

        self.image_np  : Optional[np.ndarray] = None
        self.prob_np   : Optional[np.ndarray] = None
        self.gt_np     : Optional[np.ndarray] = None
        self.mask_np   : Optional[np.ndarray] = None
        self.threshold : float = 0.5
        self.step_count: int   = 0
        self._prev_dice: float = 0.0
        self._prev_iou : float = 0.0

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(
        self,
        image_crop: np.ndarray,
        prob_map: np.ndarray,
        gt_mask_crop: np.ndarray,
    ) -> np.ndarray:
        """
        Resetta l'ambiente per un nuovo crop.
        image_crop:   [H,W] o [H,W,C] float32
        prob_map:     [H,W] mappa di probabilità in [0,1]
        gt_mask_crop: [H,W] binaria float32 (gt ritagliata alla ROI)
        """
        if image_crop.ndim == 3 and image_crop.shape[0] in (1, 3):
            image_crop = image_crop.transpose(1, 2, 0)
        if image_crop.ndim == 3:
            img_gray = image_crop.mean(axis=-1)
        else:
            img_gray = image_crop

        self.image_np   = img_gray.astype(np.float32)
        self.prob_np    = prob_map.astype(np.float32)
        self.gt_np      = gt_mask_crop.astype(np.float32)
        self.threshold  = 0.5
        self.mask_np    = (prob_map > 0.5).astype(np.float32)
        self.step_count = 0

        self._prev_dice = self._dice(self.mask_np)
        self._prev_iou  = self._iou(self.mask_np)

        return self._get_state()

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        done = False

        if action == 0:   # stop
            done = True
        elif action == 1: # accetta (no-op)
            pass
        elif action == 2: # espandi
            self.mask_np = _expand(self.mask_np)
        elif action == 3: # stringi
            self.mask_np = _shrink(self.mask_np)
        elif action == 4: # rimuovi piccoli
            self.mask_np = _remove_small(self.mask_np)
        elif action == 5: # smooth
            self.mask_np = _smooth(self.mask_np)
        elif action == 6: # threshold su
            self.mask_np, self.threshold = _threshold_up(
                self.prob_np, self.threshold, self.thresh_delta)
        elif action == 7: # threshold giù
            self.mask_np, self.threshold = _threshold_down(
                self.prob_np, self.threshold, self.thresh_delta)

        new_dice = self._dice(self.mask_np)
        new_iou  = self._iou(self.mask_np)
        delta_dice = new_dice - self._prev_dice
        delta_iou  = new_iou  - self._prev_iou

        _, n_comp = ndimage.label(self.mask_np > 0.5)
        frag_penalty = max(0, n_comp - 3) * self.w_frag

        reward = (
            self.w_dice * delta_dice
            + self.w_iou  * delta_iou
            - frag_penalty
            - self.step_pen
        )

        self._prev_dice = new_dice
        self._prev_iou  = new_iou
        self.step_count += 1

        if self.step_count >= self.max_steps:
            done = True
        if done:
            reward += self.w_terminal * new_dice

        info = {
            "dice": new_dice, "iou": new_iou,
            "n_components": n_comp, "action": action,
        }
        return self._get_state(), reward, done, info

    # ── Helpers ───────────────────────────────────────────────────────────────

    def get_refined_mask(self) -> np.ndarray:
        """Restituisce la maschera raffinata corrente [H,W]."""
        return self.mask_np

    def _get_state(self) -> np.ndarray:
        return extract_refine_state(self.image_np, self.mask_np, self.prob_np)

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
