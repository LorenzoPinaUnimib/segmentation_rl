"""
environment_roi.py — RL Environment per ROI Finding (Agent 1).

L'agente impara a spostare e ridimensionare un bounding box
per localizzare la regione di interesse (tumore) nell'immagine.

Stato: vettore di feature compatto [img_features | box_features | overlap_features]
Azioni (9):
  0 = stop
  1 = sposta su
  2 = sposta giù
  3 = sposta sinistra
  4 = sposta destra
  5 = allarga (espandi box)
  6 = stringi (riduci box)
  7 = allarga orizzontalmente
  8 = allarga verticalmente

Reward: ΔIOU_con_gt_box + bonus_terminale − penalità_passo
"""
import numpy as np
from typing import Optional, Tuple, Dict
from scipy import ndimage


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_roi_state(
    image_np: np.ndarray,   # [H, W] or [H, W, C]
    box: list,              # [x1, y1, x2, y2]
    gt_box: list,           # [x1, y1, x2, y2] ground truth (solo in training)
    img_size: int,
) -> np.ndarray:
    """
    Estrae un vettore di stato compatto per il ROI Finder.
    Dimensione: 6 (img_crop) + 6 (box_geo) + 4 (overlap) + 4 (gt_relative) = 20
    """
    if image_np.ndim == 3:
        img_gray = image_np.mean(axis=-1)
    else:
        img_gray = image_np.astype(np.float32)

    H, W = img_gray.shape
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)

    # --- Feature del crop interno al box (6)
    if x2 > x1 and y2 > y1:
        crop = img_gray[y1:y2, x1:x2]
        crop_feats = np.array([
            crop.mean(),
            crop.std(),
            crop.max(),
            np.percentile(crop, 25),
            np.percentile(crop, 75),
            float(crop.size) / (H * W),  # area relativa
        ], dtype=np.float32)
    else:
        crop_feats = np.zeros(6, dtype=np.float32)

    # --- Feature geometriche del box (normalizzate su img_size) (6)
    bw = (x2 - x1) / img_size
    bh = (y2 - y1) / img_size
    cx = ((x1 + x2) / 2) / img_size
    cy = ((y1 + y2) / 2) / img_size
    box_feats = np.array([
        cx, cy,
        bw, bh,
        bw * bh,           # area
        bw / (bh + 1e-6),  # aspect ratio
    ], dtype=np.float32)

    # --- Feature overlap con gt_box (4)
    gx1, gy1, gx2, gy2 = [int(v) for v in gt_box]
    ix1, iy1 = max(x1, gx1), max(y1, gy1)
    ix2, iy2 = min(x2, gx2), min(y2, gy2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_box = max((x2 - x1) * (y2 - y1), 1)
    area_gt  = max((gx2 - gx1) * (gy2 - gy1), 1)
    union = area_box + area_gt - inter
    iou = inter / (union + 1e-6)
    overlap_feats = np.array([
        iou,
        inter / (area_box + 1e-6),   # recall del box
        inter / (area_gt + 1e-6),    # precision del box
        float(inter > 0),            # flag overlap binario
    ], dtype=np.float32)

    # --- Posizione relativa del centro box rispetto a gt_box (4)
    gcx = ((gx1 + gx2) / 2) / img_size
    gcy = ((gy1 + gy2) / 2) / img_size
    rel_feats = np.array([
        cx - gcx,              # delta centro x
        cy - gcy,              # delta centro y
        bw - (gx2 - gx1) / img_size,  # delta width
        bh - (gy2 - gy1) / img_size,  # delta height
    ], dtype=np.float32)

    return np.concatenate([crop_feats, box_feats, overlap_feats, rel_feats]).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Ambiente ROI Finder
# ─────────────────────────────────────────────────────────────────────────────

class ROIFinderEnv:
    """
    Ambiente RL per trovare la ROI (bounding box) attorno al tumore.

    L'agente sposta e ridimensiona un box finché non coverge sulla ROI reale
    o raggiunge il limite di step.

    Usato come Agent 1 nel pipeline doppio RL.
    """

    NUM_ACTIONS = 9
    STATE_DIM   = 20  # 6 crop + 6 box_geo + 4 overlap + 4 gt_rel

    def __init__(self, cfg: dict):
        rl = cfg.get("rl", {})
        ds = cfg.get("dataset", {})

        img_size_cfg = ds.get("image_size", 256)
        if isinstance(img_size_cfg, (list, tuple)):
            self.img_size = img_size_cfg[0]
        else:
            self.img_size = int(img_size_cfg)

        self.step_size   = max(8, int(self.img_size * 0.06))
        self.max_steps   = rl.get("roi_max_steps", 100)
        self.step_pen    = rl.get("roi_step_penalty", 0.01)
        self.w_terminal  = rl.get("roi_terminal_weight", 3.0)

        # Stato interno
        self.image_np  : Optional[np.ndarray] = None
        self.gt_mask_np: Optional[np.ndarray] = None
        self.gt_box    : list = [0, 0, self.img_size, self.img_size]
        self.box       : list = [0, 0, self.img_size, self.img_size]
        self.prev_iou  : float = 0.0
        self.step_count: int   = 0
        

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self, image: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
        """
        Resetta l'ambiente per una nuova immagine.
        image:   [H,W] o [H,W,C] float32
        gt_mask: [H,W] binaria float32
        """
        if image.ndim == 3 and image.shape[0] in (1, 3):
            # Da [C,H,W] a [H,W,C]
            image = image.transpose(1, 2, 0)
        if image.ndim == 3:
            img_gray = image.mean(axis=-1)
        else:
            img_gray = image

        self.image_np   = img_gray.astype(np.float32)
        self.gt_mask_np = gt_mask.astype(np.float32)

        # Calcola gt_box dal mask
        ys, xs = np.where(gt_mask > 0.5)
        if len(ys) > 0:
            self.gt_box = [
                int(xs.min()), int(ys.min()),
                int(xs.max()), int(ys.max())
            ]
        else:
            self.gt_box = [0, 0, self.img_size, self.img_size]

        # Box iniziale: centro dell'immagine, dimensione media
        m = self.img_size // 4
        self.box = [m, m, self.img_size - m, self.img_size - m]
        self.prev_iou   = self._iou(self.box, self.gt_box)
        self.step_count = 0
        
        x1, y1, x2, y2 = self.box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        gx1, gy1, gx2, gy2 = self.gt_box
        gcx = (gx1 + gx2) / 2
        gcy = (gy1 + gy2) / 2

        # 1. Errore Centroidi (normalizzato per dimensione immagine)
        dist_x = abs(cx - gcx) / self.img_size
        dist_y = abs(cy - gcy) / self.img_size
        centroid_err = (dist_x + dist_y) / 2
        
        # 2. Errore Dimensioni (normalizzato per dimensioni GT)
        bw = max(x2 - x1, 1)
        bh = max(y2 - y1, 1)
        gbw = max(gx2 - gx1, 1)
        gbh = max(gy2 - gy1, 1)
        
        size_err = (abs(bw - gbw) / gbw + abs(bh - gbh) / gbh) / 2
        
        # Combinazione degli errori (0 = perfetto, >0 = errore)
        # Puoi dare pesi diversi se necessario
        self.prev_total_err = (0.5 * centroid_err + 0.5 * size_err)

        return self._get_state()

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Applica un'azione e restituisce (stato, reward, done, info)."""
        done = False
        s = self.step_size
        x1, y1, x2, y2 = self.box

        if action == 0:   # stop
            done = True
        elif action == 1: # su
            y1 -= s; y2 -= s
        elif action == 2: # giù
            y1 += s; y2 += s
        elif action == 3: # sinistra
            x1 -= s; x2 -= s
        elif action == 4: # destra
            x1 += s; x2 += s
        elif action == 5: # allarga tutto
            x1 -= s; y1 -= s; x2 += s; y2 += s
        elif action == 6: # stringi tutto
            x1 += s; y1 += s; x2 -= s; y2 -= s
        elif action == 7: # allarga orizzontale
            x1 -= s; x2 += s
        elif action == 8: # allarga verticale
            y1 -= s; y2 += s

        # Clipping e min size
        x1 = int(np.clip(x1, 0, self.img_size - 20))
        y1 = int(np.clip(y1, 0, self.img_size - 20))
        x2 = int(np.clip(x2, x1 + 20, self.img_size))
        y2 = int(np.clip(y2, y1 + 20, self.img_size))
        self.box = [x1, y1, x2, y2]

        # Reward
        new_iou = self._iou(self.box, self.gt_box)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        gx1, gy1, gx2, gy2 = self.gt_box
        gcx = (gx1 + gx2) / 2
        gcy = (gy1 + gy2) / 2

        # 1. Errore Centroidi (normalizzato per dimensione immagine)
        dist_x = abs(cx - gcx) / self.img_size
        dist_y = abs(cy - gcy) / self.img_size
        centroid_err = (dist_x + dist_y) / 2
        
        # 2. Errore Dimensioni (normalizzato per dimensioni GT)
        bw = max(x2 - x1, 1)
        bh = max(y2 - y1, 1)
        gbw = max(gx2 - gx1, 1)
        gbh = max(gy2 - gy1, 1)
        
        size_err = (abs(bw - gbw) / gbw + abs(bh - gbh) / gbh) / 2
        
        # Combinazione degli errori (0 = perfetto, >0 = errore)
        # Puoi dare pesi diversi se necessario
        total_err = (0.5 * centroid_err + 0.5 * size_err)
        
        # Reward basata sulla riduzione dell'errore (Delta)
        # Se total_err diminuisce, la reward è positiva
        current_score = 1.0 - total_err
        prev_score = 1.0 - self.prev_total_err # Devi aggiungere self.prev_total_err nel reset
        
        reward = (current_score - prev_score) * 10.0 - self.step_pen * self.step_count
        
        self.prev_total_err = total_err
        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True

        if done:
            reward += self.w_terminal * new_iou

        info = {
            "iou":      new_iou,
            "box":      list(self.box),
            "gt_box":   list(self.gt_box),
            "step":     self.step_count,
        }
        return self._get_state(), reward, done, info

    # ── Helpers ───────────────────────────────────────────────────────────────

    def get_box(self) -> list:
        """Restituisce il bounding box finale [x1, y1, x2, y2]."""
        return list(self.box)

    def get_crop(self, image_tensor) -> Tuple[np.ndarray, list]:
        """
        Restituisce il crop dell'immagine alla box trovata.
        image_tensor: [C,H,W] numpy o tensor
        """
        x1, y1, x2, y2 = [int(v) for v in self.box]
        if hasattr(image_tensor, 'numpy'):
            arr = image_tensor.cpu().numpy()
        else:
            arr = np.asarray(image_tensor)
        if arr.ndim == 3:
            crop = arr[:, y1:y2, x1:x2]
        else:
            crop = arr[y1:y2, x1:x2]
        return crop, self.box

    def _get_state(self) -> np.ndarray:
        return extract_roi_state(self.image_np, self.box, self.gt_box, self.img_size)

    @staticmethod
    def _iou(bA: list, bB: list) -> float:
        xA = max(bA[0], bB[0]); yA = max(bA[1], bB[1])
        xB = min(bA[2], bB[2]); yB = min(bA[3], bB[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        aA = max((bA[2]-bA[0]) * (bA[3]-bA[1]), 1)
        aB = max((bB[2]-bB[0]) * (bB[3]-bB[1]), 1)
        return inter / float(aA + aB - inter + 1e-6)
