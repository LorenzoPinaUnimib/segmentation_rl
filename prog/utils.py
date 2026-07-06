"""
utils.py
────────
Funzioni pure riusate da ambiente, callback e valutatore.
Nessuna dipendenza da Gym/SB3 qui: solo numpy/cv2.
"""
import numpy as np


def linear_schedule(initial_value: float, final_value: float = 0.0):
    """Scheduler compatibile con SB3: riceve progress_remaining (1 -> 0)."""
    def scheduler(progress_remaining: float) -> float:
        return final_value + progress_remaining * (initial_value - final_value)
    return scheduler


def build_box_vec(cx: float, cy: float, w: float, h: float, W: int, H: int) -> np.ndarray:
    """Vettore (4,) float32 con cx,cy,w,h normalizzati in [0,1].

    FIX (Dict observation space): sostituisce build_coord_planes. Stessa
    informazione (posizione/dimensione del box), ma passata come input
    NUMERICO diretto alla policy tramite la chiave "box_vec" della Dict
    observation space (vedi environment.py + policy="MultiInputPolicy" in
    train.py), invece che "spalmata" su 4 piani immagine interi a valore
    costante. Due vantaggi diretti:
      - dimezza i canali dell'immagine (8->4: 3 RGB + 1 maschera box) e quindi
        la RAM del replay buffer di SAC a parita' di buffer_size;
      - la rete riceve il box come numero esatto invece di doverlo dedurre da
        un piano spaziale a valore costante -- piu' facile da imparare.
    """
    norm = np.array([cx / W, cy / H, w / W, h / H], dtype=np.float32)
    return np.clip(norm, 0.0, 1.0)


def mask_fn(env):
    """Wrapper richiesto da ActionMasker (sb3-contrib)."""
    return env.unwrapped.action_masks()


def compute_iou(b1, b2) -> float:
    """b1, b2 in formato [x, y, w, h]."""
    xi1, yi1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    xi2 = min(b1[0] + b1[2], b2[0] + b2[2])
    yi2 = min(b1[1] + b1[3], b2[1] + b2[3])
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union_area = (b1[2] * b1[3]) + (b2[2] * b2[3]) - inter_area
    return float(inter_area / max(1e-6, union_area))


def compute_center_distance(b1, b2) -> float:
    c1_x, c1_y = b1[0] + b1[2] / 2.0, b1[1] + b1[3] / 2.0
    c2_x, c2_y = b2[0] + b2[2] / 2.0, b2[1] + b2[3] / 2.0
    return float(np.sqrt((c1_x - c2_x) ** 2 + (c1_y - c2_y) ** 2))


def to_bgr_image(image_chw_float: np.ndarray) -> np.ndarray:
    """Converte un tensore immagine CHW float [0,1] in un frame BGR uint8 per OpenCV."""
    img = image_chw_float
    if img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
        import cv2
        return cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    import cv2
    return cv2.cvtColor((img[0] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
