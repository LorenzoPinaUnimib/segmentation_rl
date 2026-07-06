"""
environment_roi.py — RL Environment per ROI Finding (Agent 1).

L'agente impara a spostare e ridimensionare un bounding box
per localizzare la regione di interesse (tumore) nell'immagine.

STATO (CNN, NON più un vettore di feature):
  Tensore [2, S, S] dove S = cfg["rl"]["roi_cnn_size"] (default 48):
    - canale 0: immagine in scala di grigi ridimensionata a S×S, normalizzata [0,1]
                (calcolata UNA SOLA VOLTA in reset() e messa in cache → ogni
                step() costa solo il ridisegno del box, non un resize
                dell'immagine intera: ottimizzazione chiave per la velocità)
    - canale 1: maschera binaria del box corrente, renderizzata sulla griglia S×S

In questo modo la Q-network (vedi agent_roi.py) è una CNN che osserva
direttamente "dove si trova il box rispetto al contenuto dell'immagine",
al posto delle 20 feature scalari estratte a mano della versione precedente.

Azioni (17):
  0  = stop
  1  = sposta su (entrambi i lati Y)
  2  = sposta giù
  3  = sposta sinistra (entrambi i lati X)
  4  = sposta destra
  5  = allarga (espandi box su tutti i lati)
  6  = stringi (riduci box su tutti i lati)
  7  = allarga orizzontalmente (entrambi i lati X)
  8  = allarga verticalmente (entrambi i lati Y)
  9  = lato sinistro dentro  (x1 += step)
  10 = lato sinistro fuori   (x1 -= step)
  11 = lato destro dentro    (x2 -= step)
  12 = lato destro fuori     (x2 += step)
  13 = lato superiore dentro (y1 += step)
  14 = lato superiore fuori  (y1 -= step)
  15 = lato inferiore dentro (y2 -= step)
  16 = lato inferiore fuori  (y2 += step)

  Le azioni 9-16 muovono UN SOLO lato per volta: permettono di adattare la
  forma/posizione del box a tumori non centrati o non quadrati senza dover
  passare per combinazioni indirette di move+resize simmetrici.

STEP SIZE ADATTIVO (coarse-to-fine):
  Lo spostamento per singola azione non è più fisso per tutto l'episodio:
  decresce a fasi (100% → 50% → 25% dello step base) man mano che
  avanzano gli step. Le prime azioni servono per un posizionamento grezzo
  veloce, le ultime per una convergenza fine — evita l'overshoot continuo
  che uno step fisso da ~15px causava vicino al target.

Reward: ΔIOU_con_gt_box + bonus_terminale − penalità_passo (costante)
"""
import numpy as np
import cv2
from typing import Optional, Tuple, Dict


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction (CNN state: immagine + box renderizzati su una griglia)
# ─────────────────────────────────────────────────────────────────────────────

def _resize_gray_normalized(img_gray: np.ndarray, size: int) -> np.ndarray:
    """Ridimensiona un'immagine [H,W] a [size,size] float32 normalizzata in [0,1]."""
    out = cv2.resize(
        img_gray.astype(np.float32), (size, size),
        interpolation=cv2.INTER_AREA,
    )
    mn, mx = float(out.min()), float(out.max())
    if mx > mn:
        out = (out - mn) / (mx - mn)
    else:
        out = np.zeros_like(out)
    return out.astype(np.float32)


def _box_to_grid(box: list, img_size: int, grid_size: int) -> np.ndarray:
    """Renderizza il box (coordinate in pixel immagine) come maschera binaria su una griglia grid_size×grid_size."""
    scale = grid_size / float(img_size)
    x1, y1, x2, y2 = box
    gx1 = int(np.clip(round(x1 * scale), 0, grid_size))
    gy1 = int(np.clip(round(y1 * scale), 0, grid_size))
    gx2 = int(np.clip(round(x2 * scale), gx1 + 1, grid_size))
    gy2 = int(np.clip(round(y2 * scale), gy1 + 1, grid_size))
    grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    grid[gy1:gy2, gx1:gx2] = 1.0
    return grid


def _gradient_magnitude_normalized(img_gray_small: np.ndarray) -> np.ndarray:
    """Canale extra di bordo/gradiente (Sobel), calcolato una sola volta per reset."""
    gx = cv2.Sobel(img_gray_small, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray_small, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    mx = float(mag.max())
    if mx > 1e-6:
        mag = mag / mx
    return mag.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Ambiente ROI Finder
# ─────────────────────────────────────────────────────────────────────────────

class ROIFinderEnv:
    """
    Ambiente RL per trovare la ROI (bounding box) attorno al tumore.

    L'agente sposta e ridimensiona un box finché non converge sulla ROI reale
    o raggiunge il limite di step.

    Usato come Agent 1 nel pipeline doppio RL.

    Lo stato restituito (vedi _get_state) è ora un'immagine a 2 canali
    (immagine + maschera del box), pensata per essere consumata da una
    Q-network convoluzionale (DuelingCNN in agent_roi.py) al posto del
    precedente vettore di 20 feature scalari.
    """

    NUM_ACTIONS     = 17
    STATE_CHANNELS  = 2     # canale immagine + canale box

    def __init__(self, cfg: dict):
        rl = cfg.get("rl", {})
        ds = cfg.get("dataset", {})

        img_size_cfg = ds.get("image_size", 256)
        if isinstance(img_size_cfg, (list, tuple)):
            self.img_size = img_size_cfg[0]
        else:
            self.img_size = int(img_size_cfg)

        # Risoluzione spaziale dello stato CNN (piccola apposta per la velocità:
        # un forward CNN su 48×48 è enormemente più economico che su 256×256,
        # e per il task di localizzazione non serve dettaglio fine).
        self.cnn_size = int(rl.get("roi_cnn_size", 48))

        # OTTIMIZZAZIONE: canali stato configurabili (2 o 3, con gradiente extra)
        self.state_channels = int(rl.get("roi_state_channels", self.STATE_CHANNELS))

        # OTTIMIZZAZIONE: box iniziale randomizzato in training (rompe il bias
        # "tumore sempre al centro" del box fisso originale)
        self.random_init = bool(rl.get("roi_random_init", False))
        scale_range = rl.get("roi_random_init_scale_range", [0.25, 0.6])
        self.random_init_scale_min = float(scale_range[0])
        self.random_init_scale_max = float(scale_range[1])
        self.random_init_jitter = float(rl.get("roi_random_init_center_jitter", 0.3))

        # Step size adattivo: base_step è il valore di partenza (fase coarse),
        # min_step è il pavimento della fase fine. Vedi _current_step_size().
        self.base_step   = max(8, int(self.img_size * 0.06))
        self.min_step    = max(2, self.base_step // 4)
        self.max_steps   = rl.get("roi_max_steps", 100)
        self.step_pen    = rl.get("roi_step_penalty", 0.01)
        self.w_terminal  = rl.get("roi_terminal_weight", 3.0)

        # Stato interno
        self.image_np   : Optional[np.ndarray] = None
        self.gt_mask_np : Optional[np.ndarray] = None
        self.gt_box     : list = [0, 0, self.img_size, self.img_size]
        self.box        : list = [0, 0, self.img_size, self.img_size]
        self.prev_iou   : float = 0.0
        self.step_count : int   = 0

        # Cache: immagine ridimensionata calcolata una sola volta per reset()
        self._img_small : Optional[np.ndarray] = None
        self._grad_small: Optional[np.ndarray] = None

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self, image: np.ndarray, gt_mask: np.ndarray, randomize: bool = False) -> np.ndarray:
        """
        Resetta l'ambiente per una nuova immagine.
        image:     [H,W] o [H,W,C] float32
        gt_mask:   [H,W] binaria float32
        randomize: se True (training) e roi_random_init e' abilitato in config,
                   il box iniziale viene campionato casualmente invece di
                   essere sempre centrato. In eval va lasciato False.
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

        # Ottimizzazione: il resize dell'immagine è il pezzo più costoso della
        # costruzione dello stato CNN → lo facciamo una volta sola qui, e lo
        # riusiamo identico per tutti gli step dell'episodio (cambia solo il
        # box, che è economico da renderizzare).
        self._img_small = _resize_gray_normalized(self.image_np, self.cnn_size)
        if self.state_channels >= 3:
            self._grad_small = _gradient_magnitude_normalized(self._img_small)

        # Calcola gt_box dal mask
        ys, xs = np.where(gt_mask > 0.5)
        if len(ys) > 0:
            self.gt_box = [
                int(xs.min()), int(ys.min()),
                int(xs.max()), int(ys.max())
            ]
        else:
            self.gt_box = [0, 0, self.img_size, self.img_size]

        # Box iniziale: centro dell'immagine, dimensione media (default),
        # oppure campionato casualmente se randomize=True (vedi docstring).
        if randomize and self.random_init:
            self.box = self._sample_random_box()
        else:
            m = self.img_size // 4
            self.box = [m, m, self.img_size - m, self.img_size - m]
        self.prev_iou   = self._iou(self.box, self.gt_box)
        self.step_count = 0

        return self._get_state()

    # ── Random init ──────────────────────────────────────────────────────────

    def _sample_random_box(self) -> list:
        """Campiona un box iniziale casuale (scala + posizione) al posto del box fisso centrato."""
        scale = np.random.uniform(self.random_init_scale_min, self.random_init_scale_max)
        w = h = max(20, int(self.img_size * scale))

        jitter = self.random_init_jitter * self.img_size
        cx = self.img_size / 2.0 + np.random.uniform(-jitter, jitter)
        cy = self.img_size / 2.0 + np.random.uniform(-jitter, jitter)

        x1 = int(np.clip(cx - w / 2, 0, self.img_size - 20))
        y1 = int(np.clip(cy - h / 2, 0, self.img_size - 20))
        x2 = int(np.clip(x1 + w, x1 + 20, self.img_size))
        y2 = int(np.clip(y1 + h, y1 + 20, self.img_size))
        return [x1, y1, x2, y2]

    # ── Step ──────────────────────────────────────────────────────────────────

    def _current_step_size(self) -> int:
        """
        Schedule coarse-to-fine: step size grande nei primi step (ricerca
        veloce e grossolana), poi via via più piccolo per permettere una
        convergenza fine senza overshoot continuo.
          [0%, 40%)  del episodio → step base (100%)
          [40%, 75%) del episodio → step/2
          [75%, 100%] del episodio → step/4 (mai sotto min_step)
        """
        frac = self.step_count / max(1, self.max_steps)
        if frac < 0.40:
            return self.base_step
        elif frac < 0.75:
            return max(self.min_step, self.base_step // 2)
        else:
            return max(self.min_step, self.base_step // 4)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Applica un'azione e restituisce (stato, reward, done, info)."""
        done = False
        s = self._current_step_size()
        x1, y1, x2, y2 = self.box

        if action == 0:    # stop
            done = True
        elif action == 1:  # su (entrambi i lati Y)
            y1 -= s; y2 -= s
        elif action == 2:  # giù
            y1 += s; y2 += s
        elif action == 3:  # sinistra (entrambi i lati X)
            x1 -= s; x2 -= s
        elif action == 4:  # destra
            x1 += s; x2 += s
        elif action == 5:  # allarga tutto
            x1 -= s; y1 -= s; x2 += s; y2 += s
        elif action == 6:  # stringi tutto
            x1 += s; y1 += s; x2 -= s; y2 -= s
        elif action == 7:  # allarga orizzontale
            x1 -= s; x2 += s
        elif action == 8:  # allarga verticale
            y1 -= s; y2 += s
        elif action == 9:  # lato sinistro dentro
            x1 += s
        elif action == 10: # lato sinistro fuori
            x1 -= s
        elif action == 11: # lato destro dentro
            x2 -= s
        elif action == 12: # lato destro fuori
            x2 += s
        elif action == 13: # lato superiore dentro
            y1 += s
        elif action == 14: # lato superiore fuori
            y1 -= s
        elif action == 15: # lato inferiore dentro
            y2 -= s
        elif action == 16: # lato inferiore fuori
            y2 += s

        # Clipping e min size
        x1 = int(np.clip(x1, 0, self.img_size - 20))
        y1 = int(np.clip(y1, 0, self.img_size - 20))
        x2 = int(np.clip(x2, x1 + 20, self.img_size))
        y2 = int(np.clip(y2, y1 + 20, self.img_size))
        self.box = [x1, y1, x2, y2]

        # Reward: variazione diretta di IoU rispetto al box GT (coerente
        # con il docstring della classe), non più un proxy basato su
        # errore di centroide/dimensione. Questo allinea il segnale di
        # apprendimento per-step alla metrica che vogliamo massimizzare.
        new_iou = self._iou(self.box, self.gt_box)

        # Reward shaping: ΔIoU scalata, penalità di step COSTANTE (non più
        # proporzionale a step_count, per non incentivare uno stop prematuro
        # negli episodi lunghi).
        reward = (new_iou - self.prev_iou) * 10.0 - self.step_pen

        self.prev_iou = new_iou
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
        """
        Stato CNN: tensore [2, cnn_size, cnn_size].
          canale 0 = immagine (cache, calcolata una sola volta in reset)
          canale 1 = box corrente renderizzato sulla griglia
        """
        box_grid = _box_to_grid(self.box, self.img_size, self.cnn_size)
        if self.state_channels >= 3 and self._grad_small is not None:
            return np.stack([self._img_small, box_grid, self._grad_small], axis=0).astype(np.float32)
        return np.stack([self._img_small, box_grid], axis=0).astype(np.float32)

    @staticmethod
    def _iou(bA: list, bB: list) -> float:
        xA = max(bA[0], bB[0]); yA = max(bA[1], bB[1])
        xB = min(bA[2], bB[2]); yB = min(bA[3], bB[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        aA = max((bA[2]-bA[0]) * (bA[3]-bA[1]), 1)
        aB = max((bB[2]-bB[0]) * (bB[3]-bB[1]), 1)
        return inter / float(aA + aB - inter + 1e-6)
