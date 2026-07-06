""""
rl_trainer.py — Trainer per il pipeline doppio RL (senza U-Net).

Fase 1: Addestra ROIFinderAgent su environment_roi
        → trova la bounding box del tumore

Fase 2: Addestra ROIRefinementAgent su environment_refine
        → raffina la maschera binaria nel crop trovato dal finder

NOVITÀ:
  - Ogni iterazione di training usa TUTTE le immagini del train set.
  - Ad ogni eval (eval_every) si valuta su TUTTO il val set (IoU/Dice).
  - Il best checkpoint viene scelto in base alle metriche sul val set.
  - Il test set non viene usato durante il training.

Parametri aggiuntivi nel config.yaml (tutti opzionali, con default):
  rl:
    eval_on_full_val: true          # validazione su tutto il val set
    reward_smooth_window: 20        # finestra media mobile reward
"""
import os
import time
import collections
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Callable, Dict, List, Tuple, Optional
from tqdm import tqdm
from scipy import ndimage

from ..rl.environment_roi import ROIFinderEnv
from ..rl.environment_refine import ROIRefinementEnv
from ..rl.agent_roi import ROIFinderAgent
from ..rl.agent_refine import ROIRefinementAgent
from ..eval.metrics import dice_coefficient, iou_score


# ─────────────────────────────────────────────────────────────────────────────
# Utility: reward buffer rolling per stabilizzare la reward media
# ─────────────────────────────────────────────────────────────────────────────

class EpisodeRewardBuffer:
    """
    Buffer a finestra scorrevole per calcolare la media mobile della reward.
    Evita che singoli episodi anomali distorcano le decisioni di salvataggio.
    """
    def __init__(self, window: int = 20):
        self.window = window
        self._buf: collections.deque = collections.deque(maxlen=window)

    def push(self, value: float) -> None:
        self._buf.append(value)

    @property
    def mean(self) -> float:
        return float(np.mean(self._buf)) if self._buf else 0.0

    @property
    def std(self) -> float:
        return float(np.std(self._buf)) if len(self._buf) > 1 else 0.0

    def __len__(self) -> int:
        return len(self._buf)


# ─────────────────────────────────────────────────────────────────────────────
# Utility: prob map senza U-Net (sogliatura adattiva sul crop)
# ─────────────────────────────────────────────────────────────────────────────

def make_prob_map_adaptive(crop_gray: np.ndarray) -> np.ndarray:
    """
    Genera una mappa di probabilità senza U-Net usando:
    1. Normalizzazione locale + CLAHE (equalizzazione a contrasto limitato)
    2. Sogliatura adattiva basata su intensità (post-CLAHE) + gradiente
    3. Smoothing gaussiano → valori in [0,1]

    OTTIMIZZAZIONE: aggiunto un passo CLAHE prima del punteggio combinato.
    La normalizzazione min-max globale da sola sovra/sotto-espone crop con
    tumori a basso contrasto locale (comuni in MRI non-contrastate): CLAHE
    rinforza il contrasto locale attorno al tumore senza amplificare troppo
    il rumore di fondo, alzando il "tetto" di qualità raggiungibile da questa
    baseline puramente classica (nessun training, nessuna U-Net).
    """
    import cv2

    crop = crop_gray.astype(np.float32)
    crop_min, crop_max = crop.min(), crop.max()
    if crop_max > crop_min:
        crop = (crop - crop_min) / (crop_max - crop_min + 1e-8)

    crop_u8 = np.clip(crop * 255.0, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    crop_eq = clahe.apply(crop_u8).astype(np.float32) / 255.0

    gx = ndimage.sobel(crop_eq, axis=0)
    gy = ndimage.sobel(crop_eq, axis=1)
    gradient = np.hypot(gx, gy)
    gradient = gradient / (gradient.max() + 1e-8)

    # Combina intensita' originale (evita di seguire artefatti puri di CLAHE),
    # intensita' equalizzata (contrasto locale) e gradiente (bordi).
    score = 0.4 * crop + 0.35 * crop_eq + 0.25 * gradient
    prob = ndimage.gaussian_filter(score, sigma=1.5)
    prob = np.clip(prob, 0, 1)
    return prob.astype(np.float32)


def make_prob_map_otsu(crop_gray: np.ndarray) -> np.ndarray:
    """
    Alternativa: sogliatura Otsu + distanza transform come prob map.
    """
    crop = crop_gray.astype(np.float32)
    crop_min, crop_max = crop.min(), crop.max()
    if crop_max > crop_min:
        crop = (crop - crop_min) / (crop_max - crop_min + 1e-8)

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    vote_map = np.zeros_like(crop)
    for t in thresholds:
        vote_map += (crop > t).astype(np.float32)
    prob = vote_map / len(thresholds)

    prob = ndimage.gaussian_filter(prob, sigma=1.5)
    return np.clip(prob, 0, 1).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: converti tensore → numpy grigio
# ─────────────────────────────────────────────────────────────────────────────

def _tensor_to_gray(t: torch.Tensor) -> np.ndarray:
    """[C,H,W] tensor → [H,W] numpy float32 in [0,1]."""
    arr = t.cpu().numpy()
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[0] == 3:
            arr = arr.mean(axis=0)
        else:
            arr = arr.mean(axis=0)
    return arr.astype(np.float32)


def _tensor_to_mask(t: torch.Tensor) -> np.ndarray:
    """[1,H,W] o [H,W] tensor → [H,W] numpy binaria float32."""
    arr = t.cpu().numpy().squeeze()
    return (arr > 0.5).astype(np.float32)


def _collect_loader_samples(loader: Optional[DataLoader]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Raccoglie tutte le (image, mask) da un DataLoader."""
    samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
    if loader is None:
        return samples
    for batch in loader:
        imgs = batch["image"]
        masks = batch["mask"]
        for i in range(imgs.size(0)):
            samples.append((imgs[i], masks[i]))
    return samples


def collect_train_samples(train_loader: DataLoader) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Raccoglie tutte le immagini del train set per il training."""
    return _collect_loader_samples(train_loader)


# ─────────────────────────────────────────────────────────────────────────────
# Valutazione full su un loader (tutte le immagini)
# ─────────────────────────────────────────────────────────────────────────────

def _full_eval_finder(
    agent: "ROIFinderAgent",
    env: "ROIFinderEnv",
    loader: DataLoader,
) -> Dict[str, float]:
    """
    Valuta il ROIFinderAgent su TUTTE le immagini del loader.
    Restituisce IoU medio, std e mediana per una stima stabile.
    """
    ious: List[float] = []
    samples = []
    for batch in loader:
        imgs    = batch["image"]
        targets = batch["mask"]
        for i in range(imgs.size(0)):
            samples.append((imgs[i], targets[i]))

    t0 = time.time()
    pbar = tqdm(samples, desc="  Eval (val)", unit="img", leave=False, dynamic_ncols=True)
    for image_t, mask_t in pbar:
        img_gray = _tensor_to_gray(image_t)
        gt_mask  = _tensor_to_mask(mask_t)
        state = env.reset(img_gray, gt_mask)
        done = False
        last_iou = 0.0
        while not done:
            action = agent.act(state, greedy=True)
            state, _, done, info = env.step(action)
            last_iou = info["iou"]
        ious.append(last_iou)
        pbar.set_postfix({"IoU": f"{np.mean(ious):.3f}"})
    pbar.close()
    elapsed = time.time() - t0

    if not ious:
        return {"mean": 0.0, "std": 0.0, "median": 0.0, "n": 0, "time_sec": elapsed}
    return {
        "mean":   float(np.mean(ious)),
        "std":    float(np.std(ious)),
        "median": float(np.median(ious)),
        "n":      len(ious),
        "time_sec": elapsed,
    }


def _full_eval_refiner(
    finder_agent: "ROIFinderAgent",
    finder_env: "ROIFinderEnv",
    refiner_agent: "ROIRefinementAgent",
    refiner_env: "ROIRefinementEnv",
    loader: DataLoader,
    img_size: int,
    prob_method: str = "adaptive",
) -> Dict[str, float]:
    """
    Valuta l'intera pipeline (finder + refiner) su TUTTE le immagini del loader.
    Restituisce Dice e IoU medi, std, mediana per stabilità statistica.
    """
    from skimage.transform import resize as sk_resize

    dices_bl: List[float] = []
    dices_rl: List[float] = []
    ious_bl:  List[float] = []
    ious_rl:  List[float] = []

    for batch in loader:
        imgs    = batch["image"]
        targets = batch["mask"]

        for i in range(imgs.size(0)):
            img_gray = _tensor_to_gray(imgs[i])
            gt_mask  = _tensor_to_mask(targets[i])

            # ── Finder ──────────────────────────────────────────────────────
            state = finder_env.reset(img_gray, gt_mask)
            done = False
            while not done:
                action = finder_agent.act(state, greedy=True)
                state, _, done, _ = finder_env.step(action)

            box = finder_env.get_box()
            x1, y1, x2, y2 = [int(v) for v in box]
            crop_gray = img_gray[y1:y2, x1:x2]
            gt_crop   = gt_mask[y1:y2, x1:x2]

            if crop_gray.size == 0:
                crop_gray = img_gray
                gt_crop   = gt_mask

            crop_r = sk_resize(crop_gray, (img_size, img_size),
                               anti_aliasing=True, preserve_range=True).astype(np.float32)
            gt_r   = (sk_resize(gt_crop, (img_size, img_size),
                                anti_aliasing=False, preserve_range=True) > 0.5).astype(np.float32)

            if prob_method == "otsu":
                prob_map = make_prob_map_otsu(crop_r)
            else:
                prob_map = make_prob_map_adaptive(crop_r)

            baseline_mask = (prob_map > 0.5).astype(np.float32)

            # ── Refiner ──────────────────────────────────────────────────────
            state = refiner_env.reset(crop_r, prob_map, gt_r)
            done = False
            while not done:
                action = refiner_agent.act(state, greedy=True)
                state, _, done, _ = refiner_env.step(action)

            refined_mask = refiner_env.get_refined_mask()

            # Metriche
            d_bl = dice_coefficient(baseline_mask, gt_r)
            d_rl = dice_coefficient(refined_mask,  gt_r)
            i_bl = iou_score(baseline_mask, gt_r)
            i_rl = iou_score(refined_mask,  gt_r)

            dices_bl.append(d_bl)
            dices_rl.append(d_rl)
            ious_bl.append(i_bl)
            ious_rl.append(i_rl)

    def _stats(vals: List[float]) -> Dict[str, float]:
        if not vals:
            return {"mean": 0.0, "std": 0.0, "median": 0.0}
        return {
            "mean":   float(np.mean(vals)),
            "std":    float(np.std(vals)),
            "median": float(np.median(vals)),
        }

    return {
        "dice_bl":  _stats(dices_bl),
        "dice_rl":  _stats(dices_rl),
        "iou_bl":   _stats(ious_bl),
        "iou_rl":   _stats(ious_rl),
        "n":        len(dices_rl),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Trainer Fase 1: ROI Finder
# ─────────────────────────────────────────────────────────────────────────────

class ROIFinderTrainer:
    """
    Addestra ROIFinderAgent a localizzare il tumore.

    Modifiche rispetto alla versione originale:
    - Ogni iterazione usa TUTTE le immagini del train set.
    - eval_every: valutazione FULL sul val set (IoU medio).
    - Best checkpoint basato sul val IoU, non sul test set.
    """

    def __init__(
        self,
        agent: ROIFinderAgent,
        env: ROIFinderEnv,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        cfg: dict,
        on_eval_callback: Optional[Callable[[int, Dict], None]] = None,
    ):
        self.agent        = agent
        self.env          = env
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.cfg          = cfg
        self.on_eval_callback = on_eval_callback

        rl = cfg.get("rl", {})
        self.episodes    = rl.get("roi_episodes", rl.get("episodes", 300))
        self.target_freq = rl.get("target_update_freq", 20)

        # Stabilizzazione reward
        smooth_w = rl.get("reward_smooth_window", 20)
        self._reward_buf = EpisodeRewardBuffer(window=smooth_w)
        self._iou_buf    = EpisodeRewardBuffer(window=smooth_w)

        # Valutazione full sul val set (default True)
        self._full_eval = rl.get("eval_on_full_val", rl.get("eval_on_full_test", True))

        self.history: Dict[str, List] = {
            "rewards":        [],   # reward raw per episodio
            "rewards_smooth": [],   # media mobile reward
            "iou":            [],   # iou raw per episodio (train)
            "iou_smooth":     [],   # media mobile iou (train)
            "epsilon":        [],
            "val_iou_mean":   [],   # full-eval mean IoU
            "val_iou_std":    [],   # full-eval std IoU
            "val_iou_median": [],   # full-eval mediana IoU
            "val_ep":         [],   # episodi in cui è stato fatto eval
        }
        self.best_val_iou = 0.0
        self._all_samples = collect_train_samples(train_loader)
        n_train = len(self._all_samples)
        n_val   = len(_collect_loader_samples(val_loader))
        print(f"  Campioni training (solo train): {n_train} | Val: {n_val}")

    def _run_episode(self, image_t, mask_t, train: bool) -> Dict:
        img_gray = _tensor_to_gray(image_t)
        gt_mask  = _tensor_to_mask(mask_t)

        # OTTIMIZZAZIONE: box iniziale randomizzato SOLO in training (train=True).
        # In eval (train=False) resta il box centrato deterministico, per una
        # valutazione riproducibile e confrontabile tra checkpoint.
        state = self.env.reset(img_gray, gt_mask, randomize=train)
        total_reward = 0.0

        while True:
            action = self.agent.act(state, greedy=not train)
            next_state, reward, done, info = self.env.step(action)

            if train:
                self.agent.remember(state, action, reward, next_state, done)
                self.agent.learn()

            total_reward += reward
            state = next_state
            if done:
                break

        return {
            "reward": total_reward,
            "iou":    info["iou"],
            "box":    info["box"],
        }

    def _evaluate_full(self) -> Dict[str, float]:
        """Valuta su TUTTE le immagini del val set."""
        return _full_eval_finder(self.agent, self.env, self.val_loader)

    def _run_iteration(self, train: bool, episode_num: int = 0) -> Dict[str, float]:
        """Un'iterazione = un passaggio su TUTTE le immagini del train set."""
        if not self._all_samples:
            return {"reward": 0.0, "iou": 0.0, "time_sec": 0.0}

        indices = np.random.permutation(len(self._all_samples))
        rewards, ious = [], []

        t0 = time.time()
        pbar = tqdm(
            indices,
            desc=f"  Iter {episode_num:04d}/{self.episodes}" if train else "  Iter (eval)",
            unit="img",
            leave=False,
            dynamic_ncols=True,
        )
        for idx in pbar:
            image_t, mask_t = self._all_samples[int(idx)]
            ep_info = self._run_episode(image_t, mask_t, train=train)
            rewards.append(ep_info["reward"])
            ious.append(ep_info["iou"])
            pbar.set_postfix({
                "R":   f"{np.mean(rewards):+.3f}",
                "IoU": f"{np.mean(ious):.3f}",
            })
        pbar.close()
        elapsed = time.time() - t0

        return {
            "reward":   float(np.mean(rewards)),
            "iou":      float(np.mean(ious)),
            "time_sec": elapsed,
        }

    def train(self, eval_every: int = 20) -> Dict:
        print(f"{'='*60}")
        print(f" [ROIFinder] Training per {self.episodes} iterazioni")
        print(f" Ogni iterazione usa TUTTE le {len(self._all_samples)} immagini (train)")
        if self._full_eval:
            print(f" Validazione su TUTTO il val set ogni {eval_every} iterazioni")
        print(f"{'='*60}")

        self.history.setdefault("iter_time_sec", [])

        for ep in range(1, self.episodes + 1):
            ep_info = self._run_iteration(train=True, episode_num=ep)

            # Aggiorna buffer rolling
            self._reward_buf.push(ep_info["reward"])
            self._iou_buf.push(ep_info["iou"])

            self.history["rewards"].append(ep_info["reward"])
            self.history["rewards_smooth"].append(self._reward_buf.mean)
            self.history["iou"].append(ep_info["iou"])
            self.history["iou_smooth"].append(self._iou_buf.mean)
            self.history["epsilon"].append(self.agent.eps)
            self.history["iter_time_sec"].append(ep_info["time_sec"])

            self.agent.update_epsilon()
            self.agent.maybe_sync_target()

            if ep % eval_every == 0 or ep == 1:
                if self._full_eval:
                    # ── Valutazione su TUTTE le immagini ─────────────────────
                    stats = self._evaluate_full()
                    val_mean   = stats["mean"]
                    val_std    = stats["std"]
                    val_median = stats["median"]
                    n_eval     = stats["n"]
                    eval_time  = stats.get("time_sec", 0.0)
                else:
                    # Fallback: campione parziale (comportamento legacy)
                    val_mean   = self._validate_partial()
                    val_std    = 0.0
                    val_median = val_mean
                    n_eval     = -1
                    eval_time  = 0.0

                self.history["val_iou_mean"].append(val_mean)
                self.history["val_iou_std"].append(val_std)
                self.history["val_iou_median"].append(val_median)
                self.history["val_ep"].append(ep)

                n_str = f"n={n_eval}" if n_eval > 0 else "campione"
                print(
                    f"Ep {ep:04d}/{self.episodes} | "
                    f"tempo iter={ep_info['time_sec']:.1f}s | "
                    f"R={ep_info['reward']:+.3f} (μ{self._reward_buf.mean:+.3f}) | "
                    f"IoU={ep_info['iou']:.4f} (μ{self._iou_buf.mean:.4f}) | "
                    f"ε={self.agent.eps:.3f} | "
                    f"Val IoU μ={val_mean:.4f} σ={val_std:.4f} med={val_median:.4f} "
                    f"[{n_str}, {eval_time:.1f}s]"
                )
            else:
                print(
                    f"Ep {ep:04d}/{self.episodes} | "
                    f"tempo iter={ep_info['time_sec']:.1f}s | "
                    f"R={ep_info['reward']:+.3f} (μ{self._reward_buf.mean:+.3f}) | "
                    f"IoU={ep_info['iou']:.4f} (μ{self._iou_buf.mean:.4f}) | "
                    f"ε={self.agent.eps:.3f}"
                )

            if ep % eval_every == 0 or ep == 1:
                # Salva best checkpoint in base al val IoU
                if val_mean > self.best_val_iou:
                    self.best_val_iou = val_mean
                    self.agent.save("best_roi_finder_agent.pth")

                if self.on_eval_callback is not None:
                    self.on_eval_callback(ep, stats if self._full_eval else {
                        "mean": val_mean, "std": val_std,
                        "median": val_median, "n": n_eval,
                    })

        self.agent.save("final_roi_finder_agent.pth")
        return self.history

    def _validate_partial(self) -> float:
        """Fallback legacy: valuta su max 20 campioni dal val_loader."""
        ious = []
        for batch in self.val_loader:
            for i in range(min(4, batch["image"].size(0))):
                info = self._run_episode(batch["image"][i], batch["mask"][i], train=False)
                ious.append(info["iou"])
            if len(ious) >= 20:
                break
        return float(np.mean(ious)) if ious else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Trainer Fase 2: ROI Refiner
# ─────────────────────────────────────────────────────────────────────────────

class ROIRefinerTrainer:
    """
    Addestra ROIRefinementAgent a raffinare la maschera
    nel crop trovato da ROIFinderAgent.

    Modifiche rispetto alla versione originale:
    - Ogni iterazione usa TUTTE le immagini del train set.
    - Ad ogni eval si valutano TUTTE le immagini del val set.
    - Best checkpoint basato sul val Dice medio.
    """

    def __init__(
        self,
        finder_agent: ROIFinderAgent,
        finder_env: ROIFinderEnv,
        refiner_agent: ROIRefinementAgent,
        refiner_env: ROIRefinementEnv,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        cfg: dict,
        prob_method: str = "adaptive",
        on_eval_callback: Optional[Callable[[int, Dict], None]] = None,
    ):
        self.finder_agent  = finder_agent
        self.finder_env    = finder_env
        self.refiner_agent = refiner_agent
        self.refiner_env   = refiner_env
        self.train_loader  = train_loader
        self.val_loader    = val_loader
        self.device        = device
        self.cfg           = cfg
        self.prob_method   = prob_method
        self.on_eval_callback = on_eval_callback

        rl = cfg.get("rl", {})
        ds = cfg.get("dataset", {})
        self.episodes    = rl.get("refine_episodes", rl.get("episodes", 300))
        self.target_freq = rl.get("target_update_freq", 20)

        img_size_cfg = ds.get("image_size", 256)
        if isinstance(img_size_cfg, (list, tuple)):
            self.img_size = img_size_cfg[0]
        else:
            self.img_size = int(img_size_cfg)

        # Rolling buffers per smoothing
        smooth_w = rl.get("reward_smooth_window", 20)
        self._reward_buf = EpisodeRewardBuffer(window=smooth_w)
        self._dice_buf   = EpisodeRewardBuffer(window=smooth_w)

        self._full_eval = rl.get("eval_on_full_val", rl.get("eval_on_full_test", True))

        self.history: Dict[str, List] = {
            "rewards":          [],
            "rewards_smooth":   [],
            "dice":             [],
            "dice_smooth":      [],
            "epsilon":          [],
            # Val full-test stats
            "val_dice_bl_mean":   [],
            "val_dice_rl_mean":   [],
            "val_dice_rl_std":    [],
            "val_dice_rl_median": [],
            "val_iou_rl_mean":    [],
            "val_ep":             [],
        }
        self.best_val_dice = 0.0
        self._all_samples = collect_train_samples(train_loader)
        n_train = len(self._all_samples)
        n_val   = len(_collect_loader_samples(val_loader))
        print(f"  Campioni training (solo train): {n_train} | Val: {n_val}")

    def _get_crop_and_gt(
        self, image_t: torch.Tensor, mask_t: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        """
        Usa il finder per ottenere la ROI, poi estrae crop e gt ridimensionati.
        Ritorna: (crop_gray, prob_map, gt_crop, box)
        """
        from skimage.transform import resize as sk_resize

        img_gray = _tensor_to_gray(image_t)
        gt_mask  = _tensor_to_mask(mask_t)

        state = self.finder_env.reset(img_gray, gt_mask)
        done = False
        while not done:
            action = self.finder_agent.act(state, greedy=True)
            state, _, done, info = self.finder_env.step(action)

        box = self.finder_env.get_box()
        x1, y1, x2, y2 = [int(v) for v in box]

        crop_gray = img_gray[y1:y2, x1:x2]
        gt_crop   = gt_mask[y1:y2, x1:x2]

        if crop_gray.size == 0 or gt_crop.size == 0:
            crop_gray = img_gray
            gt_crop   = gt_mask
            box = [0, 0, self.img_size, self.img_size]

        crop_resized = sk_resize(crop_gray, (self.img_size, self.img_size),
                                 anti_aliasing=True, preserve_range=True).astype(np.float32)
        gt_resized   = (sk_resize(gt_crop, (self.img_size, self.img_size),
                                  anti_aliasing=False, preserve_range=True) > 0.5).astype(np.float32)

        if self.prob_method == "otsu":
            prob_map = make_prob_map_otsu(crop_resized)
        else:
            prob_map = make_prob_map_adaptive(crop_resized)

        return crop_resized, prob_map, gt_resized, box

    def _run_episode(self, image_t, mask_t, train: bool) -> Dict:
        crop_gray, prob_map, gt_crop, box = self._get_crop_and_gt(image_t, mask_t)

        baseline_mask = (prob_map > 0.5).astype(np.float32)
        dice_bl = dice_coefficient(baseline_mask, gt_crop)

        state = self.refiner_env.reset(crop_gray, prob_map, gt_crop)
        total_reward = 0.0

        while True:
            action = self.refiner_agent.act(state, greedy=not train)
            next_state, reward, done, info = self.refiner_env.step(action)

            if train:
                self.refiner_agent.remember(state, action, reward, next_state, done)
                self.refiner_agent.learn()

            total_reward += reward
            state = next_state
            if done:
                break

        refined_mask = self.refiner_env.get_refined_mask()
        dice_rl = dice_coefficient(refined_mask, gt_crop)

        return {
            "reward":  total_reward,
            "dice_bl": dice_bl,
            "dice_rl": dice_rl,
            "delta":   dice_rl - dice_bl,
            "box":     box,
        }

    def _evaluate_full(self) -> Dict:
        """Valuta su TUTTE le immagini del val set."""
        return _full_eval_refiner(
            finder_agent  = self.finder_agent,
            finder_env    = self.finder_env,
            refiner_agent = self.refiner_agent,
            refiner_env   = self.refiner_env,
            loader        = self.val_loader,
            img_size      = self.img_size,
            prob_method   = self.prob_method,
        )

    def _run_iteration(self, train: bool, episode_num: int = 0) -> Dict[str, float]:
        """Un'iterazione = un passaggio su TUTTE le immagini del train set."""
        if not self._all_samples:
            return {"reward": 0.0, "dice_bl": 0.0, "dice_rl": 0.0, "delta": 0.0, "time_sec": 0.0}

        indices = np.random.permutation(len(self._all_samples))
        rewards, dices_bl, dices_rl = [], [], []

        t0 = time.time()
        pbar = tqdm(
            indices,
            desc=f"  Iter {episode_num:04d}/{self.episodes}" if train else "  Iter (eval)",
            unit="img",
            leave=False,
            dynamic_ncols=True,
        )
        for idx in pbar:
            image_t, mask_t = self._all_samples[int(idx)]
            ep_info = self._run_episode(image_t, mask_t, train=train)
            rewards.append(ep_info["reward"])
            dices_bl.append(ep_info["dice_bl"])
            dices_rl.append(ep_info["dice_rl"])
            pbar.set_postfix({"Dice": f"{np.mean(dices_rl):.3f}"})
        pbar.close()
        elapsed = time.time() - t0

        mean_bl = float(np.mean(dices_bl))
        mean_rl = float(np.mean(dices_rl))
        return {
            "reward":   float(np.mean(rewards)),
            "dice_bl":  mean_bl,
            "dice_rl":  mean_rl,
            "delta":    mean_rl - mean_bl,
            "time_sec": elapsed,
        }

    def train(self, eval_every: int = 20) -> Dict:
        print(f"{'='*60}")
        print(f" [ROIRefiner] Training per {self.episodes} iterazioni")
        print(f" Ogni iterazione usa TUTTE le {len(self._all_samples)} immagini (train)")
        if self._full_eval:
            print(f" Validazione su TUTTO il val set ogni {eval_every} iterazioni")
        print(f"{'='*60}")

        self.history.setdefault("iter_time_sec", [])

        for ep in range(1, self.episodes + 1):
            ep_info = self._run_iteration(train=True, episode_num=ep)

            self._reward_buf.push(ep_info["reward"])
            self._dice_buf.push(ep_info["dice_rl"])

            self.history["rewards"].append(ep_info["reward"])
            self.history["rewards_smooth"].append(self._reward_buf.mean)
            self.history["dice"].append(ep_info["dice_rl"])
            self.history["dice_smooth"].append(self._dice_buf.mean)
            self.history["epsilon"].append(self.refiner_agent.eps)
            self.history["iter_time_sec"].append(ep_info["time_sec"])

            self.refiner_agent.update_epsilon()
            self.refiner_agent.maybe_sync_target()

            if ep % eval_every == 0 or ep == 1:
                if self._full_eval:
                    # ── Full eval su tutte le immagini ────────────────────────
                    stats = self._evaluate_full()
                    val_dice_bl_mean   = stats["dice_bl"]["mean"]
                    val_dice_rl_mean   = stats["dice_rl"]["mean"]
                    val_dice_rl_std    = stats["dice_rl"]["std"]
                    val_dice_rl_median = stats["dice_rl"]["median"]
                    val_iou_rl_mean    = stats["iou_rl"]["mean"]
                    n_eval             = stats["n"]
                else:
                    # Fallback legacy
                    val_info = self._validate_partial()
                    val_dice_bl_mean   = val_info["val_dice_bl"]
                    val_dice_rl_mean   = val_info["val_dice_rl"]
                    val_dice_rl_std    = 0.0
                    val_dice_rl_median = val_dice_rl_mean
                    val_iou_rl_mean    = 0.0
                    n_eval             = -1

                self.history["val_dice_bl_mean"].append(val_dice_bl_mean)
                self.history["val_dice_rl_mean"].append(val_dice_rl_mean)
                self.history["val_dice_rl_std"].append(val_dice_rl_std)
                self.history["val_dice_rl_median"].append(val_dice_rl_median)
                self.history["val_iou_rl_mean"].append(val_iou_rl_mean)
                self.history["val_ep"].append(ep)

                n_str = f"n={n_eval}" if n_eval > 0 else "campione"
                print(
                    f"Ep {ep:04d}/{self.episodes} | "
                    f"R={ep_info['reward']:+.3f} (μ{self._reward_buf.mean:+.3f}) | "
                    f"Dice: BL={ep_info['dice_bl']:.4f} RL={ep_info['dice_rl']:.4f} "
                    f"(μRL={self._dice_buf.mean:.4f} Δ{ep_info['delta']:+.4f}) | "
                    f"ε={self.refiner_agent.eps:.3f} | "
                    f"Val Dice BL={val_dice_bl_mean:.4f} RL={val_dice_rl_mean:.4f} "
                    f"σ={val_dice_rl_std:.4f} med={val_dice_rl_median:.4f} [{n_str}]"
                )

                # Best checkpoint basato sul val Dice medio
                if val_dice_rl_mean > self.best_val_dice:
                    self.best_val_dice = val_dice_rl_mean
                    self.refiner_agent.save("best_roi_refiner_agent.pth")

                if self.on_eval_callback is not None:
                    eval_stats = stats if self._full_eval else {
                        "dice_bl": {"mean": val_dice_bl_mean},
                        "dice_rl": {
                            "mean": val_dice_rl_mean,
                            "std": val_dice_rl_std,
                            "median": val_dice_rl_median,
                        },
                        "iou_rl": {"mean": val_iou_rl_mean},
                        "n": n_eval,
                    }
                    self.on_eval_callback(ep, eval_stats)
        else:
            print(
                f"Ep {ep:04d}/{self.episodes} | "
                f"tempo iter={ep_info['time_sec']:.1f}s | "
                f"R={ep_info['reward']:+.3f} (μ{self._reward_buf.mean:+.3f}) | "
                f"Dice: BL={ep_info['dice_bl']:.4f} RL={ep_info['dice_rl']:.4f} "
                f"(μRL={self._dice_buf.mean:.4f} Δ{ep_info['delta']:+.4f}) | "
                f"ε={self.refiner_agent.eps:.3f}"
            )

        self.refiner_agent.save("final_roi_refiner_agent.pth")
        return self.history

    def _validate_partial(self) -> Dict:
        """Fallback legacy: valuta su max 20 campioni."""
        dices_bl, dices_rl = [], []
        for batch in self.val_loader:
            for i in range(min(4, batch["image"].size(0))):
                info = self._run_episode(batch["image"][i], batch["mask"][i], train=False)
                dices_bl.append(info["dice_bl"])
                dices_rl.append(info["dice_rl"])
            if len(dices_rl) >= 20:
                break
        return {
            "val_dice_bl": float(np.mean(dices_bl)) if dices_bl else 0.0,
            "val_dice_rl": float(np.mean(dices_rl)) if dices_rl else 0.0,
        }
