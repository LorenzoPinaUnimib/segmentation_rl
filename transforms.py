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


def robust_intensity_clip(img: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.0) -> np.ndarray:
    """
    Sezione 3 del prompt: "eventuale equalizzazione o clipping robusto delle
    intensita'". Clippa i percentili estremi PRIMA della normalizzazione, cosi'
    pochi pixel anomali (rumore da acquisizione, artefatti, bordi saturi) non
    schiacciano il range dinamico del resto dell'immagine quando poi si fa
    min-max o per_image normalize.

    IMPORTANTE - cosa NON fare: non usare un clipping troppo aggressivo
    (es. 5-95 percentile) su immagini con tumori molto piccoli e molto chiari/
    scuri, perche' un tumore che occupa il 5% dell'immagine PUO' finire dentro
    la coda che stiamo tagliando, cancellando esattamente il segnale che
    vogliamo preservare. 1-99 percentile e' un compromesso sicuro: taglia solo
    outlier estremi, non la distribuzione del tumore stesso.
    """
    img = img.astype(np.float32)
    lo = np.percentile(img, low_pct)
    hi = np.percentile(img, high_pct)
    if hi - lo > 1e-6:
        img = np.clip(img, lo, hi)
    return img


def compute_brain_mask(gray_img: np.ndarray) -> np.ndarray:
    """
    Skull-stripping approssimato (sezione 3/4 del prompt) per immagini MRI 2D
    gia' estratte da un dataset pubblico (quindi senza header/volume 3D per uno
    skull-stripping "vero" tipo BET/HD-BET). Usa una pipeline classica e
    conservativa:
      1. Otsu threshold per separare testa da sfondo nero.
      2. Chiusura morfologica per riempire piccoli buchi (es. tumori scuri
         o cavita' che altrimenti verrebbero erosi via).
      3. Tiene solo la componente connessa piu' grande (il cranio/cervello),
         scartando artefatti isolati o rumore ai bordi.

    ATTENZIONE (sezione 3 "cosa non devi fare"): questo NON e' uno skull
    stripping anatomicamente accurato (non rimuove il cranio dal cervello,
    separa solo testa da sfondo). Usarlo come CANALE DI CONTESTO aggiuntivo
    per la CNN (sezione 4), MAI come crop distruttivo dell'immagine: un crop
    troppo aggressivo rischia di tagliare via tumori vicini al bordo del
    cranio, che nel dataset possono occupare fino al 30% dell'immagine.

    Ritorna una maschera float32 [H, W] in {0, 1}.
    """
    if gray_img.ndim == 3:
        gray_img = cv2.cvtColor(gray_img.astype(np.uint8) if gray_img.dtype == np.uint8
                                 else (gray_img * 255).astype(np.uint8) if gray_img.max() <= 1.0
                                 else gray_img.astype(np.uint8),
                                 cv2.COLOR_RGB2GRAY)
        img_u8 = gray_img
    elif gray_img.dtype == np.uint8:
        img_u8 = gray_img
    elif gray_img.max() <= 1.0:
        # immagine normalizzata in [0,1]: riportarla in [0,255] prima di Otsu,
        # altrimenti il cast diretto a uint8 la azzera quasi ovunque.
        img_u8 = (np.clip(gray_img, 0.0, 1.0) * 255).astype(np.uint8)
    else:
        img_u8 = gray_img.astype(np.uint8)

    _, thresh = cv2.threshold(img_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if n_labels <= 1:
        # nessuna componente trovata (immagine quasi tutta nera): fallback a tutto 1
        return np.ones_like(img_u8, dtype=np.float32)

    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest_label).astype(np.float32)

def classify_tumor_polarity(gray_img: np.ndarray, mask: np.ndarray) -> str:
    """
    Sezione 9 del prompt: classifica il tumore come "bright" o "dark" rispetto
    al resto del cervello, confrontando l'intensita' media dentro la maschera
    GT con quella del background. Usato per lo stratified sampling e per le
    metriche separate per polarita' (sezione 8/13).
    """
    if gray_img.ndim == 3:
        gray = cv2.cvtColor(gray_img.astype(np.float32), cv2.COLOR_RGB2GRAY)
    else:
        gray = gray_img.astype(np.float32)
    fg = mask > 0.5
    if fg.sum() < 1:
        return "unknown"
    tumor_mean = float(np.mean(gray[fg]))
    background_mean = float(np.mean(gray[~fg])) if (~fg).sum() > 0 else float(np.mean(gray))
    return "bright" if tumor_mean >= background_mean else "dark"

def get_train_transform(cfg: dict, image_size: Tuple[int, int], has_brain_mask: bool = False) -> "A.Compose":
    aug = cfg.get("augmentation", {})
    size_h, size_w = image_size

    if not HAS_ALBUMENTATIONS:
        raise RuntimeError("albumentations required for augmentation.")

    transforms = [
        A.Resize(size_h, size_w),
    ]

    if aug.get("enabled", True):
        _extracted_from_get_train_transform_13(aug, transforms)
    transforms.append(ToTensorV2())
    additional_targets = {"brain_mask": "mask"} if has_brain_mask else None
    return A.Compose(transforms, additional_targets=additional_targets)


# TODO Rename this here and in `get_train_transform`
def _extracted_from_get_train_transform_13(aug, transforms):
    hflip_p = aug.get("horizontal_flip", 0.5)
    if hflip_p > 0:
        transforms.append(A.HorizontalFlip(p=hflip_p))
    # Il vertical flip e' DISATTIVATO di default (sezione 3: "quali
    # trasformazioni rischiano di peggiorare l'anatomia"). Un cervello in
    # una scansione assiale non ha simmetria alto/basso: capovolgerlo
    # produce un'immagine anatomicamente impossibile, a differenza del
    # flip orizzontale (sinistra/destra) che e' un'augmentation comune e
    # valida in radiologia. Va abilitato esplicitamente solo se il
    # dataset/protocollo lo giustifica.
    vflip_p = aug.get("vertical_flip", 0.0)
    if vflip_p > 0:
        transforms.append(A.VerticalFlip(p=vflip_p))
    # Limite di rotazione tenuto contenuto: rotazioni ampie (>15-20 gradi)
    # non sono realistiche per scansioni assiali gia' allineate e rischiano
    # di introdurre bordi neri interpolati che il modello puo' imparare a
    # sfruttare come scorciatoia spuria.
    rot = aug.get("rotate_limit", 10)
    if rot > 0:
        transforms.append(A.Rotate(limit=rot, p=0.5,
                                    border_mode=cv2.BORDER_REFLECT_101))
    # Brightness/contrast: e' la leva principale per la sezione 9
    # (robustezza al contrasto / tumori chiari e scuri). Valori piu' alti
    # di quelli di default (0.15) rischiano di trasformare un tumore
    # ipointenso in isointenso rispetto al background, cancellando il
    # segnale che l'agente deve imparare a riconoscere: 0.2-0.3 e' un
    # limite ragionevole da provare in ablation, non oltre.
    bl = aug.get("brightness_limit", 0.2)
    cl = aug.get("contrast_limit", 0.2)
    if bl > 0 or cl > 0:
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=bl, contrast_limit=cl, p=0.5
            )
        )
    if aug.get("gamma_limit", None):
        transforms.append(A.RandomGamma(gamma_limit=aug["gamma_limit"], p=0.3))
    nl = aug.get("noise_var_limit", [5.0, 20.0])
    #transforms.append(A.GaussNoise(p=0.3))


def get_val_transform(image_size: Tuple[int, int], has_brain_mask: bool = False) -> "A.Compose":
    size_h, size_w = image_size
    if not HAS_ALBUMENTATIONS:
        raise RuntimeError("albumentations required.")
    additional_targets = {"brain_mask": "mask"} if has_brain_mask else None
    return A.Compose([
        A.Resize(size_h, size_w),
        ToTensorV2(),
    ], additional_targets=additional_targets)


# ─────────────────────────────────────────────────────────────────────────────
# Fallback (no albumentations)
# ─────────────────────────────────────────────────────────────────────────────

class FallbackTransform:
    """
    Minimal numpy-only transform: resize + augmentation + to tensor.
    Usato solo se albumentations non e' installato.

    NOTA: prima questa classe ignorava completamente la config di
    augmentation passata da cfg["augmentation"] (probabilita' di flip
    hardcoded al 50%/30%) e applicava sempre un flip verticale, anatomicamente
    implausibile per una scansione assiale del cervello (sezione 3 del
    prompt). Ora rispetta la stessa config usata dalla pipeline
    albumentations, con lo stesso default (niente flip verticale a meno di
    abilitarlo esplicitamente).
    """
    def __init__(self, image_size: Tuple[int, int], augment: bool = False, aug_cfg: dict = None):
        self.h, self.w = image_size
        self.augment = augment
        self.aug_cfg = aug_cfg or {}

    def __call__(self, image: np.ndarray, mask: np.ndarray, brain_mask: np.ndarray = None) -> Dict:
        # resize
        image = cv2.resize(image, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        mask  = cv2.resize(mask,  (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        if brain_mask is not None:
            brain_mask = cv2.resize(brain_mask, (self.w, self.h), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            hflip_p = self.aug_cfg.get("horizontal_flip", 0.5)
            vflip_p = self.aug_cfg.get("vertical_flip", 0.0)  # off by default: vedi docstring
            if np.random.rand() < hflip_p:
                image = np.fliplr(image).copy()
                mask  = np.fliplr(mask).copy()
                if brain_mask is not None:
                    brain_mask = np.fliplr(brain_mask).copy()
            if vflip_p > 0 and np.random.rand() < vflip_p:
                image = np.flipud(image).copy()
                mask  = np.flipud(mask).copy()
                if brain_mask is not None:
                    brain_mask = np.flipud(brain_mask).copy()

            # brightness/contrast jitter, coerente con la pipeline albumentations
            # (sezione 9: robustezza al contrasto) anche quando albumentations
            # non e' disponibile.
            bl = self.aug_cfg.get("brightness_limit", 0.2)
            cl = self.aug_cfg.get("contrast_limit", 0.2)
            if (bl > 0 or cl > 0) and np.random.rand() < 0.5:
                brightness = 1.0 + np.random.uniform(-bl, bl)
                contrast = 1.0 + np.random.uniform(-cl, cl)
                mean = image.mean()
                image = np.clip((image - mean) * contrast + mean * brightness, 0.0, 1.0 if image.max() <= 1.0 else 255.0)

        # [H,W,C] → [C,H,W] tensor
        if image.ndim == 2:
            image = image[..., np.newaxis]
        image_t = torch.from_numpy(image.transpose(2, 0, 1)).float()

        if mask.ndim == 3:
            mask = mask[..., 0]
        mask_t = torch.from_numpy(mask).float()

        out = {"image": image_t, "mask": mask_t}
        if brain_mask is not None:
            out["brain_mask"] = torch.from_numpy(brain_mask).float()
        return out