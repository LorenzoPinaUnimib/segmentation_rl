"""
localizer.py
────────────
Piccolo regressore CNN supervisionato: predice una bounding box
approssimativa [cx, cy, w, h] direttamente dall'immagine, usata come PUNTO
DI PARTENZA per l'agente RL (invece di un box completamente casuale, o di
un box costruito attorno alla ground truth che a test time su immagini
nuove non e' disponibile).

Non e' data leakage: si allena su image->box del train/val set (le box GT
derivate dalle maschere, che il dataset fornisce comunque per calcolare il
reward), non tocca il test set. A INFERENZA (sia durante il training RL sia
nella valutazione finale) usa solo l'immagine — esattamente quello che
succederebbe in produzione su un'immagine nuova senza maschera.

Uso tipico (vedi anche train.py):
    from localizer import train_localizer, load_localizer, make_localizer_fn
    train_localizer(train_ds, val_ds, save_path="./ppo_brain_tumor_logs/localizer.pt")
    model, device = load_localizer("./ppo_brain_tumor_logs/localizer.pt")
    localizer_fn = make_localizer_fn(model, device)
    env = BrainTumorRL_Env(..., localizer_fn=localizer_fn)
"""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def mask_to_cxcywh(mask_hw: np.ndarray, W: int, H: int) -> np.ndarray:
    """Da maschera binaria [H,W] a box [cx,cy,w,h] normalizzato in [0,1].
    Se la maschera e' vuota (non dovrebbe succedere su questo dataset, ma
    per robustezza) ritorna un box neutro al centro."""
    pos = np.where(mask_hw > 0.5)
    if len(pos[0]) == 0:
        return np.array([0.5, 0.5, 0.3, 0.3], dtype=np.float32)
    ymin, ymax = float(pos[0].min()), float(pos[0].max())
    xmin, xmax = float(pos[1].min()), float(pos[1].max())
    cx = (xmin + xmax) / 2.0 / W
    cy = (ymin + ymax) / 2.0 / H
    w = (xmax - xmin) / W
    h = (ymax - ymin) / H
    return np.array([cx, cy, max(w, 1e-3), max(h, 1e-3)], dtype=np.float32)


class BoxRegressorCNN(nn.Module):
    """Backbone piccolo e veloce di proposito: non deve essere accurato al
    pixel, deve solo dare all'RL un punto di partenza molto migliore di uno
    casuale (rifinire da IoU~0.5 e' un problema molto piu' semplice che
    localizzare da zero in 200 step)."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, stride=2, padding=2), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Linear(64, 4),
        )

    def forward(self, x):
        feat = self.features(x)
        return torch.sigmoid(self.head(feat))  # cx,cy,w,h in [0,1]


def _box_iou_loss(pred_cxcywh, target_cxcywh):
    """1 - IoU medio nel batch (box center-based normalizzati -> xyxy),
    differenziabile: guida direttamente la metrica che ci interessa, non solo
    la distanza L1 sulle coordinate."""
    def to_xyxy(b):
        cx, cy, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)

    p, t = to_xyxy(pred_cxcywh), to_xyxy(target_cxcywh)
    xi1, yi1 = torch.max(p[:, 0], t[:, 0]), torch.max(p[:, 1], t[:, 1])
    xi2, yi2 = torch.min(p[:, 2], t[:, 2]), torch.min(p[:, 3], t[:, 3])
    inter = (xi2 - xi1).clamp(min=0) * (yi2 - yi1).clamp(min=0)
    area_p = (p[:, 2] - p[:, 0]).clamp(min=0) * (p[:, 3] - p[:, 1]).clamp(min=0)
    area_t = (t[:, 2] - t[:, 0]).clamp(min=0) * (t[:, 3] - t[:, 1]).clamp(min=0)
    union = area_p + area_t - inter
    iou = inter / union.clamp(min=1e-6)
    return (1.0 - iou).mean()


def _collate_with_boxes(batch):
    images = torch.stack([b["image"] for b in batch])
    boxes = []
    for b in batch:
        mask = b["mask"].numpy().squeeze(0)
        H, W = mask.shape
        boxes.append(mask_to_cxcywh(mask, W, H))
    return images, torch.from_numpy(np.stack(boxes))


def train_localizer(train_ds, val_ds, save_path, epochs=15, batch_size=32,
                     lr=1e-3, device=None, verbose=1):
    """Allena il regressore su train_ds/val_ds (usa le maschere GIA' presenti
    nel dataset per derivare le box target -- nessun dato extra necessario).
    Salva solo il checkpoint con la miglior IoU di validazione."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    in_channels = train_ds[0]["image"].shape[0]
    model = BoxRegressorCNN(in_channels=in_channels).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               collate_fn=_collate_with_boxes, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=_collate_with_boxes, num_workers=0)

    best_val_iou = -1.0
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        for images, boxes in train_loader:
            images, boxes = images.to(device).float(), boxes.to(device).float()
            if images.max() > 1.0:
                images = images / 255.0
            pred = model(images)
            loss = F.smooth_l1_loss(pred, boxes) + _box_iou_loss(pred, boxes)
            opt.zero_grad(); loss.backward(); opt.step()
            train_loss_sum += loss.item() * images.size(0)
        train_loss = train_loss_sum / max(1, len(train_ds))

        model.eval()
        val_ious = []
        with torch.no_grad():
            for images, boxes in val_loader:
                images, boxes = images.to(device).float(), boxes.to(device).float()
                if images.max() > 1.0:
                    images = images / 255.0
                pred = model(images)
                val_ious.append(float((1.0 - _box_iou_loss(pred, boxes)).item()))
        val_iou = float(np.mean(val_ious)) if val_ious else 0.0

        if verbose:
            print(f"[localizer] epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  val_iou={val_iou:.4f}")

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save({"model_state": model.state_dict(), "in_channels": in_channels}, save_path)

    if verbose:
        print(f"[localizer] migliore val_iou={best_val_iou:.4f} -> salvato in {save_path}")
    return save_path


def load_localizer(path, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device)
    model = BoxRegressorCNN(in_channels=ckpt["in_channels"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, device


def make_localizer_fn(model, device):
    """Ritorna una funzione image[C,H,W] (uint8 o float) -> (cx,cy,w,h) in
    PIXEL, nel formato che BrainTumorRL_Env(localizer_fn=...) si aspetta."""
    @torch.no_grad()
    def _predict(image_chw: np.ndarray):
        _, H, W = image_chw.shape
        img_f = image_chw.astype(np.float32)
        if img_f.max() > 1.0:
            img_f = img_f / 255.0
        x = torch.from_numpy(img_f).unsqueeze(0).float().to(device)
        cx, cy, w, h = model(x).squeeze(0).cpu().numpy()
        return float(cx * W), float(cy * H), float(max(w * W, 12)), float(max(h * H, 12))
    return _predict
