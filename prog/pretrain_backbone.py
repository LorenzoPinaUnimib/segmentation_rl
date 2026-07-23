"""
Pretraining supervisionato del backbone ResNet18 (+ testa di pooling spaziale)
su un task di segmentazione binaria (tumore sì/no per pixel) + regressione
ausiliaria del bounding box, usando le stesse maschere già disponibili nel
dataset usato per il training RL in tutto.py.

PERCHÉ:
In tutto.py, sbloccare il backbone per il fine-tuning (--backbone-freeze
diverso da 'all') costringe a ricalcolare l'intero forward/backward del
ResNet18 ad ogni singolo training step RL (ImageReplayBuffer), il che è
molto lento. La via alternativa era tenere il backbone congelato con pesi
ImageNet puri (--backbone-freeze all, EmbeddingReplayBuffer, veloce) ma
senza nessun adattamento al dominio medico.

Questo script rompe il compromesso: adatta il backbone al dominio PRIMA
del training RL, con un task supervisionato standard (segmentazione).

NOVITÀ (allineamento con l'architettura aggiornata di tutto.py, item 4):
tutto.py non fa più solo global average pooling sul backbone: usa
SpatialAttentionPool (conv + attenzione spaziale, vedi tutto.py), un modulo
CON parametri allenabili che va anch'esso addestrato per essere utile. Il
problema: quando in tutto.py si usa --backbone-freeze all (fast-path a
embedding cache, necessario per allenamenti RL lunghi), spatial_pool non
riceve MAI gradiente durante l'RL, quindi se partisse da un'inizializzazione
casuale resterebbe casuale per sempre (peggio del semplice GAP che
sostituisce). Questo script risolve il problema PREADDESTRANDO anche
spatial_pool qui, con due segnali supervisionati allineati al suo scopo
("codificare dove si trova il tumore"):
  1. Il decoder di segmentazione (come prima) allena l'encoder/backbone.
  2. Una testa ausiliaria di regressione del bounding box, che parte
     ESATTAMENTE dall'embedding prodotto da spatial_pool (stesso modulo,
     stessi pesi, importato direttamente da tutto.py) e impara a predire
     [x, y, w, h] normalizzati del tumore. Questo allena spatial_pool a
     comprimere la feature map spaziale in un embedding che preserva
     l'informazione geometrica — esattamente il compito che gli viene
     chiesto di svolgere anche nella testa Q dell'agente RL.
Il bounding box target è calcolato dalla maschera con la STESSA convenzione
usata da BatchedActiveLocalizationEnv._load_sample_into_slot in tutto.py
(x, y, w, h = xmin, ymin, xmax-xmin, ymax-ymin; box di fallback centrato se
la maschera è vuota), per coerenza tra pretraining e RL.

Vengono salvati DUE file:
1. *_backbone.pt: Contiene i pesi di encoder + spatial_pool, da usare in
   tutto.py con --pretrained-backbone (anche con --backbone-freeze all: in tal
   caso tutto.py userà SpatialAttentionPool pre-addestrata invece di
   ripiegare su AdaptiveAvgPool2d, vedi PolicyNetwork.__init__ in tutto.py).
2. *_full.pt: Contiene il modello completo (encoder+spatial_pool+decoder+
   bbox_head), da usare qui per i test.

Uso tipico (Training):
    python pretrain_backbone.py --n-epochs 30 --batch-size 32 --output ./pretrained

Uso tipico (Test locale):
    python pretrain_backbone.py --test --model ./pretrained_full.pt

Uso tipico (poi, nel training RL):
    python tutto.py --backbone-freeze all --pretrained-backbone ./pretrained_backbone.pt ...
"""

import os
import sys
import argparse
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Importa DIRETTAMENTE da tutto.py il modulo di pooling spaziale (item 4) e le
# costanti rilevanti, invece di duplicarne la definizione: garantisce che
# l'architettura pre-addestrata qui sia BIT-A-BIT identica a quella usata da
# PolicyNetwork in tutto.py (stessa classe, stessi nomi di parametri), quindi
# il caricamento dei pesi con --pretrained-backbone funziona senza sorprese.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agente import SpatialAttentionPool, EMBED_DIM, WARP_SIZE  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# MODELLO: encoder ResNet18 + spatial_pool (identico a tutto.py) + decoder
# per segmentazione + testa ausiliaria di regressione del bounding box
# ─────────────────────────────────────────────────────────────────────────────
class ResNet18SegModel(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        backbone.fc = nn.Identity()
        self.backbone = backbone  # <-- struttura IDENTICA a PolicyNetwork.backbone

        # (4) Stesso modulo usato da PolicyNetwork in tutto.py: conv leggero +
        # attenzione spaziale che comprime la feature map 7x7x512 in un
        # embedding EMBED_DIM-dimensionale, invece del semplice GAP.
        self.spatial_pool = SpatialAttentionPool(in_channels=512, embed_dim=EMBED_DIM)

        # Testa ausiliaria: dall'embedding di spatial_pool regredisce
        # [x, y, w, h] normalizzati (sigmoid -> range [0,1]) del bounding box
        # del tumore. Serve SOLO per dare a spatial_pool un segnale di
        # apprendimento allineato al suo scopo geometrico; non viene usata in
        # tutto.py (che ha la sua testa Q) e quindi NON viene salvata nel file
        # *_backbone.pt, solo nel *_full.pt per i test locali.
        self.bbox_head = nn.Sequential(
            nn.Linear(EMBED_DIM, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4),
            nn.Sigmoid(),
        )

        # Decoder: 5 blocchi di upsampling x2 = x32 totale
        def up_block(in_ch, out_ch):
            return nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.decoder = nn.Sequential(
            up_block(512, 256),
            up_block(256, 128),
            up_block(128, 64),
            up_block(64, 32),
            up_block(32, 16),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward_backbone_features(self, x):
        b = self.backbone
        x = b.conv1(x)
        x = b.bn1(x)
        x = b.relu(x)
        x = b.maxpool(x)
        x = b.layer1(x)
        x = b.layer2(x)
        x = b.layer3(x)
        x = b.layer4(x)
        return x  # [B, 512, H/32, W/32]

    def forward_embedding(self, feat):
        """(4) Applica spatial_pool (identico a tutto.py) alla feature map
        spaziale per ottenere l'embedding EMBED_DIM-dimensionale."""
        return self.spatial_pool(feat)

    def forward(self, x):
        """Ritorna (logits di segmentazione, embedding, bbox predetto
        normalizzato [0,1]). Un'unica forward pass condivisa: il decoder di
        segmentazione lavora sulla feature map spaziale grezza, spatial_pool +
        bbox_head lavorano sull'embedding compresso — allenano insieme lo
        stesso encoder ma con due segnali complementari."""
        feat = self.forward_backbone_features(x)
        logits = self.decoder(feat)
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)

        embed = self.forward_embedding(feat)
        bbox_pred = self.bbox_head(embed)
        return logits, embed, bbox_pred


# ─────────────────────────────────────────────────────────────────────────────
# GROUND-TRUTH BBOX DALLA MASCHERA (stessa convenzione di tutto.py:
# BatchedActiveLocalizationEnv._load_sample_into_slot)
# ─────────────────────────────────────────────────────────────────────────────
def masks_to_bboxes_normalized(masks):
    """masks: [B, 1, H, W] (o [B, H, W]) binaria/soglia 0.5. Ritorna
    [B, 4] con [x, y, w, h] normalizzati in [0,1] (x,y = angolo in alto a
    sinistra, w,h = larghezza/altezza), stessa convenzione (senza +1) usata
    da tutto.py per calcolare gt_boxes dalla maschera. Se una maschera non
    ha pixel positivi usa lo stesso box di fallback centrato di tutto.py
    ([W/4, H/4, W/2, H/2])."""
    if masks.dim() == 4:
        m = masks.squeeze(1)
    else:
        m = masks
    m = (m > 0.5)
    B, H, W = m.shape
    device = m.device

    rows_any = m.any(dim=2)  # [B, H]
    cols_any = m.any(dim=1)  # [B, W]
    has_fg = rows_any.any(dim=1)  # [B]

    row_idx = torch.arange(H, device=device).unsqueeze(0).expand(B, H)
    col_idx = torch.arange(W, device=device).unsqueeze(0).expand(B, W)

    ymin = torch.where(rows_any, row_idx, torch.full_like(row_idx, H)).min(dim=1).values.float()
    ymax = torch.where(rows_any, row_idx, torch.full_like(row_idx, -1)).max(dim=1).values.float()
    xmin = torch.where(cols_any, col_idx, torch.full_like(col_idx, W)).min(dim=1).values.float()
    xmax = torch.where(cols_any, col_idx, torch.full_like(col_idx, -1)).max(dim=1).values.float()

    boxes = torch.stack([xmin, ymin, xmax - xmin, ymax - ymin], dim=1)  # [B, 4], stessa convenzione di tutto.py

    fallback = torch.tensor([W / 4.0, H / 4.0, W / 2.0, H / 2.0], device=device).unsqueeze(0).expand(B, 4)
    boxes = torch.where(has_fg.unsqueeze(1), boxes, fallback)

    norm = torch.tensor([W, H, W, H], device=device, dtype=torch.float32).unsqueeze(0)
    return boxes / norm, has_fg


# ─────────────────────────────────────────────────────────────────────────────
# LOSS E METRICHE
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    
    intersection = (preds * targets).sum()
    dice = (2. * intersection + eps) / (preds.sum() + targets.sum() + eps)
    
    union = (preds + targets).sum() - intersection
    iou = (intersection + eps) / (union + eps)
    
    return dice.item(), iou.item()

def combined_loss(logits, targets):
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum()
    dice = (2. * intersection + 1e-6) / (probs.sum() + targets.sum() + 1e-6)
    return bce + (1.0 - dice), bce.item(), (1.0 - dice).item()

@torch.no_grad()
def dice_score(logits, targets, eps=1e-6):
    probs = (torch.sigmoid(logits) > 0.5).float()
    probs = probs.flatten(1)
    targets = targets.flatten(1)
    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    return ((2 * intersection + eps) / (union + eps)).mean().item()


def bbox_loss_fn(bbox_pred, bbox_target):
    """(4) Smooth-L1 sui 4 valori normalizzati [x,y,w,h]. Segnale ausiliario
    che allena spatial_pool a codificare l'informazione geometrica (dove si
    trova il tumore) nell'embedding, esattamente ciò che serve alla testa Q
    di tutto.py."""
    return F.smooth_l1_loss(bbox_pred, bbox_target)


@torch.no_grad()
def bbox_iou_metric(bbox_pred, bbox_target, eps=1e-6):
    """IoU medio tra i box predetti e quelli target (entrambi normalizzati
    [x,y,w,h]), solo a scopo di monitoraggio in validazione."""
    px1, py1 = bbox_pred[:, 0], bbox_pred[:, 1]
    px2, py2 = px1 + bbox_pred[:, 2], py1 + bbox_pred[:, 3]
    tx1, ty1 = bbox_target[:, 0], bbox_target[:, 1]
    tx2, ty2 = tx1 + bbox_target[:, 2], ty1 + bbox_target[:, 3]

    ix1, iy1 = torch.max(px1, tx1), torch.max(py1, ty1)
    ix2, iy2 = torch.min(px2, tx2), torch.min(py2, ty2)
    inter = torch.clamp(ix2 - ix1, min=0) * torch.clamp(iy2 - iy1, min=0)
    area_p = torch.clamp(bbox_pred[:, 2], min=0) * torch.clamp(bbox_pred[:, 3], min=0)
    area_t = torch.clamp(bbox_target[:, 2], min=0) * torch.clamp(bbox_target[:, 3], min=0)
    union = area_p + area_t - inter
    return ((inter + eps) / (union + eps)).mean().item()


# ─────────────────────────────────────────────────────────────────────────────
# MODALITÀ TEST (AGGIORNATA CON PLOT E BOX)
# ─────────────────────────────────────────────────────────────────────────────
def run_test(args, device, test_ds):
    model = ResNet18SegModel(pretrained=False).to(device)

    print(f"[INFO] Caricamento pesi da: {args.model}")
    checkpoint = torch.load(args.model, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print("[INFO] Modello completo caricato con successo (Encoder + spatial_pool + Decoder + BBoxHead).")
    elif "backbone_state_dict" in checkpoint:
        print("[WARNING] Stai caricando il file _backbone.pt per il test!")
        print("[WARNING] Il decoder e la bbox_head avranno pesi casuali, le predizioni potrebbero essere errate.")
        print("[WARNING] Usa il file _full.pt per eseguire il test correttamente.")
        model.backbone.load_state_dict(checkpoint["backbone_state_dict"])
        if "spatial_pool_state_dict" in checkpoint:
            model.spatial_pool.load_state_dict(checkpoint["spatial_pool_state_dict"])
    else:
        raise ValueError("Il file specificato non contiene un checkpoint valido.")

    model.eval()

    os.makedirs("output_test", exist_ok=True)
    print(f"[INFO] Running inference su {len(test_ds)} immagini di test...")

    for i in range(len(test_ds)):
        item = test_ds[i]
        img = item["image"].unsqueeze(0).to(device)
        mask = item["mask"].unsqueeze(0).to(device)

        with torch.no_grad():
            logits, embed, bbox_pred = model(img)
            dice, iou = compute_metrics(logits, mask)
            bbox_target, _ = masks_to_bboxes_normalized(mask)
            bbox_iou = bbox_iou_metric(bbox_pred, bbox_target)

        pred = (torch.sigmoid(logits) > 0.5).float()

        # --- PREPARAZIONE DATI PER MATPLOTLIB ---
        # Convertiamo tensori [1, C, H, W] in numpy array [H, W, C] o [H, W]
        img_np = img.cpu().squeeze(0).permute(1, 2, 0).numpy() # (H, W, 3)
        img_np = np.clip(img_np, 0, 1) # Sicurezza
        H, W = img_np.shape[0], img_np.shape[1]

        mask_np = mask.cpu().squeeze(0).squeeze(0).numpy() # (H, W)
        pred_np = pred.cpu().squeeze(0).squeeze(0).numpy() # (H, W)

        # --- CREAZIONE PLOT ---
        fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=150)
        ax.imshow(img_np)

        # 1. Overlay maschera predetta (Giallo)
        # Creiamo un livello RGBA della stessa grandezza dell'immagine
        yellow_overlay = np.zeros((*pred_np.shape, 4)) 
        # Dove la predizione è 1, mettiamo R=1, G=1, B=0, Alpha=0.45 (giallo semi-trasparente)
        yellow_overlay[pred_np == 1] = [1, 1, 0, 0.45]
        ax.imshow(yellow_overlay)

        # 2. Bounding Box per la Ground Truth (Rossa, vuota al centro)
        rows = np.any(mask_np, axis=1)
        cols = np.any(mask_np, axis=0)
        # Controlliamo se esiste una maschera reale (potrebbe non esserci tumore)
        if np.any(rows) and np.any(cols):
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            width = xmax - xmin
            height = ymax - ymin

            # Disegniamo la box rossa (ground truth)
            rect = patches.Rectangle((xmin, ymin), width, height,
                                     linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)

        # 3. (4) Bounding Box predetto dalla bbox_head ausiliaria (Ciano),
        # de-normalizzato da [0,1] alle dimensioni reali dell'immagine.
        # Serve a verificare visivamente quanto bene spatial_pool ha imparato
        # a codificare la posizione/estensione del tumore nell'embedding.
        bp = bbox_pred.cpu().squeeze(0).numpy()
        px, py, pw, ph = bp[0] * W, bp[1] * H, bp[2] * W, bp[3] * H
        rect_pred = patches.Rectangle((px, py), pw, ph,
                                       linewidth=2, edgecolor='cyan', facecolor='none', linestyle='--')
        ax.add_patch(rect_pred)

        # Titolo con metriche e salvataggio
        ax.set_title(f"Test {i} | Dice: {dice:.2f} | IoU: {iou:.2f} | BBox IoU: {bbox_iou:.2f}", fontsize=11)
        ax.axis('off')

        plt.savefig(f"output_test/test_{i}_dice{dice:.2f}_iou{iou:.2f}.png", bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

    print("[INFO] Test completato. Immagini salvate in ./output_test")


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
def train(args, device, train_ds, val_ds):
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, drop_last=False)

    print(f"[INFO] Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    model = ResNet18SegModel(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs, eta_min=args.learning_rate * 0.05)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    base_name = os.path.splitext(args.output)[0]
    if base_name.endswith("_backbone") or base_name.endswith("_full"):
        base_name = base_name.rsplit('_', 1)[0] 
        
    out_dir = os.path.dirname(base_name) or "."
    os.makedirs(out_dir, exist_ok=True)
    
    best_path_backbone = f"{base_name}_backbone.pt"
    best_path_full = f"{base_name}_full.pt"
    
    best_val_dice = 0.0

    print("[INFO] Inizio pretraining supervisionato del backbone...")
    print("=" * 80)

    for epoch in range(args.n_epochs):
        model.train()
        train_losses, train_bces, train_dices_l, train_bbox_losses, train_bbox_ious = [], [], [], [], []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.n_epochs} [train]")
        for batch in pbar:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)

            logits, embed, bbox_pred = model(images)
            seg_loss, bce_v, dice_v = combined_loss(logits, masks)

            # (4) Bbox target dalla stessa maschera, stessa convenzione di tutto.py.
            bbox_target, _ = masks_to_bboxes_normalized(masks)
            b_loss = bbox_loss_fn(bbox_pred, bbox_target)

            loss = seg_loss + args.bbox_loss_weight * b_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            train_losses.append(loss.item())
            train_bces.append(bce_v)
            train_dices_l.append(dice_v)
            train_bbox_losses.append(b_loss.item())
            with torch.no_grad():
                train_bbox_ious.append(bbox_iou_metric(bbox_pred, bbox_target))
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "bbox_loss": f"{b_loss.item():.4f}"})

        scheduler.step()

        model.eval()
        val_dices, val_bbox_ious = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{args.n_epochs} [val]", leave=False):
                images = batch["image"].to(device, non_blocking=True)
                masks = batch["mask"].to(device, non_blocking=True)
                if masks.dim() == 3:
                    masks = masks.unsqueeze(1)
                logits, embed, bbox_pred = model(images)
                val_dices.append(dice_score(logits, masks))
                bbox_target, _ = masks_to_bboxes_normalized(masks)
                val_bbox_ious.append(bbox_iou_metric(bbox_pred, bbox_target))

        avg_train_loss = sum(train_losses) / max(len(train_losses), 1)
        avg_train_bbox_iou = sum(train_bbox_ious) / max(len(train_bbox_ious), 1)
        avg_val_dice = sum(val_dices) / max(len(val_dices), 1)
        avg_val_bbox_iou = sum(val_bbox_ious) / max(len(val_bbox_ious), 1)

        print(f"[Epoch {epoch + 1}] Train Loss: {avg_train_loss:.4f} | Val Dice: {avg_val_dice:.4f} | "
              f"Train BBox IoU: {avg_train_bbox_iou:.4f} | Val BBox IoU: {avg_val_bbox_iou:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")

        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice

            torch.save({
                "backbone_state_dict": model.backbone.state_dict(),
                # (4) Salva anche spatial_pool: tutto.py lo carica opzionalmente
                # (checkpoint.get("spatial_pool_state_dict")) e, se presente,
                # lo usa anche con --backbone-freeze all invece di ripiegare
                # su AdaptiveAvgPool2d (vedi PolicyNetwork.__init__).
                "spatial_pool_state_dict": model.spatial_pool.state_dict(),
                "epoch": epoch + 1,
                "val_dice": avg_val_dice,
                "val_bbox_iou": avg_val_bbox_iou,
                "train_loss": avg_train_loss,
                "args": args,
                "timestamp": timestamp,
            }, best_path_backbone)

            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "val_dice": avg_val_dice,
                "val_bbox_iou": avg_val_bbox_iou,
                "train_loss": avg_train_loss,
                "args": args,
                "timestamp": timestamp,
            }, best_path_full)

            print(f"  [✓] Nuovo miglior modello salvato (Val Dice: {best_val_dice:.4f}, "
                  f"Val BBox IoU: {avg_val_bbox_iou:.4f})")
            print(f"      - Per RL (tutto.py): {best_path_backbone}")
            print(f"      - Per Test (qui):    {best_path_full}")

        print("-" * 80)

    print("=" * 80)
    print(f"[INFO] Pretraining completato. Miglior Val Dice: {best_val_dice:.4f}")
    print(f"[INFO] Per usarli nel training RL:")
    print(f"  python tutto.py --backbone-freeze all --pretrained-backbone {best_path_backbone} ...")
    print(f"[INFO] Per testare visivamente la bontà della segmentazione:")
    print(f"  python pretrain_backbone.py --test --model {best_path_full}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pretraining supervisionato del backbone ResNet18 (segmentazione) "
                     "per poi riusarlo congelato nel training RL di tutto.py."
    )
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--model", type=str, help="Path al file _full.pt per il test")
    parser.add_argument("--n-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--bbox-loss-weight", type=float, default=1.0,
                       help="(4) Peso della loss ausiliaria di regressione del bounding box "
                            "(loss totale = seg_loss + bbox_loss_weight * bbox_loss). Allena "
                            "spatial_pool a codificare informazione geometrica nell'embedding "
                            "(default: 1.0).")
    parser.add_argument("--output", type=str, default="./backbone_pretrained.pt")
    parser.add_argument("--dataset-source", type=str, default=os.environ.get("DATASET_SOURCE", "kaggle"))
    parser.add_argument("--dataset-path", type=str, default=os.environ.get("DATASET_PATH", None))
    parser.add_argument("--kaggle-id", type=str, default=os.environ.get(
        "KAGGLE_DATASET_ID", "pkdarabi/brain-tumor-image-dataset-semantic-segmentation"
    ))
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        try:
            import torch_directml
            device = torch_directml.device()
            print(f"[INFO] DirectML Device: {device}")
        except ImportError:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[INFO] Device: {device}")
    
    cfg = {
        "dataset": {
            "source": args.dataset_source,
            "kaggle_id": args.kaggle_id,
            "local_path": args.dataset_path,
            "image_size": [224, 224],
            "train_ratio": (1501 / 2145),
            "val_ratio": (429 / 2145),
        },
        "preprocessing": {
            "binarize_mask": True,
            "mask_threshold": 0.5,
            "normalization": "per_image",
            "white_balance": False,
            "clahe": False,
            "denoise": False,
        },
        "training": {"batch_size": args.batch_size, "num_workers": 0}
    }

    from dataset import get_datasets
    print("[INFO] Caricamento dataset...")
    train_ds, val_ds, test_ds = get_datasets(cfg)

    if args.test:
        if not args.model:
            raise ValueError("Devi specificare --model per la modalità test (usa il file _full.pt)")
        run_test(args, device, test_ds)
    else:
        train(args, device, train_ds, val_ds)