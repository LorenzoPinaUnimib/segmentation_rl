# Import
import cv2
import json
import kagglehub
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import traceback

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import Dict, Tuple, Optional, List

class BrainTumorRefinementDataset(Dataset):
    # Carica le annotazioni e le salva internamente
    def __init__(self, dataset_path, split='train', crop_size=224):
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.crop_size = crop_size
        
        coco_json_path = self.dataset_path / f'{split}' / '_annotations.coco.json'
        
        if not coco_json_path.exists():
            raise FileNotFoundError(f"COCO JSON non trovato: {coco_json_path}")
        
        with open(coco_json_path, 'r') as f:
            self.coco_data = json.load(f)
        
        self.img_dir = self.dataset_path / split
        
        self.annotations_by_img_id = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations_by_img_id:
                self.annotations_by_img_id[img_id] = []
            self.annotations_by_img_id[img_id].append(ann)
        
        # Filtra solo immagini con annotazioni
        self.images = [
            img for img in self.coco_data['images'] 
            if img['id'] in self.annotations_by_img_id
        ]
        
        print(f"Dataset {split}: {len(self.images)} immagini caricato")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info['id']
        h, w = img_info['height'], img_info['width']
        
        img_path = self.img_dir / img_info['file_name']
        
        if not img_path.exists():
            return None
        
        # Carica immagine
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        
        # TODO: testare senza questa riga
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Recupero annotazione dell'immagine
        annotations = self.annotations_by_img_id.get(img_id, [])
        if len(annotations) == 0:
            return None
        
        # Creazione box ground truth
        ann = annotations[0]
        bbox = ann['bbox']
        x, y, bw, bh = bbox
        
        gt_x1 = int(x)
        gt_y1 = int(y)
        gt_x2 = int(x + bw)
        gt_y2 = int(y + bh)
        
        gt_box = np.array([gt_x1, gt_y1, gt_x2, gt_y2], dtype=np.float32)
        
        # Creazione box iniziale
        init_box = self._create_initial_box_from_center(h, w)
        
        result = self._extract_crop(img, init_box, gt_box)
        if result is None:
            return None
        crop, init_box_crop, gt_box_crop = result
        
        offsets = self._compute_offsets(init_box_crop, gt_box_crop)
        
        # Normalizza immagine
        crop = crop.astype(np.float32) / 255.0
        crop = np.transpose(crop, (2, 0, 1))  # HWC → CHW
        
        return {
            'crop': torch.tensor(crop),
            'offsets': torch.tensor(offsets),
        }
    
    # Crea il box iniziale attorno al centro dell'immagine
    def _create_initial_box_from_center(self, img_h, img_w):
        w_g = 50
        h_g = 50
        
        center_x = img_w / 2
        center_y = img_h / 2
        
        init_x1 = center_x - w_g / 2
        init_y1 = center_y - h_g / 2
        init_x2 = center_x + w_g / 2
        init_y2 = center_y + h_g / 2
        
        return np.array([init_x1, init_y1, init_x2, init_y2], dtype=np.float32)
    
    def _extract_crop(self, img, init_box, gt_box):
        h, w = img.shape[:2]
        x1, y1, x2, y2 = init_box
        
        # Calcolo centro e dimensioni crop
        box_cx = (x1 + x2) / 2
        box_cy = (y1 + y2) / 2
        
        half_size = self.crop_size // 2
        cx1 = int(box_cx - half_size)
        cy1 = int(box_cy - half_size)
        cx2 = int(box_cx + half_size)
        cy2 = int(box_cy + half_size)
        
        # Salva offset per trasformazione di coordinate
        crop_offset_x = cx1
        crop_offset_y = cy1
        
        # Evita di andare fuori immagine
        cx1 = max(0, cx1)
        cy1 = max(0, cy1)
        cx2 = min(w, cx2)
        cy2 = min(h, cy2)
        
        crop = img[cy1:cy2, cx1:cx2]
        
        # Scarto file troppo piccoli per essere processati
        if crop.shape[0] < self.crop_size // 2 or crop.shape[1] < self.crop_size // 2:
            return None
        
        # Calcola scale factor per il resize
        scale_x = self.crop_size / crop.shape[1]
        scale_y = self.crop_size / crop.shape[0]
        
        crop = cv2.resize(crop, (self.crop_size, self.crop_size))
        
        def transform_box(box, offset_x, offset_y, scale_x, scale_y):
            x1, y1, x2, y2 = box
            # Trasla rispetto al ritaglio
            x1 = (x1 - offset_x) * scale_x
            y1 = (y1 - offset_y) * scale_y
            x2 = (x2 - offset_x) * scale_x
            y2 = (y2 - offset_y) * scale_y
            # Clamp al crop ridimensionato
            x1 = np.clip(x1, 0, self.crop_size)
            y1 = np.clip(y1, 0, self.crop_size)
            x2 = np.clip(x2, 0, self.crop_size)
            y2 = np.clip(y2, 0, self.crop_size)
            return np.array([x1, y1, x2, y2], dtype=np.float32)
        
        init_box_crop = transform_box(init_box, crop_offset_x, crop_offset_y, scale_x, scale_y)
        gt_box_crop = transform_box(gt_box, crop_offset_x, crop_offset_y, scale_x, scale_y)
        
        return crop, init_box_crop, gt_box_crop
    
    # Calcola gli offset tra il centro dei due box
    def _compute_offsets(self, init_box, gt_box):
        x1_i, y1_i, x2_i, y2_i = init_box
        x1_g, y1_g, x2_g, y2_g = gt_box
        
        w_i = x2_i - x1_i
        h_i = y2_i - y1_i
        
        if w_i <= 0 or h_i <= 0:
            return np.array([0., 0., 0., 0.], dtype=np.float32)
        
        cx_i = (x1_i + x2_i) / 2
        cy_i = (y1_i + y2_i) / 2
        cx_g = (x1_g + x2_g) / 2
        cy_g = (y1_g + y2_g) / 2
        
        w_g = x2_g - x1_g
        h_g = y2_g - y1_g
        
        # Offsets normalizzati
        dx = (cx_g - cx_i) / w_i
        dy = (cy_g - cy_i) / h_i
        dw = np.log(w_g / w_i) if w_g > 0 else 0.
        dh = np.log(h_g / h_i) if h_g > 0 else 0.
        
        return np.array([dx, dy, dw, dh], dtype=np.float32)

# Collate function per ignorare i campioni None
def collate_fn(batch):
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

# Definizione modello
class BoxRefinementNet(nn.Module):
    # CNN leggera che predice 4 offsets per affinare il bounding box
    
    def __init__(self, input_channels=3, hidden_dim=256):
        super().__init__()
        
        # Backbone: ResNet18 semplificata o custom CNN
        self.backbone = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Block 1
            self._make_residual_block(64, 64, 2),
            
            # Block 2
            self._make_residual_block(64, 128, 2),
            
            # Block 3
            self._make_residual_block(128, 256, 2),
        )
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # FC head per predire gli offsets
        self.fc = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 4)  # Output: [dx, dy, dw, dh]
        )
    
    def _make_residual_block(self, in_channels, out_channels, num_blocks):
        layers = []
        for i in range(num_blocks):
            stride = 2 if i == 0 else 1
            in_ch = in_channels if i == 0 else out_channels
            layers.append(self._residual_block(in_ch, out_channels, stride))
        return nn.Sequential(*layers)
    
    def _residual_block(self, in_channels, out_channels, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# ============ FUNZIONE DI TRAINING ============
def train_refinement_model(
    dataset_path,
    epochs=100,
    batch_size=32,
    learning_rate=1e-3,
    device='cuda'
):
    """
    Addestra il modello di refinement
    """
    
    print("=" * 60)
    print("TRAINING: Box Refinement Network")
    print("=" * 60)
    
    # Dataset e DataLoader
    train_dataset = BrainTumorRefinementDataset(dataset_path, split='train')
    valid_dataset = BrainTumorRefinementDataset(dataset_path, split='valid')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    # Modello
    model = BoxRefinementNet(input_channels=3, hidden_dim=256)
    model = model.to(device)
    
    # Loss function: MSE per gli offsets
    criterion = nn.MSELoss()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        train_loss = 0.0
        model.train()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [TRAIN]")
        for batch in pbar:
            if batch is None:
                continue
            
            crops = batch['crop'].to(device)
            offsets_gt = batch['offsets'].to(device)
            
            optimizer.zero_grad()
            offsets_pred = model(crops)
            loss = criterion(offsets_pred, offsets_gt)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * crops.size(0)
            pbar.set_postfix({'loss': loss.item()})
        
        train_loss /= len(train_dataset)
        
        # Validation
        val_loss = 0.0
        model.eval()
        
        with torch.no_grad():
            pbar = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{epochs} [VALID]")
            for batch in pbar:
                if batch is None:
                    continue
                
                crops = batch['crop'].to(device)
                offsets_gt = batch['offsets'].to(device)
                
                offsets_pred = model(crops)
                loss = criterion(offsets_pred, offsets_gt)
                
                val_loss += loss.item() * crops.size(0)
                pbar.set_postfix({'loss': loss.item()})
        
        val_loss /= len(valid_dataset)
        
        print(f"Epoch {epoch+1}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
        
        # Salva best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'box_refinement_best.pt')
            print(f"  ✓ Miglior modello salvato (val_loss={best_loss:.6f})")
        
        scheduler.step(val_loss)
    
    return model

# ============ FUNZIONE DI INFERENCE ============
def refine_box(model, img, init_box, crop_size=224, device='cuda'):
    # Dato un'immagine e un box iniziale, ritorna il box affinato
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    x1, y1, x2, y2 = init_box
    box_w = x2 - x1
    box_h = y2 - y1
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    
    # Estrai crop
    half_size = crop_size // 2
    cx1 = int(cx - half_size)
    cy1 = int(cy - half_size)
    cx2 = int(cx + half_size)
    cy2 = int(cy + half_size)
    
    cx1 = max(0, cx1)
    cy1 = max(0, cy1)
    cx2 = min(w, cx2)
    cy2 = min(h, cy2)
    
    crop = img[cy1:cy2, cx1:cx2]
    crop = cv2.resize(crop, (crop_size, crop_size))
    crop = crop.astype(np.float32) / 255.0
    crop = np.transpose(crop, (2, 0, 1))  # HWC → CHW
    
    # Predici offsets
    model.eval()
    with torch.no_grad():
        crop_tensor = torch.tensor(crop).unsqueeze(0).to(device)
        offsets = model(crop_tensor).cpu().numpy()[0]
    
    dx, dy, dw, dh = offsets
    
    # Applica offsets al box iniziale
    refined_cx = cx + dx * box_w
    refined_cy = cy + dy * box_h
    refined_w = box_w * np.exp(dw)
    refined_h = box_h * np.exp(dh)
    
    refined_x1 = refined_cx - refined_w / 2
    refined_y1 = refined_cy - refined_h / 2
    refined_x2 = refined_cx + refined_w / 2
    refined_y2 = refined_cy + refined_h / 2
    
    # Clamp to image bounds
    refined_x1 = max(0, refined_x1)
    refined_y1 = max(0, refined_y1)
    refined_x2 = min(w, refined_x2)
    refined_y2 = min(h, refined_y2)
    
    return np.array([refined_x1, refined_y1, refined_x2, refined_y2])

# Configurazione logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BrainTumorRefinementDataset(Dataset):
    """Dataset per il refinement dei bounding box su immagini di tumori cerebrali."""
    
    def __init__(self, dataset_path: str, split: str = 'train', crop_size: int = 224):
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.crop_size = crop_size
        
        coco_json_path = self.dataset_path / split / '_annotations.coco.json'
        
        if not coco_json_path.exists():
            raise FileNotFoundError(f"COCO JSON non trovato: {coco_json_path}")
        
        with open(coco_json_path, 'r') as f:
            self.coco_data = json.load(f)
        
        self.img_dir = self.dataset_path / split
        
        # Crea mapping immagine_id -> annotazioni
        self.annotations_by_img_id = self._build_annotations_map()
        
        # Filtra solo immagini con annotazioni
        self.images = [
            img for img in self.coco_data['images']
            if img['id'] in self.annotations_by_img_id
        ]
        
        logger.info(f"Dataset {split}: {len(self.images)} immagini caricate")
    
    def _build_annotations_map(self) -> Dict:
        """Costruisce il mapping immagine_id -> lista di annotazioni."""
        annotations_map = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in annotations_map:
                annotations_map[img_id] = []
            annotations_map[img_id].append(ann)
        return annotations_map
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Optional[Dict]:
        img_info = self.images[idx]
        img_id = img_info['id']
        h, w = img_info['height'], img_info['width']
        
        img_path = self.img_dir / img_info['file_name']
        
        if not img_path.exists():
            return None
        
        # Carica immagine
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        annotations = self.annotations_by_img_id.get(img_id, [])
        if len(annotations) == 0:
            return None
        
        ann = annotations[0]
        bbox = ann['bbox']
        x, y, bw, bh = bbox
        
        gt_box = np.array([x, y, x + bw, y + bh], dtype=np.float32)
        init_box = self._create_initial_box(h, w)
        
        result = self._extract_crop(img, init_box, gt_box)
        if result is None:
            return None
        
        crop, init_box_crop, gt_box_crop = result
        offsets = self._compute_offsets(init_box_crop, gt_box_crop)
        
        # Normalizza immagine
        crop = crop.astype(np.float32) / 255.0
        crop = np.transpose(crop, (2, 0, 1))
        
        return {
            'crop': torch.tensor(crop, dtype=torch.float32),
            'offsets': torch.tensor(offsets, dtype=torch.float32),
            'image_path': str(img_path),
            'original_image': img,
            'init_box': init_box,
            'gt_box': gt_box
        }
    
    def _create_initial_box(self, img_h: int, img_w: int) -> np.ndarray:
        """Crea un box iniziale nel centro dell'immagine."""
        box_w, box_h = 50, 50
        center_x, center_y = img_w / 2, img_h / 2
        
        return np.array([
            center_x - box_w / 2,
            center_y - box_h / 2,
            center_x + box_w / 2,
            center_y + box_h / 2
        ], dtype=np.float32)
    
    def _extract_crop(self, img: np.ndarray, init_box: np.ndarray, gt_box: np.ndarray) -> Optional[Tuple]:
        """Estrae un crop intorno al box iniziale e trasforma i box."""
        h, w = img.shape[:2]
        x1, y1, x2, y2 = init_box
        
        box_cx = (x1 + x2) / 2
        box_cy = (y1 + y2) / 2
        
        half_size = self.crop_size // 2
        cx1 = max(0, int(box_cx - half_size))
        cy1 = max(0, int(box_cy - half_size))
        cx2 = min(w, int(box_cx + half_size))
        cy2 = min(h, int(box_cy + half_size))
        
        crop = img[cy1:cy2, cx1:cx2]
        
        if crop.shape[0] < self.crop_size // 2 or crop.shape[1] < self.crop_size // 2:
            return None
        
        scale_x = self.crop_size / crop.shape[1]
        scale_y = self.crop_size / crop.shape[0]
        
        crop = cv2.resize(crop, (self.crop_size, self.crop_size))
        
        init_box_crop = self._transform_box(init_box, cx1, cy1, scale_x, scale_y)
        gt_box_crop = self._transform_box(gt_box, cx1, cy1, scale_x, scale_y)
        
        return crop, init_box_crop, gt_box_crop
    
    @staticmethod
    def _transform_box(box: np.ndarray, offset_x: float, offset_y: float,
                       scale_x: float, scale_y: float, crop_size: int = 224) -> np.ndarray:
        """Trasforma le coordinate di un box rispetto al crop."""
        x1, y1, x2, y2 = box
        x1 = (x1 - offset_x) * scale_x
        y1 = (y1 - offset_y) * scale_y
        x2 = (x2 - offset_x) * scale_x
        y2 = (y2 - offset_y) * scale_y
        
        return np.array([
            np.clip(x1, 0, crop_size),
            np.clip(y1, 0, crop_size),
            np.clip(x2, 0, crop_size),
            np.clip(y2, 0, crop_size)
        ], dtype=np.float32)
    
    @staticmethod
    def _compute_offsets(init_box: np.ndarray, gt_box: np.ndarray) -> np.ndarray:
        """Calcola gli offsets normalizzati tra il box iniziale e quello ground truth."""
        x1_i, y1_i, x2_i, y2_i = init_box
        x1_g, y1_g, x2_g, y2_g = gt_box
        
        w_i = x2_i - x1_i
        h_i = y2_i - y1_i
        
        if w_i <= 0 or h_i <= 0:
            return np.array([0., 0., 0., 0.], dtype=np.float32)
        
        cx_i = (x1_i + x2_i) / 2
        cy_i = (y1_i + y2_i) / 2
        cx_g = (x1_g + x2_g) / 2
        cy_g = (y1_g + y2_g) / 2
        
        w_g = x2_g - x1_g
        h_g = y2_g - y1_g
        
        dx = (cx_g - cx_i) / w_i
        dy = (cy_g - cy_i) / h_i
        dw = np.log(w_g / w_i) if w_g > 0 else 0.
        dh = np.log(h_g / h_i) if h_g > 0 else 0.
        
        return np.array([dx, dy, dw, dh], dtype=np.float32)


def collate_fn(batch: List) -> Optional[Dict]:
    """Collate function che scarta i campioni None."""
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


class BoxRefinementNet(nn.Module):
    """CNN per il refinement dei bounding box."""
    
    def __init__(self, input_channels: int = 3, hidden_dim: int = 256):
        super().__init__()
        
        self.backbone = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            self._make_residual_block(64, 64, 2),
            self._make_residual_block(64, 128, 2),
            self._make_residual_block(128, 256, 2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc_head = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 4)
        )
    
    def _make_residual_block(self, in_channels: int, out_channels: int, num_blocks: int) -> nn.Sequential:
        """Crea un blocco residuale."""
        layers = []
        for i in range(num_blocks):
            stride = 2 if i == 0 else 1
            in_ch = in_channels if i == 0 else out_channels
            layers.append(self._residual_block(in_ch, out_channels, stride))
        return nn.Sequential(*layers)
    
    @staticmethod
    def _residual_block(in_channels: int, out_channels: int, stride: int = 1) -> nn.Sequential:
        """Crea un singolo blocco residuale."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_head(x)
        return x


def calculate_iou(box1: Tuple, box2: Tuple) -> float:
    """Calcola l'Intersection over Union tra due box."""
    x1_gt, y1_gt, x2_gt, y2_gt = box1
    x1_pred, y1_pred, x2_pred, y2_pred = box2
    
    x_left = max(x1_gt, x1_pred)
    y_top = max(y1_gt, y1_pred)
    x_right = min(x2_gt, x2_pred)
    y_bottom = min(y2_gt, y2_pred)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)
    box2_area = (x2_pred - x1_pred) * (y2_pred - y1_pred)
    union = box1_area + box2_area - intersection
    
    if union == 0:
        return 0.0
    
    return float(np.clip(intersection / union, 0, 1))


def apply_offsets_to_box(init_box: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Applica gli offsets predetti al box iniziale."""
    x1, y1, x2, y2 = init_box
    box_w = x2 - x1
    box_h = y2 - y1
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    
    dx, dy, dw, dh = offsets
    
    refined_cx = cx + dx * box_w
    refined_cy = cy + dy * box_h
    refined_w = box_w * np.exp(dw)
    refined_h = box_h * np.exp(dh)
    
    refined_x1 = refined_cx - refined_w / 2
    refined_y1 = refined_cy - refined_h / 2
    refined_x2 = refined_cx + refined_w / 2
    refined_y2 = refined_cy + refined_h / 2
    
    return np.array([refined_x1, refined_y1, refined_x2, refined_y2], dtype=np.float32)


def draw_boxes_with_labels(img: np.ndarray, gt_box: np.ndarray, pred_box: np.ndarray,
                          iou: float) -> np.ndarray:
    """Disegna i box con etichette e IoU sull'immagine."""
    img_with_boxes = img.copy()
    
    # Box GT (verde)
    x1_gt, y1_gt, x2_gt, y2_gt = gt_box.astype(int)
    cv2.rectangle(img_with_boxes, (x1_gt, y1_gt), (x2_gt, y2_gt),
                color=(0, 255, 0), thickness=2)
    cv2.putText(img_with_boxes, 'GT', (x1_gt + 5, y1_gt - 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Box predetto (rosso)
    x1_pred, y1_pred, x2_pred, y2_pred = pred_box.astype(int)
    cv2.rectangle(img_with_boxes, (x1_pred, y1_pred), (x2_pred, y2_pred),
                color=(255, 0, 0), thickness=2)
    cv2.putText(img_with_boxes, 'Pred', (x1_pred + 5, y1_pred - 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # IoU in alto a sinistra
    iou_text = f'IoU: {iou:.4f}'
    cv2.putText(img_with_boxes, iou_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # Legenda in basso
    cv2.putText(img_with_boxes, 'Green=GT | Red=Pred', (10, img_with_boxes.shape[0] - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return img_with_boxes


def save_predictions_per_epoch(model: nn.Module, dataset: Dataset, epoch: int,
                               device: str, output_dir: str = './epoch_predictions') -> Tuple[str, Dict]:
    """
    Salva immagini originali con box GT e predetti per ogni sample nel validation set.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    epoch_dir = output_dir / f'epoch_{epoch}'
    epoch_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    all_iou_values = []
    saved_count = 0
    
    with torch.no_grad():
        pbar = tqdm(dataset, desc=f"Salvando predizioni Epoch {epoch}", leave=False)
        
        for idx, sample in enumerate(pbar):
            try:
                if sample is None or 'crop' not in sample:
                    continue
                
                crop = sample['crop'].unsqueeze(0).to(device)
                original_image = sample['original_image']
                image_path = sample['image_path']
                init_box = sample['init_box']
                gt_box = sample['gt_box']
                
                offset_pred = model(crop).cpu().numpy()[0]
                
                del crop
                torch.cuda.empty_cache()
                
                pred_box = apply_offsets_to_box(init_box, offset_pred)
                
                h, w = original_image.shape[:2]
                gt_box = np.clip(gt_box, 0, [w, h, w, h])
                pred_box = np.clip(pred_box, 0, [w, h, w, h])
                
                # Calcola IoU
                iou = calculate_iou(
                    tuple(gt_box.astype(int)),
                    tuple(pred_box.astype(int))
                )
                
                if not np.isnan(iou):
                    all_iou_values.append(iou)
                
                img_with_boxes = draw_boxes_with_labels(original_image, gt_box, pred_box, iou)
                
                base_name = Path(image_path).stem
                output_path = epoch_dir / f'{base_name}_pred.png'
                
                img_bgr = cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_path), img_bgr)
                saved_count += 1
                
                del img_with_boxes, img_bgr, original_image
                
                pbar.set_postfix({'saved': saved_count, 'iou': f'{iou:.4f}'})
                
            except Exception as e:
                logger.error(f"Errore nel processare sample {idx}: {e}")
                continue
    
    # Statistiche
    stats = {
        'iou_values': all_iou_values,
        'mean_iou': float(np.mean(all_iou_values)) if all_iou_values else 0.0,
        'median_iou': float(np.median(all_iou_values)) if all_iou_values else 0.0,
        'std_iou': float(np.std(all_iou_values)) if all_iou_values else 0.0,
        'min_iou': float(np.min(all_iou_values)) if all_iou_values else 0.0,
        'max_iou': float(np.max(all_iou_values)) if all_iou_values else 0.0,
        'saved_count': saved_count,
        'total_count': len(dataset)
    }
    
    return str(epoch_dir), stats


def get_device() -> str:
    """Seleziona il device disponibile: CUDA > MPS > CPU"""
    if torch.cuda.is_available():
        device = 'cuda'
        logger.info(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = 'mps'
        logger.info("Device: MPS (Metal Performance Shaders)")
    else:
        device = 'cpu'
        logger.info("Device: CPU")
    
    return device


def train_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module,
                optimizer: torch.optim.Optimizer, device: str) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch in pbar:
        if batch is None or batch['crop'] is None or batch['crop'].shape[0] == 0:
            continue
        
        crops = batch['crop'].to(device)
        offsets_gt = batch['offsets'].to(device)
        
        optimizer.zero_grad()
        offsets_pred = model(crops)
        loss = criterion(offsets_pred, offsets_gt)
        loss.backward()
        optimizer.step()
        
        batch_size = crops.size(0)
        loss_value = loss.item()
        
        del crops, offsets_gt, offsets_pred, loss, batch
        
        total_loss += loss_value * batch_size
        total_samples += batch_size
        pbar.set_postfix({'loss': f'{loss_value:.6f}'})
    
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    return total_loss / max(total_samples, 1)


def validate_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module,
                   device: str) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation", leave=False)
        for batch in pbar:
            if batch is None or batch['crop'] is None or batch['crop'].shape[0] == 0:
                continue
            
            crops = batch['crop'].to(device)
            offsets_gt = batch['offsets'].to(device)
            
            offsets_pred = model(crops)
            loss = criterion(offsets_pred, offsets_gt)
            
            batch_size = crops.size(0)
            loss_value = loss.item()
            
            del crops, offsets_gt, offsets_pred, loss, batch
            
            total_loss += loss_value * batch_size
            total_samples += batch_size
            pbar.set_postfix({'loss': f'{loss_value:.6f}'})
    
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    return total_loss / max(total_samples, 1)


def log_epoch_stats(epoch: int, total_epochs: int, train_loss: float, val_loss: float,
                    current_lr: float, best_loss: float, improved: bool, mean_iou: float = None):
    """Registra le statistiche dell'epoca."""
    status = "BEST" if improved else ""
    iou_str = f" | Mean IoU: {mean_iou:.4f}" if mean_iou is not None else ""
    logger.info(
        f"Epoch {epoch+1:3d}/{total_epochs} | "
        f"Train Loss: {train_loss:.6f} | "
        f"Val Loss: {val_loss:.6f} | "
        f"LR: {current_lr:.2e}{iou_str} {status}"
    )
    if improved:
        logger.info(f"  Best val_loss: {best_loss:.6f}")


def log_iou_stats(saved_dir: str, stats: Dict):
    """Registra le statistiche IoU."""
    logger.info(f"Predizioni salvate in: {saved_dir}")
    logger.info(f"Immagini salvate: {stats['saved_count']}/{stats['total_count']}")
    logger.info(f"  Mean IoU:   {stats['mean_iou']:.4f}")
    logger.info(f"  Median IoU: {stats['median_iou']:.4f}")
    logger.info(f"  Min IoU:    {stats['min_iou']:.4f}")
    logger.info(f"  Max IoU:    {stats['max_iou']:.4f}")
    logger.info(f"  Std IoU:    {stats['std_iou']:.4f}")


def save_checkpoint(checkpoint_path: Path, epoch: int, model: nn.Module,
                    optimizer: torch.optim.Optimizer, best_loss: float,
                    iou_history: List[Dict], train_loss: float, val_loss: float):
    """Salva il checkpoint del training."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_loss,
        'iou_history': iou_history,
        'train_loss': train_loss,
        'val_loss': val_loss
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(checkpoint_path: Path, model: nn.Module,
                   optimizer: torch.optim.Optimizer, device: str) -> Tuple[int, float, List[Dict]]:
    """Carica il checkpoint del training."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = checkpoint.get('epoch', 0) + 1
    best_loss = checkpoint.get('best_loss', float('inf'))
    iou_history = checkpoint.get('iou_history', [])
    
    logger.info("Checkpoint caricato con successo")
    logger.info(f"  Riprendendo dall'epoca {start_epoch + 1}")
    logger.info(f"  Best loss precedente: {best_loss:.6f}")
    logger.info(f"  Epoche completate: {len(iou_history)}\n")
    
    return start_epoch, best_loss, iou_history


def save_iou_history(output_dir: Path, iou_history: List[Dict]):
    """Salva la cronologia IoU in un file di testo."""
    iou_history_file = output_dir / 'iou_history.txt'
    
    with open(iou_history_file, 'w') as f:
        f.write("Epoch | Mean IoU | Median IoU | Min IoU | Max IoU | Std IoU\n")
        f.write("-" * 65 + "\n")
        
        if iou_history:
            for entry in iou_history:
                f.write(
                    f"{entry['epoch']:5d} | "
                    f"{entry['mean_iou']:8.4f} | "
                    f"{entry['median_iou']:9.4f} | "
                    f"{entry['min_iou']:7.4f} | "
                    f"{entry['max_iou']:7.4f} | "
                    f"{entry['std_iou']:8.4f}\n"
                )
    
    logger.info(f"Cronologia IoU salvata in: {iou_history_file}")


def main():
    """Funzione principale di training."""
    
    device = get_device()
    
    # Download dataset
    logger.info("Download del dataset...")
    dataset_path = kagglehub.dataset_download("pkdarabi/brain-tumor-image-dataset-semantic-segmentation")
    
    # Carica dataset
    logger.info("Caricamento dataset...")
    train_dataset = BrainTumorRefinementDataset(dataset_path, split='train')
    valid_dataset = BrainTumorRefinementDataset(dataset_path, split='valid')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    # Inizializza modello e optimizer
    logger.info("Inizializzazione modello...")
    model = BoxRefinementNet(input_channels=3, hidden_dim=256)
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Carica checkpoint se esiste
    checkpoint_path = Path('training_checkpoint.pt')
    start_epoch = 0
    best_loss = float('inf')
    iou_history = []
    
    pred_dir = Path('./epoch_predictions')
    pred_dir.mkdir(parents=True, exist_ok=True)
    
    if checkpoint_path.exists():
        start_epoch, best_loss, iou_history = load_checkpoint(
            checkpoint_path, model, optimizer, device
        )
    else:
        logger.info("Nessun checkpoint trovato. Inizio addestramento da zero.\n")
    
    epochs = 100
    save_interval = 5
    
    logger.info("=" * 70)
    logger.info("INIZIO ADDESTRAMENTO")
    logger.info("=" * 70 + "\n")
    
    try:
        for epoch in range(start_epoch, epochs):
            logger.info(f"Epoca {epoch+1}/{epochs}")
            
            # Training
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            
            # Validation
            val_loss = validate_epoch(model, valid_loader, criterion, device)
            
            # Log statistiche
            current_lr = optimizer.param_groups[0]['lr']
            improved = val_loss < best_loss
            if improved:
                best_loss = val_loss
            
            # Inizializza mean_iou
            current_mean_iou = None
            
            # Salva best model
            if improved:
                torch.save(model.state_dict(), 'box_refinement_best.pt')
            
            # Salva predizioni ogni N epoche
            if (epoch + 1) % save_interval == 0:
                logger.info(f"\nSalvataggio predizioni per epoca {epoch+1}...")
                try:
                    saved_dir, stats = save_predictions_per_epoch(
                        model=model,
                        dataset=valid_dataset,
                        epoch=epoch+1,
                        device=device,
                        output_dir='./epoch_predictions'
                    )
                    
                    iou_values = np.array(stats['iou_values'])
                    iou_values = iou_values[~np.isnan(iou_values)]
                    
                    if len(iou_values) > 0:
                        current_mean_iou = stats['mean_iou']
                        iou_entry = {
                            'epoch': epoch+1,
                            'mean_iou': stats['mean_iou'],
                            'median_iou': stats['median_iou'],
                            'min_iou': stats['min_iou'],
                            'max_iou': stats['max_iou'],
                            'std_iou': stats['std_iou']
                        }
                        iou_history.append(iou_entry)
                        
                        logger.info("")
                        log_iou_stats(saved_dir, stats)
                        logger.info("")
                    else:
                        logger.warning("Nessun valore IoU valido calcolato")
                        
                except Exception as e:
                    logger.error(f"Errore nel salvataggio predizioni: {e}")
            
            # Log con IoU (sempre, se disponibile)
            log_epoch_stats(epoch, epochs, train_loss, val_loss, 
                          current_lr, best_loss, improved, current_mean_iou)
            
            # Salva checkpoint ogni epoca per permettere la ripresa
            try:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_loss': best_loss,
                    'iou_history': iou_history,
                    'train_loss': train_loss,
                    'val_loss': val_loss
                }
                torch.save(checkpoint, checkpoint_path)
            except Exception as e:
                logger.error(f"Errore nel salvataggio checkpoint: {e}")
            
            # Aggiorna learning rate
            scheduler.step(val_loss)
    
    except KeyboardInterrupt:
        logger.info("\n" + "=" * 70)
        logger.info("ADDESTRAMENTO INTERROTTO DALL'UTENTE")
        logger.info("=" * 70)
    
    except Exception as e:
        logger.error(f"\nErrore durante l'addestramento: {e}")
        traceback.print_exc()
    
    # SALVA CRONOLOGIA IoU FINALE
    logger.info("\n" + "=" * 70)
    logger.info("STORIA IoU COMPLETA")
    logger.info("=" * 70)
    
    save_iou_history(pred_dir, iou_history)
    
    # Stampa la cronologia completa
    if iou_history:
        for entry in iou_history:
            logger.info(
                f"Epoch {entry['epoch']:3d}: "
                f"Mean={entry['mean_iou']:.4f}, "
                f"Median={entry['median_iou']:.4f}, "
                f"Min={entry['min_iou']:.4f}, "
                f"Max={entry['max_iou']:.4f}, "
                f"Std={entry['std_iou']:.4f}"
            )
    else:
        logger.warning("Nessun dato IoU disponibile")
    
    logger.info("\nAddestramento completato!")


if __name__ == "__main__":
    main()
