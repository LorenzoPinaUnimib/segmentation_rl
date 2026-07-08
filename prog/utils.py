"""
utils.py
────────
Aggiunta la funzione per estrarre e deformare (warp) la regione attiva.
"""
import numpy as np
import cv2

def compute_iou(b1, b2) -> float:
    """b1, b2 in formato [xmin, ymin, w, h]."""
    xi1, yi1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    xi2 = min(b1[0] + b1[2], b2[0] + b2[2])
    yi2 = min(b1[1] + b1[3], b2[1] + b2[3])
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union_area = (b1[2] * b1[3]) + (b2[2] * b2[3]) - inter_area
    return float(inter_area / max(1e-6, union_area))


def compute_giou(b1, b2) -> float:
    """Generalized IoU (Rezatofighi et al. 2019), formato [xmin, ymin, w, h].

    FIX (reward shaping): a differenza dell'IoU piano, il GIoU resta
    informativo anche quando b1 e b2 NON si sovrappongono affatto (IoU=0 in
    entrambi i casi, sia che b1 sia vicinissimo a b2 sia che sia dall'altra
    parte dell'immagine). Il termine aggiuntivo penalizza l'area del piu'
    piccolo box che li racchiude entrambi (enclosing box): piu' i due box
    sono lontani, piu' negativo diventa il GIoU, dando un gradiente utile
    esattamente nella fase (frequente qui: si parte sempre dall'intera
    immagine) in cui l'IoU piano e' zero per ogni azione possibile. Range:
    [-1, 1] (1 = box identici).
    """
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    b1x2, b1y2 = x1 + w1, y1 + h1
    b2x2, b2y2 = x2 + w2, y2 + h2

    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(b1x2, b2x2), min(b1y2, b2y2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    area1, area2 = w1 * h1, w2 * h2
    union_area = area1 + area2 - inter_area
    iou = inter_area / max(1e-6, union_area)

    xe1, ye1 = min(x1, x2), min(y1, y2)
    xe2, ye2 = max(b1x2, b2x2), max(b1y2, b2y2)
    enclosing_area = max(1e-6, (xe2 - xe1) * (ye2 - ye1))

    giou = iou - (enclosing_area - union_area) / enclosing_area
    return float(giou)

def extract_warped_region(image_chw, box_xywh, context=16, target_size=(224, 224)):
    """
    Estrae il crop, aggiunge 16 pixel di contesto e fa il warp a 224x224.
    image_chw: [C, H, W] tensore numpy
    """
    C, H, W = image_chw.shape
    x, y, w, h = box_xywh
    
    # Aggiungi contesto (16 pixel)
    x1 = int(max(0, x - context))
    y1 = int(max(0, y - context))
    x2 = int(min(W, x + w + context))
    y2 = int(min(H, y + h + context))
    
    # Gestione di box invalidi collassati
    if x2 <= x1 or y2 <= y1:
        return np.zeros((C, target_size[1], target_size[0]), dtype=np.float32)

    # Crop e Warp (OpenCV richiede formato HWC)
    img_hwc = image_chw.transpose(1, 2, 0)
    crop = img_hwc[y1:y2, x1:x2]
    
    warped = cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR)
    
    if C == 1:
        warped = np.expand_dims(warped, axis=-1)
        
    return warped.transpose(2, 0, 1) # Torna a [C, H, W]