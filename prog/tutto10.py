"""
Versione ridotta di tutto6.py, ottimizzata per:
- Training con --backbone-freeze all --pretrained-backbone ./pretrained/backbone_pretrained_backbone.pt
- Test con --test --model <checkpoint> --device cpu --backbone-freeze all --pretrained-backbone ./pretrained/backbone_pretrained_backbone.pt

Rimosse tutte le funzionalità non utilizzate (NoisyNet, data augmentation, curriculum adattivo,
buffer immagini, validazione infra-epoca, ecc.).
Le metriche si basano esclusivamente sull'ultima iterazione di ogni episodio (trigger o timeout).
Durante il training, nel progress bar e nei log viene mostrato l'IoU dello step corrente.
"""
import gc
import os
import copy
import math
import collections
import argparse
import random
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision.ops import roi_align
from pathlib import Path
import cv2
import imageio
import csv

# ─────────────────────────────────────────────────────────────────────────────
# COSTANTI (solo quelle effettivamente usate)
# ─────────────────────────────────────────────────────────────────────────────
N_ACTIONS = 9
ALPHA = 0.1
WARP_SIZE = (224, 224)
HISTORY_LENGTH = 10
CONTEXT_PIXELS = 16
MAX_STEPS_PER_EPISODE = 50
TRIGGER_REWARD = 3.0
REWARD_POSITIVE = 1.0
REWARD_NEGATIVE = -1.0
TAU_IOU = 0.6
GAMMA = 0.90
EPSILON_START = 1.0
EPSILON_END = 0.1
REWARD_CLIP = 10.0
EMBED_DIM = 512
COORD_FEAT_DIM = 5

# PER (Prioritized Experience Replay)
PER_ALPHA = 0.6
PER_BETA_START = 0.4
PER_BETA_END = 1.0
PER_EPS = 1e-5

# n-step
N_STEP = 3

# Reward scaling
REWARD_SCALING_EPS = 1e-6

# ─────────────────────────────────────────────────────────────────────────────
# 1. UTILITY FUNCTIONS (GPU)
# ─────────────────────────────────────────────────────────────────────────────
def compute_iou_tensor(b1, b2):
    # b1, b2: [N, 4] (x1, y1, x2, y2)
    xi1 = torch.max(b1[:, 0], b2[:, 0])
    yi1 = torch.max(b1[:, 1], b2[:, 1])
    xi2 = torch.min(b1[:, 2], b2[:, 2])
    yi2 = torch.min(b1[:, 3], b2[:, 3])
    inter_area = torch.clamp(xi2 - xi1, min=0) * torch.clamp(yi2 - yi1, min=0)
    area1 = torch.clamp(b1[:, 2] - b1[:, 0], min=0) * torch.clamp(b1[:, 3] - b1[:, 1], min=0)
    area2 = torch.clamp(b2[:, 2] - b2[:, 0], min=0) * torch.clamp(b2[:, 3] - b2[:, 1], min=0)
    union_area = area1 + area2 - inter_area
    return inter_area / torch.clamp(union_area, min=1e-6)

def compute_giou_tensor(b1, b2):
    # b1, b2: [N, 4] (x1, y1, x2, y2)
    xi1 = torch.max(b1[:, 0], b2[:, 0])
    yi1 = torch.max(b1[:, 1], b2[:, 1])
    xi2 = torch.min(b1[:, 2], b2[:, 2])
    yi2 = torch.min(b1[:, 3], b2[:, 3])
    inter_area = torch.clamp(xi2 - xi1, min=0) * torch.clamp(yi2 - yi1, min=0)
    area1 = torch.clamp(b1[:, 2] - b1[:, 0], min=0) * torch.clamp(b1[:, 3] - b1[:, 1], min=0)
    area2 = torch.clamp(b2[:, 2] - b2[:, 0], min=0) * torch.clamp(b2[:, 3] - b2[:, 1], min=0)
    union_area = area1 + area2 - inter_area
    iou = inter_area / torch.clamp(union_area, min=1e-6)
    cx1 = torch.min(b1[:, 0], b2[:, 0])
    cy1 = torch.min(b1[:, 1], b2[:, 1])
    cx2 = torch.max(b1[:, 2], b2[:, 2])
    cy2 = torch.max(b1[:, 3], b2[:, 3])
    enclose_area = torch.clamp(cx2 - cx1, min=0) * torch.clamp(cy2 - cy1, min=0)
    return iou - (enclose_area - union_area) / torch.clamp(enclose_area, min=1e-6)

def compute_diou_tensor(b1, b2):
    # b1, b2: [N, 4] (x1, y1, x2, y2)
    # Center points
    c1_x = (b1[:, 0] + b1[:, 2]) / 2
    c1_y = (b1[:, 1] + b1[:, 3]) / 2
    c2_x = (b2[:, 0] + b2[:, 2]) / 2
    c2_y = (b2[:, 1] + b2[:, 3]) / 2
    
    # Euclidean distance squared between centers
    rho2 = (c1_x - c2_x)**2 + (c1_y - c2_y)**2
    
    # Diagonal length squared of the smallest enclosing box
    # smallest enclosing box: min x1, min y1, max x2, max y2
    x1_min = torch.min(b1[:, 0], b2[:, 0])
    y1_min = torch.min(b1[:, 1], b2[:, 1])
    x2_max = torch.max(b1[:, 2], b2[:, 2])
    y2_max = torch.max(b1[:, 3], b2[:, 3])
    c2 = (x2_max - x1_min)**2 + (y2_max - y1_min)**2
    
    # IoU
    xi1 = torch.max(b1[:, 0], b2[:, 0])
    yi1 = torch.max(b1[:, 1], b2[:, 1])
    xi2 = torch.min(b1[:, 2], b2[:, 2])
    yi2 = torch.min(b1[:, 3], b2[:, 3])
    inter = torch.clamp(xi2 - xi1, min=0) * torch.clamp(yi2 - yi1, min=0)
    area1 = torch.clamp(b1[:, 2] - b1[:, 0], min=0) * torch.clamp(b1[:, 3] - b1[:, 1], min=0)
    area2 = torch.clamp(b2[:, 2] - b2[:, 0], min=0) * torch.clamp(b2[:, 3] - b2[:, 1], min=0)
    union = area1 + area2 - inter
    iou = inter / torch.clamp(union, min=1e-6)
    
    diou = iou - (rho2 / torch.clamp(c2, min=1e-6))
    return diou

class RunningMeanStd:
    """Stima online media/varianza per reward scaling."""
    def __init__(self, eps=1e-4, device="cpu"):
        self.mean = torch.zeros(1, device=device)
        self.var = torch.ones(1, device=device)
        self.count = eps

    def update(self, x: torch.Tensor):
        x = x.detach().reshape(-1)
        if x.numel() == 0:
            return
        batch_mean = x.mean()
        batch_var = x.var(unbiased=False)
        batch_count = x.numel()
        delta = batch_mean - self.mean.to(batch_mean.device)
        tot_count = self.count + batch_count
        new_mean = self.mean.to(batch_mean.device) + delta * batch_count / tot_count
        m_a = self.var.to(batch_mean.device) * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + (delta ** 2) * self.count * batch_count / tot_count
        new_var = m2 / tot_count
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def std(self):
        return torch.sqrt(self.var + REWARD_SCALING_EPS)

    def scale_only(self, x: torch.Tensor):
        return x / self.std().to(x.device)

    def state_dict(self):
        return {"mean": self.mean.clone(), "var": self.var.clone(), "count": self.count}

    def load_state_dict(self, sd):
        self.mean = sd["mean"].clone()
        self.var = sd["var"].clone()
        self.count = sd["count"]

# ─────────────────────────────────────────────────────────────────────────────
# 2. AMBIENTE VETTORIZZATO
# ─────────────────────────────────────────────────────────────────────────────
class BatchedActiveLocalizationEnv:
    def __init__(self, pytorch_dataset, batch_size, device):
        self.dataset = pytorch_dataset
        self.num_envs = batch_size
        self.tau_iou = TAU_IOU
        self.device = device

        sample = self.dataset[0]
        self.C, self.H, self.W = sample["image"].shape

        self.current_images = torch.zeros((batch_size, self.C, self.H, self.W),
                                          dtype=torch.float32, device=device)
        self.gt_boxes = torch.zeros((batch_size, 4), dtype=torch.float32, device=device)
        self.boxes = torch.zeros((batch_size, 4), dtype=torch.float32, device=device)
        self.histories = torch.zeros((batch_size, HISTORY_LENGTH * N_ACTIONS),
                                     dtype=torch.float32, device=device)
        self.current_steps = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.previous_ious = torch.zeros(batch_size, dtype=torch.float32, device=device)
        self.previous_gious = torch.zeros(batch_size, dtype=torch.float32, device=device)
        self.previous_dious = torch.zeros(batch_size, dtype=torch.float32, device=device)
        self.previous_dists = torch.zeros(batch_size, dtype=torch.float32, device=device)
        self.best_ious = torch.zeros(batch_size, dtype=torch.float32, device=device)
        self.best_boxes = torch.zeros((batch_size, 4), dtype=torch.float32, device=device)
        self.one_hot_buffer = torch.zeros((batch_size, N_ACTIONS), dtype=torch.float32, device=device)
        self.repeat_counter = torch.zeros(batch_size, dtype=torch.float32, device=device)
        self.last_action = torch.full((batch_size,), -1, dtype=torch.long, device=device)

    def _load_sample_into_slot(self, slot, dataset_idx):
        sample = self.dataset[dataset_idx % len(self.dataset)]
        self.current_images[slot] = sample["image"].to(self.device, non_blocking=True)

        mask = sample["mask"].squeeze(0)
        pos = torch.where(mask > 0.5)
        if len(pos[0]) > 0:
            ymin, ymax = torch.min(pos[0]), torch.max(pos[0])
            xmin, xmax = torch.min(pos[1]), torch.max(pos[1])
            # gt_boxes in formato (x1, y1, x2, y2)
            self.gt_boxes[slot] = torch.tensor(
                [xmin.float(), ymin.float(), xmax.float(), ymax.float()],
                dtype=torch.float32, device=self.device
            )
        else:
            self.gt_boxes[slot] = torch.tensor(
                [self.W / 4, self.H / 4, 3 * self.W / 4, 3 * self.H / 4],
                dtype=torch.float32, device=self.device
            )

        # box iniziale = immagine intera (x1, y1, x2, y2)
        self.boxes[slot, 0] = 0.0
        self.boxes[slot, 1] = 0.0
        self.boxes[slot, 2] = float(self.W)
        self.boxes[slot, 3] = float(self.H)

        self.histories[slot].zero_()
        self.current_steps[slot] = 0
        self.repeat_counter[slot] = 0
        self.last_action[slot] = -1

    def reset_all(self, indices, epoch=None, total_epochs=None, force_max_difficulty=None, curriculum_override=None):
        for i, idx in enumerate(indices):
            self._load_sample_into_slot(i, idx)

        curr_centers = self.get_centers(self.boxes)
        gt_centers = self.get_centers(self.gt_boxes)
        self.previous_dists = torch.norm(curr_centers - gt_centers, dim=1)
        self.previous_ious = compute_iou_tensor(self.boxes, self.gt_boxes)
        self.previous_gious = compute_giou_tensor(self.boxes, self.gt_boxes)
        self.previous_dious = compute_diou_tensor(self.boxes, self.gt_boxes)
        self.best_ious = self.previous_ious.clone()
        self.best_boxes = self.boxes.clone()
        return self._get_obs()

    def _get_obs(self):
        x1, y1, x2, y2 = self.boxes[:, 0], self.boxes[:, 1], self.boxes[:, 2], self.boxes[:, 3]

        rx1 = torch.clamp(x1 - CONTEXT_PIXELS, min=0, max=float(self.W))
        ry1 = torch.clamp(y1 - CONTEXT_PIXELS, min=0, max=float(self.H))
        rx2 = torch.clamp(x2 + CONTEXT_PIXELS, min=0, max=float(self.W))
        ry2 = torch.clamp(y2 + CONTEXT_PIXELS, min=0, max=float(self.H))
        rx2 = torch.maximum(rx2, rx1 + 1.0)
        ry2 = torch.maximum(ry2, ry1 + 1.0)

        batch_idx = torch.arange(self.num_envs, device=self.device, dtype=torch.float32)
        rois = torch.stack([batch_idx, rx1, ry1, rx2, ry2], dim=1)

        regions = roi_align(
            self.current_images, rois, output_size=WARP_SIZE,
            spatial_scale=1.0, sampling_ratio=2, aligned=True
        )

        extra = torch.stack([
            x1 / float(self.W),
            y1 / float(self.H),
            x2 / float(self.W),
            y2 / float(self.H),
            self.current_steps.float() / float(MAX_STEPS_PER_EPISODE),
        ], dim=1)

        return {"regions": regions, "histories": self.histories, "extra": extra}

    def get_centers(self, boxes):
        x_c = boxes[:, 0] + boxes[:, 2] / 2
        y_c = boxes[:, 1] + boxes[:, 3] / 2
        return torch.stack([x_c, y_c], dim=1)

    def step(self, actions):
        self.current_steps += 1

        same_action = (actions == self.last_action) & (actions < 8)
        self.repeat_counter = torch.where(
            same_action, self.repeat_counter + 1.0, torch.zeros_like(self.repeat_counter)
        )
        self.last_action = actions.clone()

        self.one_hot_buffer.zero_()
        self.one_hot_buffer.scatter_(1, actions.unsqueeze(1), 1.0)
        self.histories = torch.cat([self.histories[:, N_ACTIONS:], self.one_hot_buffer], dim=1)

        # w, h derivati dal box corrente (x1, y1, x2, y2)
        w, h = self.boxes[:, 2] - self.boxes[:, 0], self.boxes[:, 3] - self.boxes[:, 1]
        aw, ah = ALPHA * w, ALPHA * h
 # delta espliciti sui 4 estremi (x1, y1, x2, y2)
        dx1 = torch.zeros(self.num_envs, device=self.device)
        dy1 = torch.zeros(self.num_envs, device=self.device)
        dx2 = torch.zeros(self.num_envs, device=self.device)
        dy2 = torch.zeros(self.num_envs, device=self.device)

        m0 = (actions == 0); m1 = (actions == 1); m2 = (actions == 2); m3 = (actions == 3)
        m4 = (actions == 4); m5 = (actions == 5); m6 = (actions == 6); m7 = (actions == 7)

        # traslazione destra/sinistra: entrambi gli estremi x si spostano insieme
        dx1[m0] = aw[m0]; dx2[m0] = aw[m0]
        dx1[m1] = -aw[m1]; dx2[m1] = -aw[m1]

        # traslazione su/giù: entrambi gli estremi y si spostano insieme
        dy1[m2] = -ah[m2]; dy2[m2] = -ah[m2]
        dy1[m3] = ah[m3]; dy2[m3] = ah[m3]

        # shrink orizzontale (centrato sull'asse x): x1 avanza, x2 arretra
        dx1[m4] = aw[m4]; dx2[m4] = -aw[m4]
        # shrink verticale (centrato sull'asse y): y1 avanza, y2 arretra
        dy1[m5] = ah[m5]; dy2[m5] = -ah[m5]

        # expand orizzontale (centrato): x1 arretra, x2 avanza
        dx1[m6] = -aw[m6]; dx2[m6] = aw[m6]
        # expand verticale (centrato): y1 arretra, y2 avanza
        dy1[m7] = -ah[m7]; dy2[m7] = ah[m7]

        new_x1 = torch.clamp(self.boxes[:, 0] + dx1, 0, self.W)
        new_y1 = torch.clamp(self.boxes[:, 1] + dy1, 0, self.H)
        new_x2 = torch.clamp(self.boxes[:, 2] + dx2, 0, self.W)
        new_y2 = torch.clamp(self.boxes[:, 3] + dy2, 0, self.H)

        # garantisce larghezza/altezza minima 10 px senza invertire x1/x2 o y1/y2
        new_x2 = torch.maximum(new_x2, new_x1 + 10)
        new_y2 = torch.maximum(new_y2, new_y1 + 10)
        new_x2 = torch.clamp(new_x2, max=self.W)
        new_y2 = torch.clamp(new_y2, max=self.H)
        new_x1 = torch.minimum(new_x1, new_x2 - 10)
        new_y1 = torch.minimum(new_y1, new_y2 - 10)

        self.boxes[:, 0] = new_x1
        self.boxes[:, 1] = new_y1
        self.boxes[:, 2] = new_x2
        self.boxes[:, 3] = new_y2

        new_ious = compute_iou_tensor(self.boxes, self.gt_boxes)

        new_gious = compute_giou_tensor(self.boxes, self.gt_boxes)
        new_dious = compute_diou_tensor(self.boxes, self.gt_boxes)
        
        # 1. Shaping basato solo su GIoU (teoricamente fondato)
        giou_shaped = (GAMMA * new_dious) - self.previous_dious
        rewards = giou_shaped * 5.0

        # 2. Piccola penalità per stallo/loop combinata
        iou_delta = new_ious - self.previous_ious
        stall_penalty = torch.where(
            (actions < 8) & (iou_delta < 0.005),
            -0.02 * torch.clamp(self.repeat_counter, min=0.0),
            torch.zeros_like(iou_delta)
        )
        rewards = rewards + stall_penalty

        # 3. Trigger reward semplice e forte
        terminated = (actions == 8)
        rewards[terminated] = torch.where(
            new_ious[terminated] >= self.tau_iou,
            5.0 + 20.0 * (new_ious[terminated] - self.tau_iou),  # bonus fino a 11
            -5.0 - 20.0 * (self.tau_iou - new_ious[terminated])   # penalità fino a -13
        )

        # 4. Nessuna penalità di regressione
        
        '''
        curr_centers = self.get_centers(self.boxes)
        gt_centers = self.get_centers(self.gt_boxes)
        curr_dists = torch.norm(curr_centers - gt_centers, dim=1)

        giou_shaped = (GAMMA * new_gious) - self.previous_gious
        dist_shaped = self.previous_dists - (GAMMA * curr_dists)
        rewards = (giou_shaped * 10.0) + (dist_shaped * 0.5)

        stagnation_mask = (actions < 8) & (torch.abs(iou_diff) < 0.005)
        stagnation_scale = torch.clamp(1.0 - (new_ious / self.tau_iou), min=0.0, max=1.0)
        rewards[stagnation_mask] -= 0.05 * stagnation_scale[stagnation_mask]

        loop_penalty = torch.clamp(self.repeat_counter - 3.0, min=0.0) * 0.08
        rewards = rewards - loop_penalty

        regressed_mask = (actions < 8) & ((self.previous_ious - new_ious) > 0)
        rewards[regressed_mask] -=z * (self.best_ious[regressed_mask] - new_ious[regressed_mask])

        terminated = (actions == 8)
        good_trigger = terminated & (new_ious >= self.tau_iou)
        bad_trigger = terminated & (new_ious < self.tau_iou)
        rewards[good_trigger] += 5.0 + (new_ious[good_trigger] - self.tau_iou) * 10.0
        remaining_steps = (MAX_STEPS_PER_EPISODE - self.current_steps.float())
        rewards[bad_trigger] -= (1.0 + (self.tau_iou - new_ious[bad_trigger]) * 2.0) + (remaining_steps[bad_trigger] * 0.2)
        

        self.previous_dists = curr_dists.clone()

        '''
        improved = new_ious > self.best_ious
        if improved.any():
            self.best_ious[improved] = new_ious[improved]
            self.best_boxes[improved] = self.boxes[improved].clone()
        truncated = (self.current_steps >= MAX_STEPS_PER_EPISODE)
        #self.previous_ious = new_ious
        #self.previous_gious = new_gious
        #self.previous_dious = new_dious
        
        return self._get_obs(), rewards, (terminated | truncated), new_ious, new_gious, new_dious

    def _simulate_move_boxes(self, boxes):
        num = boxes.shape[0]
        # w, h derivati dai box (x1, y1, x2, y2)
        w, h = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
        aw, ah = ALPHA * w, ALPHA * h
        zeros = torch.zeros(num, device=self.device)

        
        deltas = {
            0: (aw, zeros, aw, zeros),        # trasla destra: x1+=aw, x2+=aw
            1: (-aw, zeros, -aw, zeros),      # trasla sinistra
            2: (zeros, -ah, zeros, -ah),      # trasla su
            3: (zeros, ah, zeros, ah),        # trasla giù
            4: (aw, zeros, -aw, zeros),       # shrink orizzontale: x1+=aw, x2-=aw
            5: (zeros, ah, zeros, -ah),       # shrink verticale
            6: (-aw, zeros, aw, zeros),       # expand orizzontale: x1-=aw, x2+=aw
            7: (zeros, -ah, zeros, ah),       # expand verticale
        }

        candidates = []
        for a in range(8):
            dx1, dy1, dx2, dy2 = deltas[a]
            new_x1 = boxes[:, 0] + dx1
            new_y1 = boxes[:, 1] + dy1
            new_x2 = boxes[:, 2] + dx2
            new_y2 = boxes[:, 3] + dy2

            # clamp coordinate ai bordi immagine
            new_x1 = torch.clamp(new_x1, 0, self.W)
            new_y1 = torch.clamp(new_y1, 0, self.H)
            new_x2 = torch.clamp(new_x2, 0, self.W)
            new_y2 = torch.clamp(new_y2, 0, self.H)

            # garantisci larghezza/altezza minima 10, senza spostare l'estremo opposto
            new_x2 = torch.maximum(new_x2, new_x1 + 10)
            new_y2 = torch.maximum(new_y2, new_y1 + 10)

            nb = boxes.clone()
            nb[:, 0] = new_x1
            nb[:, 1] = new_y1
            nb[:, 2] = new_x2
            nb[:, 3] = new_y2
            candidates.append(nb)
        return candidates

    def set_tau_iou(self, value):
        self.tau_iou = float(value)

    def _param_distance(self, boxes):
        dx = (boxes[:,0] - self.gt_boxes[:,0]) / self.W
        dy = (boxes[:,1] - self.gt_boxes[:,1]) / self.H
        dw = (boxes[:,2] - self.gt_boxes[:,2]) / self.W
        dh = (boxes[:,3] - self.gt_boxes[:,3]) / self.H
        return dx**2 + dy**2 + dw**2 + dh**2

    def compute_oracle_actions(self):
        candidates = self._simulate_move_boxes(self.boxes)
        dious = torch.stack([compute_diou_tensor(cb, self.gt_boxes) for cb in candidates], dim=1)

        # Nota: non mascheriamo più le mosse "clippate" dal bordo (quelle il cui
        # effetto viene troncato dal clamp sui limiti dell'immagine). L'oracolo deve
        # poter scegliere anche un'azione che si scontra col bordo, se è comunque
        # quella che massimizza il DIoU risultante: in tal caso il box si sposterà
        # fin dove il clamp lo consente, invece di essere escluso a priori.
        best_move_action = dious.argmax(dim=1)

        current_iou = compute_iou_tensor(self.boxes, self.gt_boxes)
        should_trigger = current_iou >= self.tau_iou
        return torch.where(should_trigger, torch.full_like(best_move_action, 8), best_move_action)

# ─────────────────────────────────────────────────────────────────────────────
# 3. REPLAY BUFFER (solo EmbeddingReplayBuffer con PER)
# ─────────────────────────────────────────────────────────────────────────────
class PrioritizedReplayMixin:
    def _per_init(self, capacity, alpha=PER_ALPHA, priority_device=None):
        self.per_alpha = alpha
        self.per_beta = PER_BETA_START
        self.max_priority = 1.0
        self._priority_device = priority_device if priority_device is not None else self.device
        self.priorities = torch.zeros(capacity, dtype=torch.float32, device=self._priority_device)

    def _per_mark_new(self, idx):
        idx = idx.to(self._priority_device)
        self.priorities[idx] = self.max_priority

    def set_beta(self, beta):
        self.per_beta = float(min(1.0, max(0.0, beta)))

    def sample_indices_per(self, batch_size):
        valid_priorities = self.priorities[:self.size].clamp(min=PER_EPS)
        probs = valid_priorities ** self.per_alpha
        probs = probs / probs.sum()
        idx = torch.multinomial(probs, batch_size, replacement=True)
        is_weights = (self.size * probs[idx]).clamp(min=1e-8) ** (-self.per_beta)
        is_weights = is_weights / is_weights.max()
        return idx.to(self._priority_device), is_weights.to(self.device)

    def update_priorities(self, idx, td_errors):
        idx = idx.to(self._priority_device)
        new_p = td_errors.detach().abs().to(self._priority_device) + PER_EPS
        self.priorities[idx] = new_p
        self.max_priority = max(self.max_priority, new_p.max().item())

class EmbeddingReplayBuffer(PrioritizedReplayMixin):
    def __init__(self, capacity, embed_dim, history_dim, device, coord_dim=COORD_FEAT_DIM):
        self.device = device
        self.capacity, self.pos, self.size = capacity, 0, 0
        self.embeds = torch.zeros((capacity, embed_dim), dtype=torch.float32, device=device)
        self.next_embeds = torch.zeros((capacity, embed_dim), dtype=torch.float32, device=device)
        self.histories = torch.zeros((capacity, history_dim), dtype=torch.float32, device=device)
        self.next_histories = torch.zeros((capacity, history_dim), dtype=torch.float32, device=device)
        self.extra = torch.zeros((capacity, coord_dim), dtype=torch.float32, device=device)
        self.next_extra = torch.zeros((capacity, coord_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros(capacity, dtype=torch.long, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.terminals = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.is_expert = torch.zeros(capacity, dtype=torch.float32, device=device)
        self._per_init(capacity, priority_device=device)

    def push_batch(self, embeds, histories, extra, actions, rewards, next_embeds, next_histories,
                   next_extra, terminals, is_expert):
        b_size = embeds.shape[0]
        idx = (self.pos + torch.arange(b_size, device=self.device)) % self.capacity

        self.embeds[idx] = embeds
        self.next_embeds[idx] = next_embeds
        self.histories[idx] = histories
        self.next_histories[idx] = next_histories
        self.extra[idx] = extra
        self.next_extra[idx] = next_extra
        self.actions[idx] = actions
        self.rewards[idx] = rewards
        self.terminals[idx] = terminals
        self.is_expert[idx] = is_expert
        self._per_mark_new(idx)

        self.pos = (self.pos + b_size) % self.capacity
        self.size = min(self.size + b_size, self.capacity)

    def sample_per(self, batch_size):
        idx, is_weights = self.sample_indices_per(batch_size)
        idx_dev = idx.to(self.device)
        return (
            self.embeds[idx_dev], self.histories[idx_dev], self.extra[idx_dev], self.actions[idx_dev],
            self.rewards[idx_dev], self.next_embeds[idx_dev], self.next_histories[idx_dev],
            self.next_extra[idx_dev], self.terminals[idx_dev], self.is_expert[idx_dev],
            idx, is_weights
        )

    # Campionamento uniforme per quando il PER è disattivato
    def sample_uniform(self, batch_size):
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.embeds[idx], self.histories[idx], self.extra[idx], self.actions[idx],
            self.rewards[idx], self.next_embeds[idx], self.next_histories[idx],
            self.next_extra[idx], self.terminals[idx], self.is_expert[idx]
        )

# ─────────────────────────────────────────────────────────────────────────────
# 4. MODELLI
# ─────────────────────────────────────────────────────────────────────────────
class SpatialAttentionPool(nn.Module):
    def __init__(self, in_channels=512, embed_dim=512):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.attn_logits = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.project = nn.Sequential(
            nn.Linear(in_channels * 2, embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, feat_map):
        refined = self.refine(feat_map)
        b, c, h, w = refined.shape
        attn = self.attn_logits(refined).view(b, 1, h * w)
        attn = torch.softmax(attn, dim=-1).view(b, 1, h, w)
        attn_pooled = (refined * attn).sum(dim=[2, 3])
        avg_pooled = self.avg_pool(refined).view(b, c)
        combined = torch.cat([attn_pooled, avg_pooled], dim=1)
        return self.project(combined)

class QNetwork(nn.Module):
    def __init__(self, embed_dim, history_dim, n_actions, hidden=512, coord_dim=COORD_FEAT_DIM):
        super().__init__()
        self.coord_dim = coord_dim
        input_dim = embed_dim + history_dim + coord_dim
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, n_actions)
        )

    def forward(self, embeds, histories, extra_feats):
        x = torch.cat([embeds, histories, extra_feats], dim=1)
        x = self.shared(x)
        v = self.value_stream(x)
        a = self.advantage_stream(x)
        return v + (a - a.mean(dim=1, keepdim=True))

class PolicyNetwork(nn.Module):
    def __init__(self, history_dim, n_actions, pretrained_backbone=None, use_spatial_attention=None):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights

        if pretrained_backbone is not None:
            weights = None
        else:
            weights = ResNet18_Weights.DEFAULT

        backbone = resnet18(weights=weights)
        backbone.avgpool = nn.Identity()
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self._backbone_channels = 512
        self._backbone_spatial = WARP_SIZE[0] // 32

        # Forza l'uso di SpatialAttentionPool se richiesto esplicitamente
        if use_spatial_attention is True:
            self.spatial_pool = SpatialAttentionPool(in_channels=self._backbone_channels, embed_dim=EMBED_DIM)
            self.use_spatial_attention = True
        elif use_spatial_attention is False:
            self.spatial_pool = nn.AdaptiveAvgPool2d(1)
            self.use_spatial_attention = False
        else:
            # Comportamento originale: decide in base al checkpoint del backbone preaddestrato
            if pretrained_backbone is not None:
                checkpoint = torch.load(pretrained_backbone, map_location="cpu", weights_only=False)
                if isinstance(checkpoint, dict):
                    backbone_state_dict = checkpoint.get("backbone_state_dict", checkpoint)
                    spatial_pool_state_dict = checkpoint.get("spatial_pool_state_dict", None)
                else:
                    backbone_state_dict = checkpoint
                    spatial_pool_state_dict = None

                # Rimuovi prefissi
                fixed_state_dict = {}
                for k, v in backbone_state_dict.items():
                    if k.startswith("backbone."):
                        fixed_state_dict[k[9:]] = v
                    else:
                        fixed_state_dict[k] = v
                fixed_state_dict = {k: v for k, v in fixed_state_dict.items() if not k.startswith("fc.")}
                self.backbone.load_state_dict(fixed_state_dict, strict=False)

                # Usa SpatialAttentionPool solo se presenti pesi preaddestrati
                if spatial_pool_state_dict is not None:
                    self.spatial_pool = SpatialAttentionPool(in_channels=self._backbone_channels, embed_dim=EMBED_DIM)
                    self.spatial_pool.load_state_dict(spatial_pool_state_dict, strict=True)
                    for p in self.spatial_pool.parameters():
                        p.requires_grad_(False)
                    self.spatial_pool.eval()
                    self.use_spatial_attention = True
                else:
                    self.spatial_pool = nn.AdaptiveAvgPool2d(1)
                    self.use_spatial_attention = False
            else:
                self.spatial_pool = nn.AdaptiveAvgPool2d(1)
                self.use_spatial_attention = False

        # Congela tutto il backbone
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()

        self.head = QNetwork(EMBED_DIM, history_dim, n_actions)

    def embed_regions(self, regions):
        feat_flat = self.backbone(regions)
        b = feat_flat.shape[0]
        feat_map = feat_flat.view(b, self._backbone_channels,
                                   self._backbone_spatial, self._backbone_spatial)
        pooled = self.spatial_pool(feat_map)
        if self.use_spatial_attention:
            return pooled
        return pooled.view(b, self._backbone_channels)

    def forward(self, regions, histories, extra_feats):
        if regions.dim() == 4:
            embeds = self.embed_regions(regions)
        else:
            embeds = regions  # assume già embedding
        return self.head(embeds, histories, extra_feats)

    def set_inference_mode(self):
        self.backbone.eval()
        self.spatial_pool.eval()
        self.head.eval()

    def train(self, mode=True):
        super().train(mode)
        # Forza backbone in eval
        self.backbone.eval()
        self.spatial_pool.eval()
        self.head.train(mode)

# ─────────────────────────────────────────────────────────────────────────────
# 5. VALIDAZIONE (solo metriche finali)
# ─────────────────────────────────────────────────────────────────────────────
def validate(env, policy_net, val_indices, device, writer, global_step, epoch, n_epochs, tag="Validation"):
    policy_net.eval()
    policy_net.set_inference_mode()

    obs = env.reset_all(val_indices)
    num_envs = env.num_envs
    reward_sums = torch.zeros(num_envs, device=device)   # somma reward per episodio
    step_counts = torch.zeros(num_envs, device=device)   # conteggio passi per episodio (per media)
    active_mask = torch.ones(num_envs, dtype=torch.bool, device=device)
    last_iou_per_slot = torch.zeros(num_envs, device=device)
    last_diou_per_slot = torch.zeros(num_envs, device=device)
    last_giou_per_slot = torch.zeros(num_envs, device=device)
    final_ious = torch.zeros(num_envs, device=device)
    final_gious = torch.zeros(num_envs, device=device)
    final_dious = torch.zeros(num_envs, device=device)

    with torch.no_grad():
        for _ in range(MAX_STEPS_PER_EPISODE):
            if not active_mask.any():
                break
            q_values = policy_net(obs["regions"], obs["histories"], obs["extra"])
            actions = q_values.argmax(dim=1)
            step_counts[active_mask] += 1
            next_obs, rewards, dones, ious, gious, dious = env.step(actions)
            
            reward_sums[active_mask] += rewards[active_mask]

        newly_done = active_mask & dones
        final_ious[newly_done]  = ious[newly_done]
        final_gious[newly_done] = gious[newly_done]
        final_dious[newly_done] = dious[newly_done]

        active_mask = active_mask & (~dones)
        last_iou_per_slot  = torch.where(active_mask, ious, last_iou_per_slot)
        last_diou_per_slot = torch.where(active_mask, dious, last_diou_per_slot)
        last_giou_per_slot = torch.where(active_mask, gious, last_giou_per_slot)
        obs = next_obs

    if active_mask.any():
        final_ious[active_mask] = last_iou_per_slot[active_mask]
        final_dious[active_mask] = last_diou_per_slot[active_mask]
        final_gious[active_mask] = last_giou_per_slot[active_mask]

    final_avg_iou = final_ious.mean().item()
    final_avg_diou = final_dious.mean().item()
    final_avg_giou = final_gious.mean().item()
    mean_reward = reward_sums.mean().item()
    mean_step = step_counts.mean().item()
    final_best_iou = final_ious.max().item()
    final_best_diou = final_dious.max().item()
    final_best_giou = final_gious.max().item()
    max_reward = reward_sums.max().item()
    max_step = step_counts.max().item()
    final_std_iou = final_ious.std(unbiased=False).item()
    final_std_diou = final_dious.std(unbiased=False).item()
    final_std_giou = final_gious.std(unbiased=False).item()
    std_reward = reward_sums.std(unbiased=False).item()
    std_step = step_counts.std(unbiased=False).item()
    success_rate = (final_ious >= TAU_IOU).float().sum().item()

    print(f"[Epoch {epoch}] {tag} (fine episodio) - "
          f"Final IoU: {final_avg_iou:.4f}, Best IoU: {final_best_iou:.4f}, Success Rate: {success_rate:.4f}")

    writer.add_scalar(f"{tag}/Final_Avg_IoU", final_avg_iou, global_step)
    writer.add_scalar(f"{tag}/Final_Std_IoU", final_std_iou, global_step)
    writer.add_scalar(f"{tag}/Final_Best_IoU", final_best_iou, global_step)
    writer.add_scalar(f"{tag}/Final_Avg_DIoU", final_avg_diou, global_step)
    writer.add_scalar(f"{tag}/Final_Std_DIoU", final_std_diou, global_step)
    writer.add_scalar(f"{tag}/Final_Best_DIoU", final_best_diou, global_step)
    writer.add_scalar(f"{tag}/Final_Avg_GIoU", final_avg_giou, global_step)
    writer.add_scalar(f"{tag}/Final_Std_GIoU", final_std_giou, global_step)
    writer.add_scalar(f"{tag}/Final_Best_GIoU", final_best_giou, global_step)
    writer.add_scalar(f"{tag}/Success_Rate", success_rate, global_step)
    
    writer.add_scalar(f"{tag}/Reward_mean", mean_reward, global_step)
    writer.add_scalar(f"{tag}/Reward_std", std_reward, global_step)
    writer.add_scalar(f"{tag}/Reward_max", max_reward, global_step)
    
    writer.add_scalar(f"{tag}/Step_max", mean_step, global_step)
    writer.add_scalar(f"{tag}/Step_std", std_step, global_step)
    writer.add_scalar(f"{tag}/Step_avg", max_step, global_step)

    policy_net.train()
      # Ritorna tutte le metriche in un dizionario
    return {
        'avg_iou': final_avg_iou,
        'std_iou': final_std_iou,
        'best_iou': final_best_iou,
        'avg_diou': final_avg_diou,
        'std_diou': final_std_diou,
        'best_diou': final_best_diou,
        'avg_giou': final_avg_giou,
        'std_giou': final_std_giou,
        'best_giou': final_best_giou,
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'max_reward': max_reward,
        'mean_step': mean_step,
        'std_step': std_step,
        'max_step': max_step,
        'success_rate': success_rate,
    }

# ─────────────────────────────────────────────────────────────────────────────
# 6. N-STEP ACCUMULATOR
# ─────────────────────────────────────────────────────────────────────────────
class NStepAccumulator:
    def __init__(self, num_envs, n_step, gamma):
        self.num_envs = num_envs
        self.n_step = n_step
        self.gamma = gamma
        self.queues = [collections.deque() for _ in range(num_envs)]

    def push_and_pop(self, states, histories, extra, actions, rewards, next_states, next_histories,
                      next_extra, dones, is_expert, active_mask=None):
        ready = []
        rewards_cpu = rewards.detach()
        dones_cpu = dones.detach()
        is_expert_cpu = is_expert.detach()
        active_cpu = active_mask.detach() if active_mask is not None else None
        for slot in range(self.num_envs):
            if active_cpu is not None and not bool(active_cpu[slot]):
                continue
            transition = (
                states[slot], histories[slot], extra[slot], actions[slot],
                rewards_cpu[slot].item(), next_states[slot], next_histories[slot],
                next_extra[slot], dones_cpu[slot].item(), is_expert_cpu[slot].item()
            )
            self.queues[slot].append(transition)

            if dones_cpu[slot].item() > 0.5:
                ready.extend(self._flush_ready_v2(slot, force_all=True))
            elif len(self.queues[slot]) >= self.n_step:
                ready.extend(self._flush_ready_v2(slot, force_all=False))
        return ready

    def _flush_ready_v2(self, slot, force_all=False):
        out = []
        q = self.queues[slot]
        while q and (force_all or len(q) >= self.n_step):
            k = min(self.n_step, len(q))
            R = 0.0
            done_within = False
            last_state = None
            for i in range(k):
                _, _, _, _, r_i, ns_i, nh_i, ne_i, d_i, _ = q[i]
                R += (self.gamma ** i) * r_i
                last_state = (ns_i, nh_i, ne_i)
                if d_i:
                    done_within = True
                    break
            s0, h0, e0, a0, _, _, _, _, _, is_expert0 = q[0]
            ns_final, nh_final, ne_final = last_state
            out.append((s0, h0, e0, a0, R, ns_final, nh_final, ne_final,
                        1.0 if done_within else 0.0, is_expert0))
            q.popleft()
        return out

    def reset_slot(self, slot):
        self.queues[slot].clear()

# ─────────────────────────────────────────────────────────────────────────────
# 7. TRAINING LOOP (semplificato)
# ─────────────────────────────────────────────────────────────────────────────
def train(args, device, train_ds, val_ds):
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(args.output_root, "logs", timestamp)
    checkpoint_dir = os.path.join(args.output_root, "checkpoints", timestamp)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir)
    print(f"[INFO] Log directory: {log_dir}")
    print(f"[INFO] Checkpoint directory: {checkpoint_dir}")

    train_env = BatchedActiveLocalizationEnv(train_ds, batch_size=args.batch_size, device=device)
    val_env = BatchedActiveLocalizationEnv(val_ds, batch_size=args.batch_size, device=device)

    print(f"[INFO] Creazione PolicyNetwork (backbone ResNet18 congelato) + Target Network...")
    policy_net = PolicyNetwork(HISTORY_LENGTH * N_ACTIONS, N_ACTIONS,
                                pretrained_backbone=args.pretrained_backbone).to(device)
    target_net = copy.deepcopy(policy_net).to(device)
    target_net.eval()
    for p in target_net.parameters():
        p.requires_grad_(False)

    optimizer = optim.AdamW(policy_net.head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    total_training_steps = args.n_epochs * MAX_STEPS_PER_EPISODE
    total_optimizer_steps = max(total_training_steps * args.gradient_steps, 1)
    warmup_steps = max(int(0.05 * total_optimizer_steps), 1)
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_optimizer_steps - warmup_steps, 1), eta_min=args.learning_rate * 0.05
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps]
    )

    memory = EmbeddingReplayBuffer(args.replay_buffer_size, EMBED_DIM,
                                    HISTORY_LENGTH * N_ACTIONS, device)
    print(f"[INFO] EmbeddingReplayBuffer (GPU) attivo: capacità={args.replay_buffer_size}")

    memory.per_alpha = args.per_alpha

    # Ripresa da checkpoint
    start_epoch = 0
    global_step = 0
    best_iou = 0.0
    if args.model is not None:
        print(f"\n[INFO] --model fornito: riprendo il training da '{args.model}'")
        checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
        policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        target_net.load_state_dict(checkpoint.get('target_net_state_dict', policy_net.state_dict()))
        start_epoch = checkpoint.get('epoch', 0)
        global_step = checkpoint.get('global_step', 0)
        best_iou = checkpoint.get('best_iou', 0.0)
        print(f"  [✓] Ripresa da epoca {start_epoch}, global_step={global_step}, best_iou={best_iou:.4f}")
        if start_epoch >= args.n_epochs:
            print("[INFO] Epoca di partenza già >= --n-epochs: fine.")
            writer.close()
            return checkpoint_dir, best_iou

    def get_epsilon(step, total_steps):
        decay_steps = max(1, args.epsilon_decay_steps)
        return EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-1.0 * step / decay_steps)

    def get_teacher_prob(epoch):
        ramp = args.curriculum_ramp_frac
        return max(args.teacher_prob_floor, 1.0 - min(1.0, epoch / (args.n_epochs * ramp)))

    def get_tau_iou(epoch):
        if not args.use_tau_curriculum:
            return TAU_IOU
        factor = min(1.0, epoch / (args.n_epochs * args.curriculum_ramp_frac))
        return args.tau_iou_start + (TAU_IOU - args.tau_iou_start) * factor

    def get_per_beta(step, total_steps):
        frac = min(1.0, step / max(total_steps, 1))
        return PER_BETA_START + (PER_BETA_END - PER_BETA_START) * frac

    reward_scaler = RunningMeanStd(device=device)
    n_step_acc = NStepAccumulator(args.batch_size, args.n_step, GAMMA)
    per_enabled = args.use_per
    n_step_gamma = GAMMA ** args.n_step

    # Funzione locale per impacchettare le transizioni n-step
    def stack_ready(ready):
        """Impacchetta una lista di transizioni n-step in tensori batched."""
        s0 = torch.stack([r[0] for r in ready])
        h0 = torch.stack([r[1] for r in ready])
        e0 = torch.stack([r[2] for r in ready])
        a0 = torch.stack([r[3] for r in ready])
        R = torch.tensor([r[4] for r in ready], dtype=torch.float32, device=device)
        ns = torch.stack([r[5] for r in ready])
        nh = torch.stack([r[6] for r in ready])
        ne = torch.stack([r[7] for r in ready])
        term = torch.tensor([r[8] for r in ready], dtype=torch.float32, device=device)
        is_exp = torch.tensor([r[9] for r in ready], dtype=torch.float32, device=device)
        return s0, h0, e0, a0, R, ns, nh, ne, term, is_exp

    print("[INFO] Inizio training..." if args.model is None else "[INFO] Ripresa training...")
    print("=" * 80)

    for epoch in range(start_epoch, args.n_epochs):
        
        reward_sums = torch.zeros(args.batch_size, device=device)   # reward cumulativa per slot
        step_counts = torch.zeros(args.batch_size, device=device)   # contatore step per slot
        print(f"\n[Epoch {epoch + 1}/{args.n_epochs}]")

        p_teacher = get_teacher_prob(epoch)
        writer.add_scalar("Train/Teacher_Prob", p_teacher, epoch)
        print(f"  [INFO] Teacher prob: {p_teacher:.3f}")

        current_tau_iou = get_tau_iou(epoch)
        train_env.set_tau_iou(current_tau_iou)
        writer.add_scalar("Train/Tau_IOU", current_tau_iou, epoch)
        print(f"  [INFO] Soglia trigger corrente: {current_tau_iou:.3f}")

        # Accumulatori per metriche di epoca
        epoch_final_ious = []
        epoch_final_gious = []
        epoch_final_dious = []
        epoch_rewards = []
        epoch_steps = []
        epoch_losses = []

        # Tensori per tracking per-slot
        reward_sums = torch.zeros(args.batch_size, device=device)
        step_counts = torch.zeros(args.batch_size, device=device)

        train_indices = np.random.choice(len(train_ds), size=args.batch_size, replace=True)
        obs = train_env.reset_all(train_indices)

        active_mask = torch.ones(args.batch_size, dtype=torch.bool, device=device)
        last_iou_per_slot = torch.zeros(args.batch_size, device=device)
        epoch_final_ious = []   # solo IoU di fine episodio
        epoch_losses = []
        epoch_step_ious = []    # per mostrare IoU dello step corrente

        pbar = tqdm(range(MAX_STEPS_PER_EPISODE), desc="Training steps", leave=True)

        for step in pbar:
            if not active_mask.any():
                pbar.close()
                break

            epsilon = get_epsilon(global_step, total_training_steps)

            policy_net.set_inference_mode()
            with torch.no_grad():
                q_values = policy_net(obs["regions"], obs["histories"], obs["extra"])
                agent_actions = q_values.argmax(dim=1)

                teacher_mask = torch.rand(args.batch_size, device=device) < p_teacher
                actions = agent_actions.clone()
                if teacher_mask.any():
                    oracle_actions = train_env.compute_oracle_actions()
                    actions[teacher_mask] = oracle_actions[teacher_mask]

                explore_mask = (~teacher_mask) & (torch.rand(args.batch_size, device=device) < epsilon)
                if explore_mask.any():
                    n_explore = int(explore_mask.sum().item())
                    actions[explore_mask] = torch.randint(0, N_ACTIONS, (n_explore,), device=device)

                is_expert_step = teacher_mask.float()

            next_obs, rewards, dones, ious, gious, dious = train_env.step(actions)
            # Aggiorna accumulatori per gli slot attivi
            reward_sums[active_mask] += rewards[active_mask]
            step_counts[active_mask] += 1
            rewards = torch.clamp(rewards, -REWARD_CLIP, REWARD_CLIP)

            step_active_mask = active_mask.clone()
            n_active = int(step_active_mask.sum().item())

            if n_active > 0:
                reward_scaler.update(rewards[step_active_mask])
            scaled_rewards = reward_scaler.scale_only(rewards) if args.use_reward_scaling else rewards

            # Calcola embedding per il buffer (backbone sempre congelato)
            with torch.no_grad():
                step_states = policy_net.embed_regions(obs["regions"])
                step_next_states = policy_net.embed_regions(next_obs["regions"])

            ready = n_step_acc.push_and_pop(
                step_states, obs["histories"], obs["extra"], actions, scaled_rewards,
                step_next_states, next_obs["histories"], next_obs["extra"], dones.float(),
                is_expert_step, active_mask=step_active_mask
            )
            if ready:
                b0, h0, e0, a0, R, ns, nh, ne, term, is_exp = stack_ready(ready)
                memory.push_batch(b0, h0, e0, a0, R, ns, nh, ne, term, is_exp)

            if n_active > 0:
                iou_active = ious[step_active_mask]
                giou_active = gious[step_active_mask]
                diou_active = dious[step_active_mask]

                writer.add_scalar("Train/Step/IoU_mean", iou_active.mean().item(), global_step)
                writer.add_scalar("Train/Step/IoU_std", iou_active.std(unbiased=False).item(), global_step)
                writer.add_scalar("Train/Step/IoU_max", iou_active.max().item(), global_step)
                writer.add_scalar("Train/Step/GIoU_mean", giou_active.mean().item(), global_step)
                writer.add_scalar("Train/Step/GIoU_std", giou_active.std(unbiased=False).item(), global_step)
                writer.add_scalar("Train/Step/GIoU_max", giou_active.max().item(), global_step)
                writer.add_scalar("Train/Step/DIoU_mean", diou_active.mean().item(), global_step)
                writer.add_scalar("Train/Step/DIoU_std", diou_active.std(unbiased=False).item(), global_step)
                writer.add_scalar("Train/Step/DIoU_max", diou_active.max().item(), global_step)

                writer.add_scalar("Train/Step/Epsilon", epsilon, global_step)
                writer.add_scalar("Train/Step/Teacher_Prob", p_teacher, global_step)
                writer.add_scalar("Train/Step/Reward_Std", reward_scaler.std().item(), global_step)

                current_step_iou = iou_active.mean().item()
                epoch_step_ious.append(current_step_iou)
                writer.add_scalar("Train/Step_IoU", current_step_iou, global_step)

            last_iou_per_slot = torch.where(step_active_mask, ious.detach(), last_iou_per_slot)
            newly_done = step_active_mask & dones
            if newly_done.any():
                done_indices = torch.where(newly_done)[0]
                final_iou_new = ious[done_indices]
                final_giou_new = gious[done_indices]
                final_diou_new = dious[done_indices]
                final_reward_new = reward_sums[done_indices]
                final_step_new = step_counts[done_indices]
                
                epoch_final_ious.extend(final_iou_new.cpu().tolist())
                epoch_final_gious.extend(final_giou_new.cpu().tolist())
                epoch_final_dious.extend(final_diou_new.cpu().tolist())
                epoch_rewards.extend(final_reward_new.cpu().tolist())
                epoch_steps.extend(final_step_new.cpu().tolist())

            active_mask = active_mask & (~dones)
            obs = next_obs
            # Aggiornamento Q-network
            if memory.size > args.batch_size * 10:
                for _ in range(args.gradient_steps):
                    if per_enabled:
                        beta = get_per_beta(global_step, total_training_steps)
                        memory.set_beta(beta)
                        b_s, b_hist, b_extra, b_act, b_rew, b_ns, b_nhist, b_nextra, b_term, b_is_expert, \
                            per_idx, is_weights = memory.sample_per(args.batch_size)
                    else:
                        b_s, b_hist, b_extra, b_act, b_rew, b_ns, b_nhist, b_nextra, b_term, b_is_expert = \
                            memory.sample_uniform(args.batch_size)
                        is_weights = torch.ones_like(b_rew)

                    policy_net.train()
                    q_all = policy_net(b_s, b_hist, b_extra)
                    q_vals = q_all.gather(1, b_act.unsqueeze(1)).squeeze(1)

                    with torch.no_grad():
                        next_actions = policy_net(b_ns, b_nhist, b_nextra).argmax(dim=1, keepdim=True)
                        next_q_vals = target_net(b_ns, b_nhist, b_nextra).gather(1, next_actions).squeeze(1)
                        target_q = b_rew + (n_step_gamma * next_q_vals * (1.0 - b_term))

                    td_errors = (q_vals - target_q)
                    per_sample_loss = F.smooth_l1_loss(q_vals, target_q, reduction="none")
                    td_loss = (per_sample_loss * is_weights).mean()

                    # DQfD margin loss
                    if args.use_dqfd_margin and b_is_expert.sum() > 0:
                        margin_matrix = torch.full_like(q_all, args.dqfd_margin)
                        margin_matrix.scatter_(1, b_act.unsqueeze(1), 0.0)
                        margin_target = (q_all + margin_matrix).max(dim=1)[0] - q_vals
                        margin_per_sample = margin_target * b_is_expert
                        margin_loss = margin_per_sample.sum() / b_is_expert.sum().clamp(min=1.0)
                    else:
                        margin_loss = torch.zeros((), device=device)

                    loss = td_loss + args.dqfd_lambda * margin_loss

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy_net.head.parameters(), max_norm=10.0)
                    optimizer.step()
                    scheduler.step()

                    if per_enabled:
                        memory.update_priorities(per_idx, td_errors)

                    with torch.no_grad():
                        for tp, p in zip(target_net.parameters(), policy_net.parameters()):
                            tp.data.mul_(1.0 - args.target_tau).add_(args.target_tau * p.data)

                    epoch_losses.append(loss.item())

                writer.add_scalar("Train/Loss", loss.item(), global_step)
                writer.add_scalar("Train/LR", scheduler.get_last_lr()[0], global_step)

            global_step += 1
            pbar.set_postfix({
                'IoU_step': f"{current_step_iou if n_active>0 else 0:.3f}",
                'Loss': f"{epoch_losses[-1] if epoch_losses else 0:.3f}",
                'Eps': f"{epsilon:.3f}"
            })

        # --- FINE EPOCA: gestione slot ancora attivi (timeout) ---
        if active_mask.any():
            last_boxes = train_env.boxes[active_mask]
            last_gt = train_env.gt_boxes[active_mask]
            last_iou = compute_iou_tensor(last_boxes, last_gt)
            last_giou = compute_giou_tensor(last_boxes, last_gt)
            last_diou = compute_diou_tensor(last_boxes, last_gt)
            last_reward = reward_sums[active_mask]
            last_step = step_counts[active_mask]

            epoch_final_ious.extend(last_iou.cpu().tolist())
            epoch_final_gious.extend(last_giou.cpu().tolist())
            epoch_final_dious.extend(last_diou.cpu().tolist())
            epoch_rewards.extend(last_reward.cpu().tolist())
            epoch_steps.extend(last_step.cpu().tolist())

        # --- Calcolo metriche aggregate dell'epoca ---
        if epoch_final_ious:
            mean_iou = np.mean(epoch_final_ious)
            std_iou = np.std(epoch_final_ious)
            max_iou = np.max(epoch_final_ious)
            mean_giou = np.mean(epoch_final_gious)
            std_giou = np.std(epoch_final_gious)
            max_giou = np.max(epoch_final_gious)
            mean_diou = np.mean(epoch_final_dious)
            std_diou = np.std(epoch_final_dious)
            max_diou = np.max(epoch_final_dious)
            mean_reward = np.mean(epoch_rewards)
            std_reward = np.std(epoch_rewards)
            max_reward = np.max(epoch_rewards)
            mean_steps = np.mean(epoch_steps)
            std_steps = np.std(epoch_steps)
            max_steps = np.max(epoch_steps)
            success_rate = (np.array(epoch_final_ious) >= current_tau_iou).mean()
        else:
            mean_iou = std_iou = max_iou = float('nan')
            mean_giou = std_giou = max_giou = float('nan')
            mean_diou = std_diou = max_diou = float('nan')
            mean_reward = std_reward = max_reward = float('nan')
            mean_steps = std_steps = max_steps = float('nan')
            success_rate = 0.0

        # Best metriche (da env.best_ious e env.best_boxes)
        best_iou_all = train_env.best_ious  # [batch_size]
        best_boxes_all = train_env.best_boxes
        best_giou_all = compute_giou_tensor(best_boxes_all, train_env.gt_boxes)
        best_diou_all = compute_diou_tensor(best_boxes_all, train_env.gt_boxes)
        # Calcolo solo sugli slot che hanno almeno un episodio (tutti)
        mean_best_iou = best_iou_all.mean().item()
        std_best_iou = best_iou_all.std().item()
        max_best_iou = best_iou_all.max().item()
        mean_best_giou = best_giou_all.mean().item()
        std_best_giou = best_giou_all.std().item()
        max_best_giou = best_giou_all.max().item()
        mean_best_diou = best_diou_all.mean().item()
        std_best_diou = best_diou_all.std().item()
        max_best_diou = best_diou_all.max().item()

        # Logging delle metriche di epoca in TensorBoard (Training)
        writer.add_scalar("Epoch/Train_Final_IoU_mean", mean_iou, epoch)
        writer.add_scalar("Epoch/Train_Final_IoU_std", std_iou, epoch)
        writer.add_scalar("Epoch/Train_Final_IoU_max", max_iou, epoch)
        writer.add_scalar("Epoch/Train_Final_GIoU_mean", mean_giou, epoch)
        writer.add_scalar("Epoch/Train_Final_GIoU_std", std_giou, epoch)
        writer.add_scalar("Epoch/Train_Final_GIoU_max", max_giou, epoch)
        writer.add_scalar("Epoch/Train_Final_DIoU_mean", mean_diou, epoch)
        writer.add_scalar("Epoch/Train_Final_DIoU_std", std_diou, epoch)
        writer.add_scalar("Epoch/Train_Final_DIoU_max", max_diou, epoch)
        writer.add_scalar("Epoch/Train_Reward_mean", mean_reward, epoch)
        writer.add_scalar("Epoch/Train_Reward_std", std_reward, epoch)
        writer.add_scalar("Epoch/Train_Reward_max", max_reward, epoch)
        writer.add_scalar("Epoch/Train_Steps_mean", mean_steps, epoch)
        writer.add_scalar("Epoch/Train_Steps_std", std_steps, epoch)
        writer.add_scalar("Epoch/Train_Steps_max", max_steps, epoch)
        writer.add_scalar("Epoch/Train_Success_Rate", success_rate, epoch)

        # Best metriche (training)
        writer.add_scalar("Epoch/Train_Best_IoU_mean", mean_best_iou, epoch)
        writer.add_scalar("Epoch/Train_Best_IoU_std", std_best_iou, epoch)
        writer.add_scalar("Epoch/Train_Best_IoU_max", max_best_iou, epoch)
        writer.add_scalar("Epoch/Train_Best_GIoU_mean", mean_best_giou, epoch)
        writer.add_scalar("Epoch/Train_Best_GIoU_std", std_best_giou, epoch)
        writer.add_scalar("Epoch/Train_Best_GIoU_max", max_best_giou, epoch)
        writer.add_scalar("Epoch/Train_Best_DIoU_mean", mean_best_diou, epoch)
        writer.add_scalar("Epoch/Train_Best_DIoU_std", std_best_diou, epoch)
        writer.add_scalar("Epoch/Train_Best_DIoU_max", max_best_diou, epoch)

        epoch_avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        writer.add_scalar("Epoch/Train_Avg_Loss", epoch_avg_loss, epoch)

        # Stampa a schermo
        print(f"[Epoch {epoch+1}] Train Summary:")
        print(f"  Final IoU  : {mean_iou:.4f} ± {std_iou:.4f} (max {max_iou:.4f})")
        print(f"  Final GIoU : {mean_giou:.4f} ± {std_giou:.4f} (max {max_giou:.4f})")
        print(f"  Final DIoU : {mean_diou:.4f} ± {std_diou:.4f} (max {max_diou:.4f})")
        print(f"  Best IoU   : {mean_best_iou:.4f} ± {std_best_iou:.4f} (max {max_best_iou:.4f})")
        print(f"  Reward     : {mean_reward:.2f} ± {std_reward:.2f} (max {max_reward:.2f})")
        print(f"  Steps      : {mean_steps:.1f} ± {std_steps:.1f} (max {max_steps})")
        print(f"  Success Rate (IoU ≥ {current_tau_iou:.2f}): {success_rate:.4f}")
        print(f"  Avg Loss   : {epoch_avg_loss:.4f}")

        # --- VALIDAZIONE ---
        print(f"\n[Epoch {epoch+1}] Validazione in corso...")
        val_indices = np.random.choice(len(val_ds), size=min(args.batch_size, len(val_ds)), replace=False)
        val_indices = np.pad(val_indices, (0, args.batch_size - len(val_indices)), 'wrap')

        val_metrics = validate(
            val_env, policy_net, val_indices, device, writer, global_step, epoch + 1, args.n_epochs
        )

        # Estraggo le metriche principali
        val_final_iou = val_metrics['avg_iou']
        val_final_best_iou = val_metrics['best_iou']
        val_success_rate = val_metrics['success_rate']
        # (opzionale: posso estrarre anche le altre, ad es. per il checkpoint)
        val_avg_giou = val_metrics['avg_giou']
        val_avg_diou = val_metrics['avg_diou']
        val_mean_reward = val_metrics['mean_reward']
        val_mean_step = val_metrics['mean_step']

        # Log delle metriche principali (alcune già loggate dentro validate, ma posso aggiungere)
        writer.add_scalar("Epoch/Val_Final_IoU", val_final_iou, epoch)
        writer.add_scalar("Epoch/Val_Final_Best_IoU", val_final_best_iou, epoch)
        writer.add_scalar("Epoch/Val_Success_Rate", val_success_rate, epoch)

        # Checkpoint (incluse le nuove metriche)
        checkpoint = {
            'epoch': epoch + 1,
            'global_step': global_step,
            'policy_net_state_dict': policy_net.state_dict(),
            'target_net_state_dict': target_net.state_dict(),
            'train_loss': epoch_avg_loss,
            'val_final_iou': val_final_iou,
            'val_final_best_iou': val_final_best_iou,
            'val_success_rate': val_success_rate,
            'val_avg_giou': val_avg_giou,
            'val_avg_diou': val_avg_diou,
            'val_mean_reward': val_mean_reward,
            'val_mean_step': val_mean_step,
            'best_iou': max(best_iou, val_final_iou),
            'args': args,
            'reward_scaler_state': reward_scaler.state_dict(),
        }

        # Salvataggio checkpoint come prima...
        epoch_checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1:03d}.pt")
        torch.save(checkpoint, epoch_checkpoint_path)
        print(f"  [✓] Checkpoint salvato: {epoch_checkpoint_path}")

        if val_final_iou > best_iou:
            best_iou = val_final_iou
            best_checkpoint_path = os.path.join(checkpoint_dir, "best_checkpoint.pt")
            torch.save(checkpoint, best_checkpoint_path)
            print(f"  [✓] Best checkpoint salvato (IoU best-seen: {best_iou:.4f})")

        latest_checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
        torch.save(checkpoint, latest_checkpoint_path)
        print("=" * 80)
        if device.type == 'mps':
            torch.mps.empty_cache()
            gc.collect()

    print("\n" + "=" * 80)
    print("[INFO] Training completato!")
    print(f"[INFO] Best Validation IoU: {best_iou:.4f}")
    writer.close()
    return checkpoint_dir, best_iou

# ─────────────────────────────────────────────────────────────────────────────
# 8. TEST
# ─────────────────────────────────────────────────────────────────────────────
def load_checkpoint(checkpoint_path, device):
    print(f"[INFO] Caricamento checkpoint da: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint['policy_net_state_dict']
    # Verifica se il modello salvato contiene SpatialAttentionPool
    has_spatial = any(k.startswith('spatial_pool.') for k in state_dict.keys())
    policy_net = PolicyNetwork(
        HISTORY_LENGTH * N_ACTIONS, N_ACTIONS,
        pretrained_backbone=None,
        use_spatial_attention=has_spatial
    ).to(device)
    policy_net.load_state_dict(state_dict)
    print(f"[INFO] Checkpoint caricato da epoca {checkpoint['epoch']}")
    return policy_net, checkpoint

def run_test(args, device, test_ds):
    policy_net, checkpoint = load_checkpoint(args.model, device)
    policy_net.eval()
    policy_net.set_inference_mode()
    env = BatchedActiveLocalizationEnv(test_ds, batch_size=1, device=device)

    output_gif_dir = os.path.join(args.output_root, "gifs")
    os.makedirs(output_gif_dir, exist_ok=True)

    n_test = len(test_ds)
    print(f"[INFO] Generazione GIF sull'intero test set ({n_test} immagini) in: {output_gif_dir}")

    # Liste per metriche aggregate
    final_ious_all = []
    final_gious_all = []
    final_dious_all = []
    best_ious_all = []
    best_gious_all = []
    best_dious_all = []
    total_rewards_all = []
    steps_all = []
    triggered_all = []

    for i in tqdm(range(n_test), desc="Test set"):
        obs = env.reset_all([i])
        frames = []
        final_iou = 0.0
        final_giou = 0.0
        final_diou = 0.0
        best_iou = -1.0
        best_giou = -1.0
        best_diou = -1.0
        total_reward = 0.0
        steps = 0
        triggered = False

        with torch.no_grad():
            for _ in range(MAX_STEPS_PER_EPISODE):
                q_values = policy_net(obs["regions"], obs["histories"], obs["extra"])
                actions = q_values.argmax(dim=1)
                is_trigger = (actions[0].item() == 8)

                # Estrai frame e disegna bounding box (come prima)
                img = env.current_images[0].cpu().numpy().transpose(1, 2, 0)
                img = (img * 255).astype(np.uint8)

                gx, gy, gx2, gy2 = env.gt_boxes[0].cpu().numpy()
                frame = cv2.rectangle(img.copy(), (int(gx), int(gy)), (int(gx2), int(gy2)), (0, 255, 255), 2)

                color = (0, 0, 255) if is_trigger else (0, 255, 0)
                x1, y1, x2, y2 = env.boxes[0].cpu().numpy()
                frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                # Step dell'ambiente (ora restituisce 6 valori)
                next_obs, rewards, dones, ious, gious, dious = env.step(actions)

                # Aggiorna metriche correnti
                final_iou = ious[0].item()
                final_giou = gious[0].item()
                final_diou = dious[0].item()
                total_reward += rewards[0].item()
                steps += 1

                # Aggiorna best values
                if final_iou > best_iou:
                    best_iou = final_iou
                    best_giou = final_giou
                    best_diou = final_diou

                cv2.putText(frame, f"IoU: {final_iou:.3f}", (5, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                cv2.putText(frame, "Giallo=GT  Verde=movimento  Rosso=trigger", (5, img.shape[0] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
                frames.append(frame)

                if dones[0]:
                    triggered = is_trigger
                    break

                obs = next_obs

        # Fine episodio: registra metriche
        final_ious_all.append(final_iou)
        final_gious_all.append(final_giou)
        final_dious_all.append(final_diou)
        best_ious_all.append(best_iou)
        best_gious_all.append(best_giou)
        best_dious_all.append(best_diou)
        total_rewards_all.append(total_reward)
        steps_all.append(steps)
        triggered_all.append(triggered)

        # Creazione GIF (come prima)
        result_label = "TRIGGER" if triggered else "TIMEOUT"
        if frames:
            final_result_frame = frames[-1].copy()
            cv2.putText(final_result_frame, f"FINAL ({result_label}) IoU: {final_iou:.3f}", (5, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            frames += [final_result_frame] * 6

        # Best frame (usa env.best_boxes e env.best_ious)
        best_iou_env = env.best_ious[0].item()
        bx1, by1, bx2, by2 = env.best_boxes[0].cpu().numpy()   # ora (x1,y1,x2,y2)
        best_frame = env.current_images[0].cpu().numpy().transpose(1, 2, 0)
        best_frame = (best_frame * 255).astype(np.uint8)
        gx1, gy1, gx2, gy2 = env.gt_boxes[0].cpu().numpy()     # ora (x1,y1,x2,y2)

        # Disegna GT
        best_frame = cv2.rectangle(best_frame.copy(), (int(gx1), int(gy1)), (int(gx2), int(gy2)),
                                    (0, 255, 255), 2)
        # Disegna best box
        best_frame = cv2.rectangle(best_frame, (int(bx1), int(by1)), (int(bx2), int(by2)),(255, 0, 0), 3)
        cv2.putText(best_frame, f"BEST IoU: {best_iou_env:.3f}", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(best_frame, "Giallo=GT  Blu=miglior box trovato", (5, best_frame.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
        frames += [best_frame] * 8

        gif_name = f"test_{i:04d}_finalIoU_{final_iou:.3f}_bestIoU_{best_iou_env:.3f}.gif"
        imageio.mimsave(os.path.join(output_gif_dir, gif_name), frames, fps=5)

    # Calcolo metriche aggregate
    mean_final_iou = np.mean(final_ious_all)
    std_final_iou = np.std(final_ious_all)
    max_final_iou = np.max(final_ious_all)
    mean_final_giou = np.mean(final_gious_all)
    std_final_giou = np.std(final_gious_all)
    max_final_giou = np.max(final_gious_all)
    mean_final_diou = np.mean(final_dious_all)
    std_final_diou = np.std(final_dious_all)
    max_final_diou = np.max(final_dious_all)

    mean_best_iou = np.mean(best_ious_all)
    std_best_iou = np.std(best_ious_all)
    max_best_iou = np.max(best_ious_all)
    mean_best_giou = np.mean(best_gious_all)
    std_best_giou = np.std(best_gious_all)
    max_best_giou = np.max(best_gious_all)
    mean_best_diou = np.mean(best_dious_all)
    std_best_diou = np.std(best_dious_all)
    max_best_diou = np.max(best_dious_all)

    mean_reward = np.mean(total_rewards_all)
    std_reward = np.std(total_rewards_all)
    max_reward = np.max(total_rewards_all)
    mean_steps = np.mean(steps_all)
    std_steps = np.std(steps_all)
    max_steps = np.max(steps_all)

    success_rate = np.mean(np.array(final_ious_all) >= TAU_IOU)

    # Salvataggio CSV dettagliato
    summary_path = os.path.join(args.output_root, "test_summary.csv")
    with open(summary_path, "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([
            "image_index", "final_iou", "final_giou", "final_diou",
            "best_iou", "best_giou", "best_diou",
            "total_reward", "steps", "triggered"
        ])
        for idx in range(n_test):
            csv_writer.writerow([
                idx,
                f"{final_ious_all[idx]:.4f}",
                f"{final_gious_all[idx]:.4f}",
                f"{final_dious_all[idx]:.4f}",
                f"{best_ious_all[idx]:.4f}",
                f"{best_gious_all[idx]:.4f}",
                f"{best_dious_all[idx]:.4f}",
                f"{total_rewards_all[idx]:.2f}",
                steps_all[idx],
                triggered_all[idx]
            ])

    print(f"[INFO] Riepilogo CSV salvato in: {summary_path}")
    print(f"[INFO] Test completato su {n_test} immagini.")
    print(f"[INFO] ── Metriche di fine episodio ──")
    print(f"[INFO] Final IoU  : media {mean_final_iou:.4f} ± {std_final_iou:.4f}, max {max_final_iou:.4f}")
    print(f"[INFO] Final GIoU : media {mean_final_giou:.4f} ± {std_final_giou:.4f}, max {max_final_giou:.4f}")
    print(f"[INFO] Final DIoU : media {mean_final_diou:.4f} ± {std_final_diou:.4f}, max {max_final_diou:.4f}")
    print(f"[INFO] ── Migliori metriche durante l'episodio ──")
    print(f"[INFO] Best IoU   : media {mean_best_iou:.4f} ± {std_best_iou:.4f}, max {max_best_iou:.4f}")
    print(f"[INFO] Best GIoU  : media {mean_best_giou:.4f} ± {std_best_giou:.4f}, max {max_best_giou:.4f}")
    print(f"[INFO] Best DIoU  : media {mean_best_diou:.4f} ± {std_best_diou:.4f}, max {max_best_diou:.4f}")
    print(f"[INFO] ── Reward e step ──")
    print(f"[INFO] Reward totale : media {mean_reward:.2f} ± {std_reward:.2f}, max {max_reward:.2f}")
    print(f"[INFO] Step          : media {mean_steps:.1f} ± {std_steps:.1f}, max {max_steps}")
    print(f"[INFO] Success rate (IoU ≥ {TAU_IOU:.2f}): {success_rate:.4f} ({int(success_rate*n_test)}/{n_test})")
    print(f"[INFO] Episodi terminati con trigger: {sum(triggered_all)}/{n_test}")

# ─────────────────────────────────────────────────────────────────────────────
# 9. MAIN
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[INFO] Seed fissato a {seed}")

if __name__ == "__main__":
    set_seed()
    parser = argparse.ArgumentParser(description="Active Localization (versione ridotta)")

    # Parametri principali
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--replay-buffer-size", type=int, default=100000,
                        help="Capacità del replay buffer (EmbeddingReplayBuffer, leggero).")
    parser.add_argument("--output-root", type=str, default="./ppo_logs")
    parser.add_argument("--dataset-source", type=str, default=os.environ.get("DATASET_SOURCE", "kaggle"))
    parser.add_argument("--dataset-path", type=str, default=os.environ.get("DATASET_PATH", None))
    parser.add_argument("--kaggle-id", type=str, default=os.environ.get(
        "KAGGLE_DATASET_ID", "pkdarabi/brain-tumor-image-dataset-semantic-segmentation"
    ))
    parser.add_argument("--backbone", type=str, default="resnet18")  # non usato
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--pretrained-backbone", type=str, default=None,
                        help="Path al checkpoint custom del backbone (con spatial_pool opzionale)")

    # Modalità test
    parser.add_argument("--test", action="store_true", help="Esegui test")
    parser.add_argument("--model", type=str, default=None, help="Checkpoint da caricare (training o test)")

    # Iperparametri RL (con valori di default)
    parser.add_argument("--target-tau", type=float, default=0.01)
    parser.add_argument("--gradient-steps", type=int, default=2)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--use-per", action="store_true", default=True)
    parser.add_argument("--no-per", dest="use_per", action="store_false")
    parser.add_argument("--per-alpha", type=float, default=PER_ALPHA)
    parser.add_argument("--n-step", type=int, default=N_STEP)
    parser.add_argument("--use-reward-scaling", action="store_true", default=True)
    parser.add_argument("--no-reward-scaling", dest="use_reward_scaling", action="store_false")
    parser.add_argument("--curriculum-ramp-frac", type=float, default=0.6)
    parser.add_argument("--teacher-prob-floor", type=float, default=0.05)
    parser.add_argument("--use-tau-curriculum", action="store_true", default=True)
    parser.add_argument("--no-tau-curriculum", dest="use_tau_curriculum", action="store_false")
    parser.add_argument("--tau-iou-start", type=float, default=0.4)
    parser.add_argument("--epsilon-decay-steps", type=int, default=2500)
    parser.add_argument("--use-dqfd-margin", action="store_true", default=True)
    parser.add_argument("--no-dqfd-margin", dest="use_dqfd_margin", action="store_false")
    parser.add_argument("--dqfd-margin", type=float, default=0.8)
    parser.add_argument("--dqfd-lambda", type=float, default=1.0)

    args = parser.parse_args()

    # Device
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

    from dataset import get_datasets

    cfg = {
        "dataset": {
            "source": args.dataset_source,
            "kaggle_id": args.kaggle_id,
            "local_path": args.dataset_path,
            "image_size": [224, 224],
            "in_channels": 3,
            "train_ratio": (1501/2145),
            "val_ratio": (429/2145),
            "cache_pairs": False
        },
        "preprocessing": {
            "normalization": "per_image",
            "binarize_mask": True,
            "mask_threshold": 0.5,
            "white_balance": False,
            "clahe": False,
            "denoise": False
        },
        "training": {
            "batch_size": args.batch_size,
            "num_workers": 0
        },
        "seed": 42
    }

    print("[INFO] Caricamento dataset...")
    train_ds, val_ds, test_ds = get_datasets(cfg)


    if args.test:
        if args.model is None:
            raise ValueError("Devi specificare --model per la modalità --test")
        run_test(args, device, test_ds)
    else:
        train(args, device, train_ds, val_ds)