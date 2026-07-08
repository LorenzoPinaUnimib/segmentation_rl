"""
Codice iper-ottimizzato per AMD Windows (DirectML) con:
- Logging completo TensorBoard (metriche training e validation)
- Validazione ad ogni epoca
- Salvataggio checkpoint ad ogni epoca
- Ambiente vettorizzato (Parallel Batched Environment)
"""

import os
import argparse
import random
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# COSTANTI
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
GAMMA = 0.90  # Discount factor
EPSILON_START = 1.0
EPSILON_END = 0.1

# ─────────────────────────────────────────────────────────────────────────────
# 1. FUNZIONI DI UTILITÀ VETTORIZZATE (Interamente su GPU)
# ─────────────────────────────────────────────────────────────────────────────
def compute_iou_tensor(b1, b2):
    """
    Calcola l'IoU in parallelo per un batch di box.
    b1, b2: Tensori [B, 4] nel formato [xmin, ymin, w, h]
    """
    xi1 = torch.max(b1[:, 0], b2[:, 0])
    yi1 = torch.max(b1[:, 1], b2[:, 1])
    xi2 = torch.min(b1[:, 0] + b1[:, 2], b2[:, 0] + b2[:, 2])
    yi2 = torch.min(b1[:, 1] + b1[:, 3], b2[:, 1] + b2[:, 3])
    
    inter_area = torch.clamp(xi2 - xi1, min=0) * torch.clamp(yi2 - yi1, min=0)
    union_area = (b1[:, 2] * b1[:, 3]) + (b2[:, 2] * b2[:, 3]) - inter_area
    return inter_area / torch.clamp(union_area, min=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# 2. AMBIENTE VETTORIZZATO (Parallel Batched Env)
# ─────────────────────────────────────────────────────────────────────────────
class BatchedActiveLocalizationEnv:
    def __init__(self, pytorch_dataset, batch_size, device):
        self.dataset = pytorch_dataset
        self.num_envs = batch_size
        self.device = device
        
        # Estrai dimensioni di esempio
        sample = self.dataset[0]
        self.C, self.H, self.W = sample["image"].shape
        
        # Allocazione tensori di stato per tutti gli ambienti in parallelo
        self.current_images = torch.zeros((batch_size, self.C, self.H, self.W), 
                                          dtype=torch.float32, device=device)
        self.gt_boxes = torch.zeros((batch_size, 4), dtype=torch.float32, device=device)
        self.boxes = torch.zeros((batch_size, 4), dtype=torch.float32, device=device)
        self.histories = torch.zeros((batch_size, HISTORY_LENGTH * N_ACTIONS), 
                                     dtype=torch.float32, device=device)
        self.current_steps = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.previous_ious = torch.zeros(batch_size, dtype=torch.float32, device=device)
        self.one_hot_buffer = torch.zeros((batch_size, N_ACTIONS), 
                                          dtype=torch.float32, device=device)

    def reset_all(self, indices):
        """Inizializza o reinstanzia tutti gli ambienti in parallelo."""
        for i, idx in enumerate(indices):
            sample = self.dataset[idx % len(self.dataset)]
            self.current_images[i] = sample["image"].to(self.device, non_blocking=True)
            
            # Trasformazione maschera in bounding box
            mask = sample["mask"].squeeze(0)
            pos = torch.where(mask > 0.5)
            if len(pos[0]) > 0:
                ymin, ymax = torch.min(pos[0]), torch.max(pos[0])
                xmin, xmax = torch.min(pos[1]), torch.max(pos[1])
                self.gt_boxes[i] = torch.tensor(
                    [xmin.float(), ymin.float(), 
                     (xmax - xmin).float(), (ymax - ymin).float()], 
                    dtype=torch.float32, device=self.device
                )
            else:
                self.gt_boxes[i] = torch.tensor(
                    [self.W/4, self.H/4, self.W/2, self.H/2], 
                    dtype=torch.float32, device=self.device
                )

        self.boxes[:, 0] = 0
        self.boxes[:, 1] = 0
        self.boxes[:, 2] = self.W
        self.boxes[:, 3] = self.H
        
        # Inizializza le distanze al momento del reset
        curr_centers = self.get_centers(self.boxes)
        gt_centers = self.get_centers(self.gt_boxes)
        self.previous_dists = torch.norm(curr_centers - gt_centers, dim=1)
        
        self.histories.zero_()
        self.current_steps.zero_()
        self.previous_ious = compute_iou_tensor(self.boxes, self.gt_boxes)
        return self._get_obs()

    def _get_obs(self):
        """Ritaglio e Warping parallelo (Batch 4D) direttamente in VRAM."""
        regions = torch.zeros((self.num_envs, self.C, *WARP_SIZE), 
                             dtype=torch.float32, device=self.device)
        
        for i in range(self.num_envs):
            x, y, w, h = self.boxes[i]
            x1 = max(0, int(x.item() - CONTEXT_PIXELS))
            y1 = max(0, int(y.item() - CONTEXT_PIXELS))
            x2 = min(self.W, int(x.item() + w.item() + CONTEXT_PIXELS))
            y2 = min(self.H, int(y.item() + h.item() + CONTEXT_PIXELS))
            
            crop = self.current_images[i, :, y1:y2, x1:x2]
            if crop.shape[1] == 0 or crop.shape[2] == 0:
                x_int, y_int, w_int, h_int = int(x.item()), int(y.item()), int(w.item()), int(h.item())
                crop = self.current_images[i, :, y_int:max(y_int+1, y_int+h_int), 
                                         x_int:max(x_int+1, x_int+w_int)]
            
            if crop.numel() > 0:
                regions[i] = F.interpolate(crop.unsqueeze(0), size=WARP_SIZE, 
                                          mode="bilinear", align_corners=False).squeeze(0)
            
        return {"regions": regions, "histories": self.histories}

    def get_centers(self,boxes):
        # boxes: [B, 4] dove 4 = [xmin, ymin, w, h]
        x_c = boxes[:, 0] + boxes[:, 2] / 2
        y_c = boxes[:, 1] + boxes[:, 3] / 2
        return torch.stack([x_c, y_c], dim=1)
    def step(self, actions):
        """
        Esegue uno step in parallelo per tutti gli ambienti del batch.
        actions: Tensore GPU [B] contenente le azioni discrete
        """
        self.current_steps += 1
        
        # Aggiornamento dello storico azioni vettorizzato
        self.one_hot_buffer.zero_()
        self.one_hot_buffer.scatter_(1, actions.unsqueeze(1), 1.0)
        self.histories = torch.cat([self.histories[:, N_ACTIONS:], self.one_hot_buffer], dim=1)

        rewards = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Applica modifiche geometriche in base all'azione per ogni env
        for i in range(self.num_envs):
            act = actions[i].item()
            x, y, w, h = self.boxes[i].clone()
            aw, ah = ALPHA * w, ALPHA * h
            
            if act == 0: x = x + aw
            elif act == 1: x = x - aw
            elif act == 2: y = y - ah
            elif act == 3: y = y + ah
            elif act == 4: x = x - aw; y = y - ah; w = w + 2*aw; h = h + 2*ah
            elif act == 5: x = x + aw; y = y + ah; w = w - 2*aw; h = h - 2*ah
            elif act == 6: x = x - aw; w = w + 2*aw
            elif act == 7: y = y - ah; h = h + 2*ah
            elif act == 8:
                terminated[i] = True
                
            self.boxes[i, 0] = torch.clamp(x, 0, self.W - 1)
            self.boxes[i, 1] = torch.clamp(y, 0, self.H - 1)
            self.boxes[i, 2] = torch.clamp(w, min=10, max=self.W)
            self.boxes[i, 3] = torch.clamp(h, min=10, max=self.H)

        # Calcolo dei reward vettorizzato globale
        new_ious = compute_iou_tensor(self.boxes, self.gt_boxes)
        
        # Assegnazione Condizionale Vettorizzata
        trigger_mask = (actions == 8)
        correct_trigger = trigger_mask & (self.previous_ious >= TAU_IOU)
        wrong_trigger = trigger_mask & (self.previous_ious < TAU_IOU)
        movement = ~trigger_mask
        improved_movement = movement & (new_ious > self.previous_ious)
        regressed_movement = movement & (new_ious <= self.previous_ious)
        truncated = (self.current_steps >= MAX_STEPS_PER_EPISODE)

        rewards[correct_trigger] = TRIGGER_REWARD
        rewards[wrong_trigger] = -TRIGGER_REWARD
        rewards[improved_movement] = REWARD_POSITIVE- 0.01
        rewards[regressed_movement] = - 0.03
        rewards[truncated] = -3.0 * (1 - new_ious[truncated])
        

        # 2. Calcola la Distanza Reward (LA INTEGRAZIONE)
        curr_centers = self.get_centers(self.boxes)
        gt_centers = self.get_centers(self.gt_boxes)
        curr_dists = torch.norm(curr_centers - gt_centers, dim=1)
        
        # La formula (vecchia - nuova) premia la diminuzione di distanza
        # Moltiplica per uno scalare (es 0.1) perché questa reward 
        # verrà data AD OGNI step e non deve sovrastare il premio finale.
        dist_reward = (self.previous_dists - curr_dists) * 0.1
        
        # Integrazione: SOMMA alla reward esistente
        rewards += dist_reward
        
        # 3. Update reference per il prossimo step
        self.previous_dists = curr_dists.clone()

        self.previous_ious = new_ious
        dones = terminated | truncated

        return self._get_obs(), rewards, dones, new_ious


# ─────────────────────────────────────────────────────────────────────────────
# 3. REPLAY BUFFER
# ─────────────────────────────────────────────────────────────────────────────
class GPUReplayBuffer:
    def __init__(self, capacity, embed_dim, history_dim, device):
        self.device = device
        self.capacity, self.pos, self.size = capacity, 0, 0
        self.embeds = torch.zeros((capacity, embed_dim), dtype=torch.float32, device=device)
        self.next_embeds = torch.zeros((capacity, embed_dim), dtype=torch.float32, device=device)
        self.histories = torch.zeros((capacity, history_dim), dtype=torch.float32, device=device)
        self.next_histories = torch.zeros((capacity, history_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros(capacity, dtype=torch.long, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.terminals = torch.zeros(capacity, dtype=torch.float32, device=device)

    def push_batch(self, embeds, histories, actions, rewards, next_embeds, next_histories, terminals):
        b_size = embeds.shape[0]
        for i in range(b_size):
            idx = self.pos
            self.embeds[idx] = embeds[i]
            self.next_embeds[idx] = next_embeds[i]
            self.histories[idx] = histories[i]
            self.next_histories[idx] = next_histories[i]
            self.actions[idx] = actions[i]
            self.rewards[idx] = rewards[i]
            self.terminals[idx] = terminals[i]
            self.pos = (idx + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.embeds[idx], self.histories[idx], self.actions[idx], self.rewards[idx],
            self.next_embeds[idx], self.next_histories[idx], self.terminals[idx]
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. MODELLI
# ─────────────────────────────────────────────────────────────────────────────
class QNetwork(nn.Module):
    def __init__(self, embed_dim, history_dim, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim + history_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n_actions)
        
    def forward(self, embeds, histories):
        x = torch.cat([embeds, histories], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ─────────────────────────────────────────────────────────────────────────────
# 5. VALIDAZIONE
# ─────────────────────────────────────────────────────────────────────────────
def validate(env, backbone, q_net, val_indices, device, writer, global_step, epoch):
    """Esegui validazione su un set di validazione."""
    q_net.eval()
    backbone.eval()
    
    obs = env.reset_all(val_indices)
    
    episode_rewards = torch.zeros(env.num_envs, device=device)
    episode_ious = torch.zeros((env.num_envs, MAX_STEPS_PER_EPISODE), device=device)
    final_ious = torch.zeros(env.num_envs, device=device)
    
    with torch.no_grad():
        for step in range(MAX_STEPS_PER_EPISODE):
            embs = backbone(obs["regions"])
            q_values = q_net(embs, obs["histories"])
            actions = q_values.argmax(dim=1)
            
            next_obs, rewards, dones, ious = env.step(actions)
            
            episode_rewards += rewards
            episode_ious[:, step] = ious
            obs = next_obs
            
            # Per gli ambienti terminati, registra l'IoU finale
            final_ious[dones] = ious[dones]
    
    avg_reward = episode_rewards.mean().item()
    avg_iou = episode_ious.mean().item()
    final_avg_iou = final_ious.mean().item()
    max_iou = episode_ious.max().item()
    
    print(f"[Epoch {epoch}] Validation - Avg Reward: {avg_reward:.4f}, Avg IoU: {avg_iou:.4f}, Final IoU: {final_avg_iou:.4f}, Max IoU: {max_iou:.4f}")
    
    writer.add_scalar("Validation/Avg_Reward", avg_reward, global_step)
    writer.add_scalar("Validation/Avg_IoU", avg_iou, global_step)
    writer.add_scalar("Validation/Final_Avg_IoU", final_avg_iou, global_step)
    writer.add_scalar("Validation/Max_IoU", max_iou, global_step)
    
    q_net.train()
    return avg_reward, avg_iou, final_avg_iou, max_iou


# ─────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# 6. TRAINING LOOP VETTORIZZATO COMPLETO
# ─────────────────────────────────────────────────────────────────────────────
def train(args, device):
    """Training loop principale con validazione e checkpoint saving."""
    from dataset import get_datasets
    
    # Configurazione dataset
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
    
    # Carica dataset
    print("[INFO] Caricamento dataset...")
    train_ds, val_ds, test_ds = get_datasets(cfg)
    
    # Setup logging e checkpoint directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(args.output_root, "logs", timestamp)
    checkpoint_dir = os.path.join(args.output_root, "checkpoints", timestamp)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    writer = SummaryWriter(log_dir)
    print(f"[INFO] Log directory: {log_dir}")
    print(f"[INFO] Checkpoint directory: {checkpoint_dir}")
    
    # Inizializza ambiente batched per training e validation
    print("[INFO] Inizializzazione ambienti batched...")
    train_env = BatchedActiveLocalizationEnv(train_ds, batch_size=args.batch_size, device=device)
    val_env = BatchedActiveLocalizationEnv(val_ds, batch_size=args.batch_size, device=device)
    
    # Carica backbone (ResNet18 con feature pre-trained)
    print("[INFO] Caricamento backbone ResNet18...")
    from torchvision.models import resnet18
    backbone = resnet18(pretrained=True)
    backbone.fc = nn.Identity() 
    backbone = backbone.to(device)
    backbone.eval()  # Mantieni in eval mode
    embed_dim = 512
    
    # Crea Q-Network
    print("[INFO] Creazione Q-Network...")
    q_net = QNetwork(embed_dim, HISTORY_LENGTH * N_ACTIONS, N_ACTIONS).to(device)
    optimizer = optim.Adam(q_net.parameters(), lr=args.learning_rate)
    
    # Replay buffer
    memory = GPUReplayBuffer(args.replay_buffer_size, embed_dim, HISTORY_LENGTH * N_ACTIONS, device)
    
    # Schedule epsilon-greedy
    def get_epsilon(step, total_steps):
        return EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-1.0 * step / (0.5 * total_steps))
    
    total_training_steps = args.n_epochs * MAX_STEPS_PER_EPISODE
    global_step = 0
    best_val_iou = 0.0
    
    print("[INFO] Inizio training...")
    print(f"[INFO] Epochs: {args.n_epochs}, Batch Size: {args.batch_size}, Steps per epoch: {MAX_STEPS_PER_EPISODE}")
    print(f"[INFO] Learning Rate: {args.learning_rate}, Replay Buffer Size: {args.replay_buffer_size}")
    print("=" * 80)
    
    # ─────────────────────────────────────────────────────────────────────────
    # MAIN TRAINING LOOP
    # ─────────────────────────────────────────────────────────────────────────
    for epoch in range(args.n_epochs):
        print(f"\n[Epoch {epoch + 1}/{args.n_epochs}]")
        
        # Reset train environment con nuovi indici
        train_indices = np.random.choice(len(train_ds), size=args.batch_size, replace=True)
        obs = train_env.reset_all(train_indices)
        
        epoch_rewards = []
        epoch_ious = []
        epoch_losses = []
        epoch_q_values = []
        
        # Loop di training per ogni step dell'episodio
        pbar = tqdm(range(MAX_STEPS_PER_EPISODE), desc="Training steps", leave=True)
        
        for step in pbar:
            # Calcola epsilon attuale
            epsilon = get_epsilon(global_step, total_training_steps)
            
            with torch.no_grad():
                # Feature extraction massivo su GPU
                embs = backbone(obs["regions"])
                q_values = q_net(embs, obs["histories"])
                
                # Epsilon-greedy action selection (vettorizzato)
                if random.random() < epsilon:
                    actions = torch.randint(0, N_ACTIONS, (args.batch_size,), device=device)
                else:
                    actions = q_values.argmax(dim=1)
            
            # Esegui step nell'ambiente batched
            next_obs, rewards, dones, ious = train_env.step(actions)
            
            # Salva nel replay buffer
            with torch.no_grad():
                next_embs = backbone(next_obs["regions"])
            memory.push_batch(embs, obs["histories"], actions, rewards, 
                            next_embs, next_obs["histories"], dones.float())
            
            # Logging metriche di step
            epoch_rewards.append(rewards.mean().item())
            epoch_ious.append(ious.mean().item())
            avg_q = q_values.max(dim=1)[0].mean().item()
            epoch_q_values.append(avg_q)
            
            writer.add_scalar("Train/Step_Reward_Mean", rewards.mean().item(), global_step)
            writer.add_scalar("Train/Step_IoU_Mean", ious.mean().item(), global_step)
            writer.add_scalar("Train/Step_Q_Value_Max", avg_q, global_step)
            writer.add_scalar("Train/Epsilon", epsilon, global_step)
            writer.add_scalar("Train/Buffer_Size", memory.size, global_step)
            
            # ─────────────────────────────────────────────────────────────────
            # TRAINING DEL Q-NETWORK
            # ─────────────────────────────────────────────────────────────────
            if memory.size > args.batch_size * 2:
                # Sample dal replay buffer
                b_emb, b_hist, b_act, b_rew, b_nemb, b_nhist, b_term = memory.sample(args.batch_size)
                
                # Compute Q-values attuali
                q_vals = q_net(b_emb, b_hist).gather(1, b_act.unsqueeze(1)).squeeze(1)
                
                # Compute target Q-values (senza gradient)
                with torch.no_grad():
                    next_q_vals = q_net(b_nemb, b_nhist)
                    max_next_q = next_q_vals.max(dim=1)[0]
                    target_q = b_rew + (GAMMA * max_next_q * (1.0 - b_term))
                
                # MSE Loss
                loss = F.mse_loss(q_vals, target_q)
                
                # Backward pass
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=10.0)
                optimizer.step()
                
                epoch_losses.append(loss.item())
                writer.add_scalar("Train/Loss", loss.item(), global_step)
                writer.add_scalar("Train/Target_Q_Mean", target_q.mean().item(), global_step)
            
            # Update observation
            obs = next_obs
            global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'Reward': f"{np.mean(epoch_rewards[-10:]):.3f}",
                'IoU': f"{np.mean(epoch_ious[-10:]):.3f}",
                'Loss': f"{np.mean(epoch_losses[-10:]) if epoch_losses else 0:.3f}",
                'Epsilon': f"{epsilon:.3f}"
            })
        
        # ─────────────────────────────────────────────────────────────────────
        # EPOCH SUMMARY METRICS (TRAINING)
        # ─────────────────────────────────────────────────────────────────────
        epoch_avg_reward = np.mean(epoch_rewards)
        epoch_avg_iou = np.mean(epoch_ious)
        epoch_avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        epoch_avg_q = np.mean(epoch_q_values)
        
        writer.add_scalar("Epoch/Train_Avg_Reward", epoch_avg_reward, epoch)
        writer.add_scalar("Epoch/Train_Avg_IoU", epoch_avg_iou, epoch)
        writer.add_scalar("Epoch/Train_Avg_Loss", epoch_avg_loss, epoch)
        writer.add_scalar("Epoch/Train_Avg_Q_Value", epoch_avg_q, epoch)
        writer.add_scalar("Epoch/Memory_Size", memory.size, epoch)
        
        print(f"[Epoch {epoch + 1}] Train Summary:")
        print(f"  Avg Reward: {epoch_avg_reward:.4f}")
        print(f"  Avg IoU: {epoch_avg_iou:.4f}")
        print(f"  Avg Loss: {epoch_avg_loss:.4f}")
        print(f"  Avg Q-Value: {epoch_avg_q:.4f}")
        print(f"  Epsilon: {epsilon:.4f}")
        
        # ─────────────────────────────────────────────────────────────────────
        # VALIDATION
        # ─────────────────────────────────────────────────────────────────────
        print(f"\n[Epoch {epoch + 1}] Validazione in corso...")
        val_indices = np.random.choice(len(val_ds), size=min(args.batch_size, len(val_ds)), replace=False)
        val_indices = np.pad(val_indices, (0, args.batch_size - len(val_indices)), 'wrap')
        
        val_reward, val_iou, val_final_iou, val_max_iou = validate(
            val_env, backbone, q_net, val_indices, device, writer, global_step, epoch + 1
        )
        
        writer.add_scalar("Epoch/Val_Avg_Reward", val_reward, epoch)
        writer.add_scalar("Epoch/Val_Avg_IoU", val_iou, epoch)
        writer.add_scalar("Epoch/Val_Final_IoU", val_final_iou, epoch)
        writer.add_scalar("Epoch/Val_Max_IoU", val_max_iou, epoch)
        
        # ─────────────────────────────────────────────────────────────────────
        # CHECKPOINT SAVING
        # ─────────────────────────────────────────────────────────────────────
        checkpoint = {
            'epoch': epoch + 1,
            'global_step': global_step,
            'q_net_state_dict': q_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'backbone_state_dict': backbone.state_dict(),
            'train_loss': epoch_avg_loss,
            'train_reward': epoch_avg_reward,
            'train_iou': epoch_avg_iou,
            'val_reward': val_reward,
            'val_iou': val_iou,
            'val_final_iou': val_final_iou,
            'val_max_iou': val_max_iou,
            'args': args
        }
        
        # Salva checkpoint di ogni epoca
        epoch_checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1:03d}.pt")
        torch.save(checkpoint, epoch_checkpoint_path)
        print(f"  [✓] Checkpoint salvato: {epoch_checkpoint_path}")
        
        # Salva best checkpoint
        if val_final_iou > best_val_iou:
            best_val_iou = val_final_iou
            best_checkpoint_path = os.path.join(checkpoint_dir, "best_checkpoint.pt")
            torch.save(checkpoint, best_checkpoint_path)
            print(f"  [✓] Best checkpoint salvato (IoU: {best_val_iou:.4f}): {best_checkpoint_path}")
        
        # Salva latest checkpoint
        latest_checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
        torch.save(checkpoint, latest_checkpoint_path)
        
        print("=" * 80)
    
    # ─────────────────────────────────────────────────────────────────────────
    # TRAINING COMPLETE
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("[INFO] Training completato!")
    print(f"[INFO] Best Validation IoU: {best_val_iou:.4f}")
    print(f"[INFO] Log directory: {log_dir}")
    print(f"[INFO] Checkpoint directory: {checkpoint_dir}")
    print(f"[INFO] Total global steps: {global_step}")
    print("=" * 80)
    
    writer.close()
    return checkpoint_dir, best_val_iou


# ─────────────────────────────────────────────────────────────────────────────
# 7. UTILITY FUNCTIONS PER LOADING CHECKPOINT
# ─────────────────────────────────────────────────────────────────────────────
def load_checkpoint(checkpoint_path, device):
    """Carica un checkpoint precedente."""
    print(f"[INFO] Caricamento checkpoint da: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Ricostruisci il modello
    embed_dim = 512
    q_net = QNetwork(embed_dim, HISTORY_LENGTH * N_ACTIONS, N_ACTIONS).to(device)
    q_net.load_state_dict(checkpoint['q_net_state_dict'])
    
    from torchvision.models import resnet18
    backbone = resnet18(pretrained=False)
    backbone.fc = nn.Identity() 
    backbone = backbone.to(device)
    backbone.load_state_dict(checkpoint['backbone_state_dict'])
    
    print(f"[INFO] Checkpoint caricato da epoca {checkpoint['epoch']}")
    return q_net, backbone, checkpoint

import cv2
import imageio

def run_test(args, device):
    from dataset import get_datasets
    
    # 1. Caricamento Modello
    q_net, backbone, checkpoint = load_checkpoint(args.model, device)
    q_net.eval()
    backbone.eval()
    
    # 2. Caricamento Dataset (usa solo il test set)
    cfg = {"dataset": {"source": "kaggle", "kaggle_id": args.kaggle_id, "local_path": args.dataset_path, "image_size": [224, 224], "in_channels": 3, "train_ratio": (1501/2145), "val_ratio": (429/2145), "cache_pairs": False},
           "preprocessing": {"normalization": "per_image", "binarize_mask": True, "mask_threshold": 0.5, "white_balance": False, "clahe": False, "denoise": False},
           "training": {"batch_size": 512, "num_workers": 0},
           "output": {"root": args.output_root},
           "seed": 42,
           "backbone": args.backbone,
    }
    _, _, test_ds = get_datasets(cfg)
    
    # Setup ambiente (batch_size=1 per fare test su singola immagine alla volta)
    env = BatchedActiveLocalizationEnv(test_ds, batch_size=1, device=device)
    
    output_gif_dir = os.path.join(args.output_root, "gifs")
    os.makedirs(output_gif_dir, exist_ok=True)
    
    print(f"[INFO] Generazione GIF in: {output_gif_dir}")
    
    # Test su un numero limitato di immagini (es. prime 10)
    for i in range(min(10, len(test_ds))):
        print(f"Generazione GIF per immagine {i}...")
        obs = env.reset_all([i]) # Reset dell'env per l'immagine i
        frames = []
        
        with torch.no_grad():
            for step in range(MAX_STEPS_PER_EPISODE):
                # Ottieni predizione
                embs = backbone(obs["regions"])
                q_values = q_net(embs, obs["histories"])
                actions = q_values.argmax(dim=1)
                
                # Visualizzazione: prendi l'immagine corrente
                # Assumiamo che env.current_images sia [1, 3, H, W]
                img = env.current_images[0].cpu().numpy().transpose(1, 2, 0)
                img = (img * 255).astype(np.uint8) # denormalizza se necessario
                
                # Disegna Box (x, y, w, h)
                x, y, w, h = env.boxes[0].cpu().numpy()
                color = (0, 255, 0) if actions[0] != 8 else (0, 0, 255) # Rosso se trigger, Verde se movimento
                img = cv2.rectangle(img.copy(), (int(x), int(y)), (int(x+w), int(y+h)), color, 2)
                frames.append(img)
                
                # Step env
                obs, _, dones, _ = env.step(actions)
                if dones[0]: break
        
        # Salva GIF
        imageio.mimsave(os.path.join(output_gif_dir, f"test_{i}.gif"), frames, fps=5)
    
    print("[INFO] Test completato.")

# ─────────────────────────────────────────────────────────────────────────────
# 8. MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    
    
    # Argument parser
    parser = argparse.ArgumentParser(
        description="Active Localization con Q-Learning su AMD Windows (DirectML)"
    )
    parser.add_argument("--n-epochs", type=int, default=10,
                       help="Numero di epoche (default: 10)")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Dimensione del batch (default: 32)")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                       help="Learning rate (default: 1e-4)")
    parser.add_argument("--replay-buffer-size", type=int, default=100000,
                       help="Capacità replay buffer (default: 100000)")
    
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.1)
    parser.add_argument("--output-root", type=str, default="./ppo_logs")
    parser.add_argument("--dataset-source", type=str, default=os.environ.get("DATASET_SOURCE", "kaggle"))
    parser.add_argument("--dataset-path", type=str, default=os.environ.get("DATASET_PATH", None))
    parser.add_argument("--kaggle-id", type=str, default=os.environ.get(
        "KAGGLE_DATASET_ID", "pkdarabi/brain-tumor-image-dataset-semantic-segmentation"
    ))
    parser.add_argument("--backbone", type=str, default="vgg16")
    parser.add_argument("--test", action="store_true", help="Esegui in modalità test")
    parser.add_argument("--model", type=str, default=None, help="Path al file .pt del modello")
    parser.add_argument("--device", type=str, default="auto", help="Device da utilizzare (default: cuda se disponibile, altrimenti cpu)")
    args = parser.parse_args()
    
    if args.device == "auto":
        try:
            import torch_directml
            device = torch_directml.device()
            print(f"[INFO] DirectML Device: {device}")
        except ImportError:
            print("[WARNING] torch_directml non disponibile. Uso CUDA se disponibile...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[INFO] Device: {device}")
    
    if args.test:
        if args.model is None:
            raise ValueError("Devi specificare --model per la modalità --test")
        run_test(args, device)
    else:
        train(args, device)