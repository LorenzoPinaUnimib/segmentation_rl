"""
agent_roi.py — DQN Agent per ROI Finding (Agent 1).

Usa uno stato vettoriale compatto (STATE_DIM=20) estratto da environment_roi.
Architettura: MLP più profondo con dueling network per una migliore
stima dei Q-values in task di localizzazione.
"""
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .replay_buffer import ReplayBuffer
from .environment_roi import ROIFinderEnv


# ─────────────────────────────────────────────────────────────────────────────
# Dueling Q-Network
# ─────────────────────────────────────────────────────────────────────────────

class DuelingQNetwork(nn.Module):
    """
    Dueling DQN: separa stima del valore dello stato V(s)
    dal vantaggio per ogni azione A(s,a).
    Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
    """

    def __init__(self, state_dim: int, num_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        # Stream valore
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )
        # Stream vantaggio
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, num_actions),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared = self.shared(x)
        value  = self.value_stream(shared)
        adv    = self.advantage_stream(shared)
        return value + adv - adv.mean(dim=1, keepdim=True)


# ─────────────────────────────────────────────────────────────────────────────
# ROI Finder Agent (Agent 1)
# ─────────────────────────────────────────────────────────────────────────────

class ROIFinderAgent:
    """
    Double Dueling DQN per localizzare la ROI (bounding box).

    Compatibile con environment_roi.ROIFinderEnv.
    Salva/carica checkpoint separati da DQNAgent (agent.py).
    """

    def __init__(self, cfg: dict, device: torch.device):
        rl = cfg.get("rl", {})
        self.device      = device
        self.num_actions = ROIFinderEnv.NUM_ACTIONS
        self.gamma       = rl.get("gamma", 0.99)
        self.batch_size  = rl.get("batch_size", 32)
        self.target_freq = rl.get("target_update_freq", 20)

        # Epsilon greedy
        self.eps       = rl.get("epsilon_start", 1.0)
        self.eps_end   = rl.get("epsilon_end", 0.05)
        self.eps_decay = rl.get("epsilon_decay", 300)

        state_dim  = ROIFinderEnv.STATE_DIM
        hidden_dim = rl.get("roi_hidden_dim", 256)

        self.online = DuelingQNetwork(state_dim, self.num_actions, hidden_dim).to(device)
        self.target = DuelingQNetwork(state_dim, self.num_actions, hidden_dim).to(device)
        self._sync_target()

        self.optimizer = torch.optim.Adam(
            self.online.parameters(),
            lr=rl.get("roi_lr", 5e-4),
            weight_decay=1e-5,
        )
        self.buffer = ReplayBuffer(rl.get("replay_buffer_size", 10000))

        self._episode = 0
        self.checkpoint_dir = cfg["output"]["checkpoint_dir"]

    # ── Epsilon ───────────────────────────────────────────────────────────────

    def update_epsilon(self) -> None:
        self._episode += 1
        self.eps = self.eps_end + (1.0 - self.eps_end) * math.exp(
            -self._episode / self.eps_decay
        )

    # ── Azione ────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        if not greedy and np.random.rand() < self.eps:
            return np.random.randint(self.num_actions)
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q = self.online(s)
        return int(q.argmax(dim=1).item())

    # ── Memoria ───────────────────────────────────────────────────────────────

    def remember(self, state, action, reward, next_state, done) -> None:
        self.buffer.push(state, action, reward, next_state, done)

    # ── Apprendimento ─────────────────────────────────────────────────────────

    def learn(self) -> Optional[float]:
        if not self.buffer.is_ready(self.batch_size):
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        s  = torch.FloatTensor(states).to(self.device)
        a  = torch.LongTensor(actions).to(self.device)
        r  = torch.FloatTensor(rewards).to(self.device)
        s2 = torch.FloatTensor(next_states).to(self.device)
        d  = torch.FloatTensor(dones).to(self.device)

        with torch.no_grad():
            best_actions = self.online(s2).argmax(dim=1, keepdim=True)
            q_next  = self.target(s2).gather(1, best_actions).squeeze(1)
            q_target = r + self.gamma * q_next * (1 - d)

        q_online = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(q_online, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    # ── Target sync ───────────────────────────────────────────────────────────

    def _sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    def maybe_sync_target(self) -> None:
        if self._episode % self.target_freq == 0:
            self._sync_target()

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def save(self, name: str = "best_roi_finder_agent.pth") -> None:
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.checkpoint_dir, name)
        torch.save({
            "online":  self.online.state_dict(),
            "target":  self.target.state_dict(),
            "episode": self._episode,
            "eps":     self.eps,
        }, path)
        print(f"[agent_roi] Salvato ROI Finder → {path}")

    def load(self, name: str = "best_roi_finder_agent.pth") -> None:
        path = os.path.join(self.checkpoint_dir, name)
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device)
            self.online.load_state_dict(ckpt["online"])
            self.target.load_state_dict(ckpt["target"])
            self._episode = ckpt.get("episode", 0)
            self.eps = ckpt.get("eps", self.eps_end)
            print(f"[agent_roi] Caricato ROI Finder da episodio {self._episode}")
        else:
            print(f"[agent_roi] Nessun checkpoint trovato: {path}")
