"""
agent_refine.py — DQN Agent per ROI Refinement (Agent 2).

Usa lo stesso state_dim di MaskRefinementEnv (21) ed è compatibile
con environment_refine.ROIRefinementEnv.

Architettura: MLP standard identica a DQNAgent (agent.py) per permettere
un eventuale pre-training dai pesi dell'agente originale.
"""
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .replay_buffer import ReplayBuffer
from .environment_refine import ROIRefinementEnv


# ─────────────────────────────────────────────────────────────────────────────
# Q-Network (stessa architettura di agent.py QNetwork)
# ─────────────────────────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, num_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, num_actions),
        )
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# ROI Refinement Agent (Agent 2)
# ─────────────────────────────────────────────────────────────────────────────

class ROIRefinementAgent:
    """
    Double DQN per il raffinamento della maschera sul crop ROI.

    Compatibile con environment_refine.ROIRefinementEnv.
    Può opzionalmente essere inizializzato dai pesi di DQNAgent (agent.py)
    per transfer learning.
    """

    def __init__(self, cfg: dict, device: torch.device):
        rl = cfg.get("rl", {})
        self.device      = device
        self.num_actions = ROIRefinementEnv.NUM_ACTIONS
        self.gamma       = rl.get("gamma", 0.99)
        self.batch_size  = rl.get("batch_size", 32)
        self.target_freq = rl.get("target_update_freq", 20)

        self.eps       = rl.get("epsilon_start", 1.0)
        self.eps_end   = rl.get("epsilon_end", 0.05)
        self.eps_decay = rl.get("epsilon_decay", 200)

        state_dim  = ROIRefinementEnv.STATE_DIM
        hidden_dim = rl.get("hidden_dim", 128)

        self.online = QNetwork(state_dim, self.num_actions, hidden_dim).to(device)
        self.target = QNetwork(state_dim, self.num_actions, hidden_dim).to(device)
        self._sync_target()

        self.optimizer = torch.optim.Adam(
            self.online.parameters(), lr=rl.get("lr", 1e-4)
        )
        self.buffer = ReplayBuffer(rl.get("replay_buffer_size", 5000))

        self._episode = 0
        self.checkpoint_dir = cfg["output"]["checkpoint_dir"]

    # ── Transfer learning opzionale ───────────────────────────────────────────

    def load_from_base_agent(self, base_agent_path: str) -> bool:
        """
        Tenta di caricare i pesi dall'agente di raffinamento base (agent.py).
        Utile per transfer learning se le architetture coincidono.
        """
        if not os.path.exists(base_agent_path):
            return False
        try:
            ckpt = torch.load(base_agent_path, map_location=self.device)
            self.online.load_state_dict(ckpt["online"], strict=False)
            self.target.load_state_dict(ckpt["target"], strict=False)
            print(f"[agent_refine] Transfer learning da {base_agent_path}")
            return True
        except Exception as e:
            print(f"[agent_refine] Transfer learning fallito: {e}")
            return False

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

    def save(self, name: str = "best_roi_refiner_agent.pth") -> None:
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.checkpoint_dir, name)
        torch.save({
            "online":  self.online.state_dict(),
            "target":  self.target.state_dict(),
            "episode": self._episode,
            "eps":     self.eps,
        }, path)
        print(f"[agent_refine] Salvato ROI Refiner → {path}")

    def load(self, name: str = "best_roi_refiner_agent.pth") -> None:
        path = os.path.join(self.checkpoint_dir, name)
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device)
            self.online.load_state_dict(ckpt["online"])
            self.target.load_state_dict(ckpt["target"])
            self._episode = ckpt.get("episode", 0)
            self.eps = ckpt.get("eps", self.eps_end)
            print(f"[agent_refine] Caricato ROI Refiner da episodio {self._episode}")
        else:
            print(f"[agent_refine] Nessun checkpoint trovato: {path}")
