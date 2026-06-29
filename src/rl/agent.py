"""
agent.py — DQN agent for mask refinement.

Policy network: MLP(state_dim → hidden → hidden → num_actions)
Training: double DQN with target network, epsilon-greedy exploration,
          experience replay, Huber loss.
"""
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .replay_buffer import ReplayBuffer
from .environment import MaskRefinementEnv


# ─────────────────────────────────────────────────────────────────────────────
# Q-Network
# ─────────────────────────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    """
    MLP Q-network.
    Input: state vector (STATE_DIM,)
    Output: Q-values for each action (num_actions,)
    """

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
# DQN Agent
# ─────────────────────────────────────────────────────────────────────────────

class DQNAgent:
    """
    Double DQN agent.
    - online network  : updated every step
    - target network  : hard-updated every target_update_freq episodes
    - epsilon-greedy  : linear decay over epsilon_decay episodes
    """

    def __init__(self, cfg: dict, device: torch.device):
        rl = cfg.get("rl", {})
        self.device      = device
        self.num_actions = rl.get("num_actions", 8)
        self.gamma       = rl.get("gamma", 0.99)
        self.batch_size  = rl.get("batch_size", 32)
        self.target_freq = rl.get("target_update_freq", 20)

        # Epsilon
        self.eps         = rl.get("epsilon_start", 1.0)
        self.eps_end     = rl.get("epsilon_end", 0.05)
        self.eps_decay   = rl.get("epsilon_decay", 200)

        state_dim   = MaskRefinementEnv.STATE_DIM
        hidden_dim  = rl.get("hidden_dim", 128)

        self.online = QNetwork(state_dim, self.num_actions, hidden_dim).to(device)
        self.target = QNetwork(state_dim, self.num_actions, hidden_dim).to(device)
        self._sync_target()

        self.optimizer = torch.optim.Adam(
            self.online.parameters(), lr=rl.get("lr", 1e-4)
        )
        self.buffer = ReplayBuffer(rl.get("replay_buffer_size", 5000))

        self._episode = 0
        self.checkpoint_dir = cfg["output"]["checkpoint_dir"]

    # ── Epsilon ───────────────────────────────────────────────────────────────

    def update_epsilon(self) -> None:
        self._episode += 1
        # Exponential decay
        self.eps = self.eps_end + (1.0 - self.eps_end) * math.exp(
            -self._episode / self.eps_decay
        )

    # ── Action selection ──────────────────────────────────────────────────────

    @torch.no_grad()
    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        if not greedy and np.random.rand() < self.eps:
            return np.random.randint(self.num_actions)
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q = self.online(s)
        return int(q.argmax(dim=1).item())

    # ── Memory ───────────────────────────────────────────────────────────────

    def remember(self, state, action, reward, next_state, done) -> None:
        self.buffer.push(state, action, reward, next_state, done)

    # ── Training step ─────────────────────────────────────────────────────────

    def learn(self) -> Optional[float]:
        if not self.buffer.is_ready(self.batch_size):
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        s  = torch.FloatTensor(states).to(self.device)
        a  = torch.LongTensor(actions).to(self.device)
        r  = torch.FloatTensor(rewards).to(self.device)
        s2 = torch.FloatTensor(next_states).to(self.device)
        d  = torch.FloatTensor(dones).to(self.device)

        # Online → select best action
        with torch.no_grad():
            best_actions = self.online(s2).argmax(dim=1, keepdim=True)
            # Target → evaluate
            q_next = self.target(s2).gather(1, best_actions).squeeze(1)
            q_target = r + self.gamma * q_next * (1 - d)

        q_online = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(q_online, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    # ── Target network ────────────────────────────────────────────────────────

    def _sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    def maybe_sync_target(self) -> None:
        if self._episode % self.target_freq == 0:
            self._sync_target()

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def save(self, name: str = "best_rl_agent.pth") -> None:
        path = os.path.join(self.checkpoint_dir, name)
        torch.save({
            "online":  self.online.state_dict(),
            "target":  self.target.state_dict(),
            "episode": self._episode,
            "eps":     self.eps,
        }, path)

    def load(self, name: str = "best_rl_agent.pth") -> None:
        path = os.path.join(self.checkpoint_dir, name)
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device)
            self.online.load_state_dict(ckpt["online"])
            self.target.load_state_dict(ckpt["target"])
            self._episode = ckpt.get("episode", 0)
            self.eps = ckpt.get("eps", self.eps_end)
            print(f"[agent] Loaded RL agent from episode {self._episode}")
        else:
            print(f"[agent] No checkpoint found at {path}")
