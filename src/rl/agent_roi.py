"""
agent_roi.py — DQN Agent per ROI Finding (Agent 1).

Usa uno stato immagine [2, S, S] (immagine + maschera box) prodotto da
environment_roi.ROIFinderEnv. Architettura: CNN dueling leggera (al posto
del precedente MLP su 20 feature scalari), pensata per imparare a
"guardare" l'immagine e il box correnti invece di basarsi su feature
ingegnerizzate a mano.

Ottimizzazioni per la velocità (la CNN gira per ogni step di ogni episodio,
quindi deve restare economica):
  - input piccolo (default 48×48, configurabile via rl.roi_cnn_size)
  - solo 3 conv layer con stride 2 (nessun pooling separato → meno operazioni)
  - GroupNorm al posto di BatchNorm: stabile anche con batch_size=1
    (necessario per act() che valuta un singolo stato per volta), evita la
    gestione esplicita di model.train()/eval() per le statistiche running
  - AdaptiveAvgPool2d(1) prima delle teste FC: le teste restano piccole
    indipendentemente dalla risoluzione spaziale di input
"""
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from .environment_roi import ROIFinderEnv


# ─────────────────────────────────────────────────────────────────────────────
# Dueling CNN Q-Network
# ─────────────────────────────────────────────────────────────────────────────

class DuelingCNN(nn.Module):
    """
    Dueling DQN convoluzionale: separa la stima del valore dello stato V(s)
    dal vantaggio per ogni azione A(s,a).
    Q(s,a) = V(s) + A(s,a) - mean(A(s,:))

    `gradcam_layer` è un nn.Identity() posizionato subito dopo l'ultima
    feature map convoluzionale: serve come punto di aggancio per gli hook
    di Grad-CAM (vedi src/utils/gradcam_utils.py), così la saliency map
    riflette esattamente le attivazioni che alimentano le teste value/advantage.
    """

    def __init__(
        self,
        in_channels: int,
        num_actions: int,
        conv_channels: tuple = (16, 32, 64),
        fc_dim: int = 128,
    ):
        super().__init__()
        c1, c2, c3 = conv_channels

        def gn(ch: int) -> nn.GroupNorm:
            groups = max(1, ch // 4)
            while ch % groups != 0:
                groups -= 1
            return nn.GroupNorm(groups, ch)

        self.conv1 = nn.Conv2d(in_channels, c1, kernel_size=3, stride=2, padding=1)
        self.norm1 = gn(c1)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1)
        self.norm2 = gn(c2)
        self.conv3 = nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1)
        self.norm3 = gn(c3)

        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)
        self.relu3 = nn.ReLU(inplace=False)

        # Punto di aggancio Grad-CAM: ultima feature map prima del pooling.
        self.gradcam_layer = nn.Identity()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.value_stream = nn.Sequential(
            nn.Linear(c3, fc_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fc_dim, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(c3, fc_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fc_dim, num_actions),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.relu2(self.norm2(self.conv2(x)))
        x = self.relu3(self.norm3(self.conv3(x)))
        x = self.gradcam_layer(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        pooled = self.pool(feat).flatten(1)
        value = self.value_stream(pooled)
        adv = self.advantage_stream(pooled)
        return value + adv - adv.mean(dim=1, keepdim=True)


# ─────────────────────────────────────────────────────────────────────────────
# ROI Finder Agent (Agent 1)
# ─────────────────────────────────────────────────────────────────────────────

class ROIFinderAgent:
    """
    Double Dueling DQN (CNN) per localizzare la ROI (bounding box).

    Compatibile con environment_roi.ROIFinderEnv (stato immagine [2,S,S]).
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
        # OTTIMIZZAZIONE: usa roi_epsilon_decay se presente (dedicato), invece
        # di ereditare sempre epsilon_decay condiviso col refiner, cosi' il
        # decadimento si allinea al numero effettivo di roi_episodes.
        self.eps_decay = rl.get("roi_epsilon_decay", rl.get("epsilon_decay", 300))

        # Stato CNN: stessa risoluzione/numero canali usati da ROIFinderEnv
        # (letti dalla stessa chiave di config per restare sempre coerenti).
        self.cnn_size    = int(rl.get("roi_cnn_size", 48))
        # OTTIMIZZAZIONE: legge la stessa chiave usata da ROIFinderEnv per
        # restare sempre coerente (2 canali default, 3 se si abilita il
        # canale gradiente extra via roi_state_channels: 3 in config).
        self.in_channels = int(rl.get("roi_state_channels", ROIFinderEnv.STATE_CHANNELS))
        conv_channels    = tuple(rl.get("roi_conv_channels", [16, 32, 64]))
        fc_dim           = rl.get("roi_hidden_dim", 128)

        self.online = DuelingCNN(self.in_channels, self.num_actions, conv_channels, fc_dim).to(device)
        self.target = DuelingCNN(self.in_channels, self.num_actions, conv_channels, fc_dim).to(device)
        self._sync_target()
        self.target.eval()

        self.optimizer = torch.optim.Adam(
            self.online.parameters(),
            lr=rl.get("roi_lr", 5e-4),
            weight_decay=1e-5,
        )
        # OTTIMIZZAZIONE: Prioritized Experience Replay opzionale (default off
        # per compatibilita', ma abilitato di default nel config.yaml fornito).
        self.use_per = bool(rl.get("use_prioritized_replay", False))
        self.per_alpha = float(rl.get("per_alpha", 0.6))
        self.per_beta_start = float(rl.get("per_beta_start", 0.4))
        self.per_beta_frames = float(rl.get("per_beta_frames", 20000))
        self._learn_steps = 0
        buf_size = rl.get("replay_buffer_size", 10000)
        if self.use_per:
            self.buffer = PrioritizedReplayBuffer(buf_size, alpha=self.per_alpha)
        else:
            self.buffer = ReplayBuffer(buf_size)

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
        s = torch.from_numpy(np.ascontiguousarray(state)).float().unsqueeze(0).to(self.device)
        q = self.online(s)
        return int(q.argmax(dim=1).item())

    # ── Memoria ───────────────────────────────────────────────────────────────

    def remember(self, state, action, reward, next_state, done) -> None:
        self.buffer.push(state, action, reward, next_state, done)

    # ── Apprendimento ─────────────────────────────────────────────────────────

    def learn(self) -> Optional[float]:
        if not self.buffer.is_ready(self.batch_size):
            return None

        if self.use_per:
            self._learn_steps += 1
            beta = min(1.0, self.per_beta_start + self._learn_steps *
                       (1.0 - self.per_beta_start) / max(1.0, self.per_beta_frames))
            states, actions, rewards, next_states, dones, indices, is_weights = \
                self.buffer.sample(self.batch_size, beta=beta)
            w = torch.from_numpy(is_weights).float().to(self.device)
        else:
            states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
            w = None

        s  = torch.from_numpy(states).float().to(self.device)
        a  = torch.LongTensor(actions).to(self.device)
        r  = torch.FloatTensor(rewards).to(self.device)
        s2 = torch.from_numpy(next_states).float().to(self.device)
        d  = torch.FloatTensor(dones).to(self.device)

        with torch.no_grad():
            best_actions = self.online(s2).argmax(dim=1, keepdim=True)
            q_next  = self.target(s2).gather(1, best_actions).squeeze(1)
            q_target = r + self.gamma * q_next * (1 - d)

        q_online = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)

        td_errors = (q_online - q_target).detach()
        elementwise_loss = F.smooth_l1_loss(q_online, q_target, reduction="none")
        if w is not None:
            loss = (elementwise_loss * w).mean()
        else:
            loss = elementwise_loss.mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 1.0)
        self.optimizer.step()

        if self.use_per:
            self.buffer.update_priorities(indices, td_errors.cpu().numpy())

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
