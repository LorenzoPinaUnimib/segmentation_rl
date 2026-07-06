"""
replay_buffer.py — Experience replay buffer for DQN.
"""
import numpy as np
from collections import deque
from typing import Tuple
import random


class ReplayBuffer:
    """Fixed-size circular buffer storing (s, a, r, s', done) transitions."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)

    def is_ready(self, min_size: int) -> bool:
        return len(self) >= min_size


# ─────────────────────────────────────────────────────────────────────────────
# OTTIMIZZAZIONE: Prioritized Experience Replay (variante proporzionale)
# ─────────────────────────────────────────────────────────────────────────────

class PrioritizedReplayBuffer:
    """
    Replay buffer con campionamento pesato per priorita' (TD-error).

    Motivazione: con campionamento uniforme, i casi difficili (tumori piccoli,
    periferici, a basso contrasto) vengono visti dal gradiente con la stessa
    frequenza dei casi facili, pur essendo rari e pur essendo quelli su cui
    l'agente sbaglia di piu' — il segnale di apprendimento utile viene diluito.
    Pesare il campionamento per |TD-error| concentra gli update sui casi dove
    la Q-network sbaglia di piu' (in stile Schaul et al., 2016), accelerando
    la convergenza su un budget di iterazioni fisso.

    Implementazione volutamente semplice (array piatto, non sum-tree) per
    restare leggera: adeguata per capacita' fino a decine di migliaia di
    transizioni come in questo progetto.
    """

    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer: list = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self._eps = 1e-5

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        max_prio = self.priorities.max() if self.buffer else 1.0
        data = (state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.pos] = data
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4):
        n = len(self.buffer)
        prios = self.priorities[:n]
        probs = prios ** self.alpha
        probs_sum = probs.sum()
        if probs_sum <= 0:
            probs = np.ones_like(probs) / max(1, n)
        else:
            probs = probs / probs_sum

        replace = n < batch_size
        indices = np.random.choice(n, batch_size, replace=replace, p=probs)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        weights = (n * probs[indices]) ** (-beta)
        weights = weights / (weights.max() + 1e-8)

        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
            indices,
            np.array(weights, dtype=np.float32),
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        prios = np.abs(td_errors) + self._eps
        for idx, p in zip(indices, prios):
            self.priorities[idx] = p

    def __len__(self) -> int:
        return len(self.buffer)

    def is_ready(self, min_size: int) -> bool:
        return len(self) >= min_size
