"""
config.py
─────────
Costanti allineate al paper: Active Object Localization with Deep Reinforcement Learning (2015).
"""

# ── Azioni (9 gradi di libertà) ──────────────────────────────────────────
N_ACTIONS = 9
ALPHA = 0.2  # Fattore di trasformazione della bounding box (Eq. 1)

ACTION_NAMES = {
    0: "RIGHT", 1: "LEFT", 2: "UP", 3: "DOWN",
    4: "BIGGER", 5: "SMALLER", 6: "FATTER", 7: "TALLER",
    8: "TRIGGER",
}

# ── Reward Shaping (Eq. 2 e Eq. 3) ───────────────────────────────────────
REWARD_POSITIVE = 1.0
REWARD_NEGATIVE = -1.0
TRIGGER_REWARD = 3.0       # Valore di eta (\eta)
TAU_IOU = 0.6              # Soglia minima per considerare il trigger un successo (\tau)

# ── Stato e Ambiente ─────────────────────────────────────────────────────
HISTORY_LENGTH = 10        # Ultime 10 azioni (codificate one-hot a 9 dim -> 90 dim)
CONTEXT_PIXELS = 16        # Pixel di contesto estratti attorno al box
WARP_SIZE = (224, 224)     # Dimensione input CNN
MAX_STEPS_PER_EPISODE = 200 # L'agente si ferma forzatamente a 200 step