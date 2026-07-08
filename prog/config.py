"""
config.py
─────────
Costanti allineate al paper: Active Object Localization with Deep Reinforcement Learning (2015).
"""

# ── Azioni (9 gradi di libertà) ──────────────────────────────────────────
N_ACTIONS = 9
ALPHA = 0.1  # Fattore di trasformazione della bounding box (Eq. 1)

ACTION_NAMES = {
    0: "RIGHT", 1: "LEFT", 2: "UP", 3: "DOWN",
    4: "BIGGER", 5: "SMALLER", 6: "FATTER", 7: "TALLER",
    8: "TRIGGER",
}

# ── Reward Shaping (Eq. 2 e Eq. 3 del paper -- ORA POTENZIATO) ──────────
# FIX (reward migliorato, vedi environment.py::step): il paper originale usa
# un reward binario +-1 basato solo sul SEGNO del delta IoU. Due problemi:
#   1. Nessuna informazione sulla MAGNITUDO del miglioramento (spostarsi di
#      1px o di 50px verso il target da' lo stesso +1).
#   2. IoU piano e' sempre 0 quando i due box non si sovrappongono affatto
#      (tipico nei primi step, si parte dall'intera immagine): ogni azione,
#      buona o pessima, riceve lo stesso -1, quindi zero segnale utile
#      proprio nella fase piu' critica.
# ORA: reward continuo = delta di GIoU (Generalized IoU, vedi utils.py),
# che resta informativo anche su box disgiunti, piu' una piccola penalita'
# di step per scoraggiare percorsi inutilmente lunghi. REWARD_POSITIVE/
# REWARD_NEGATIVE restano definite (compatibilita'/riferimento) ma non sono
# piu' usate direttamente in environment.py.
REWARD_POSITIVE = 1.0
REWARD_NEGATIVE = -1.0
GIOU_SHAPING_SCALE = 3.0   # amplifica il delta-GIoU (tipicamente piccolo, <<1) alla stessa scala di TRIGGER_REWARD
STEP_PENALTY = 0.02        # piccola penalita' ad ogni step non-trigger: incoraggia episodi piu' brevi/efficienti

TRIGGER_REWARD = 3.0       # Valore di eta (\eta), assegnato per un trigger corretto (iou >= TAU_IOU)
# FIX: un trigger sbagliato NON e' sempre ugualmente sbagliato -- fallire di
# poco (iou=0.55 con tau=0.6) e fallire completamente (iou=0.05) prima
# ricevevano la STESSA penalita' fissa (-TRIGGER_REWARD), senza alcun
# gradiente che aiuti la rete a distinguere "quasi giusto" da "pessimo".
# Ora la penalita' e' scalata in base a quanto ci si discosta da TAU_IOU
# (vedi environment.py::step): resta comunque sempre negativa e comunque
# nello stesso ordine di grandezza di TRIGGER_REWARD, ma con un gradiente.
TAU_IOU = 0.6              # Soglia minima per considerare il trigger un successo (\tau)

# ── Stato e Ambiente ─────────────────────────────────────────────────────
HISTORY_LENGTH = 10        # Ultime 10 azioni (codificate one-hot a 9 dim -> 90 dim)
CONTEXT_PIXELS = 16        # Pixel di contesto estratti attorno al box
WARP_SIZE = (224, 224)     # Dimensione input CNN
MAX_STEPS_PER_EPISODE = 50 # L'agente si ferma forzatamente a 200 step