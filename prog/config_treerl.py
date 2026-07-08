"""
config_treerl.py
─────────────────
Costanti fedeli a: Jie, Liang, Feng, Jin, Lu, Yan, "Tree-Structured
Reinforcement Learning for Sequential Object Localization", NeurIPS 2016
(arXiv:1703.02710). https://arxiv.org/pdf/1703.02710

Ogni valore qui sotto e' preso LETTERALMENTE dal testo del paper (sezioni
3.1 e 3.4), non dal codice del progetto Caicedo & Lazebnik gia' presente in
config.py/environment.py/q_network.py/train.py -- questo e' un metodo
diverso (multi-oggetto, nessuna azione di trigger, ricerca ad albero),
quindi vive in file paralleli con suffisso "_treerl" per non mischiarsi con
la pipeline single-object esistente.

Dove il paper NON specifica un valore (es. ottimizzatore, learning rate,
dimensioni della testa MLP), il default scelto qui e' segnalato esplicitamente
come ASSUNZIONE NON VERIFICATA, con un flag CLI in train_treerl.py per
cambiarlo senza editare il codice.
"""

# ── Azioni (Sezione 3.1, "Actions") ─────────────────────────────────────
# "The available actions of the agent consist of two groups, one for
# scaling the current window to a sub-window, and the other one for
# translating the current window locally."
N_SCALE_ACTIONS = 5       # sub-finestra a 0.55x del lato corrente (Fig. 2)
N_TRANSLATE_ACTIONS = 8   # right/left/up/down/shorter_h/longer_h/shorter_v/longer_v
N_ACTIONS = N_SCALE_ACTIONS + N_TRANSLATE_ACTIONS  # 13, NESSUN trigger

# "each corresponding to a certain sub-window with the size 0.55 times as
# the current window"
SCALE_FACTOR = 0.55

# "Each local translation action moves the window by 0.25 times of the
# current window size."
TRANSLATE_ALPHA = 0.25

# Indici azione -> nome. Azioni 0-4: scaling group (ordine ASSUNTO come nello
# schema "5-crop" standard: 4 angoli + centro, dato che il paper mostra solo
# Fig. 2 senza didascalia testuale sull'ordine esatto -- ASSUNZIONE NON
# VERIFICATA, ma l'insieme di 5 sotto-finestre a 0.55x e' quello descritto
# nel testo). Azioni 5-12: translation group, ordine ESATTO dal testo:
# "horizontal moving to left/right, vertical moving to up/down, becoming
# shorter/longer horizontally and becoming shorter/longer vertically".
ACTION_NAMES = {
    0: "SCALE_TOP_LEFT", 1: "SCALE_TOP_RIGHT", 2: "SCALE_BOTTOM_LEFT",
    3: "SCALE_BOTTOM_RIGHT", 4: "SCALE_CENTER",
    5: "RIGHT", 6: "LEFT", 7: "UP", 8: "DOWN",
    9: "SHORTER_H", 10: "LONGER_H", 11: "SHORTER_V", 12: "LONGER_V",
}
SCALE_ACTIONS = list(range(0, N_SCALE_ACTIONS))                       # [0..4]
TRANSLATE_ACTIONS = list(range(N_SCALE_ACTIONS, N_ACTIONS))           # [5..12]

# ── Stato (Sezione 3.1, "States") ───────────────────────────────────────
# "50 past actions are encoded in the state" -- 50 (non 10 come in Caicedo).
HISTORY_LENGTH = 50
# "Each action is represented by a 13-d binary vector" -> HISTORY_LENGTH*13 = 650
HISTORY_DIM = HISTORY_LENGTH * N_ACTIONS

# Feature visive: VGG-16 fc6, calcolata via ROI-pooling sulle feature map
# "conv5_3" pre-calcolate (condivise per l'intera immagine, un solo forward
# convoluzionale per episodio -- vedi q_network_treerl.py).
ROI_POOL_SIZE = (7, 7)   # dimensione spaziale attesa in ingresso a fc6 (7x7x512=25088)
CONV5_3_STRIDE = 16      # VGG-16: 4 maxpool prima di conv5_3 -> stride 2^4
VISUAL_FEAT_DIM = 4096   # dim di fc6 (sia per la finestra corrente sia per l'immagine intera)

# ── Reward (Sezione 3.1, "Rewards", Eq. 1 e Eq. 2) ──────────────────────
# Eq. 1: r(s,a) = max_i sign(IoU(w', g_i) - IoU(w, g_i))
# Eq. 2: +5 se un qualunque oggetto viene "hittato" (IoU>0.5) per la prima
#        volta in questo episodio, altrimenti Eq. 1.
HIT_IOU_THRESH = 0.5
HIT_REWARD = 5.0
REWARD_POSITIVE = 1.0   # sign(+) del delta IoU (Eq. 1)
REWARD_NEGATIVE = -1.0  # sign(-) del delta IoU (Eq. 1)

# ── Deep Q-learning (Sezione 3.3/3.4, "Implementation Details") ────────
GAMMA = 0.9
MAX_STEPS_PER_EPISODE = 50   # "We run each episode with maximal 50 steps during training."
REPLAY_SIZE = 800_000        # "The replay memory size is set to 800,000"
BATCH_SIZE = 64              # "The mini batch size in training is set to 64."
N_EPOCHS = 25                # "We train a deep Q-network ... for 25 epochs."

# "epsilon is annealed linearly from 1 to 0.1 over the first 10 epochs.
#  Then epsilon is fixed to 0.1 in the last 15 epochs."
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY_EPOCHS = 10   # 10 + 15 = 25 epoche totali, esattamente come nel paper

# ── Ricerca ad albero (Sezione 3.2 e 4, Tabella 2) ──────────────────────
# "1+2+4+8+16=31 proposals" con 5 livelli -> ogni livello raddoppia i nodi
# (bifurcazione: 1 azione di scaling + 1 di traslazione per nodo).
DEFAULT_TREE_LEVELS = 5
