"""
config.py
─────────
Costanti condivise da tutto il pacchetto: azioni, reward shaping,
soglie di valutazione e durate di default del curriculum.
Tenerle in un solo posto evita che i moduli vadano fuori sincrono.
"""

# ── Azioni ────────────────────────────────────────────────────────────────
N_ACTIONS = 9  # 0-3: sposta centro, 4-7: ridimensiona (w/h), 8: STOP
MAX_STEPS_PER_EPISODE = 50
# FIX (Dict observation space): N_COORD_CHANNELS ora e' la dimensione del
# vettore box (cx,cy,w,h) passato come input NUMERICO separato alla policy
# (chiave "box_vec" della Dict obs, vedi environment.py/utils.py), non piu'
# "spalmato" su 4 piani immagine interi come in precedenza. Dimezza i canali
# immagine (8->4: 3 RGB + 1 maschera box) -> dimezza la RAM del replay buffer
# di SAC a parita' di buffer_size, e da' alla rete il box come numero esatto
# invece che da dedurre da un piano a valore costante.
N_COORD_CHANNELS = 4

OPPOSITE_ACTIONS = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4, 6: 7, 7: 6}

ACTION_NAMES = {
    0: "LEFT", 1: "RIGHT", 2: "UP", 3: "DOWN",
    4: "WIDER", 5: "NARROWER", 6: "TALLER", 7: "SHORTER",
    8: "STOP",
}

# ── Reward shaping ───────────────────────────────────────────────────────
DELTA_IOU_SCALE = 25.0
DELTA_IOU_CLIP = 10.0

DISTANCE_REWARD_SCALE = 0.05
DISTANCE_REWARD_CLIP = 1.0

TIME_PENALTY = -0.01

STOP_IOU_BASELINE = 0.50
# FIX v2: scale=12/clip=8 (tentativo precedente) rendeva troppo economico
# fermarsi subito con un box mediocre rispetto al costo di continuare
# (TIME_PENALTY=-0.01/step): l'agente ha imparato a premere STOP quasi subito
# (vedi action_distribution/action_8 salire a ~0.30 e mean_ep_length crollare
# a 1-3 step). Si rialza la severita', ma non ai livelli originali (20/15) che
# causavano il problema opposto (mai fermarsi).
STOP_BONUS_SCALE = 18.0
STOP_BONUS_CLIP = 12.0

# FIX v4: STOP_IOU_BASELINE fisso a 0.5 e' un traguardo sempre uguale
# indipendentemente da quanto e' difficile il punto di partenza. A difficolta'
# alta il box iniziale e' completamente casuale (spesso senza overlap con il
# target): raggiungere 0.5 di IoU in 200 step da li' e' molto piu' difficile
# che da un box gia' vicino, e il reward diventa quasi sempre fortemente
# negativo indipendentemente da quanto l'agente si sia impegnato, un segnale
# poco informativo. Questo bonus supplementare premia il MIGLIORAMENTO
# relativo rispetto al punto di partenza dell'episodio (iou_final - iou_iniziale),
# indipendente dalla difficolta': anche quando 0.5 assoluto e' irrealistico,
# l'agente riceve comunque un segnale utile per il progresso genuino fatto.
IMPROVEMENT_BONUS_SCALE = 6.0
IMPROVEMENT_BONUS_CLIP = 4.0

NO_STOP_PENALTY = -1.0

OVERSIZE_AREA_RATIO_THRESHOLD = 0.50
OVERSIZE_PENALTY_SCALE = 4.0

OSCILLATION_PENALTY = 0.01

# FIX v7 (action space continuo): penalita' sul "jerk" (variazione tra azioni
# consecutive) nel ramo continuo di environment.py -- equivalente di
# OSCILLATION_PENALTY ma per vettori continui invece che azioni discrete
# opposte. Tenuta volutamente piccola: deve scoraggiare micro-jitter senza
# impedire correzioni di rotta grandi e legittime.
ACTION_SMOOTHNESS_PENALTY_SCALE = 0.02

# ── Valutazione ───────────────────────────────────────────────────────────
SIZE_BUCKET_EDGES = (0.10, 0.20)
IOU_THRESHOLDS = (0.3, 0.5, 0.7)
SUCCESS_IOU_THRESHOLD = 0.5

# ── Curriculum di default (sovrascrivibili da CLI in train.py) ────────────
DEFAULT_STOP_CURRICULUM_STEPS = 600_000
DEFAULT_STEP_SIZE_CURRICULUM_STEPS = 800_000
DEFAULT_ENTROPY_SCHEDULE_STEPS = 1_000_000
DEFAULT_INIT_BOX_CURRICULUM_STEPS = 1_000_000
