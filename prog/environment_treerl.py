"""
environment_treerl.py
──────────────────────
Ambiente fedele a Jie et al. 2016, "Tree-Structured Reinforcement Learning
for Sequential Object Localization" (arXiv:1703.02710).

DIFFERENZE STRUTTURALI rispetto a environment.py (Caicedo & Lazebnik) gia'
presente nel progetto -- NON e' una variante, e' un altro MDP:
  - NESSUNA azione di trigger: 13 azioni, tutte di trasformazione della
    finestra (5 scaling + 8 traslazione). L'episodio non "termina" mai per
    scelta dell'agente, solo per limite di step (training) o per profondita'
    dell'albero (inferenza, vedi tree_search_treerl.py).
  - MULTI-oggetto: la reward (Eq. 1/2 del paper) e' definita rispetto a TUTTI
    i box ground-truth dell'immagine, non a uno solo. Qui i box multipli
    vengono estratti dalla maschera binaria via connected components
    (cv2.connectedComponentsWithStats), NON dal singolo bounding-box del
    blob piu' grande come faceva prepare_state/reset() in environment.py.
    Se il dataset ha davvero un solo tumore per immagine (come documentato
    nel resto del progetto), questo degenera naturalmente a n=1 -- ma il
    codice resta corretto anche se una maschera contiene piu' componenti
    connesse distinte (es. tumori satellite separati).
  - Il reward dipende dalla STORIA COMPLETA dell'episodio (max IoU mai
    raggiunto per ciascun oggetto, non solo lo step precedente), per il
    termine "+5 al primo hit" (Eq. 2) -- vedi hit_flags/best_iou_per_obj.
  - Formato box: [x, y, w, h] in pixel dell'immagine (stessa convenzione di
    utils.py::compute_iou, riusata cosi' com'e' da questo modulo).

NOTA (fallback per maschere vuote): a differenza di environment.py (che
assegna un gt_box fittizio "quarto centrale" quando la maschera e' vuota --
un canale di segnale spurio gia' segnalato come problematico in analisi
precedenti), qui un'immagine senza alcun oggetto ha semplicemente gt_boxes=[]
ed episodi da SALTARE in training (vedi train_treerl.py::filter_dataset_with_objects),
invece di inventare un target arbitrario.
"""
import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces
from collections import deque

from config_treerl import (
    N_ACTIONS, N_SCALE_ACTIONS, SCALE_ACTIONS, TRANSLATE_ACTIONS,
    SCALE_FACTOR, TRANSLATE_ALPHA, HISTORY_LENGTH, HISTORY_DIM,
    HIT_IOU_THRESH, HIT_REWARD, REWARD_POSITIVE, REWARD_NEGATIVE,
    MAX_STEPS_PER_EPISODE,
)
from utils import compute_iou


# ─────────────────────────────────────────────────────────────────────────────
# Estrazione multi-oggetto dalla maschera binaria
# ─────────────────────────────────────────────────────────────────────────────

def get_gt_boxes_from_mask(mask, min_area=20):
    """Estrae uno o piu' bounding box [x,y,w,h] dalla maschera binaria,
    uno per ogni componente connessa (cv2.connectedComponentsWithStats),
    scartando componenti piu' piccole di min_area pixel (rumore di
    segmentazione). Ritorna lista vuota se non c'e' nessun oggetto valido.

    mask: array 2D, valori >0 = foreground.
    """
    mask_u8 = (mask > 0.5).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    boxes = []
    for label in range(1, n_labels):  # 0 = background
        x, y, w, h, area = stats[label]
        if area < min_area:
            continue
        boxes.append(np.array([x, y, w, h], dtype=np.float32))
    return boxes


# ─────────────────────────────────────────────────────────────────────────────
# Trasformazioni geometriche della finestra (condivise da env.step() e da
# tree_search_treerl.py per la ricerca ad albero in inferenza -- STESSA
# funzione pura, cosi' training e inferenza applicano ESATTAMENTE la stessa
# geometria).
# ─────────────────────────────────────────────────────────────────────────────

def apply_action(box, action, W, H, min_size=10):
    """Applica l'azione 'action' (0..12) alla finestra box=[x,y,w,h] e
    ritorna la nuova finestra, clippata dentro [0,W]x[0,H].

    Azioni 0-4 (scaling, Fig. 2): sotto-finestra a SCALE_FACTOR (0.55) del
    lato corrente, posizionata nei 4 angoli o al centro (vedi ASSUNZIONE in
    config_treerl.py sull'ordine esatto -- il paper non lo specifica a testo).
    Azioni 5-12 (traslazione): spostamento/ridimensionamento di
    TRANSLATE_ALPHA (0.25) del lato corrente. Le azioni SHORTER/LONGER
    (asimmetriche nel paper, che non ne specifica l'ancoraggio esatto) sono
    implementate qui centrate (il centro della finestra resta fisso mentre
    la larghezza/altezza cambia), per evitare una deriva sistematica verso
    un lato ad ogni singola applicazione -- ASSUNZIONE NON VERIFICATA contro
    il testo del paper, che descrive solo l'effetto ("shorter/longer") non
    l'ancoraggio geometrico.
    """
    x, y, w, h = box.copy()

    if action in SCALE_ACTIONS:
        new_w = SCALE_FACTOR * w
        new_h = SCALE_FACTOR * h
        if action == 0:      # top-left
            nx, ny = x, y
        elif action == 1:    # top-right
            nx, ny = x + (w - new_w), y
        elif action == 2:    # bottom-left
            nx, ny = x, y + (h - new_h)
        elif action == 3:    # bottom-right
            nx, ny = x + (w - new_w), y + (h - new_h)
        else:                 # action == 4: center
            nx, ny = x + (w - new_w) / 2.0, y + (h - new_h) / 2.0
        x, y, w, h = nx, ny, new_w, new_h
    else:
        a = TRANSLATE_ALPHA
        aw, ah = a * w, a * h
        if action == 5:       # RIGHT
            x += aw
        elif action == 6:     # LEFT
            x -= aw
        elif action == 7:     # UP
            y -= ah
        elif action == 8:     # DOWN
            y += ah
        elif action == 9:     # SHORTER_H (centrata)
            x += aw / 2.0
            w -= aw
        elif action == 10:    # LONGER_H (centrata)
            x -= aw / 2.0
            w += aw
        elif action == 11:    # SHORTER_V (centrata)
            y += ah / 2.0
            h -= ah
        elif action == 12:    # LONGER_V (centrata)
            y -= ah / 2.0
            h += ah

    x = np.clip(x, 0, W - 1)
    y = np.clip(y, 0, H - 1)
    w = np.clip(w, min_size, W - x)
    h = np.clip(h, min_size, H - y)
    return np.array([x, y, w, h], dtype=np.float32)


class TreeRLEnv(gym.Env):
    """Ambiente multi-oggetto senza trigger, fedele a Jie et al. 2016.

    A differenza di ActiveLocalizationEnv (environment.py), NON produce crop
    warpati: le feature visive sono calcolate una volta per immagine sulla
    feature map conv5_3 condivisa (vedi q_network_treerl.py, TreeRLBackbone),
    quindi l'ambiente espone solo la geometria della finestra (box) e lascia
    al training loop / al Q-network il compito di calcolare le feature via
    ROI-pooling su quella mappa condivisa -- molto piu' efficiente del
    crop+resize+forward-CNN ad ogni step usato in Caicedo & Lazebnik.
    """

    def __init__(self, pytorch_dataset, min_object_area=20):
        super().__init__()
        self.dataset = pytorch_dataset
        self.min_object_area = min_object_area
        sample = self.dataset[0]
        self.C, self.H, self.W = sample["image"].shape

        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Dict({
            "window": spaces.Box(low=0, high=max(self.W, self.H), shape=(4,), dtype=np.float32),
            "history": spaces.Box(low=0.0, high=1.0, shape=(HISTORY_DIM,), dtype=np.float32),
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options is not None and "idx" in options:
            idx = int(options["idx"])
        else:
            idx = self.np_random.integers(0, len(self.dataset))
        sample = self.dataset[idx]

        self.current_image = sample["image"].numpy()  # [C,H,W], usato per calcolare conv5_3 una volta per episodio
        mask = sample["mask"].numpy().squeeze(0)
        self.gt_boxes = get_gt_boxes_from_mask(mask, min_area=self.min_object_area)
        self.n_objects = len(self.gt_boxes)

        # "Starting with taking the entire image as a proposal"
        self.box = np.array([0, 0, self.W, self.H], dtype=np.float32)
        self.history = deque([0] * HISTORY_DIM, maxlen=HISTORY_DIM)
        self.current_step = 0

        # Tracking per-oggetto per la reward Eq. 1/2: IoU dello stato
        # corrente (per il delta-sign) e IoU MASSIMA MAI raggiunta
        # nell'intero episodio (per il "+5 al primo hit", che dipende dalla
        # storia completa dei box visitati, non solo dall'ultimo step).
        self.previous_ious = np.array(
            [compute_iou(self.box, g) for g in self.gt_boxes], dtype=np.float32
        ) if self.n_objects > 0 else np.zeros(0, dtype=np.float32)
        self.best_iou_per_obj = self.previous_ious.copy()
        self.hit_flags = (self.best_iou_per_obj > HIT_IOU_THRESH)

        return self._get_obs(), {"idx": idx, "n_objects": self.n_objects}

    def _get_obs(self):
        return {
            "window": self.box.copy(),
            "history": np.array(self.history, dtype=np.float32),
        }

    def step(self, action):
        self.current_step += 1

        one_hot = [0] * N_ACTIONS
        one_hot[action] = 1
        self.history.extend(one_hot)

        self.box = apply_action(self.box, action, self.W, self.H)

        if self.n_objects == 0:
            # Nessun oggetto in questa immagine: nessuna reward informativa
            # possibile (vedi docstring del modulo). Il training loop deve
            # filtrare a monte queste immagini (filter_dataset_with_objects
            # in train_treerl.py); questo ramo e' solo una rete di sicurezza.
            reward = 0.0
            new_ious = np.zeros(0, dtype=np.float32)
            newly_hit = False
        else:
            new_ious = np.array([compute_iou(self.box, g) for g in self.gt_boxes], dtype=np.float32)

            # Eq. 2: +5 se un qualunque oggetto supera HIT_IOU_THRESH per la
            # PRIMA volta nell'intero episodio (confronto contro il massimo
            # storico, non contro lo step precedente).
            new_best = np.maximum(self.best_iou_per_obj, new_ious)
            became_hit = (new_best > HIT_IOU_THRESH) & (~self.hit_flags)
            newly_hit = bool(became_hit.any())

            if newly_hit:
                reward = HIT_REWARD
            else:
                # Eq. 1: max_i sign(IoU(w',g_i) - IoU(w,g_i))
                deltas = new_ious - self.previous_ious
                signs = np.sign(deltas)
                reward = float(REWARD_POSITIVE if signs.max() > 0 else REWARD_NEGATIVE)

            self.best_iou_per_obj = new_best
            self.hit_flags = self.hit_flags | became_hit
            self.previous_ious = new_ious

        # Nessuna azione di trigger -> "terminated" nel senso RL classico non
        # esiste in questo MDP: l'episodio finisce SOLO per limite di step
        # durante il training (vedi MAX_STEPS_PER_EPISODE). A differenza di
        # Caicedo & Lazebnik, qui NON c'e' nemmeno un bug time-limit da
        # correggere: il paper stesso non usa mai un flag "done" nella sua
        # equazione di update (Eq. 3), quindi il bootstrap TD non va mai
        # azzerato (vedi train_treerl.py, che infatti non salva alcun flag
        # "done" nel replay buffer).
        terminated = False
        truncated = self.current_step >= MAX_STEPS_PER_EPISODE

        info = {
            "ious": new_ious.tolist() if self.n_objects > 0 else [],
            "newly_hit": newly_hit,
            "n_hit_total": int(self.hit_flags.sum()) if self.n_objects > 0 else 0,
            "n_objects": self.n_objects,
        }
        return self._get_obs(), reward, terminated, truncated, info
