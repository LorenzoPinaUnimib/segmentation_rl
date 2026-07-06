"""
environment.py
───────────────
Ambiente fedele al paper: reward quantizzati, crop (warped regions), 
e storico delle ultime 10 azioni.

DIFFERENZE NOTE rispetto al paper (segnalate, non "bug" da correggere: sono
adattamenti legittimi al dominio diverso di questo progetto):

- Dominio: il paper (Caicedo & Lazebnik 2015, "Active Object Localization
  with Deep Reinforcement Learning") lavora su Pascal VOC (20 classi,
  multi-oggetto, un Q-network per categoria + un SVM esterno per il
  re-ranking delle detection ai fini della valutazione AP). Qui il task e'
  tumor-localization single-class su immagini MRI: stessa metodologia (DQN +
  9 azioni + Apprenticeship Learning), applicata a un problema diverso, non
  "lo stesso esperimento".
- Procedura di test del paper (sez. 4.3) non implementata: dopo il training
  il paper prevede un IoR mark (croce nera dopo il trigger), riavvio della
  ricerca dopo 40 step senza trigger o dopo il trigger stesso, e un box di
  riavvio al 75% dell'immagine posizionato negli angoli in ordine fisso --
  perche' in Pascal VOC un'immagine puo' contenere piu' istanze dello stesso
  oggetto. Questo ambiente gestisce un solo box/gt_box per episodio: ha senso
  per un dominio con un solo tumore per immagine, ma resta un pezzo di spec
  del paper assente se in futuro si volesse gestire multi-istanza.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque

from config import (
    N_ACTIONS, ALPHA, REWARD_POSITIVE, REWARD_NEGATIVE, 
    TRIGGER_REWARD, TAU_IOU, HISTORY_LENGTH, CONTEXT_PIXELS, WARP_SIZE,
    MAX_STEPS_PER_EPISODE
)
from utils import compute_iou, extract_warped_region

class ActiveLocalizationEnv(gym.Env):
    def __init__(self, pytorch_dataset):
        super().__init__()
        self.dataset = pytorch_dataset
        sample = self.dataset[0]
        self.C, self.H, self.W = sample["image"].shape
        
        self.action_space = spaces.Discrete(N_ACTIONS)
        
        # L'osservazione ora è il crop 224x224 + il vettore history
        self.observation_space = spaces.Dict({
            "region": spaces.Box(low=0, high=1.0, shape=(self.C, WARP_SIZE[1], WARP_SIZE[0]), dtype=np.float32),
            "history": spaces.Box(low=0.0, high=1.0, shape=(HISTORY_LENGTH * N_ACTIONS,), dtype=np.float32),
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # FIX (definizione di "epoca" fedele al paper): il paper definisce
        # un'epoca come un passaggio COMPLETO e SENZA RIMPIAZZO su ogni
        # immagine di training esattamente una volta. Campionare qui con
        # self.np_random.integers(...) e' un campionamento CON rimpiazzo:
        # alcune immagini potrebbero non essere mai viste in un'epoca, altre
        # viste piu' volte. train.py ora costruisce esplicitamente una
        # permutazione del training set per ogni epoca e passa l'indice da
        # usare tramite options={"idx": ...}; il campionamento casuale resta
        # solo come fallback per chi chiama reset() senza specificare idx
        # (es. modalita' --simulate / --watch-idx).
        if options is not None and "idx" in options:
            idx = int(options["idx"])
        else:
            idx = self.np_random.integers(0, len(self.dataset))
        sample = self.dataset[idx]
        
        self.current_image = sample["image"].numpy()
        mask = sample["mask"].numpy().squeeze(0)
        
        pos = np.where(mask > 0.5)
        if len(pos[0]) > 0:
            ymin, ymax = np.min(pos[0]), np.max(pos[0])
            xmin, xmax = np.min(pos[1]), np.max(pos[1])
            self.gt_box = np.array([xmin, ymin, xmax - xmin, ymax - ymin], dtype=np.float32)
        else:
            self.gt_box = np.array([self.W/4, self.H/4, self.W/2, self.H/2], dtype=np.float32)

        # "The proposed model... starts by analyzing the whole scene"
        self.box = np.array([0, 0, self.W, self.H], dtype=np.float32)
        self.history = deque([0] * (HISTORY_LENGTH * N_ACTIONS), maxlen=HISTORY_LENGTH * N_ACTIONS)
        
        self.current_step = 0
        self.previous_iou = compute_iou(self.box, self.gt_box)
        
        return self._get_obs(), {}

    def _get_obs(self):
        region = extract_warped_region(self.current_image, self.box, CONTEXT_PIXELS, WARP_SIZE)
        history_vec = np.array(self.history, dtype=np.float32)
        return {"region": region, "history": history_vec}

    def _apply_action(self, box, action):
        x, y, w, h = box.copy()
        aw = ALPHA * w
        ah = ALPHA * h
        
        if action == 0: x += aw                  # RIGHT
        elif action == 1: x -= aw                # LEFT
        elif action == 2: y -= ah                # UP
        elif action == 3: y += ah                # DOWN
        elif action == 4: x -= aw; y -= ah; w += 2*aw; h += 2*ah # BIGGER
        elif action == 5: x += aw; y += ah; w -= 2*aw; h -= 2*ah # SMALLER
        elif action == 6: x -= aw; w += 2*aw     # FATTER
        elif action == 7: y -= ah; h += 2*ah     # TALLER
        
        # Clip strictly within image boundaries
        x = np.clip(x, 0, self.W - 1)
        y = np.clip(y, 0, self.H - 1)
        w = np.clip(w, 10, self.W - x)
        h = np.clip(h, 10, self.H - y)
        
        return np.array([x, y, w, h], dtype=np.float32)

    def get_positive_actions(self):
        """Apprenticeship Learning: restituisce le azioni che aumentano l'IoU."""
        positive = []
        for a in range(N_ACTIONS - 1): # Escludi il trigger
            new_box = self._apply_action(self.box, a)
            if compute_iou(new_box, self.gt_box) > self.previous_iou:
                positive.append(a)
        
        # Verifica se il trigger è un'azione positiva (IoU >= tau)
        if self.previous_iou >= TAU_IOU:
            positive.append(8)
            
        return positive

    def step(self, action):
        self.current_step += 1
        
        # Aggiorna history one-hot
        one_hot = [0]*N_ACTIONS
        one_hot[action] = 1
        self.history.extend(one_hot)

        terminated = False
        truncated = self.current_step >= MAX_STEPS_PER_EPISODE

        if action == 8: # TRIGGER
            iou = compute_iou(self.box, self.gt_box)
            reward = TRIGGER_REWARD if iou >= TAU_IOU else -TRIGGER_REWARD
            terminated = True
        else:
            self.box = self._apply_action(self.box, action)
            new_iou = compute_iou(self.box, self.gt_box)
            
            # Eq 2: Reward binario basato sul segno del delta IoU
            if new_iou > self.previous_iou:
                reward = REWARD_POSITIVE
            else:
                reward = REWARD_NEGATIVE
                
            self.previous_iou = new_iou

        return self._get_obs(), reward, terminated, truncated, {}
    
    
    # In environment.py, aggiungi questo metodo
    def clear_memory(self):
        """Rilascia esplicitamente i riferimenti ai dati pesanti."""
        if hasattr(self, 'current_image'):
            del self.current_image
        # Se hai altre strutture pesanti, puliscile qui
        import gc
        gc.collect()