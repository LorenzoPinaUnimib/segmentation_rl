import os
import cv2
import gymnasium as gym
import numpy as np
import torch
from gymnasium  import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CallbackList,
    CheckpointCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

# Importiamo la funzione dal tuo file dataset.py
from dataset import get_datasets


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def linear_schedule(initial_value: float, final_value: float = 0.0):
    """Schedule lineare per learning_rate / clip_range.
    SB3 chiama la funzione con 'progress_remaining' che va da 1 (inizio) a 0 (fine).
    """
    def scheduler(progress_remaining: float) -> float:
        return final_value + progress_remaining * (initial_value - final_value)
    return scheduler


def compute_action_steps(W: int, H: int):
    """Calcola due granularita' di movimento (coarse/fine) proporzionali alla
    dimensione dell'immagine, cosi' l'agente puo' sia esplorare velocemente sia
    rifinire il box con precisione per avvicinarsi a IoU alti.
    """
    big_step = max(6.0, 0.045 * ((W + H) / 2.0))
    small_step = max(2.0, big_step / 3.0)
    return big_step, small_step


# Numero di azioni discrete: 6 movimenti "coarse" + 6 movimenti "fine" + 1 STOP
N_ACTIONS = 13
STOP_ACTION = 12


class VisionMetricsCallback(BaseCallback):
    """Callback modificata per strutturare i 4 grafici personalizzati richiesti su TensorBoard."""

    def __init__(self, val_dataset, verbose=0, save_dir="./ppo_gradcam_outputs"):
        super(VisionMetricsCallback, self).__init__(verbose)
        self.val_dataset = val_dataset
        self.save_dir = save_dir
        self.iteration_count = 0
        os.makedirs(self.save_dir, exist_ok=True)

        # Selezione campione fisso per la validazione visiva
        self.fixed_sample = None
        for i in range(len(self.val_dataset)):
            sample = self.val_dataset[i]
            mask = sample["mask"].numpy().squeeze(0)
            pos = np.where(mask > 0.5)
            if len(pos[0]) > 0:
                self.fixed_sample = sample
                ymin, ymax = np.min(pos[0]), np.max(pos[0])
                xmin, xmax = np.min(pos[1]), np.max(pos[1])
                self.fixed_gt_box = np.array([xmin, ymin, xmax - xmin, ymax - ymin], dtype=np.float32)
                break

        if self.fixed_sample is None:
            self.fixed_sample = self.val_dataset[0]
            _, H, W = self.fixed_sample["image"].shape
            self.fixed_gt_box = np.array([W/4, H/4, W/2, H/2], dtype=np.float32)

        _, self.H, self.W = self.fixed_sample["image"].shape
        self.fixed_initial_box = np.array([self.W/3, self.H/3, self.W/3, self.H/3], dtype=np.float32)
        self.big_step, self.small_step = compute_action_steps(self.W, self.H)

    def _on_training_start(self) -> None:
        """Crea il layout a 4 grafici richiesto dall'utente all'avvio del modello."""
        output_formats = self.logger.output_formats
        from stable_baselines3.common.logger import TensorBoardOutputFormat
        
        for fmt in output_formats:
            if isinstance(fmt, TensorBoardOutputFormat):
                # Definiamo la disposizione esatta dei grafici nella tab "Custom Scalars"
                layout = {
                    "Analisi_Integrazione_RL": {
                        "1_IoU_Media_Std_Finale": ["Multiline", ["custom_plots/1_iou_mean", "custom_plots/1_iou_std", "custom_plots/1_iou_final"]],
                        "2_Delta_IoU_per_step": ["Multiline", ["custom_plots/2_delta_iou"]],
                        "3_Terminal_Bonus": ["Multiline", ["custom_plots/3_terminal_bonus"]],
                        "4_Reward_Completa": ["Multiline", ["custom_plots/4_total_reward"]],
                    }
                }
                fmt.writer.add_custom_scalars(layout)
                print("[TensorBoard] Layout a 4 grafici personalizzati registrato con successo!")
                break

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if infos is not None and len(infos) > 0:
            # FIX: prima si leggeva solo infos[0], cioe' un solo sub-environment
            # su N_ENVS paralleli (SubprocVecEnv). I grafici mostravano quindi
            # l'andamento di UN SOLO env su 4, rumoroso e non rappresentativo
            # della media reale della policy. Ora aggreghiamo su tutti gli info
            # disponibili in questo step, su tutti gli env paralleli.

            # Componenti di reward: media su tutti gli env che le hanno prodotte
            delta_vals, terminal_vals, total_vals = [], [], []
            for info in infos:
                comp = info.get("rew_components")
                if comp is not None:
                    delta_vals.append(float(comp["delta_iou"]))
                    terminal_vals.append(float(comp["terminal_bonus"]))
                    total_vals.append(float(comp["total"]))
            if delta_vals:
                self.logger.record("custom_plots/2_delta_iou", float(np.mean(delta_vals)))
                self.logger.record("custom_plots/3_terminal_bonus", float(np.mean(terminal_vals)))
                self.logger.record("custom_plots/4_total_reward", float(np.mean(total_vals)))

            # Metriche di fine episodio (Media, Deviazione Standard, IoU finale):
            # media su tutti gli episodi terminati in questo step, su tutti gli env
            mean_vals, std_vals, final_vals = [], [], []
            for info in infos:
                metrics = info.get("episode_metrics")
                if metrics is not None:
                    mean_vals.append(float(metrics["ep_iou_mean"]))
                    std_vals.append(float(metrics["ep_iou_std"]))
                    final_vals.append(float(metrics["ep_iou_final"]))
            if mean_vals:
                self.logger.record("custom_plots/1_iou_mean", float(np.mean(mean_vals)))
                self.logger.record("custom_plots/1_iou_std", float(np.mean(std_vals)))
                self.logger.record("custom_plots/1_iou_final", float(np.mean(final_vals)))
        return True

    def _on_rollout_end(self) -> None:
        self.iteration_count += 1
        if self.iteration_count == 1 or self.iteration_count % 20 == 0:
            self.generate_and_save_gradcam()

    def generate_and_save_gradcam(self):
        # 1. Costruiamo il 4° canale (il Box) per il campione fisso nello spazio geometrico
        box_mask = np.zeros((self.H, self.W), dtype=np.uint8)
        
        # Estraiamo le coordinate correnti (formato centro: cx, cy, w, h)
        cx, cy, w, h = self.fixed_initial_box[0], self.fixed_initial_box[1], self.fixed_initial_box[2], self.fixed_initial_box[3]
        x1, y1 = int(cx - w / 2.0), int(cy - h / 2.0)
        x2, y2 = int(cx + w / 2.0), int(cy + h / 2.0)
        
        cv2.rectangle(box_mask, (x1, y1), (x2, y2), 255, thickness=-1)
        box_mask_channel = np.expand_dims(box_mask, axis=0)

        # 2. Prepariamo l'immagine a 4 canali (Numpy uint8 per model.predict)
        img_uint8 = (self.fixed_sample["image"].numpy() * 255).astype(np.uint8)
        fixed_obs = np.concatenate([img_uint8, box_mask_channel], axis=0)

        # 3. Prepariamo il tensore a 4 canali per PyTorch (Aggiungendo la dimensione del batch -> 1, 4, H, W)
        obs_tensor = torch.tensor(np.expand_dims(fixed_obs, axis=0), dtype=torch.float32).to(self.model.device)

        with torch.enable_grad():
            self.model.policy.eval()
            activations, gradients = None, None

            def forward_hook(module, input, output): nonlocal activations; activations = output
            def backward_hook(module, grad_input, grad_output): nonlocal gradients; gradients = grad_output[0]

            # Target layer corretto per CnnPolicy (NatureCNN)
            target_layer = self.model.policy.features_extractor.cnn[4]
            
            h1 = target_layer.register_forward_hook(forward_hook)
            h2 = target_layer.register_full_backward_hook(backward_hook)

            # Esecuzione del calcolo del valore sul tensore puro a 4 canali
            values = self.model.policy.predict_values(obs_tensor)
            self.model.policy.zero_grad()
            values.sum().backward()
            h1.remove(); h2.remove()

        if activations is not None and gradients is not None:
            pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
            for i in range(activations.shape[1]):
                activations[:, i, :, :] *= pooled_gradients[i]

            heatmap = torch.mean(activations, dim=1).squeeze(0)
            heatmap = torch.max(heatmap, torch.zeros_like(heatmap))
            if torch.max(heatmap) > 0:
                heatmap /= torch.max(heatmap)

            heatmap = heatmap.cpu().detach().numpy()
            orig_img = self.fixed_sample["image"].numpy()
            if orig_img.shape[0] == 3:
                orig_img = np.transpose(orig_img, (1, 2, 0))
                orig_img = cv2.cvtColor((orig_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            else:
                orig_img = cv2.cvtColor((orig_img[0] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

            heatmap_resized = cv2.resize(heatmap, (self.W, self.H))
            heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(orig_img, 0.6, heatmap_colored, 0.4, 0)

            gt = self.fixed_gt_box
            cv2.rectangle(overlay, (int(gt[0]), int(gt[1])), (int(gt[0]+gt[2]), int(gt[1]+gt[3])), (0, 255, 0), 2)
            
            with torch.no_grad():
                # Il modello riceve l'osservazione a 4 canali nativa, senza dizionari
                action, _ = self.model.predict(fixed_obs, deterministic=True)
            
            # Applichiamo la logica dell'azione discreta (coarse/fine + STOP) per aggiornare il disegno visivo
            bs, ss = self.big_step, self.small_step
            if action == 0:   self.fixed_initial_box[0] -= bs   # cx - coarse
            elif action == 1: self.fixed_initial_box[0] += bs   # cx + coarse
            elif action == 2: self.fixed_initial_box[1] -= bs   # cy - coarse
            elif action == 3: self.fixed_initial_box[1] += bs   # cy + coarse
            elif action == 4:
                self.fixed_initial_box[2] += bs; self.fixed_initial_box[3] += bs  # w, h + coarse
            elif action == 5:
                self.fixed_initial_box[2] -= bs; self.fixed_initial_box[3] -= bs  # w, h - coarse
            elif action == 6:  self.fixed_initial_box[0] -= ss   # cx - fine
            elif action == 7:  self.fixed_initial_box[0] += ss   # cx + fine
            elif action == 8:  self.fixed_initial_box[1] -= ss   # cy - fine
            elif action == 9:  self.fixed_initial_box[1] += ss   # cy + fine
            elif action == 10:
                self.fixed_initial_box[2] += ss; self.fixed_initial_box[3] += ss  # w, h + fine
            elif action == 11:
                self.fixed_initial_box[2] -= ss; self.fixed_initial_box[3] -= ss  # w, h - fine
            # action == STOP_ACTION (12): nessun movimento, il box resta fermo

            # Limiti di sicurezza per il disegno
            self.fixed_initial_box[2] = np.clip(self.fixed_initial_box[2], 12, self.W)
            self.fixed_initial_box[3] = np.clip(self.fixed_initial_box[3], 12, self.H)
            self.fixed_initial_box[0] = np.clip(self.fixed_initial_box[0], self.fixed_initial_box[2]/2, self.W - self.fixed_initial_box[2]/2)
            self.fixed_initial_box[1] = np.clip(self.fixed_initial_box[1], self.fixed_initial_box[3]/2, self.H - self.fixed_initial_box[3]/2)

            # Conversione finale da centro a standard [xmin, ymin] per cv2.rectangle
            draw_x = self.fixed_initial_box[0] - self.fixed_initial_box[2] / 2.0
            draw_y = self.fixed_initial_box[1] - self.fixed_initial_box[3] / 2.0

            cv2.rectangle(overlay, (int(draw_x), int(draw_y)), (int(draw_x+self.fixed_initial_box[2]), int(draw_y+self.fixed_initial_box[3])), (0, 0, 255), 2)
            save_path = os.path.join(self.save_dir, f"gradcam_iter_{self.iteration_count}.png")
            cv2.imwrite(save_path, overlay)

class BrainTumorRL_Env(gym.Env):
    """Ambiente ottimizzato a 4 canali per risolvere la cecità spaziale della CNN."""
    metadata = {"render_modes": ["human"]}

    def __init__(self, pytorch_dataset, max_steps=256):
        super(BrainTumorRL_Env, self).__init__()
        self.dataset = pytorch_dataset
        self.max_steps = max_steps
        
        sample = self.dataset[0]
        # Assumiamo immagini a 3 canali dal dataset (es. Synthetic di dataset.py)
        self.channels, self.H, self.W = sample["image"].shape 

        # 1. Spazio delle azioni discreto: 6 movimenti coarse + 6 movimenti fine + 1 STOP.
        # Il doppio livello di granularita' e' cio' che permette all'agente di prima
        # avvicinarsi velocemente al target e poi rifinire il box con precisione,
        # necessario per superare IoU ~0.5-0.6 e avvicinarsi a 0.8.
        self.action_space = spaces.Discrete(N_ACTIONS)
        
        # 2. NUOVO SPAZIO OSSERVAZIONI: Un'unica immagine a 4 canali (3 canali MRI + 1 canale Box)
        # Se le tue immagini originali sono in scala di grigi (1 canale), qui metterai shape=(2, self.H, self.W)
        self.observation_space = spaces.Box(
            low=0, 
            high=255, 
            shape=(self.channels + 1, self.H, self.W), 
            dtype=np.uint8
        )
        
        self.big_step, self.small_step = compute_action_steps(self.W, self.H)
        self.max_diag = np.sqrt(self.W**2 + self.H**2)

    # ... (il metodo reset rimane invariato nella logica, inizializza cx, cy, w, h) ...
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_idx = self.np_random.integers(0, len(self.dataset))
        sample = self.dataset[self.current_idx]

        self.current_image = (sample["image"].numpy() * 255).astype(np.uint8)
        mask = sample["mask"].numpy().squeeze(0)

        pos = np.where(mask > 0.5)
        if len(pos[0]) > 0:
            ymin, ymax = np.min(pos[0]), np.max(pos[0])
            xmin, xmax = np.min(pos[1]), np.max(pos[1])
            self.gt_box = np.array([xmin, ymin, xmax - xmin, ymax - ymin], dtype=np.float32)
        else:
            self.gt_box = np.array([self.W/4, self.H/4, self.W/2, self.H/2], dtype=np.float32)

        # Inizializzazione randomizzata di centro E dimensione del box iniziale:
        # nella versione originale w,h erano sempre fissi a W/4,H/4, il che riduceva
        # la diversita' delle traiettorie viste in training e favoriva l'overfitting
        # a una singola strategia di ridimensionamento.
        self.cx = self.W / 2.0 + self.np_random.uniform(-30, 30)
        self.cy = self.H / 2.0 + self.np_random.uniform(-30, 30)
        self.w = self.W * self.np_random.uniform(0.20, 0.45)
        self.h = self.H * self.np_random.uniform(0.20, 0.45)
        
        self.current_step = 0
        self.episode_ious = []
        
        current_box_xywh = self._get_xywh_box()
        self.previous_iou = self._compute_iou(current_box_xywh, self.gt_box)

        return self._get_obs(), {}

    def _get_xywh_box(self):
        xmin = self.cx - self.w / 2.0
        ymin = self.cy - self.h / 2.0
        return np.array([xmin, ymin, self.w, self.h], dtype=np.float32)

    def _get_obs(self):
        """Genera l'osservazione a 4 canali iniettando il box geometrico nei pixel."""
        # Creiamo il 4° canale: una matrice vuota (H, W) riempita di zeri
        box_mask = np.zeros((self.H, self.W), dtype=np.uint8)

        # Recuperiamo i punti del box corrente in formato xywh
        box_xywh = self._get_xywh_box()
        x1, y1 = int(box_xywh[0]), int(box_xywh[1])
        x2, y2 = int(box_xywh[0] + box_xywh[2]), int(box_xywh[1] + box_xywh[3])

        # Disegnamo un rettangolo pieno (thickness=-1) con intensità 255
        cv2.rectangle(box_mask, (x1, y1), (x2, y2), 255, thickness=-1)

        # Aggiungiamo la dimensione del canale alla maschera del box -> (1, H, W)
        box_mask_channel = np.expand_dims(box_mask, axis=0)

        return np.concatenate([self.current_image, box_mask_channel], axis=0)

    def _compute_iou(self, b1, b2):
        xi1, yi1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        xi2, yi2 = min(b1[0]+b1[2], b2[0]+b2[2]), min(b1[1]+b1[3], b2[1]+b2[3])
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union_area = (b1[2]*b1[3]) + (b2[2]*b2[3]) - inter_area
        return inter_area / max(1e-6, union_area)

    def _compute_center_distance(self, cx, cy, b2):
        gt_cx = b2[0] + b2[2] / 2.0
        gt_cy = b2[1] + b2[3] / 2.0
        return np.linalg.norm(np.array([cx, cy]) - np.array([gt_cx, gt_cy]))

    def step(self, action):
        self.current_step += 1
        terminated = False

        bs, ss = self.big_step, self.small_step

        # Esecuzione dell'azione discreta scelta dalla policy (coarse + fine + STOP)
        if action == 0:   self.cx -= bs                # Sinistra (coarse)
        elif action == 1: self.cx += bs                # Destra (coarse)
        elif action == 2: self.cy -= bs                # Su (coarse)
        elif action == 3: self.cy += bs                # Giu' (coarse)
        elif action == 4:                               # Ingrandisci (coarse), centro fisso
            self.w += bs; self.h += bs
        elif action == 5:                               # Rimpicciolisci (coarse), centro fisso
            self.w -= bs; self.h -= bs
        elif action == 6:  self.cx -= ss                # Sinistra (fine)
        elif action == 7:  self.cx += ss                # Destra (fine)
        elif action == 8:  self.cy -= ss                # Su (fine)
        elif action == 9:  self.cy += ss                # Giu' (fine)
        elif action == 10:                              # Ingrandisci (fine)
            self.w += ss; self.h += ss
        elif action == 11:                              # Rimpicciolisci (fine)
            self.w -= ss; self.h -= ss
        elif action == STOP_ACTION:                     # AZIONE DI STOP ESPLICITA
            terminated = True

        # Clip di sicurezza per evitare che il box esca o collassi
        self.w = np.clip(self.w, 12, self.W)
        self.h = np.clip(self.h, 12, self.H)
        self.cx = np.clip(self.cx, self.w/2, self.W - self.w/2)
        self.cy = np.clip(self.cy, self.h/2, self.H - self.h/2)

        # Calcolo metriche correnti
        current_box_xywh = self._get_xywh_box()
        iou = self._compute_iou(current_box_xywh, self.gt_box)
        self.episode_ious.append(iou)

        # ─── REWARD SHAPING BASATO DIRETTAMENTE SULL'IoU ───────────────────────
        # Il reward shaping precedente (distanza dal centro + errore di scala)
        # era solo debolmente correlato all'IoU: un agente poteva massimizzarlo
        # senza migliorare l'IoU reale. Premiare direttamente il delta di IoU
        # (approccio standard nella letteratura di active localization, es.
        # Caicedo & Lazebnik 2015) allinea l'obiettivo di training alla metrica
        # che vogliamo davvero ottimizzare.
        #
        # Ogni componente e' tenuta separata (invece di sommarla subito dentro
        # 'reward') cosi' la callback puo' loggarle su grafici distinti in
        # TensorBoard e si puo' capire a colpo d'occhio quale parte del reward
        # sta guidando (o frenando) l'apprendimento.
        delta_iou = iou - self.previous_iou
        delta_iou_component = delta_iou * 100.0   # segnale denso ad ogni step
        time_penalty_component = -0.02            # piccola penalita' di tempo per incentivare efficienza
        terminal_bonus = 0.0

        if terminated:
            # FIX: la versione precedente aveva una scogliera netta a IoU=0.5:
            # +iou*60 sopra la soglia, -15 fisso sotto. All'inizio del training
            # l'IoU medio e' ~0.1-0.15 (vedi custom_plots/1_iou_mean), quindi
            # raggiungere 0.5 e' quasi impossibile e l'agente prende quasi
            # sempre -15 quando chiama STOP. La strategia ottimale che PPO
            # impara e' allora smettere di chiamare STOP del tutto (per non
            # rischiare -15) e lasciare che l'episodio vada sempre in timeout,
            # il che spiega perche' 'terminal_bonus' resta vicino/sotto zero
            # per quasi tutto il training e non spinge mai la policy verso IoU
            # migliori.
            #
            # Ora il bonus/penalita' e' continuo e centrato su una soglia
            # ragionevole (0.3): l'agente riceve un segnale proporzionale a
            # quanto e' sopra o sotto la soglia, senza salti bruschi, e puo'
            # quindi imparare gradualmente che fermarsi con IoU 0.2 e' meglio
            # di fermarsi con IoU 0.05, invece di essere punito allo stesso
            # modo (-15) in entrambi i casi.
            terminal_bonus = (iou - 0.3) * 40.0

        self.previous_iou = iou
        truncated = self.current_step >= self.max_steps

        if truncated and not terminated:
            # Anche se l'episodio finisce per timeout (senza STOP esplicito),
            # diamo comunque un credito finale proporzionale all'IoU raggiunto,
            # cosi' la qualita' del box finale conta sempre. Usiamo la stessa
            # scala continua del ramo STOP (ma dimezzata: il timeout non e' una
            # decisione attiva dell'agente, quindi non deve essere ne' premiato
            # ne' punito quanto una STOP esplicita corretta/sbagliata).
            terminal_bonus = (iou - 0.3) * 20.0

        reward = delta_iou_component + time_penalty_component + terminal_bonus

        info = {
            "iou_instant": float(iou),
            "rew_components": {
                "delta_iou": float(delta_iou_component),
                "time_penalty": float(time_penalty_component),
                "terminal_bonus": float(terminal_bonus),
                "total": float(reward),
            }
        }

        if terminated or truncated:
            info["episode_metrics"] = {
                "ep_iou_mean": float(np.mean(self.episode_ious)),
                "ep_iou_std": float(np.std(self.episode_ious)),
                "ep_iou_final": float(iou),
            }

        return self._get_obs(), float(reward), terminated, truncated, info


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE TRAINING
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = {
        "dataset": {
            "source": "kaggle",
            "kaggle_id": "pkdarabi/brain-tumor-image-dataset-semantic-segmentation",
            "image_size": [224, 224],
            "in_channels": 3,
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "cache_pairs": False,
        },
        "preprocessing": {"normalization": "minmax", "binarize_mask": True, "mask_threshold": 0.5},
        "training": {"batch_size": 16, "num_workers": 0},
        "output": {"root": "./output"},
        "seed": 42,
    }

    train_ds, val_ds, test_ds = get_datasets(cfg)

    # Lunghezza massima di un episodio: 256 step per un box gia' abbastanza vicino
    # al target erano eccessivi (episodi troppo lunghi = pochi episodi per rollout,
    # segnale di reward molto diluito). 100 step sono piu' che sufficienti con lo
    # spazio azioni coarse+fine e favoriscono un training molto piu' denso.
    MAX_STEPS_PER_EPISODE = 200

    # Verifica di conformita' Gymnasium su un'istanza singola dell'ambiente
    check_env(BrainTumorRL_Env(pytorch_dataset=train_ds, max_steps=MAX_STEPS_PER_EPISODE))

    # ─── ENVIRONMENT VETTORIZZATI ──────────────────────────────────────────────
    # Il bug piu' impattante della versione originale: un solo environment e
    # solo 150 * 256 = 38.400 timesteps totali. E' un budget che con un'immagine
    # come input e una CNN da allenare da zero non arriva nemmeno a stabilizzare
    # la policy. Vettorizzare su piu' processi (SubprocVecEnv) permette di
    # raccogliere molta piu' esperienza per unita' di tempo, ed e' lo standard
    # per allenare PPO su policy CNN.
    #
    # Su Windows l'avvio di ogni sotto-processo e' LENTO (spawn deve reimportare
    # torch/cv2/albumentations/tensorboard in ognuno, spesso 10-15s a processo
    # per via dell'overhead di creazione processi + antivirus). Non e' un loop
    # rotto: e' semplicemente l'avvio sequenziale degli N_ENVS worker. Se vuoi
    # verificarlo, metti USE_SUBPROCESS=False per allenare con un singolo
    # processo (DummyVecEnv): se il training parte subito, conferma che il
    # "loop" era solo overhead di avvio dei sotto-processi.
    USE_SUBPROCESS = True
    N_ENVS = 4  # ridotto da 8: meno processi da avviare = avvio piu' rapido su Windows

    def make_env():
        def _init():
            return BrainTumorRL_Env(pytorch_dataset=train_ds, max_steps=MAX_STEPS_PER_EPISODE)
        return _init

    print(f"[setup] Creazione di {N_ENVS} environment "
          f"({'SubprocVecEnv' if USE_SUBPROCESS else 'DummyVecEnv'})... "
          f"su Windows puo' richiedere fino a ~15s per processo, attendere.", flush=True)

    if USE_SUBPROCESS:
        vec_env = SubprocVecEnv([make_env() for _ in range(N_ENVS)])
    else:
        from stable_baselines3.common.vec_env import DummyVecEnv
        vec_env = DummyVecEnv([make_env() for _ in range(N_ENVS)])

    vec_env = VecMonitor(vec_env)
    print("[setup] Environment vettorizzati pronti.", flush=True)

    eval_env = Monitor(BrainTumorRL_Env(pytorch_dataset=val_ds, max_steps=MAX_STEPS_PER_EPISODE))

    N_STEPS = 1024  # step di rollout per singolo env -> buffer totale = N_STEPS * N_ENVS = 4096

    # Budget di training realistico: con l'ambiente e la policy CNN attuali,
    # servono ordini di grandezza in piu' rispetto alle 38k transizioni originali
    # per avvicinarsi a IoU ~0.8. 2M e' un punto di partenza ragionevole;
    # monitora TensorBoard (custom_plots/1_iou_mean) e allunga se la curva
    # non si e' ancora appiattita.
    TOTAL_TIMESTEPS = 2_000_000

    visual_callback = VisionMetricsCallback(val_dataset=val_ds)

    stop_on_no_improve = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=15, min_evals=20, verbose=1
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./ppo_brain_tumor_logs/best_model/",
        log_path="./ppo_brain_tumor_logs/eval_results/",
        eval_freq=max(N_STEPS * 4, 2000),
        n_eval_episodes=20,
        deterministic=True,
        callback_after_eval=stop_on_no_improve,
        verbose=1,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(N_STEPS * 20, 10000),
        save_path="./ppo_brain_tumor_logs/checkpoints/",
        name_prefix="ppo_brain_tumor",
    )

    callbacks = CallbackList([visual_callback, eval_callback, checkpoint_callback])

    model = PPO(
        policy="CnnPolicy",
        env=vec_env,
        # Learning rate e clip range decrescenti: alto all'inizio per esplorare
        # velocemente lo spazio delle azioni, basso verso la fine per rifinire
        # la policy senza destabilizzarla (utile per superare il plateau attorno
        # a IoU 0.5-0.6 tipico di questi task).
        learning_rate=linear_schedule(3e-4, 1e-5),
        n_steps=N_STEPS,
        batch_size=256,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        # FIX: con 13 azioni discrete, una CnnPolicy allenata da zero e un
        # reward shaping non banale, il clip_range decadeva troppo in fretta
        # verso 0.05 (in soli 2M step), riducendo l'esplorazione prima che la
        # policy avesse davvero imparato una strategia decente. Rallentiamo il
        # decay mantenendo un clip_range finale piu' alto.
        clip_range=linear_schedule(0.2, 0.1),
        # FIX: ent_coef=0.01 e' basso per un action space da 13 azioni discrete
        # con reward sparso/rumoroso; la policy collassava rapidamente verso
        # un comportamento quasi deterministico (vedi calo simultaneo di
        # iou_mean E iou_std nei grafici, sintomo classico di policy collapse).
        # Alziamo l'entropia per mantenere piu' esplorazione piu' a lungo.
        ent_coef=0.03,   # incoraggia l'esplorazione su 13 azioni discrete
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="./ppo_brain_tumor_logs/",
    )

    print(f"Avvio addestramento PPO ({N_ENVS} env paralleli, {TOTAL_TIMESTEPS} timesteps totali)...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks)

    model.save("./ppo_brain_tumor_logs/final_model")
    print("Training completato. Modello finale salvato in ./ppo_brain_tumor_logs/final_model")