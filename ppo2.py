import os
import csv
import cv2
import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
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
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor, VecNormalize

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

# ─────────────────────────────────────────────────────────────────────────────
# REWARD SHAPING — parametri centralizzati
# ─────────────────────────────────────────────────────────────────────────────
# FIX PRINCIPALE rispetto alla versione precedente: il vecchio terminal_bonus
# era (iou - 0.3) * 40, cioe' una "scogliera" quasi sempre negativa quando la
# IoU media reale del training e' 0.1-0.2 (vedi custom_plots/1_iou_mean nei
# grafici forniti). Un reward terminale quasi sempre negativo:
#   1) non da' MAI un segnale chiaro "questa IoU e' meglio di quella" quando
#      entrambe sono sotto soglia (sono solo "meno negative" l'una dell'altra,
#      un gradiente debolissimo all'inizio del training);
#   2) disincentiva la policy dal chiamare STOP esplicitamente (l'azione
#      "sicura" diventa non-fermarsi mai, il timeout da' comunque meta' pena),
#      il che spiega perche' terminal_bonus restava vicino/sotto zero per
#      tutto il training osservato e la policy collassava (iou_mean E iou_std
#      scendono insieme = collasso, non convergenza).
#
# La nuova formulazione e' PROPORZIONALE E SEMPRE NON-NEGATIVA rispetto alla
# IoU raggiunta: non esiste soglia sotto la quale scatta una penalita' fissa.
# Il segnale e' quindi denso e monotono in ogni fase del training, dal primo
# step (IoU~0.05) fino a IoU alte (0.8+), invece di essere utile solo a
# training gia' avanzato.
DELTA_IOU_SCALE = 25.0          # era 100: riduce la varianza del segnale dominante
DELTA_IOU_CLIP = 10.0           # clip per evitare spike da salti di IoU anomali
TIME_PENALTY = -0.01            # leggera pressione all'efficienza

TERMINAL_SCALE_STOP = 60.0      # STOP esplicito: iou * questo fattore
TERMINAL_SCALE_TIMEOUT = 40.0   # timeout: iou * questo fattore (piu' basso: STOP attivo e' premiato di piu')

# Bonus a scaglioni: spinta extra man mano che ci si avvicina a un box "buono",
# senza mai introdurre un valore negativo. Aiuta a superare i plateau tipici
# 0.5 -> 0.7 -> 0.8 dei task di active localization.
IOU_MILESTONES = [(0.5, 5.0), (0.7, 10.0), (0.8, 20.0)]

# FIX (robustezza): senza un minimo di step obbligatori, un episodio puo'
# terminare per puro caso con pochissimi step. All'inizio del training la
# policy e' quasi uniforme su 13 azioni (alta entropia voluta), quindi la
# probabilita' di pescare STOP ad ogni step e' ~1/13: la lunghezza media
# attesa di un episodio "casuale" e' infatti ~13 step, che e' esattamente
# quello che si osserva nei primi log (ep_len_mean 7-13, stop_rate=1). Non e'
# un bug del reward, ma e' comunque un problema pratico: episodi cosi' corti
# non danno all'agente il tempo di esplorare/rifinire il box, quindi il
# segnale di delta_iou resta povero troppo a lungo. Blocchiamo STOP come
# no-op per i primi MIN_STEPS_BEFORE_STOP step: l'episodio continua (il box
# resta fermo quello step, piccola time_penalty) finche' non e' passato un
# numero minimo di step, garantendo che ogni episodio dia effettivamente
# occasione di migliorare la IoU prima di poter chiudere.
MIN_STEPS_BEFORE_STOP = 20  # 20% di MAX_STEPS_PER_EPISODE=100


class VisionMetricsCallback(BaseCallback):
    """Struttura i grafici personalizzati su TensorBoard e salva Grad-CAM periodici
    di un campione fisso durante il training (per monitorare visivamente l'evoluzione)."""

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
            self.fixed_gt_box = np.array([W / 4, H / 4, W / 2, H / 2], dtype=np.float32)

        _, self.H, self.W = self.fixed_sample["image"].shape
        self.fixed_initial_box = np.array([self.W / 3, self.H / 3, self.W / 3, self.H / 3], dtype=np.float32)
        self.big_step, self.small_step = compute_action_steps(self.W, self.H)

    def _on_training_start(self) -> None:
        """Crea il layout a 5 grafici richiesto dall'utente all'avvio del modello
        (aggiunto il 5° grafico: tasso di STOP esplicito, diagnostico chiave del fix)."""
        output_formats = self.logger.output_formats
        from stable_baselines3.common.logger import TensorBoardOutputFormat

        for fmt in output_formats:
            if isinstance(fmt, TensorBoardOutputFormat):
                layout = {
                    "Analisi_Integrazione_RL": {
                        "1_IoU_Media_Std_Finale": ["Multiline", ["custom_plots/1_iou_mean", "custom_plots/1_iou_std", "custom_plots/1_iou_final"]],
                        "2_Delta_IoU_per_step": ["Multiline", ["custom_plots/2_delta_iou"]],
                        "3_Terminal_Bonus": ["Multiline", ["custom_plots/3_terminal_bonus"]],
                        "4_Reward_Completa": ["Multiline", ["custom_plots/4_total_reward"]],
                        "5_Tasso_STOP_Esplicito": ["Multiline", ["custom_plots/5_stop_rate"]],
                    }
                }
                fmt.writer.add_custom_scalars(layout)
                print("[TensorBoard] Layout a 5 grafici personalizzati registrato con successo!")
                break

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if infos is not None and len(infos) > 0:
            # Media su tutti gli env paralleli (SubprocVecEnv), non solo infos[0]
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

            mean_vals, std_vals, final_vals, stop_vals = [], [], [], []
            for info in infos:
                metrics = info.get("episode_metrics")
                if metrics is not None:
                    mean_vals.append(float(metrics["ep_iou_mean"]))
                    std_vals.append(float(metrics["ep_iou_std"]))
                    final_vals.append(float(metrics["ep_iou_final"]))
                    stop_vals.append(float(metrics["ep_stopped_explicitly"]))
            if mean_vals:
                self.logger.record("custom_plots/1_iou_mean", float(np.mean(mean_vals)))
                self.logger.record("custom_plots/1_iou_std", float(np.mean(std_vals)))
                self.logger.record("custom_plots/1_iou_final", float(np.mean(final_vals)))
                self.logger.record("custom_plots/5_stop_rate", float(np.mean(stop_vals)))
        return True

    def _on_rollout_end(self) -> None:
        self.iteration_count += 1
        if self.iteration_count == 1 or self.iteration_count % 20 == 0:
            self.generate_and_save_gradcam()

    def generate_and_save_gradcam(self):
        box_mask = np.zeros((self.H, self.W), dtype=np.uint8)

        cx, cy, w, h = self.fixed_initial_box[0], self.fixed_initial_box[1], self.fixed_initial_box[2], self.fixed_initial_box[3]
        x1, y1 = int(cx - w / 2.0), int(cy - h / 2.0)
        x2, y2 = int(cx + w / 2.0), int(cy + h / 2.0)

        cv2.rectangle(box_mask, (x1, y1), (x2, y2), 255, thickness=-1)
        box_mask_channel = np.expand_dims(box_mask, axis=0)

        img_uint8 = (self.fixed_sample["image"].numpy() * 255).astype(np.uint8)
        fixed_obs = np.concatenate([img_uint8, box_mask_channel], axis=0)

        obs_tensor = torch.tensor(np.expand_dims(fixed_obs, axis=0), dtype=torch.float32).to(self.model.device)

        with torch.enable_grad():
            self.model.policy.eval()
            activations, gradients = None, None

            def forward_hook(module, input, output): nonlocal activations; activations = output
            def backward_hook(module, grad_input, grad_output): nonlocal gradients; gradients = grad_output[0]

            target_layer = self.model.policy.features_extractor.cnn[4]

            h1 = target_layer.register_forward_hook(forward_hook)
            h2 = target_layer.register_full_backward_hook(backward_hook)

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
            cv2.rectangle(overlay, (int(gt[0]), int(gt[1])), (int(gt[0] + gt[2]), int(gt[1] + gt[3])), (0, 255, 0), 2)

            with torch.no_grad():
                action, _ = self.model.predict(fixed_obs, deterministic=True)

            bs, ss = self.big_step, self.small_step
            if action == 0:   self.fixed_initial_box[0] -= bs
            elif action == 1: self.fixed_initial_box[0] += bs
            elif action == 2: self.fixed_initial_box[1] -= bs
            elif action == 3: self.fixed_initial_box[1] += bs
            elif action == 4:
                self.fixed_initial_box[2] += bs; self.fixed_initial_box[3] += bs
            elif action == 5:
                self.fixed_initial_box[2] -= bs; self.fixed_initial_box[3] -= bs
            elif action == 6:  self.fixed_initial_box[0] -= ss
            elif action == 7:  self.fixed_initial_box[0] += ss
            elif action == 8:  self.fixed_initial_box[1] -= ss
            elif action == 9:  self.fixed_initial_box[1] += ss
            elif action == 10:
                self.fixed_initial_box[2] += ss; self.fixed_initial_box[3] += ss
            elif action == 11:
                self.fixed_initial_box[2] -= ss; self.fixed_initial_box[3] -= ss

            self.fixed_initial_box[2] = np.clip(self.fixed_initial_box[2], 12, self.W)
            self.fixed_initial_box[3] = np.clip(self.fixed_initial_box[3], 12, self.H)
            self.fixed_initial_box[0] = np.clip(self.fixed_initial_box[0], self.fixed_initial_box[2] / 2, self.W - self.fixed_initial_box[2] / 2)
            self.fixed_initial_box[1] = np.clip(self.fixed_initial_box[1], self.fixed_initial_box[3] / 2, self.H - self.fixed_initial_box[3] / 2)

            draw_x = self.fixed_initial_box[0] - self.fixed_initial_box[2] / 2.0
            draw_y = self.fixed_initial_box[1] - self.fixed_initial_box[3] / 2.0

            cv2.rectangle(overlay, (int(draw_x), int(draw_y)), (int(draw_x + self.fixed_initial_box[2]), int(draw_y + self.fixed_initial_box[3])), (0, 0, 255), 2)
            save_path = os.path.join(self.save_dir, f"gradcam_iter_{self.iteration_count}.png")
            cv2.imwrite(save_path, overlay)


class EntropyScheduleCallback(BaseCallback):
    """FIX: SB3 non supporta uno schedule nativo per ent_coef (a differenza di
    learning_rate/clip_range che accettano una funzione). Con 13 azioni discrete
    e un reward shaping non banale, un ent_coef fisso o troppo basso porta a
    collasso rapido della policy (visibile nei grafici: iou_mean E iou_std
    scendono insieme). Con questo callback anneal manualmente ent_coef da un
    valore alto (molta esplorazione, utile a inizio training per non convergere
    subito su una strategia degenere) a uno basso (per rifinire la policy
    verso la fine, quando serve precisione più che esplorazione).
    """

    def __init__(self, initial_ent: float, final_ent: float, total_timesteps: int, verbose=0):
        super().__init__(verbose)
        self.initial_ent = initial_ent
        self.final_ent = final_ent
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        progress = min(1.0, self.num_timesteps / max(1, self.total_timesteps))
        new_ent = self.initial_ent + progress * (self.final_ent - self.initial_ent)
        self.model.ent_coef = new_ent
        if self.num_timesteps % 50_000 < self.training_env.num_envs:
            self.logger.record("custom_plots/6_ent_coef", float(new_ent))
        return True


class BrainTumorRL_Env(gym.Env):
    """Ambiente a 4 canali (3 MRI + 1 box) per la localizzazione del tumore via RL."""
    metadata = {"render_modes": ["human"]}

    def __init__(self, pytorch_dataset, max_steps=100):
        super(BrainTumorRL_Env, self).__init__()
        self.dataset = pytorch_dataset
        self.max_steps = max_steps

        sample = self.dataset[0]
        self.channels, self.H, self.W = sample["image"].shape

        self.action_space = spaces.Discrete(N_ACTIONS)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.channels + 1, self.H, self.W),
            dtype=np.uint8
        )

        self.big_step, self.small_step = compute_action_steps(self.W, self.H)
        self.max_diag = np.sqrt(self.W ** 2 + self.H ** 2)

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
            self.gt_box = np.array([self.W / 4, self.H / 4, self.W / 2, self.H / 2], dtype=np.float32)

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
        box_mask = np.zeros((self.H, self.W), dtype=np.uint8)

        box_xywh = self._get_xywh_box()
        x1, y1 = int(box_xywh[0]), int(box_xywh[1])
        x2, y2 = int(box_xywh[0] + box_xywh[2]), int(box_xywh[1] + box_xywh[3])

        cv2.rectangle(box_mask, (x1, y1), (x2, y2), 255, thickness=-1)

        box_mask_channel = np.expand_dims(box_mask, axis=0)

        return np.concatenate([self.current_image, box_mask_channel], axis=0)

    def _compute_iou(self, b1, b2):
        xi1, yi1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        xi2, yi2 = min(b1[0] + b1[2], b2[0] + b2[2]), min(b1[1] + b1[3], b2[1] + b2[3])
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union_area = (b1[2] * b1[3]) + (b2[2] * b2[3]) - inter_area
        return inter_area / max(1e-6, union_area)

    def step(self, action):
        self.current_step += 1
        terminated = False

        bs, ss = self.big_step, self.small_step

        if action == 0:   self.cx -= bs
        elif action == 1: self.cx += bs
        elif action == 2: self.cy -= bs
        elif action == 3: self.cy += bs
        elif action == 4:
            self.w += bs; self.h += bs
        elif action == 5:
            self.w -= bs; self.h -= bs
        elif action == 6:  self.cx -= ss
        elif action == 7:  self.cx += ss
        elif action == 8:  self.cy -= ss
        elif action == 9:  self.cy += ss
        elif action == 10:
            self.w += ss; self.h += ss
        elif action == 11:
            self.w -= ss; self.h -= ss
        elif action == STOP_ACTION:
            if self.current_step >= MIN_STEPS_BEFORE_STOP:
                terminated = True
            # se siamo sotto la soglia minima, STOP e' un no-op: il box non si
            # muove e l'episodio prosegue (vedi commento a MIN_STEPS_BEFORE_STOP)

        self.w = np.clip(self.w, 12, self.W)
        self.h = np.clip(self.h, 12, self.H)
        self.cx = np.clip(self.cx, self.w / 2, self.W - self.w / 2)
        self.cy = np.clip(self.cy, self.h / 2, self.H - self.h / 2)

        current_box_xywh = self._get_xywh_box()
        iou = self._compute_iou(current_box_xywh, self.gt_box)
        self.episode_ious.append(iou)

        # ─── REWARD SHAPING (vedi commento ai parametri DELTA_IOU_SCALE ecc. in testa al file) ───
        delta_iou = iou - self.previous_iou
        delta_iou_component = float(np.clip(delta_iou * DELTA_IOU_SCALE, -DELTA_IOU_CLIP, DELTA_IOU_CLIP))
        time_penalty_component = TIME_PENALTY
        terminal_bonus = 0.0

        truncated = self.current_step >= self.max_steps

        if terminated or truncated:
            scale = TERMINAL_SCALE_STOP if terminated else TERMINAL_SCALE_TIMEOUT
            terminal_bonus = iou * scale
            for milestone, bonus in IOU_MILESTONES:
                if iou >= milestone:
                    terminal_bonus += bonus

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

        self.previous_iou = iou

        if terminated or truncated:
            info["episode_metrics"] = {
                "ep_iou_mean": float(np.mean(self.episode_ious)),
                "ep_iou_std": float(np.std(self.episode_ious)),
                "ep_iou_final": float(iou),
                "ep_stopped_explicitly": bool(terminated),
            }

        return self._get_obs(), float(reward), terminated, truncated, info


# ─────────────────────────────────────────────────────────────────────────────
# VALUTAZIONE SU TEST SET — Grad-CAM + overlay box + metriche pixel-level
# ─────────────────────────────────────────────────────────────────────────────

class SingleSampleDataset:
    """Wrapper minimale per far vedere all'ambiente un solo campione alla volta,
    cosi' controlliamo esattamente quale immagine del test set viene processata."""

    def __init__(self, sample):
        self.sample = sample

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.sample


def compute_gradcam(model, obs_4ch_uint8, device):
    """Stessa logica Grad-CAM di VisionMetricsCallback, generalizzata a una
    singola osservazione a 4 canali (numpy uint8, CHW)."""
    obs_tensor = torch.tensor(np.expand_dims(obs_4ch_uint8, axis=0), dtype=torch.float32).to(device)

    activations, gradients = None, None

    def forward_hook(module, inp, out):
        nonlocal activations
        activations = out

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    target_layer = model.policy.features_extractor.cnn[4]
    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    with torch.enable_grad():
        model.policy.eval()
        values = model.policy.predict_values(obs_tensor)
        model.policy.zero_grad()
        values.sum().backward()

    h1.remove()
    h2.remove()

    if activations is None or gradients is None:
        return None

    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim=1).squeeze(0)
    heatmap = torch.max(heatmap, torch.zeros_like(heatmap))
    if torch.max(heatmap) > 0:
        heatmap /= torch.max(heatmap)

    return heatmap.cpu().detach().numpy()


def _box_metrics(pred_box, gt_box):
    xi1, yi1 = max(pred_box[0], gt_box[0]), max(pred_box[1], gt_box[1])
    xi2 = min(pred_box[0] + pred_box[2], gt_box[0] + gt_box[2])
    yi2 = min(pred_box[1] + pred_box[3], gt_box[1] + gt_box[3])
    inter_px = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)

    gt_px = gt_box[2] * gt_box[3]
    pred_px = pred_box[2] * pred_box[3]
    union_px = gt_px + pred_px - inter_px

    return {
        "iou": float(inter_px / max(1e-6, union_px)),
        "intersection_px": float(inter_px),
        "gt_px": float(gt_px),
        "pred_px": float(pred_px),
        "coverage_ratio": float(inter_px / max(1e-6, gt_px)),   # quanto della GT e' coperta dalla predizione
        "size_ratio": float(pred_px / max(1e-6, gt_px)),        # area predetta / area GT
    }


def _to_bgr_image(image_chw_float):
    img = image_chw_float
    if img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
        img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    else:
        img = cv2.cvtColor((img[0] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return img


def _draw_boxes(img_bgr, gt_box, pred_box):
    out = img_bgr.copy()
    cv2.rectangle(out, (int(gt_box[0]), int(gt_box[1])),
                   (int(gt_box[0] + gt_box[2]), int(gt_box[1] + gt_box[3])), (0, 255, 0), 2)
    cv2.rectangle(out, (int(pred_box[0]), int(pred_box[1])),
                   (int(pred_box[0] + pred_box[2]), int(pred_box[1] + pred_box[3])), (0, 0, 255), 2)
    return out


def _rollout_single(env, model, seed):
    obs, _ = env.reset(seed=seed)
    stopped_explicitly = False
    last_obs = obs

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        last_obs = obs
        if int(action) == STOP_ACTION:
            stopped_explicitly = True
        if terminated or truncated:
            break

    pred_box = env._get_xywh_box()
    gt_box = env.gt_box
    return pred_box, gt_box, last_obs, stopped_explicitly, env.current_step


def evaluate_on_test_set(model, test_ds, output_dir, max_steps=100, seed=42, max_samples=0):
    """Elabora tutto (o parte del) il test set: rollout deterministico, metriche
    pixel-level (IoU, coverage GT, size ratio) e salvataggio di Grad-CAM + box
    overlay per ogni immagine. Chiamata automaticamente a fine training."""
    gradcam_dir = os.path.join(output_dir, "gradcam")
    boxes_dir = os.path.join(output_dir, "boxes")
    os.makedirs(gradcam_dir, exist_ok=True)
    os.makedirs(boxes_dir, exist_ok=True)

    n_samples = len(test_ds) if max_samples <= 0 else min(max_samples, len(test_ds))
    print(f"[eval] Valutazione su {n_samples}/{len(test_ds)} campioni del test set...")

    device = model.device
    csv_path = os.path.join(output_dir, "metrics_per_sample.csv")
    fieldnames = ["idx", "iou", "intersection_px", "gt_px", "pred_px",
                  "coverage_ratio", "size_ratio", "steps_used", "stopped_explicitly"]
    all_metrics = []

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx in range(n_samples):
            sample = test_ds[idx]
            single_ds = SingleSampleDataset(sample)
            env = BrainTumorRL_Env(pytorch_dataset=single_ds, max_steps=max_steps)

            pred_box, gt_box, last_obs, stopped_explicitly, steps_used = _rollout_single(env, model, seed=seed)
            m = _box_metrics(pred_box, gt_box)
            m["idx"] = idx
            m["steps_used"] = steps_used
            m["stopped_explicitly"] = stopped_explicitly
            writer.writerow(m)
            all_metrics.append(m)

            orig_bgr = _to_bgr_image(sample["image"].numpy())
            boxes_img = _draw_boxes(orig_bgr, gt_box, pred_box)
            cv2.imwrite(os.path.join(boxes_dir, f"{idx:04d}.png"), boxes_img)

            heatmap = compute_gradcam(model, last_obs, device)
            if heatmap is not None:
                H, W = orig_bgr.shape[:2]
                heatmap_resized = cv2.resize(heatmap, (W, H))
                heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(orig_bgr, 0.6, heatmap_colored, 0.4, 0)
                overlay = _draw_boxes(overlay, gt_box, pred_box)
                cv2.imwrite(os.path.join(gradcam_dir, f"{idx:04d}.png"), overlay)

            if (idx + 1) % 25 == 0 or idx == n_samples - 1:
                print(f"[eval] {idx + 1}/{n_samples}  IoU={m['iou']:.3f}  "
                      f"coverage={m['coverage_ratio']:.3f}  size_ratio={m['size_ratio']:.3f}")

    ious = np.array([m["iou"] for m in all_metrics])
    coverages = np.array([m["coverage_ratio"] for m in all_metrics])
    size_ratios = np.array([m["size_ratio"] for m in all_metrics])
    stop_rate = np.mean([m["stopped_explicitly"] for m in all_metrics])

    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        def log(line):
            print(line)
            f.write(line + "\n")

        log("─" * 60)
        log(f"Campioni valutati: {n_samples}")
        log(f"IoU          -> media: {ious.mean():.4f}  mediana: {np.median(ious):.4f}  std: {ious.std():.4f}")
        log(f"Coverage GT  -> media: {coverages.mean():.4f}  mediana: {np.median(coverages):.4f}  "
            f"(intersezione / area GT)")
        log(f"Size ratio   -> media: {size_ratios.mean():.4f}  mediana: {np.median(size_ratios):.4f}  "
            f"(area predetta / area GT)")
        log(f"Tasso STOP esplicito: {stop_rate:.2%} (resto = timeout)")
        log("─" * 60)

    print(f"[eval] CSV: {csv_path}")
    print(f"[eval] Riepilogo: {summary_path}")
    print(f"[eval] Box overlay: {boxes_dir}")
    print(f"[eval] Grad-CAM: {gradcam_dir}")


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

    MAX_STEPS_PER_EPISODE = 100

    check_env(BrainTumorRL_Env(pytorch_dataset=train_ds, max_steps=MAX_STEPS_PER_EPISODE))

    USE_SUBPROCESS = True
    N_ENVS = 4  # su Windows l'avvio di ogni processo e' lento; aumenta se non sei su Windows

    def make_env():
        def _init():
            return BrainTumorRL_Env(pytorch_dataset=train_ds, max_steps=MAX_STEPS_PER_EPISODE)
        return _init

    print(f"[setup] Creazione di {N_ENVS} environment "
          f"({'SubprocVecEnv' if USE_SUBPROCESS else 'DummyVecEnv'})...", flush=True)

    if USE_SUBPROCESS:
        vec_env = SubprocVecEnv([make_env() for _ in range(N_ENVS)])
    else:
        vec_env = DummyVecEnv([make_env() for _ in range(N_ENVS)])

    vec_env = VecMonitor(vec_env)

    # FIX: normalizziamo solo il reward (non le osservazioni: le immagini vengono
    # gia' normalizzate internamente da CnnPolicy, e normalizzare ulteriormente
    # romperebbe la semantica binaria 0/255 del 4° canale "box"). La normalizzazione
    # del reward stabilizza la scala molto variabile tra delta_iou per-step e
    # terminal_bonus di fine episodio, aiutando la value function a convergere.
    GAMMA = 0.99
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=10.0, gamma=GAMMA)

    print("[setup] Environment vettorizzati pronti.", flush=True)

    # FIX (bug che causava il crash): EvalCallback chiama internamente
    # sync_envs_normalization(training_env, eval_env) ad ogni valutazione, per
    # copiare le statistiche di normalizzazione dal training env all'eval env.
    # sync_envs_normalization scorre la catena di wrapper STRATO PER STRATO e
    # pretende che entrambi gli env abbiano la STESSA sequenza di wrapper, non
    # solo lo stesso wrapper piu' esterno. Il training env e':
    #     VecNormalize -> VecMonitor -> SubprocVecEnv/DummyVecEnv
    # Un primo tentativo di fix con solo "VecNormalize -> DummyVecEnv" (senza
    # VecMonitor in mezzo) falliva ancora, perche' a met  catena la funzione si
    # aspetta un altro VecEnvWrapper (il VecMonitor) e trova invece la
    # DummyVecEnv "nuda". La catena di eval_env deve rispecchiare ESATTAMENTE
    # quella del training env, layer per layer:
    #     VecNormalize -> VecMonitor -> DummyVecEnv
    #   - norm_obs=False: stessa scelta del training env, le osservazioni
    #     immagine non vanno normalizzate ulteriormente.
    #   - norm_reward=False, training=False: l'eval env NON deve ne'
    #     aggiornare le proprie statistiche ne' scalare il reward. Vogliamo
    #     che i reward riportati in eval (usati da StopTrainingOnNoModelImprovement
    #     e per scegliere il best_model) siano il reward "vero" e interpretabile,
    #     non quello scalato per la stabilita' del training.
    eval_env_raw = DummyVecEnv([
        lambda: BrainTumorRL_Env(pytorch_dataset=val_ds, max_steps=MAX_STEPS_PER_EPISODE)
    ])
    eval_env_raw = VecMonitor(eval_env_raw)
    eval_env = VecNormalize(eval_env_raw, norm_obs=False, norm_reward=False, gamma=GAMMA, training=False)

    N_STEPS = 1024  # buffer totale = N_STEPS * N_ENVS = 4096 (prima il commento diceva 4096 ma il valore reale era 2048)

    TOTAL_TIMESTEPS = 2_000_000
    # Nota: con reward shaping corretto la curva dovrebbe iniziare a salire
    # in modo monotono entro le prime 100-200k transizioni. Se a 500k
    # iou_mean e' ancora sotto 0.3, controlla per prima cosa il tasso di
    # STOP esplicito (custom_plots/5_stop_rate): se resta vicino a 0, il
    # problema e' probabilmente nell'ambiente/dataset (es. GT box degeneri),
    # non nel reward. Per IoU ~0.8 aspettati di aver bisogno di 2-4M step
    # complessivi: questo e' un punto di partenza, non un valore garantito.

    visual_callback = VisionMetricsCallback(val_dataset=val_ds)
    entropy_callback = EntropyScheduleCallback(initial_ent=0.05, final_ent=0.01, total_timesteps=TOTAL_TIMESTEPS)

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

    callbacks = CallbackList([visual_callback, entropy_callback, eval_callback, checkpoint_callback])

    # FIX: rete piu' capiente (features_dim 512 invece del default 256, e head
    # pi/vf a 2 layer da 256) per dare piu' capacita' al task di localizzazione
    # fine, che richiede di distinguere spostamenti di pochi pixel.
    policy_kwargs = dict(
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )

    model = PPO(
        policy="CnnPolicy",
        env=vec_env,
        policy_kwargs=policy_kwargs,
        learning_rate=linear_schedule(3e-4, 1e-5),
        n_steps=N_STEPS,
        batch_size=256,
        n_epochs=10,
        gamma=GAMMA,           # FIX: 0.995 -> 0.99, piu' adatto a episodi di 100 step con reward denso
        gae_lambda=0.95,
        clip_range=linear_schedule(0.2, 0.1),
        ent_coef=0.05,          # valore iniziale; annealed a runtime da EntropyScheduleCallback fino a 0.01
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="./ppo_brain_tumor_logs/",
    )

    print(f"Avvio addestramento PPO ({N_ENVS} env paralleli, {TOTAL_TIMESTEPS} timesteps totali)...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks)

    model.save("./ppo_brain_tumor_logs/final_model")
    vec_env.save("./ppo_brain_tumor_logs/vecnormalize_stats.pkl")
    print("Training completato. Modello finale salvato in ./ppo_brain_tumor_logs/final_model")

    # ─── VALUTAZIONE AUTOMATICA SUL TEST SET (Grad-CAM + box overlay + CSV metriche) ───
    print("\nAvvio valutazione automatica sul test set completo...")
    evaluate_on_test_set(
        model=model,
        test_ds=test_ds,
        output_dir="./ppo_brain_tumor_logs/test_eval/",
        max_steps=MAX_STEPS_PER_EPISODE,
        seed=cfg["seed"],
        max_samples=0,  # 0 = tutto il test set
    )
    print("Valutazione completata. Risultati in ./ppo_brain_tumor_logs/test_eval/")