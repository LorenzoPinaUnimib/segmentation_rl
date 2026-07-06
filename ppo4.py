import os
import csv
import argparse
import cv2
import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    StopTrainingOnNoModelImprovement,
)

# FIX 6 (action masking collegato): passiamo da PPO standard a MaskablePPO
# (sb3-contrib). action_masks() era gia' implementato in BrainTumorRL_Env ma
# non era usato: il modello sprecava esplorazione su azioni di resize che
# sono no-op quando w/h sono gia' al limite (12px o W/H). Serve:
#   pip install sb3-contrib --break-system-packages
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.policies import MaskableActorCriticCnnPolicy  # noqa: F401 (registra "CnnPolicy")
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker

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
    """Calcola le granularita' di movimento/resize (coarse/fine) proporzionali
    alla dimensione dell'immagine, cosi' l'agente puo' sia esplorare
    velocemente sia rifinire il box con precisione per avvicinarsi a IoU alti.

    Step separati per l'asse w e per l'asse h, proporzionali rispettivamente
    a W e H, cosi' il resize per-asse non eredita una granularita' "sbagliata"
    presa dalla dimensione media (importante su immagini/tumori non quadrati).
    """
    big_step = max(6.0, 0.045 * ((W + H) / 2.0))
    small_step = max(2.0, big_step / 3.0)

    big_step_w = max(6.0, 0.045 * W)
    small_step_w = max(2.0, big_step_w / 3.0)

    big_step_h = max(6.0, 0.045 * H)
    small_step_h = max(2.0, big_step_h / 3.0)

    return big_step, small_step, big_step_w, small_step_w, big_step_h, small_step_h


# Layout (21 azioni):
#   0-3   : movimento coarse (sx, dx, su, giu)
#   4-7   : movimento fine   (sx, dx, su, giu)
#   8-11  : resize coarse per asse (w+, w-, h+, h-)
#   12-15 : resize fine per asse   (w+, w-, h+, h-)
#   16-17 : resize coarse uniforme (w+h+, w-h-)  -- mantenuto per scalare in fretta
#   18-19 : resize fine uniforme   (w+h+, w-h-)
#   20    : STOP
N_ACTIONS = 21
STOP_ACTION = 20

# ─────────────────────────────────────────────────────────────────────────────
# REWARD SHAPING — parametri centralizzati
# ─────────────────────────────────────────────────────────────────────────────
# FIX 7 (ribilanciamento reward): nella versione precedente il terminal_bonus
# arrivava fino a iou*60 + milestone (~80-90 nei casi migliori), mentre il
# contributo denso per-step (delta_iou_component) e' per costruzione clippato
# a +-10 a ogni step: sommato su un episodio, il suo contributo netto tende a
# telescopare a circa DELTA_IOU_SCALE*(iou_finale - iou_iniziale), quindi
# nell'ordine di poche decine al massimo. Un terminal_bonus 3-8 volte piu'
# grande del segnale denso totale rende il ritorno per-episodio dominato da
# un evento singolo, molto rumoroso da un episodio all'altro: e' la causa
# piu' probabile del crollo di train/explained_variance osservato nei log
# (il critic non riesce a spiegare la varianza dei ritorni, perche' la
# varianza e' concentrata quasi tutta nell'ultimo step). Qui riduciamo
# terminal_bonus a una scala COMPARABILE (non dominante) rispetto al segnale
# denso, cosi' i ritorni sono meno "a scogliera" e piu' facili da fittare.
DELTA_IOU_SCALE = 25.0
DELTA_IOU_CLIP = 10.0
TIME_PENALTY = -0.01

TERMINAL_SCALE_STOP = 18.0      # era 60.0
TERMINAL_SCALE_TIMEOUT = 12.0   # era 40.0

# Bonus a scaglioni ridotti in proporzione (mantenuto il rapporto relativo
# STOP > timeout implicito, ma senza dominare il segnale denso)
IOU_MILESTONES = [(0.5, 2.0), (0.7, 4.0), (0.8, 8.0)]

# FIX (robustezza, invariato): minimo di step obbligatori prima che STOP
# abbia effetto, per evitare episodi troppo corti a inizio training quando
# la policy e' quasi uniforme su 21 azioni.
MIN_STEPS_BEFORE_STOP = 20  # 20% di MAX_STEPS_PER_EPISODE=100


def mask_fn(env: gym.Env) -> np.ndarray:
    """Funzione richiesta da ActionMasker/MaskablePPO per ottenere la maschera
    delle azioni valide dall'ambiente sottostante."""
    return env.unwrapped.action_masks()


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
        (self.big_step, self.small_step,
         self.big_step_w, self.small_step_w,
         self.big_step_h, self.small_step_h) = compute_action_steps(self.W, self.H)

    def _on_training_start(self) -> None:
        """Crea il layout a 5 grafici richiesto dall'utente all'avvio del modello."""
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

            # FIX 6: con MaskablePPO, model.predict richiede la maschera delle
            # azioni valide corrente, altrimenti puo' scegliere azioni no-op
            # (es. resize gia' al limite) diverse da quelle usate in training.
            current_mask = self._compute_mask_for_fixed_box()
            with torch.no_grad():
                action, _ = self.model.predict(fixed_obs, deterministic=True, action_masks=current_mask)

            bs, ss = self.big_step, self.small_step
            bs_w, ss_w = self.big_step_w, self.small_step_w
            bs_h, ss_h = self.big_step_h, self.small_step_h
            if action == 0:   self.fixed_initial_box[0] -= bs
            elif action == 1: self.fixed_initial_box[0] += bs
            elif action == 2: self.fixed_initial_box[1] -= bs
            elif action == 3: self.fixed_initial_box[1] += bs
            elif action == 4: self.fixed_initial_box[0] -= ss
            elif action == 5: self.fixed_initial_box[0] += ss
            elif action == 6: self.fixed_initial_box[1] -= ss
            elif action == 7: self.fixed_initial_box[1] += ss
            elif action == 8:  self.fixed_initial_box[2] += bs_w
            elif action == 9:  self.fixed_initial_box[2] -= bs_w
            elif action == 10: self.fixed_initial_box[3] += bs_h
            elif action == 11: self.fixed_initial_box[3] -= bs_h
            elif action == 12: self.fixed_initial_box[2] += ss_w
            elif action == 13: self.fixed_initial_box[2] -= ss_w
            elif action == 14: self.fixed_initial_box[3] += ss_h
            elif action == 15: self.fixed_initial_box[3] -= ss_h
            elif action == 16:
                self.fixed_initial_box[2] += bs; self.fixed_initial_box[3] += bs
            elif action == 17:
                self.fixed_initial_box[2] -= bs; self.fixed_initial_box[3] -= bs
            elif action == 18:
                self.fixed_initial_box[2] += ss; self.fixed_initial_box[3] += ss
            elif action == 19:
                self.fixed_initial_box[2] -= ss; self.fixed_initial_box[3] -= ss
            # action == STOP_ACTION (20): nessun movimento, il box resta fermo

            self.fixed_initial_box[2] = np.clip(self.fixed_initial_box[2], 12, self.W)
            self.fixed_initial_box[3] = np.clip(self.fixed_initial_box[3], 12, self.H)
            self.fixed_initial_box[0] = np.clip(self.fixed_initial_box[0], self.fixed_initial_box[2] / 2, self.W - self.fixed_initial_box[2] / 2)
            self.fixed_initial_box[1] = np.clip(self.fixed_initial_box[1], self.fixed_initial_box[3] / 2, self.H - self.fixed_initial_box[3] / 2)

            draw_x = self.fixed_initial_box[0] - self.fixed_initial_box[2] / 2.0
            draw_y = self.fixed_initial_box[1] - self.fixed_initial_box[3] / 2.0

            cv2.rectangle(overlay, (int(draw_x), int(draw_y)), (int(draw_x + self.fixed_initial_box[2]), int(draw_y + self.fixed_initial_box[3])), (0, 0, 255), 2)
            save_path = os.path.join(self.save_dir, f"gradcam_iter_{self.iteration_count}.png")
            cv2.imwrite(save_path, overlay)

    def _compute_mask_for_fixed_box(self):
        """FIX 6: replica action_masks() di BrainTumorRL_Env per il box fisso
        usato nel Grad-CAM, senza dover istanziare un env completo."""
        w, h = self.fixed_initial_box[2], self.fixed_initial_box[3]
        mask = np.ones(N_ACTIONS, dtype=bool)
        at_w_min, at_w_max = w <= 12, w >= self.W
        at_h_min, at_h_max = h <= 12, h >= self.H
        if at_w_max: mask[8] = mask[12] = False
        if at_w_min: mask[9] = mask[13] = False
        if at_h_max: mask[10] = mask[14] = False
        if at_h_min: mask[11] = mask[15] = False
        if at_w_max or at_h_max: mask[16] = mask[18] = False
        if at_w_min or at_h_min: mask[17] = mask[19] = False
        return mask


class EntropyScheduleCallback(BaseCallback):
    """FIX 8 (entropia annealed piu' aggressivamente): SB3 non supporta uno
    schedule nativo per ent_coef. Nel run precedente train/entropy_loss
    restava vicino al massimo teorico (ln(21)~3.04) per tutto il training
    osservato: la policy restava quasi uniforme, coerente con stop_rate=1
    costante (con 21 azioni quasi-random, la probabilita' di pescare STOP
    entro gli step rimanenti e' comunque altissima per puro caso, non perche'
    la policy abbia imparato QUANDO fermarsi). Qui abbassiamo il valore
    iniziale e finale rispetto alla versione precedente, cosi' la policy ha
    piu' margine per specializzarsi una volta che il segnale di reward e'
    piu' pulito (vedi FIX 7).
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


class CurriculumCallback(BaseCallback):
    """FIX 2 (invariato nella logica, esteso in durata — vedi FIX 9 in fondo):
    curriculum sull'inizializzazione del box iniziale in reset().
    p=0 -> box iniziale vicino alla GT (task piu' facile a inizio training).
    p=1 -> box iniziale con range originale, ampio e centrato sull'immagine.
    """

    def __init__(self, curriculum_timesteps: int, verbose=0):
        super().__init__(verbose)
        self.curriculum_timesteps = max(1, curriculum_timesteps)

    def _on_step(self) -> bool:
        progress = min(1.0, self.num_timesteps / self.curriculum_timesteps)
        self.training_env.env_method("set_curriculum_progress", progress)
        if self.num_timesteps % 50_000 < self.training_env.num_envs:
            self.logger.record("custom_plots/7_curriculum_progress", float(progress))
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

        (self.big_step, self.small_step,
         self.big_step_w, self.small_step_w,
         self.big_step_h, self.small_step_h) = compute_action_steps(self.W, self.H)
        self.max_diag = np.sqrt(self.W ** 2 + self.H ** 2)

        # default = 1.0 (nessun curriculum attivo). Abbassato solo dal
        # CurriculumCallback sugli env di training; eval_env e gli env creati
        # in evaluate_on_test_set restano sempre a piena difficolta'.
        self.curriculum_progress = 1.0

    def set_curriculum_progress(self, progress: float):
        self.curriculum_progress = float(np.clip(progress, 0.0, 1.0))

    def action_masks(self):
        """Maschera le azioni di movimento/resize che sarebbero no-op perche'
        w/h sono gia' al limite (min 12px o max W/H). FIX 6: ora collegata
        al training tramite ActionMasker + MaskablePPO."""
        mask = np.ones(N_ACTIONS, dtype=bool)
        at_w_min, at_w_max = self.w <= 12, self.w >= self.W
        at_h_min, at_h_max = self.h <= 12, self.h >= self.H
        if at_w_max: mask[8] = mask[12] = False
        if at_w_min: mask[9] = mask[13] = False
        if at_h_max: mask[10] = mask[14] = False
        if at_h_min: mask[11] = mask[15] = False
        if at_w_max or at_h_max: mask[16] = mask[18] = False
        if at_w_min or at_h_min: mask[17] = mask[19] = False
        return mask

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

        p = self.curriculum_progress

        gt_cx = self.gt_box[0] + self.gt_box[2] / 2.0
        gt_cy = self.gt_box[1] + self.gt_box[3] / 2.0
        img_cx, img_cy = self.W / 2.0, self.H / 2.0

        pos_center_x = (1 - p) * gt_cx + p * img_cx
        pos_center_y = (1 - p) * gt_cy + p * img_cy
        pos_noise_amp = 10.0 + p * 20.0
        self.cx = pos_center_x + self.np_random.uniform(-pos_noise_amp, pos_noise_amp)
        self.cy = pos_center_y + self.np_random.uniform(-pos_noise_amp, pos_noise_amp)

        near_w = self.gt_box[2] * self.np_random.uniform(0.85, 1.15)
        near_h = self.gt_box[3] * self.np_random.uniform(0.85, 1.15)
        far_w = self.W * self.np_random.uniform(0.20, 0.45)
        far_h = self.H * self.np_random.uniform(0.20, 0.45)
        self.w = (1 - p) * near_w + p * far_w
        self.h = (1 - p) * near_h + p * far_h

        self.w = np.clip(self.w, 12, self.W)
        self.h = np.clip(self.h, 12, self.H)
        self.cx = np.clip(self.cx, self.w / 2, self.W - self.w / 2)
        self.cy = np.clip(self.cy, self.h / 2, self.H - self.h / 2)

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
        bs_w, ss_w = self.big_step_w, self.small_step_w
        bs_h, ss_h = self.big_step_h, self.small_step_h

        if action == 0:   self.cx -= bs
        elif action == 1: self.cx += bs
        elif action == 2: self.cy -= bs
        elif action == 3: self.cy += bs
        elif action == 4: self.cx -= ss
        elif action == 5: self.cx += ss
        elif action == 6: self.cy -= ss
        elif action == 7: self.cy += ss
        elif action == 8:  self.w += bs_w
        elif action == 9:  self.w -= bs_w
        elif action == 10: self.h += bs_h
        elif action == 11: self.h -= bs_h
        elif action == 12: self.w += ss_w
        elif action == 13: self.w -= ss_w
        elif action == 14: self.h += ss_h
        elif action == 15: self.h -= ss_h
        elif action == 16:
            self.w += bs; self.h += bs
        elif action == 17:
            self.w -= bs; self.h -= bs
        elif action == 18:
            self.w += ss; self.h += ss
        elif action == 19:
            self.w -= ss; self.h -= ss
        elif action == STOP_ACTION:
            if self.current_step >= MIN_STEPS_BEFORE_STOP:
                terminated = True

        self.w = np.clip(self.w, 12, self.W)
        self.h = np.clip(self.h, 12, self.H)
        self.cx = np.clip(self.cx, self.w / 2, self.W - self.w / 2)
        self.cy = np.clip(self.cy, self.h / 2, self.H - self.h / 2)

        current_box_xywh = self._get_xywh_box()
        iou = self._compute_iou(current_box_xywh, self.gt_box)
        self.episode_ious.append(iou)

        # ─── REWARD SHAPING (vedi FIX 7 in testa al file per il razionale) ───
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
    """Wrapper minimale per far vedere all'ambiente un solo campione alla volta."""

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
        "coverage_ratio": float(inter_px / max(1e-6, gt_px)),
        "size_ratio": float(pred_px / max(1e-6, gt_px)),
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
    """FIX 6: passiamo esplicitamente action_masks a model.predict a ogni
    step, dato che qui l'env non e' vettorizzato (MaskablePPO non ha modo
    di recuperare la maschera automaticamente come farebbe con una VecEnv
    wrappata da ActionMasker)."""
    obs, _ = env.reset(seed=seed)
    stopped_explicitly = False
    last_obs = obs

    while True:
        current_mask = env.action_masks()
        action, _ = model.predict(obs, deterministic=True, action_masks=current_mask)
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
    arg_parser = argparse.ArgumentParser(description="Training PPO (masked) per localizzazione tumori cerebrali via RL")
    arg_parser.add_argument("--total-timesteps", type=int, default=None,
                             help="Budget totale di timesteps (default: env var TOTAL_TIMESTEPS o 6_000_000)")
    cli_args, _unknown = arg_parser.parse_known_args()

    if cli_args.total_timesteps is not None:
        TOTAL_TIMESTEPS = cli_args.total_timesteps
    else:
        TOTAL_TIMESTEPS = int(os.environ.get("TOTAL_TIMESTEPS", 6_000_000))

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
        "training": {"batch_size": 512, "num_workers": 0},
        "output": {"root": "./output"},
        "seed": 42,
    }

    train_ds, val_ds, test_ds = get_datasets(cfg)

    MAX_STEPS_PER_EPISODE = 100

    check_env(BrainTumorRL_Env(pytorch_dataset=train_ds, max_steps=MAX_STEPS_PER_EPISODE))

    # FIX 9 (piu' env paralleli, buffer piu' grande): con solo 4 env e un
    # reward "a scogliera" (vedi FIX 7), il critic vedeva pochi ritorni per
    # rollout su cui stimare la varianza. Aumentiamo N_ENVS a 8 (abbassa a 4
    # se la macchina non regge l'overhead di avvio dei sottoprocessi, tipico
    # su Windows) cosi' ogni rollout aggrega piu' episodi indipendenti.
    USE_SUBPROCESS = True
    N_ENVS = 8
    N_EPOCHS = 5

    def make_env():
        def _init():
            env = BrainTumorRL_Env(pytorch_dataset=train_ds, max_steps=MAX_STEPS_PER_EPISODE)
            # FIX 6: ogni env viene wrappato con ActionMasker cosi' MaskablePPO
            # puo' recuperare la maschera valida ad ogni step tramite mask_fn.
            env = ActionMasker(env, mask_fn)
            return env
        return _init

    print(f"[setup] Creazione di {N_ENVS} environment "
          f"({'SubprocVecEnv' if USE_SUBPROCESS else 'DummyVecEnv'})...", flush=True)

    if USE_SUBPROCESS:
        vec_env = SubprocVecEnv([make_env() for _ in range(N_ENVS)])
    else:
        vec_env = DummyVecEnv([make_env() for _ in range(N_ENVS)])

    vec_env = VecMonitor(vec_env)

    GAMMA = 0.99
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=10.0, gamma=GAMMA)

    print("[setup] Environment vettorizzati pronti.", flush=True)

    # Catena eval_env identica (layer per layer) a quella del training env,
    # come richiesto da sync_envs_normalization: VecNormalize -> VecMonitor ->
    # DummyVecEnv. Anche l'eval env viene wrappato con ActionMasker (FIX 6).
    def make_eval_env():
        env = BrainTumorRL_Env(pytorch_dataset=val_ds, max_steps=MAX_STEPS_PER_EPISODE)
        env = ActionMasker(env, mask_fn)
        return env

    eval_env_raw = DummyVecEnv([make_eval_env])
    eval_env_raw = VecMonitor(eval_env_raw)
    eval_env = VecNormalize(eval_env_raw, norm_obs=False, norm_reward=False, gamma=GAMMA, training=False)

    N_STEPS = 1024  # buffer totale = N_STEPS * N_ENVS = 8192 (era 4096 con N_ENVS=4)

    # FIX 9 (curriculum esteso): nel run precedente il training si e' fermato
    # (per lo StopTrainingOnNoModelImprovement troppo aggressivo, vedi sotto)
    # a meta' del curriculum (20% del budget totale), quindi il task stava
    # ancora diventando piu' difficile quando e' stato interrotto. Estendiamo
    # il curriculum al 50% del budget totale, cosi' la transizione verso il
    # range di inizializzazione originale e' molto piu' graduale.
    CURRICULUM_TIMESTEPS = max(1, int(0.5 * TOTAL_TIMESTEPS))

    visual_callback = VisionMetricsCallback(val_dataset=val_ds)
    # FIX 8: ent_coef iniziale/finale abbassati rispetto alla versione
    # precedente (0.05 -> 0.02 iniziale, 0.01 -> 0.003 finale), per permettere
    # alla policy di specializzarsi una volta che il reward e' meno rumoroso.
    entropy_callback = EntropyScheduleCallback(initial_ent=0.02, final_ent=0.003, total_timesteps=TOTAL_TIMESTEPS)
    curriculum_callback = CurriculumCallback(curriculum_timesteps=CURRICULUM_TIMESTEPS)

    # FIX 9 (early stopping meno aggressivo): con min_evals=20 e
    # max_no_improvement_evals=15 il training precedente si e' fermato al 10%
    # del budget pianificato (6M), su una curva di iou_mean che oscillava
    # molto (rumore da value function ancora instabile, vedi FIX 7) invece di
    # essere davvero collassata. Alziamo entrambe le soglie per dare al
    # training tempo di uscire dal rumore iniziale prima di poter fermarsi.
    stop_on_no_improve = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=40, min_evals=60, verbose=1
    )
    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path="./ppo_brain_tumor_logs/best_model/",
        log_path="./ppo_brain_tumor_logs/eval_results/",
        eval_freq=max(N_STEPS * 4, 2000),
        n_eval_episodes=20,
        deterministic=True,
        use_masking=True,
        callback_after_eval=stop_on_no_improve,
        verbose=1,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(N_STEPS * 20, 10000),
        save_path="./ppo_brain_tumor_logs/checkpoints/",
        name_prefix="ppo_brain_tumor",
    )

    callbacks = CallbackList([visual_callback, entropy_callback, curriculum_callback, eval_callback, checkpoint_callback])

    policy_kwargs = dict(
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )

    # FIX 6: PPO -> MaskablePPO. FIX 9: batch_size alzato a 1024 (era 512)
    # dato il buffer piu' grande (8192 con N_ENVS=8), per mantenere un numero
    # di minibatch per epoca comparabile alla versione precedente.
    model = MaskablePPO(
        policy="CnnPolicy",
        env=vec_env,
        policy_kwargs=policy_kwargs,
        learning_rate=linear_schedule(1.5e-4, 1e-5),
        n_steps=N_STEPS,
        batch_size=1024,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=0.95,
        clip_range=linear_schedule(0.1, 0.03),
        ent_coef=0.02,           # valore iniziale; annealed a runtime (FIX 8)
        vf_coef=1.0,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="./ppo_brain_tumor_logs/",
        target_kl=0.03
    )

    print(f"Avvio addestramento PPO mascherato ({N_ENVS} env paralleli, {TOTAL_TIMESTEPS} timesteps totali, "
          f"curriculum sui primi {CURRICULUM_TIMESTEPS} timesteps)...")
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


# ─────────────────────────────────────────────────────────────────────────────
# RIEPILOGO FIX APPLICATI IN QUESTA REVISIONE
# ─────────────────────────────────────────────────────────────────────────────
# FIX 6 — Action masking collegato: PPO -> MaskablePPO (sb3-contrib), env
#         wrappati con ActionMasker, MaskableEvalCallback al posto di
#         EvalCallback, model.predict aggiornato ovunque (Grad-CAM callback,
#         evaluate_on_test_set) per passare action_masks esplicitamente dove
#         l'env non e' vettorizzato. Richiede: pip install sb3-contrib.
#
# FIX 7 — Reward ribilanciato: TERMINAL_SCALE_STOP 60->18, TERMINAL_SCALE_
#         TIMEOUT 40->12, IOU_MILESTONES scalati in proporzione. Il terminal
#         bonus non domina piu' di un ordine di grandezza il segnale denso
#         (delta_iou), riducendo la varianza dei ritorni per episodio: questo
#         e' il fix con maggiore impatto atteso su train/explained_variance,
#         che nel run precedente era vicino a zero e in peggioramento.
#
# FIX 8 — Entropia annealed piu' aggressivamente (0.05->0.02 iniziale,
#         0.01->0.003 finale), per permettere alla policy di uscire dal
#         comportamento quasi-uniforme osservato (entropy_loss vicino al
#         massimo teorico ln(21) per tutto il run precedente).
#
# FIX 9 — Early stopping meno aggressivo (min_evals 20->60,
#         max_no_improvement_evals 15->40): il run precedente si e' fermato
#         al ~10% del budget pianificato. Curriculum esteso dal 20% al 50%
#         del budget totale, per una transizione di difficolta' piu' graduale
#         e coerente con un budget di 6M step. N_ENVS 4->8 e buffer/batch
#         proporzionalmente piu' grandi, per stime di vantaggio meno rumorose.
#
# Suggerimento pratico: rilancia prima un run diagnostico piu' corto (es.
# --total-timesteps 1500000) per verificare che train/explained_variance
# smetta di degradare e che custom_plots/1_iou_mean cresca in modo piu'
# monotono nelle prime 300-500k transizioni, prima di lanciare il run
# completo a 6M step.