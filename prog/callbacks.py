"""
callbacks.py
────────────
Tutti i callback SB3 usati durante il training:
  - VisionMetricsCallback : log metriche + GradCAM periodico su un
                             campione fisso di validazione.
  - EntropyScheduleCallback, StopCurriculumCallback,
    StepSizeScheduleCallback, InitBoxCurriculumCallback : curriculum.
  - ModelCheckpointCallback : salva modello + VecNormalize INSIEME
                              a ogni checkpoint (nello script originale
                              VecNormalize veniva salvato solo a fine
                              training: se il training si interrompeva
                              prima, le statistiche di normalizzazione
                              andavano perse).
"""
import os
from collections import deque

import cv2
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

from config import N_ACTIONS
from utils import build_coord_planes


class VisionMetricsCallback(BaseCallback):
    """Logga le componenti di reward/IoU su TensorBoard e salva una GradCAM
    periodica su un campione fisso, per capire dove guarda la CNN."""

    def __init__(self, val_dataset, verbose=0, save_dir="./ppo_gradcam_outputs", gradcam_every=20):
        super().__init__(verbose)
        self.val_dataset = val_dataset
        self.save_dir = save_dir
        self.gradcam_every = gradcam_every
        self.iteration_count = 0
        os.makedirs(self.save_dir, exist_ok=True)

        self.action_counts = np.zeros(N_ACTIONS, dtype=np.int64)

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
        self.viz_step_frac = 0.15

    def _on_training_start(self) -> None:
        from stable_baselines3.common.logger import TensorBoardOutputFormat
        for fmt in self.logger.output_formats:
            if isinstance(fmt, TensorBoardOutputFormat):
                layout = {
                    "Analisi_Integrazione_RL": {
                        "1_IoU_Media_Std_Finale": ["Multiline", ["custom_plots/1_iou_mean", "custom_plots/1_iou_std", "custom_plots/1_iou_final"]],
                        "2_Guida_Risoluzione": ["Multiline", ["custom_plots/2_delta_iou", "custom_plots/2_delta_dist"]],
                        "4_Reward_Completa": ["Multiline", ["custom_plots/4_total_reward"]],
                        "5_Curriculum": ["Multiline", ["custom_plots/6_ent_coef", "custom_plots/7_min_steps_before_stop", "custom_plots/8_step_frac"]],
                    }
                }
                fmt.writer.add_custom_scalars(layout)
                break

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        actions = self.locals.get("actions")
        if actions is not None:
            for a in np.atleast_1d(actions):
                a = int(a)
                if 0 <= a < N_ACTIONS:
                    self.action_counts[a] += 1

        if infos:
            delta_vals, dist_vals, total_vals, oversize_vals, oscillation_vals = [], [], [], [], []
            for info in infos:
                comp = info.get("rew_components")
                if comp is not None:
                    delta_vals.append(float(comp["delta_iou"]))
                    dist_vals.append(float(comp.get("delta_dist", 0.0)))
                    total_vals.append(float(comp["total"]))
                    oversize_vals.append(float(comp.get("oversize_penalty", 0.0)))
                    oscillation_vals.append(float(comp.get("oscillation_penalty", 0.0)))
            if delta_vals:
                self.logger.record("custom_plots/2_delta_iou", float(np.mean(delta_vals)))
                self.logger.record("custom_plots/2_delta_dist", float(np.mean(dist_vals)))
                self.logger.record("custom_plots/4_total_reward", float(np.mean(total_vals)))
                self.logger.record("custom_plots/3_oversize_penalty", float(np.mean(oversize_vals)))
                self.logger.record("custom_plots/3_oscillation_penalty", float(np.mean(oscillation_vals)))

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
                self.logger.record("custom_plots/1_success_rate", float(np.mean(np.array(final_vals) >= 0.5)))

        if self.action_counts.sum() > 0 and self.num_timesteps % 50_000 < self.training_env.num_envs:
            freqs = self.action_counts / max(1, self.action_counts.sum())
            for a in range(N_ACTIONS):
                self.logger.record(f"action_distribution/action_{a}", float(freqs[a]))
            self.action_counts[:] = 0
        return True

    def _on_rollout_end(self) -> None:
        self.iteration_count += 1
        if self.iteration_count == 1 or self.iteration_count % self.gradcam_every == 0:
            self.generate_and_save_gradcam()

    def generate_and_save_gradcam(self):
        box_mask = np.zeros((self.H, self.W), dtype=np.uint8)
        cx, cy, w, h = self.fixed_initial_box
        x1, y1 = int(cx - w / 2.0), int(cy - h / 2.0)
        x2, y2 = int(cx + w / 2.0), int(cy + h / 2.0)
        cv2.rectangle(box_mask, (x1, y1), (x2, y2), 255, thickness=-1)
        box_mask_channel = np.expand_dims(box_mask, axis=0)

        img_uint8 = (self.fixed_sample["image"].numpy() * 255).astype(np.uint8)
        coord_planes = build_coord_planes(cx, cy, w, h, self.W, self.H)
        fixed_obs = np.concatenate([img_uint8, box_mask_channel, coord_planes], axis=0)
        obs_tensor = torch.tensor(np.expand_dims(fixed_obs, axis=0), dtype=torch.float32).to(self.model.device)

        with torch.enable_grad():
            self.model.policy.eval()
            activations, gradients = None, None

            def forward_hook(module, inp, out): nonlocal activations; activations = out
            def backward_hook(module, gin, gout): nonlocal gradients; gradients = gout[0]

            target_layer = self.model.policy.features_extractor.cnn[4]
            h1 = target_layer.register_forward_hook(forward_hook)
            h2 = target_layer.register_full_backward_hook(backward_hook)

            values = self.model.policy.predict_values(obs_tensor)
            self.model.policy.zero_grad()
            values.sum().backward()
            h1.remove(); h2.remove()

        if activations is None or gradients is None:
            return

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

        mask = self._compute_mask_for_fixed_box()
        with torch.no_grad():
            action, _ = self.model.predict(fixed_obs, deterministic=True, action_masks=mask)

        frac = self.viz_step_frac
        dx, dy = frac * self.fixed_initial_box[2], frac * self.fixed_initial_box[3]
        if action == 0:   self.fixed_initial_box[0] -= dx
        elif action == 1: self.fixed_initial_box[0] += dx
        elif action == 2: self.fixed_initial_box[1] -= dy
        elif action == 3: self.fixed_initial_box[1] += dy
        elif action == 4: self.fixed_initial_box[2] *= (1.0 + frac)
        elif action == 5: self.fixed_initial_box[2] *= (1.0 - frac)
        elif action == 6: self.fixed_initial_box[3] *= (1.0 + frac)
        elif action == 7: self.fixed_initial_box[3] *= (1.0 - frac)

        self.fixed_initial_box[2] = np.clip(self.fixed_initial_box[2], 12, self.W)
        self.fixed_initial_box[3] = np.clip(self.fixed_initial_box[3], 12, self.H)
        self.fixed_initial_box[0] = np.clip(self.fixed_initial_box[0], self.fixed_initial_box[2] / 2, self.W - self.fixed_initial_box[2] / 2)
        self.fixed_initial_box[1] = np.clip(self.fixed_initial_box[1], self.fixed_initial_box[3] / 2, self.H - self.fixed_initial_box[3] / 2)

        draw_x = self.fixed_initial_box[0] - self.fixed_initial_box[2] / 2.0
        draw_y = self.fixed_initial_box[1] - self.fixed_initial_box[3] / 2.0
        cv2.rectangle(overlay, (int(draw_x), int(draw_y)),
                      (int(draw_x + self.fixed_initial_box[2]), int(draw_y + self.fixed_initial_box[3])), (0, 0, 255), 2)
        cv2.imwrite(os.path.join(self.save_dir, f"gradcam_iter_{self.iteration_count}.png"), overlay)

    def _compute_mask_for_fixed_box(self):
        w, h = self.fixed_initial_box[2], self.fixed_initial_box[3]
        mask = np.ones(N_ACTIONS, dtype=bool)
        if w >= self.W: mask[4] = False
        if w <= 12: mask[5] = False
        if h >= self.H: mask[6] = False
        if h <= 12: mask[7] = False
        mask[8] = True
        return mask


class EntropyScheduleCallback(BaseCallback):
    def __init__(self, initial_ent: float, final_ent: float, schedule_steps: int, verbose=0):
        super().__init__(verbose)
        self.initial_ent, self.final_ent, self.schedule_steps = initial_ent, final_ent, schedule_steps

    def _on_step(self) -> bool:
        progress = min(1.0, self.num_timesteps / max(1, self.schedule_steps))
        self.model.ent_coef = self.initial_ent + progress * (self.final_ent - self.initial_ent)
        if self.num_timesteps % 50_000 < self.training_env.num_envs:
            self.logger.record("custom_plots/6_ent_coef", float(self.model.ent_coef))
        return True


class StopCurriculumCallback(BaseCallback):
    def __init__(self, curriculum_steps, initial_min_steps, final_min_steps=0, verbose=0):
        super().__init__(verbose)
        self.curriculum_steps, self.initial_min_steps, self.final_min_steps = curriculum_steps, initial_min_steps, final_min_steps

    def _on_step(self) -> bool:
        progress = min(1.0, self.num_timesteps / max(1, self.curriculum_steps))
        current = int(round(self.initial_min_steps + progress * (self.final_min_steps - self.initial_min_steps)))
        self.training_env.env_method("set_min_steps_before_stop", current)
        if self.num_timesteps % 50_000 < self.training_env.num_envs:
            self.logger.record("custom_plots/7_min_steps_before_stop", float(current))
        return True


class StepSizeScheduleCallback(BaseCallback):
    def __init__(self, curriculum_steps, initial_frac=0.25, final_frac=0.04, verbose=0):
        super().__init__(verbose)
        self.curriculum_steps, self.initial_frac, self.final_frac = curriculum_steps, initial_frac, final_frac

    def _on_step(self) -> bool:
        progress = min(1.0, self.num_timesteps / max(1, self.curriculum_steps))
        frac = self.initial_frac + progress * (self.final_frac - self.initial_frac)
        self.training_env.env_method("set_step_frac", frac)
        if self.num_timesteps % 50_000 < self.training_env.num_envs:
            self.logger.record("custom_plots/8_step_frac", float(frac))
        return True


class InitBoxCurriculumCallback(BaseCallback):
    def __init__(self, curriculum_steps, initial_difficulty=0.0, final_difficulty=1.0, verbose=0):
        super().__init__(verbose)
        self.curriculum_steps, self.initial_difficulty, self.final_difficulty = curriculum_steps, initial_difficulty, final_difficulty

    def _on_step(self) -> bool:
        progress = min(1.0, self.num_timesteps / max(1, self.curriculum_steps))
        difficulty = self.initial_difficulty + progress * (self.final_difficulty - self.initial_difficulty)
        self.training_env.env_method("set_init_difficulty", difficulty)
        if self.num_timesteps % 50_000 < self.training_env.num_envs:
            self.logger.record("custom_plots/9_init_difficulty", float(difficulty))
        return True


class StagedCurriculumCallback(BaseCallback):
    """Curriculum di difficolta' A GRADINI (invece di rampa lineare continua),
    con "reheat" di ent_coef sincronizzato ad ogni salto di gradino.

    Perche': con una rampa lineare l'ambiente non e' mai stazionario -> la
    policy non fa in tempo a consolidarsi prima che la difficolta' cambi di
    nuovo sotto di lei, e un pavimento di entropia fisso per tutto il training
    o e' troppo alto quando servirebbe sfruttare cio' che si e' imparato, o
    troppo basso quando arrivano scenari piu' difficili che richiederebbero
    esplorazione fresca.

    Con questo callback la difficolta' resta COSTANTE per blocchi lunghi
    ("stage"), e ad ogni cambio di stage ent_coef torna temporaneamente alto
    (reheat) per poi decadere gradualmente al pavimento PRIMA del prossimo
    salto di difficolta' -- quindi l'agente ha sempre una finestra di
    esplorazione fresca quando i dati/scenari cambiano, seguita da una finestra
    piu' lunga di sfruttamento/consolidamento a entropia bassa.
    """

    def __init__(self, total_curriculum_steps, n_stages=5,
                 initial_difficulty=0.0, final_difficulty=1.0,
                 reheat_ent=0.03, floor_ent=0.01, reheat_frac=0.35, verbose=0):
        super().__init__(verbose)
        self.n_stages = max(1, n_stages)
        self.stage_duration = max(1, total_curriculum_steps // self.n_stages)
        self.reheat_ent, self.floor_ent, self.reheat_frac = reheat_ent, floor_ent, reheat_frac
        if self.n_stages > 1:
            self.difficulties = [initial_difficulty + (final_difficulty - initial_difficulty) * i / (self.n_stages - 1)
                                  for i in range(self.n_stages)]
        else:
            self.difficulties = [final_difficulty]
        self._last_stage = -1

    def _on_step(self) -> bool:
        stage = min(self.n_stages - 1, self.num_timesteps // self.stage_duration)
        difficulty = self.difficulties[stage]
        if stage != self._last_stage:
            self.training_env.env_method("set_init_difficulty", difficulty)
            self._last_stage = stage
            if self.verbose:
                print(f"[curriculum] nuovo stage {stage} -> difficulty={difficulty:.2f} "
                      f"@ step {self.num_timesteps} (ent_coef torna a {self.reheat_ent})")

        # sawtooth: reheat -> decadimento lineare -> pavimento, prima del prossimo stage
        step_within_stage = self.num_timesteps - stage * self.stage_duration
        reheat_steps = self.reheat_frac * self.stage_duration
        if step_within_stage < reheat_steps:
            frac = step_within_stage / max(1, reheat_steps)
            ent = self.reheat_ent + frac * (self.floor_ent - self.reheat_ent)
        else:
            ent = self.floor_ent
        self.model.ent_coef = ent

        if self.num_timesteps % 50_000 < self.training_env.num_envs:
            self.logger.record("custom_plots/9_init_difficulty", float(difficulty))
            self.logger.record("custom_plots/6_ent_coef", float(ent))
        return True


class AdaptiveCurriculumCallback(BaseCallback):
    """Curriculum di difficolta' AUTO-PACED (gated dalle performance), non a
    tempo fisso. Gestisce anche ent_coef e step_frac con un reheat sincronizzato.

    Perche' un curriculum a step_count fisso (versione precedente) puo'
    "complicare le cose": la difficolta' sale comunque anche se l'agente non
    ha ancora imparato lo stage corrente, e (bug della versione precedente)
    una volta raggiunto l'ultimo stage il reheat di ent_coef non si ripeteva
    piu' per il resto del training, lasciando la fase piu' difficile - quella
    che ne avrebbe piu' bisogno - senza alcuna esplorazione fresca.

    Qui invece:
      - si avanza di stage solo quando il success_rate (iou_final>=0.5) sulle
        ultime `window` episodi supera `advance_threshold`;
      - si REGREDISCE di uno stage (mai sotto lo stage 0) se il success_rate
        crolla sotto `regress_threshold` -- se la difficolta' e' davvero troppa,
        il curriculum torna indietro invece di lasciare l'agente a marcire;
      - ogni transizione (avanti o indietro) fa scattare un reheat di ent_coef
        e step_frac (torna a un valore alto e decade in `reheat_duration` step);
      - se l'agente resta bloccato sullo stesso stage per piu' di `stall_patience`
        step senza ne' avanzare ne' regredire, scatta comunque un reheat
        periodico ("nuovo tentativo" con piu' esplorazione) -- questo e' il fix
        del bug: il reheat si ripete SEMPRE, anche all'ultimo stage.
    """

    def __init__(self, n_stages=6, initial_difficulty=0.0, final_difficulty=1.0,
                 window=100, min_steps_per_stage=30_000, advance_threshold=0.35,
                 regress_threshold=0.08, stall_patience=150_000,
                 reheat_ent=0.03, floor_ent=0.01,
                 reheat_step_frac=0.05, floor_step_frac=0.012,
                 reheat_duration=60_000, verbose=1):
        super().__init__(verbose)
        self.n_stages = max(1, n_stages)
        self.difficulties = ([initial_difficulty + (final_difficulty - initial_difficulty) * i / (self.n_stages - 1)
                               for i in range(self.n_stages)] if self.n_stages > 1 else [final_difficulty])
        self.window = window
        self.min_steps_per_stage = min_steps_per_stage
        self.advance_threshold = advance_threshold
        self.regress_threshold = regress_threshold
        self.stall_patience = stall_patience
        self.reheat_ent, self.floor_ent = reheat_ent, floor_ent
        self.reheat_step_frac, self.floor_step_frac = reheat_step_frac, floor_step_frac
        self.reheat_duration = reheat_duration

        self.stage = 0
        self._iou_final_buffer = deque(maxlen=window)
        self._last_transition_step = 0
        self._last_reheat_step = 0

    def _apply_stage(self, new_stage, reason):
        new_stage = int(np.clip(new_stage, 0, self.n_stages - 1))
        if new_stage != self.stage:
            self.stage = new_stage
            self.training_env.env_method("set_init_difficulty", self.difficulties[self.stage])
            self._iou_final_buffer.clear()
            self._last_transition_step = self.num_timesteps
            self._last_reheat_step = self.num_timesteps
            if self.verbose:
                print(f"[curriculum] {reason}: stage -> {self.stage} "
                      f"(difficulty={self.difficulties[self.stage]:.2f}) @ step {self.num_timesteps}")

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if infos:
            for info in infos:
                metrics = info.get("episode_metrics")
                if metrics is not None:
                    self._iou_final_buffer.append(float(metrics["ep_iou_final"]))

        steps_in_stage = self.num_timesteps - self._last_transition_step
        success_rate = None
        if len(self._iou_final_buffer) >= self.window:
            success_rate = float(np.mean(np.array(self._iou_final_buffer) >= 0.5))
            if steps_in_stage >= self.min_steps_per_stage:
                if success_rate >= self.advance_threshold and self.stage < self.n_stages - 1:
                    self._apply_stage(self.stage + 1, "avanzo (performance buone)")
                elif success_rate < self.regress_threshold and self.stage > 0:
                    self._apply_stage(self.stage - 1, "regredisco (performance crollate)")

        # reheat periodico se bloccato troppo a lungo sullo stesso stage senza transizioni
        # (FIX del bug precedente: qui si ripete sempre, anche all'ultimo stage)
        if self.num_timesteps - self._last_reheat_step >= self.stall_patience:
            self._last_reheat_step = self.num_timesteps
            if self.verbose:
                print(f"[curriculum] stallo allo stage {self.stage}: reheat esplorativo @ step {self.num_timesteps}")

        # sawtooth ent_coef / step_frac dal reheat piu' recente (transizione o stallo)
        step_since_reheat = self.num_timesteps - self._last_reheat_step
        frac = min(1.0, step_since_reheat / max(1, self.reheat_duration))
        ent = self.reheat_ent + frac * (self.floor_ent - self.reheat_ent)
        step_frac = self.reheat_step_frac + frac * (self.floor_step_frac - self.reheat_step_frac)
        self.model.ent_coef = ent
        self.training_env.env_method("set_step_frac", step_frac)

        if self.num_timesteps % 50_000 < self.training_env.num_envs:
            self.logger.record("custom_plots/9_init_difficulty", float(self.difficulties[self.stage]))
            self.logger.record("custom_plots/6_ent_coef", float(ent))
            self.logger.record("custom_plots/8_step_frac", float(step_frac))
            if success_rate is not None:
                self.logger.record("custom_plots/10_curriculum_success_rate", success_rate)
        return True


class ModelCheckpointCallback(BaseCallback):
    """Salva modello + statistiche VecNormalize INSIEME ogni `save_freq` step,
    con il numero di timestep nel nome, cosi' ogni checkpoint e' autosufficiente
    per essere ripreso o valutato in isolamento (a differenza dello script
    originale, dove VecNormalize veniva scritto solo a fine training)."""

    def __init__(self, save_freq: int, save_dir: str, vec_env, name_prefix="ppo_brain_tumor",
                 keep_last=10, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_dir = save_dir
        self.vec_env = vec_env
        self.name_prefix = name_prefix
        self.keep_last = keep_last
        self._saved = []
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq < self.training_env.num_envs:
            self._save_checkpoint()
        return True

    def _save_checkpoint(self):
        tag = f"{self.name_prefix}_{self.num_timesteps}"
        model_path = os.path.join(self.save_dir, f"{tag}.zip")
        vecnorm_path = os.path.join(self.save_dir, f"{tag}_vecnormalize.pkl")

        self.model.save(model_path)
        self.vec_env.save(vecnorm_path)

        if self.verbose:
            print(f"[checkpoint] salvato modello + vecnormalize a {self.num_timesteps} step -> {model_path}")

        self._saved.append((model_path, vecnorm_path))
        if self.keep_last is not None and len(self._saved) > self.keep_last:
            old_model, old_vecnorm = self._saved.pop(0)
            for p in (old_model, old_vecnorm):
                if os.path.exists(p):
                    os.remove(p)
