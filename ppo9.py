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

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.policies import MaskableActorCriticCnnPolicy
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker

from dataset import get_datasets

# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def linear_schedule(initial_value: float, final_value: float = 0.0):
    def scheduler(progress_remaining: float) -> float:
        return final_value + progress_remaining * (initial_value - final_value)
    return scheduler

def build_coord_planes(cx: float, cy: float, w: float, h: float, W: int, H: int) -> np.ndarray:
    norm = np.array([cx / W, cy / H, w / W, h / H], dtype=np.float32)
    norm = np.clip(norm, 0.0, 1.0)
    planes = np.zeros((N_COORD_CHANNELS, H, W), dtype=np.uint8)
    for i, v in enumerate(norm):
        planes[i, :, :] = np.uint8(v * 255)
    return planes

N_ACTIONS = 9
MAX_STEPS_PER_EPISODE = 200
N_COORD_CHANNELS = 4

OPPOSITE_ACTIONS = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4, 6: 7, 7: 6}

# ─────────────────────────────────────────────────────────────────────────────
# REWARD SHAPING
# ─────────────────────────────────────────────────────────────────────────────
DELTA_IOU_SCALE = 25.0
DELTA_IOU_CLIP = 10.0

# -- NUOVA REWARD DI DISTANZA --
# Aiuta l'agente a trovare la direzione anche quando l'IoU e' 0.0 (nessuna intersezione)
DISTANCE_REWARD_SCALE = 0.05
DISTANCE_REWARD_CLIP = 1.0

TIME_PENALTY = -0.01

STOP_IOU_BASELINE = 0.50
STOP_BONUS_SCALE = 20.0
STOP_BONUS_CLIP = 15.0

NO_STOP_PENALTY = -1.0

OVERSIZE_AREA_RATIO_THRESHOLD = 0.50
OVERSIZE_PENALTY_SCALE = 4.0

OSCILLATION_PENALTY = 0.01


def mask_fn(env: gym.Env) -> np.ndarray:
    return env.unwrapped.action_masks()


class VisionMetricsCallback(BaseCallback):
    def __init__(self, val_dataset, verbose=0, save_dir="./ppo_gradcam_outputs"):
        super(VisionMetricsCallback, self).__init__(verbose)
        self.val_dataset = val_dataset
        self.save_dir = save_dir
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
        output_formats = self.logger.output_formats
        from stable_baselines3.common.logger import TensorBoardOutputFormat

        for fmt in output_formats:
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

        if infos is not None and len(infos) > 0:
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
                success_rate = float(np.mean(np.array(final_vals) >= 0.5))
                self.logger.record("custom_plots/1_success_rate", success_rate)

        if self.action_counts.sum() > 0 and self.num_timesteps % 50_000 < self.training_env.num_envs:
            freqs = self.action_counts / max(1, self.action_counts.sum())
            for a in range(N_ACTIONS):
                self.logger.record(f"action_distribution/action_{a}", float(freqs[a]))
            self.action_counts[:] = 0
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
        coord_planes = build_coord_planes(cx, cy, w, h, self.W, self.H)
        fixed_obs = np.concatenate([img_uint8, box_mask_channel, coord_planes], axis=0)

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

            current_mask = self._compute_mask_for_fixed_box()
            with torch.no_grad():
                action, _ = self.model.predict(fixed_obs, deterministic=True, action_masks=current_mask)

            frac = self.viz_step_frac
            dx = frac * self.fixed_initial_box[2]
            dy = frac * self.fixed_initial_box[3]

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

            cv2.rectangle(overlay, (int(draw_x), int(draw_y)), (int(draw_x + self.fixed_initial_box[2]), int(draw_y + self.fixed_initial_box[3])), (0, 0, 255), 2)
            save_path = os.path.join(self.save_dir, f"gradcam_iter_{self.iteration_count}.png")
            cv2.imwrite(save_path, overlay)

    def _compute_mask_for_fixed_box(self):
        w, h = self.fixed_initial_box[2], self.fixed_initial_box[3]
        mask = np.ones(N_ACTIONS, dtype=bool)
        at_w_min, at_w_max = w <= 12, w >= self.W
        at_h_min, at_h_max = h <= 12, h >= self.H
        if at_w_max: mask[4] = False
        if at_w_min: mask[5] = False
        if at_h_max: mask[6] = False
        if at_h_min: mask[7] = False
        mask[8] = True
        return mask


class EntropyScheduleCallback(BaseCallback):
    def __init__(self, initial_ent: float, final_ent: float, schedule_steps: int, verbose=0):
        super().__init__(verbose)
        self.initial_ent = initial_ent
        self.final_ent = final_ent
        self.schedule_steps = schedule_steps

    def _on_step(self) -> bool:
        progress = min(1.0, self.num_timesteps / max(1, self.schedule_steps))
        new_ent = self.initial_ent + progress * (self.final_ent - self.initial_ent)
        self.model.ent_coef = new_ent
        if self.num_timesteps % 50_000 < self.training_env.num_envs:
            self.logger.record("custom_plots/6_ent_coef", float(new_ent))
        return True


class StopCurriculumCallback(BaseCallback):
    def __init__(self, curriculum_steps, initial_min_steps, final_min_steps=0, verbose=0):
        super().__init__(verbose)
        self.curriculum_steps = curriculum_steps
        self.initial_min_steps = initial_min_steps
        self.final_min_steps = final_min_steps

    def _on_step(self) -> bool:
        progress = min(1.0, self.num_timesteps / max(1, self.curriculum_steps))
        current_min_steps = int(round(
            self.initial_min_steps + progress * (self.final_min_steps - self.initial_min_steps)
        ))
        self.training_env.env_method("set_min_steps_before_stop", current_min_steps)
        if self.num_timesteps % 50_000 < self.training_env.num_envs:
            self.logger.record("custom_plots/7_min_steps_before_stop", float(current_min_steps))
        return True


class StepSizeScheduleCallback(BaseCallback):
    def __init__(self, curriculum_steps, initial_frac=0.25, final_frac=0.04, verbose=0):
        super().__init__(verbose)
        self.curriculum_steps = curriculum_steps
        self.initial_frac = initial_frac
        self.final_frac = final_frac

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
        self.curriculum_steps = curriculum_steps
        self.initial_difficulty = initial_difficulty
        self.final_difficulty = final_difficulty

    def _on_step(self) -> bool:
        progress = min(1.0, self.num_timesteps / max(1, self.curriculum_steps))
        difficulty = self.initial_difficulty + progress * (self.final_difficulty - self.initial_difficulty)
        self.training_env.env_method("set_init_difficulty", difficulty)
        if self.num_timesteps % 50_000 < self.training_env.num_envs:
            self.logger.record("custom_plots/9_init_difficulty", float(difficulty))
        return True


class BrainTumorRL_Env(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, pytorch_dataset, max_steps=MAX_STEPS_PER_EPISODE, min_steps_before_stop=0,
                 step_frac=0.05, init_difficulty=1.0):
        super(BrainTumorRL_Env, self).__init__()
        self.dataset = pytorch_dataset
        self.max_steps = max_steps

        sample = self.dataset[0]
        self.channels, self.H, self.W = sample["image"].shape

        self.action_space = spaces.Discrete(N_ACTIONS)

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.channels + 1 + N_COORD_CHANNELS, self.H, self.W), dtype=np.uint8
        )

        self.step_frac = step_frac
        self.min_steps_before_stop = min_steps_before_stop
        self.init_difficulty = init_difficulty
        self.last_action = None

    def set_min_steps_before_stop(self, value):
        self.min_steps_before_stop = int(value)

    def set_step_frac(self, frac):
        self.step_frac = float(frac)

    def set_init_difficulty(self, value):
        self.init_difficulty = float(np.clip(value, 0.0, 1.0))

    def action_masks(self):
        mask = np.ones(N_ACTIONS, dtype=bool)
        at_w_min, at_w_max = self.w <= 12, self.w >= self.W
        at_h_min, at_h_max = self.h <= 12, self.h >= self.H

        if at_w_max: mask[4] = False
        if at_w_min: mask[5] = False
        if at_h_max: mask[6] = False
        if at_h_min: mask[7] = False

        mask[8] = self.current_step >= self.min_steps_before_stop
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

        gt_cx = self.gt_box[0] + self.gt_box[2] / 2.0
        gt_cy = self.gt_box[1] + self.gt_box[3] / 2.0
        gt_w = self.gt_box[2]
        gt_h = self.gt_box[3]

        d = self.init_difficulty
        pos_jitter_frac = 0.10 + d * 0.50
        size_jitter_frac = 0.10 + d * 0.60

        easy_cx = gt_cx + self.np_random.uniform(-pos_jitter_frac, pos_jitter_frac) * gt_w
        easy_cy = gt_cy + self.np_random.uniform(-pos_jitter_frac, pos_jitter_frac) * gt_h
        easy_w = gt_w * (1.0 + self.np_random.uniform(-size_jitter_frac, size_jitter_frac))
        easy_h = gt_h * (1.0 + self.np_random.uniform(-size_jitter_frac, size_jitter_frac))

        random_cx = self.np_random.uniform(self.W * 0.2, self.W * 0.8)
        random_cy = self.np_random.uniform(self.H * 0.2, self.H * 0.8)
        random_w = self.W * self.np_random.uniform(0.15, 0.40)
        random_h = self.H * self.np_random.uniform(0.15, 0.40)

        if self.np_random.uniform(0.0, 1.0) < d:
            self.cx, self.cy, self.w, self.h = random_cx, random_cy, random_w, random_h
        else:
            self.cx, self.cy, self.w, self.h = easy_cx, easy_cy, easy_w, easy_h

        self.w = np.clip(self.w, 12, self.W)
        self.h = np.clip(self.h, 12, self.H)
        self.cx = np.clip(self.cx, self.w / 2, self.W - self.w / 2)
        self.cy = np.clip(self.cy, self.h / 2, self.H - self.h / 2)

        self.current_step = 0
        self.episode_ious = []
        self.last_action = None

        current_box_xywh = self._get_xywh_box()
        self.previous_iou = self._compute_iou(current_box_xywh, self.gt_box)
        
        # Inizializza la distanza
        self.previous_dist = self._compute_center_distance(current_box_xywh, self.gt_box)

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
        
        coord_planes = build_coord_planes(
            self.cx, self.cy, self.w, self.h, self.W, self.H
        )
        
        return np.concatenate([self.current_image, box_mask_channel, coord_planes], axis=0)

    def _compute_iou(self, b1, b2):
        xi1, yi1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        xi2, yi2 = min(b1[0] + b1[2], b2[0] + b2[2]), min(b1[1] + b1[3], b2[1] + b2[3])
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union_area = (b1[2] * b1[3]) + (b2[2] * b2[3]) - inter_area
        return inter_area / max(1e-6, union_area)
        
    def _compute_center_distance(self, b1, b2):
        """ Calcola la distanza tra i centri di due bounding box. """
        c1_x, c1_y = b1[0] + b1[2] / 2.0, b1[1] + b1[3] / 2.0
        c2_x, c2_y = b2[0] + b2[2] / 2.0, b2[1] + b2[3] / 2.0
        return np.sqrt((c1_x - c2_x)**2 + (c1_y - c2_y)**2)

    def _terminal_bonus(self, iou: float) -> float:
        return float(np.clip((iou - STOP_IOU_BASELINE) * STOP_BONUS_SCALE, -STOP_BONUS_CLIP, STOP_BONUS_CLIP))

    def _oversize_penalty(self) -> float:
        area_ratio = (self.w * self.h) / float(self.W * self.H)
        excess = max(0.0, area_ratio - OVERSIZE_AREA_RATIO_THRESHOLD)
        return float(-excess * OVERSIZE_PENALTY_SCALE)

    def _oscillation_penalty(self, action: int) -> float:
        if self.last_action is not None and OPPOSITE_ACTIONS.get(action) == self.last_action:
            return OSCILLATION_PENALTY
        return 0.0

    def step(self, action):
        self.current_step += 1

        # ─── AZIONE STOP ───
        if action == 8:
            current_box_xywh = self._get_xywh_box()
            iou = self._compute_iou(current_box_xywh, self.gt_box)
            self.episode_ious.append(iou)

            reward = self._terminal_bonus(iou)
            terminated = True
            truncated = False

            reward += self._oversize_penalty()

            info = {
                "iou_instant": float(iou),
                "rew_components": {
                    "delta_iou": 0.0,
                    "delta_dist": 0.0,
                    "oversize_penalty": float(self._oversize_penalty()),
                    "oscillation_penalty": 0.0,
                    "total": float(reward),
                },
                "episode_metrics": {
                    "ep_iou_mean": float(np.mean(self.episode_ious)),
                    "ep_iou_std": float(np.std(self.episode_ious)),
                    "ep_iou_final": float(iou),
                },
                "action_taken": int(action),
            }
            self.previous_iou = iou
            self.last_action = action
            return self._get_obs(), float(reward), terminated, truncated, info

        # ─── AZIONI DI MOVIMENTO/RESIZE (0-7) ───
        # FIX: Assicura un movimento minimo di 5 pixel anche se il box diventa piccolissimo
        bs_x = max(self.step_frac * self.w, 2.0)
        bs_y = max(self.step_frac * self.h, 2.0)
        bs_w = max(self.step_frac * self.w, 2.0)
        bs_h = max(self.step_frac * self.h, 2.0)

        if action == 0:   self.cx -= bs_x
        elif action == 1: self.cx += bs_x
        elif action == 2: self.cy -= bs_y
        elif action == 3: self.cy += bs_y
        elif action == 4: self.w += bs_w
        elif action == 5: self.w -= bs_w
        elif action == 6: self.h += bs_h
        elif action == 7: self.h -= bs_h

        self.w = np.clip(self.w, 12, self.W)
        self.h = np.clip(self.h, 12, self.H)
        self.cx = np.clip(self.cx, self.w / 2, self.W - self.w / 2)
        self.cy = np.clip(self.cy, self.h / 2, self.H - self.h / 2)

        current_box_xywh = self._get_xywh_box()
        
        # ─── CALCOLO DELTA IOU ───
        iou = self._compute_iou(current_box_xywh, self.gt_box)
        self.episode_ious.append(iou)
        delta_iou = iou - self.previous_iou
        delta_iou_component = float(np.clip(delta_iou * DELTA_IOU_SCALE, -DELTA_IOU_CLIP, DELTA_IOU_CLIP))

        # ─── CALCOLO DELTA DISTANZA ───
        current_dist = self._compute_center_distance(current_box_xywh, self.gt_box)
        delta_dist = self.previous_dist - current_dist # Positivo = ci stiamo avvicinando
        delta_dist_component = float(np.clip(delta_dist * DISTANCE_REWARD_SCALE, -DISTANCE_REWARD_CLIP, DISTANCE_REWARD_CLIP))

        time_penalty_component = TIME_PENALTY
        oversize_penalty_component = self._oversize_penalty()
        oscillation_penalty_component = self._oscillation_penalty(action)

        reward = (
            delta_iou_component
            + delta_dist_component
            + time_penalty_component
            + oversize_penalty_component
            + oscillation_penalty_component
        )

        truncated = self.current_step >= self.max_steps
        terminated = False

        info = {
            "iou_instant": float(iou),
            "rew_components": {
                "delta_iou": float(delta_iou_component),
                "delta_dist": float(delta_dist_component),
                "oversize_penalty": float(oversize_penalty_component),
                "oscillation_penalty": float(oscillation_penalty_component),
                "total": float(reward),
            },
            "action_taken": int(action),
        }

        self.previous_iou = iou
        self.previous_dist = current_dist
        self.last_action = action

        if truncated:
            terminal_bonus = self._terminal_bonus(iou)
            reward += terminal_bonus + NO_STOP_PENALTY
            info["rew_components"]["total"] = float(reward)
            info["episode_metrics"] = {
                "ep_iou_mean": float(np.mean(self.episode_ious)),
                "ep_iou_std": float(np.std(self.episode_ious)),
                "ep_iou_final": float(iou)
            }
        
        # SALVA IL FRAME CORRENTE
        if self.current_idx % 100 == 0:
            obs_for_save = self._get_obs()
            os.makedirs(f"./episode_debug/ep_{self.current_idx}", exist_ok=True)
            self.save_observation_debug(obs_for_save, filename=f"./episode_debug/ep_{self.current_idx}/step_{self.current_step}.png")
        return self._get_obs(), float(reward), terminated, truncated, info

    def save_observation_debug(self, obs, filename="debug_obs.png"):
        """Salva l'osservazione a 8 canali come immagine leggibile."""
        # Prendi i primi 3 canali (immagine RGB)
        img = obs[:3].transpose(1, 2, 0)
        # Prendi il 4° canale (il box attuale)
        box = obs[3]
        # Combina o salva separatamente
        combined = (img * 255).astype(np.uint8)
        # Sovrapponi il box in rosso
        combined[:, :, 0] = np.where(box > 0, 255, combined[:, :, 0])
        cv2.imwrite(filename, combined)

# ─────────────────────────────────────────────────────────────────────────────
# VALUTAZIONE SU TEST SET
# ─────────────────────────────────────────────────────────────────────────────

class SingleSampleDataset:
    def __init__(self, sample):
        self.sample = sample
    def __len__(self):
        return 1
    def __getitem__(self, idx):
        return self.sample

def compute_gradcam(model, obs_4ch_uint8, device):
    obs_tensor = torch.tensor(np.expand_dims(obs_4ch_uint8, axis=0), dtype=torch.float32).to(device)
    activations, gradients = None, None

    def forward_hook(module, inp, out): nonlocal activations; activations = out
    def backward_hook(module, grad_in, grad_out): nonlocal gradients; gradients = grad_out[0]

    target_layer = model.policy.features_extractor.cnn[4]
    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    with torch.enable_grad():
        model.policy.eval()
        values = model.policy.predict_values(obs_tensor)
        model.policy.zero_grad()
        values.sum().backward()

    h1.remove(); h2.remove()

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


SIZE_BUCKET_EDGES = (0.10, 0.20)
IOU_THRESHOLDS = (0.3, 0.5, 0.7)
SUCCESS_IOU_THRESHOLD = 0.5


def _size_bucket(gt_area_ratio: float) -> str:
    lo, hi = SIZE_BUCKET_EDGES
    if gt_area_ratio < lo:
        return "small"
    if gt_area_ratio < hi:
        return "medium"
    return "large"


def _intensity_bucket(image_chw_float, gt_box) -> str:
    img = image_chw_float
    gray = img[0] if img.shape[0] != 3 else np.mean(img, axis=0)
    H, W = gray.shape
    x1 = int(np.clip(gt_box[0], 0, W - 1))
    y1 = int(np.clip(gt_box[1], 0, H - 1))
    x2 = int(np.clip(gt_box[0] + gt_box[2], x1 + 1, W))
    y2 = int(np.clip(gt_box[1] + gt_box[3], y1 + 1, H))
    tumor_mean = float(np.mean(gray[y1:y2, x1:x2])) if (y2 > y1 and x2 > x1) else float(np.mean(gray))
    background_mean = float(np.mean(gray))
    return "bright" if tumor_mean >= background_mean else "dark"


def _box_metrics(pred_box, gt_box, image_size=None):
    xi1, yi1 = max(pred_box[0], gt_box[0]), max(pred_box[1], gt_box[1])
    xi2 = min(pred_box[0] + pred_box[2], gt_box[0] + gt_box[2])
    yi2 = min(pred_box[1] + pred_box[3], gt_box[1] + gt_box[3])
    inter_px = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)

    gt_px = gt_box[2] * gt_box[3]
    pred_px = pred_box[2] * pred_box[3]
    union_px = gt_px + pred_px - inter_px

    metrics = {
        "iou": float(inter_px / max(1e-6, union_px)),
        "intersection_px": float(inter_px),
        "gt_px": float(gt_px),
        "pred_px": float(pred_px),
        "coverage_ratio": float(inter_px / max(1e-6, gt_px)),
        "size_ratio": float(pred_px / max(1e-6, gt_px)),
    }
    if image_size is not None:
        W, H = image_size
        gt_area_ratio = float(gt_px / max(1e-6, W * H))
        metrics["gt_area_ratio"] = gt_area_ratio
        metrics["size_bucket"] = _size_bucket(gt_area_ratio)
    return metrics


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
    last_obs = obs

    while True:
        current_mask = env.action_masks()
        action, _ = model.predict(obs, deterministic=True, action_masks=current_mask)
        obs, reward, terminated, truncated, info = env.step(action)
        last_obs = obs
        if terminated or truncated:
            break

    pred_box = env._get_xywh_box()
    gt_box = env.gt_box
    return pred_box, gt_box, last_obs, env.current_step


def evaluate_on_test_set(model, test_ds, output_dir, max_steps=MAX_STEPS_PER_EPISODE, seed=42,
                          max_samples=0, n_failure_cases=20):
    gradcam_dir = os.path.join(output_dir, "gradcam")
    boxes_dir = os.path.join(output_dir, "boxes")
    failures_dir = os.path.join(output_dir, "failure_cases")
    os.makedirs(gradcam_dir, exist_ok=True)
    os.makedirs(boxes_dir, exist_ok=True)
    os.makedirs(failures_dir, exist_ok=True)

    n_samples = len(test_ds) if max_samples <= 0 else min(max_samples, len(test_ds))
    print(f"[eval] Valutazione su {n_samples}/{len(test_ds)} campioni del test set...")

    device = model.device
    csv_path = os.path.join(output_dir, "metrics_per_sample.csv")
    fieldnames = ["idx", "iou", "intersection_px", "gt_px", "pred_px",
                  "coverage_ratio", "size_ratio", "gt_area_ratio", "size_bucket",
                  "intensity_bucket", "steps_used", "stopped_explicitly"]
    all_metrics = []

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx in range(n_samples):
            sample = test_ds[idx]
            single_ds = SingleSampleDataset(sample)
            env = BrainTumorRL_Env(pytorch_dataset=single_ds, max_steps=max_steps, min_steps_before_stop=0)

            pred_box, gt_box, last_obs, steps_used = _rollout_single(env, model, seed=seed)
            img_np = sample["image"].numpy()
            _, H, W = img_np.shape

            m = _box_metrics(pred_box, gt_box, image_size=(W, H))
            m["idx"] = idx
            m["steps_used"] = steps_used
            m["stopped_explicitly"] = bool(steps_used < max_steps)
            m["intensity_bucket"] = _intensity_bucket(img_np, gt_box)
            writer.writerow({k: m.get(k) for k in fieldnames})
            all_metrics.append(m)

            orig_bgr = _to_bgr_image(img_np)
            boxes_img = _draw_boxes(orig_bgr, gt_box, pred_box)
            cv2.imwrite(os.path.join(boxes_dir, f"{idx:04d}.png"), boxes_img)

            heatmap = compute_gradcam(model, last_obs, device)
            if heatmap is not None:
                heatmap_resized = cv2.resize(heatmap, (W, H))
                heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(orig_bgr, 0.6, heatmap_colored, 0.4, 0)
                overlay = _draw_boxes(overlay, gt_box, pred_box)
                cv2.imwrite(os.path.join(gradcam_dir, f"{idx:04d}.png"), overlay)

            if (idx + 1) % 25 == 0 or idx == n_samples - 1:
                print(f"[eval] {idx + 1}/{n_samples}  IoU={m['iou']:.3f}")

    ious = np.array([m["iou"] for m in all_metrics])

    worst_idx = np.argsort(ious)[:n_failure_cases]
    for rank, i in enumerate(worst_idx):
        idx = all_metrics[i]["idx"]
        src = os.path.join(boxes_dir, f"{idx:04d}.png")
        if os.path.exists(src):
            dst = os.path.join(failures_dir, f"rank{rank:02d}_idx{idx:04d}_iou{ious[i]:.3f}.png")
            img = cv2.imread(src)
            if img is not None:
                cv2.imwrite(dst, img)

    def _bucket_report(log, key, values):
        buckets = {}
        for m in all_metrics:
            buckets.setdefault(m[key], []).append(m["iou"])
        for name, vals in sorted(buckets.items()):
            vals = np.array(vals)
            log(f"  {name:8s} (n={len(vals):4d}) -> IoU media: {vals.mean():.4f}  mediana: {np.median(vals):.4f}")

    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        def log(line):
            print(line)
            f.write(line + "\n")

        log("─" * 60)
        log(f"Campioni valutati: {n_samples}")
        log(f"IoU -> media: {ious.mean():.4f}  mediana: {np.median(ious):.4f}  std: {ious.std():.4f}")
        for thr in IOU_THRESHOLDS:
            pct = float(np.mean(ious >= thr)) * 100.0
            log(f"IoU >= {thr:.1f}: {pct:.1f}% dei campioni")
        success_rate = float(np.mean(ious >= SUCCESS_IOU_THRESHOLD)) * 100.0
        log(f"Success rate (IoU >= {SUCCESS_IOU_THRESHOLD}): {success_rate:.1f}%")
        stop_rate = float(np.mean([m["stopped_explicitly"] for m in all_metrics])) * 100.0
        log(f"Episodi terminati con STOP esplicito (non timeout): {stop_rate:.1f}%")
        log("─" * 60)
        log("Breakdown per dimensione del tumore:")
        _bucket_report(log, "size_bucket", ious)
        log("─" * 60)
        log("Breakdown per polarita' di contrasto:")
        _bucket_report(log, "intensity_bucket", ious)
        log("─" * 60)
        log(f"Peggiori {len(worst_idx)} casi salvati in: {failures_dir}")
        log("─" * 60)

    return all_metrics


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE TRAINING
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--total-timesteps", type=int, default=None)
    arg_parser.add_argument(
        "--dataset-source", type=str, default=None, choices=["kaggle", "local", "synthetic"]
    )
    arg_parser.add_argument(
        "--dataset-path", type=str, default=None
    )
    arg_parser.add_argument(
        "--kaggle-id", type=str, default=None
    )
    cli_args, _unknown = arg_parser.parse_known_args()

    if cli_args.total_timesteps is not None:
        TOTAL_TIMESTEPS = cli_args.total_timesteps
    else:
        TOTAL_TIMESTEPS = int(os.environ.get("TOTAL_TIMESTEPS", 100_000))

    DATASET_SOURCE = cli_args.dataset_source or os.environ.get("DATASET_SOURCE", "kaggle")
    DATASET_LOCAL_PATH = cli_args.dataset_path or os.environ.get("DATASET_PATH", None)
    KAGGLE_ID = cli_args.kaggle_id or os.environ.get(
        "KAGGLE_DATASET_ID", "pkdarabi/brain-tumor-image-dataset-semantic-segmentation"
    )

    if DATASET_SOURCE == "local" and not DATASET_LOCAL_PATH:
        arg_parser.error("--dataset-source=local richiede anche --dataset-path")

    cfg = {
        "dataset": {
            "source": DATASET_SOURCE,
            "kaggle_id": KAGGLE_ID,
            "local_path": DATASET_LOCAL_PATH,
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

    os.makedirs(cfg["output"]["root"], exist_ok=True)
    print(f"[data] Sorgente dataset selezionata: '{DATASET_SOURCE}'")

    train_ds, val_ds, test_ds = get_datasets(cfg)

    check_env(BrainTumorRL_Env(pytorch_dataset=train_ds, max_steps=MAX_STEPS_PER_EPISODE))

    USE_SUBPROCESS = True
    N_ENVS = 6
    N_EPOCHS = 6

    # ─── MODIFICA: TEMPI DI CURRICULUM ALLUNGATI ───
    # Allungati per dare tempo all'agente di imparare a usare la bussola
    STOP_CURRICULUM_STEPS       = 600_000   # min_steps_before_stop: 201 → 0
    STEP_SIZE_CURRICULUM_STEPS  = 800_000   # step_frac: 0.045 → 0.012 (passi più fini/lenti)
    ENTROPY_SCHEDULE_STEPS      = 1_000_000 # ent_coef: 0.05 → 0.01 (meno esplorazione)
    INIT_BOX_CURRICULUM_STEPS   = 1_000_000 # init_difficulty: 0.0 → 1.0 (box iniziale più lontano/casuale)   

    INITIAL_MIN_STEPS_BEFORE_STOP = MAX_STEPS_PER_EPISODE + 1
    FINAL_MIN_STEPS_BEFORE_STOP = 0

    def make_env():
        def _init():
            env = BrainTumorRL_Env(
                pytorch_dataset=train_ds,
                max_steps=MAX_STEPS_PER_EPISODE,
                min_steps_before_stop=INITIAL_MIN_STEPS_BEFORE_STOP,
                init_difficulty=0.0,
            )
            env = ActionMasker(env, mask_fn)
            return env
        return _init

    if USE_SUBPROCESS:
        vec_env = SubprocVecEnv([make_env() for _ in range(N_ENVS)])
    else:
        vec_env = DummyVecEnv([make_env() for _ in range(N_ENVS)])

    vec_env = VecMonitor(vec_env)
    GAMMA = 0.99
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=10.0, gamma=GAMMA)

    def make_eval_env():
        env = BrainTumorRL_Env(pytorch_dataset=val_ds, max_steps=MAX_STEPS_PER_EPISODE, min_steps_before_stop=0)
        env = ActionMasker(env, mask_fn)
        return env

    eval_env_raw = DummyVecEnv([make_eval_env])
    eval_env_raw = VecMonitor(eval_env_raw)
    eval_env = VecNormalize(eval_env_raw, norm_obs=False, norm_reward=False, gamma=GAMMA, training=False)

    N_STEPS = 512

    visual_callback = VisionMetricsCallback(val_dataset=val_ds)
    entropy_callback = EntropyScheduleCallback(
        initial_ent=0.05, final_ent=0.01, schedule_steps=ENTROPY_SCHEDULE_STEPS
    )
    stop_curriculum_callback = StopCurriculumCallback(
        curriculum_steps=STOP_CURRICULUM_STEPS,
        initial_min_steps=INITIAL_MIN_STEPS_BEFORE_STOP,
        final_min_steps=FINAL_MIN_STEPS_BEFORE_STOP,
    )
    step_size_callback = StepSizeScheduleCallback(
        curriculum_steps=STEP_SIZE_CURRICULUM_STEPS,
        initial_frac=0.045,
        final_frac=0.012,
    )
    init_box_callback = InitBoxCurriculumCallback(
        curriculum_steps=INIT_BOX_CURRICULUM_STEPS,
        initial_difficulty=0.0,
        final_difficulty=1.0,
    )

    stop_on_no_improve = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=60, min_evals=90, verbose=1
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

    callbacks = CallbackList([
        visual_callback,
        entropy_callback,
        stop_curriculum_callback,
        step_size_callback,
        init_box_callback,
        eval_callback,
        checkpoint_callback,
    ])

    policy_kwargs = dict(
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )

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
        ent_coef=0.05,
        vf_coef=1.0,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="./ppo_brain_tumor_logs/",
        target_kl=0.03
    )
    for i in range(1, 110):
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks)
        model.save(F"./ppo_brain_tumor_logs/{TOTAL_TIMESTEPS*i}")
        i += 1
    vec_env.save("./ppo_brain_tumor_logs/vecnormalize_stats.pkl")

    evaluate_on_test_set(
        model=model, test_ds=test_ds, output_dir="./ppo_brain_tumor_logs/test_eval/",
        max_steps=MAX_STEPS_PER_EPISODE, seed=cfg["seed"], max_samples=0
    )