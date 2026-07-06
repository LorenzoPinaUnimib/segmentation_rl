"""
environment.py
───────────────
BrainTumorRL_Env: l'agente sposta/ridimensiona un bounding box su
un'immagine per farlo coincidere con la maschera del tumore.

Novità rispetto allo script originale:
  - render(mode="human")     -> finestra cv2 in tempo reale (richiede display)
  - render(mode="rgb_array") -> ritorna il frame BGR corrente come array,
                                 utile per registrare video/GIF durante la
                                 valutazione senza bisogno di un display.
  - action_masks(), step(), reset() invariati nella logica di reward.
"""
import os

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import (
    ACTION_NAMES,
    ACTION_SMOOTHNESS_PENALTY_SCALE,
    IMPROVEMENT_BONUS_CLIP,
    IMPROVEMENT_BONUS_SCALE,
    MAX_STEPS_PER_EPISODE,
    N_ACTIONS,
    N_COORD_CHANNELS,
    NO_STOP_PENALTY,
    OSCILLATION_PENALTY,
    OPPOSITE_ACTIONS,
    OVERSIZE_AREA_RATIO_THRESHOLD,
    OVERSIZE_PENALTY_SCALE,
    STOP_BONUS_CLIP,
    STOP_BONUS_SCALE,
    STOP_IOU_BASELINE,
    TIME_PENALTY,
    DELTA_IOU_CLIP,
    DELTA_IOU_SCALE,
    DISTANCE_REWARD_CLIP,
    DISTANCE_REWARD_SCALE,
)
from utils import build_box_vec, compute_center_distance, compute_iou


class BrainTumorRL_Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}

    def __init__(self, pytorch_dataset, max_steps=MAX_STEPS_PER_EPISODE, min_steps_before_stop=0,
                 step_frac=0.05, init_difficulty=1.0, render_mode=None, window_name="BrainTumorRL",
                 localizer_fn=None, continuous_actions=False):
        super().__init__()
        self.dataset = pytorch_dataset
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.window_name = window_name
        # FIX (warm-start supervisionato): funzione opzionale image[C,H,W] ->
        # (cx,cy,w,h) in pixel. Se fornita, il box iniziale parte dalla sua
        # predizione + rumore invece che da un jitter attorno alla GT o da
        # una posizione totalmente casuale (vedi reset() sotto). None ->
        # comportamento originale invariato (retrocompatibile).
        self.localizer_fn = localizer_fn
        # FIX v7 (ceiling strutturale): con azioni discrete a passo fisso
        # (step_frac), la precisione massima raggiungibile e' intrinsecamente
        # limitata dalla granularita' del passo -- non si puo' convergere su un
        # box che richiede un aggiustamento piu' fine dello step corrente. In
        # piu', l'intera famiglia di bug/comportamenti degeneri legati a STOP
        # (fermarsi troppo presto, mai fermarsi, reward shaping fragile) esiste
        # SOLO perche' STOP e' un'azione discreta appresa. continuous_actions=True
        # sostituisce le 9 azioni discrete con un vettore continuo [dx,dy,dw,dh]
        # (regressione diretta del delta, non un "nudge" fisso) e rimuove STOP:
        # l'episodio dura sempre max_steps, niente piu' da decidere su "quando
        # fermarsi". Vedi train.py --continuous per come si allena (SAC).
        self.continuous_actions = continuous_actions

        sample = self.dataset[0]
        self.channels, self.H, self.W = sample["image"].shape

        if self.continuous_actions:
            # [dx, dy, dw, dh] normalizzati in [-1,1], scalati da step_frac*dimensione
            # corrente del box in step() -- niente STOP, l'episodio dura max_steps.
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(N_ACTIONS)
        # FIX (Dict observation space): l'osservazione ora e' un Dict con due
        # chiavi invece di un unico array a canali concatenati:
        #   - "image": immagine + maschera box disegnata (channels+1 canali,
        #     invariato rispetto a prima MENO i 4 piani di coordinate);
        #   - "box_vec": (cx,cy,w,h) normalizzati come vettore numerico diretto,
        #     non piu' spalmati su piani immagine interi (vedi build_box_vec in
        #     utils.py per il perche').
        # Richiede policy="MultiInputPolicy" lato SB3/sb3-contrib (invece di
        # "CnnPolicy") in train.py, che smista automaticamente ogni chiave al
        # sotto-estrattore giusto (CNN per "image", flatten per "box_vec") e
        # concatena le feature prima delle head policy/value.
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(self.channels + 1, self.H, self.W), dtype=np.uint8),
            "box_vec": spaces.Box(low=0.0, high=1.0, shape=(N_COORD_CHANNELS,), dtype=np.float32),
        })

        self.step_frac = step_frac
        self.min_steps_before_stop = min_steps_before_stop
        self.init_difficulty = init_difficulty
        self.last_action = None

        # stato usato solo per il render (ultima ricompensa/iou/azione)
        self._last_reward = 0.0
        self._last_iou = 0.0

    # ── setter usati dai callback di curriculum ────────────────────────
    def set_min_steps_before_stop(self, value):
        self.min_steps_before_stop = int(value)

    def set_step_frac(self, frac):
        self.step_frac = float(frac)

    def set_init_difficulty(self, value):
        self.init_difficulty = float(np.clip(value, 0.0, 1.0))

    # ── action masking ──────────────────────────────────────────────────
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

    # ── reset ────────────────────────────────────────────────────────────
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

        if self.localizer_fn is not None:
            # FIX (warm-start supervisionato): il box iniziale parte dalla
            # predizione del regressore supervisionato + rumore, NON da un
            # jitter attorno alla GT (che a test time su un'immagine nuova
            # non esiste) e NON da una posizione totalmente casuale. La scala
            # del rumore usa le dimensioni PREDETTE (pred_w/pred_h), non quelle
            # della GT: e' un'informazione disponibile anche a inferenza reale,
            # quindi niente leakage. `d` ora significa "quanto rumore aggiungo
            # sopra la predizione", non piu' "quanto lontano parto dalla GT".
            pred_cx, pred_cy, pred_w, pred_h = self.localizer_fn(self.current_image)
            pos_jitter_frac = 0.03 + d * 0.35
            size_jitter_frac = 0.03 + d * 0.35
            self.cx = pred_cx + self.np_random.uniform(-pos_jitter_frac, pos_jitter_frac) * pred_w
            self.cy = pred_cy + self.np_random.uniform(-pos_jitter_frac, pos_jitter_frac) * pred_h
            self.w = pred_w * (1.0 + self.np_random.uniform(-size_jitter_frac, size_jitter_frac))
            self.h = pred_h * (1.0 + self.np_random.uniform(-size_jitter_frac, size_jitter_frac))
        else:
            # comportamento originale (nessun localizzatore disponibile):
            # blend fra jitter attorno alla GT ("easy") e posizione totalmente
            # casuale ("hard"), con probabilita' pari a `d`.
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
        self.last_action_vec = np.zeros(4, dtype=np.float32)
        self._last_reward = 0.0

        current_box_xywh = self._get_xywh_box()
        self.previous_iou = compute_iou(current_box_xywh, self.gt_box)
        self.previous_dist = compute_center_distance(current_box_xywh, self.gt_box)
        self._last_iou = self.previous_iou
        # FIX v4: serve per il bonus di miglioramento relativo in _terminal_bonus.
        self.initial_iou = self.previous_iou

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), {}

    # ── geometria/osservazione ──────────────────────────────────────────
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

        image_obs = np.concatenate([self.current_image, box_mask_channel], axis=0)
        box_vec = build_box_vec(self.cx, self.cy, self.w, self.h, self.W, self.H)
        return {"image": image_obs, "box_vec": box_vec}

    def _terminal_bonus(self, iou: float) -> float:
        absolute = np.clip((iou - STOP_IOU_BASELINE) * STOP_BONUS_SCALE, -STOP_BONUS_CLIP, STOP_BONUS_CLIP)
        # FIX v4: bonus supplementare per il miglioramento rispetto al punto di
        # partenza dell'episodio, indipendente dalla difficolta' -- vedi
        # commento su IMPROVEMENT_BONUS_SCALE in config.py.
        improvement = np.clip((iou - self.initial_iou) * IMPROVEMENT_BONUS_SCALE,
                               -IMPROVEMENT_BONUS_CLIP, IMPROVEMENT_BONUS_CLIP)
        return float(absolute + improvement)

    def _oversize_penalty(self) -> float:
        area_ratio = (self.w * self.h) / float(self.W * self.H)
        excess = max(0.0, area_ratio - OVERSIZE_AREA_RATIO_THRESHOLD)
        return float(-excess * OVERSIZE_PENALTY_SCALE)

    def _oscillation_penalty(self, action: int) -> float:
        if self.last_action is not None and OPPOSITE_ACTIONS.get(action) == self.last_action:
            return OSCILLATION_PENALTY
        return 0.0

    def _smoothness_penalty(self, action_vec: np.ndarray) -> float:
        """Equivalente continuo di _oscillation_penalty: penalizza cambi bruschi
        di direzione/intensita' tra un'azione e la successiva (jerk), non la
        magnitudine assoluta -- altrimenti scoraggerebbe anche i movimenti grandi
        legittimi necessari a inizio episodio quando si e' ancora lontani dal target.
        """
        jerk = float(np.sum((action_vec - self.last_action_vec) ** 2))
        return -jerk * ACTION_SMOOTHNESS_PENALTY_SCALE

    # ── step ─────────────────────────────────────────────────────────────
    def step(self, action):
        self.current_step += 1

        if self.continuous_actions:
            return self._step_continuous(action)

        if action == 8:  # STOP
            current_box_xywh = self._get_xywh_box()
            iou = compute_iou(current_box_xywh, self.gt_box)
            self.episode_ious.append(iou)

            reward = self._terminal_bonus(iou) + self._oversize_penalty()
            terminated, truncated = True, False

            info = {
                "iou_instant": float(iou),
                "rew_components": {
                    "delta_iou": 0.0, "delta_dist": 0.0,
                    "oversize_penalty": float(self._oversize_penalty()),
                    "oscillation_penalty": 0.0, "total": float(reward),
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
            self._last_reward, self._last_iou = reward, iou
            if self.render_mode == "human":
                self.render()
            return self._get_obs(), float(reward), terminated, truncated, info

        # ── movimento/resize (0-7) ──
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

        iou = compute_iou(current_box_xywh, self.gt_box)
        self.episode_ious.append(iou)
        delta_iou = iou - self.previous_iou
        delta_iou_component = float(np.clip(delta_iou * DELTA_IOU_SCALE, -DELTA_IOU_CLIP, DELTA_IOU_CLIP))

        current_dist = compute_center_distance(current_box_xywh, self.gt_box)
        delta_dist = self.previous_dist - current_dist
        delta_dist_component = float(np.clip(delta_dist * DISTANCE_REWARD_SCALE, -DISTANCE_REWARD_CLIP, DISTANCE_REWARD_CLIP))

        time_penalty_component = TIME_PENALTY
        oversize_penalty_component = self._oversize_penalty()
        oscillation_penalty_component = self._oscillation_penalty(action)

        reward = (delta_iou_component + delta_dist_component + time_penalty_component
                  + oversize_penalty_component + oscillation_penalty_component)

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
        self._last_reward, self._last_iou = reward, iou

        if truncated:
            reward += self._terminal_bonus(iou) + NO_STOP_PENALTY
            info["rew_components"]["total"] = float(reward)
            info["episode_metrics"] = {
                "ep_iou_mean": float(np.mean(self.episode_ious)),
                "ep_iou_std": float(np.std(self.episode_ious)),
                "ep_iou_final": float(iou),
            }

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), float(reward), terminated, truncated, info

    def _step_continuous(self, action: np.ndarray):
        """Ramo continuo: action = [dax, day, daw, dah] in [-1,1]. Niente STOP:
        l'episodio dura sempre max_steps, il reward finale (_terminal_bonus)
        viene applicato SEMPRE alla truncation, non solo se l'agente "sceglie"
        di fermarsi -- rimuove l'intera classe di patologie legate al timing
        di STOP che ha causato la maggior parte dei problemi visti finora."""
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        bs_x = max(self.step_frac * self.w, 1.0)
        bs_y = max(self.step_frac * self.h, 1.0)
        bs_w = max(self.step_frac * self.w, 1.0)
        bs_h = max(self.step_frac * self.h, 1.0)

        self.cx += action[0] * bs_x
        self.cy += action[1] * bs_y
        self.w += action[2] * bs_w
        self.h += action[3] * bs_h

        self.w = np.clip(self.w, 12, self.W)
        self.h = np.clip(self.h, 12, self.H)
        self.cx = np.clip(self.cx, self.w / 2, self.W - self.w / 2)
        self.cy = np.clip(self.cy, self.h / 2, self.H - self.h / 2)

        current_box_xywh = self._get_xywh_box()

        iou = compute_iou(current_box_xywh, self.gt_box)
        self.episode_ious.append(iou)
        delta_iou = iou - self.previous_iou
        delta_iou_component = float(np.clip(delta_iou * DELTA_IOU_SCALE, -DELTA_IOU_CLIP, DELTA_IOU_CLIP))

        current_dist = compute_center_distance(current_box_xywh, self.gt_box)
        delta_dist = self.previous_dist - current_dist
        delta_dist_component = float(np.clip(delta_dist * DISTANCE_REWARD_SCALE, -DISTANCE_REWARD_CLIP, DISTANCE_REWARD_CLIP))

        time_penalty_component = TIME_PENALTY
        oversize_penalty_component = self._oversize_penalty()
        smoothness_penalty_component = self._smoothness_penalty(action)

        reward = (delta_iou_component + delta_dist_component + time_penalty_component
                  + oversize_penalty_component + smoothness_penalty_component)

        truncated = self.current_step >= self.max_steps
        terminated = False

        info = {
            "iou_instant": float(iou),
            "rew_components": {
                "delta_iou": float(delta_iou_component),
                "delta_dist": float(delta_dist_component),
                "oversize_penalty": float(oversize_penalty_component),
                "oscillation_penalty": float(smoothness_penalty_component),
                "total": float(reward),
            },
        }

        self.previous_iou = iou
        self.previous_dist = current_dist
        self.last_action_vec = action
        self._last_reward, self._last_iou = reward, iou

        if truncated:
            reward += self._terminal_bonus(iou)
            info["rew_components"]["total"] = float(reward)
            info["episode_metrics"] = {
                "ep_iou_mean": float(np.mean(self.episode_ious)),
                "ep_iou_std": float(np.std(self.episode_ious)),
                "ep_iou_final": float(iou),
            }

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), float(reward), terminated, truncated, info

    # ── rendering ────────────────────────────────────────────────────────
    def _render_frame_bgr(self) -> np.ndarray:
        """Costruisce il frame BGR con GT box (verde), box predetto (rosso) e HUD testuale."""
        img = self.current_image
        if img.shape[0] == 3:
            frame = np.transpose(img, (1, 2, 0)).copy()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame = cv2.cvtColor(img[0], cv2.COLOR_GRAY2BGR)

        gt = self.gt_box
        cv2.rectangle(frame, (int(gt[0]), int(gt[1])),
                      (int(gt[0] + gt[2]), int(gt[1] + gt[3])), (0, 255, 0), 2)

        pred = self._get_xywh_box()
        cv2.rectangle(frame, (int(pred[0]), int(pred[1])),
                      (int(pred[0] + pred[2]), int(pred[1] + pred[3])), (0, 0, 255), 2)

        action_label = ACTION_NAMES.get(self.last_action, "-")
        hud_lines = [
            f"step {self.current_step}/{self.max_steps}",
            f"action: {action_label}",
            f"IoU: {self._last_iou:.3f}",
            f"reward: {self._last_reward:+.3f}",
        ]
        # riquadro semi-trasparente per rendere leggibile il testo
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (150, 16 + 16 * len(hud_lines)), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)
        for i, line in enumerate(hud_lines):
            cv2.putText(frame, line, (6, 16 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (255, 255, 255), 1, cv2.LINE_AA)
        return frame

    def render(self):
        frame = self._render_frame_bgr()
        if self.render_mode == "human":
            cv2.imshow(self.window_name, frame)
            cv2.waitKey(1)
            return None
        if self.render_mode == "rgb_array" or self.render_mode is None:
            return frame
        return frame

    def close(self):
        if self.render_mode == "human":
            cv2.destroyWindow(self.window_name)

    def save_observation_debug(self, obs, filename="debug_obs.png"):
        """Salva l'osservazione multi-canale come immagine leggibile (debug rapido su disco)."""
        # FIX (Dict observation space): obs ora e' {"image":..., "box_vec":...}
        # invece di un array unico -- si legge il canale immagine da li'.
        img_obs = obs["image"] if isinstance(obs, dict) else obs
        img = img_obs[:3].transpose(1, 2, 0)
        box = img_obs[3]
        combined = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        combined[:, :, 0] = np.where(box > 0, 255, combined[:, :, 0])
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        cv2.imwrite(filename, combined)
