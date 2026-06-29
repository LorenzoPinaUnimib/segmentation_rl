"""
rl_trainer.py — RL training loop for DQN mask refinement agent.

For each episode:
  1) Sample a random image from the training set
  2) Run baseline model → get prob map + initial mask
  3) Run RL episode: agent refines mask step-by-step
  4) Collect experiences → replay buffer
  5) Train agent

Evaluation: after every eval_every episodes, run greedy policy on val set
            and record dice improvement vs baseline.
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List
from tqdm import tqdm

from ..rl.environment import MaskRefinementEnv
from ..rl.agent import DQNAgent
from ..eval.metrics import dice_coefficient, iou_score


class RLTrainer:
    """RL training and evaluation for mask refinement."""

    def __init__(
        self,
        model: nn.Module,          # pretrained baseline (eval mode)
        agent: DQNAgent,
        env: MaskRefinementEnv,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        cfg: dict,
    ):
        self.model       = model.to(device)
        self.model.eval()
        self.agent       = agent
        self.env         = env
        self.train_loader= train_loader
        self.val_loader  = val_loader
        self.device      = device
        self.cfg         = cfg

        rl = cfg.get("rl", {})
        self.episodes    = rl.get("episodes", 300)
        self.target_freq = rl.get("target_update_freq", 20)

        self.history: Dict[str, List] = {
            "rewards":          [],
            "dice_improvement": [],
            "epsilon":          [],
            "val_dice_rl":      [],
            "val_dice_bl":      [],
        }
        self.best_val_dice = 0.0
        self._train_iter = None

    # ── Inference helper ─────────────────────────────────────────────────────

    @torch.no_grad()
    def _get_prob_map(self, image_t: torch.Tensor) -> np.ndarray:
        """Run baseline model, return [H,W] probability numpy array."""
        logits = self.model(image_t.unsqueeze(0).to(self.device))
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()
        return prob  # [H, W]

    def _sample_from_loader(self, loader: DataLoader):
        """Infinite cycling over a DataLoader, return one random batch sample."""
        if self._train_iter is None:
            self._train_iter = iter(loader)
        try:
            batch = next(self._train_iter)
        except StopIteration:
            self._train_iter = iter(loader)
            batch = next(self._train_iter)

        # Pick first sample from batch
        idx    = np.random.randint(batch["image"].size(0))
        image  = batch["image"][idx]    # [C, H, W]
        mask   = batch["mask"][idx]     # [1, H, W]
        return image, mask

    @staticmethod
    def _tensor_to_np(t: torch.Tensor) -> np.ndarray:
        """[C,H,W] or [1,H,W] tensor → [H,W] numpy float32."""
        arr = t.cpu().numpy()
        if arr.ndim == 3:
            arr = arr[0] if arr.shape[0] in [1, 3] else arr.mean(0)
        return arr.astype(np.float32)

    # ── Single RL episode ────────────────────────────────────────────────────

    def _run_episode(self, image_t, mask_gt_t, train: bool) -> Dict:
        """
        Run one RL episode for a single image.
        Returns episode info: total_reward, dice_bl, dice_rl.
        """
        image_np  = self._tensor_to_np(image_t)
        gt_np     = self._tensor_to_np(mask_gt_t)
        prob_np   = self._get_prob_map(image_t)

        # Baseline dice
        baseline_mask = (prob_np > 0.5).astype(np.float32)
        dice_bl = dice_coefficient(baseline_mask, gt_np)

        # Reset env
        state = self.env.reset(image_np, prob_np, gt_np)
        total_reward = 0.0

        while True:
            action = self.agent.act(state, greedy=not train)
            next_state, reward, done, info = self.env.step(action)

            if train:
                self.agent.remember(state, action, reward, next_state, done)
                self.agent.learn()

            total_reward += reward
            state = next_state

            if done:
                break

        refined_mask = self.env.get_refined_mask()
        dice_rl = dice_coefficient(refined_mask, gt_np)

        return {
            "reward":    total_reward,
            "dice_bl":   dice_bl,
            "dice_rl":   dice_rl,
            "delta":     dice_rl - dice_bl,
        }

    # ── Validation ───────────────────────────────────────────────────────────

    def _validate(self) -> Dict:
        """Run greedy policy on val_loader, return mean metrics."""
        dice_bl_list, dice_rl_list = [], []

        for batch in self.val_loader:
            for i in range(min(4, batch["image"].size(0))):  # max 4 per batch
                image_t  = batch["image"][i]
                mask_gt_t = batch["mask"][i]

                ep_info = self._run_episode(image_t, mask_gt_t, train=False)
                dice_bl_list.append(ep_info["dice_bl"])
                dice_rl_list.append(ep_info["dice_rl"])

            if len(dice_rl_list) >= 20:  # limit val episodes
                break

        return {
            "val_dice_bl": float(np.mean(dice_bl_list)),
            "val_dice_rl": float(np.mean(dice_rl_list)),
        }

    # ── Main training loop ───────────────────────────────────────────────────

    def train(self, eval_every: int = 20) -> Dict:
        print(f"\n{'='*60}")
        print(f" RL Training for {self.episodes} episodes")
        print(f"{'='*60}")

        for ep in range(1, self.episodes + 1):
            image_t, mask_gt_t = self._sample_from_loader(self.train_loader)
            ep_info = self._run_episode(image_t, mask_gt_t, train=True)

            self.history["rewards"].append(ep_info["reward"])
            self.history["dice_improvement"].append(ep_info["delta"])
            self.history["epsilon"].append(self.agent.eps)

            # Update epsilon and target net
            self.agent.update_epsilon()
            self.agent.maybe_sync_target()

            # Periodic validation
            if ep % eval_every == 0:
                val_info = self._validate()
                self.history["val_dice_rl"].append(val_info["val_dice_rl"])
                self.history["val_dice_bl"].append(val_info["val_dice_bl"])

                print(
                    f"Ep {ep:04d}/{self.episodes} | "
                    f"Reward={ep_info['reward']:+.3f} | "
                    f"Dice: BL={ep_info['dice_bl']:.4f} RL={ep_info['dice_rl']:.4f} "
                    f"(Δ{ep_info['delta']:+.4f}) | "
                    f"ε={self.agent.eps:.3f} | "
                    f"ValDice RL={val_info['val_dice_rl']:.4f} "
                    f"BL={val_info['val_dice_bl']:.4f}"
                )

                if val_info["val_dice_rl"] > self.best_val_dice:
                    self.best_val_dice = val_info["val_dice_rl"]
                    self.agent.save("best_rl_agent.pth")

        self.agent.save("final_rl_agent.pth")
        return self.history
