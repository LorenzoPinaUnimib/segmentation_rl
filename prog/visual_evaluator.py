"""
visual_evaluator.py
────────────────────
VisualEvaluator: fa girare il modello addestrato sul test set e mostra/
salva graficamente cosa succede passo per passo su ogni immagine:

  - watch(idx)                 -> finestra cv2 live (richiede display),
                                   utile per "vedere l'agente lavorare".
  - save_episode_video(idx)    -> .mp4 con un frame per step (box che si
                                   muove sull'immagine + HUD con IoU/reward).
  - save_episode_steps_grid(idx)-> un'unica immagine con i frame chiave
                                   (inizio, alcuni step intermedi, fine).
  - evaluate_dataset(...)       -> valutazione quantitativa su tutto il
                                   test set (CSV per campione + summary.txt,
                                   come nello script originale) PIU' grafici
                                   aggregati salvati come PNG (matplotlib):
                                   istogramma IoU, barre per bucket
                                   dimensione/contrasto, curva media di
                                   reward/IoU nel tempo, pie del success rate.
"""
import csv
import os

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    IOU_THRESHOLDS,
    MAX_STEPS_PER_EPISODE,
    SIZE_BUCKET_EDGES,
    SUCCESS_IOU_THRESHOLD,
)
from environment import BrainTumorRL_Env
from utils import to_bgr_image


class _SingleSampleDataset:
    """Dataset fittizio con un solo campione, per far girare BrainTumorRL_Env
    su un'immagine specifica del test set durante la valutazione."""
    def __init__(self, sample):
        self.sample = sample
    def __len__(self):
        return 1
    def __getitem__(self, idx):
        return self.sample


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


def compute_gradcam(model, obs, device):
    # FIX (Dict observation space): obs e' ora {"image":..., "box_vec":...}
    # (vedi BrainTumorRL_Env._get_obs) invece di un array unico -- si
    # costruisce un tensore per chiave, e il target_layer si legge dal
    # sotto-estrattore "image" del CombinedExtractor (MultiInputPolicy),
    # non piu' direttamente da policy.features_extractor.cnn[4].
    image_tensor = torch.tensor(np.expand_dims(obs["image"], axis=0), dtype=torch.float32).to(device)
    box_vec_tensor = torch.tensor(np.expand_dims(obs["box_vec"], axis=0), dtype=torch.float32).to(device)
    obs_tensor = {"image": image_tensor, "box_vec": box_vec_tensor}
    activations, gradients = None, None

    def forward_hook(m, i, o): nonlocal activations; activations = o
    def backward_hook(m, gi, go): nonlocal gradients; gradients = go[0]

    target_layer = model.policy.features_extractor.extractors["image"].cnn[4]
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
    pooled = torch.mean(gradients, dim=[0, 2, 3])
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled[i]
    heatmap = torch.mean(activations, dim=1).squeeze(0)
    heatmap = torch.max(heatmap, torch.zeros_like(heatmap))
    if torch.max(heatmap) > 0:
        heatmap /= torch.max(heatmap)
    return heatmap.cpu().detach().numpy()


class VisualEvaluator:
    def __init__(self, model, test_ds, output_dir, max_steps=MAX_STEPS_PER_EPISODE, seed=42,
                 localizer_fn=None, continuous_actions=False):
        self.model = model
        self.test_ds = test_ds
        self.output_dir = output_dir
        self.max_steps = max_steps
        self.seed = seed
        # FIX (warm-start supervisionato): stesso localizzatore usato in
        # training, cosi' la valutazione finale riflette le condizioni reali
        # con cui l'agente e' stato allenato (partenza dalla predizione del
        # regressore, non da un box completamente casuale come prima).
        self.localizer_fn = localizer_fn
        # FIX v7: con SAC/action space continuo non esistono action_masks da
        # passare a model.predict() (SAC non accetta quel parametro).
        self.continuous_actions = continuous_actions
        os.makedirs(self.output_dir, exist_ok=True)

    # ── rollout di un singolo episodio, con frame ad ogni step ──────────
    def _rollout_with_frames(self, idx, render_mode="rgb_array"):
        sample = self.test_ds[idx]
        env = BrainTumorRL_Env(
            pytorch_dataset=_SingleSampleDataset(sample),
            max_steps=self.max_steps, min_steps_before_stop=0,
            render_mode=render_mode, localizer_fn=self.localizer_fn,
            continuous_actions=self.continuous_actions,
        )
        obs, _ = env.reset(seed=self.seed)
        frames = [env.render()]
        step_log = []  # (action, reward, iou) per step

        while True:
            if self.continuous_actions:
                action, _ = self.model.predict(obs, deterministic=True)
            else:
                mask = env.action_masks()
                action, _ = self.model.predict(obs, deterministic=True, action_masks=mask)
            obs, reward, terminated, truncated, info = env.step(action)
            frames.append(env.render())
            step_log.append((int(action) if not self.continuous_actions else action.tolist(), float(reward), float(info["iou_instant"])))
            if terminated or truncated:
                break

        pred_box = env._get_xywh_box()
        gt_box = env.gt_box
        last_obs = obs
        env.close()
        return {
            "frames": frames, "step_log": step_log, "pred_box": pred_box,
            "gt_box": gt_box, "last_obs": last_obs, "steps_used": env.current_step,
            "image": sample["image"].numpy(),
        }

    # ── visualizzazione umana in tempo reale (richiede display) ─────────
    def watch(self, idx, fps=8):
        """Apre una finestra cv2 e mostra l'agente lavorare passo-passo
        sull'immagine idx del test set. Richiede un display disponibile."""
        result = self._rollout_with_frames(idx, render_mode="human")
        # _rollout_with_frames con render_mode="human" ha già mostrato i frame
        # in tempo reale via imshow; qui restiamo un attimo sull'ultimo frame.
        cv2.waitKey(int(1000 / fps) * 5)
        cv2.destroyAllWindows()
        return result

    # ── video/GIF dell'episodio ──────────────────────────────────────────
    def save_episode_video(self, idx, fps=6, fmt="mp4"):
        result = self._rollout_with_frames(idx, render_mode="rgb_array")
        frames = result["frames"]
        H, W = frames[0].shape[:2]
        out_dir = os.path.join(self.output_dir, "episode_videos")
        os.makedirs(out_dir, exist_ok=True)

        if fmt == "mp4":
            path = os.path.join(out_dir, f"episode_{idx:04d}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(path, fourcc, fps, (W, H))
            for f in frames:
                writer.write(f)
            writer.release()
        else:  # gif via imageio, se disponibile
            import imageio
            path = os.path.join(out_dir, f"episode_{idx:04d}.gif")
            rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
            imageio.mimsave(path, rgb_frames, fps=fps)

        print(f"[visual] video episodio salvato in: {path}")
        return path

    # ── griglia di step chiave in una sola immagine ──────────────────────
    def save_episode_steps_grid(self, idx, n_frames=6):
        result = self._rollout_with_frames(idx, render_mode="rgb_array")
        frames = result["frames"]
        picks = np.linspace(0, len(frames) - 1, num=min(n_frames, len(frames)), dtype=int)
        chosen = [frames[i] for i in picks]

        h, w = chosen[0].shape[:2]
        grid = np.zeros((h, w * len(chosen), 3), dtype=np.uint8)
        for i, f in enumerate(chosen):
            grid[:, i * w:(i + 1) * w] = f

        out_dir = os.path.join(self.output_dir, "episode_grids")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"episode_{idx:04d}_grid.png")
        cv2.imwrite(path, grid)
        print(f"[visual] griglia step salvata in: {path}")
        return path

    # ── valutazione quantitativa completa sul test set ──────────────────
    def evaluate_dataset(self, max_samples=0, n_failure_cases=20, save_videos_for_worst=5):
        gradcam_dir = os.path.join(self.output_dir, "gradcam")
        boxes_dir = os.path.join(self.output_dir, "boxes")
        failures_dir = os.path.join(self.output_dir, "failure_cases")
        plots_dir = os.path.join(self.output_dir, "plots")
        for d in (gradcam_dir, boxes_dir, failures_dir, plots_dir):
            os.makedirs(d, exist_ok=True)

        n_samples = len(self.test_ds) if max_samples <= 0 else min(max_samples, len(self.test_ds))
        print(f"[eval] Valutazione su {n_samples}/{len(self.test_ds)} campioni del test set...")

        device = self.model.device
        csv_path = os.path.join(self.output_dir, "metrics_per_sample.csv")
        fieldnames = ["idx", "iou", "intersection_px", "gt_px", "pred_px",
                      "coverage_ratio", "size_ratio", "gt_area_ratio", "size_bucket",
                      "intensity_bucket", "steps_used", "stopped_explicitly"]
        all_metrics = []
        all_step_logs = []  # per grafico "reward/IoU medi nel tempo"

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for idx in range(n_samples):
                result = self._rollout_with_frames(idx, render_mode="rgb_array")
                img_np = result["image"]
                _, H, W = img_np.shape

                m = _box_metrics(result["pred_box"], result["gt_box"], image_size=(W, H))
                m["idx"] = idx
                m["steps_used"] = result["steps_used"]
                m["stopped_explicitly"] = bool(result["steps_used"] < self.max_steps)
                m["intensity_bucket"] = _intensity_bucket(img_np, result["gt_box"])
                writer.writerow({k: m.get(k) for k in fieldnames})
                all_metrics.append(m)
                all_step_logs.append(result["step_log"])

                boxes_img = result["frames"][-1]
                cv2.imwrite(os.path.join(boxes_dir, f"{idx:04d}.png"), boxes_img)

                if not self.continuous_actions:
                    # FIX v7: compute_gradcam usa model.policy.predict_values,
                    # specifico delle ActorCriticPolicy di PPO -- SAC ha una
                    # struttura actor/critic diversa e non espone quel metodo.
                    heatmap = compute_gradcam(self.model, result["last_obs"], device)
                    if heatmap is not None:
                        orig_bgr = to_bgr_image(img_np)
                        heatmap_resized = cv2.resize(heatmap, (W, H))
                        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
                        overlay = cv2.addWeighted(orig_bgr, 0.6, heatmap_colored, 0.4, 0)
                        gt, pred = result["gt_box"], result["pred_box"]
                        cv2.rectangle(overlay, (int(gt[0]), int(gt[1])), (int(gt[0]+gt[2]), int(gt[1]+gt[3])), (0,255,0), 2)
                        cv2.rectangle(overlay, (int(pred[0]), int(pred[1])), (int(pred[0]+pred[2]), int(pred[1]+pred[3])), (0,0,255), 2)
                        cv2.imwrite(os.path.join(gradcam_dir, f"{idx:04d}.png"), overlay)

                if (idx + 1) % 25 == 0 or idx == n_samples - 1:
                    print(f"[eval] {idx + 1}/{n_samples}  IoU={m['iou']:.3f}")

        ious = np.array([m["iou"] for m in all_metrics])

        # salva video per i peggiori N casi, cosi' si vede esattamente cosa e' andato storto
        worst_idx = np.argsort(ious)[:n_failure_cases]
        for rank, i in enumerate(worst_idx):
            idx = all_metrics[i]["idx"]
            src = os.path.join(boxes_dir, f"{idx:04d}.png")
            if os.path.exists(src):
                dst = os.path.join(failures_dir, f"rank{rank:02d}_idx{idx:04d}_iou{ious[i]:.3f}.png")
                img = cv2.imread(src)
                if img is not None:
                    cv2.imwrite(dst, img)
        for rank, i in enumerate(worst_idx[:save_videos_for_worst]):
            self.save_episode_video(int(all_metrics[i]["idx"]))

        self._make_plots(all_metrics, all_step_logs, plots_dir)
        self._write_summary(all_metrics, ious, failures_dir)
        return all_metrics

    # ── grafici aggregati ────────────────────────────────────────────────
    def _make_plots(self, all_metrics, all_step_logs, plots_dir):
        ious = np.array([m["iou"] for m in all_metrics])

        # 1) istogramma IoU finale
        plt.figure(figsize=(6, 4))
        plt.hist(ious, bins=20, color="#4C72B0", edgecolor="black")
        plt.axvline(SUCCESS_IOU_THRESHOLD, color="red", linestyle="--", label=f"soglia successo={SUCCESS_IOU_THRESHOLD}")
        plt.xlabel("IoU finale"); plt.ylabel("numero campioni")
        plt.title("Distribuzione IoU finale sul test set")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "iou_histogram.png")); plt.close()

        # 2) IoU media per bucket dimensione e contrasto
        for key, fname, title in [("size_bucket", "iou_by_size.png", "IoU media per dimensione tumore"),
                                   ("intensity_bucket", "iou_by_intensity.png", "IoU media per contrasto tumore")]:
            buckets = {}
            for m in all_metrics:
                buckets.setdefault(m[key], []).append(m["iou"])
            names = sorted(buckets.keys())
            means = [np.mean(buckets[n]) for n in names]
            counts = [len(buckets[n]) for n in names]
            plt.figure(figsize=(5, 4))
            bars = plt.bar(names, means, color="#55A868")
            for b, c in zip(bars, counts):
                plt.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01, f"n={c}", ha="center", fontsize=8)
            plt.ylim(0, 1.0); plt.ylabel("IoU media"); plt.title(title)
            plt.tight_layout(); plt.savefig(os.path.join(plots_dir, fname)); plt.close()

        # 3) curva media di IoU e reward nel tempo (tutti gli episodi allineati per step index)
        max_len = max(len(log) for log in all_step_logs) if all_step_logs else 0
        if max_len > 0:
            iou_matrix = np.full((len(all_step_logs), max_len), np.nan)
            reward_matrix = np.full((len(all_step_logs), max_len), np.nan)
            for i, log in enumerate(all_step_logs):
                for t, (_, r, iou) in enumerate(log):
                    iou_matrix[i, t] = iou
                    reward_matrix[i, t] = r
            mean_iou_curve = np.nanmean(iou_matrix, axis=0)
            mean_reward_curve = np.nanmean(reward_matrix, axis=0)

            fig, ax1 = plt.subplots(figsize=(7, 4))
            ax1.plot(mean_iou_curve, color="#4C72B0", label="IoU media")
            ax1.set_xlabel("step"); ax1.set_ylabel("IoU media", color="#4C72B0")
            ax2 = ax1.twinx()
            ax2.plot(mean_reward_curve, color="#C44E52", alpha=0.6, label="reward media")
            ax2.set_ylabel("reward media", color="#C44E52")
            plt.title("Andamento medio di IoU e reward durante l'episodio (test set)")
            fig.tight_layout()
            fig.savefig(os.path.join(plots_dir, "iou_reward_over_steps.png")); plt.close(fig)

        # 4) pie del success rate
        success = np.mean(ious >= SUCCESS_IOU_THRESHOLD)
        plt.figure(figsize=(4, 4))
        plt.pie([success, 1 - success], labels=["successo", "fallimento"],
                autopct="%1.1f%%", colors=["#55A868", "#C44E52"])
        plt.title(f"Success rate (IoU >= {SUCCESS_IOU_THRESHOLD})")
        plt.tight_layout(); plt.savefig(os.path.join(plots_dir, "success_rate_pie.png")); plt.close()

        print(f"[visual] grafici aggregati salvati in: {plots_dir}")

    def _write_summary(self, all_metrics, ious, failures_dir):
        summary_path = os.path.join(self.output_dir, "summary.txt")
        with open(summary_path, "w") as f:
            def log(line):
                print(line)
                f.write(line + "\n")

            log("─" * 60)
            log(f"Campioni valutati: {len(all_metrics)}")
            log(f"IoU -> media: {ious.mean():.4f}  mediana: {np.median(ious):.4f}  std: {ious.std():.4f}")
            for thr in IOU_THRESHOLDS:
                pct = float(np.mean(ious >= thr)) * 100.0
                log(f"IoU >= {thr:.1f}: {pct:.1f}% dei campioni")
            success_rate = float(np.mean(ious >= SUCCESS_IOU_THRESHOLD)) * 100.0
            log(f"Success rate (IoU >= {SUCCESS_IOU_THRESHOLD}): {success_rate:.1f}%")
            stop_rate = float(np.mean([m["stopped_explicitly"] for m in all_metrics])) * 100.0
            log(f"Episodi terminati con STOP esplicito (non timeout): {stop_rate:.1f}%")
            log("─" * 60)

            def bucket_report(key):
                buckets = {}
                for m in all_metrics:
                    buckets.setdefault(m[key], []).append(m["iou"])
                for name, vals in sorted(buckets.items()):
                    vals = np.array(vals)
                    log(f"  {name:8s} (n={len(vals):4d}) -> IoU media: {vals.mean():.4f}  mediana: {np.median(vals):.4f}")

            log("Breakdown per dimensione del tumore:")
            bucket_report("size_bucket")
            log("─" * 60)
            log("Breakdown per polarita' di contrasto:")
            bucket_report("intensity_bucket")
            log("─" * 60)
            log(f"Casi peggiori salvati in: {failures_dir}")
            log("─" * 60)
