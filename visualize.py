"""
visualize_rollout.py
─────────────────────────────────────────────────────────────────────────────
Ricarica un modello MaskablePPO salvato (da ppo9.py) e mostra passo-passo
come l'agente muove/ridimensiona il box su uno o piu' campioni:
  - stampa a terminale azione scelta, reward, componenti reward, IoU corrente
  - salva un'immagine per ogni step con box GT (verde) e box agente (rosso)
  - genera una GIF animata del rollout completo (opzionale)

USO:
    python visualize_rollout.py --model ./ppo_brain_tumor_logs/best_model/best_model.zip
    python visualize_rollout.py --model final_model.zip --n-samples 5 --split test
    python visualize_rollout.py --model final_model.zip --sample-idx 12 --gif

Richiede che dataset.py, dataset_inspector.py, transforms.py e ppo9.py
siano nella stessa cartella (o nel PYTHONPATH), perche' l'env, le classi
e le funzioni di rendering vengono riusate direttamente da ppo9.py.
"""
import os
import argparse
import numpy as np
import cv2

from sb3_contrib import MaskablePPO

# Riusiamo l'ambiente e le utility di disegno gia' definite in ppo9.py,
# cosi' la visualizzazione e' garantita coerente con l'ambiente di training
# (stessa logica di IoU, stesso spazio di azione, stesso reward shaping).
from ppo import (
    BrainTumorRL_Env,
    MAX_STEPS_PER_EPISODE,
    N_ACTIONS,
    _to_bgr_image,
)
from dataset import get_datasets


ACTION_NAMES = {
    0: "cx -=  (sinistra)",
    1: "cx +=  (destra)",
    2: "cy -=  (su)",
    3: "cy +=  (giu')",
    4: "w  +=  (allarga)",
    5: "w  -=  (restringi)",
    6: "h  +=  (allunga)",
    7: "h  -=  (accorcia)",
    8: "STOP",
}


def build_cfg(dataset_source, dataset_path, kaggle_id, image_size):
    return {
        "dataset": {
            "source": dataset_source,
            "kaggle_id": kaggle_id,
            "local_path": dataset_path,
            "image_size": list(image_size),
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


def draw_step_frame(img_bgr, gt_box, pred_box, step_idx, action, reward, iou):
    """Disegna GT (verde), box agente (rosso) e un pannello info in alto."""
    out = img_bgr.copy()
    H, W = out.shape[:2]

    cv2.rectangle(out, (int(gt_box[0]), int(gt_box[1])),
                  (int(gt_box[0] + gt_box[2]), int(gt_box[1] + gt_box[3])),
                  (0, 255, 0), 2)
    cv2.rectangle(out, (int(pred_box[0]), int(pred_box[1])),
                  (int(pred_box[0] + pred_box[2]), int(pred_box[1] + pred_box[3])),
                  (0, 0, 255), 2)

    # Pannello info sopra l'immagine (banda nera + testo)
    banner_h = 60
    canvas = np.zeros((H + banner_h, W, 3), dtype=np.uint8)
    canvas[banner_h:, :, :] = out

    action_name = ACTION_NAMES.get(int(action), str(action))
    line1 = f"step {step_idx:03d}  action={action_name}"
    line2 = f"reward={reward:+.3f}   IoU={iou:.3f}"
    cv2.putText(canvas, line1, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, line2, (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return canvas


def run_rollout(model, env, sample_idx, out_dir, seed, make_gif, verbose=True):
    """Esegue un rollout deterministico su un singolo campione e salva ogni step."""
    sample_dir = os.path.join(out_dir, f"sample_{sample_idx:04d}")
    os.makedirs(sample_dir, exist_ok=True)

    obs, _ = env.reset(seed=seed)
    gt_box = env.gt_box
    img_np = env.dataset[env.current_idx]["image"].numpy()
    orig_bgr = _to_bgr_image(img_np)

    frames = []
    step = 0

    # frame iniziale (step 0, prima di qualunque azione)
    pred_box0 = env._get_xywh_box()
    iou0 = env._compute_iou(pred_box0, gt_box)
    frame0 = draw_step_frame(orig_bgr, gt_box, pred_box0, step, action=-1, reward=0.0, iou=iou0)
    cv2.putText(frame0, "INIZIO", (8, frame0.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(sample_dir, f"step_{step:03d}.png"), frame0)
    frames.append(frame0)

    if verbose:
        print(f"\n=== Sample {sample_idx} (idx dataset={env.current_idx}) ===")
        print(f"GT box (x,y,w,h) = {gt_box.round(1).tolist()}")
        print(f"step 000  box iniziale={pred_box0.round(1).tolist()}  IoU={iou0:.3f}")

    while True:
        action_mask = env.action_masks()
        action, _ = model.predict(obs, deterministic=True, action_masks=action_mask)
        action = int(action)

        obs, reward, terminated, truncated, info = env.step(action)
        step += 1

        pred_box = env._get_xywh_box()
        iou = info.get("iou_instant", env._compute_iou(pred_box, gt_box))

        if verbose:
            comp = info.get("rew_components", {})
            comp_str = "  ".join(f"{k}={v:+.3f}" for k, v in comp.items() if k != "total")
            print(f"step {step:03d}  action={ACTION_NAMES.get(action, action):18s} "
                  f"reward={reward:+.3f}  IoU={iou:.3f}   box={pred_box.round(1).tolist()}")
            if comp_str:
                print(f"          {comp_str}")

        frame = draw_step_frame(orig_bgr, gt_box, pred_box, step, action, reward, iou)
        if terminated:
            cv2.putText(frame, "STOP ESPLICITO", (8, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        elif truncated:
            cv2.putText(frame, "TIMEOUT (max_steps)", (8, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        cv2.imwrite(os.path.join(sample_dir, f"step_{step:03d}.png"), frame)
        frames.append(frame)

        if terminated or truncated:
            break

    if verbose:
        outcome = "STOP esplicito" if terminated else "timeout (max_steps raggiunto)"
        print(f"--- Fine episodio: {outcome} dopo {step} step. IoU finale = {iou:.3f} ---")

    if make_gif:
        gif_path = os.path.join(sample_dir, "rollout.gif")
        try:
            import imageio
            frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
            imageio.mimsave(gif_path, frames_rgb, duration=0.35)
            if verbose:
                print(f"GIF salvata in: {gif_path}")
        except ImportError:
            print("[warn] 'imageio' non installato: pip install imageio  -> GIF non generata.")

    return {"sample_idx": sample_idx, "steps": step, "final_iou": float(iou), "stopped": bool(terminated)}


def main():
    p = argparse.ArgumentParser(description="Visualizza passo-passo il rollout di un modello PPO salvato.")
    p.add_argument("--model", type=str, required=True, help="Path al file .zip del modello salvato.", default="./ppo_brain_tumor_logs/final_model")
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"],
                   help="Quale split del dataset usare per il rollout.")
    p.add_argument("--sample-idx", type=int, default=None,
                   help="Indice specifico nel dataset da usare. Se omesso, si usano i primi --n-samples.")
    p.add_argument("--n-samples", type=int, default=3, help="Numero di campioni da visualizzare.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, default="./rollout_viz")
    p.add_argument("--gif", action="store_true", help="Genera anche una GIF animata per ogni rollout.")
    p.add_argument("--max-steps", type=int, default=MAX_STEPS_PER_EPISODE)

    # stessi argomenti dataset usati in ppo9.py, per ricostruire lo stesso split
    p.add_argument("--dataset-source", type=str, default="kaggle", choices=["kaggle", "local", "synthetic"])
    p.add_argument("--dataset-path", type=str, default=None)
    p.add_argument("--kaggle-id", type=str, default="pkdarabi/brain-tumor-image-dataset-semantic-segmentation")
    p.add_argument("--image-size", type=int, nargs=2, default=[224, 224])

    args = p.parse_args()

    print(f"[load] Carico il modello da: {args.model}")
    model = MaskablePPO.load(args.model)

    print(f"[data] Carico dataset (source={args.dataset_source}, split={args.split})...")
    cfg = build_cfg(args.dataset_source, args.dataset_path, args.kaggle_id, args.image_size)
    train_ds, val_ds, test_ds = get_datasets(cfg)
    ds = {"train": train_ds, "val": val_ds, "test": test_ds}[args.split]
    print(f"[data] Split '{args.split}': {len(ds)} campioni.")

    os.makedirs(args.out_dir, exist_ok=True)

    if args.sample_idx is not None:
        indices = [args.sample_idx]
    else:
        indices = list(range(min(args.n_samples, len(ds))))

    results = []
    for idx in indices:
        sample = ds[idx]

        class _Single:
            def __len__(self_inner):
                return 1
            def __getitem__(self_inner, i):
                return sample

        # min_steps_before_stop=0: vogliamo vedere il comportamento "vero"
        # del modello, incluso quando decide di fermarsi, senza vincoli
        # di curriculum residui.
        env = BrainTumorRL_Env(
            pytorch_dataset=_Single(),
            max_steps=args.max_steps,
            min_steps_before_stop=0,
        )
        env.current_idx = 0  # coerente con _Single, serve per il salvataggio debug interno

        res = run_rollout(model, env, sample_idx=idx, out_dir=args.out_dir,
                           seed=args.seed, make_gif=args.gif)
        results.append(res)

    print("\n" + "─" * 60)
    print("Riepilogo:")
    for r in results:
        stato = "STOP esplicito" if r["stopped"] else "timeout"
        print(f"  sample {r['sample_idx']:>4d}: {r['steps']:>3d} step, "
              f"IoU finale={r['final_iou']:.3f}  ({stato})")
    print(f"\nImmagini per-step salvate in: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()