"""
train_treerl.py
─────────────────
Training + valutazione (recall@IoU, come Tabelle 1/2 del paper) per la
pipeline Tree-RL, fedele a Jie, Liang, Feng, Jin, Lu, Yan, "Tree-Structured
Reinforcement Learning for Sequential Object Localization", NeurIPS 2016
(arXiv:1703.02710) -- https://arxiv.org/pdf/1703.02710

Riusa dataset.py, utils.py::compute_iou cosi' come sono (formato box e
pipeline immagini identici al resto del progetto); NON riusa environment.py/
q_network.py/train.py di Caicedo & Lazebnik, che implementano un altro MDP
(single-object, con trigger) -- vedi environment_treerl.py per il dettaglio
delle differenze strutturali.

ASSUNZIONI NON VERIFICATE contro il testo del paper (segnalate anche nei
singoli file), riassunte qui per trasparenza:
  - Ordine/posizionamento esatto delle 5 sotto-finestre di scaling (Fig. 2
    non ha didascalia testuale sull'ordine) -> 4 angoli + centro.
  - Ancoraggio geometrico delle azioni SHORTER/LONGER (il paper descrive
    l'effetto, non il punto di ancoraggio) -> centrate.
  - Dimensioni della testa MLP (Fig. 4 non da' numeri) -> 2 hidden layer da
    1024, per coerenza con la testa "fedele al paper" gia' usata altrove nel
    progetto per Caicedo & Lazebnik.
  - Ottimizzatore e learning rate (non menzionati nel testo) -> Adam,
    lr=1e-4 di default, entrambi configurabili da CLI.
  - Nessun target network separato: l'Eq. 3 del paper usa LETTERALMENTE lo
    stesso theta_i sia per la stima corrente sia per il bootstrap sul
    next-state (nessuna menzione di target network separato, a differenza
    del DQN "Nature" standard) -- implementato cosi' di default, con
    --use-target-net come deviazione opzionale se l'instabilita' lo richiede.
"""
import argparse
import csv
import datetime
import os
import random
import time
import gc

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
from tqdm import tqdm
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter

from dataset import get_datasets
from utils import compute_iou
from environment_treerl import TreeRLEnv, get_gt_boxes_from_mask, apply_action
from q_network_treerl import TreeRLQNet, build_backbone
from tree_search_treerl import tree_search
from config_treerl import (
    N_ACTIONS, SCALE_ACTIONS, TRANSLATE_ACTIONS, HISTORY_DIM,
    GAMMA, MAX_STEPS_PER_EPISODE, REPLAY_SIZE, BATCH_SIZE, N_EPOCHS,
    EPSILON_START, EPSILON_END, EPSILON_DECAY_EPOCHS, DEFAULT_TREE_LEVELS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Replay buffer: window_feat/next_window_feat (4096) + global_feat CONDIVISO
# tra s e s' (la feature dell'intera immagine non cambia durante l'episodio,
# quindi si salva UNA sola volta per transizione invece di due) + history/
# next_history (650) + action + reward. Nessun flag "done": l'Eq. 3 del
# paper non ne ha uno, il bootstrap TD non va mai azzerato (vedi docstring
# del modulo e di environment_treerl.py).
# ─────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity, feat_dim, history_dim):
        self.capacity = capacity
        self.pos = 0
        self.size = 0
        self.window_feats = np.zeros((capacity, feat_dim), dtype=np.float32)
        self.next_window_feats = np.zeros((capacity, feat_dim), dtype=np.float32)
        self.global_feats = np.zeros((capacity, feat_dim), dtype=np.float32)
        self.histories = np.zeros((capacity, history_dim), dtype=np.float32)
        self.next_histories = np.zeros((capacity, history_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)

    def estimated_bytes(self):
        return (
            self.window_feats.nbytes + self.next_window_feats.nbytes
            + self.global_feats.nbytes + self.histories.nbytes
            + self.next_histories.nbytes + self.actions.nbytes + self.rewards.nbytes
        )

    def push(self, w_feat, g_feat, hist, action, reward, next_w_feat, next_hist):
        idx = self.pos
        self.window_feats[idx] = w_feat
        self.next_window_feats[idx] = next_w_feat
        self.global_feats[idx] = g_feat
        self.histories[idx] = hist
        self.next_histories[idx] = next_hist
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.pos = (idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def __len__(self):
        return self.size

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.window_feats[idx], self.global_feats[idx], self.histories[idx],
            self.actions[idx], self.rewards[idx],
            self.next_window_feats[idx], self.next_histories[idx],
        )


# ─────────────────────────────────────────────────────────────────────────────
# Dataset: config + filtro immagini senza oggetti (vedi docstring di
# environment_treerl.py sul perche' NON si usa un fallback fittizio).
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset_config(args):
    dataset_source = args.dataset_source or os.environ.get("DATASET_SOURCE", "kaggle")
    dataset_local_path = args.dataset_path or os.environ.get("DATASET_PATH", None)
    kaggle_id = args.kaggle_id or os.environ.get(
        "KAGGLE_DATASET_ID", "pkdarabi/brain-tumor-image-dataset-semantic-segmentation"
    )
    if dataset_source == "local" and not dataset_local_path:
        raise SystemExit("--dataset-source=local richiede anche --dataset-path")
    return {
        "dataset": {
            "source": dataset_source, "kaggle_id": kaggle_id, "local_path": dataset_local_path,
            "image_size": [224, 224], "in_channels": 3,
            "train_ratio": (1501 / 2145), "val_ratio": (429 / 2145), "cache_pairs": False,
        },
        "preprocessing": {
            "normalization": "per_image", "binarize_mask": True, "mask_threshold": 0.5,
            "white_balance": True, "clahe": True, "denoise": False,
        },
        "training": {"batch_size": 512, "num_workers": 0},
        "output": {"root": args.output_root},
        "seed": 42,
    }


def filter_dataset_with_objects(dataset, min_area, name):
    """Tiene solo le immagini con almeno un oggetto valido (>= min_area
    pixel) nella maschera, secondo la stessa estrazione a componenti
    connesse usata dall'ambiente (get_gt_boxes_from_mask). A differenza di
    environment.py (che assegna un gt_box fittizio "quarto centrale" alle
    maschere vuote), qui le immagini senza oggetto vengono ESCLUSE
    dall'epoca invece di iniettare un target arbitrario -- vedi docstring di
    environment_treerl.py."""
    keep_idx = []
    n_objects_total = 0
    for i in range(len(dataset)):
        mask = dataset[i]["mask"].numpy().squeeze(0)
        boxes = get_gt_boxes_from_mask(mask, min_area=min_area)
        if boxes:
            keep_idx.append(i)
            n_objects_total += len(boxes)
    n = len(dataset)
    print(f"[filtro oggetti] {name}: {len(keep_idx)}/{n} immagini con >=1 oggetto "
          f"({n_objects_total} oggetti totali, {n_objects_total/max(1,len(keep_idx)):.2f}/immagine) -- "
          f"{n - len(keep_idx)} immagini SENZA oggetto escluse dall'epoca")
    return Subset(dataset, keep_idx)


def subsample_dataset(dataset, frac, seed, name):
    if frac >= 1.0:
        return dataset
    n = len(dataset)
    n_keep = max(1, int(round(n * frac)))
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.permutation(n)[:n_keep]).tolist()
    print(f"[image-usage] {name}: {n_keep}/{n} campioni ({frac:.0%})")
    return Subset(dataset, idx)


# ─────────────────────────────────────────────────────────────────────────────
# Rete
# ─────────────────────────────────────────────────────────────────────────────

def build_nets(device, use_target_net):
    backbone = build_backbone().to(device)
    policy_net = TreeRLQNet(backbone).to(device)
    target_net = None
    if use_target_net:
        # Backbone condiviso (frozen, identico), solo la testa MLP viene
        # duplicata/aggiornata -- stesso schema usato in train.py per
        # Caicedo & Lazebnik.
        target_net = TreeRLQNet(backbone).to(device)
        target_net.q_net.load_state_dict(policy_net.q_net.state_dict())
        target_net.eval()
    return backbone, policy_net, target_net


def soft_update(target_net, policy_net, tau):
    with torch.no_grad():
        for tp, pp in zip(target_net.q_net.parameters(), policy_net.q_net.parameters()):
            tp.data.mul_(1.0 - tau).add_(pp.data, alpha=tau)


# ─────────────────────────────────────────────────────────────────────────────
# Valutazione: recall@IoU con ricerca ad albero, come Tabelle 1/2 del paper.
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_recall(dataset, q_net, device, num_levels, iou_thresholds, min_object_area,
                     large_area_frac=0.04, desc="Valutazione recall"):
    """Per ogni immagine del dataset genera l'albero di proposal (tree_search)
    e calcola, per ciascun livello cumulativo (1, 1+2, 1+2+4, ...) e per
    ciascuna soglia IoU, la frazione di oggetti ground-truth "coperti" da
    almeno una proposal fino a quel livello (== recall di Tabella 1/2).

    large_area_frac: un oggetto e' considerato "large" (vedi nota 1 del
    paper: >2000 pixel, definiti sull'immagine a risoluzione originale VOC)
    se la sua area supera questa frazione dell'area totale dell'immagine --
    ASSUNZIONE di traduzione alla risoluzione fissa di questo progetto
    (224x224), non un valore preso dal paper.
    """
    n_levels_cum = [2 ** L - 1 for L in range(1, num_levels + 1)]
    hits = {n: {t: 0 for t in iou_thresholds} for n in n_levels_cum}
    hits_large = {n: {t: 0 for t in iou_thresholds} for n in n_levels_cum}
    hits_small = {n: {t: 0 for t in iou_thresholds} for n in n_levels_cum}
    n_objects, n_large, n_small = 0, 0, 0

    q_net.eval()
    for i in tqdm(range(len(dataset)), desc=desc, unit="img", leave=False):
        sample = dataset[i]
        image = sample["image"].to(device)
        mask = sample["mask"].numpy().squeeze(0)
        gt_boxes = get_gt_boxes_from_mask(mask, min_area=min_object_area)
        if not gt_boxes:
            continue

        H, W = mask.shape
        img_area = H * W
        proposals = tree_search(q_net, image, num_levels, device)

        for gt in gt_boxes:
            n_objects += 1
            is_large = (gt[2] * gt[3]) > large_area_frac * img_area
            if is_large:
                n_large += 1
            else:
                n_small += 1

            best_iou_at = {}
            running_best = 0.0
            for k, prop in enumerate(proposals):
                running_best = max(running_best, compute_iou(prop, gt))
                if (k + 1) in n_levels_cum:
                    best_iou_at[k + 1] = running_best

            for n_cum, best in best_iou_at.items():
                for t in iou_thresholds:
                    if best >= t:
                        hits[n_cum][t] += 1
                        if is_large:
                            hits_large[n_cum][t] += 1
                        else:
                            hits_small[n_cum][t] += 1

    def _to_recall(hit_dict, denom):
        return {n: {t: (hit_dict[n][t] / denom if denom > 0 else 0.0) for t in iou_thresholds}
                for n in n_levels_cum}

    return {
        "all": _to_recall(hits, n_objects),
        "large": _to_recall(hits_large, n_large),
        "small": _to_recall(hits_small, n_small),
        "n_objects": n_objects, "n_large": n_large, "n_small": n_small,
    }


def print_recall_table(recall, iou_thresholds, tag=""):
    print(f"[recall {tag}]")
    for n_cum in sorted(recall["all"].keys()):
        row = "  ".join(f"IoU={t}: {recall['all'][n_cum][t]*100:5.1f}%" for t in iou_thresholds)
        print(f"  #proposals={n_cum:4d}  {row}")


def render_proposals(image_chw, proposals, num_levels_to_draw=3):
    """Visualizza le proposal (Fig. 6 del paper): livello 2 -> verde,
    livello 3 -> giallo, livello 4 -> rosso, ciclando i colori se
    num_levels_to_draw > 3."""
    img = np.clip(image_chw, 0.0, 1.0)
    img = (img.transpose(1, 2, 0) * 255.0).astype(np.uint8).copy()
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    colors = [(0, 255, 0), (0, 255, 255), (0, 0, 255), (255, 0, 255), (255, 255, 0)]

    idx = 0
    level_size = 1
    for level in range(num_levels_to_draw):
        color = colors[level % len(colors)]
        for _ in range(level_size):
            if idx >= len(proposals):
                break
            x, y, w, h = [int(round(v)) for v in proposals[idx]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
            idx += 1
        level_size *= 2
    return img


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-source", type=str, default=None, choices=["kaggle", "local", "synthetic"])
    p.add_argument("--dataset-path", type=str, default=None)
    p.add_argument("--kaggle-id", type=str, default=None)
    p.add_argument("--output-root", type=str, default="./treerl_logs")
    p.add_argument("--run-name", type=str, default=None)

    p.add_argument("--n-epochs", type=int, default=N_EPOCHS,
                    help=f"default {N_EPOCHS}, come nel paper ('trained for 25 epochs')")
    p.add_argument("--epsilon-decay-epochs", type=int, default=EPSILON_DECAY_EPOCHS,
                    help=f"epoche di decadimento lineare di epsilon (default {EPSILON_DECAY_EPOCHS}, "
                         f"'annealed linearly from 1 to 0.1 over the first 10 epochs')")
    p.add_argument("--gamma", type=float, default=GAMMA)
    p.add_argument("--max-steps", type=int, default=MAX_STEPS_PER_EPISODE)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--replay-size", type=int, default=REPLAY_SIZE,
                    help=f"default {REPLAY_SIZE} (valore letterale del paper: 'about 1 epoch of "
                         f"transitions'). ATTENZIONE: con feature a 4096-dim questo puo' richiedere "
                         f"decine di GB di RAM -- abbassalo se necessario (vedi stima stampata a "
                         f"inizio training).")
    p.add_argument("--min-object-area", type=int, default=20,
                    help="area minima (pixel) di una componente connessa della maschera per essere "
                         "considerata un oggetto valido")
    p.add_argument("--tree-levels", type=int, default=DEFAULT_TREE_LEVELS,
                    help=f"livelli dell'albero in valutazione/test (default {DEFAULT_TREE_LEVELS} -> "
                         f"31 proposal, come Tabella 2 del paper)")
    p.add_argument("--eval-every", type=int, default=1, help="valuta la recall sul val set ogni N epoche")
    p.add_argument("--iou-thresholds", type=float, nargs="+", default=[0.5, 0.6, 0.7])

    # Non specificati dal paper -- vedi ASSUNZIONI nel docstring del modulo.
    p.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--sgd-momentum", type=float, default=0.9)
    p.add_argument("--use-target-net", action="store_true",
                    help="DEVIAZIONE dal paper (Eq. 3 usa lo stesso theta_i per stima e bootstrap, "
                         "nessun target network separato): se instabile, usa questo flag per un target "
                         "network standard con soft update.")
    p.add_argument("--target-tau", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=0.0,
                    help="norma massima di gradient clipping, DEVIAZIONE dal paper (non menzionato). "
                         "0 = disabilitato (default, fedele al paper).")

    p.add_argument("--image-usage", type=float, default=1.0)
    p.add_argument("--test", action="store_true")
    p.add_argument("--model", type=str, default=None)
    return p


def build_run_name(args, prefix):
    if args.run_name:
        return args.run_name
    return f"{prefix}_{datetime.datetime.now():%Y%m%d_%H%M%S}"


def epsilon_for_epoch(epoch, decay_epochs, eps_start=EPSILON_START, eps_end=EPSILON_END):
    """"epsilon is annealed linearly from 1 to 0.1 over the first 10 epochs.
    Then epsilon is fixed to 0.1 in the last 15 epochs." (epoch e' 1-indexed)"""
    if epoch >= decay_epochs:
        return eps_end
    progress = (epoch - 1) / max(1, decay_epochs - 1)
    return eps_start - (eps_start - eps_end) * progress


def select_action(q_vals, epsilon):
    """Comportamento epsilon-greedy ESATTO del paper (Sezione 3.3):
    'the agent selects a random action from the whole action set with
    probability epsilon, and selects a random action from the two best
    actions in the two action groups [...] with probability 1-epsilon'.

    q_vals: tensore [13] di Q-value per lo stato corrente.
    """
    if random.random() < epsilon:
        return random.randrange(N_ACTIONS)
    scale_lo, scale_hi = SCALE_ACTIONS[0], SCALE_ACTIONS[-1] + 1
    translate_lo, translate_hi = TRANSLATE_ACTIONS[0], TRANSLATE_ACTIONS[-1] + 1
    best_scale = scale_lo + int(torch.argmax(q_vals[scale_lo:scale_hi]).item())
    best_translate = translate_lo + int(torch.argmax(q_vals[translate_lo:translate_hi]).item())
    return random.choice([best_scale, best_translate])


def train(args, device):
    cfg = build_dataset_config(args)
    train_ds, val_ds, _ = get_datasets(cfg)
    train_ds = subsample_dataset(train_ds, args.image_usage, cfg["seed"], "train")
    val_ds = subsample_dataset(val_ds, args.image_usage, cfg["seed"] + 1, "val")

    train_ds = filter_dataset_with_objects(train_ds, args.min_object_area, "train")
    val_ds = filter_dataset_with_objects(val_ds, args.min_object_area, "val")

    env = TreeRLEnv(train_ds, min_object_area=args.min_object_area)

    backbone, policy_net, target_net = build_nets(device, args.use_target_net)
    eval_net = target_net if target_net is not None else policy_net

    if args.optimizer == "sgd":
        optimizer = optim.SGD(policy_net.q_net.parameters(), lr=args.lr, momentum=args.sgd_momentum)
    else:
        optimizer = optim.Adam(policy_net.q_net.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()  # coerente con il gradiente dell'Eq. 3 (discesa sull'errore TD al quadrato)

    memory = ReplayBuffer(args.replay_size, backbone.out_dim, HISTORY_DIM)
    print(f"[memory] Replay buffer: capacity={args.replay_size} "
          f"~{memory.estimated_bytes() / (1024**3):.2f} GiB stimati")

    run_name = build_run_name(args, "treerl_train")
    tb_dir = os.path.join(cfg["output"]["root"], "tensorboard", run_name)
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)
    print(f"[tensorboard] logging in {tb_dir}")
    print(f"[config] optimizer={args.optimizer}(lr={args.lr}) use_target_net={args.use_target_net} "
          f"grad_clip={args.grad_clip} gamma={args.gamma} batch_size={args.batch_size} "
          f"n_epochs={args.n_epochs} epsilon_decay_epochs={args.epsilon_decay_epochs}")

    epoch_rng = np.random.default_rng(cfg["seed"])
    best_recall = -1.0
    best_ckpt_path = os.path.join(cfg["output"]["root"], "model_best.pth")
    os.makedirs(cfg["output"]["root"], exist_ok=True)

    n_grad_updates = 0
    for epoch in range(1, args.n_epochs + 1):
        epsilon = epsilon_for_epoch(epoch, args.epsilon_decay_epochs)
        # "Each epoch is ended after performing an episode in each training image."
        # -- permutazione SENZA rimpiazzo, un episodio per immagine per epoca.
        epoch_order = epoch_rng.permutation(len(train_ds))

        epoch_rewards, epoch_losses, epoch_hits = [], [], []
        epoch_start = time.time()

        progress_bar = tqdm(epoch_order, desc=f"Epoca {epoch}/{args.n_epochs}", unit="img")
        for img_idx in progress_bar:
            obs, info = env.reset(options={"idx": int(img_idx)})
            if info["n_objects"] == 0:
                continue  # rete di sicurezza, filter_dataset_with_objects dovrebbe gia' escluderle

            image_t = torch.from_numpy(env.current_image).to(device)
            feature_map, global_feat = policy_net.encode_episode(image_t)

            done = False
            episode_reward = 0.0
            while not done:
                box_t = torch.from_numpy(obs["window"]).unsqueeze(0)
                hist_t = torch.from_numpy(obs["history"]).unsqueeze(0).to(device)

                with torch.no_grad():
                    w_feat = policy_net.window_feature(feature_map, box_t, (env.H, env.W))
                    q_vals = policy_net(w_feat, global_feat, hist_t).squeeze(0)

                action = select_action(q_vals, epsilon)
                next_obs, reward, terminated, truncated, step_info = env.step(action)
                done = terminated or truncated

                next_box_t = torch.from_numpy(next_obs["window"]).unsqueeze(0)
                next_hist_t = torch.from_numpy(next_obs["history"]).unsqueeze(0).to(device)
                with torch.no_grad():
                    next_w_feat = policy_net.window_feature(feature_map, next_box_t, (env.H, env.W))

                memory.push(
                    w_feat.squeeze(0).cpu().numpy(), global_feat.squeeze(0).cpu().numpy(),
                    obs["history"], action, float(reward),
                    next_w_feat.squeeze(0).cpu().numpy(), next_obs["history"],
                )

                obs = next_obs
                episode_reward += reward
                if step_info.get("newly_hit"):
                    epoch_hits.append(1)

                if len(memory) >= args.batch_size:
                    w_b, g_b, h_b, a_b, r_b, nw_b, nh_b = memory.sample(args.batch_size)
                    w_b = torch.from_numpy(w_b).to(device)
                    g_b = torch.from_numpy(g_b).to(device)
                    h_b = torch.from_numpy(h_b).to(device)
                    a_b = torch.from_numpy(a_b).unsqueeze(1).to(device)
                    r_b = torch.from_numpy(r_b).to(device)
                    nw_b = torch.from_numpy(nw_b).to(device)
                    nh_b = torch.from_numpy(nh_b).to(device)

                    q_sa = policy_net(w_b, g_b, h_b).gather(1, a_b).squeeze(1)
                    with torch.no_grad():
                        # Eq. 3: max_a' Q(s',a'; theta_i) -- STESSA rete
                        # (policy_net) se --use-target-net non e' passato,
                        # letteralmente come scritto nel paper; altrimenti
                        # target_net (deviazione opzionale, vedi CLI).
                        next_q = eval_net(nw_b, g_b, nh_b).max(dim=1)[0]
                        target = r_b + args.gamma * next_q  # NESSUN (1-done): Eq. 3 non ne ha uno

                    loss = loss_fn(q_sa, target)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(policy_net.q_net.parameters(), args.grad_clip)
                    optimizer.step()
                    n_grad_updates += 1
                    if target_net is not None:
                        soft_update(target_net, policy_net, args.target_tau)

                    loss_val = loss.item()
                    epoch_losses.append(loss_val)
                    writer.add_scalar("train/loss_step", loss_val, n_grad_updates)

            epoch_rewards.append(episode_reward)
            progress_bar.set_postfix({
                "Rew": f"{np.mean(epoch_rewards):.2f}",
                "Hits/ep": f"{np.sum(epoch_hits)/max(1,len(epoch_rewards)):.2f}",
                "eps": f"{epsilon:.2f}",
            })

        writer.add_scalar("train/epsilon", epsilon, epoch)
        writer.add_scalar("train/reward_mean", float(np.mean(epoch_rewards)), epoch)
        writer.add_scalar("train/hits_per_episode", float(np.sum(epoch_hits) / max(1, len(epoch_rewards))), epoch)
        if epoch_losses:
            writer.add_scalar("train/loss_epoch_mean", float(np.mean(epoch_losses)), epoch)
        writer.add_scalar("replay/size", len(memory), epoch)
        print(f"[epoca {epoch}] {time.time()-epoch_start:.1f}s  reward_medio={np.mean(epoch_rewards):.2f}  "
              f"hit/episodio={np.sum(epoch_hits)/max(1,len(epoch_rewards)):.2f}  epsilon={epsilon:.2f}")

        ckpt_path = os.path.join(cfg["output"]["root"], f"model_epoch_{epoch}.pth")
        torch.save({"q_net_state_dict": policy_net.q_net.state_dict(), "epoch": epoch}, ckpt_path)

        if epoch % args.eval_every == 0 or epoch == args.n_epochs:
            recall = evaluate_recall(
                val_ds, eval_net, device, args.tree_levels, args.iou_thresholds,
                args.min_object_area, desc=f"Val epoca {epoch}",
            )
            max_n = max(recall["all"].keys())
            for t in args.iou_thresholds:
                writer.add_scalar(f"val/recall_IoU{t}_all", recall["all"][max_n][t], epoch)
                writer.add_scalar(f"val/recall_IoU{t}_large", recall["large"][max_n][t], epoch)
                writer.add_scalar(f"val/recall_IoU{t}_small", recall["small"][max_n][t], epoch)
            print_recall_table(recall, args.iou_thresholds, tag=f"val epoca {epoch}")

            main_recall = recall["all"][max_n][args.iou_thresholds[0]]
            writer.add_scalar("val/best_recall_so_far", max(best_recall, main_recall), epoch)
            if main_recall > best_recall:
                best_recall = main_recall
                torch.save(
                    {"q_net_state_dict": policy_net.q_net.state_dict(), "epoch": epoch, "recall": best_recall},
                    best_ckpt_path,
                )
                print(f"[val epoca {epoch}] Nuovo best checkpoint (recall@IoU{args.iou_thresholds[0]}="
                      f"{best_recall:.3f}) -> {best_ckpt_path}")

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    writer.close()
    print(f"[train] Completato. Miglior recall di validazione: {best_recall:.3f} -> {best_ckpt_path}")


def run_test(args, device):
    if not args.model:
        raise SystemExit("--test richiede anche --model=<percorso checkpoint>")
    cfg = build_dataset_config(args)
    _, _, test_ds = get_datasets(cfg)
    test_ds = subsample_dataset(test_ds, args.image_usage, cfg["seed"] + 2, "test")
    test_ds = filter_dataset_with_objects(test_ds, args.min_object_area, "test")

    backbone, policy_net, _ = build_nets(device, use_target_net=False)
    ckpt = torch.load(args.model, map_location=device)
    policy_net.q_net.load_state_dict(ckpt["q_net_state_dict"])
    policy_net.eval()
    print(f"[test] Checkpoint caricato da {args.model} (epoca {ckpt.get('epoch', '?')})")

    out_root = cfg["output"]["root"]
    viz_dir = os.path.join(out_root, "test_proposals")
    os.makedirs(viz_dir, exist_ok=True)

    recall = evaluate_recall(
        test_ds, policy_net, device, args.tree_levels, args.iou_thresholds,
        args.min_object_area, desc="Test",
    )
    print_recall_table(recall, args.iou_thresholds, tag="test")

    csv_path = os.path.join(out_root, "test_recall.csv")
    with open(csv_path, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["n_proposals", "iou_threshold", "recall_all", "recall_large", "recall_small"])
        for n_cum in sorted(recall["all"].keys()):
            for t in args.iou_thresholds:
                wcsv.writerow([n_cum, t, recall["all"][n_cum][t], recall["large"][n_cum][t], recall["small"][n_cum][t]])
    print(f"[test] Metriche in: {csv_path}")

    n_viz = min(20, len(test_ds))
    for i in range(n_viz):
        sample = test_ds[i]
        image = sample["image"].to(device)
        proposals = tree_search(policy_net, image, args.tree_levels, device)
        img = render_proposals(sample["image"].numpy(), proposals, num_levels_to_draw=min(3, args.tree_levels))
        cv2.imwrite(os.path.join(viz_dir, f"sample_{i:04d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"[test] Visualizzazioni proposal in: {viz_dir}")


if __name__ == "__main__":
    _args = build_arg_parser().parse_args()
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if _args.test:
        run_test(_args, _device)
    else:
        train(_args, _device)
