"""
train.py
────────
Loop custom DQN con "Guided Exploration" (Apprenticeship Learning).

FIX PRINCIPALE (fedelta' al paper): il backbone visivo (vedi q_network.py,
classe VisualBackbone) e' congelato ma ora e' un vero ResNet18 pre-addestrato
su ImageNet, non piu' un proiettore casuale mai addestrato -- vedi q_network.py
per i dettagli. policy_net e target_net CONDIVIDONO la stessa istanza di
backbone (stessi Tensor in VRAM, non due copie): e' identico e mai aggiornato
in nessuno dei due, quindi duplicarlo sprecherebbe memoria senza motivo, e il
target update / il salvataggio dei checkpoint ora toccano solo il q_net
(la parte davvero allenata), non piu' l'intero state_dict.

ALTRE CORREZIONI (vedi commenti "FIX" nel corpo del file):
  - Epoca fedele al paper: permutazione SENZA rimpiazzo del training set ad
    ogni epoca (prima si campionava CON rimpiazzo in env.reset()).
  - Schedule di epsilon corretto: 1.0 esatto alla prima epoca, lineare fino
    a 0.1 alla quinta (prima la prima epoca partiva da 0.82, non da 1.0).

FIX P1 (instabilita' del training DQN -- IoU/success rate che peggioravano
man mano che epsilon scendeva, vedi analisi in chat):
  - Adam (lr=3e-4) al posto di SGD(lr=1e-5): convergenza molto piu' rapida
    sulla testa MLP entro il numero di epoche disponibile.
  - Double DQN: l'azione migliore sul next-state e' scelta da policy_net e
    valutata da target_net, invece che scegliere e valutare entrambe con
    target_net.max(1) (overestimation bias del DQN "vanilla").
  - Huber loss (SmoothL1) al posto di MSE: meno sensibile a TD-error grandi.
  - Gradient clipping (norm max 10) su ogni update.
  - UPDATE_FREQ 4 -> 1: molti piu' gradienti utili per epoca.
  Insieme, questi fix mirano a far si' che la Q-network converga verso una
  policy sensata PRIMA che l'annealing di epsilon smetta di "coprire" i suoi
  errori con la guida di apprenticeship.

FIX -- ANALISI APPROFONDITA (dopo P1: train/iou_mean restava alto ma
val/iou_mean (policy greedy pura, epsilon=0) restava vicino a 0.02-0.09,
quasi casuale, con val/reward_mean sempre piu' negativo -- la Q-network non
stava imparando una policy USABILE, nonostante le metriche di training
sembrassero decenti). Cause strutturali individuate (non risolvibili con un
semplice cambio di iperparametri):
  1. get_positive_actions() e' quasi sempre non-vuota, quindi l'esplorazione
     epsilon-greedy era SEMPRE guidata dal "maestro" (apprenticeship) e non
     esponeva mai la Q-network a stati di errore/recupero. Risultato: la
     policy greedy, una volta senza guida (val, --test), si trovava su stati
     mai visti in training e collassava (extrapolation/distributional shift,
     un problema noto nell'imitation-augmented RL). FIX: --random-explore-frac
     (default 0.3) -- quando scatta epsilon, con questa probabilita' si
     sceglie un'azione DAVVERO casuale tra tutte e 9 invece che una garantita
     IoU-positiva, per dare alla rete esperienza reale di errore/recupero.
  2. Il replay buffer (20_000) ruotava piu' veloce di un'epoca (con episodi
     fino a 50-100 step e ~1500 immagini/epoca, un'epoca supera facilmente
     20_000 step), espellendo le transizioni buone raccolte con epsilon alto
     prima che la rete potesse consolidarle, e rendendo la distribuzione di
     training sempre piu' dominata da esperienza recente e correlata. FIX:
     REPLAY_SIZE portato a 200_000 (ancora poche centinaia di MB con
     l'embedding cache).
  3. L'hard-copy del target network ogni 1000 step causava un salto
     discontinuo nei target TD, coerente con la loss che ricominciava a
     salire dopo un minimo iniziale invece di stabilizzarsi. FIX: soft/Polyak
     update (soft_update(), --target-tau, default 0.005) ad ogni gradient
     step, per un bersaglio che si muove con continuita'.
  4. Nessun periodo di "warm-up": il primo update partiva appena il buffer
     superava BATCH_SIZE (32 transizioni), quindi sui dati piu' scarsi e meno
     rappresentativi possibili. FIX: --learning-starts (default 5000 step)
     prima del primo gradient update.
  Se il gap train/val resta ampio anche dopo questi fix, il prossimo livello
  da considerare (piu' invasivo, non ancora implementato) sono i ritorni
  n-step per accelerare il credit assignment sull'orizzonte lungo (fino a 100
  step) e/o il warm-start supervisionato del localizzatore (vedi P0 discusso
  in chat, localizer.py) per accorciare drasticamente lo spazio di ricerca.

NUOVO:
  - TensorBoard: tutte le metriche (loss, IoU, reward, step, epsilon, LR,
    success rate, sia in training che in validazione) vengono loggate in
    <output-root>/tensorboard, cosi' si vedono tutti i grafici con
    `tensorboard --logdir <output-root>/tensorboard`.
  - Ad ogni epoca, dopo il salvataggio del modello, si valuta la policy
    (greedy, epsilon=0) sull'INTERO validation set e si logga
    IoU/reward/step/success-rate medi sotto val/... in TensorBoard.
  - Best-checkpoint su validazione: oltre a model_epoch_N.pth (uno per
    epoca), si mantiene sempre model_best.pth = il checkpoint con la IoU di
    validazione piu' alta mai osservata durante il training, aggiornato ogni
    volta che la validazione migliora. L'ultima epoca non e' detto sia la
    migliore (specie con training lunghi/instabili), quindi model_best.pth
    e' il checkpoint da preferire per --test.
  - Schedule ADATTIVI rispetto a --n-epochs: sia il decadimento di epsilon
    (--epsilon-decay-frac, default 1/3 delle epoche totali, invece di un
    numero fisso di 5 epoche) sia il learning rate (cosine annealing con
    T_max=n_epochs, --lr-min-fraction) si dilatano automaticamente quando si
    aumenta --n-epochs, invece di restare tarati per un training di 15
    epoche specificamente. Utile perche' con un dominio ad alta variabilita'
    (MRI, tumori di forma/dimensione/contrasto molto diversi) 15 epoche
    possono non bastare alla Q-network per convergere prima che la guida di
    apprenticeship (epsilon alto) sparisca: ora si puo' allungare il
    training con --n-epochs 40/60/100 e gli schedule si adattano da soli,
    mantenendo le stesse proporzioni relative (es. 1/3 delle epoche in
    esplorazione guidata, poi lineare/costante) invece di dover ricalcolare
    a mano ogni soglia.
  - Modalita' --test --model=<checkpoint>: carica un checkpoint salvato
    (vedi formato in fondo al loop di training) e valuta sul TEST set,
    salvando per ogni immagine una GIF animata del percorso del box (verde =
    ground truth, rosso = box predetto) passo-passo, piu' un CSV riepilogativo
    e le metriche aggregate in TensorBoard.

DIFFERENZE NOTE (segnalate, non corrette qui perche' non sono bug ma
adattamenti legittimi al dominio -- vedi anche il docstring di
environment.py per il dettaglio):
  - Procedura di test del paper (sez. 4.3, IoR mark + riavvio della ricerca
    negli angoli) non implementata: ha senso per Pascal VOC multi-istanza,
    meno per questo dominio single-tumor-per-immagine.
  - Dominio: MRI single-class invece di Pascal VOC multi-classe/multi-oggetto
    -- stessa metodologia applicata a un problema diverso, non "lo stesso
    esperimento" del paper.

OTTIMIZZAZIONI MEMORIA (vedi commenti "MEMFIX"/"FIX embedding cache"):
  - Il vecchio ReplayBuffer teneva in RAM un dict Python + array numpy float32
    per OGNI transizione, sia per lo stato che per il next-state (praticamente
    duplicato). Con region 224x224x3 float32 (~0.57MB) e 50_000 transizioni,
    questo arrivava a decine di GB (da qui i 20GB/epoca osservati).
  - Poi il buffer e' passato ad array numpy pre-allocati con region
    quantizzate a uint8 (4x in meno rispetto a float32) e senza duplicare il
    next-state (stesso trucco usato da Stable-Baselines3 con
    optimize_memory_usage / dal Nature DQN).
  - ORA (dato che il backbone visivo e' congelato per sempre) il buffer non
    salva piu' i pixel grezzi ma l'EMBEDDING visivo gia' calcolato (512
    float32 = 2KB, contro i ~150KB di una region 224x224x3 uint8: circa 75x
    meno RAM per transizione). L'embedding viene calcolato una sola volta per
    step (in policy_net.encode()) e riusato sia per la scelta dell'azione sia
    per il push nel buffer, invece di essere ricalcolato ad ogni singolo
    sample durante l'update (prima: fino a 2*BATCH_SIZE forward del backbone
    ad ogni update, uno per lo stato corrente e uno per il next-state di ogni
    elemento del batch). Il flag "done" azzera comunque il bootstrap quando
    il next-state non e' valido (fine episodio), quindi la correttezza
    matematica e' preservata esattamente come prima.
  - Risultato: con capacita' 20_000 il buffer passa da qualche GB a poche
    decine di MB: si puo' aumentare tranquillamente --replay-size se serve
    piu' diversita' di esperienza, senza preoccupazioni di RAM.
"""
import argparse
import csv
import datetime
import os
import random
import time
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter

try:
    from PIL import Image
except ImportError as _exc:
    raise ImportError(
        "La generazione delle GIF di test richiede Pillow: pip install Pillow"
    ) from _exc
import cv2

from utils import compute_iou

from dataset import get_datasets
from environment import ActiveLocalizationEnv
from q_network import ActiveLocalizationQNet, VisualBackbone, VGG16FC7Backbone, build_backbone
from config import N_ACTIONS, TAU_IOU

# Hyperparametri DQN
BATCH_SIZE = 32
GAMMA = 0.90

# FIX P1 (instabilita' del training): con SGD(lr=1e-5) la testa MLP (512+90
# -> 512 -> 256 -> 9) impara troppo lentamente per tenere il passo con
# l'annealing di epsilon in 15 epoche -- da qui il collasso di IoU/success
# rate osservato non appena l'agente inizia a fidarsi della propria policy
# invece che della guida di apprenticeship. Adam converge molto piu' in
# fretta su reti di queste dimensioni; LR alzato di conseguenza.
LR = 3e-4

REPLAY_SIZE = 200_000  # FIX (analisi approfondita): con ~1500 immagini/epoca
                       # e episodi che arrivano a 50-100 step, un epoca intera
                       # supera abbondantemente i 20_000 usati prima -- il
                       # buffer ruotava piu' veloce di un'epoca, espellendo le
                       # transizioni buone raccolte quando epsilon era alto
                       # PRIMA che la rete potesse consolidarle. Con embedding
                       # da 512 float invece di pixel grezzi, 200_000
                       # transizioni pesano ancora poche centinaia di MB.

# FIX (analisi approfondita): l'hard-copy del target ogni 1000 step causava
# un salto discontinuo nei target TD (bootstrap che cambia di colpo invece
# che con continuita'), coerente con l'oscillazione vista in train/loss_step
# e con la loss che ricomincia a salire dopo un minimo iniziale. Sostituito
# con un soft update (Polyak averaging) ad ogni gradient step: il target si
# muove morbidamente verso il policy_net invece di saltare periodicamente.
TARGET_TAU = 0.005

# FIX P1: era 4 (un update ogni 4 step ambiente). Con cosi' pochi update per
# epoca la rete non fa in tempo a convergere prima che epsilon scenda:
# aggiornare ad ogni step da' molti piu' gradienti utili nello stesso
# numero di epoche.
UPDATE_FREQ = 1

# FIX P1: gradient clipping per evitare che TD-error occasionalmente grandi
# (es. dopo un TRIGGER_REWARD=3 su una transizione poco vista) destabilizzino
# l'update -- coerente con il loss/step molto rumoroso osservato in
# TensorBoard.
GRAD_CLIP_NORM = 10.0


class ReplayBuffer:
    """
    Buffer a capacita' fissa, interamente pre-allocato (nessun overhead di
    oggetti Python per transizione).

    FIX (bug del time-limit -- analisi approfondita): la versione precedente
    deduceva il "next state" dalla posizione (idx+1) nel buffer circolare, e
    azzerava il bootstrap (1-done) sia per il vero TERMINATED (trigger) sia
    per il semplice TRUNCATED (fine dei 100 step). Questo e' un bug noto
    dell'RL ("time-limit bug", vedi Pardo et al., "Time Limits in
    Reinforcement Learning"): un troncamento per limite di tempo NON e' una
    vera fine, lo stato a step 100 avrebbe comunque un valore futuro, che
    veniva sistematicamente azzerato -- un bias che cresce quanto piu' gli
    episodi si allungano (esattamente il pattern di IoU/reward che peggiora
    nel tempo osservato in training). Inoltre il trucco "idx+1" non era
    nemmeno corretto in quel caso: alla fine di un episodio (per qualunque
    motivo) il push successivo appartiene a un episodio DIVERSO (immagine
    diversa, box resettato), quindi (idx+1) non era comunque il vero
    next-state del transition troncato.

    FIX: ogni transizione salva ESPLICITAMENTE il proprio next_embed/
    next_history (calcolato una volta per step, vedi train()), invece di
    dedurlo dalla posizione adiacente nel buffer. Il flag salvato e'
    "terminal" (vero SOLO se l'episodio e' davvero terminato, cioe' azione
    TRIGGER), non "done" (che prima includeva anche il truncation): per le
    transizioni troncate si bootstrappa regolarmente sul vero next-state,
    perche' e' comunque un'osservazione valida, solo che l'episodio e' stato
    tagliato artificialmente.

    Costo: raddoppia lo spazio per embedding/history rispetto al trucco
    "adiacente", ma con embedding da 512 float (2KB) invece di pixel grezzi
    il costo assoluto resta trascurabile (~4.7KB/transizione, vedi
    estimated_bytes()).
    """

    def __init__(self, capacity, embed_dim, history_dim):
        self.capacity = capacity
        self.pos = 0
        self.size = 0
        self.embeds = np.zeros((capacity, embed_dim), dtype=np.float32)
        self.next_embeds = np.zeros((capacity, embed_dim), dtype=np.float32)
        self.histories = np.zeros((capacity, history_dim), dtype=np.float32)
        self.next_histories = np.zeros((capacity, history_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminals = np.zeros(capacity, dtype=np.float32)  # 1.0 solo se veramente TERMINATED (trigger)
        # OTTIMIZZAZIONE (n-step returns): ogni transizione salva anche il
        # proprio GAMMA EFFETTIVO (GAMMA**n, dove n e' il numero di step
        # ambiente effettivamente collassati in quella transizione). Serve
        # perche' vicino alla fine di un episodio la finestra n-step puo'
        # essere piu' corta di --n-step (n < N_STEP), quindi lo sconto da
        # applicare al bootstrap NON e' sempre lo stesso GAMMA**N_STEP fisso
        # -- va ricalcolato per-transizione. Con n-step=1 (default legacy)
        # gamma_n coincide sempre con GAMMA, comportamento invariato.
        self.gammas = np.zeros(capacity, dtype=np.float32)

    def estimated_bytes(self):
        return (
            self.embeds.nbytes + self.next_embeds.nbytes
            + self.histories.nbytes + self.next_histories.nbytes
            + self.actions.nbytes + self.rewards.nbytes + self.terminals.nbytes
            + self.gammas.nbytes
        )

    def push(self, embed, history, action, reward, next_embed, next_history, terminal, gamma_n):
        idx = self.pos
        self.embeds[idx] = embed
        self.next_embeds[idx] = next_embed
        self.histories[idx] = history
        self.next_histories[idx] = next_history
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.terminals[idx] = terminal
        self.gammas[idx] = gamma_n
        self.pos = (idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def __len__(self):
        return self.size

    def sample(self, batch_size):
        # FIX: ogni transizione e' ormai auto-contenuta (embed E next_embed
        # salvati esplicitamente), quindi non serve piu' nessuna gestione
        # speciale dei bordi del buffer circolare: qualunque indice in
        # [0, size) e' valido.
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.embeds[idx], self.histories[idx], self.actions[idx], self.rewards[idx],
            self.next_embeds[idx], self.next_histories[idx], self.terminals[idx], self.gammas[idx],
        )


def prepare_state(obs, device):
    reg = torch.from_numpy(obs["region"]).unsqueeze(0).to(device, non_blocking=True)
    hist = torch.from_numpy(obs["history"]).unsqueeze(0).to(device, non_blocking=True)
    return reg, hist


def diagnose_tumor_presence(dataset, name):
    """DIAGNOSTICA (persistente non-convergenza nonostante reward/rete/
    curriculum/action-space diversi): environment.py, quando la maschera di
    un campione e' completamente vuota (nessun pixel di tumore), assegna un
    gt_box FISSO e arbitrario (il quarto centrale dell'immagine):

        self.gt_box = np.array([self.W/4, self.H/4, self.W/2, self.H/2])

    Se una frazione non trascurabile del dataset non contiene tumore, questi
    campioni insegnano all'agente un target completamente slegato dal
    contenuto visivo reale -- un possibile canale di segnale spurio che
    nessun fix a livello di reward/architettura/curriculum puo' correggere,
    perche' e' un problema di ETICHETTE, non di algoritmo. Questa funzione
    conta quanti campioni del dataset dato hanno maschera vuota, PRIMA di
    lanciare il training, cosi' si puo' confermare o escludere l'ipotesi.
    """
    n = len(dataset)
    n_empty = 0
    for i in range(n):
        if dataset[i]["mask"].sum().item() <= 0:
            n_empty += 1
    frac = n_empty / max(1, n)
    print(f"[diagnostica] {name}: {n_empty}/{n} campioni ({frac:.1%}) SENZA tumore nella maschera "
          f"-> gt_box fallback 'quarto centrale' in environment.py per questi campioni")
    return frac


def subsample_dataset(dataset, frac, seed, name):
    """NUOVO (--image-usage): riduce un dataset a una frazione (0, 1] dei
    suoi campioni, scelti con un sottoinsieme casuale ma RIPRODUCIBILE (seed
    fisso, diverso per train/val/test cosi' i tre sottoinsiemi non si
    accavallano per costruzione). Utile per iterare velocemente in fase di
    debug (es. verificare che il modello riesca ad OVERFITTARE una manciata
    di immagini, vedi diagnosi in chat) senza aspettare un'epoca sull'intero
    dataset. frac=1.0 (default) lascia il dataset invariato -- nessun costo
    quando l'opzione non e' usata.
    """
    if frac >= 1.0:
        return dataset
    if not (0.0 < frac < 1.0):
        raise SystemExit(f"--image-usage deve essere in (0, 1], ricevuto {frac}")
    n = len(dataset)
    n_keep = max(1, int(round(n * frac)))
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.permutation(n)[:n_keep]).tolist()
    print(f"[image-usage] {name}: {n_keep}/{n} campioni ({frac:.0%})")
    return Subset(dataset, idx)


def _push_nstep(memory, nstep_buffer, next_visual_emb, next_obs, terminated):
    """OTTIMIZZAZIONE (n-step returns, vedi ReplayBuffer e --n-step):
    finalizza la transizione piu' vecchia nella finestra scorrevole
    'nstep_buffer' (lista di tuple (embed, history, action, reward), la piu'
    vecchia in posizione 0) e la spinge nel replay buffer.

    reward salvato = somma scontata delle ricompense della finestra (r_0 +
    GAMMA*r_1 + ... + GAMMA^(n-1)*r_(n-1)); next-state salvato = lo stato
    CORRENTE (next_visual_emb/next_obs), cioe' n step dopo l'ancora; gamma
    effettivo salvato = GAMMA**n, da usare al posto del GAMMA fisso nel
    bootstrap TD (serve perche' a fine episodio n puo' essere piu' corto di
    --n-step, vedi commento in ReplayBuffer).
    """
    n = len(nstep_buffer)
    discounted_reward = 0.0
    for k, (_, _, _, r) in enumerate(nstep_buffer):
        discounted_reward += (GAMMA ** k) * r
    embed0, hist0, action0, _ = nstep_buffer[0]
    memory.push(
        embed0, hist0, action0, discounted_reward,
        next_visual_emb.squeeze(0).cpu().numpy(), next_obs["history"],
        float(terminated), GAMMA ** n,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Rollout condiviso: usato sia per la valutazione ad ogni epoca sul val set,
# sia per --test sul test set (con record_frames=True per le GIF).
# ─────────────────────────────────────────────────────────────────────────────

def render_frame(image_chw, box, gt_box):
    """Disegna il box ground-truth (verde) e quello predetto (rosso) sopra
    l'immagine originale. image_chw e' un array [C,H,W] float in [0,1] (come
    prodotto dalla pipeline di preprocessing, normalization='per_image')."""
    img = np.clip(image_chw, 0.0, 1.0)
    img = (img.transpose(1, 2, 0) * 255.0).astype(np.uint8).copy()
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)

    gx, gy, gw, gh = [int(round(v)) for v in gt_box]
    cv2.rectangle(img, (gx, gy), (gx + gw, gy + gh), (0, 255, 0), 2)  # GT: verde

    px, py, pw, ph = [int(round(v)) for v in box]
    cv2.rectangle(img, (px, py), (px + pw, py + ph), (255, 0, 0), 2)  # predetto: rosso

    return img


def save_gif(frames, path, duration_ms=180):
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=duration_ms, loop=0)


@torch.no_grad()
def run_episode(env, policy_net, device, idx, epsilon=0.0, record_frames=False):
    """Esegue un episodio completo su dataset[idx] con policy greedy
    (eventualmente epsilon-random per un po' di rumore in valutazione).
    Nessun update, nessun push nel replay buffer: e' un rollout puro, usato
    per la valutazione (val ad ogni epoca) e per --test.

    Se record_frames=True, ritorna anche la lista di frame (uno per step,
    piu' quello iniziale) con GT (verde) e box predetto (rosso) disegnati,
    pronti per essere assemblati in una GIF con save_gif().
    """
    obs, _ = env.reset(options={"idx": int(idx)})
    done = False
    episode_reward = 0.0
    ep_steps = 0

    frames = None
    if record_frames:
        frames = [render_frame(env.current_image, env.box, env.gt_box)]

    policy_net.eval()
    while not done:
        reg, hist = prepare_state(obs, device)
        visual_emb = policy_net.encode(reg)

        if epsilon > 0.0 and random.random() < epsilon:
            action = random.randrange(N_ACTIONS)
        else:
            q_vals = policy_net(visual_emb, hist)
            action = q_vals.argmax(dim=1).item()

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        episode_reward += reward
        ep_steps += 1

        if record_frames:
            frames.append(render_frame(env.current_image, env.box, env.gt_box))

    final_iou = compute_iou(env.box, env.gt_box)
    result = {"reward": float(episode_reward), "iou": float(final_iou), "steps": ep_steps}
    if record_frames:
        result["frames"] = frames
    return result


def evaluate_dataset(env, policy_net, device, dataset, tag, writer, global_step, desc=None):
    """Valuta la policy greedy (epsilon=0) su TUTTO il dataset dato (env deve
    gia' avvolgere quel dataset) e logga le metriche aggregate in TensorBoard
    sotto '<tag>/...'. Usata sia per la validazione ad ogni epoca (tag='val')
    sia potenzialmente per altre valutazioni ad-hoc."""
    rewards, ious, steps_list = [], [], []
    iterator = tqdm(range(len(dataset)), desc=desc or f"Valutazione ({tag})", unit="img", leave=False)
    for i in iterator:
        res = run_episode(env, policy_net, device, i, epsilon=0.0, record_frames=False)
        rewards.append(res["reward"])
        ious.append(res["iou"])
        steps_list.append(res["steps"])

    metrics = {
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "iou_mean": float(np.mean(ious)),
        "iou_std": float(np.std(ious)),
        "steps_mean": float(np.mean(steps_list)),
        "success_rate": float(np.mean(np.array(ious) >= TAU_IOU)),
    }
    if writer is not None:
        for name, value in metrics.items():
            writer.add_scalar(f"{tag}/{name}", value, global_step)
    return metrics


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
            "train_ratio": (1501/2145), "val_ratio": (429/2145), "cache_pairs": False,
        },
        "preprocessing": {
            "normalization": "per_image", "binarize_mask": True, "mask_threshold": 0.5,
            "white_balance": True, "clahe": False, "denoise": False,
        },
        "training": {"batch_size": 512, "num_workers": 0},
        "output": {"root": args.output_root},
        "seed": 42,
        "backbone": args.backbone,
    }

def build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--total-timesteps", type=int, default=None)
    p.add_argument("--dataset-source", type=str, default=None, choices=["kaggle", "local", "synthetic"])
    p.add_argument("--dataset-path", type=str, default=None)
    p.add_argument("--kaggle-id", type=str, default=None)
    p.add_argument("--n-envs", type=int, default=6)
    p.add_argument("--n-epochs", type=int, default=15)
    p.add_argument("--n-iterations", type=int, default=110, help="quante volte ripetere model.learn(total_timesteps)")
    p.add_argument("--checkpoint-every", type=int, default=10_000, help="ogni quanti step salvare modello+vecnormalize")
    p.add_argument("--output-root", type=str, default="./ppo_brain_tumor_logs")

    # NUOVO: ogni esecuzione scrive in una sotto-cartella TensorBoard diversa
    # (<output-root>/tensorboard/<run-name>), cosi' run successive NON
    # sovrascrivono i grafici di quelle precedenti e restano tutte visibili
    # e confrontabili fianco a fianco in TensorBoard. Se non specificato, il
    # nome e' generato automaticamente da data/ora.
    p.add_argument("--run-name", type=str, default=None,
                    help="nome della run per la sotto-cartella TensorBoard (default: timestamp "
                         "automatico train_YYYYMMDD_HHMMSS / test_YYYYMMDD_HHMMSS)")

    # MEMFIX: dimensione del replay buffer ora configurabile da CLI, cosi'
    # non serve piu' modificare il codice per bilanciare la RAM disponibile.
    p.add_argument("--replay-size", type=int, default=REPLAY_SIZE,
                    help="capacita' del replay buffer DQN (buffer basato su embedding: "
                         "poche decine di byte/KB per transizione, vedi commenti in cima al file)")

    p.add_argument("--backbone", type=str, default="vgg16_fc7", choices=["resnet18_spatial", "vgg16_fc7"],
                    help="NUOVO: quale backbone visivo pre-addestrato/frozen usare (vedi q_network.py, "
                         "build_backbone). ENTRAMBI sono pre-addestrati ImageNet e completamente congelati: "
                         "'vgg16_fc7' (default) e' quello del paper originale (Caicedo & Lazebnik), embedding "
                         "a 4096-dim da fc7, un layer denso pre-addestrato che codifica nativamente anche "
                         "l'informazione posizionale (pesi diversi per posizione spaziale, non una media); "
                         "'resnet18_spatial' e' un'alternativa piu' leggera (ResNet18, griglia 3x3 mediata, "
                         "4608-dim) utile per confronto/velocita'. IMPORTANTE per --test: va sempre passato lo "
                         "stesso --backbone usato in fase di training per quel checkpoint, altrimenti il "
                         "caricamento dei pesi del q_net fallisce (dimensioni di input incompatibili).")

    p.add_argument("--image-usage", type=float, default=1.0,
                    help="NUOVO: frazione (0, 1] di immagini da usare da OGNI blocco del dataset "
                         "(train/val/test), scelte con un sottoinsieme casuale riproducibile. Utile "
                         "per iterare velocemente in debug (es. testare se il modello riesce a "
                         "OVERFITTARE una manciata di immagini prima di lanciare un training completo) "
                         "senza aspettare un'epoca sull'intero dataset. Es. --image-usage 0.5 usa meta' "
                         "del train set e meta' del val set (e del test set con --test). Default 1.0 = "
                         "usa tutto, nessun cambiamento rispetto a prima.")

    # NUOVO (schedule adattivi): prima epsilon decadeva SEMPRE nelle prime 5
    # epoche indipendentemente da --n-epochs, quindi con training piu' lunghi
    # (utili qui: il dominio ha molta variabilita' e 15 epoche sono poche per
    # imparare bene) la fase random/apprenticeship finiva troppo presto
    # rispetto alla durata totale. Ora il decadimento e' una FRAZIONE delle
    # epoche totali: con --epsilon-decay-frac 0.333 (default, = 5/15, la
    # stessa proporzione del paper) e --n-epochs 15 il comportamento e'
    # identico a prima; con --n-epochs 60 il decadimento si dilata sulle
    # prime 20 epoche invece che sulle prime 5, dando alla Q-network molto
    # piu' tempo guidato prima di dover reggersi da sola.
    p.add_argument("--epsilon-start", type=float, default=1.0,
                    help="epsilon alla prima epoca")
    p.add_argument("--epsilon-end", type=float, default=0.1,
                    help="epsilon minimo raggiunto a fine decadimento e mantenuto per le epoche restanti")
    p.add_argument("--epsilon-decay-frac", type=float, default=1/3,
                    help="frazione di --n-epochs su cui epsilon scende linearmente da "
                         "--epsilon-start a --epsilon-end (default 1/3, cioe' 5 epoche su 15 come nel paper; "
                         "si dilata automaticamente all'aumentare di --n-epochs)")

    # NUOVO (schedule adattivi): learning rate su cosine annealing spalmato
    # su TUTTE le epoche (T_max=n_epochs), cosi' anche la LR "si dilata" alla
    # stessa maniera di epsilon invece di essere una costante fissa pensata
    # per un training di 15 epoche specificamente.
    p.add_argument("--lr", type=float, default=None,
                    help="learning rate iniziale per Adam (default: costante LR nel modulo, 3e-4)")
    p.add_argument("--lr-min-fraction", type=float, default=0.05,
                    help="frazione della LR iniziale raggiunta all'ultima epoca (cosine annealing su --n-epochs)")

    # NUOVO (analisi approfondita -- vedi diagnosi in chat): tre fix
    # strutturali contro il grosso scarto train/val osservato (IoU alta in
    # training, quasi nulla in validazione greedy):
    p.add_argument("--learning-starts", type=int, default=5000,
                    help="step ambiente da accumulare nel replay buffer PRIMA del primo gradient "
                         "update (warm-up): evita di allenare sui primi dati, scarsi e poco "
                         "rappresentativi, appena il buffer supera BATCH_SIZE")
    p.add_argument("--random-explore-frac", type=float, default=0.3,
                    help="frazione MASSIMA (raggiunta a fine curriculum, vedi --random-explore-frac-start) "
                         "delle volte in cui, quando scatta epsilon, si sceglie un'azione DAVVERO casuale "
                         "(tra tutte e 9, anche quelle che peggiorano l'IoU) invece di una guidata da "
                         "get_positive_actions(): senza questo, l'esplorazione e' quasi sempre 'guidata dal "
                         "maestro' (positive_actions e' raramente vuota), quindi la Q-network non vede quasi "
                         "mai stati di errore/recupero e la policy greedy (val, epsilon=0) si trova spiazzata "
                         "su stati mai visti in training")
    p.add_argument("--random-explore-frac-start", type=float, default=0.05,
                    help="OTTIMIZZAZIONE (curriculum): frazione di esplorazione davvero casuale usata "
                         "alla PRIMA epoca. Sale linearmente fino a --random-explore-frac nell'arco delle "
                         "stesse epoche di decadimento di epsilon (--epsilon-decay-frac), poi resta "
                         "costante. Motivazione: nelle primissime epoche la Q-network non ha ancora nulla "
                         "da cui generalizzare, quindi conviene che l'esplorazione sia quasi tutta guidata "
                         "dal maestro (apprenticeship) per imparare in fretta una base sensata; man mano che "
                         "la policy migliora, aumentare gradualmente la quota di esplorazione indipendente "
                         "la espone sempre di piu' a stati di errore/recupero PRIMA che il training finisca, "
                         "invece di un'unica frazione fissa dall'inizio alla fine.")
    p.add_argument("--n-step", type=int, default=3,
                    help="OTTIMIZZAZIONE (n-step returns): invece di bootstrappare sul next-state a un solo "
                         "step di distanza, ogni transizione salvata nel replay buffer accumula la ricompensa "
                         "scontata di N step consecutivi e bootstrappa sullo stato N step piu' avanti. Con "
                         "episodi lunghi fino a 100 step, questo accelera enormemente il credit assignment "
                         "(l'informazione su 'questa sequenza di azioni porta al trigger giusto' si propaga "
                         "N volte piu' in fretta all'indietro nel tempo). --n-step 1 disattiva l'ottimizzazione "
                         "e ripristina il comportamento originale a 1 step.")
    p.add_argument("--target-tau", type=float, default=TARGET_TAU,
                    help="coefficiente di soft/Polyak update del target network ad ogni gradient step "
                         "(sostituisce l'hard-copy periodico, che causava salti discontinui nei target TD)")

    # NUOVO (convergenza al paper -- vedi analisi differenze in chat): il paper
    # (Caicedo & Lazebnik 2015) NON menziona reward continua/GIoU-shaped, Adam,
    # Double DQN, Huber loss, gradient clipping ne' soft/Polyak update: usa
    # reward binaria (Eq.2/3), SGD, DQN 'vanilla' (Mnih et al. 2015, Nature),
    # e (implicitamente) un hard-copy periodico del target come nel Nature DQN.
    # Questi flag permettono di spegnere selettivamente ciascuna di queste
    # tecniche 'moderne' per un confronto diretto A/B con la ricetta originale,
    # invece di dover editare il codice ogni volta.
    p.add_argument("--reward-mode", type=str, default="paper", choices=["paper", "shaped"],
                    help="'paper': reward binaria pura, sign(delta-IoU) in {-1,+1}, trigger +-eta fisso "
                         "(Eq. 2/3 esatte, DEFAULT). 'shaped': reward continua delta-GIoU + step-penalty, "
                         "deviazione deliberata dal paper (vedi environment.py). Passato a "
                         "ActiveLocalizationEnv per train/val/test env.")
    p.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"],
                    help="'sgd' e' quello usato nel paper (nessun Adam menzionato); 'adam' (default) "
                         "e' il FIX P1 introdotto per convergere piu' in fretta entro poche epoche.")
    p.add_argument("--sgd-momentum", type=float, default=0.9,
                    help="momentum per --optimizer sgd (il paper non specifica il valore esatto)")
    p.add_argument("--dqn-mode", type=str, default="double", choices=["double", "vanilla"],
                    help="'vanilla' = DQN originale (Mnih et al. 2015): target_net sceglie E valuta "
                         "l'azione migliore sul next-state (max diretto). 'double' (default, FIX P1) = "
                         "Double DQN: policy_net sceglie, target_net valuta (riduce overestimation bias, "
                         "MA non e' menzionato nel paper Caicedo & Lazebnik).")
    p.add_argument("--loss", type=str, default="huber", choices=["huber", "mse"],
                    help="'mse' e' la loss 'standard' del DQN vanilla non specificata altrimenti dal "
                         "paper; 'huber'/SmoothL1 (default, FIX P1) e' piu' robusta a TD-error grandi "
                         "ma e' un'aggiunta rispetto alla ricetta originale.")
    p.add_argument("--grad-clip", type=float, default=GRAD_CLIP_NORM,
                    help="norma massima per il gradient clipping (FIX P1, non nel paper). "
                         "<=0 disabilita completamente il clipping.")
    p.add_argument("--target-update-mode", type=str, default="soft", choices=["soft", "hard"],
                    help="'hard' = hard-copy periodico del target ogni --target-update-every gradient "
                         "step, come nel DQN vanilla/Nature (il paper non specifica soft-update). "
                         "'soft' (default) = Polyak averaging ad ogni step (vedi --target-tau, FIX "
                         "introdotto qui per l'instabilita' osservata, non nel paper).")
    p.add_argument("--target-update-every", type=int, default=1000,
                    help="gradient step tra un hard-copy e il successivo, usato solo con "
                         "--target-update-mode hard")
    p.add_argument("--paper-faithful", action="store_true",
                    help="NUOVO: scorciatoia che sovrascrive TUTTI i flag rilevanti con la ricetta "
                         "esatta del paper in un colpo solo (equivalente a passare a mano: "
                         "--optimizer sgd --dqn-mode vanilla --loss mse --grad-clip 0 "
                         "--target-update-mode hard --n-step 1 --random-explore-frac 0 "
                         "--random-explore-frac-start 0 --reward-mode paper --backbone vgg16_fc7). "
                         "Utile per un run di controllo 'massimamente fedele' con cui confrontare i run "
                         "con le tecniche moderne attivate, per isolare se il gap train/val e' dovuto al "
                         "dominio MRI o alla ricetta di training.")
    
    p.add_argument("--mix", action="store_true",
                    help="NUOVO: scorciatoia che sovrascrive TUTTI i flag rilevanti con la ricetta "
                         "esatta del paper in un colpo solo (equivalente a passare a mano: "
                         "--optimizer adam --dqn-mode double --loss smooth_l1 --grad-clip 10 "
                         "--target-update-mode hard --n-step 1 --random-explore-frac 0 "
                         "--random-explore-frac-start 0 --reward-mode paper --backbone vgg16_fc7). "
                         "Utile per un run di controllo 'massimamente fedele' con cui confrontare i run "
                         "con le tecniche moderne attivate, per isolare se il gap train/val e' dovuto al "
                         "dominio MRI o alla ricetta di training.")

    # NUOVO: modalita' di test -- carica un checkpoint e valuta sul test set,
    # salvando una GIF della traiettoria per ogni immagine.
    p.add_argument("--test", action="store_true",
                    help="modalita' di valutazione: carica --model e testa sul test set, "
                         "salvando GIF delle traiettorie + CSV + metriche in TensorBoard")
    p.add_argument("--model", type=str, default=None,
                    help="percorso del checkpoint da caricare per --test (formato salvato da "
                         "train(): dict con 'q_net_state_dict' e 'backbone_arch')")

    # modalità alternative: valutazione o osservazione dal vivo senza (ri)allenare
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--watch-idx", type=int, default=None, help="indice del test set da osservare dal vivo (richiede display)")
    p.add_argument("--model-path", type=str, default=None)
    p.add_argument("--vecnorm-path", type=str, default=None)

    # NUOVO: Simulazione interattiva manuale gestita dall'utente
    p.add_argument("--simulate", action="store_true", help="avvia un pannello interattivo per controllare manualmente il box e vedere i reward")

    # warm-start supervisionato (vedi localizer.py)
    p.add_argument("--train-localizer", action="store_true",
                    help="allena il regressore CNN supervisionato prima del training RL")
    p.add_argument("--use-localizer", action="store_true",
                    help="usa il localizzatore (da --localizer-path, o quello appena allenato) come warm-start per l'RL")
    p.add_argument("--localizer-path", type=str, default=None,
                    help="percorso del checkpoint del localizzatore (default: <output-root>/localizer.pt)")
    p.add_argument("--localizer-epochs", type=int, default=15)

    # FIX v7: action space continuo + SAC al posto delle 9 azioni discrete +
    # MaskablePPO. Vedi commento FIX v7 in environment.py per il perche'.
    p.add_argument("--continuous", action="store_true",
                    help="usa action space continuo [dx,dy,dw,dh] + algoritmo SAC invece delle 9 azioni discrete + MaskablePPO")
    p.add_argument("--sac-buffer-size", type=int, default=15_000,
                    help="dimensione replay buffer SAC (osservazioni immagine: tienilo basso, e' in RAM). "
                         "FIX: DictReplayBuffer (richiesto dalla Dict observation space image+box_vec) non "
                         "supporta optimize_memory_usage, quindi la RAM raddoppia rispetto a un buffer normale "
                         "(osservazioni + next_osservazioni duplicate) -- il default e' stato abbassato di "
                         "conseguenza da 50_000 a 15_000 (~5.6 GiB con osservazioni a 4 canali).")
    p.add_argument("--sac-learning-starts", type=int, default=5_000)
    return p


def build_run_name(args, prefix):
    """Nome della run per la sotto-cartella TensorBoard: se l'utente passa
    --run-name lo si usa cosi' com'e' (utile per confrontare run con nomi
    leggibili, es. 'adam_lr3e-4'); altrimenti si genera un timestamp, cosi'
    ogni esecuzione finisce in una cartella diversa e non sovrascrive i
    grafici di quelle precedenti."""
    if args.run_name:
        return args.run_name
    return f"{prefix}_{datetime.datetime.now():%Y%m%d_%H%M%S}"


def build_nets(cfg, device):
    """Backbone condiviso + policy/target net -- fattorizzato perche' serve
    identico sia in train() sia in run_test(). Il backbone da usare e'
    scelto da cfg['backbone'] (vedi --backbone / build_backbone in
    q_network.py) -- default 'vgg16_fc7', il piu' fedele al paper."""
    backbone = build_backbone(cfg.get("backbone", "vgg16_fc7")).to(device)
    policy_net = ActiveLocalizationQNet(backbone, in_channels=cfg["dataset"]["in_channels"]).to(device)
    target_net = ActiveLocalizationQNet(backbone, in_channels=cfg["dataset"]["in_channels"]).to(device)
    target_net.q_net.load_state_dict(policy_net.q_net.state_dict())
    target_net.eval()
    return backbone, policy_net, target_net


def soft_update(target_net, policy_net, tau):
    """Polyak averaging: target <- (1-tau)*target + tau*policy, applicato ad
    ogni gradient step invece di un hard-copy periodico. FIX (analisi
    approfondita): l'hard-copy ogni TARGET_UPDATE step creava un salto
    discontinuo nei target TD; qui il target si muove con continuita'.
    Il backbone NON va toccato: e' la STESSA istanza condivisa tra le due
    reti (vedi build_nets), quindi e' gia' identico by design."""
    with torch.no_grad():
        for target_param, policy_param in zip(target_net.q_net.parameters(), policy_net.q_net.parameters()):
            target_param.data.mul_(1.0 - tau).add_(policy_param.data, alpha=tau)


def load_checkpoint(policy_net, path, device, expected_backbone=None):
    ckpt = torch.load(path, map_location=device)
    saved_backbone = ckpt.get("backbone_arch", "sconosciuto")
    if expected_backbone is not None and saved_backbone not in (expected_backbone, "sconosciuto"):
        # IMPORTANTE: se i due non combaciano, out_dim del backbone istanziato
        # NON corrisponde alle dimensioni con cui il q_net e' stato allenato
        # -> load_state_dict qui sotto fallira' con un errore di shape poco
        # leggibile. Questo warning esplicito rende la causa immediata.
        print(f"[ATTENZIONE] Il checkpoint e' stato salvato con backbone='{saved_backbone}', "
              f"ma e' stato richiesto --backbone={expected_backbone}. Il caricamento dei pesi "
              f"del q_net probabilmente fallira' (dimensioni di input incompatibili). "
              f"Rilancia con --backbone={saved_backbone}.")
    policy_net.q_net.load_state_dict(ckpt["q_net_state_dict"])
    return saved_backbone


# ─────────────────────────────────────────────────────────────────────────────
# Modalita' --test: carica un checkpoint, valuta sul test set, salva le GIF.
# ─────────────────────────────────────────────────────────────────────────────

def run_test(args, device):
    if not args.model:
        raise SystemExit("--test richiede anche --model=<percorso checkpoint>")

    cfg = build_dataset_config(args)
    _, _, test_ds = get_datasets(cfg)
    test_ds = subsample_dataset(test_ds, args.image_usage, cfg["seed"] + 2, "test")
    env = ActiveLocalizationEnv(test_ds, reward_mode=args.reward_mode)

    _, policy_net, _ = build_nets(cfg, device)
    backbone_arch = load_checkpoint(policy_net, args.model, device, expected_backbone=cfg["backbone"])
    policy_net.eval()
    print(f"[test] Checkpoint caricato da {args.model} (backbone: {backbone_arch})")

    out_root = cfg["output"]["root"]
    gif_dir = os.path.join(out_root, "test_gifs")
    os.makedirs(gif_dir, exist_ok=True)

    # NUOVO: sotto-cartella TensorBoard dedicata a questa esecuzione di test
    # (vedi build_run_name), cosi' test ripetuti su modelli diversi non si
    # sovrascrivono a vicenda in TensorBoard.
    run_name = build_run_name(args, "test")
    tb_dir = os.path.join(out_root, "tensorboard", run_name)
    writer = SummaryWriter(log_dir=tb_dir)
    print(f"[tensorboard] logging in {tb_dir}")
    csv_path = os.path.join(out_root, "test_metrics.csv")

    rewards, ious, steps_list = [], [], []
    with open(csv_path, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["idx", "reward", "iou", "steps", "success", "gif_path"])

        for i in tqdm(range(len(test_ds)), desc="Test", unit="img"):
            res = run_episode(env, policy_net, device, i, epsilon=0.0, record_frames=True)
            success = res["iou"] >= TAU_IOU
            gif_name = f"sample_{i:04d}_iou{res['iou']:.2f}{'_OK' if success else ''}.gif"
            gif_path = os.path.join(gif_dir, gif_name)
            save_gif(res["frames"], gif_path)

            wcsv.writerow([i, res["reward"], res["iou"], res["steps"], int(success), gif_path])
            rewards.append(res["reward"])
            ious.append(res["iou"])
            steps_list.append(res["steps"])

            # gc esplicito dei frame appena salvati: su un test set grande le
            # liste di frame (uint8 HxWx3 per step) possono accumularsi se il
            # garbage collector tarda a intervenire.
            del res
        gc.collect()

    mean_reward = float(np.mean(rewards))
    mean_iou = float(np.mean(ious))
    success_rate = float(np.mean(np.array(ious) >= TAU_IOU))
    mean_steps = float(np.mean(steps_list))

    writer.add_scalar("test/reward_mean", mean_reward, 0)
    writer.add_scalar("test/iou_mean", mean_iou, 0)
    writer.add_scalar("test/iou_std", float(np.std(ious)), 0)
    writer.add_scalar("test/steps_mean", mean_steps, 0)
    writer.add_scalar("test/success_rate", success_rate, 0)
    writer.close()

    print(f"[test] IoU medio={mean_iou:.3f}  success_rate={success_rate:.3f}  "
          f"reward medio={mean_reward:.2f}  step medi={mean_steps:.1f}")
    print(f"[test] GIF salvate in: {gif_dir}")
    print(f"[test] Metriche per-campione in: {csv_path}")
    print(f"[test] Riepilogo in TensorBoard: {tb_dir}")


def train(args, device):
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # input a dimensione fissa -> conviene l'autotuner cudnn

    cfg = build_dataset_config(args)

    train_ds, val_ds, _ = get_datasets(cfg)

    # NUOVO (--image-usage): riduce train/val alla frazione richiesta PRIMA
    # di ogni altra cosa (diagnostica inclusa), cosi' tutto il resto del
    # training/logging riflette il dataset effettivamente usato.
    train_ds = subsample_dataset(train_ds, args.image_usage, cfg["seed"], "train")
    val_ds = subsample_dataset(val_ds, args.image_usage, cfg["seed"] + 1, "val")

    # DIAGNOSTICA (vedi diagnose_tumor_presence): esegue UNA volta prima del
    # training per confermare/escludere l'ipotesi che una parte del dataset
    # senza tumore stia iniettando target spuri (gt_box fallback fisso).
    diagnose_tumor_presence(train_ds, "train")
    diagnose_tumor_presence(val_ds, "val")

    env = ActiveLocalizationEnv(train_ds, reward_mode=args.reward_mode)
    val_env = ActiveLocalizationEnv(val_ds, reward_mode=args.reward_mode)  # env separato per la valutazione ad ogni epoca

    # FIX (backbone pre-addestrato condiviso): un solo VisualBackbone (ResNet18
    # ImageNet, congelato) istanziato una volta e condiviso da policy_net e
    # target_net -- vedi q_network.py per il perche' (era un proiettore
    # casuale mai addestrato, ora e' un vero extractor pre-addestrato; e
    # condividerlo evita di duplicare inutilmente pesi congelati in VRAM).
    backbone, policy_net, target_net = build_nets(cfg, device)

    # FIX P1: Adam invece di SGD -- converge molto piu' velocemente su una
    # testa MLP di queste dimensioni, fondamentale per imparare una policy
    # decente entro il numero di epoche disponibile prima che epsilon scenda.
    # Solo il q_net e' allenato: il backbone e' congelato (requires_grad=False).
    lr = args.lr if args.lr is not None else LR
    if args.optimizer == "sgd":
        # Ricetta del paper: SGD (nessun Adam menzionato in Caicedo & Lazebnik 2015).
        optimizer = optim.SGD(policy_net.q_net.parameters(), lr=lr, momentum=args.sgd_momentum)
    else:
        # FIX P1: Adam converge molto piu' in fretta sulla testa MLP entro il
        # numero di epoche disponibili -- deviazione deliberata dal paper.
        optimizer = optim.Adam(policy_net.q_net.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss() if args.loss == "huber" else nn.MSELoss()

    print(f"[config] optimizer={args.optimizer}(lr={lr}"
          f"{', momentum=' + str(args.sgd_momentum) if args.optimizer == 'sgd' else ''}) "
          f"loss={args.loss} dqn_mode={args.dqn_mode} target_update_mode={args.target_update_mode}"
          f"{'(' + str(args.target_update_every) + ' step)' if args.target_update_mode == 'hard' else '(tau=' + str(args.target_tau) + ')'} "
          f"grad_clip={args.grad_clip} reward_mode={args.reward_mode} n_step={args.n_step} "
          f"backbone={cfg['backbone']} random_explore_frac=[{args.random_explore_frac_start} -> "
          f"{args.random_explore_frac}]")

    # NUOVO (schedule adattivi): cosine annealing della LR spalmato su TUTTE
    # le epoche (T_max=EPOCHS), cosi' che aumentando --n-epochs la discesa
    # della LR si dilati automaticamente sull'intera durata del training
    # invece di restare una costante pensata per un numero fisso di epoche.
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.n_epochs), eta_min=lr * args.lr_min_fraction
    )

    first_obs, _ = env.reset()
    history_dim = first_obs["history"].shape[0]
    embed_dim = backbone.out_dim
    memory = ReplayBuffer(args.replay_size, embed_dim, history_dim)
    print(f"[memory] Replay buffer: capacity={args.replay_size} "
          f"~{memory.estimated_bytes() / (1024**3):.4f} GiB stimati")

    # NUOVO: TensorBoard -- tutte le metriche (training + validazione) vanno
    # qui. Ogni esecuzione scrive in una sotto-cartella DIVERSA (vedi
    # build_run_name), cosi' training successivi NON sovrascrivono i grafici
    # di quelli precedenti e restano tutti confrontabili fianco a fianco con:
    #   tensorboard --logdir <output-root>/tensorboard
    run_name = build_run_name(args, "train")
    tb_dir = os.path.join(cfg["output"]["root"], "tensorboard", run_name)
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)
    print(f"[tensorboard] logging in {tb_dir}")

    EPOCHS = args.n_epochs  # FIX: prima era hard-codato a 15 e ignorava --n-epochs
    steps_done = 0
    n_grad_updates = 0  # NUOVO: contatore di gradient step, usato da --target-update-mode hard

    # NUOVO: tracking del miglior checkpoint su validazione (vedi fondo del
    # loop epoche). L'ultima epoca NON e' detto sia la migliore, specie con
    # training piu' lunghi dove puo' esserci un picco intermedio prima di
    # un eventuale overfitting/instabilita' residua.
    best_val_iou = -1.0
    best_ckpt_path = os.path.join(cfg["output"]["root"], "model_best.pth")

    # FIX (epoca fedele al paper): generatore dedicato per le permutazioni di
    # ogni epoca, seedato per riproducibilita' ma indipendente da env.np_random
    # (che gestisce solo il fallback random di reset() quando non si passa idx).
    epoch_rng = np.random.default_rng(cfg["seed"])

    # NUOVO (schedule adattivi): il decadimento di epsilon e' calcolato come
    # FRAZIONE delle epoche totali (args.epsilon_decay_frac), non piu' come
    # un numero fisso di 5 epoche -- vedi help di --epsilon-decay-frac in
    # build_arg_parser(). Con EPOCHS=15 e frazione di default 1/3 il
    # comportamento e' identico al fix precedente (decadimento su 5 epoche);
    # con EPOCHS piu' alto il decadimento si dilata proporzionalmente.
    decay_epochs = max(1, round(EPOCHS * args.epsilon_decay_frac))
    print(f"[schedule] epsilon: {args.epsilon_start} -> {args.epsilon_end} "
          f"su {decay_epochs} epoche (poi costante), totale epoche={EPOCHS}")
    # OTTIMIZZAZIONE (curriculum su random_explore_frac): stessa finestra
    # temporale del decadimento di epsilon (decay_epochs), cosi' le due fasi
    # restano coerenti fra loro invece di avere due schedule scollegati.
    print(f"[schedule] random_explore_frac: {args.random_explore_frac_start} -> {args.random_explore_frac} "
          f"su {decay_epochs} epoche (poi costante)")
    print(f"[schedule] n_step={args.n_step}  learning_starts={args.learning_starts} step  "
          f"target_tau={args.target_tau}")
    N_STEP = max(1, args.n_step)

    for epoch in range(1, EPOCHS + 1):
        # FIX (schedule di epsilon fedele al paper, ora adattivo): epsilon
        # scende linearmente da --epsilon-start a --epsilon-end nelle prime
        # decay_epochs epoche, poi resta costante a --epsilon-end.
        if epoch <= decay_epochs:
            progress = ((epoch - 1) / (decay_epochs - 1)) if decay_epochs > 1 else 1.0
        else:
            progress = 1.0
        epsilon = args.epsilon_start - (args.epsilon_start - args.epsilon_end) * progress
        # OTTIMIZZAZIONE (curriculum su random_explore_frac): stesso
        # 'progress' di epsilon (0 alla prima epoca, 1 a fine decadimento e
        # oltre), cosi' la quota di esplorazione DAVVERO indipendente sale
        # gradualmente in parallelo al calare della guida di apprenticeship,
        # invece di essere una frazione fissa fin dalla prima epoca (quando
        # la Q-network non ha ancora nulla da cui generalizzare).
        random_explore_frac_epoch = (
            args.random_explore_frac_start
            + (args.random_explore_frac - args.random_explore_frac_start) * progress
        )

        # FIX (definizione di "epoca" fedele al paper): permutazione SENZA
        # rimpiazzo dell'intero training set, invece del campionamento CON
        # rimpiazzo che c'era prima dentro env.reset(). Ogni immagine viene
        # vista esattamente una volta per epoca.
        epoch_order = epoch_rng.permutation(len(train_ds))

        epoch_rewards = []
        epoch_ious = []
        epoch_episode_steps = []
        epoch_losses = []
        # DIAGNOSTICA (persistente non-convergenza): distingue episodi finiti
        # per trigger corretto, trigger sbagliato, o truncation (mai trovato
        # nulla in MAX_STEPS_PER_EPISODE). Se e' la componente 'truncated' a
        # crescere, l'agente non trova mai il target; se cresce
        # 'trigger_fail', il ramo greedy (1-epsilon) sta scegliendo TRIGGER
        # prematuramente su stati sbagliati.
        epoch_trigger_success = 0
        epoch_trigger_fail = 0
        epoch_truncated = 0

        # DIAGNOSTICA (rallentamento progressivo epoca-su-epoca): snapshot di
        # tempo e step totali all'inizio dell'epoca, per calcolare a fine
        # epoca sia la durata assoluta sia gli step-ambiente totali fatti in
        # questa epoca. Se la durata cresce PROPORZIONALMENTE agli step
        # totali (cioe' i secondi/step restano costanti), l'epoca e'
        # semplicemente piu' lunga perche' gli episodi durano di piu' (piu'
        # truncation/trigger tardivi, coerente con una policy che fatica a
        # convergere) -- non e' un bug di performance. Se invece anche i
        # secondi/step crescono, e' un vero rallentamento (frammentazione
        # GPU, leak, I/O, ecc.) da indagare separatamente.
        epoch_start_time = time.time()
        steps_done_at_epoch_start = steps_done

        progress_bar = tqdm(
            epoch_order,
            desc=f"Epoca {epoch}/{EPOCHS}",
            unit="img",
            leave=True
        )

        for img_idx in progress_bar:
            obs, _ = env.reset(options={"idx": int(img_idx)})
            done = False
            episode_reward = 0
            ep_steps = 0
            # OTTIMIZZAZIONE (n-step returns): finestra scorrevole per
            # questo episodio, resettata ad ogni nuovo episodio (le
            # transizioni n-step non attraversano mai il confine fra due
            # episodi/immagini diverse).
            nstep_buffer = []

            while not done:
                reg, hist = prepare_state(obs, device)

                # FIX (embedding cache): un solo forward del backbone per
                # step, riusato sia per la scelta dell'azione sia per il push
                # nel replay buffer (vedi ReplayBuffer piu' sopra).
                with torch.no_grad():
                    visual_emb = policy_net.encode(reg)

                # Epsilon-greedy con Apprenticeship Learning MISTA (FIX,
                # analisi approfondita): get_positive_actions() e' quasi
                # sempre non-vuota, quindi senza random_explore_frac
                # l'esplorazione sarebbe SEMPRE guidata dal maestro e la
                # Q-network non vedrebbe mai stati di errore/recupero --
                # esattamente il motivo per cui la policy greedy (val) crolla
                # su stati mai incontrati in training. Con probabilita'
                # random_explore_frac si sceglie invece un'azione DAVVERO
                # casuale tra tutte e 9 (anche quelle che peggiorano l'IoU).
                if random.random() < epsilon:
                    if random.random() < random_explore_frac_epoch:
                        action = random.randrange(N_ACTIONS)
                    else:
                        positive_actions = env.get_positive_actions()
                        action = random.choice(positive_actions) if positive_actions else random.randrange(N_ACTIONS)
                else:
                    policy_net.eval()  # FIX: disattiva il Dropout durante la selezione greedy dell'azione
                    with torch.inference_mode():
                        q_vals = policy_net(visual_emb, hist)
                        action = q_vals.argmax(dim=1).item()

                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # FIX: push() richiede esplicitamente next_embed/next_history
                # (calcolati sul next_obs) e "terminal" (vero SOLO se davvero
                # TERMINATED/trigger, non anche per il semplice truncation da
                # limite di step -- vedi docstring di ReplayBuffer sul
                # time-limit bug). Un solo forward aggiuntivo del backbone per
                # ottenere l'embedding del next-state.
                next_reg, next_hist = prepare_state(next_obs, device)
                with torch.no_grad():
                    next_visual_emb = policy_net.encode(next_reg)

                # OTTIMIZZAZIONE (n-step returns, vedi _push_nstep/--n-step
                # in cima al file): accumula (embed, history, action, reward)
                # dello step corrente in una finestra scorrevole lunga
                # N_STEP. Appena la finestra e' piena, la transizione PIU'
                # VECCHIA puo' essere finalizzata: bootstrap sullo stato
                # CORRENTE (N_STEP passi dopo la propria ancora), con reward
                # = somma scontata delle N ricompense intermedie -- invece
                # del bootstrap a 1 solo step di prima. Con --n-step 1 questo
                # blocco si comporta esattamente come il push singolo
                # originale (finestra sempre piena a lunghezza 1).
                nstep_buffer.append((
                    visual_emb.squeeze(0).cpu().numpy(),
                    obs["history"],
                    action,
                    float(reward),
                ))
                if len(nstep_buffer) >= N_STEP:
                    _push_nstep(memory, nstep_buffer, next_visual_emb, next_obs, terminated)
                    nstep_buffer.pop(0)

                obs = next_obs
                steps_done += 1

                episode_reward += reward
                ep_steps += 1

                if len(memory) > BATCH_SIZE and steps_done >= args.learning_starts and (steps_done % UPDATE_FREQ == 0):
                    policy_net.train()  # riattiva il Dropout per l'update (il backbone resta comunque in eval, vedi VisualBackbone.train())

                    embeds, histories, actions, rewards, n_embeds, n_histories, dones, gammas = memory.sample(BATCH_SIZE)

                    emb_b = torch.from_numpy(embeds).to(device, non_blocking=True)
                    hist_b = torch.from_numpy(histories).to(device, non_blocking=True)
                    act_b = torch.from_numpy(actions).unsqueeze(1).to(device, non_blocking=True)
                    rew_b = torch.from_numpy(rewards).to(device, non_blocking=True)
                    n_emb_b = torch.from_numpy(n_embeds).to(device, non_blocking=True)
                    n_hist_b = torch.from_numpy(n_histories).to(device, non_blocking=True)
                    done_b = torch.from_numpy(dones).to(device, non_blocking=True)
                    # OTTIMIZZAZIONE (n-step returns): sconto EFFETTIVO per
                    # transizione (GAMMA**n, n<=N_STEP -- vedi ReplayBuffer),
                    # non un GAMMA**N_STEP fisso: le transizioni vicino alla
                    # fine di un episodio hanno finestre piu' corte.
                    gamma_b = torch.from_numpy(gammas).to(device, non_blocking=True)

                    state_action_values = policy_net(emb_b, hist_b).gather(1, act_b).squeeze(1)

                    # FIX P1 (Double DQN): il DQN "vanilla" usa
                    # target_net(next).max(1) sia per SCEGLIERE che per
                    # VALUTARE l'azione migliore sul next-state, il che tende
                    # a sovrastimare sistematicamente i Q-value (max di stime
                    # rumorose e' un stimatore distorto verso l'alto) -- una
                    # causa nota di instabilita' e di policy che non
                    # convergono. Qui la scelta dell'azione migliore usa
                    # policy_net (rete online), la sua VALUTAZIONE usa
                    # target_net: le due reti raramente sono d'accordo su un
                    # sovrastima, quindi il bias si riduce.
                    if args.dqn_mode == "double":
                        # FIX P1 (non nel paper): Double DQN -- policy_net
                        # sceglie l'azione migliore sul next-state, target_net
                        # la valuta, per ridurre l'overestimation bias del
                        # DQN vanilla (max di stime rumorose e' distorto verso
                        # l'alto).
                        with torch.inference_mode():
                            policy_net.eval()  # scelta dell'azione non deve essere rumorosa (niente dropout)
                            next_actions = policy_net(n_emb_b, n_hist_b).argmax(dim=1, keepdim=True)
                            policy_net.train()
                            next_state_values = target_net(n_emb_b, n_hist_b).gather(1, next_actions).squeeze(1)
                    else:
                        # DQN 'vanilla' (Mnih et al. 2015, Nature): target_net
                        # sceglie E valuta l'azione migliore sul next-state
                        # con un unico max -- e' l'algoritmo con cui il paper
                        # (Caicedo & Lazebnik) ha effettivamente dimostrato
                        # convergenza, nessuna menzione di Double DQN.
                        with torch.inference_mode():
                            next_state_values = target_net(n_emb_b, n_hist_b).max(dim=1)[0]
                    expected_state_action_values = rew_b + (gamma_b * next_state_values * (1 - done_b))

                    # --loss: 'huber' (FIX P1, default) e' piu' robusta a
                    # TD-error grandi ma non e' nel paper; 'mse' e' la scelta
                    # 'standard' non specificata altrimenti dal paper.
                    loss = loss_fn(state_action_values, expected_state_action_values)

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    # --grad-clip: FIX P1 (non nel paper), <=0 la disabilita.
                    if args.grad_clip is not None and args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(policy_net.q_net.parameters(), args.grad_clip)
                    optimizer.step()
                    n_grad_updates += 1

                    if args.target_update_mode == "soft":
                        # FIX (analisi approfondita, non nel paper): soft/
                        # Polyak update del target ad OGNI gradient step, al
                        # posto dell'hard-copy periodico (vedi soft_update()
                        # e TARGET_TAU/--target-tau in cima al file).
                        soft_update(target_net, policy_net, args.target_tau)
                    else:
                        # Hard-copy periodico (DQN vanilla/Nature, coerente
                        # col paper: nessuna menzione di soft-update): il
                        # target resta fisso per --target-update-every
                        # gradient step, poi viene sovrascritto in blocco.
                        if n_grad_updates % args.target_update_every == 0:
                            target_net.q_net.load_state_dict(policy_net.q_net.state_dict())

                    loss_val = loss.item()
                    epoch_losses.append(loss_val)
                    writer.add_scalar("train/loss_step", loss_val, steps_done)

            # OTTIMIZZAZIONE (n-step returns): a fine episodio possono
            # restare nella finestra fino a N_STEP-1 transizioni non ancora
            # finalizzate (l'episodio e' finito prima che la finestra si
            # riempisse di nuovo). Sono comunque valide: si finalizzano ORA,
            # tutte con lo stesso vero next-state finale (next_visual_emb/
            # next_obs) e lo stesso terminal flag, semplicemente con una
            # finestra piu' corta (n < N_STEP) -- per questo _push_nstep usa
            # sempre GAMMA**len(nstep_buffer) e non un GAMMA**N_STEP fisso.
            while nstep_buffer:
                _push_nstep(memory, nstep_buffer, next_visual_emb, next_obs, terminated)
                nstep_buffer.pop(0)

            final_iou = compute_iou(env.box, env.gt_box)
            epoch_rewards.append(float(episode_reward))
            epoch_ious.append(float(final_iou))
            epoch_episode_steps.append(ep_steps)
            if terminated:
                if final_iou >= TAU_IOU:
                    epoch_trigger_success += 1
                else:
                    epoch_trigger_fail += 1
            else:
                epoch_truncated += 1

            progress_bar.set_postfix({
                "IoU": f"{np.mean(epoch_ious):.3f}",
                "Rew": f"{np.mean(epoch_rewards):.2f}",
                "Steps": f"{np.mean(epoch_episode_steps):.1f}",
                "eps": f"{epsilon:.2f}"
            })

        # ── TensorBoard: metriche di training aggregate per l'epoca ────────
        writer.add_scalar("train/epsilon", epsilon, epoch)
        writer.add_scalar("train/random_explore_frac", random_explore_frac_epoch, epoch)
        writer.add_scalar("train/lr", lr_scheduler.get_last_lr()[0], epoch)
        writer.add_scalar("train/reward_mean", float(np.mean(epoch_rewards)), epoch)
        writer.add_scalar("train/reward_std", float(np.std(epoch_rewards)), epoch)
        writer.add_scalar("train/iou_mean", float(np.mean(epoch_ious)), epoch)
        writer.add_scalar("train/iou_std", float(np.std(epoch_ious)), epoch)
        writer.add_scalar("train/steps_mean", float(np.mean(epoch_episode_steps)), epoch)
        writer.add_scalar("train/success_rate", float(np.mean(np.array(epoch_ious) >= TAU_IOU)), epoch)
        if epoch_losses:
            writer.add_scalar("train/loss_epoch_mean", float(np.mean(epoch_losses)), epoch)
        writer.add_scalar("replay/size", len(memory), epoch)
        # DIAGNOSTICA: composizione delle terminazioni episodio nell'epoca.
        n_eps = max(1, len(epoch_ious))
        writer.add_scalar("train/term_trigger_success_rate", epoch_trigger_success / n_eps, epoch)
        writer.add_scalar("train/term_trigger_fail_rate", epoch_trigger_fail / n_eps, epoch)
        writer.add_scalar("train/term_truncated_rate", epoch_truncated / n_eps, epoch)

        # DIAGNOSTICA (rallentamento progressivo): durata dell'epoca, step
        # ambiente totali fatti in questa epoca, e secondi/step normalizzato.
        epoch_duration_sec = time.time() - epoch_start_time
        epoch_total_steps = steps_done - steps_done_at_epoch_start
        writer.add_scalar("perf/epoch_seconds", epoch_duration_sec, epoch)
        writer.add_scalar("perf/epoch_total_env_steps", epoch_total_steps, epoch)
        writer.add_scalar("perf/seconds_per_env_step", epoch_duration_sec / max(1, epoch_total_steps), epoch)
        print(f"[perf epoca {epoch}] {epoch_duration_sec:.1f}s totali, "
              f"{epoch_total_steps} step ambiente, "
              f"{1000 * epoch_duration_sec / max(1, epoch_total_steps):.2f} ms/step")

        # NUOVO (schedule adattivi): un passo di cosine annealing per epoca,
        # spalmato su EPOCHS totali -- si dilata automaticamente insieme a
        # --n-epochs esattamente come lo schedule di epsilon qui sopra.
        lr_scheduler.step()

        os.makedirs(cfg["output"]["root"], exist_ok=True)
        # FIX: si salva solo il q_net (la parte allenata), non piu' l'intero
        # state_dict di policy_net -- il backbone e' un ResNet18 ImageNet
        # sempre identico e ricostruibile da VisualBackbone(), non ha senso
        # riscrivere ~45MB di pesi congelati identici ad ogni singola epoca.
        # NOTA formato checkpoint: per ricaricare, ricreare VisualBackbone() +
        # ActiveLocalizationQNet(backbone) e fare load_state_dict solo su
        # policy_net.q_net con il contenuto di "q_net_state_dict" (vedi anche
        # load_checkpoint() / --test).
        ckpt_path = f"{cfg['output']['root']}/model_epoch_{epoch}.pth"
        torch.save(
            {"q_net_state_dict": policy_net.q_net.state_dict(), "backbone_arch": cfg["backbone"],
             "epoch": epoch},
            ckpt_path,
        )

        # ── Validazione ad ogni epoca sull'INTERO validation set ────────────
        # Policy greedy (epsilon=0, nessuna guida di apprenticeship): misura
        # cio' che la policy fa davvero da sola, non quanto viene aiutata.
        val_metrics = evaluate_dataset(
            val_env, policy_net, device, val_ds, tag="val", writer=writer,
            global_step=epoch, desc=f"Val epoca {epoch}/{EPOCHS}",
        )
        print(f"[val epoca {epoch}] IoU={val_metrics['iou_mean']:.3f} "
              f"success_rate={val_metrics['success_rate']:.3f} "
              f"reward={val_metrics['reward_mean']:.2f} "
              f"steps={val_metrics['steps_mean']:.1f}")

        # NUOVO: salvataggio best-checkpoint su validazione. L'ultima epoca
        # non e' necessariamente la migliore (vedi anche il collasso di IoU
        # osservato quando epsilon scende troppo in fretta rispetto alla
        # velocita' di convergenza della Q-network): qui si tiene sempre a
        # disposizione il checkpoint con la IoU di validazione piu' alta mai
        # vista, utilizzabile direttamente con --test --model=model_best.pth.
        writer.add_scalar("val/best_iou_so_far", max(best_val_iou, val_metrics["iou_mean"]), epoch)
        if val_metrics["iou_mean"] > best_val_iou:
            best_val_iou = val_metrics["iou_mean"]
            torch.save(
                {"q_net_state_dict": policy_net.q_net.state_dict(), "backbone_arch": cfg["backbone"],
                 "epoch": epoch, "val_iou": best_val_iou},
                best_ckpt_path,
            )
            print(f"[val epoca {epoch}] Nuovo best checkpoint (IoU={best_val_iou:.3f}) -> {best_ckpt_path}")

        # gc.collect() a fine epoca (non ad ogni step: sarebbe troppo
        # costoso) libera eventuali cicli di riferimento residui (es. grafici
        # autograd non piu' referenziati) prima della prossima epoca.
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    writer.close()
    print(f"[train] Training completato. Miglior IoU di validazione: {best_val_iou:.3f} "
          f"-> checkpoint: {best_ckpt_path}")


if __name__ == "__main__":
    _args = build_arg_parser().parse_args()

    if _args.paper_faithful:
        # NUOVO: un solo flag per convergere alla ricetta ESATTA del paper
        # (Caicedo & Lazebnik 2015), sovrascrivendo qualunque altro valore
        # passato per queste opzioni. Serve come run di controllo per isolare
        # se il gap train/val osservato e' un problema del dominio MRI o
        # della ricetta di training modificata.
        print("[paper-faithful] Sovrascrivo gli argomenti con la ricetta esatta del paper: "
              "SGD, DQN vanilla (no Double), MSE, nessun gradient clipping, hard target update "
              "periodico, n-step=1, nessuna esplorazione davvero casuale (solo guidata + fallback "
              "casuale se positive_actions e' vuoto, come nel paper), reward binaria pura, "
              "backbone VGG16-fc6.")
        _args.optimizer = "sgd"
        _args.dqn_mode = "vanilla"
        _args.loss = "mse"
        _args.grad_clip = 0.0
        _args.target_update_mode = "hard"
        _args.n_step = 1
        _args.random_explore_frac = 0.0
        _args.random_explore_frac_start = 0.0
        _args.reward_mode = "paper"
        _args.backbone = "vgg16_fc7"

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if _args.test:
        run_test(_args, _device)
    else:
        train(_args, _device)
