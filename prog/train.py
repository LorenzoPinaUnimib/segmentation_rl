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
import os
import random
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from utils import compute_iou

from dataset import get_datasets
from environment import ActiveLocalizationEnv
from q_network import ActiveLocalizationQNet, VisualBackbone
from config import N_ACTIONS

# Hyperparametri DQN
BATCH_SIZE = 32
GAMMA = 0.90
LR = 1e-5
REPLAY_SIZE = 20_000   # con il buffer basato su embedding questo e' ormai
                       # pochissimi MB (vedi ReplayBuffer.estimated_bytes()):
                       # si puo' alzare molto senza preoccupazioni di RAM.
TARGET_UPDATE = 1000
UPDATE_FREQ = 4  # aggiorna la rete ogni 4 step totali, non ad ogni singolo step


class ReplayBuffer:
    """
    Buffer a capacita' fissa, interamente pre-allocato (nessun overhead di
    oggetti Python per transizione).

    FIX (embedding cache): dato che il backbone visivo e' congelato per
    sempre, non ha senso salvare i pixel grezzi della region e ricalcolare il
    backbone ad ogni sample: si salva direttamente l'embedding visivo
    (512-dim float32) gia' calcolato al momento della raccolta. Questo taglia
    la RAM per transizione di circa 75x (2KB contro ~150KB di una region
    224x224x3 uint8) ed elimina la ricomputazione ridondante del backbone
    durante il training (prima: fino a 2*BATCH_SIZE forward del backbone per
    ogni update, uno per stato e uno per next-state di ogni elemento del
    batch; ora: zero, perche' policy_net/target_net ricevono direttamente
    l'embedding nel percorso "dim==2" del forward -- vedi q_network.py).

    Il next-state della transizione salvata in posizione idx e' l'embedding
    salvato in posizione (idx+1) % capacity: dato che push() viene chiamata
    una volta per step, in ordine, questo e' esattamente lo stato osservato
    subito dopo l'azione. Quando un episodio finisce, il "next" apparente
    (posizione idx+1, che verra' sovrascritta da un episodio futuro) e'
    comunque ignorato nel calcolo del target perche' moltiplicato per
    (1 - done) = 0.
    """

    def __init__(self, capacity, embed_dim, history_dim):
        self.capacity = capacity
        self.pos = 0
        self.size = 0
        self.embeds = np.zeros((capacity, embed_dim), dtype=np.float32)
        self.histories = np.zeros((capacity, history_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def estimated_bytes(self):
        return (
            self.embeds.nbytes + self.histories.nbytes + self.actions.nbytes
            + self.rewards.nbytes + self.dones.nbytes
        )

    def push(self, embed, history, action, reward, done):
        idx = self.pos
        self.embeds[idx] = embed
        self.histories[idx] = history
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = done
        self.pos = (idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def __len__(self):
        return self.size

    def sample(self, batch_size):
        if self.size < self.capacity:
            # buffer non ancora pieno: gli indici validi come "stato corrente"
            # sono [0, size-2], perche' lo stato in size-1 non ha ancora un
            # "next" scritto in memoria.
            idx = np.random.randint(0, self.size - 1, size=batch_size)
        else:
            # buffer pieno e circolare: l'unico indice da evitare come
            # "stato corrente" e' pos-1 (il piu' recente), perche' la sua
            # posizione next (pos) e' la piu' vecchia, sul punto di essere
            # sovrascritta / gia' logicamente "futura" rispetto ad esso.
            offset = np.random.randint(0, self.capacity - 1, size=batch_size)
            idx = (self.pos + offset) % self.capacity

        next_idx = (idx + 1) % self.capacity

        embeds = self.embeds[idx]
        next_embeds = self.embeds[next_idx]
        histories = self.histories[idx]
        next_histories = self.histories[next_idx]
        actions = self.actions[idx]
        rewards = self.rewards[idx]
        dones = self.dones[idx]
        return embeds, histories, actions, rewards, next_embeds, next_histories, dones


def prepare_state(obs, device):
    reg = torch.from_numpy(obs["region"]).unsqueeze(0).to(device, non_blocking=True)
    hist = torch.from_numpy(obs["history"]).unsqueeze(0).to(device, non_blocking=True)
    return reg, hist


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
            "white_balance": True, "clahe": True, "denoise": False,
        },
        "training": {"batch_size": 512, "num_workers": 0},
        "output": {"root": args.output_root},
        "seed": 42,
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

    # MEMFIX: dimensione del replay buffer ora configurabile da CLI, cosi'
    # non serve piu' modificare il codice per bilanciare la RAM disponibile.
    p.add_argument("--replay-size", type=int, default=REPLAY_SIZE,
                    help="capacita' del replay buffer DQN (buffer basato su embedding: "
                         "poche decine di byte/KB per transizione, vedi commenti in cima al file)")

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


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # input a dimensione fissa -> conviene l'autotuner cudnn

    args = build_arg_parser().parse_args()
    cfg = build_dataset_config(args)

    train_ds, val_ds, _ = get_datasets(cfg)

    env = ActiveLocalizationEnv(train_ds)

    # FIX (backbone pre-addestrato condiviso): un solo VisualBackbone (ResNet18
    # ImageNet, congelato) istanziato una volta e condiviso da policy_net e
    # target_net -- vedi q_network.py per il perche' (era un proiettore
    # casuale mai addestrato, ora e' un vero extractor pre-addestrato; e
    # condividerlo evita di duplicare inutilmente pesi congelati in VRAM).
    backbone = VisualBackbone().to(device)
    policy_net = ActiveLocalizationQNet(backbone, in_channels=cfg["dataset"]["in_channels"]).to(device)
    target_net = ActiveLocalizationQNet(backbone, in_channels=cfg["dataset"]["in_channels"]).to(device)
    target_net.q_net.load_state_dict(policy_net.q_net.state_dict())
    target_net.eval()

    # Solo il q_net e' allenato: il backbone e' congelato (requires_grad=False).
    optimizer = optim.SGD(policy_net.q_net.parameters(), lr=LR, momentum=0.9)

    first_obs, _ = env.reset()
    history_dim = first_obs["history"].shape[0]
    embed_dim = backbone.out_dim
    memory = ReplayBuffer(args.replay_size, embed_dim, history_dim)
    print(f"[memory] Replay buffer: capacity={args.replay_size} "
          f"~{memory.estimated_bytes() / (1024**3):.4f} GiB stimati")

    EPOCHS = args.n_epochs  # FIX: prima era hard-codato a 15 e ignorava --n-epochs
    steps_done = 0

    # FIX (epoca fedele al paper): generatore dedicato per le permutazioni di
    # ogni epoca, seedato per riproducibilita' ma indipendente da env.np_random
    # (che gestisce solo il fallback random di reset() quando non si passa idx).
    epoch_rng = np.random.default_rng(cfg["seed"])

    for epoch in range(1, EPOCHS + 1):
        # FIX (schedule di epsilon): il paper vuole epsilon=1.0 ESATTO alla
        # prima epoca, poi lineare fino a 0.1 alla quinta. La vecchia formula
        # 1.0 - 0.9*(epoch/5.0) dava 0.82 alla prima epoca (raggiungeva 0.1
        # solo alla quinta, ma con una curva shiftata). Con (epoch-1)/4.0:
        # epoch=1 -> 1.0, epoch=5 -> 0.1, lineare nel mezzo.
        if epoch <= 5:
            epsilon = 1.0 - (0.9 * ((epoch - 1) / 4.0))
        else:
            epsilon = 0.1

        # FIX (definizione di "epoca" fedele al paper): permutazione SENZA
        # rimpiazzo dell'intero training set, invece del campionamento CON
        # rimpiazzo che c'era prima dentro env.reset(). Ogni immagine viene
        # vista esattamente una volta per epoca.
        epoch_order = epoch_rng.permutation(len(train_ds))

        epoch_rewards = []
        epoch_ious = []
        epoch_episode_steps = []

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

            while not done:
                reg, hist = prepare_state(obs, device)

                # FIX (embedding cache): un solo forward del backbone per
                # step, riusato sia per la scelta dell'azione sia per il push
                # nel replay buffer (vedi ReplayBuffer piu' sopra).
                with torch.no_grad():
                    visual_emb = policy_net.encode(reg)

                # Epsilon-greedy con Apprenticeship Learning
                if random.random() < epsilon:
                    positive_actions = env.get_positive_actions()
                    action = random.choice(positive_actions) if positive_actions else random.randrange(N_ACTIONS)
                else:
                    policy_net.eval()  # FIX: disattiva il Dropout durante la selezione greedy dell'azione
                    with torch.inference_mode():
                        q_vals = policy_net(visual_emb, hist)
                        action = q_vals.argmax(dim=1).item()

                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                memory.push(visual_emb.squeeze(0).cpu().numpy(), obs["history"], action, float(reward), done)
                obs = next_obs
                steps_done += 1

                episode_reward += reward
                ep_steps += 1

                if len(memory) > BATCH_SIZE and (steps_done % UPDATE_FREQ == 0):
                    policy_net.train()  # riattiva il Dropout per l'update (il backbone resta comunque in eval, vedi VisualBackbone.train())

                    embeds, histories, actions, rewards, n_embeds, n_histories, dones = memory.sample(BATCH_SIZE)

                    emb_b = torch.from_numpy(embeds).to(device, non_blocking=True)
                    hist_b = torch.from_numpy(histories).to(device, non_blocking=True)
                    act_b = torch.from_numpy(actions).unsqueeze(1).to(device, non_blocking=True)
                    rew_b = torch.from_numpy(rewards).to(device, non_blocking=True)
                    n_emb_b = torch.from_numpy(n_embeds).to(device, non_blocking=True)
                    n_hist_b = torch.from_numpy(n_histories).to(device, non_blocking=True)
                    done_b = torch.from_numpy(dones).to(device, non_blocking=True)

                    state_action_values = policy_net(emb_b, hist_b).gather(1, act_b).squeeze(1)
                    with torch.inference_mode():
                        next_state_values = target_net(n_emb_b, n_hist_b).max(1)[0]
                    expected_state_action_values = rew_b + (GAMMA * next_state_values * (1 - done_b))

                    loss = nn.MSELoss()(state_action_values, expected_state_action_values)

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

                if steps_done % TARGET_UPDATE == 0:
                    # Solo il q_net va sincronizzato: il backbone e' la STESSA
                    # istanza condivisa tra policy_net e target_net, quindi e'
                    # gia' identico e non va mai ricopiato.
                    target_net.q_net.load_state_dict(policy_net.q_net.state_dict())

            final_iou = compute_iou(env.box, env.gt_box)
            epoch_rewards.append(float(episode_reward))
            epoch_ious.append(float(final_iou))
            epoch_episode_steps.append(ep_steps)

            progress_bar.set_postfix({
                "IoU": f"{np.mean(epoch_ious):.3f}",
                "Rew": f"{np.mean(epoch_rewards):.2f}",
                "Steps": f"{np.mean(epoch_episode_steps):.1f}",
                "eps": f"{epsilon:.2f}"
            })

        os.makedirs(cfg["output"]["root"], exist_ok=True)
        # FIX: si salva solo il q_net (la parte allenata), non piu' l'intero
        # state_dict di policy_net -- il backbone e' un ResNet18 ImageNet
        # sempre identico e ricostruibile da VisualBackbone(), non ha senso
        # riscrivere ~45MB di pesi congelati identici ad ogni singola epoca.
        # NOTA formato checkpoint: per ricaricare, ricreare VisualBackbone() +
        # ActiveLocalizationQNet(backbone) e fare load_state_dict solo su
        # policy_net.q_net con il contenuto di "q_net_state_dict".
        torch.save(
            {"q_net_state_dict": policy_net.q_net.state_dict(), "backbone_arch": "resnet18_imagenet1k"},
            f"{cfg['output']['root']}/model_epoch_{epoch}.pth",
        )

        # gc.collect() a fine epoca (non ad ogni step: sarebbe troppo
        # costoso) libera eventuali cicli di riferimento residui (es. grafici
        # autograd non piu' referenziati) prima della prossima epoca.
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    train()
