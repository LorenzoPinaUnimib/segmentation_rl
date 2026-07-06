"""
q_network.py
────────────
FIX (fedelta' al paper): la versione precedente aveva una CNN "congelata"
ma inizializzata a pesi CASUALI e mai pre-addestrata su nulla -- quindi non
era un extractor con conoscenza visiva pregressa (alla Zeiler-Fergus/ImageNet,
come nel paper), ma un semplice proiettore casuale fisso (512-dim). Questo
eliminava il vantaggio centrale che il paper attribuisce all'uso di un CNN
pre-addestrato: rappresentazioni visive gia' generiche/utili fin dal primo
step, senza doverle re-imparare da zero con un segnale di reward scarso e
ritardato.

Ora il backbone visivo e' un vero ResNet18 pre-addestrato su ImageNet,
congelato (nessun parametro aggiornato, sempre in eval() per non toccare le
running stats delle BatchNorm), che produce un embedding a 512 dimensioni
(l'output nativo del suo avgpool). Non e' il 4096-dim di VGG16-fc7 usato nel
paper originale (quello resta un'opzione se preferita, vedi nota sotto), ma
e' comunque un vero extractor pre-addestrato, non piu' un proiettore casuale.

Il backbone va istanziato UNA volta sola (classe VisualBackbone) e condiviso
tra policy_net e target_net in train.py: essendo identico e mai aggiornato in
nessuno dei due, tenerne due copie sprecherebbe VRAM senza alcun beneficio (e
target_net.load_state_dict(...) non deve piu' ricopiare pesi congelati che
non cambiano mai).

OTTIMIZZAZIONE (vedi anche train.py, ReplayBuffer): dato che il backbone e'
congelato per sempre, il replay buffer NON salva piu' i pixel grezzi ma
l'embedding visivo gia' calcolato (512 float invece di 224x224x3 uint8: circa
75x meno RAM per transizione) e lo calcola una sola volta per step invece di
ricalcolarlo ad ogni sample durante l'update. Per questo forward() accetta sia
la region grezza [B,C,H,W] (la incapsula automaticamente tramite encode())
sia un embedding gia' pronto [B,512] (percorso usato in training/replay).

Nota (se si vuole restare piu' fedeli al 4096-dim di VGG16-fc7 del paper):
si puo' sostituire resnet18 con torchvision.models.vgg16(weights=VGG16_Weights
.IMAGENET1K_V1).features + AdaptiveAvgPool2d((7,7)) + i due fc pre-addestrati
(classifier[0], classifier[3], 4096-dim ciascuno), congelati; l'interfaccia di
VisualBackbone (out_dim, encode/forward) resta identica, cambia solo out_dim.
"""
import torch
import torch.nn as nn

from config import N_ACTIONS, HISTORY_LENGTH


class VisualBackbone(nn.Module):
    """Estrattore visivo ResNet18 pre-addestrato su ImageNet, congelato.

    Sempre in eval() (vedi override di train()): e' un estrattore fisso, non
    deve mai aggiornare le running stats delle BatchNorm ne' uscire da eval
    mode, indipendentemente da chi lo possiede (policy_net o target_net) e da
    quando questi vengono messi in .train().
    """

    def __init__(self):
        super().__init__()
        try:
            from torchvision.models import resnet18, ResNet18_Weights
        except ImportError as exc:
            raise ImportError(
                "VisualBackbone richiede torchvision (per i pesi ImageNet "
                "pre-addestrati). Installa con: pip install torchvision"
            ) from exc

        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Tutto tranne l'ultimo fc di classificazione: conv1 ... avgpool ->
        # output (B, 512, 1, 1). 512 e' comodo perche' coincide con il "512"
        # gia' usato altrove nel progetto (q_net, history, ecc).
        self.body = nn.Sequential(*list(net.children())[:-1])
        for param in self.body.parameters():
            param.requires_grad = False
        self.body.eval()

        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.out_dim = 512

    def train(self, mode=True):
        super().train(False)  # ignora sempre 'mode': backbone congelato, mai in training mode
        return self

    @torch.no_grad()
    def forward(self, region):
        x = region
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] != 3:
            raise ValueError(f"VisualBackbone si aspetta 1 o 3 canali in input, ricevuti {x.shape[1]}")
        x = (x - self.imagenet_mean) / self.imagenet_std
        feat = self.body(x)
        return feat.flatten(1)  # (B, 512)


class ActiveLocalizationQNet(nn.Module):
    def __init__(self, backbone, in_channels=3):
        super().__init__()
        self.backbone = backbone  # condiviso tra policy_net/target_net, vedi train.py
        self.in_channels = in_channels

        # Q-Network (512 visual + 90 history) -> 9 Q-Values finali.
        # Questa e' l'UNICA parte allenata: il backbone resta congelato.
        self.q_net = nn.Sequential(
            nn.Linear(backbone.out_dim + (HISTORY_LENGTH * N_ACTIONS), 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, N_ACTIONS)
        )

    def encode(self, region):
        """Calcola l'embedding visivo (512-dim) dalla region grezza [B,C,H,W].
        Frozen: nessun gradiente. Da chiamare una sola volta per stato: in
        train.py viene chiamato una volta per step e il risultato viene
        riusato sia per la scelta dell'azione sia per il push nel replay
        buffer, invece di essere ricalcolato ad ogni sample del replay."""
        return self.backbone(region)

    def forward(self, visual_input, history):
        # Dual-mode: accetta sia region grezza (dim==4, [B,C,H,W]) sia
        # embedding gia' calcolato (dim==2, [B,512]) -- quest'ultimo e' il
        # percorso usato durante gli update di training, per non ricalcolare
        # il backbone congelato ad ogni sample del replay buffer.
        visual_emb = self.encode(visual_input) if visual_input.dim() == 4 else visual_input
        state_repr = torch.cat([visual_emb, history], dim=1)
        q_values = self.q_net(state_repr)
        return q_values
