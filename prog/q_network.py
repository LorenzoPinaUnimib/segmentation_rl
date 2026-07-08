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
running stats delle BatchNorm). Non e' il 4096-dim di VGG16-fc7 usato nel
paper originale (quello resta un'opzione se preferita, vedi nota sotto), ma
e' comunque un vero extractor pre-addestrato, non piu' un proiettore casuale.

FIX (perdita di informazione SPAZIALE -- diagnosi "punto 1", train.py in
calo mentre val sale nonostante reward/rete/curriculum diversi): la versione
precedente usava l'avgpool NATIVO di ResNet18, che riduce la feature map
finale (512, 7, 7) a (512, 1, 1) -- un singolo vettore mediato su TUTTO lo
spazio del crop. Per un task di classificazione (per cui ResNet18 e' stato
pre-addestrato) questo e' voluto: l'obiettivo e' essere il piu' possibile
INVARIANTI a dove si trova l'oggetto nell'immagine. Ma qui l'obiettivo e'
l'esatto OPPOSTO: la Q-network deve capire se il tumore e' spostato a
sinistra o a destra DENTRO il crop attuale per scegliere l'azione giusta --
informazione che l'avgpool a (1,1) tende a schiacciare via.
FIX: si droppano SIA l'avgpool nativo SIA il fc finale (fino a 'layer4'),
e si applica un AdaptiveAvgPool2d PROPRIO a una griglia 3x3 (invece che
1x1): il risultato e' un embedding piu' grande (512*3*3 = 4608 invece di
512) che pero' mantiene un'informazione posizionale grossolana ("il segnale
forte e' nel terzo in alto a sinistra della griglia" eccetera). Resta
un'operazione COMPLETAMENTE frozen (nessun parametro nuovo, nessun
gradiente): fondamentale per non invalidare l'ottimizzazione a "embedding
cache" del replay buffer (che presuppone che l'embedding di uno stato non
cambi mai una volta calcolato -- se ci fosse anche un solo parametro
allenabile prima della cache, gli embedding gia' salvati diventerebbero
obsoleti man mano che quel parametro si aggiorna).

Il backbone va istanziato UNA volta sola (classe VisualBackbone) e condiviso
tra policy_net e target_net in train.py: essendo identico e mai aggiornato in
nessuno dei due, tenerne due copie sprecherebbe VRAM senza alcun beneficio (e
target_net.load_state_dict(...) non deve piu' ricopiare pesi congelati che
non cambiano mai).

OTTIMIZZAZIONE (vedi anche train.py, ReplayBuffer): dato che il backbone e'
congelato per sempre, il replay buffer NON salva piu' i pixel grezzi ma
l'embedding visivo gia' calcolato (4608 float invece di 224x224x3 uint8: pur
con la griglia 3x3, ancora ~10x meno RAM per transizione rispetto ai pixel
grezzi) e lo calcola una sola volta per step invece di ricalcolarlo ad ogni
sample durante l'update. Per questo forward() accetta sia la region grezza
[B,C,H,W] (la incapsula automaticamente tramite encode()) sia un embedding
gia' pronto [B,out_dim] (percorso usato in training/replay).

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
        # FIX (informazione spaziale): si droppano SIA l'avgpool nativo SIA
        # il fc finale (i.e. tutto tranne le ultime due voci di children()),
        # fermandosi all'output di 'layer4': (B, 512, 7, 7) per un input
        # 224x224. L'avgpool nativo a (1,1) verrebbe applicato DOPO e
        # distruggerebbe la posizione; qui si sostituisce con un pooling a
        # griglia 3x3 (vedi spatial_pool sotto), che mantiene un minimo di
        # struttura spaziale nell'embedding finale.
        self.body = nn.Sequential(*list(net.children())[:-2])
        # FIX (informazione spaziale): griglia 3x3 invece di collassare tutto
        # a un solo vettore (1x1) come faceva l'avgpool nativo di ResNet18 --
        # vedi spiegazione estesa nel docstring del modulo. Resta un pooling
        # FISSO (nessun parametro), quindi compatibile con l'embedding cache
        # del replay buffer (l'embedding di uno stato non cambia mai).
        self.spatial_pool = nn.AdaptiveAvgPool2d((3, 3))
        for param in self.body.parameters():
            param.requires_grad = False
        self.body.eval()

        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.out_dim = 512 * 3 * 3  # FIX: era 512 (embedding globale a 1x1)

        # OTTIMIZZAZIONE (performance, --paper-faithful su dataset intero): il
        # backbone e' SEMPRE in inference pura (congelato, @torch.no_grad, mai
        # un gradiente), e viene chiamato ad ogni singolo step ambiente: e' il
        # costo GPU dominante dell'intero training loop (conv stack + pooling
        # su un batch di 1 immagine, ripetuto decine di migliaia di volte per
        # epoca). Eseguirlo in autocast fp16 sfrutta i Tensor Core delle GPU
        # NVIDIA moderne per le convoluzioni, senza toccare in alcun modo i
        # pesi (restano fp32, il cast e' solo per il calcolo del forward) ne'
        # il training del q_net (che resta sempre fp32). Disattivabile con
        # --no-amp-backbone se si vuole la massima riproducibilita' numerica
        # bit-a-bit rispetto a un run precedente.
        self.use_amp = True

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
        # OTTIMIZZAZIONE (performance): autocast fp16 solo sul tratto
        # convoluzionale/pooling (il costo dominante), invariato su CPU o se
        # self.use_amp=False. L'output torna esplicitamente a fp32 cosi' il
        # replay buffer (embeds float32) e il resto della pipeline non notano
        # alcuna differenza di dtype.
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(x.is_cuda and self.use_amp)):
            feat = self.body(x)              # (B, 512, 7, 7)
            feat = self.spatial_pool(feat)    # (B, 512, 3, 3) -- FIX: preserva posizione grossolana
        return feat.flatten(1).float()        # (B, 4608)


class VGG16FC7Backbone(nn.Module):
    """Backbone FEDELE AL PAPER: VGG16 pre-addestrato su ImageNet, embedding
    a 4096-dim preso da **fc6** (l'output di classifier[0], DOPO la ReLU).

    CORREZIONE (era fc6+fc7 nella prima versione di questo file): il testo
    del paper dice esplicitamente *"We forward the region up to the layer 6
    (fc6) and use the 4,096 dimensional feature vector to represent its
    content"* -- un SOLO layer denso oltre il tronco convoluzionale, non due.
    Il nome della classe resta VGG16FC7Backbone per compatibilita' con il
    codice/checkpoint esistenti, ma il forward ora si ferma davvero a fc6.

    Perche' e' preferibile alla semplice media spaziale di VisualBackbone:
    fc6 e' un layer DENSO applicato al flatten dell'INTERA feature map
    spaziale (7x7x512 = 25088 valori in ingresso), con un peso DIVERSO per
    ciascuna delle 49 posizioni spaziali -- pesi pre-addestrati su ImageNet,
    non inizializzati a caso. A differenza di un average pooling (che tratta
    ogni posizione allo stesso modo per costruzione, quindi non puo' MAI
    codificare 'dove' si trova qualcosa), una combinazione lineare
    pre-addestrata con pesi posizione-specifici PUO' nativamente codificare
    sia le feature semantiche di alto livello sia l'informazione posizionale.
    Resta comunque un backbone COMPLETAMENTE frozen (nessun gradiente), quindi
    compatibile con l'embedding cache del replay buffer esattamente come
    VisualBackbone.
    """

    def __init__(self):
        super().__init__()
        try:
            from torchvision.models import vgg16, VGG16_Weights
        except ImportError as exc:
            raise ImportError(
                "VGG16FC7Backbone richiede torchvision (per i pesi ImageNet "
                "pre-addestrati). Installa con: pip install torchvision"
            ) from exc

        net = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.conv_features = net.features   # conv1..conv5+maxpool -> (B, 512, 7, 7) per input 224x224
        self.avgpool = net.avgpool          # AdaptiveAvgPool2d((7,7)) nativo di torchvision (no-op qui)
        # classifier: 0=fc6(25088->4096) 1=ReLU 2=Dropout 3=fc7(4096->4096)
        # 4=ReLU 5=Dropout 6=fc8(4096->1000). CORREZIONE: si usa SOLO fc6+ReLU
        # (layer 6 del paper), fc7/fc8 NON vengono piu' applicati.
        self.fc6 = net.classifier[0]
        self.relu6 = net.classifier[1]

        for module in (self.conv_features, self.fc6):
            for param in module.parameters():
                param.requires_grad = False
        self.eval()

        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.out_dim = 4096

        # OTTIMIZZAZIONE (performance): vedi commento identico in
        # VisualBackbone.__init__ -- qui e' ANCORA piu' rilevante, perche'
        # fc6 e' un singolo layer denso 25088->4096 (un matmul enorme per
        # step), il candidato ideale per i Tensor Core in fp16.
        self.use_amp = True

    def train(self, mode=True):
        super().train(False)  # ignora sempre 'mode': backbone congelato, mai in training mode
        return self

    @torch.no_grad()
    def forward(self, region):
        x = region
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] != 3:
            raise ValueError(f"VGG16FC7Backbone si aspetta 1 o 3 canali in input, ricevuti {x.shape[1]}")
        x = (x - self.imagenet_mean) / self.imagenet_std
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(x.is_cuda and self.use_amp)):
            feat = self.conv_features(x)   # (B, 512, 7, 7)
            feat = self.avgpool(feat)      # (B, 512, 7, 7), no-op per input 224x224
            feat = torch.flatten(feat, 1)  # (B, 25088)
            feat = self.relu6(self.fc6(feat))  # (B, 4096) -- fc6 pre-addestrato, CORREZIONE: fermo qui
        return feat.float()


def build_backbone(name="resnet18_spatial"):
    """Factory: sceglie l'implementazione del backbone visivo frozen.
    ENTRAMBE le opzioni sono backbone ImageNet PRE-ADDESTRATI e completamente
    congelati -- cambiare backbone non significa mai perdere il pre-training,
    solo scegliere quale "lettura" delle feature pretrained usare (vedi
    docstring delle due classi per il confronto dettagliato).
      - 'resnet18_spatial': ResNet18, griglia 3x3 mediata (4608-dim).
      - 'vgg16_fc7'       : VGG16-fc7, come nel paper originale (4096-dim,
                            posizione codificata nativamente dai pesi densi).
    """
    if name == "resnet18_spatial":
        return VisualBackbone()
    if name == "vgg16_fc7":
        return VGG16FC7Backbone()
    raise ValueError(f"Backbone sconosciuto: {name!r} (valori validi: 'resnet18_spatial', 'vgg16_fc7')")


class PaperQHead(nn.Module):
    """Testa Q-network FEDELE AL PAPER (Fig. 3, Caicedo & Lazebnik): un MLP
    semplice con DUE hidden layer da 1024 unita' ciascuno, ReLU + dropout,
    nessuna architettura dueling:

        (visual_embed + history) -> 1024 -> 1024 -> n_actions

    NUOVO (uniformita' tra backbone diversi): questa testa dipende solo da
    in_dim (backbone.out_dim + history_dim), quindi la stessa architettura
    identica (stessa capacita' di apprendimento) si applica automaticamente
    sia con VGG16-fc6 (4096-dim) sia con ResNet18 spatial (4608-dim) sia con
    qualunque altro backbone si aggiunga in futuro -- il confronto tra
    backbone diversi resta quindi "pulito": le differenze di risultato si
    possono attribuire al backbone, non a una testa diversa per ciascuno.

    NOTA: il paper non specifica il tasso di dropout esatto; si usa 0.3
    (stesso valore gia' in uso in questo codebase) come scelta ragionevole,
    non verificata contro il testo del paper.
    """

    def __init__(self, in_dim, n_actions, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(1024, n_actions),
        )

    def forward(self, state_repr):
        return self.net(state_repr)


class DuelingQHead(nn.Module):
    """Architettura Dueling DQN (Wang et al. 2016): separa la stima del
    valore dello stato V(s) da quella del vantaggio relativo di ogni azione
    A(s,a), invece di un'unica testa che stima Q(s,a) direttamente.

    OTTIMIZZAZIONE (perche' aiuta a convergere piu' in fretta e con meno
    guida): in questo dominio molte delle azioni hanno un impatto sul VALORE
    dello stato molto simile in tanti stati (es. quando il box e' gia' vicino
    al target, muoversi in una direzione piuttosto che un'altra conta poco
    rispetto al "quanto vale genericamente essere in questo stato"). Una
    singola testa Q deve re-imparare V(s) separatamente per OGNUNA delle 9
    azioni, diluendo i dati disponibili per azione; la testa dueling impara
    V(s) una volta sola (condivisa da tutte le azioni) e puo' concentrare la
    capacita' residua sulla stima del vantaggio relativo, che e' la parte
    davvero rilevante per la scelta dell'azione greedy -- tipicamente riduce
    in modo sensibile il numero di episodi necessari a stabilizzare la
    policy rispetto a una singola testa Q.
    """

    def __init__(self, in_dim, n_actions):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, n_actions)
        )

    def forward(self, state_repr):
        shared = self.shared(state_repr)
        value = self.value_stream(shared)
        advantage = self.advantage_stream(shared)
        # Sottrarre la media dell'advantage e' il trucco di identificabilita'
        # del paper originale: senza, Q=V+A ammette infinite coppie (V,A)
        # equivalenti (basta spostare una costante dall'una all'altra), il
        # che rende l'ottimizzazione instabile/non identificabile.
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


class ActiveLocalizationQNet(nn.Module):
    def __init__(self, backbone, in_channels=3, head_type="paper"):
        super().__init__()
        self.backbone = backbone  # condiviso tra policy_net/target_net, vedi train.py
        self.in_channels = in_channels

        # NUOVO (diagnosi dataset vs architettura): head_type sceglie tra la
        # testa fedele al paper (MLP 1024->1024, default -- vedi PaperQHead)
        # e la Dueling head introdotta in questa chat (--q-head in train.py).
        # Il nome dell'attributo (q_net) e la sua interfaccia
        # (parameters()/state_dict()) restano invariati in ENTRAMBI i casi:
        # train.py (ottimizzatore, soft-update del target, checkpoint)
        # continua a funzionare senza modifiche qualunque testa si scelga.
        in_dim = backbone.out_dim + (HISTORY_LENGTH * N_ACTIONS)
        if head_type == "paper":
            self.q_net = PaperQHead(in_dim, N_ACTIONS)
        elif head_type == "dueling":
            self.q_net = DuelingQHead(in_dim, N_ACTIONS)
        else:
            raise ValueError(f"head_type sconosciuto: {head_type!r} (valori validi: 'paper', 'dueling')")

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
