"""
q_network_treerl.py
────────────────────
Backbone visivo e Q-network fedeli a Jie et al. 2016 (arXiv:1703.02710),
Sezione 3.1 ("States") e Fig. 4.

Citazione esatta (Sezione 3.1): "The features of both the current window and
the whole image are extracted using a VGG-16 layer CNN model pre-trained on
ImageNet. We use the feature vector of layer fc6 in our problem. To
accelerate the feature extraction, all the feature vectors are computed on
top of pre-computed feature maps of the layer conv5_3 after using ROI
Pooling operation to obtain a fixed-length feature representation of the
specific windows, which shares the spirit of Fast R-CNN."

Questo e' STRUTTURALMENTE diverso da q_network.py (Caicedo & Lazebnik):
  - Un solo forward convoluzionale PER IMMAGINE (non per step): la mappa
    conv5_3 (stride 16, 512 canali) viene calcolata una volta all'inizio
    dell'episodio; ogni step successivo fa solo ROI-pooling + fc6 su quella
    mappa gia' calcolata per la finestra corrente -- niente crop/resize/
    ricalcolo del backbone ad ogni step (il "trucco Fast R-CNN" citato dagli
    autori).
  - Lo stato include ANCHE la feature dell'INTERA immagine (stessa fc6,
    stessa mappa condivisa, ROI = tutta l'immagine), non solo quella della
    finestra corrente -- vedi TreeRLVisualBackbone.encode_episode().
  - Nessuna Dueling/Double-DQN qui: la testa e' un MLP semplice (Fig. 4:
    "The MLP predicts the estimated values of the 13 actions"), il paper non
    ne specifica le dimensioni esatte -- vedi ASSUNZIONE in TreeRLQHead.
"""
import torch
import torch.nn as nn
from torchvision.ops import roi_align

from config_treerl import (
    N_ACTIONS, HISTORY_DIM, ROI_POOL_SIZE, CONV5_3_STRIDE, VISUAL_FEAT_DIM,
)


class TreeRLVisualBackbone(nn.Module):
    """VGG-16 pre-addestrato ImageNet, congelato, fermato a conv5_3+ReLU
    (torchvision vgg16().features[:30] -- esclude l'ultimo maxpool, quindi
    stride complessivo 16, non 32). fc6 (+ReLU) e' applicato via ROI-pooling
    sia alla finestra corrente sia (con lo stesso identico layer, stessi
    pesi) alla finestra che copre l'intera immagine, per ottenere la
    "whole image feature" del paper -- interpretazione naturale del testo,
    che non descrive un layer separato per la feature globale.

    Frozen end-to-end (nessun requires_grad=True), coerente con "similar to
    [23] [Caicedo & Lazebnik], we also use the pre-trained CNN as the
    regional feature extractor instead of training the whole hierarchy".
    """

    def __init__(self):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights
        net = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

        # features[:30] = tutto fino a ReLU(conv5_3) incluso, ESCLUSO l'ultimo
        # MaxPool2d (indice 30 su 31 elementi totali) -> stride 16, 512 canali.
        self.conv_body = nn.Sequential(*list(net.features.children())[:30])
        # classifier: 0=fc6(25088->4096) 1=ReLU 2=Dropout 3=fc7 ... -- si usa
        # SOLO fc6+ReLU, esattamente come nel testo ("the feature vector of
        # layer fc6").
        self.fc6 = net.classifier[0]
        self.relu6 = net.classifier[1]

        for module in (self.conv_body, self.fc6):
            for p in module.parameters():
                p.requires_grad = False
        self.eval()

        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.out_dim = VISUAL_FEAT_DIM

    def train(self, mode=True):
        super().train(False)  # backbone sempre congelato, ignora .train()
        return self

    @torch.no_grad()
    def compute_feature_map(self, image_chw):
        """Un solo forward convoluzionale per l'intera immagine (fatto UNA
        volta per episodio, non per step). image_chw: [C,H,W] o [B,C,H,W].
        Ritorna la mappa conv5_3 [B,512,H/16,W/16]."""
        x = image_chw
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = (x - self.imagenet_mean) / self.imagenet_std
        return self.conv_body(x)

    @torch.no_grad()
    def roi_feature(self, feature_map, boxes_xywh, image_size):
        """ROI-pooling (roi_align) + fc6 + ReLU per un batch di finestre
        sulla STESSA feature map (stesso episodio/immagine).

        feature_map: [1, 512, Hf, Wf] (da compute_feature_map).
        boxes_xywh: array/tensore [N, 4] in coordinate pixel dell'immagine
            ORIGINALE (stessa convenzione [x,y,w,h] di utils.py::compute_iou).
        image_size: (H, W) dell'immagine originale, per clippare i box.
        Ritorna [N, 4096].
        """
        device = feature_map.device
        if not torch.is_tensor(boxes_xywh):
            boxes_xywh = torch.as_tensor(boxes_xywh, dtype=torch.float32)
        boxes_xywh = boxes_xywh.to(device).float()

        x1 = boxes_xywh[:, 0]
        y1 = boxes_xywh[:, 1]
        x2 = x1 + boxes_xywh[:, 2]
        y2 = y1 + boxes_xywh[:, 3]
        batch_idx = torch.zeros((boxes_xywh.shape[0], 1), device=device)
        rois = torch.cat([batch_idx, x1[:, None], y1[:, None], x2[:, None], y2[:, None]], dim=1)

        pooled = roi_align(
            feature_map, rois, output_size=ROI_POOL_SIZE,
            spatial_scale=1.0 / CONV5_3_STRIDE, sampling_ratio=-1, aligned=True,
        )  # [N, 512, 7, 7]
        flat = torch.flatten(pooled, 1)  # [N, 25088]
        return self.relu6(self.fc6(flat))  # [N, 4096]


class TreeRLQHead(nn.Module):
    """MLP per la stima dei 13 Q-value (Fig. 4: "The MLP predicts the
    estimated values of the 13 actions"). Il paper NON specifica il numero
    di hidden layer/unita': ASSUNZIONE NON VERIFICATA, si usa qui una testa
    a due hidden layer da 1024 unita' (stessa capacita' della testa "fedele
    al paper" gia' usata in q_network.py per Caicedo & Lazebnik, scelta
    ragionevole per coerenza col resto del progetto, non per un riferimento
    testuale specifico di QUESTO paper)."""

    def __init__(self, in_dim, n_actions=N_ACTIONS, hidden=1024, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, state_repr):
        return self.net(state_repr)


class TreeRLQNet(nn.Module):
    """Combina backbone + testa. Lo "stato" per la testa MLP e' la
    concatenazione [window_feat(4096) ; global_feat(4096) ; history(650)] =
    8842-dim, esattamente i tre componenti elencati nel testo del paper
    ("the feature vector of the current window, the feature vector of the
    whole image and the history of taken actions")."""

    def __init__(self, backbone: TreeRLVisualBackbone, hidden=1024, dropout=0.3):
        super().__init__()
        self.backbone = backbone
        in_dim = 2 * backbone.out_dim + HISTORY_DIM
        self.q_net = TreeRLQHead(in_dim, N_ACTIONS, hidden=hidden, dropout=dropout)

    def encode_episode(self, image_chw):
        """Da chiamare UNA volta a inizio episodio: calcola la mappa
        conv5_3 condivisa e la feature dell'immagine intera (fc6 su una ROI
        = tutta l'immagine). Ritorna (feature_map, global_feat[1,4096])."""
        H, W = image_chw.shape[-2], image_chw.shape[-1]
        feature_map = self.backbone.compute_feature_map(image_chw)
        whole_box = torch.tensor([[0.0, 0.0, float(W), float(H)]])
        global_feat = self.backbone.roi_feature(feature_map, whole_box, (H, W))
        return feature_map, global_feat

    def window_feature(self, feature_map, boxes_xywh, image_size):
        """ROI-pooling + fc6 per una o piu' finestre sulla mappa condivisa."""
        return self.backbone.roi_feature(feature_map, boxes_xywh, image_size)

    def forward(self, window_feat, global_feat, history):
        """window_feat/global_feat: [B,4096]; history: [B,650]."""
        if global_feat.shape[0] == 1 and window_feat.shape[0] > 1:
            global_feat = global_feat.expand(window_feat.shape[0], -1)
        state_repr = torch.cat([window_feat, global_feat, history], dim=1)
        return self.q_net(state_repr)


def build_backbone():
    return TreeRLVisualBackbone()
