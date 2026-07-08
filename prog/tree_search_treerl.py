"""
tree_search_treerl.py
──────────────────────
Ricerca ad albero (Sezione 3.2, "Tree-Structured Search") fedele a Jie et
al. 2016 -- usata SOLO in inferenza/valutazione, non in training (durante il
training l'agente segue un solo percorso sequenziale per episodio, vedi
train_treerl.py).

Citazione esatta: "For each window, the actions with the highest predicted
value in both the scaling action group and the local translation action
group are selected respectively. The two best actions are both taken to
obtain two next windows [...] Such bifurcation is performed recursively by
each window starting from the whole image in a top-down fashion."

Con L livelli si ottengono 2^L - 1 proposal (1 + 2 + 4 + ... + 2^(L-1)),
ordinate per livello di profondita' (livello 1 = radice/immagine intera,
poi via via piu' fini), esattamente come Tabella 2 del paper ("1+2+4+8+16=31
proposals" con L=5).
"""
import torch

from config_treerl import SCALE_ACTIONS, TRANSLATE_ACTIONS, N_ACTIONS, HISTORY_DIM
from environment_treerl import apply_action


@torch.no_grad()
def tree_search(q_net, image_chw, num_levels, device):
    """Esegue la ricerca ad albero completa su una singola immagine e
    ritorna la lista di proposal (box [x,y,w,h] in coordinate pixel
    dell'immagine originale), ORDINATE per livello di profondita' crescente
    (radice esclusa, dato che coincide sempre con l'intera immagine).

    q_net: TreeRLQNet gia' su `device`, in eval().
    image_chw: tensore [C,H,W] (float, normalizzato come nel resto della
        pipeline).
    num_levels: numero di livelli dell'albero DOPO la radice (livello 0).
        Es. num_levels=5 -> 1+2+4+8+16 = 31 proposal (come Tabella 2).
    """
    q_net.eval()
    H, W = image_chw.shape[-2], image_chw.shape[-1]
    image_chw = image_chw.to(device)

    feature_map, global_feat = q_net.encode_episode(image_chw.unsqueeze(0) if image_chw.dim() == 3 else image_chw)

    root_box = torch.tensor([0.0, 0.0, float(W), float(H)], dtype=torch.float32)
    root_history = torch.zeros(HISTORY_DIM, dtype=torch.float32)

    # Ogni nodo della frontiera corrente: (box, history_vector)
    frontier = [(root_box, root_history)]
    proposals = []  # accumula le finestre generate livello per livello

    for _level in range(num_levels):
        next_frontier = []
        if not frontier:
            break

        boxes = torch.stack([b for b, _h in frontier]).to(device)          # [N,4]
        histories = torch.stack([h for _b, h in frontier]).to(device)      # [N,650]

        window_feats = q_net.window_feature(feature_map, boxes, (H, W))    # [N,4096]
        q_vals = q_net(window_feats, global_feat, histories)               # [N,13]

        best_scale = q_vals[:, SCALE_ACTIONS].argmax(dim=1) + SCALE_ACTIONS[0]
        best_translate = q_vals[:, TRANSLATE_ACTIONS].argmax(dim=1) + TRANSLATE_ACTIONS[0]

        for i, (box, hist) in enumerate(frontier):
            box_np = box.numpy()
            for action in (int(best_scale[i].item()), int(best_translate[i].item())):
                child_box = apply_action(box_np, action, W, H)
                child_box_t = torch.from_numpy(child_box)

                one_hot = torch.zeros(N_ACTIONS)
                one_hot[action] = 1.0
                child_hist = torch.cat([hist[N_ACTIONS:], one_hot])  # shift a sinistra, append in coda

                proposals.append(child_box)
                next_frontier.append((child_box_t, child_hist))

        frontier = next_frontier

    return proposals
