"""
Script standalone per il calcolo di statistiche descrittive sul dataset
(train / val / test): dimensione della GT rispetto all'immagine, colore medio
del tumore, concavità/feature geometriche, distanza dal centro immagine e
gradiente medio dentro l'area della GT.

In più:
  - genera e salva grafici riassuntivi (istogrammi di ogni metrica + heatmap di
    correlazione tra le metriche) che spiegano la varianza presente nel dataset;
  - salva, per ogni categoria di metriche, gli indici delle 5 immagini "migliori"
    (valore più alto) e delle 5 "peggiori" (valore più basso).

Estratto dal ramo --testDataset di tutto15OracoloConvergeMio.py: contiene solo
il codice necessario a questa funzionalità (nessuna parte di training/RL).

Uso:
    python dataset_stats.py --output-root ./ppo_logs \
        --dataset-source kaggle --kaggle-id pkdarabi/brain-tumor-image-dataset-semantic-segmentation

    python dataset_stats.py --output-root ./ppo_logs \
        --dataset-source local --dataset-path /percorso/al/dataset

Dipendenze aggiuntive rispetto al progetto originale: matplotlib (per i grafici).
"""

import argparse
import csv
import math
import os

import cv2
import numpy as np
from tqdm import tqdm

try:
    import matplotlib
    matplotlib.use("Agg")  # backend non interattivo, per salvare su file senza display
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False


# Numero di immagini "migliori"/"peggiori" da salvare per ogni metrica
TOP_BOTTOM_K = 5

# Raggruppamento delle metriche nelle 5 categorie richieste (usato sia per la
# stampa a schermo sia per il salvataggio dei top/bottom). Le chiavi
# "color_mean_chX" vengono aggiunte dinamicamente in base al numero di canali.
CATEGORY_DEFINITIONS = {
    "dimensione_gt": ["area_ratio"],
    "colore_medio_tumore": ["color_mean_overall"],  # + color_mean_chX aggiunte a runtime
    "concavita_e_geometria": [
        "solidity", "circularity", "extent", "aspect_ratio",
        "eccentricity", "num_convexity_defects", "max_defect_depth_norm"
    ],
    "distanza_dal_centro": ["dist_from_center_norm"],
    "gradiente_medio_tumore": ["grad_mean"],
}


def _analyze_sample_for_stats(image_t, mask_t):
    """
    Calcola le feature statistiche per una singola immagine + ground truth.

    Assunzioni sul formato dati:
      - image_t: tensore/array (C, H, W) con valori già in [0, 1].
      - mask_t: tensore/array (1, H, W) oppure (H, W) con valori in [0, 1],
        binarizzata con soglia 0.5 (mask > 0.5).
    """
    # Porto tutto su CPU/numpy (funziona sia con tensori torch che array numpy)
    if hasattr(image_t, "detach"):
        image = image_t.detach().cpu().float().numpy()
    else:
        image = np.asarray(image_t, dtype=np.float32)
    if image.ndim == 2:
        image = image[None, ...]
    C, H, W = image.shape

    if hasattr(mask_t, "detach"):
        mask = mask_t.detach().cpu().float().numpy()
    else:
        mask = np.asarray(mask_t, dtype=np.float32)
    if mask.ndim == 3:
        mask = mask[0]

    mask_bin = (mask > 0.5).astype(np.uint8)
    area = int(mask_bin.sum())
    total_pixels = H * W

    out = {
        "area_ratio": (area / total_pixels) if total_pixels > 0 else np.nan,
        "color_mean_overall": np.nan,
        "solidity": np.nan,
        "circularity": np.nan,
        "extent": np.nan,
        "aspect_ratio": np.nan,
        "eccentricity": np.nan,
        "num_convexity_defects": 0,
        "max_defect_depth_norm": np.nan,
        "dist_from_center_norm": np.nan,
        "grad_mean": np.nan,
    }

    # Colore medio del tumore (per-canale + media complessiva)
    img_hwc = np.transpose(image, (1, 2, 0))  # H, W, C
    color_per_channel = np.full((C,), np.nan, dtype=np.float64)
    if area > 0:
        tumor_pixels = img_hwc[mask_bin.astype(bool)]  # (N, C)
        color_per_channel = tumor_pixels.mean(axis=0)
        out["color_mean_overall"] = float(tumor_pixels.mean())
    for c in range(C):
        out[f"color_mean_ch{c}"] = float(color_per_channel[c]) if C > 0 else np.nan

    # Centroide della GT (default: centro immagine, se manca la GT)
    centroid = (W / 2.0, H / 2.0)

    # Concavità e feature geometriche (via contorni OpenCV)
    if area > 0:
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) > 0:
            cnt = max(contours, key=cv2.contourArea)
            cnt_area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            x, y, w, h = cv2.boundingRect(cnt)
            bbox_area = w * h

            # Centroide "reale" della GT (momenti)
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                centroid = (M["m10"] / M["m00"], M["m01"] / M["m00"])
            else:
                centroid = (x + w / 2.0, y + h / 2.0)

            # Solidity: area / area dell'involucro convesso -> quanto la forma è "concava"
            # (1 = perfettamente convessa, valori più bassi = più concava)
            if hull_area > 0:
                out["solidity"] = float(cnt_area / hull_area)

            # Circolarità: 4*pi*Area / Perimetro^2 (1 = cerchio perfetto)
            if perimeter > 0:
                out["circularity"] = float((4.0 * math.pi * cnt_area) / (perimeter ** 2))

            # Extent: area / area del bounding box
            if bbox_area > 0:
                out["extent"] = float(cnt_area / bbox_area)

            # Aspect ratio del bounding box
            if min(w, h) > 0:
                out["aspect_ratio"] = float(max(w, h) / min(w, h))

            # Eccentricità (richiede almeno 5 punti per fittare un'ellisse)
            if len(cnt) >= 5:
                (_, _), (MA, ma), _ = cv2.fitEllipse(cnt)
                major, minor = max(MA, ma), min(MA, ma)
                if major > 0:
                    out["eccentricity"] = float(math.sqrt(max(0.0, 1.0 - (minor / major) ** 2)))

            # Convexity defects: numero e profondità massima normalizzata sulla diagonale
            hull_idx = cv2.convexHull(cnt, returnPoints=False)
            if hull_idx is not None and len(hull_idx) > 3:
                try:
                    hull_idx = np.sort(hull_idx, axis=0)
                    defects = cv2.convexityDefects(cnt, hull_idx)
                    if defects is not None:
                        depths = defects[:, 0, 3] / 256.0
                        diag = math.sqrt(H ** 2 + W ** 2)
                        out["num_convexity_defects"] = int(len(depths))
                        out["max_defect_depth_norm"] = float(depths.max() / diag) if diag > 0 else np.nan
                    else:
                        out["num_convexity_defects"] = 0
                        out["max_defect_depth_norm"] = 0.0
                except cv2.error:
                    pass

    # Distanza del centroide della GT dal centro dell'immagine (normalizzata sulla
    # semi-diagonale, cosi' il valore sta in [0, 1] circa)
    diag_half = math.sqrt(H ** 2 + W ** 2) / 2.0
    img_center = (W / 2.0, H / 2.0)
    dist = math.sqrt((centroid[0] - img_center[0]) ** 2 + (centroid[1] - img_center[1]) ** 2)
    out["dist_from_center_norm"] = float(dist / diag_half) if diag_half > 0 else np.nan

    # Media dei gradienti dell'immagine calcolati solo dentro l'area della GT
    gray = image.mean(axis=0) if C > 1 else image[0]
    gx = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    if area > 0:
        out["grad_mean"] = float(grad_mag[mask_bin.astype(bool)].mean())

    return out


def _summarize_column(values):
    """Ritorna (min, max, media, std) ignorando i NaN, oppure None se non ci sono dati validi."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return None
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "n": int(arr.size),
    }


def _build_category_metrics(all_keys):
    """Espande CATEGORY_DEFINITIONS aggiungendo le colonne color_mean_chX rilevate a runtime."""
    color_channel_keys = sorted([k for k in all_keys if k.startswith("color_mean_ch")])
    categories = {k: list(v) for k, v in CATEGORY_DEFINITIONS.items()}
    categories["colore_medio_tumore"] += color_channel_keys
    return categories


def compute_dataset_statistics(dataset, partition_name, output_dir):
    """
    Itera su tutte le immagini di una partizione del dataset e calcola:
      - dimensione della GT rispetto all'immagine totale
      - colore medio del tumore
      - concavità e feature geometriche della GT
      - distanza del centroide della GT dal centro dell'immagine
      - media dei gradienti dell'immagine dentro l'area della GT

    Salva un CSV con i valori per-immagine e uno con il riepilogo (min/max/mean/std),
    e ritorna (summary, rows, per_image_path, summary_path).
    """
    n = len(dataset)
    rows = []

    for idx in tqdm(range(n), desc=f"[testDataset] Analisi partizione '{partition_name}'"):
        sample = dataset[idx]
        feats = _analyze_sample_for_stats(sample["image"], sample["mask"])
        feats["idx"] = idx
        rows.append(feats)

    # Determino dinamicamente tutte le colonne (incluse quelle color_mean_chX)
    all_keys = []
    for r in rows:
        for k in r.keys():
            if k not in all_keys:
                all_keys.append(k)

    os.makedirs(output_dir, exist_ok=True)

    # --- CSV per-immagine ---
    per_image_path = os.path.join(output_dir, f"dataset_stats_{partition_name}_per_image.csv")
    with open(per_image_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # --- Riepilogo min/max/mean/std per ogni metrica ---
    metric_cols = [k for k in all_keys if k != "idx"]
    summary = {}
    for col in metric_cols:
        values = [r.get(col, np.nan) for r in rows]
        s = _summarize_column(values)
        if s is not None:
            summary[col] = s

    summary_path = os.path.join(output_dir, f"dataset_stats_{partition_name}_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "min", "max", "mean", "std", "n"])
        for col, s in summary.items():
            writer.writerow([col, f"{s['min']:.6f}", f"{s['max']:.6f}", f"{s['mean']:.6f}", f"{s['std']:.6f}", s["n"]])

    return summary, rows, per_image_path, summary_path


def _print_stats_group(title, summary, keys, unit=""):
    print(f"[INFO]   -- {title} --")
    for k in keys:
        s = summary.get(k)
        if s is None:
            print(f"[INFO]     {k:28s}: dati non disponibili (nessuna GT valida)")
            continue
        print(f"[INFO]     {k:28s}: min {s['min']:.4f}{unit}  max {s['max']:.4f}{unit}  "
              f"media {s['mean']:.4f}{unit}  std {s['std']:.4f}{unit}  (n={s['n']})")


# --------------------------------------------------------------------------- #
# Grafici riassuntivi della varianza del dataset
# --------------------------------------------------------------------------- #

def plot_partition_variance(rows, metric_cols, partition_name, output_dir):
    """
    Genera e salva due grafici per la partizione:
      1) griglia di istogrammi, uno per metrica, per visualizzare la distribuzione
         (e quindi la varianza) di ogni statistica sul dataset;
      2) heatmap di correlazione tra tutte le metriche (per capire quali variano
         insieme, es. dimensione GT vs distanza dal centro).

    Ritorna (hist_path, corr_path); se matplotlib non è disponibile, ritorna
    (None, None) e stampa un avviso.
    """
    if not _HAS_MATPLOTLIB:
        print("[WARN] matplotlib non installato: salto la generazione dei grafici "
              "(pip install matplotlib per abilitarli).")
        return None, None

    os.makedirs(output_dir, exist_ok=True)

    # --- 1) Griglia di istogrammi (varianza di ogni singola metrica) ---
    ncols = 4
    nrows = math.ceil(len(metric_cols) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.0 * nrows))
    axes = np.atleast_1d(axes).reshape(-1)

    for i, metric in enumerate(metric_cols):
        ax = axes[i]
        values = np.array([r.get(metric, np.nan) for r in rows], dtype=np.float64)
        values = values[~np.isnan(values)]
        if values.size == 0:
            ax.set_title(f"{metric}\n(dati non disponibili)", fontsize=8)
            ax.axis("off")
            continue
        ax.hist(values, bins=20, color="#4C72B0", edgecolor="black", alpha=0.85)
        ax.axvline(values.mean(), color="red", linestyle="--", linewidth=1,
                   label=f"media={values.mean():.3f}\nstd={values.std():.3f}")
        ax.set_title(metric, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc="upper right")

    for j in range(len(metric_cols), len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Distribuzione delle metriche (varianza) - partizione '{partition_name}'", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    hist_path = os.path.join(output_dir, f"dataset_stats_{partition_name}_histograms.png")
    fig.savefig(hist_path, dpi=140)
    plt.close(fig)

    # --- 2) Heatmap di correlazione tra le metriche ---
    valid_metrics = []
    data_by_metric = []
    for metric in metric_cols:
        values = np.array([r.get(metric, np.nan) for r in rows], dtype=np.float64)
        if np.all(np.isnan(values)) or np.nanstd(values) == 0:
            continue
        valid_metrics.append(metric)
        data_by_metric.append(values)

    corr_path = None
    if len(valid_metrics) >= 2:
        m = len(valid_metrics)
        corr = np.full((m, m), np.nan)
        for i in range(m):
            for j in range(m):
                a, b = data_by_metric[i], data_by_metric[j]
                mask = ~np.isnan(a) & ~np.isnan(b)
                if mask.sum() > 1 and np.std(a[mask]) > 0 and np.std(b[mask]) > 0:
                    corr[i, j] = np.corrcoef(a[mask], b[mask])[0, 1]

        fig2, ax2 = plt.subplots(figsize=(0.55 * m + 3, 0.55 * m + 3))
        im = ax2.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
        ax2.set_xticks(range(m))
        ax2.set_yticks(range(m))
        ax2.set_xticklabels(valid_metrics, rotation=90, fontsize=7)
        ax2.set_yticklabels(valid_metrics, fontsize=7)
        for i in range(m):
            for j in range(m):
                if not np.isnan(corr[i, j]):
                    ax2.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=5.5)
        fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label="correlazione")
        ax2.set_title(f"Correlazione tra le metriche - '{partition_name}'", fontsize=11)
        fig2.tight_layout()
        corr_path = os.path.join(output_dir, f"dataset_stats_{partition_name}_correlation.png")
        fig2.savefig(corr_path, dpi=140)
        plt.close(fig2)

    return hist_path, corr_path


def plot_cross_partition_comparison(rows_by_partition, metric_cols, output_dir):
    """
    Genera un grafico riassuntivo che confronta la distribuzione (boxplot) di
    ogni metrica tra le diverse partizioni (train/val/test), utile per capire
    se la varianza del dataset è distribuita in modo omogeneo tra le partizioni.
    """
    if not _HAS_MATPLOTLIB:
        return None
    partitions = [p for p, rows in rows_by_partition.items() if rows]
    if len(partitions) < 2:
        return None

    ncols = 4
    nrows = math.ceil(len(metric_cols) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.0 * nrows))
    axes = np.atleast_1d(axes).reshape(-1)

    for i, metric in enumerate(metric_cols):
        ax = axes[i]
        data = []
        labels = []
        for p in partitions:
            values = np.array([r.get(metric, np.nan) for r in rows_by_partition[p]], dtype=np.float64)
            values = values[~np.isnan(values)]
            if values.size > 0:
                data.append(values)
                labels.append(p)
        if not data:
            ax.set_title(f"{metric}\n(dati non disponibili)", fontsize=8)
            ax.axis("off")
            continue
        ax.boxplot(data, labels=labels, showmeans=True)
        ax.set_title(metric, fontsize=9)
        ax.tick_params(labelsize=7)

    for j in range(len(metric_cols), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Confronto della varianza tra le partizioni (train / val / test)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(output_dir, "dataset_stats_cross_partition_comparison.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Top-5 / bottom-5 per categoria
# --------------------------------------------------------------------------- #

def _top_bottom_for_metric(rows, metric, k=TOP_BOTTOM_K):
    """Ritorna (best, worst): liste di (idx, value) con i k valori più alti e più bassi."""
    valid = [(r["idx"], r.get(metric, np.nan)) for r in rows]
    valid = [(idx, v) for idx, v in valid if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not valid:
        return [], []
    valid_sorted = sorted(valid, key=lambda t: t[1])
    worst = valid_sorted[:k]
    best = list(reversed(valid_sorted[-k:]))
    return best, worst


def save_top_bottom_per_category(rows, all_keys, partition_name, output_dir, k=TOP_BOTTOM_K):
    """
    Per ogni categoria di metriche (dimensione GT, colore, concavità/geometria,
    distanza dal centro, gradiente) e per ogni metrica al suo interno, salva gli
    indici delle k immagini con valore più alto ("migliori") e più basso
    ("peggiori"). Ritorna (path_csv, dict_riassuntivo).
    """
    categories = _build_category_metrics(all_keys)

    path = os.path.join(output_dir, f"dataset_stats_{partition_name}_top_bottom.csv")
    highlights = {}  # per stampa rapida a schermo: {categoria: {metrica: (best, worst)}}

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "metric", "rank", "type", "image_idx", "value"])
        for category, metrics in categories.items():
            highlights[category] = {}
            for metric in metrics:
                best, worst = _top_bottom_for_metric(rows, metric, k)
                highlights[category][metric] = (best, worst)
                for rank, (idx, val) in enumerate(best, start=1):
                    writer.writerow([category, metric, rank, "migliore (top)", idx, f"{val:.6f}"])
                for rank, (idx, val) in enumerate(worst, start=1):
                    writer.writerow([category, metric, rank, "peggiore (bottom)", idx, f"{val:.6f}"])

    return path, highlights


def _print_top_bottom_highlights(highlights):
    """Stampa a schermo un estratto rapido: per ogni categoria, la metrica 'principale'."""
    headline_metric = {
        "dimensione_gt": "area_ratio",
        "colore_medio_tumore": "color_mean_overall",
        "concavita_e_geometria": "solidity",
        "distanza_dal_centro": "dist_from_center_norm",
        "gradiente_medio_tumore": "grad_mean",
    }
    print("[INFO]   -- Estratto top/bottom (vedi CSV per l'elenco completo di tutte le metriche) --")
    for category, metric in headline_metric.items():
        best_worst = highlights.get(category, {}).get(metric)
        if not best_worst:
            continue
        best, worst = best_worst
        best_str = ", ".join(f"idx={idx}({val:.4f})" for idx, val in best)
        worst_str = ", ".join(f"idx={idx}({val:.4f})" for idx, val in worst)
        print(f"[INFO]     [{category}] {metric}")
        print(f"[INFO]       migliori (top-{len(best)}) : {best_str}")
        print(f"[INFO]       peggiori (bottom-{len(worst)}): {worst_str}")


# --------------------------------------------------------------------------- #
# Entry point del ramo testDataset
# --------------------------------------------------------------------------- #

def run_dataset_statistics(output_root, partitions):
    """
    Calcola e stampa/salva le statistiche, i grafici riassuntivi e i top/bottom
    per ciascuna partizione (train, val, test) del dataset passato in
    'partitions' (dict: nome -> oggetto Dataset).
    """
    output_dir = os.path.join(output_root, "dataset_stats")
    os.makedirs(output_dir, exist_ok=True)

    print("[INFO] ==================== testDataset ====================")

    rows_by_partition = {}
    metric_cols_ref = None

    for partition_name, dataset in partitions.items():
        if dataset is None or len(dataset) == 0:
            print(f"[INFO] Partizione '{partition_name}' vuota o non disponibile, la salto.")
            rows_by_partition[partition_name] = []
            continue

        print(f"[INFO] Partizione '{partition_name}': {len(dataset)} immagini")
        summary, rows, per_image_path, summary_path = compute_dataset_statistics(
            dataset, partition_name, output_dir
        )
        rows_by_partition[partition_name] = rows

        all_keys = [k for k in rows[0].keys() if k != "idx"]
        metric_cols_ref = all_keys  # riuso identico per tutte le partizioni

        # Determino dinamicamente le colonne colore per-canale presenti
        color_channel_keys = sorted([k for k in summary.keys() if k.startswith("color_mean_ch")])

        print(f"[INFO] --- Statistiche partizione '{partition_name}' ---")
        _print_stats_group(
            "Dimensione GT rispetto all'immagine totale (area_ratio)",
            summary, ["area_ratio"]
        )
        _print_stats_group(
            "Colore medio del tumore",
            summary, ["color_mean_overall"] + color_channel_keys
        )
        _print_stats_group(
            "Concavità e feature geometriche",
            summary,
            ["solidity", "circularity", "extent", "aspect_ratio", "eccentricity",
             "num_convexity_defects", "max_defect_depth_norm"]
        )
        _print_stats_group(
            "Distanza del centroide GT dal centro immagine (normalizzata)",
            summary, ["dist_from_center_norm"]
        )
        _print_stats_group(
            "Media dei gradienti dell'immagine dentro l'area della GT",
            summary, ["grad_mean"]
        )

        print(f"[INFO] CSV per-immagine salvato in : {per_image_path}")
        print(f"[INFO] CSV riepilogo salvato in    : {summary_path}")

        # --- Grafici riassuntivi della varianza ---
        hist_path, corr_path = plot_partition_variance(rows, all_keys, partition_name, output_dir)
        if hist_path:
            print(f"[INFO] Grafico istogrammi salvato in : {hist_path}")
        if corr_path:
            print(f"[INFO] Grafico correlazione salvato in: {corr_path}")

        # --- Top-5 / bottom-5 per categoria ---
        top_bottom_path, highlights = save_top_bottom_per_category(rows, all_keys, partition_name, output_dir)
        print(f"[INFO] CSV top/bottom (5 migliori/5 peggiori per metrica) salvato in: {top_bottom_path}")
        _print_top_bottom_highlights(highlights)

        print("[INFO] ---------------------------------------------------")

    # --- Grafico riassuntivo di confronto tra le partizioni ---
    if metric_cols_ref is not None:
        cross_path = plot_cross_partition_comparison(rows_by_partition, metric_cols_ref, output_dir)
        if cross_path:
            print(f"[INFO] Grafico di confronto tra partizioni salvato in: {cross_path}")

    print("[INFO] ==================== Fine testDataset ====================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calcola statistiche descrittive, grafici e top/bottom sul dataset (train/val/test)."
    )

    # Cartella di output (dentro verrà creata la sottocartella dataset_stats/)
    parser.add_argument("--output-root", type=str, default="./ppo_logs")

    # Sorgente del dataset (stessa interfaccia di get_datasets(cfg) usata nel progetto)
    parser.add_argument("--dataset-source", type=str, default=os.environ.get("DATASET_SOURCE", "kaggle"))

    # Percorso del dataset (se locale)
    parser.add_argument("--dataset-path", type=str, default=os.environ.get("DATASET_PATH", None))

    # ID del dataset Kaggle
    parser.add_argument("--kaggle-id", type=str, default=os.environ.get(
        "KAGGLE_DATASET_ID", "pkdarabi/brain-tumor-image-dataset-semantic-segmentation"
    ))

    args = parser.parse_args()

    # Configurazione minima necessaria a get_datasets(cfg): stessa struttura usata
    # nello script principale, cosi' il dataset viene caricato/normalizzato allo stesso modo.
    cfg = {
        "dataset": {
            "source": args.dataset_source,
            "kaggle_id": args.kaggle_id,
            "local_path": args.dataset_path,
            "image_size": [224, 224],
            "in_channels": 3,
            "train_ratio": (1501 / 2145),
            "valid_ratio": (429 / 2145),
            "cache_pairs": False
        },
        "preprocessing": {
            "normalization": "per_image",
            "binarize_mask": True,
            "mask_threshold": 0.5,
            "white_balance": False,
            "clahe": False,
            "denoise": False
        },
        "training": {
            "batch_size": 1,
            "num_workers": 0
        },
        "seed": 42
    }

    print("[INFO] Caricamento dataset...")
    from dataset import get_datasets
    train_ds, valid_ds, test_ds = get_datasets(cfg)

    run_dataset_statistics(args.output_root, {"train": train_ds, "val": valid_ds, "test": test_ds})
