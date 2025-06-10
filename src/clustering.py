# clustering.py
"""
Utility functions for expression binning / clustering.

Provides
--------
- `bin_reaction_expression`: bin log10‑transformed reaction activities and
  return
    * a reaction × bin‑count matrix (``pd.DataFrame``) **with a leading
      ``reaction`` column**
    * the verwendeten (ggf. angepassten) Bin‑Kanten als ``np.ndarray``

Bin‑Logik
---------
Für *k* Edge‑Werte entstehen *k + 1* Bins:

* Bin‑0   … Werte <  1. Edge
* Bin‑i   … Werte ∈ [Edge_i, Edge_{i+1})
* Bin‑k   … Werte ≥ letzter Edge

Bei ``adjust_bins=True`` wird die Edge‑Liste so skaliert, dass der erste
Edge dem 5‑Perzentil und der letzte dem 95‑Perzentil aller >0 Werte (log10)
entspricht. Somit landen ca. 5 % der Werte im ersten und letzten Bin.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from typing import Tuple
import matplotlib.pyplot as plt


__all__ = [
    "bin_reaction_expression",
]

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _prepare_edges(
    edges: Sequence[float],
    low: float,
    high: float,
    adjust: bool,
) -> np.ndarray:
    """Return a strictly increasing edge array."""
    e = np.asarray(edges, dtype=float).copy()
    if e.ndim != 1 or e.size < 2:
        raise ValueError("`edges` must be 1‑D with ≥2 entries.")

    if adjust:
        scale = (high - low) / (e[-1] - e[0])
        e = (e - e[0]) * scale + low

    # ensure unique & ascending
    e = np.unique(e)
    if e.size < 2:
        raise ValueError("Edge array collapsed to <2 unique values.")
    if e[1] < e[0]:
        e = e[::-1]
    return e


def _log10_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Log10‑transform numeric columns; keep non‑numeric intact."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) == 0:
        raise ValueError("No numeric columns found.")

    df_num = df[num_cols].astype(float).copy()
    df_num[df_num <= 0] = np.nan
    with np.errstate(divide="ignore"):
        df_log = np.log10(df_num)

    non_num = df.columns.difference(num_cols, sort=False)
    return pd.concat([df[non_num], df_log], axis=1)

def _undo_log2_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    „Undo“ einer Log2-Transformation, indem numerische Werte quadriert werden.
    Nicht-numerische Spalten bleiben erhalten.
    """
    # numerische Spalten ermitteln
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) == 0:
        raise ValueError("Keine numerischen Spalten gefunden.")

    # kopieren & quadrieren
    df_num = df[num_cols].astype(float).copy()
    df_num = df_num.pow(2)

    # nicht-numerische Spalten anhängen
    non_num = df.columns.difference(num_cols, sort=False)
    return pd.concat([df[non_num], df_num], axis=1)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def bin_reaction_expression(
    rxn_df: pd.DataFrame,
    edges: Sequence[float],
    adjust_bins: bool = False,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Return bin counts and (possibly adjusted) edges."""

    if rxn_df.empty:
        return pd.DataFrame(), np.asarray(edges, dtype=float)

    rxn_counts = _undo_log2_transform(rxn_df)
    rxn_log = _log10_transform(rxn_counts)
    num_val = rxn_log.select_dtypes(include=[np.number]).to_numpy()

    flat = num_val[~np.isnan(num_val)]
    if flat.size == 0:
        raise ValueError("Matrix contains only NaNs after log10 transform.")

    p1, p99 = np.nanpercentile(flat, [1, 99])
    edges_adj = _prepare_edges(edges, p1, p99, adjust_bins)

    # number of bins: len(edges)+1 (two open bins at ends)
    n_bins = len(edges_adj) + 1

    # digitize
    bin_ids = np.digitize(num_val, edges_adj, right=False)  # 0 … len(edges)

    counts = np.zeros((num_val.shape[0], n_bins), dtype=int)
    for i in range(num_val.shape[0]):
        ids = bin_ids[i][~np.isnan(num_val[i])]  # remove NaNs
        if ids.size:
            counts[i] = np.bincount(ids.astype(int), minlength=n_bins)

    bin_cols = [f"bin_{i}" for i in range(n_bins)]
    counts_df = pd.DataFrame(counts, index=rxn_df.index, columns=bin_cols)
    counts_df.insert(0, "reaction", counts_df.index)
    return counts_df, edges_adj

def norm_bin_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize bin-count DataFrame per reaction:
    - Für jede Zeile werden die bin_*-Spalten durch ihre Zeilensumme geteilt.
    - Reaktionen, die in **keiner** Bin sitzen (Summe=0), bleiben 0.
    - Die Spalte 'reaction' (falls vorhanden) bleibt unverändert.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame aus bin_reaction_expression, mit 'reaction' und bin_*-Spalten.

    Returns
    -------
    pd.DataFrame
        Gleiches Format, bin_*-Werte sind nun relative Häufigkeiten,
        und es gibt **keine** NaNs mehr.
    """
    df_norm = df.copy()
    # alle bin_ Spalten ermitteln
    bins = [c for c in df_norm.columns if c.startswith("bin_")]

    # Zeilensummen
    row_sums = df_norm[bins].sum(axis=1)

    # Division, aber vermeide NaN: 0/0 → NaN, daher direkt mit fillna(0)
    df_norm[bins] = df_norm[bins].div(row_sums, axis=0).fillna(0)

    return df_norm

def _compute_cluster_thresholds(
    labels: np.ndarray,
    reactions: np.ndarray,
    expr_numeric: pd.DataFrame
) -> Dict[int, float]:
    """
    Compute Threshold_c per cluster as specified:
      M  = global mean of all log10 values
      D  = global std  of all log10 values
      u_c   = mean of cluster c values
      sig_c = std  of cluster c values
      g(u_c)   = -(u_c - M)
      f(sig_c) = (sig_c - D) / max(sig_c - D over clusters)
      theta_c  = f + g
      Threshold_c = (theta_c - min(theta)) * 100 / max(theta - min(theta))
    Returns dict {cluster_label : Threshold_c}
    """
    # flatten global stats
    global_vals = expr_numeric.to_numpy().flatten()
    global_vals = global_vals[~np.isnan(global_vals)]
    M = np.mean(global_vals)
    D = np.std(global_vals, ddof=1)

    cluster_ids = np.unique(labels)
    u_list = []
    sig_list = []
    for cid in cluster_ids:
        idx = labels == cid
        rxns = reactions[idx]
        vals = expr_numeric.loc[rxns].to_numpy().flatten()
        vals = vals[~np.isnan(vals)]
        u_list.append(np.mean(vals))
        sig_list.append(np.std(vals, ddof=1))
    u_arr = np.asarray(u_list)
    sig_arr = np.asarray(sig_list)

    # g and f
    g = -(u_arr - M)
    diff_sig = sig_arr - D
    max_diff = np.max(diff_sig) if np.max(diff_sig) != 0 else 1.0
    f = diff_sig / max_diff
    theta = f + g
    theta_min = np.min(theta)
    theta_range = np.max(theta) - theta_min if np.max(theta) - theta_min != 0 else 1.0
    thresholds = (theta - theta_min) * 100.0 / theta_range
    return {cid: thr for cid, thr in zip(cluster_ids, thresholds)}

def _make_binary_core_matrix(
    labels: np.ndarray,
    reactions: np.ndarray,
    expr_numeric: pd.DataFrame,
    thresholds: Dict[int, float]
) -> pd.DataFrame:
    """
    Build binary matrix (reactions x samples) using cluster thresholds.
    Expression values are scaled to 0–100 using global min/max before comparison.
    """
    # global min/max for scaling
    vals = expr_numeric.to_numpy()
    min_val = np.nanmin(vals)
    max_val = np.nanmax(vals)
    rng = max_val - min_val if max_val - min_val != 0 else 1.0

    samples = expr_numeric.columns
    bin_mat = pd.DataFrame(0, index=reactions, columns=samples, dtype=int)

    for rxn, cid in zip(reactions, labels):
        thr = thresholds[cid]
        x = expr_numeric.loc[rxn].to_numpy()
        x_scaled = (x - min_val) * 100.0 / rng
        bin_mat.loc[rxn] = (x_scaled >= thr).astype(int)
    return bin_mat

def _core_reaction_set(bin_mat: pd.DataFrame) -> set:
    """Return set of reactions active in all samples (row sum == n_samples)."""
    n_samples = bin_mat.shape[1]
    core = bin_mat.index[bin_mat.sum(axis=1) == n_samples]
    return set(core)

def cluster_reactions_iterative(
    counts_df: pd.DataFrame,
    expr_df: pd.DataFrame,
    max_k: int = 100,
    similarity_threshold: float = 0.9
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Determine cluster number k using core-reaction stability criterion.
    - counts_df: bin-count matrix (reaction first col 'reaction' or index),
    - expr_df  : log10 expression matrix (reaction rows, sample cols).
    Returns:
      df_clusters: reaction -> cluster for chosen k
      core_bin   : binary matrix (reactions x samples) for chosen k
    """

    # extract reaction names & numeric bin matrix
    if "reaction" in counts_df.columns:
        reactions = counts_df["reaction"].astype(str).to_numpy()
        bins_numeric = counts_df.drop(columns=["reaction"]).select_dtypes(include=[np.number])
    else:
        reactions = counts_df.index.astype(str).to_numpy()
        bins_numeric = counts_df.select_dtypes(include=[np.number])

    X = bins_numeric.to_numpy()
    n_reactions = X.shape[0]
    if n_reactions < 2:
        raise ValueError("Need at least 2 reactions.")

    # reorder expr_df to align with counts_df order
    expr_numeric = expr_df.set_index(expr_df.columns[0]) if expr_df.columns[0] != bins_numeric.index.name else expr_df
    # ensure same reaction order; assume 'reaction' column exists
    if "reaction" in expr_numeric.index.names:
        expr_numeric.index = expr_numeric.index.astype(str)
    if "reaction" in counts_df.columns:
        expr_numeric = expr_numeric.loc[reactions]
    else:
        expr_numeric = expr_numeric.loc[bins_numeric.index.astype(str)]
    # compute linkage on binned counts
    Z = linkage(X, method="complete", metric="euclidean")

    chosen_k = max_k
    max_k = min(max_k, n_reactions - 1)

    for k in range(14, max_k + 1):
        # cluster for k and k+1
        labels_k = fcluster(Z, t=k, criterion="maxclust")
        labels_k1 = fcluster(Z, t=k+1, criterion="maxclust")

        # thresholds per cluster
        thr_k = _compute_cluster_thresholds(labels_k, reactions, expr_numeric)
        thr_k1 = _compute_cluster_thresholds(labels_k1, reactions, expr_numeric)

        # binary core matrices
        bin_k = _make_binary_core_matrix(labels_k, reactions, expr_numeric, thr_k)
        bin_k1 = _make_binary_core_matrix(labels_k1, reactions, expr_numeric, thr_k1)

        # core reaction sets
        core_k = _core_reaction_set(bin_k)
        core_k1 = _core_reaction_set(bin_k1)

        # Jaccard similarity of core sets
        inter = len(core_k & core_k1)
        union = len(core_k | core_k1)
        sim = inter / union if union else 1.0

        if sim >= similarity_threshold:
            chosen_k = k
            chosen_labels = labels_k
            chosen_bin = bin_k
            break
    else:
        chosen_labels = fcluster(Z, t=chosen_k, criterion="maxclust")
        thr_final = _compute_cluster_thresholds(chosen_labels, reactions, expr_numeric)
        chosen_bin = _make_binary_core_matrix(chosen_labels, reactions, expr_numeric, thr_final)

    df_clusters = pd.DataFrame({"reaction": reactions, "cluster": chosen_labels})
    return df_clusters, chosen_bin

def plot_jaccard_curve(
    counts_df: pd.DataFrame,
    max_k: int = 100,
    save_path: str | None = None
):
    """
    Berechnet und plottet den durchschnittlichen Jaccard-Index für jedes k von 2 bis max_k
    basierend auf hierarchischem Clustering (Euclidean + complete linkage).
    Speichert den Plot, wenn save_path angegeben ist.

    Parameters
    ----------
    counts_df : pd.DataFrame
        DataFrame mit 'reaction' und bin_*-Spalten oder nur bin_*-Spalten.
    max_k : int
        Maximale Clusterzahl, bis zu der der Jaccard berechnet wird (default=100).
    save_path : str or None
        Pfad, unter dem der Plot gespeichert wird (z.B. 'jaccard_plot.png').
        Wenn None, wird der Plot nur angezeigt.
    """
    # 1. Reaktionen extrahieren und numerische Matrix
    if "reaction" in counts_df.columns:
        reactions = counts_df["reaction"].astype(str).to_numpy()
        numeric_df = counts_df.drop(columns=["reaction"])
    else:
        reactions = counts_df.index.astype(str).to_numpy()
        numeric_df = counts_df.copy()

    numeric_df = numeric_df.select_dtypes(include=[np.number])
    X = numeric_df.values
    n = X.shape[0]
    if n < 2:
        raise ValueError("Mindestens 2 Reaktionen benötigt zum Clustern.")

    # 2. Linkage berechnen
    Z = linkage(X, method="complete", metric="euclidean")

    # 3. Jaccard-Werte vorbereiten
    jaccard_vals = np.full(max_k + 1, np.nan)

    for k in range(2, min(max_k, n - 1) + 1):
        labels_k  = fcluster(Z, t=k,   criterion="maxclust")
        labels_k1 = fcluster(Z, t=k+1, criterion="maxclust")

        sets_k  = {i: set(reactions[labels_k  == i]) for i in range(1, k+1)}
        sets_k1 = {j: set(reactions[labels_k1 == j]) for j in range(1, k+2)}

        vals = []
        for Si in sets_k.values():
            best = 0.0
            for Sj in sets_k1.values():
                inter = len(Si & Sj)
                union = len(Si | Sj)
                if union > 0:
                    best = max(best, inter / union)
            vals.append(best)
        jaccard_vals[k] = np.mean(vals)

    # 4. Plot
    plt.figure()
    plt.plot(range(max_k + 1), jaccard_vals, marker='o')
    plt.xlabel('k')
    plt.ylabel('Mean Jaccard similarity')
    plt.title('Jaccard similarity vs. number of clusters k')
    plt.tight_layout()

    # 5. Save or show
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()