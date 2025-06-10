from __future__ import annotations
from typing import Sequence, Tuple, Set, Dict

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _prepare_edges(
    edges: Sequence[float],
    low: float,
    high: float,
    adjust: bool,
) -> np.ndarray:
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
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) == 0:
        raise ValueError("Keine numerischen Spalten gefunden.")

    # kopieren & quadrieren
    df_num = df[num_cols].astype(float).copy()
    df_num = df_num.pow(2)

    # nicht-numerische Spalten anhängen
    non_num = df.columns.difference(num_cols, sort=False)
    return pd.concat([df[non_num], df_num], axis=1)

def _compute_cluster_thresholds(
    labels: np.ndarray,
    reactions: np.ndarray,
    expr_numeric: pd.DataFrame
) -> Dict[int, float]:
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
    samples = expr_numeric.columns
    bin_mat = pd.DataFrame(0, index=reactions, columns=samples, dtype=int)

    for rxn, cid in zip(reactions, labels):
        thr = thresholds[cid]
        # 1) Werte aller Reaktionen im Cluster cid über alle Samples
        cluster_rxns = reactions[labels == cid]    # musst labels hier reinholen
        vals_cluster = expr_numeric.loc[cluster_rxns].to_numpy().flatten()
        vals_nonzero = vals_cluster[~np.isnan(vals_cluster)]
        # 2) Bestimme den tatsächlichen Cut-off als Perzentil thr
        cutoff = np.nanpercentile(vals_nonzero, thr)
        # 3) Vergleiche **unveränderte** Log10-Werte x mit diesem cutoff
        x = expr_numeric.loc[rxn]
        bin_mat.loc[rxn] = (x >= cutoff).astype(int)
    return bin_mat

def _active_entry_set(bin_mat: pd.DataFrame) -> Set[Tuple[str, str]]:
    arr = bin_mat.to_numpy()
    rows, cols = np.nonzero(arr)
    rxns = bin_mat.index.to_numpy()
    samples = bin_mat.columns.to_numpy()
    return set(zip(rxns[rows], samples[cols]))

def _norm_bin_counts(df: pd.DataFrame) -> pd.DataFrame:
    df_norm = df.copy()
    # alle bin_ Spalten ermitteln
    bins = [c for c in df_norm.columns if c.startswith("bin_")]

    # Zeilensummen
    row_sums = df_norm[bins].sum(axis=1)

    # Division, aber vermeide NaN: 0/0 → NaN, daher direkt mit fillna(0)
    df_norm[bins] = df_norm[bins].div(row_sums, axis=0).fillna(0)

    return df_norm


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def bin_reaction_expression(
    rxn_df: pd.DataFrame,
    edges: Sequence[float],
    adjust_bins: bool = False,
) -> Tuple[pd.DataFrame, np.ndarray]:
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
    counts_df_norm = _norm_bin_counts(counts_df)
    return counts_df_norm, edges_adj

def cluster_reactions_iterative(
    counts_df: pd.DataFrame,
    expr_df: pd.DataFrame,
    max_k: int = 100,
    similarity_threshold: float = 0.9
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, float]]:
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

    # Initial computation for k = 2
    labels_current = fcluster(Z, t=2, criterion="maxclust")
    thr_current = _compute_cluster_thresholds(labels_current, reactions, expr_numeric)
    bin_current = _make_binary_core_matrix(labels_current, reactions, expr_numeric, thr_current)
    active_current = _active_entry_set(bin_current)

    for k in range(2, max_k):
        # Compute k+1; k from previous iteration is labels_current
        labels_next = fcluster(Z, t=k+1, criterion="maxclust")
        thr_next = _compute_cluster_thresholds(labels_next, reactions, expr_numeric)
        bin_next = _make_binary_core_matrix(labels_next, reactions, expr_numeric, thr_next)
        active_next = _active_entry_set(bin_next)

        # Jaccard similarity of active sets
        inter = len(active_current & active_next)
        union = len(active_current | active_next)
        sim = inter / union if union else 1.0

        if sim >= similarity_threshold:
            chosen_k = k
            chosen_labels = labels_current
            chosen_bin = bin_current
            thresholds = thr_current
            break

        # Next iteration: k+1 becomes new k
        labels_current = labels_next
        thr_current = thr_next
        bin_current = bin_next
        active_current = active_next
    else:
        chosen_labels = fcluster(Z, t=chosen_k, criterion="maxclust")
        thr_final = _compute_cluster_thresholds(chosen_labels, reactions, expr_numeric)
        chosen_bin = _make_binary_core_matrix(chosen_labels, reactions, expr_numeric, thr_final)
        thresholds = thr_final

    # Ensure the binary matrix includes a first column with reaction names
    if "reaction" not in chosen_bin.columns:
        chosen_bin = chosen_bin.copy()
        chosen_bin.insert(0, "reaction", reactions)

    df_clusters = pd.DataFrame({"reaction": reactions, "cluster": chosen_labels})
    return df_clusters, chosen_bin, thresholds