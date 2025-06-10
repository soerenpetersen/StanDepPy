# standep_py/models.py

import os
import numpy as np
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering, cophenet, dendrogram
from typing import Optional, Tuple, List, Dict, Any
from cobra import Model
from .clustering import ClustObj
from scipy.sparse import csr_matrix, coo_matrix


def get_jaccard_sim_matrix(A: np.ndarray) -> np.ndarray:
    """
    Compute Jaccard similarity between columns of a boolean matrix A
    in a memory-efficient way.
    
    For matrices with a small number of columns (n < threshold),
    the standard sparse dot-product is used.
    Otherwise, a column-wise iteration computes only nonzero entries.

    Args:
        A: Boolean array, shape (n_items, n_sets)

    Returns:
        If n < threshold: a dense NumPy array.
        Otherwise: a sparse COO matrix converted to a dense array 
                     only when explicitly requested later.
    """
    n = A.shape[1]
    threshold = 10000  # Schwelle für direkte Berechnung per Dot-Product

    if n < threshold:
        A_bool = csr_matrix(A.astype(int))
        intersections = A_bool.T.dot(A_bool).tocoo()
        col_sums = np.array(A_bool.sum(axis=0)).flatten()
        rows = intersections.row
        cols = intersections.col
        data = intersections.data.astype(float)
        union = col_sums[rows] + col_sums[cols] - data
        with np.errstate(divide='ignore', invalid='ignore'):
            jaccard_data = np.divide(
                data,
                union,
                out=np.zeros_like(data, dtype=float),
                where=union != 0
            )
        return coo_matrix(
            (jaccard_data, (rows, cols)),
            shape=(n, n)
        ).toarray()
    else:
        A_bool = csr_matrix(A.astype(int)).tocsc()
        col_sums = np.array(A_bool.sum(axis=0)).flatten()
        rows_list = []
        cols_list = []
        data_list = []
        for i in range(n):
            idx_i = set(A_bool.getcol(i).indices)
            if not idx_i:
                continue
            for j in range(i, n):
                idx_j = set(A_bool.getcol(j).indices)
                if not idx_j:
                    continue
                inter = len(idx_i & idx_j)
                if inter == 0:
                    continue
                union = col_sums[i] + col_sums[j] - inter
                sim = inter / union if union > 0 else 0.0
                if sim > 0:
                    rows_list.append(i)
                    cols_list.append(j)
                    data_list.append(sim)
                    if i != j:
                        rows_list.append(j)
                        cols_list.append(i)
                        data_list.append(sim)
        return coo_matrix(
            (np.array(data_list), (np.array(rows_list), np.array(cols_list))),
            shape=(n, n)
        ).toarray()


def models4m_clusters1(
    clustObj: ClustObj,
    tisNames: List[str],
    model: Optional[Model],
    xedge: np.ndarray,
    folderName: Optional[str] = None,
    cutOff: Optional[np.ndarray] = None,
    figFlag: bool = False,
    adjustValue: float = 0.0,
    weightage: List[float] = [1.0, 1.0]
) -> Tuple[
    np.ndarray,      # geneTis (or activeRxns if model gegeben)
    np.ndarray,      # enzTis
    np.ndarray,      # cutOff
    np.ndarray,      # thr
    np.ndarray,      # enzSel
    Dict[str, Any]   # tisClust
]:
    """
    Portierte Funktion von MATLAB models4mClusters1.
    Führt pro Cluster und Tissue eine Schwellenwert-Filterung durch,
    berechnet cluster-spezifische Jaccard-Matrizen (optional),
    und erstellt eine globale Jaccard-Matrix mit Tissue-Clustering.
    Wenn ein Model übergeben wird, werden ausgewählte Gene/Enzyme
    auf Reaktionen abgebildet.

    Returns:
        geneTis:   Boolean-Matrix (#Objekte × #Tis), oder aktivesRxns (#Rxn × #Tis), wenn model nicht None
        enzTis:    Ursprüngliche Boolean-Matrix vor Mapping auf Reaktionen
        cutOff:    Per-Cluster-Prozentilwerte [0–100]
        thr:       Absolute Schwellenwerte pro Cluster
        enzSel:    Integer-Matrix (#Objekte × #Tis), Cluster-Zugehörigkeit pro Objekt/Tissue
        tisClust:  Dict mit Tissue-Clustering-Ergebnissen
    """
    # 1) Falls cutOff nicht vorgegeben, berechne alles via cluster_variability1
    if cutOff is None:
        cutOff, thr, m, s, term1, term2, term3, stdExpr, meanExpr = cluster_variability1(
            clustObj=clustObj,
            xedge=xedge,
            figFlag=figFlag,
            adjustValue=adjustValue,
            weightage=weightage
        )

    nTis = len(tisNames)
    nObjs = clustObj.Data.shape[0]
    geneTis = np.zeros((nObjs, nTis), dtype=bool)
    enzSel = np.zeros((nObjs, nTis), dtype=int)

    # 2) Pro-Cluster und Tissue: selektiere Objekte ≥ Schwellenwert
    for cid in range(1, clustObj.C.shape[0] + 1):
        mask_objs = (clustObj.cindex == cid)
        if not np.any(mask_objs):
            continue
        cluster_data = clustObj.Data[mask_objs, :]  # (#inCluster × nTis)
        sel = cluster_data >= thr[cid - 1]          # denselben Index wie thr verwenden
        idx_objs = np.where(mask_objs)[0]
        for ti in range(nTis):
            sel_idx = idx_objs[sel[:, ti]]
            geneTis[sel_idx, ti] = True
            enzSel[sel_idx, ti] = cid

        # 3) Cluster-spezifische Jaccard (wenn folderName gesetzt)
        if folderName:
            clustMat = sel.copy()  # Boolean (#inCluster × nTis)
            Jc = get_jaccard_sim_matrix(clustMat)
            os.makedirs(folderName, exist_ok=True)
            path = os.path.join(folderName, f"C{cid}.txt")
            np.savetxt(path, Jc, fmt="%0.2f", delimiter="\t")

    # 4) Setze enzTis als Kopie von geneTis vor Mapping
    enzTis = geneTis.copy()

    # 5) Globaler Jaccard über alle Tissues (für geneTis/enzTis)
    J_global = get_jaccard_sim_matrix(geneTis)
    if folderName:
        os.makedirs(folderName, exist_ok=True)
        np.savetxt(
            os.path.join(folderName, "Call.txt"),
            J_global,
            fmt="%0.2f",
            delimiter="\t"
        )

    # 6) Tissue-Clustering (Dendrogramm, optional figFlag)
    tisClust: Dict[str, Any] = {}
    # Distanzmatrix: 
    #    In MATLAB: Y = pdist(J, 'euclidean')
    #    pdist erwartet Beobachtungen als Zeilen; hier sind Zeilen Tissues
    #    Da J_global shape = (nTis × nTis), wir clustern anhand dieser Ähnlichkeit.
    #    Distance-Matrix fürs Clustering:
    Y = pdist(J_global, metric='euclidean')
    Z = linkage(Y, method='complete')
    leaf_order = optimal_leaf_ordering(Z, Y)
    if figFlag:
        # Plot Dendrogramm
        dendrogram(Z, labels=tisNames, orientation='left', no_plot=False, reorder=leaf_order)
    # Speichere im Dictionary
    tisClust["Z"] = Z
    tisClust["Y"] = Y
    tisClust["outperm"] = leaf_order
    tisClust["names"] = [tisNames[i] for i in leaf_order.tolist()]

    # 7) Mapping von Genen/Enzymen auf Reaktionen, falls model und objectMaps vorhanden
    if model is not None and hasattr(clustObj, "objectMaps"):
        # Für Gene-Fall (keine objectMaps): unique(model.genes) und gene-Indices
        if not hasattr(clustObj, "objectMaps") or clustObj.objectMaps is None:
            # Stelle sicher, dass model.genes einzigartig sind
            unique_genes = list(dict.fromkeys(model.genes))
            active_rxns = np.zeros((len(unique_genes), nTis), dtype=bool)
            for ti in range(nTis):
                selected_genes = np.where(geneTis[:, ti])[0]
                sel_names = [clustObj.objects[i] for i in selected_genes]
                # Finde Indizes in model.genes
                for g in sel_names:
                    if g in unique_genes:
                        gi = unique_genes.index(g)
                        active_rxns[gi, ti] = True
            geneTis = active_rxns.copy()
        else:
            # Enzym-Fall: objectMaps[i] ist Liste von Reaktionen pro Enzym-Index i
            rxn_list_per_tissue: List[List[Any]] = [[] for _ in range(nTis)]
            for ti in range(nTis):
                selected_objs = np.where(geneTis[:, ti])[0]
                for obj_idx in selected_objs:
                    maps = clustObj.objectMaps[obj_idx]
                    # Flatten maps, falls verschachtelt
                    flat, _ = linearization_index(
                        A = maps if isinstance(maps, list) else [maps],
                        flag = 'rows'
                    )
                    rxn_list_per_tissue[ti].extend(flat)
                # eindeutige Reaktionen
                rxn_list_per_tissue[ti] = list(set(rxn_list_per_tissue[ti]))
            # Erstelle Boolean-Matrix (#Rxns × #Tissues)
            all_rxns = model.rxns
            active_rxns = np.zeros((len(all_rxns), nTis), dtype=bool)
            for ti in range(nTis):
                for rxn in rxn_list_per_tissue[ti]:
                    if rxn in all_rxns:
                        ri = all_rxns.index(rxn)
                        active_rxns[ri, ti] = True
            geneTis = active_rxns.copy()

    # 8) Rückgabe aller Ergebnisse
    return geneTis, enzTis, cutOff, thr, enzSel, tisClust


def linearization_index(
    A: List[List[Any]],
    flag: str = 'rows'
) -> Tuple[List[Any], List[int]]:
    """
    Flatten a list-of-lists by one level and record original indices.

    Args:
        A:       List where A[i] is itself a list of items.
        flag:    Ignored in this Python port (orientation is irrelevant).

    Returns:
        B:       Flattened list of all items in A (one level deep).
        index:   For each item in B, the integer i such that it came from A[i].
    """
    B = []
    index = []
    for i, sub in enumerate(A):
        if isinstance(sub, list):
            for elem in sub:
                B.append(elem)
                index.append(i)
        else:
            B.append(sub)
            index.append(i)
    return B, index


def cluster_variability1(
    clustObj: ClustObj,
    xedge: np.ndarray,
    figFlag: bool,
    adjustValue: float,
    weightage: List[float]
) -> Tuple[
    np.ndarray,  # cutOff (nClust,)
    np.ndarray,  # thrval (nClust,)
    np.ndarray,  # m (nClust,)
    np.ndarray,  # s (nClust,)
    np.ndarray,  # term1 (nClust,)
    np.ndarray,  # term2 (nClust,)
    np.ndarray,  # term3 (nClust,)  (unused)
    np.ndarray,  # stdExpr (nObjs,)
    np.ndarray   # meanExpr (nObjs,)
]:
    """
    Port von MATLAB clusterVariability1:
    Berechnet pro Objekt (Gen/Enzym) Mittelwert, Std und Noise,
    dann pro Cluster standardisierte Metriken und pro-Cluster Cutoffs.

    Returns:
        cutOff    (nClust,) Prozentile [0–100]
        thrval    (nClust,) absolute Schwellwerte
        m         (nClust,) Mittelwert je Cluster
        s         (nClust,) Std je Cluster
        term1     (nClust,) normalisierte Std-Term
        term2     (nClust,) Mittelwert-Abweichung vom Overall-Mean
        term3     (nClust,) UNUSED placeholder (zeros)
        stdExpr   (nObjs,)  Std je Objekt
        meanExpr  (nObjs,)  Mean je Objekt
    """
    data_all = clustObj.Data.copy()
    data_flat = data_all[np.isfinite(data_all)]
    overall_mean = np.mean(data_flat) if data_flat.size else 0.0

    nObjs = clustObj.Data.shape[0]
    stdExpr = np.zeros(nObjs)
    meanExpr = np.zeros(nObjs)
    varExpr = np.zeros(nObjs)
    noiseExpr = np.zeros(nObjs)

    for i in range(nObjs):
        row = clustObj.Data[i, :]
        finite = row[np.isfinite(row)]
        if finite.size:
            meanExpr[i] = finite.mean()
            stdExpr[i] = finite.std(ddof=0)
            varExpr[i] = finite.var(ddof=0)
            noiseExpr[i] = stdExpr[i] / meanExpr[i] if meanExpr[i] != 0 else 0.0

    sigData = np.sqrt(np.mean(varExpr)) if varExpr.size else 0.0

    nClust = clustObj.C.shape[0]
    s = np.zeros(nClust)
    m = np.zeros(nClust)
    mu_sm = np.zeros(nClust)
    noi = np.zeros(nClust)

    for ci in range(1, nClust + 1):
        mask = (clustObj.cindex == ci)
        if not np.any(mask):
            continue
        s[ci - 1] = np.sqrt(np.mean(varExpr[mask]))
        m[ci - 1] = np.mean(meanExpr[mask])
        mu_sm[ci - 1] = (
            np.mean(stdExpr[mask] / meanExpr[mask])
            if np.all(meanExpr[mask] != 0) else 0.0
        )
        noi[ci - 1] = np.mean(noiseExpr[mask])

    diff = s - sigData
    denom = diff.max() if diff.max() != 0 else 1.0
    term1 = diff / denom
    term2 = m - overall_mean
    term3 = np.zeros_like(term1)

    cutOff = weightage[0] * term1 - weightage[1] * term2
    cutOff -= cutOff.min()
    cutOff = cutOff * (100.0 / cutOff.max()) if cutOff.max() != 0 else cutOff

    cx = np.argwhere(cutOff == 100.0).flatten()

    thrval = np.zeros(nClust)
    for i in range(nClust):
        mask = (clustObj.cindex == (i + 1))
        vals = clustObj.Data[mask, :].flatten()
        vals = vals[np.isfinite(vals)]
        if i in cx:
            thrval[i] = overall_mean
        else:
            adj_pct = cutOff[i]
            if adjustValue >= 0:
                adj_pct += (100.0 - adj_pct) * adjustValue
            else:
                adj_pct += adj_pct * adjustValue
            thrval[i] = np.percentile(vals, adj_pct) if vals.size else np.nan

    return cutOff, thrval, m, s, term1, term2, term3, stdExpr, meanExpr
