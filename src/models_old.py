# standep_py/models.py

import os
import numpy as np
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering, cophenet, dendrogram
from typing import Optional, Tuple, List, Dict, Any
from cobra import Model
from .clustering import ClustObj

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
        # sub must be iterable (list of elements)
        for elem in sub:
            B.append(elem)
            index.append(i)
    return B, index


def cluster_variability1(
    clustObj: ClustObj,
    xedge: np.ndarray,
    figFlag: bool,
    adjustValue: float,
    weightage: List[float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Port of MATLAB clusterVariability1:
    Computes per-cluster percentile cutoffs, absolute thresholds, and
    several variability metrics used by models4mClusters1.

    Returns:
        cutOff    (nClust,) percentiles [0–100]
        thrval    (nClust,) threshold in log10 units
        m         (nClust,) mean(log10) per cluster
        s         (nClust,) std(log10) per cluster
        term1     (nClust,) normalized std deviation term
        term2     (nClust,) mean deviation term
        term3     UNUSED placeholder (zeros)
        stdExpr   (nObjs,)  std(log10) per object
        meanExpr  (nObjs,)  mean(log10) per object
    """
    data = clustObj.Data.copy()
    data = data[np.isfinite(data)]
    overall_mean = np.mean(data)
    # per-object statistics
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
    # overall std of cluster-means
    sigData = np.sqrt(np.mean(varExpr))

    nClust = clustObj.C.shape[0]
    s = np.zeros(nClust)
    m = np.zeros(nClust)
    mu_sm = np.zeros(nClust)
    noi = np.zeros(nClust)

    for ci in range(1, nClust+1):
        mask = clustObj.cindex == ci
        s[ci-1] = np.sqrt(np.mean(varExpr[mask]))
        m[ci-1] = np.mean(meanExpr[mask])
        # average noise per cluster
        mu_sm[ci-1] = np.mean(
            stdExpr[mask] / meanExpr[mask]
            if np.all(meanExpr[mask] != 0) else 0.0
        )
        noi[ci-1] = np.mean(noiseExpr[mask])

    # term1 = (s - sigData) / max(s - sigData)
    diff = s - sigData
    denom = diff.max() if diff.max() != 0 else 1.0
    term1 = diff / denom
    # term2 = m - overall_mean
    term2 = m - overall_mean
    term3 = np.zeros_like(term1)  # unused in MATLAB port

    # combine and scale to [0,100]
    cutOff = weightage[0]*term1 - weightage[1]*term2
    cutOff -= cutOff.min()
    cutOff *= (100.0 / cutOff.max()) if cutOff.max() != 0 else 0.0

    # determine which cluster gets 100% cutoff
    cx = np.argwhere(cutOff == 100.0).flatten()
    # flatten all cluster Data values for percentile
    thrval = np.zeros(nClust)
    for i in range(nClust):
        mask = clustObj.cindex == (i+1)
        vals = clustObj.Data[mask, :].flatten()
        vals = vals[np.isfinite(vals)]
        if i in cx:
            thrval[i] = overall_mean
        else:
            adj = cutOff[i]
            if adjustValue >= 0:
                adj += (100.0 - adj) * adjustValue
            else:
                adj += adj * adjustValue
            thrval[i] = np.percentile(vals, adj) if vals.size else np.nan

    return cutOff, thrval, m, s, term1, term2, term3, stdExpr, meanExpr


def get_jaccard_sim_matrix(
    A: np.ndarray
) -> np.ndarray:
    """
    Compute Jaccard similarity between columns of a boolean matrix A
    in O(n_cols^2) time and O(n_cols*n_rows) memory.

    Args:
        A: Boolean array, shape (n_items, n_sets)

    Returns:
        J: float array, shape (n_sets, n_sets)
    """
    n = A.shape[1]
    J = np.zeros((n, n), dtype=float)
    for i in range(n):
        col_i = A[:, i]
        for j in range(i, n):
            col_j = A[:, j]
            inter = np.logical_and(col_i, col_j).sum()
            union = np.logical_or(col_i, col_j).sum()
            J[i, j] = J[j, i] = inter / union if union > 0 else 0.0
    return J


def models4m_clusters1(
    clustObj: ClustObj,
    tisNames: List[str],
    model: Model,
    xedge: np.ndarray,
    folderName: Optional[str] = None,
    cutOff: Optional[np.ndarray] = None,
    figFlag: bool = False,
    adjustValue: float = 0.0,
    weightage: List[float] = [1.0, 1.0]
) -> Tuple[
    np.ndarray,      # geneTis (or activeRxns if model given)
    np.ndarray,      # enzTis
    np.ndarray,      # cutOff
    np.ndarray,      # thr
    np.ndarray,      # enzSel
    Dict[str, Any]   # tisClust
]:
    """
    Python port of MATLAB models4mClusters1:
    Determines for each tissue which genes/enzymes (and reactions) are 'active'.

    Returns:
        geneTis (bool array): n_objects × n_tissues mask of selected objects
        enzTis (bool array): identical to geneTis for enzymes
        cutOff (float[]): percentile cutoffs per cluster
        thr (float[]):     threshold in log10 units per cluster
        enzSel (int[]):    cluster ID assigned per object × tissue
        tisClust (dict):   tissue-clustering results if figFlag is True
    """
    # 1) Threshold calculation
    if cutOff is None:
        cutOff, thr, *_ = cluster_variability1(
            clustObj, xedge, figFlag, adjustValue, weightage
        )
    else:
        _, thr, *_ = cluster_variability1(
            clustObj, xedge, figFlag, adjustValue, weightage
        )

    nObjs = len(clustObj.objects)
    nTis = len(tisNames)

    geneTis = np.zeros((nObjs, nTis), dtype=bool)
    enzSel = np.zeros((nObjs, nTis), dtype=int)

    # 2) Per-cluster selection
    for i in range(clustObj.C.shape[0]):
        cid = i + 1
        mask_objs = (clustObj.cindex == cid)
        cluster_data = clustObj.Data[mask_objs, :]  # shape (#inCluster, nTis)
        # apply threshold
        sel = cluster_data >= thr[i]
        idx_objs = np.where(mask_objs)[0]
        for ti in range(nTis):
            sel_idx = idx_objs[sel[:, ti]]
            geneTis[sel_idx, ti] = True
            enzSel[sel_idx, ti] = cid

        # 3) Cluster-specific Jaccard (optional)
        if folderName:
            clustMat = sel.copy()
            Jc = get_jaccard_sim_matrix(clustMat)
            os.makedirs(folderName, exist_ok=True)
            np.savetxt(
                os.path.join(folderName, f"C{cid}.txt"),
                Jc, fmt="%0.2f", delimiter="\t"
            )

    # copy for enzyme output
    enzTis = geneTis.copy()

    # 4) Global tissue Jaccard & optional write
    J = get_jaccard_sim_matrix(geneTis)
    if folderName:
        np.savetxt(
            os.path.join(folderName, "Call.txt"),
            J, fmt="%0.2f", delimiter="\t"
        )

    # 5) Tissue clustering (if requested)
    tisClust: Dict[str, Any] = {}
    if figFlag:
        Zt = linkage(pdist(J), method='complete')
        leaf = optimal_leaf_ordering(Zt, pdist(J))
        dendro = dendrogram(
            Zt, labels=tisNames, orientation='left',
            no_plot=True, reorder=leaf
        )
        tisClust = {
            'Z': Zt,
            'outperm': leaf,
            'dendrogram': dendro,
            'names': tisNames
        }

    # 6) Map selected enzymes → reactions if model given
    if model is not None and hasattr(clustObj, 'objectMaps'):
        rxn_ids = [r.id for r in model.reactions]
        activeRxns = np.zeros((len(rxn_ids), nTis), dtype=bool)
        for ti in range(nTis):
            selected = np.where(geneTis[:, ti])[0]
            rxns = []
            for obj in selected:
                maps = clustObj.objectMaps[obj]
                # flatten lists
                if isinstance(maps, list):
                    rxns.extend(maps)
                else:
                    rxns.append(maps)
            for rid in set(rxns):
                if rid in rxn_ids:
                    activeRxns[rxn_ids.index(rid), ti] = True
        geneTis = activeRxns

    return geneTis, enzTis, cutOff, thr, enzSel, tisClust
