import numpy as np
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering, cophenet, fcluster, dendrogram
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Dict, Any
from .enzymes import EnzymeData


class ClustObj:
    """
    Container for clustering results.

    Attributes:
        Distribution (np.ndarray): Histogram-based features (enzymes x bins).
        objects (List[str]): Names of enzymes clustered.
        altObjects (Optional[List[str]]): Not used for EnzymeData.
        objectMaps (List[List[str]]): Reaction lists for each enzyme.
        Data (np.ndarray): Log-transformed expression matrix (enzymes x tissues).
        cindex (np.ndarray): Cluster labels for each enzyme (1..k).
        C (np.ndarray): Cluster centroids (k x bins).
        numObjs (np.ndarray): Number of tissues in which each enzyme is expressed.
        numObjInClust (np.ndarray): Number of enzymes in each of the k clusters.
    """
    def __init__(
        self,
        Distribution: np.ndarray,
        objects: List[str],
        altObjects: Optional[List[str]],
        objectMaps: List[List[str]],
        Data: np.ndarray,
        cindex: np.ndarray,
        C: np.ndarray,
        numObjs: np.ndarray,
        numObjInClust: np.ndarray,
    ):
        self.Distribution = Distribution
        self.objects = objects
        self.altObjects = altObjects
        self.objectMaps = objectMaps
        self.Data = Data
        self.cindex = cindex
        self.C = C
        self.numObjs = numObjs
        self.numObjInClust = numObjInClust


def gene_expr_dist_hierarchy(
    enzyme_data: EnzymeData,
    remove_objects: Optional[List[str]],
    edgeX: np.ndarray,
    k: int,
    dist_method: str = 'euclidean',
    linkage_method: str = 'complete',
    plot: bool = True
) -> Tuple[ClustObj, np.ndarray, Dict[str, Any]]:
    """
    Perform hierarchical clustering on enzyme expression distributions.

    Binned histogram features per enzyme werden über Tissues berechnet,
    dann mit SciPy pdist, linkage und optimal_leaf_ordering geclustert.

    Args:
        enzyme_data (EnzymeData): Input with .enzyme, .value, .Tissue, .rxns
        remove_objects (List[str] or None): Enzymes to exclude before clustering.
        edgeX (np.ndarray): Bin edges for histogramming log10-values.
        k (int): Number of clusters.
        dist_method (str): Distance metric for pdist (z.B. 'euclidean').
        linkage_method (str): Linkage method (z.B. 'complete').
        plot (bool): If True, erzeugt einen Dendrogram-Plot.

    Returns:
        clustObj (ClustObj): Struktur mit allen Cluster-Ergebnissen.
        Z_ord (np.ndarray): Geordnete Linkage-Matrix.
        dendro (dict): Ergebnis von dendrogram mit Knoten-Layout.
    """
    # 1) Datenfilterung
    X = enzyme_data.value.copy()  # (n_enzymes, n_tissues)
    objs = np.array(enzyme_data.enzyme)

    # Entferne explizit gewünschte Enzyme
    if remove_objects:
        mask = ~np.isin(objs, remove_objects)
        objs = objs[mask]
        X = X[mask, :]
        maps = [m for m, keep in zip(enzyme_data.rxns, mask) if keep]
    else:
        maps = enzyme_data.rxns

    # Entferne Enzyme ohne Expression in allen Tissues
    nonzero = X.sum(axis=1) > 0
    objs = objs[nonzero]
    X = X[nonzero, :]
    maps = [m for m, nz in zip(maps, nonzero) if nz]

    n_objs, n_tissues = X.shape

    # 2) Log10-Transformation (mit eps)
    with np.errstate(divide='ignore'):
        V = np.log10(X + np.finfo(float).eps)

    # 3) Histogramm-Merkmale
    n_bins = len(edgeX) - 1
    Yhist = np.zeros((n_objs, n_bins), dtype=float)
    for i in range(n_objs):
        vals = V[i, :]
        vals = vals[np.isfinite(vals)]
        counts, _ = np.histogram(vals, bins=edgeX)
        Yhist[i, :] = counts / n_tissues

    # 4) Distanzmatrix & Linkage
    Y = pdist(Yhist, metric=dist_method)
    Z = linkage(Y, method=linkage_method)
    Z_ord = optimal_leaf_ordering(Z, Y)
    coph_corr, _ = cophenet(Z_ord, Y)
    print(f"Cophenetic correlation (method={linkage_method}, metric={dist_method}): {coph_corr:.4f}")

    # 5) Cluster-Zuweisung
    labels = fcluster(Z_ord, t=k, criterion='maxclust')

    # 6) Centroid-Berechnung & Cluster-Größen
    C = np.zeros((k, n_bins), dtype=float)
    num_in_clust = np.zeros(k, dtype=int)
    for ci in range(1, k + 1):
        idx = labels == ci
        num_in_clust[ci-1] = np.count_nonzero(idx)
        if num_in_clust[ci-1] > 0:
            C[ci-1, :] = Yhist[idx, :].mean(axis=0)

    # 7) Objekt-Zählung pro Tissue
    num_objs = (X > 0).sum(axis=1)

    # 8) ClustObj befüllen
    clustObj = ClustObj(
        Distribution=Yhist,
        objects=objs.tolist(),
        altObjects=None,
        objectMaps=maps,
        Data=V,
        cindex=labels,
        C=C,
        numObjs=num_objs,
        numObjInClust=num_in_clust
    )

    # 9) Dendrogramm (optional geplottet)
    dendro = dendrogram(Z_ord, orientation='left', no_plot=not plot)
    if plot:
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Distance')
        plt.ylabel('Enzymes')
        plt.tight_layout()

    return clustObj, Z_ord, dendro
