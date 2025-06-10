import cobra
import numpy as np
import pandas as pd
from typing import List, Optional, Union
from .data import ExpressionData

class ModelData:
    """
    Container for model-specific expression data.

    Attributes:
        gene (List[str]): Unique gene IDs present in both expression and model.
        value (np.ndarray): Aggregated expression values (genes x tissues).
        Tissue (List[str]): Names of tissues/conditions.
        genesymbol (Optional[List[str]]): Alternative gene symbols per unique gene.
        ID_geneMissing (List[str]): Genes in model but not in expression data.
        ID_genePresent (List[str]): Genes in model and expression data (pre-aggregation).
    """
    def __init__(
        self,
        gene: List[str],
        value: np.ndarray,
        Tissue: List[str],
        genesymbol: Optional[List[str]],
        ID_geneMissing: List[str],
        ID_genePresent: List[str],
    ):
        self.gene = gene
        self.value = value
        self.Tissue = Tissue
        self.genesymbol = genesymbol
        self.ID_geneMissing = ID_geneMissing
        self.ID_genePresent = ID_genePresent


def load_model(sbml_path: str) -> cobra.Model:
    """
    Lade ein SBML-Modell (.xml oder .sbml) und gib ein cobra.Model zurück.

    Args:
        sbml_path (str): Pfad zur SBML-Datei.

    Returns:
        cobra.Model: Geladenes Stoffwechselmodell.
    """
    return cobra.io.read_sbml_model(sbml_path)


def get_model_data(
    expr: Union[ExpressionData, pd.DataFrame],
    model: cobra.Model
) -> ModelData:
    """
    Portierung von MATLAB getModelData:
    Akzeptiert entweder ein ExpressionData-Objekt oder ein pd.DataFrame (genes x samples).
    Filtere Transcriptom-Daten auf Gene, die im Modell vorkommen,
    aggregiere Duplikate und bestimme fehlende/present Gene.
    """
    # Wenn DataFrame übergeben, in Variables umwandeln
    if isinstance(expr, pd.DataFrame):
        genes = expr.index.astype(str).tolist()
        values = expr.values
        tissue_names = expr.columns.astype(str).tolist()
        genesymbol = None
    else:
        genes = expr.gene
        values = expr.value
        tissue_names = expr.Tissue
        genesymbol = getattr(expr, 'genesymbol', None)

    # Modell-Gene-IDs extrahieren
    model_gene_ids = [g.id for g in model.genes]

    # Boolean-Maske: welche expr-Gene sind in model.genes?
    present_mask = [gene in model_gene_ids for gene in genes]

    # Listen der präsenten und fehlenden Gene
    genes_present = [g for g, ok in zip(genes, present_mask) if ok]
    ID_genePresent = genes_present.copy()
    ID_geneMissing = [g for g in model_gene_ids if g not in genes]

    # Werte der präsenten Gene
    values_present = values[np.array(present_mask), :]

    # Einmalige Gene und Mittelwert bei Duplikaten
    unique_genes, inv_idx = np.unique(genes_present, return_inverse=True)
    n_genes = unique_genes.shape[0]
    n_tissues = values_present.shape[1]
    aggregated = np.zeros((n_genes, n_tissues), dtype=values_present.dtype)
    for i, grp in enumerate(inv_idx):
        aggregated[grp] += values_present[i]
    counts = np.bincount(inv_idx)
    aggregated = aggregated / counts[:, None]

    # Alternative Symbolnamen pro einzigartigem Gen
    genesymbol_unique = None
    if genesymbol is not None:
        symbol_map = {g: s for g, s in zip(genes, genesymbol)}
        genesymbol_unique = [symbol_map.get(g) for g in unique_genes]

    return ModelData(
        gene=unique_genes.tolist(),
        value=aggregated,
        Tissue=tissue_names,
        genesymbol=genesymbol_unique,
        ID_geneMissing=ID_geneMissing,
        ID_genePresent=ID_genePresent,
    )
