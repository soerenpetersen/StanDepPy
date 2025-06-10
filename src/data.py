import os
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread, mmwrite
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri


class ExpressionData:
    """
    Container for transcriptomic data.

    Attributes:
        gene (List[str]): List of gene identifiers.
        value (np.ndarray): Expression values matrix (genes x samples).
        Tissue (List[str]): Names of samples or pseudobulk groups.
        genesymbol (Optional[List[str]]): Alternative gene symbols.
        cell_types (Optional[List[str]]): Cell type annotation per sample.
        meta (Optional[pd.DataFrame]): Full metadata DataFrame if provided.
    """
    def __init__(
        self,
        gene: list,
        value: np.ndarray,
        Tissue: list,
        genesymbol: list = None,
        cell_types: list = None,
        meta: pd.DataFrame = None,
    ):
        self.gene = gene
        self.value = value
        self.Tissue = Tissue
        self.genesymbol = genesymbol
        self.cell_types = cell_types
        self.meta = meta


def extract_data_from_rds(file_path: str):
    """
    Extract ExpressionData and optional metadata DataFrame from a Seurat .rds file.

    Returns:
        expr (ExpressionData): Loaded expression data (without cell_types).
        meta_df (pd.DataFrame): Cell metadata with cell barcodes as index.
    """
    pandas2ri.activate()

    rds_obj = ro.r['readRDS'](file_path)
    assays = ro.r['slot'](rds_obj, 'assays')
    rna_assay = assays.rx2('RNA')
    counts_matrix = ro.r['slot'](rna_assay, 'counts')

    data_matrix = ro.r['as.matrix'](counts_matrix)
    values = np.array(data_matrix, dtype=np.uint32)
    rownames = list(ro.r['rownames'](counts_matrix))
    colnames = list(ro.r['colnames'](counts_matrix))

    expr = ExpressionData(
        gene=rownames,
        value=values,
        Tissue=colnames
    )

    meta_df = None
    try:
        r_meta = ro.r['slot'](rds_obj, 'meta.data')
        meta_df = pandas2ri.rpy2py(r_meta)
        meta_df.index = colnames
    except Exception:
        pass

    return expr, meta_df


def data_to_df(counts_path: str, genes_path: str) -> (np.ndarray, list, list):
    """
    Read count matrix (MTX) and gene list into numpy arrays and gene names.

    Returns:
        values (np.ndarray): Raw counts array.
        genes (list): Gene identifiers.
        _ (ignored): placeholder for samples.
    """
    counts = mmread(counts_path)
    if sparse.issparse(counts):
        counts = counts.toarray()

    with open(genes_path, 'r') as f:
        genes = [line.strip() for line in f if line.strip()]

    n_genes, n_samples = counts.shape
    if len(genes) != n_genes:
        raise ValueError(f"Number of genes ({len(genes)}) != rows in matrix ({n_genes})")

    return counts, genes, None


def load_expression_data(
    counts_path: str,
    genes_path: str,
    samples_path: str,
    cell_type_column: int = None
) -> ExpressionData:
    """
    Load counts, gene list and (optional) metadata, return ExpressionData.

    If cell_type_column is None, samples_path is a simple list of sample IDs.
    Otherwise it's a CSV with a header (or without), where:
      - col 0 = sample ID
      - col `cell_type_column` = cell type annotation
    """
    # --- 1) Counts und Gene laden ---
    counts = mmread(counts_path)
    if sparse.issparse(counts):
        counts = counts.toarray()
    with open(genes_path, 'r') as f:
        genes = [l.strip() for l in f if l.strip()]
    if counts.shape[0] != len(genes):
        raise ValueError(f"Genes ({len(genes)}) != matrix rows ({counts.shape[0]})")

    # --- 2) Metadaten oder Sample-Liste ---
    cell_types = None
    meta_df = None
    if cell_type_column is None:
        # einfache Sample-Liste
        with open(samples_path, 'r') as f:
            samples = [l.strip() for l in f if l.strip()]
        if counts.shape[1] != len(samples):
            raise ValueError(f"Samples ({len(samples)}) != matrix cols ({counts.shape[1]})")
        tissue_names = samples
    else:
        # Metadaten-CSV mit oder ohne Header einlesen
        try:
            meta_df = pd.read_csv(samples_path, header=0)
        except Exception:
            meta_df = pd.read_csv(samples_path, header=None)
        if meta_df.shape[0] != counts.shape[1]:
            # nochmal ohne Header probieren
            meta_df = pd.read_csv(samples_path, header=None)
        if meta_df.shape[0] != counts.shape[1]:
            raise ValueError(f"Metadata rows ({meta_df.shape[0]}) != matrix cols ({counts.shape[1]})")
        sample_ids = meta_df.iloc[:, 0].astype(str).tolist()
        cell_types = meta_df.iloc[:, cell_type_column].astype(str).tolist()
        tissue_names = sample_ids

    # --- 3) ExpressionData erzeugen ---
    expr = ExpressionData(
        gene=genes,
        value=counts,
        Tissue=tissue_names,
        genesymbol=None,
        cell_types=cell_types,
        meta=meta_df
    )
    return expr



def save_core_files(
    expr: ExpressionData,
    out_dir: str = "data/core_files"
):
    """
    Save count matrix, rownames, colnames and (if present) metadata to disk.
    """
    os.makedirs(out_dir, exist_ok=True)

    df = pd.DataFrame(expr.value, index=expr.gene, columns=expr.Tissue)
    mtx = sparse.csr_matrix(df.values)
    mmwrite(os.path.join(out_dir, "count_matrix.mtx"), mtx)

    df.index.to_series().to_csv(
        os.path.join(out_dir, "rownames.csv"), index=False, header=False
    )
    if expr.meta is not None:
        # ensure 'cell' column first
        df_meta = expr.meta.copy()
        if df_meta.columns[0].lower() != 'cell':
            df_meta.insert(0, 'cell', df_meta.iloc[:, 0])
        df_meta.to_csv(
            os.path.join(out_dir, "colnames.csv"), index=False
        )
    else:
        pd.Series(expr.Tissue).to_csv(
            os.path.join(out_dir, "colnames.csv"), index=False, header=False
        )

def load_expression_csv(path: str) -> pd.DataFrame:
    """
    Read a gene-by-sample CSV and clean quotes.

    Returns
    -------
    pd.DataFrame  (rows = genes, cols = samples)
    """
    df = pd.read_csv(path, index_col=0)
    # strip potential leading/trailing quotes in index & columns
    df.index   = df.index.astype(str).str.strip('"')
    df.columns = df.columns.astype(str).str.strip('"')
    return df