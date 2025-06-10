import pandas as pd

def load_expression_csv_as_df(path: str) -> pd.DataFrame:
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