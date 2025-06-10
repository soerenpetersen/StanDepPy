import os
from scipy import sparse
from scipy.io import mmwrite
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
import pandas as pd
import numpy as np



def extractDataFromRds(file_path):
    """
    Extracts data from an RDS file and converts it to a pandas DataFrame.
    Args:
        file_path (str): Path to the RDS file.
    Returns:
        pd.DataFrame: DataFrame containing the data from the RDS file.
    """

    pandas2ri.activate()

    # Read the RDS file
    rds_obj = ro.r['readRDS'](file_path)

    # Extract the RNA assay from the RDS object
    assays = ro.r['slot'](rds_obj, 'assays')
    rna_assay = assays.rx2('RNA')
    counts_matrix = ro.r['slot'](rna_assay, 'counts')

    # Extract the data
    data_matrix = ro.r['as.matrix'](counts_matrix)
    rownames = list(ro.r['rownames'](counts_matrix))
    colnames = list(ro.r['colnames'](counts_matrix))

    df = pd.DataFrame(data_matrix, index=rownames, columns=colnames, dtype=np.uint32)

    # Try to extract the meta data
    meta_df = None
    try:
        r_meta   = ro.r['slot'](rds_obj, 'meta.data')
        meta_df  = pandas2ri.rpy2py(r_meta)
        meta_df.index = colnames
    except Exception:
        pass

    return df, meta_df

def save_core_files(df, meta_df=None, out_dir="data/core_files"):
    """
    Save the core files in the specified directory.
    Args:
        df (pd.DataFrame): DataFrame to save.
        out_dir (str): Directory to save the files.
    """

    os.makedirs(out_dir, exist_ok=True)

    # Save sparse matrix as .mtx file
    mtx = sparse.csr_matrix(df.values)
    mmwrite(os.path.join(out_dir, "count_matrix.mtx"), mtx)

    # Save row and column names and meta data as .csv files
    pd.Series(df.index)\
      .to_csv(os.path.join(out_dir, "rownames.csv"),
              index=False, header=False)
    if meta_df is not None:
        meta_df = meta_df.copy()
        meta_df.insert(0, "cell", meta_df.index)
        meta_df.to_csv(os.path.join(out_dir, "colnames.csv"),
                       index=False)
    else:
        pd.Series(df.columns)\
          .to_csv(os.path.join(out_dir, "colnames.csv"),
                  index=False, header=False)

    
