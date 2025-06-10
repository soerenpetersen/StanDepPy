import cobra as cb
import numpy as np
import pandas as pd
from scipy.io import mmread


def dataToDf(counts_path, genes_path, samples_path):
    """
    Converts a counts matrix and gene/sample names into a pandas DataFrame.
    
    Parameters:
    - counts: The counts matrix (sparse or dense).
    - genes: A list or Series of gene names.
    - samples: A list or Series of sample names.
    
    Returns:
    - A pandas DataFrame with genes as rows and samples as columns.
    """

    counts = mmread(counts_path)

    # Convert sparse matrix to dense if necessary
    if hasattr(counts, 'toarray'):
        counts = counts.toarray()
    
    genes = pd.read_csv(genes_path, header=None, squeeze=True)
    samples = pd.read_csv(samples_path, header=None, squeeze=True)

    #if genes.is_numeric:
    #    genes = genes.astype(str) # ???? testen + functioniert nicht

    # Create DataFrame
    df = pd.DataFrame(counts, index=genes, columns=samples)
    
    return df


def getModelData(model_path, counts_path, genes_path, samples_path):
    """
    Reads a model and a counts matrix and returns a DataFrame with the counts data.
    
    Parameters:
    - model_path: Path to the metabolic model.
    - counts_path: Path to the counts matrix file.
    - genes_path: Path to the genes file.
    - samples_path: Path to the samples file.
    
    Returns:
    - A pandas DataFrame with genes as rows and samples as columns.
    """
    # Read the model
    mod = cb.io.read_sbml_model(model_path)
    
    # Read the counts matrix and convert it to a DataFrame
    df = dataToDf(counts_path, genes_path, samples_path)
    
    return df