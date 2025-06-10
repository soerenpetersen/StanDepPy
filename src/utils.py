# plot_utils.py
"""
Utility for visualising reaction‑level expression distributions.

Functions
---------
plot_reaction_distribution(rxn_df, outfile, *, bins="auto", log10=False,
                          include_zeros=False)
    Create and save a histogram (line plot) of all numeric values in
    *rxn_df*.
"""
from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

__all__ = [
    "plot_reaction_distribution",
]

# -----------------------------------------------------------------------------
# Helper
# -----------------------------------------------------------------------------

def _flatten_numeric(df: pd.DataFrame) -> np.ndarray:
    """Return 1‑D array of finite numeric values from *df*."""
    values = df.select_dtypes(include=[np.number]).to_numpy().ravel()
    return values[np.isfinite(values)]

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def plot_reaction_distribution(
    rxn_df: pd.DataFrame,
    outfile: str,
    *,
    bins: Union[int, str] = "auto",
    log10: bool = False,
    include_zeros: bool = False,
) -> None:
    """Plot the value distribution of *rxn_df* and save to *outfile*.

    Parameters
    ----------
    rxn_df : pd.DataFrame
        Reaction × sample expression matrix.
    outfile : str
        Path where the figure will be saved (format inferred from extension).
    bins : int or "auto", default "auto"
        Histogram binning (passed to :pyfunc:`numpy.histogram`).
    log10 : bool, default False
        When *True*, apply log10 transform *before* plotting (values ≤0 are
        removed automatically).
    include_zeros : bool, default False
        If *False*, 0‑Werte werden vor dem Plotten entfernt (nur relevant,
        wenn *log10=False*).
    """
    values = _flatten_numeric(rxn_df)
    if values.size == 0:
        raise ValueError("No numeric values found in rxn_df.")

    # Remove zeros if desired (linear scale only) -------------------------
    if not include_zeros and not log10:
        values = values[values != 0]

    # Log‑transform -------------------------------------------------------
    if log10:
        values = values[values > 0]  # drop ≤0
        with np.errstate(divide="ignore"):
            values = np.log10(values)

    # Histogram -----------------------------------------------------------
    counts, bin_edges = np.histogram(values, bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    fig, ax = plt.subplots()
    ax.plot(bin_centers, counts, drawstyle="steps-mid")

    ax.set_xlabel("log10(value)" if log10 else "value")
    ax.set_ylabel("frequency")
    ax.set_title("Reaction activity distribution")

    fig.tight_layout()
    fig.savefig(outfile)
    plt.close(fig)
