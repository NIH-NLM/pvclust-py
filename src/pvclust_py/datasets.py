"""Small datasets shipped with the package, so an install has something to run.

ORIENTATION MATTERS AND IS EASY TO GET BACKWARDS. Everything here returns matrices
with **rows = resampling units** (samples/patients) and **columns = the objects
clustered** (genes/proteins), because that is the orientation the bootstrap assumes:
rows must be things you could have drawn more of.

Expression matrices usually arrive the other way round -- genes as rows -- so they
need transposing first. ``lung_expression`` is already transposed for you.
"""
from __future__ import annotations

from pathlib import Path

_DATA = Path(__file__).parent / "data"


def load_lung():
    """Lung tumour expression: 73 samples x 150 most-variable genes.

    From the ``lung`` dataset shipped with R's pvclust (Garber et al.), transposed so
    that **rows are samples** and **columns are genes**. Clustering the columns groups
    genes, resampling the 73 samples -- the statistically sound direction.

    Note:
        R's own ``pvclust(lung)`` example does the opposite: it clusters the 73
        SAMPLES by resampling genes. That is fine as a demonstration of the software
        and is what the parity fixtures reproduce, but it is the anti-conservative
        orientation -- genes are co-expressed rather than independent draws.
    """
    import pandas as pd
    return pd.read_csv(_DATA / "lung_expression.csv", index_col=0,
                       float_precision="round_trip")


def load_cohorts(n_cohorts: int = 3, seed: int = 0, proportions=None):
    """The lung data split by rows into disjoint cohorts, for the federation demos.

    Splits **samples**, not genes -- which is the real situation and the only one that
    federates: cohorts hold different patients but measure the same analytes. Splitting
    the other axis would mean cohorts holding different genes for shared patients,
    which is not the problem this package solves.

    Args:
        proportions: relative cohort sizes. Defaults to UNEQUAL sizes, because equal
            cohorts hide the scale question -- each project's r is relative to its own
            n, so cohorts of different sizes contribute measurements at genuinely
            different scales (see :func:`pvclust_py.aggregate.pool_counts`).
    """
    import numpy as np
    df = load_lung()
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(df))

    if proportions is None and n_cohorts <= 3:
        proportions = (0.55, 0.28, 0.17)[:n_cohorts]
    if proportions is None:
        parts = np.array_split(perm, n_cohorts)
    else:
        w = np.array(proportions, dtype=float); w = w / w.sum()
        parts = np.split(perm, np.cumsum((w * len(df)).astype(int))[:-1])
    return {f"cohort_{chr(65 + i)}": df.iloc[sorted(idx)] for i, idx in enumerate(parts)}
