"""
Adapter for SomaLogic SomaScan output.

One vendor's file layout, kept out of :mod:`pvclust_py.io` so the generic path stays
generic. Nothing else in the package imports this module; its only job is to produce
the same labelled matrix :func:`pvclust_py.io.read_matrix` produces, after which the
two are indistinguishable.

THE LAYOUT, which is positional rather than labelled::

    RFU_<panel>_<stage>.txt   headerless, tab-separated, n_samples x n_somamers
    sample_<panel>.txt        one row per sample, WITH a header
    somamer_<panel>.txt       one row per SOMAmer, WITH a header

The RFU file carries no labels at all: row i is ``sample_<panel>`` row i, column j is
``somamer_<panel>`` row j. Nothing enforces that, so :func:`read_somascan` checks the
shapes agree and refuses rather than silently attaching labels to the wrong
measurements.

**If your SomaScan data arrived as a plain labelled CSV** -- as public depositions
usually do -- you do not need this module. Use :func:`pvclust_py.io.read_matrix` with
``feature_map=`` and skip past it entirely.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .io import _unique_labels

#: The SomaScan normalisation cascade, in the order it is applied. Useful for batch
#: diagnostics: cluster at each stage and watch plate structure recede.
STAGES = ("raw", "hyb", "hyb.msnCal.ps", "hyb.msnCal.ps.cal", "hyb.msnCal.ps.cal.msnAll")


#: Sample-annotation columns that are technical rather than biological. Pass these as
#: ``technical=`` to :func:`pvclust_py.diagnostics.report` so it knows which
#: associations mean "batch effect" rather than "finding".
TECHNICAL_COLUMNS = ("PlateId", "PlateRunDate", "ScannerID", "PlatePosition",
                     "SlideId", "Subarray", "Batch", "RowCheck", "SampleType",
                     "PercentDilution")

#: SOMAmer annotation columns usable as column labels. SeqId is the only unique one;
#: the rest collide (in the 7K panel 904 SOMAmers share a gene symbol, because several
#: proteins are measured by more than one reagent).
LABEL_COLUMNS = ("SeqId", "Target", "TargetFullName", "UniProt", "EntrezGeneSymbol")


def read_somascan(rfu, samples, somamers, *,
                  sample_id: str = "SampleId",
                  somamer_id: str = "SeqId",
                  sample_type: Optional[str] = "Sample",
                  log2: bool = True) -> tuple:
    """Read the SomaScan three-file layout into one labelled matrix.

    Args:
        rfu: the headerless RFU matrix (samples x SOMAmers).
        samples: sample annotation, one row per RFU row.
        somamers: SOMAmer annotation, one row per RFU column.
        sample_id: sample annotation column to label rows with. Not unique in
            practice (replicates share an id), so a positional suffix is appended
            where needed rather than silently collapsing rows.
        somamer_id: SOMAmer annotation column to label columns with -- one of
            :data:`LABEL_COLUMNS`. ``SeqId`` (default) is unique; the protein-name
            columns are far more readable but collide, so duplicates get the SeqId
            appended, e.g. ``ABL2 (5261-13)`` and ``ABL2 (3342-76)``. Those pairs are
            the same protein measured by two reagents, and they will cluster together
            with very high AU -- a useful built-in positive control, and not a
            biological finding.
        sample_type: keep only rows with this ``SampleType``; ``None`` keeps all.
            Buffer wells are excluded by default -- they are process controls, not
            biology, and would otherwise form their own confident cluster.
        log2: take log2 of the RFU. RFU is strongly right-skewed and correlation on
            the raw scale is dominated by a handful of abundant analytes; log2 is
            standard for SomaScan and is on by default.

    Returns:
        ``(matrix, sample_annotation, somamer_annotation)`` -- the DataFrame ready to
        cluster, plus both annotation tables aligned to it (the sample table carries
        PlateId / ScannerID / Batch for the batch diagnostics).
    """
    X = pd.read_csv(rfu, sep="\t", header=None, float_precision="round_trip")
    smp = pd.read_csv(samples, sep="\t")
    som = pd.read_csv(somamers, sep="\t")

    if X.shape != (len(smp), len(som)):
        raise ValueError(
            f"shapes disagree: RFU is {X.shape[0]}x{X.shape[1]} but there are "
            f"{len(smp)} sample rows and {len(som)} somamer rows. These files are "
            f"aligned by POSITION, so a mismatch means labels would be attached to "
            f"the wrong measurements.")

    for col, tbl, what in ((sample_id, smp, "sample"), (somamer_id, som, "somamer")):
        if col not in tbl.columns:
            raise ValueError(f"{what} annotation has no {col!r} column; "
                             f"available: {list(tbl.columns)}")
    if somamer_id != "SeqId" and "SeqId" not in som.columns:
        raise ValueError("somamer annotation needs a SeqId column to disambiguate "
                         f"duplicate {somamer_id} values")

    labels = smp[sample_id].astype(str)
    if labels.duplicated().any():
        # replicates legitimately share a SampleId; keep them distinguishable
        labels = labels + "_" + np.arange(len(labels)).astype(str)
    X.index = labels
    if somamer_id == "SeqId":
        X.columns = som["SeqId"].astype(str)
    else:
        X.columns = _unique_labels(som[somamer_id], som["SeqId"], somamer_id)

    smp = smp.set_index(X.index)
    if sample_type is not None:
        if "SampleType" not in smp.columns:
            raise ValueError("no SampleType column to filter on; pass sample_type=None")
        keep = smp["SampleType"] == sample_type
        X, smp = X.loc[keep.values], smp.loc[keep.values]

    if log2:
        if (X <= 0).any().any():
            raise ValueError("non-positive RFU values cannot be log2-transformed; "
                             "pass log2=False and handle them explicitly")
        X = np.log2(X)

    som = som.copy()
    som.insert(0, "label", X.columns)
    return X, smp, som.set_index("label")


