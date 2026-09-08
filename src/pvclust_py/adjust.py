"""
Removing batch effects before clustering -- without removing the biology with them.

Clustering will happily reproduce the plate a sample was run on, and AU will call
that cluster highly supported, because a batch effect *is* reproducible. So batch
often has to be removed first.

THE TRAP
--------
The obvious move -- subtract each batch's mean -- is wrong whenever batch is
correlated with the variable you care about. In the SLE dataset batch A is 83% SLE
and batch B is 56% SLE, so batch A's mean is shifted toward SLE simply because of who
is in it. Subtracting that mean removes the batch effect AND a share of the disease
signal, and the correction quietly destroys the thing you were looking for.

The fix is to fit batch and the protected variables TOGETHER, then subtract only the
batch part. The protected variable's contribution is estimated and left alone, so a
group difference survives even when the groups are unevenly distributed across
batches.

    adjust_batch(X, batch)                  # naive -- only safe if batch is balanced
    adjust_batch(X, batch, protect=group)   # what you almost always want

WHAT THIS IS AND IS NOT
-----------------------
This is the location (and optionally scale) step of ComBat, without ComBat's
empirical-Bayes shrinkage. The shrinkage matters when batches are small -- it pools
per-feature estimates toward a common prior, so a batch of five samples does not get
a wildly overfitted correction. With batches of tens or hundreds the two agree
closely. For small batches, prefer a real ComBat implementation.

Adjusting is not free: it costs degrees of freedom and can only ever *reduce*
apparent structure. Always re-run :mod:`pvclust_py.diagnostics` afterwards and check
that batch fell AND the biology did not.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _dummies(f: pd.Series, drop_first: bool) -> pd.DataFrame:
    """Treatment-coded dummies for one factor.

    ``drop_first`` matters more than it looks. Coding every level of every factor
    with no intercept makes the design rank-deficient -- the batch columns and the
    protected columns each sum to 1, so they are collinear with one another. numpy's
    least squares then returns the minimum-norm solution, which splits the shared
    intercept arbitrarily across collinear columns; the "batch coefficients" pick up
    part of the overall mean and change unpredictably as protected variables are
    added. With an explicit intercept and one reference level dropped per factor, the
    batch coefficients are identified as deviations from a reference and behave.
    """
    d = pd.get_dummies(f.astype(str), prefix=f.name or "f", dtype=float)
    return d.iloc[:, 1:] if (drop_first and d.shape[1] > 1) else d


def adjust_batch(X: pd.DataFrame, batch, protect=None, *, scale: bool = False,
                 min_per_batch: int = 3) -> pd.DataFrame:
    """Remove batch effects from a samples x features matrix.

    Args:
        X: rows are samples, columns are features. Log-transform first if the data
            is on a ratio scale -- batch effects are multiplicative there, and
            subtracting means only makes sense once they are additive.
        batch: batch label per sample, aligned to ``X.index``.
        protect: variable(s) whose effect must be PRESERVED -- typically the disease
            group. A Series, or a DataFrame of several. Omit only if you have checked
            that batch is balanced with respect to everything you care about.
        scale: also equalise per-batch variance. Location-only is the safer default;
            scale correction can flatten real biological heterogeneity if one batch
            genuinely contains more diverse samples.
        min_per_batch: refuse batches smaller than this, where per-feature estimates
            are too noisy to subtract responsibly.

    Returns:
        The adjusted matrix, same shape and labels.

    Raises:
        ValueError: if a batch is too small, or if batch is *completely* confounded
            with a protected variable -- in which case no method can separate them
            and the honest answer is that the design cannot support the question.
    """
    batch = pd.Series(batch).reindex(X.index)
    if batch.isna().any():
        raise ValueError(f"{int(batch.isna().sum())} samples have no batch label")

    counts = batch.value_counts()
    if (counts < min_per_batch).any():
        small = counts[counts < min_per_batch].to_dict()
        raise ValueError(f"batches too small to adjust: {small} "
                         f"(min_per_batch={min_per_batch})")

    prot = None
    if protect is not None:
        prot = pd.DataFrame(protect).reindex(X.index)
        for col in prot.columns:
            table = pd.crosstab(batch, prot[col].astype(str))
            # complete confounding: every batch contains exactly one level
            if (table > 0).sum(axis=1).max() == 1:
                raise ValueError(
                    f"batch is completely confounded with {col!r} -- each batch holds "
                    f"only one level, so no method can tell them apart. This is a "
                    f"design limitation, not something to correct around.")

    A = X.to_numpy(float)
    grand = A.mean(axis=0)

    # Intercept + treatment-coded factors: full rank, so the batch coefficients mean
    # "deviation from the reference batch" and do not absorb the overall level.
    Db = _dummies(batch.rename("batch"), drop_first=True)
    blocks = [pd.DataFrame({"intercept": np.ones(len(X))}, index=X.index), Db]
    if prot is not None:
        blocks += [_dummies(prot[c].fillna("__missing__").rename(c), drop_first=True)
                   for c in prot.columns]
    design = pd.concat(blocks, axis=1)

    M = design.to_numpy(float)
    if np.linalg.matrix_rank(M) < M.shape[1]:
        raise ValueError(
            "the batch and protected variables are collinear -- they cannot be "
            "estimated separately on this design. Check confounding() first.")

    coef, *_ = np.linalg.lstsq(M, A, rcond=None)
    lo = 1                                   # skip the intercept
    hi = lo + Db.shape[1]
    batch_fit = Db.to_numpy(float) @ coef[lo:hi]

    # Remove batch DIFFERENCES, not the overall level.
    adjusted = A - batch_fit + batch_fit.mean(axis=0)

    if scale:
        out = adjusted.copy()
        target = adjusted.std(axis=0, ddof=1)
        for b in counts.index:
            m = (batch == b).to_numpy()
            sd = adjusted[m].std(axis=0, ddof=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                factor = np.where(sd > 0, target / sd, 1.0)
            centre = adjusted[m].mean(axis=0)
            out[m] = (adjusted[m] - centre) * factor + centre
        adjusted = out

    adjusted = adjusted - adjusted.mean(axis=0) + grand
    return pd.DataFrame(adjusted, index=X.index, columns=X.columns)


def confounding(batch, annotations: pd.DataFrame) -> pd.DataFrame:
    """How entangled is batch with each annotation? Run this BEFORE adjusting.

    Returns one row per annotation with a chi-square p-value, bias-corrected
    Cramer's V, and a verdict:

      * ``balanced``  -- adjust freely
      * ``partial``   -- adjustable, but pass the variable to ``protect=``
      * ``complete``  -- every batch holds one level; the design cannot separate them
    """
    from scipy.stats import chi2_contingency

    from .diagnostics import _cramers_v

    batch = pd.Series(batch)
    rows = []
    for col in annotations.columns:
        v = annotations[col].reindex(batch.index)
        ok = v.notna()
        if ok.sum() < 3 or v[ok].nunique() < 2:
            rows.append({"variable": col, "verdict": "skipped"})
            continue
        table = pd.crosstab(batch[ok], v[ok].astype(str))
        if min(table.shape) < 2:
            rows.append({"variable": col, "verdict": "skipped"})
            continue
        chi2, p, _, _ = chi2_contingency(table)
        n = int(table.to_numpy().sum())
        cv = _cramers_v(chi2, n, *table.shape)
        verdict = ("complete" if (table > 0).sum(axis=1).max() == 1
                   else "partial" if (p < 0.05 and cv >= 0.10) else "balanced")
        rows.append({"variable": col, "p_value": float(p), "cramers_v": float(cv),
                     "verdict": verdict,
                     "action": {"complete": "cannot adjust -- design limitation",
                                "partial": "adjust with protect=",
                                "balanced": "adjust freely"}[verdict]})
    return pd.DataFrame(rows)


def combat(X: pd.DataFrame, batch, protect=None, *, mean_only: bool = False,
           ref_batch=None) -> pd.DataFrame:
    """ComBat: the standard batch correction for omics data.

    Johnson, Li & Rabinovic (2007), *Adjusting batch effects in microarray expression
    data using empirical Bayes methods*, Biostatistics 8(1):118-127. Wraps
    ``inmoose.pycombat``, a maintained Python port of the R implementation.

    ComBat is :func:`adjust_batch` plus **empirical-Bayes shrinkage**: per-feature
    location and scale corrections are pooled toward a common prior, so a small batch
    does not receive a wildly overfitted correction. On this project's SLE data the
    two agree to r = 0.9996, because the batches are large; the difference matters
    when batches are small.

    Args:
        X: samples x features, log-transformed.
        batch: batch label per sample.
        protect: variable(s) to preserve, passed as ComBat's ``covar_mod``.

            **Think carefully before using this in an unsupervised analysis.**
            Protecting the disease label conditions the data on the very thing the
            clustering is meant to discover, which weakens any claim that the
            structure was found without supervision. ``covar_mod`` is standard
            practice for *differential expression*, where nothing is being
            discovered; for unsupervised clustering it invites the objection that
            the labels leaked in. Prefer no ``protect``, and report the
            batch-versus-biology confounding openly instead.
        mean_only: correct location only, leaving per-batch variance alone.
        ref_batch: adjust all batches toward this one, rather than the grand mean.

    Raises:
        ImportError: if ``inmoose`` is not installed (``pip install inmoose``).
    """
    try:
        from inmoose.pycombat import pycombat_norm
    except ImportError as exc:                                   # pragma: no cover
        raise ImportError(
            "ComBat needs the optional 'combat' extra, which is not installed."
        ) from exc

    batch = pd.Series(batch).reindex(X.index)
    if batch.isna().any():
        raise ValueError(f"{int(batch.isna().sum())} samples have no batch label")

    covar = None
    if protect is not None:
        prot = pd.DataFrame(protect).reindex(X.index)
        covar = pd.concat([_dummies(prot[c].fillna("__missing__").rename(c),
                                    drop_first=True) for c in prot.columns], axis=1)

    # ComBat works on features x samples, as in R.
    out = pycombat_norm(X.T, batch.tolist(), covar_mod=covar,
                        mean_only=mean_only, ref_batch=ref_batch)
    return pd.DataFrame(out).T
