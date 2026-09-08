"""
Is the clustering driven by biology, or by the plate it was run on?

Clustering will happily group samples by plate, scanner or run date and hand you
confident AU values for it. AU says a cluster is *reproducible*, not that it is
*meaningful* -- a batch effect is perfectly reproducible. These tests turn "the
colour strips look mixed" into a number.

For each annotation the association with the cluster assignment is measured two ways,
because they answer different questions:

  * a HYPOTHESIS TEST (chi-square for categorical, Kruskal-Wallis for numeric) --
    "could this association be chance?" Its p-value shrinks with sample size, so on
    a few hundred samples almost anything reaches significance.
  * an EFFECT SIZE (Cramer's V, adjusted Rand, eta-squared) -- "how strong is it?"
    This is the number to look at. A p of 1e-10 with V = 0.08 is a real but trivial
    association; V = 0.7 means your clusters ARE the plates.

Adjusted Rand is the strictest reading for a categorical variable: it asks whether
the two partitions are the same partition, corrected for chance agreement.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

#: Rules of thumb for Cramer's V / adjusted Rand. Conventional, not laws.
EFFECT_BANDS = ((0.10, "negligible"), (0.30, "weak"), (0.50, "moderate"), (1.01, "strong"))


def _cramers_v(chi2: float, n: int, r: int, c: int) -> float:
    """Bias-corrected Cramer's V (Bergsma 2013).

    The uncorrected form is inflated when a variable has many levels relative to the
    sample size -- a 96-level plate position over 301 samples scored 0.56 ("strong")
    on data whose chi-square p-value was 0.46, i.e. noise. The correction subtracts
    the expected value under independence, which is exactly that inflation.
    """
    phi2 = chi2 / n
    phi2c = max(0.0, phi2 - (r - 1) * (c - 1) / (n - 1))
    rc = r - (r - 1) ** 2 / (n - 1)
    cc = c - (c - 1) ** 2 / (n - 1)
    denom = min(rc - 1, cc - 1)
    return float(np.sqrt(phi2c / denom)) if denom > 0 else 0.0


def _band(v: float) -> str:
    for cut, name in EFFECT_BANDS:
        if abs(v) < cut:
            return name
    return "strong"


def cluster_labels(result, k: Optional[int] = None, alpha: float = 0.95) -> pd.Series:
    """One cluster label per object, for testing against annotations.

    Args:
        result: a pvclust or kmeans_pv result.
        k: cut the dendrogram into this many groups. Ignored for k-means, which
            already has its groups. Defaults to the number of clusters pvpick keeps
            at ``alpha`` (minimum 2), so the test uses the clusters you would report.
        alpha: AU threshold used when ``k`` is not given.

    Returns:
        Series indexed by object label, values are integer cluster ids.
    """
    from scipy.cluster.hierarchy import fcluster

    from .core import pvpick

    if result.is_kmeans:              # k-means: the groups ARE the assignment, and
                                     # its linkage is over centroids, not objects
        out = {}
        for i, e in enumerate(result.edges):
            for m in e["members"]:
                out[m] = i
        return pd.Series(out, name="cluster")

    if k is None:
        k = max(2, len(pvpick(result, alpha)))
    labels = fcluster(result.linkage, t=k, criterion="maxclust")
    return pd.Series(labels, index=result.labels, name="cluster")


def association(clusters: pd.Series, annotations: pd.DataFrame,
                numeric: Optional[list] = None) -> pd.DataFrame:
    """Test each annotation against the cluster assignment.

    Args:
        clusters: from :func:`cluster_labels`, indexed by object label.
        annotations: DataFrame indexed the same way, one column per variable.
        numeric: columns to treat as continuous (Kruskal-Wallis). Inferred from dtype
            when omitted.

    Returns:
        One row per annotation: test used, statistic, p-value, effect size, and a
        plain-language reading. Sorted by effect size, strongest first -- that is the
        order you care about, not the p-value order.
    """
    from scipy.stats import chi2_contingency, kruskal
    from sklearn.metrics import adjusted_rand_score

    common = clusters.index.intersection(annotations.index)
    if len(common) < 3:
        raise ValueError(f"only {len(common)} objects in common between the clustering "
                         f"and the annotations -- are they indexed the same way?")
    cl = clusters.loc[common]
    ann = annotations.loc[common]

    rows = []
    for col in ann.columns:
        v = ann[col]
        is_num = (col in numeric) if numeric is not None else pd.api.types.is_numeric_dtype(v)
        ok = v.notna()
        if ok.sum() < 3 or v[ok].nunique() < 2:
            rows.append({"variable": col, "test": "skipped",
                         "reason": "constant or too few values"})
            continue

        c, x = cl[ok], v[ok]
        if is_num:
            groups = [x[c == g].to_numpy(float) for g in c.unique() if (c == g).sum() > 0]
            groups = [g for g in groups if len(g) > 0]
            if len(groups) < 2:
                rows.append({"variable": col, "test": "skipped", "reason": "one cluster"})
                continue
            stat, p = kruskal(*groups)
            # eta-squared from H: the share of rank variance explained by cluster
            n, kk = len(x), len(groups)
            eff = max(0.0, (stat - kk + 1) / (n - kk)) if n > kk else np.nan
            rows.append({"variable": col, "test": "Kruskal-Wallis", "n": int(n),
                         "levels": kk, "statistic": float(stat), "p_value": float(p),
                         "effect": float(eff), "effect_name": "eta squared",
                         "strength": _band(np.sqrt(eff) if eff == eff else 0.0)})
        else:
            table = pd.crosstab(c, x.astype(str))
            if table.shape[0] < 2 or table.shape[1] < 2:
                rows.append({"variable": col, "test": "skipped", "reason": "one level"})
                continue
            stat, p, dof, _ = chi2_contingency(table)
            n = int(table.to_numpy().sum())
            cramers_v = _cramers_v(stat, n, *table.shape)
            ari = adjusted_rand_score(c.astype(str), x.astype(str))
            rows.append({"variable": col, "test": "chi-square", "n": n,
                         "levels": int(table.shape[1]), "statistic": float(stat),
                         "p_value": float(p), "effect": float(cramers_v),
                         "effect_name": "Cramers V", "adjusted_rand": float(ari),
                         "strength": _band(cramers_v)})

    df = pd.DataFrame(rows)
    if "effect" in df.columns:
        df = df.sort_values("effect", ascending=False, ignore_index=True)
    return df


def report(assoc: pd.DataFrame, batch_like=("PlateId", "ScannerID", "PlateRunDate",
                                            "Batch", "PlatePosition", "Subarray",
                                            "SlideId", "RowCheck")) -> str:
    """A short verdict on whether the clustering looks like a batch effect.

    Flags any batch-like variable with a more than negligible effect. Deliberately
    blunt: this is the check that stops a plate effect being published as biology.
    """
    lines = []
    # BOTH conditions: a large effect that is not significant is small-sample noise,
    # and a significant effect that is tiny does not matter. Requiring only one of
    # them produces false alarms, which is worse than no diagnostic at all.
    strong = (assoc.get("effect", 0) >= 0.30) & (assoc.get("p_value", 1) < 0.05)
    flagged = assoc[assoc["variable"].isin(batch_like) & strong]
    if len(flagged):
        lines.append("WARNING -- the clustering tracks technical variables:")
        for _, r in flagged.iterrows():
            lines.append(f"  {r['variable']}: {r['effect_name']} = {r['effect']:.2f} "
                         f"({r['strength']}), p = {r['p_value']:.2g}")
        lines.append("  Clusters may be reproducing the assay run, not biology. AU will "
                     "be high either way -- a batch effect is perfectly reproducible.")
    else:
        lines.append("No batch-like variable shows more than a weak association.")

    other = assoc[~assoc["variable"].isin(batch_like) & strong]
    if len(other):
        lines.append("Non-technical variables associated with the clustering:")
        for _, r in other.iterrows():
            lines.append(f"  {r['variable']}: {r['effect_name']} = {r['effect']:.2f} "
                         f"({r['strength']}), p = {r['p_value']:.2g}")
    return "\n".join(lines)
