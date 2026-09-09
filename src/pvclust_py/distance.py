"""
Distances between the objects being clustered, and the sufficient statistics they
reduce to.

A port of ``dist.pvclust`` (pvclust-internal.R:334). pvclust clusters the COLUMNS of
the data matrix, computing distance across the rows, so every distance here treats
``X`` as ``n`` rows (resampling units) by ``p`` columns (objects) and returns a
``p x p`` matrix.

WHY THIS MODULE IS BUILT ON SUFFICIENT STATISTICS
-------------------------------------------------
Each supported distance is a function of four ``p x p`` matrices, every entry of
which is a sum over ROWS:

    N[j,k] = number of rows where both j and k are observed
    S[j,k] = sum of x_j over those rows          (asymmetric: S[k,j] sums x_k)
    Q[j,k] = sum of x_j^2 over those rows
    G[j,k] = sum of x_j*x_k over those rows      (the Gram matrix proper)

Because they are row sums they are ADDITIVE across projects holding disjoint rows:
pool by adding, and the resulting distance matrix is bit-for-bit the one you would
get from pooling the raw data. That is what lets the federated dendrogram be exact
rather than approximate, and it is why the local path and the federated path here
are the same code -- :func:`distance` just computes the statistics first.

The payload is ``4*p^2`` numbers regardless of how many rows a project holds.

NA HANDLING IS NOT UNIFORM, AND THAT IS FAITHFUL
------------------------------------------------
pvclust is internally inconsistent about missing data and the port reproduces it:

  * ``correlation`` / ``abscor`` use PAIRWISE complete observations (R's
    ``cor(use="pairwise.complete.obs")``) -- each pair uses whatever rows it has.
  * ``uncentered`` uses LISTWISE deletion -- R calls ``na.omit(x)`` and drops any
    row with a missing value anywhere, warning as it goes. Note the consequence for
    federation: the listwise mask depends on the whole column set, so changing the
    shared-feature vocabulary changes which rows survive.
  * ``euclidean`` goes through R's ``dist()``, which drops incomplete pairs and then
    SCALES THE SUM UP by ``n/count`` to compensate. Forgetting the scaling gives
    distances that are too small whenever data is missing.
"""
from __future__ import annotations

from typing import Dict, Iterable, Mapping

import numpy as np

#: Distances that reduce to the sufficient statistics, and so federate exactly.
#: ``minkowski`` is here because pvclust never passes ``p`` to R's ``dist()``, so it
#: always means p=2, i.e. euclidean -- verified against R (max difference 0.0).
METHODS = ("correlation", "abscor", "uncentered", "euclidean", "minkowski")

#: Distances R supports via dist() that are NOT recoverable from the statistics.
#: Listed so the aggregator can refuse them in exact mode with a useful message.
NOT_GRAM_DERIVABLE = ("manhattan", "canberra", "binary", "maximum")


def pairwise_stats(X) -> Dict[str, np.ndarray]:
    """Pairwise-complete sufficient statistics of an ``n x p`` matrix.

    Returns ``{"n", "N", "S", "Q", "G"}``. Every matrix entry is a sum over rows,
    hence additive across projects -- see :func:`pool_stats`.

    Warning:
        DISCLOSURE RISK IS GOVERNED BY n RELATIVE TO p, not by n alone.
        ``rank(G) = min(n, p)``, so when ``n <= p`` the statistics expose the exact
        n-dimensional subspace the rows occupy; individuals are hidden only by a
        rotation within it. At ``n = 1`` the Gram is rank-1 and an eigendecomposition
        returns the row exactly. Only when ``n > p`` is the Gram genuinely averaging.

        A fixed minimum-n is therefore the weaker guard. The stronger rule is to keep
        the object count below the row count -- which also happens to be where the
        exact path is computationally comfortable. See Homer et al. (2008) for the
        canonical demonstration that aggregate summary statistics are not
        automatically safe.
    """
    X = np.asarray(X, dtype=float)
    M = ~np.isnan(X)
    Z = np.nan_to_num(X)          # missing -> 0 so they drop out of every sum
    Mf = M.astype(float)
    return {
        "n": X.shape[0],
        "N": Mf.T @ Mf,
        "S": Z.T @ Mf,
        "Q": (Z * Z).T @ Mf,
        "G": Z.T @ Z,
    }


def listwise_stats(X) -> Dict[str, np.ndarray]:
    """Sufficient statistics after dropping every row with any missing value.

    This is what ``uncentered`` needs, because R applies ``na.omit`` before taking
    the crossproduct rather than handling pairs independently.
    """
    X = np.asarray(X, dtype=float)
    complete = X[~np.isnan(X).any(axis=1)]
    return {"n": complete.shape[0], "G": complete.T @ complete}


def pool_stats(parts: Iterable[Mapping[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Add per-project statistics into the pooled statistics.

    Exact, not an approximation: the result equals :func:`pairwise_stats` of the
    concatenated raw data, so the pooled distance matrix equals the pooled-raw one.
    """
    parts = list(parts)
    if not parts:
        raise ValueError("no statistics to pool")

    # Labels first. Matching SHAPES are not enough: two projects that each selected
    # their own top-50 features by variance produce two 50x50 matrices describing
    # DIFFERENT proteins, and adding them silently yields a meaningless Gram. Run
    # shared-features first so every project clusters the same vocabulary.
    labelled = [s.get("labels") for s in parts]
    if any(l is not None for l in labelled):
        if any(l is None for l in labelled):
            raise ValueError("some statistics carry labels and others do not -- "
                             "regenerate them all with the current project-stats")
        ref = [str(x) for x in labelled[0]]
        for i, l in enumerate(labelled[1:], start=1):
            other = [str(x) for x in l]
            if other != ref:
                extra = sorted(set(other) - set(ref))[:4]
                miss = sorted(set(ref) - set(other))[:4]
                raise ValueError(
                    f"statistics {i} describes different objects from statistics 0, so "
                    f"they cannot be added. Only in this one: {extra}; missing from it: "
                    f"{miss}. Run shared-features across the projects and rebuild the "
                    f"statistics on that common vocabulary.")

    keys = set(parts[0]) - {"labels"}
    for i, s in enumerate(parts[1:], start=1):
        if set(s) - {"labels"} != keys:
            raise ValueError(f"statistics {i} has keys {sorted(set(s) - {'labels'})}, "
                             f"expected {sorted(keys)}")
        if s.get("G") is not None and s["G"].shape != parts[0]["G"].shape:
            raise ValueError("statistics disagree on the number of objects -- "
                             "pool one shared vocabulary")

    pooled = {k: sum(s[k] for s in parts) for k in keys}
    if labelled[0] is not None:
        pooled["labels"] = labelled[0]
    return pooled


def _zero_diagonal(D: np.ndarray) -> np.ndarray:
    """R's as.dist() keeps only the off-diagonal, so the diagonal is exactly 0
    rather than whatever rounding produced."""
    np.fill_diagonal(D, 0.0)
    return D


def _corr_from_stats(st: Mapping[str, np.ndarray]) -> np.ndarray:
    """Pairwise-complete Pearson correlation, from statistics alone."""
    N, S, Q, G = st["N"], st["S"], st["Q"], st["G"]
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = (G - S * S.T / N) / (N - 1)
        var = (Q - S * S / N) / (N - 1)
        return cov / np.sqrt(var * var.T)


def distance_from_stats(st: Mapping[str, np.ndarray], method: str = "correlation") -> np.ndarray:
    """The ``p x p`` distance matrix implied by sufficient statistics.

    Args:
        st: from :func:`pairwise_stats` (or :func:`listwise_stats` for ``uncentered``),
            possibly pooled across projects by :func:`pool_stats`.
        method: one of :data:`METHODS`.

    Raises:
        ValueError: for a method that cannot be recovered from statistics.
    """
    method = method.lower()
    if method in NOT_GRAM_DERIVABLE:
        raise ValueError(
            f"{method!r} is not recoverable from sufficient statistics, so it cannot "
            f"be used for exact federation -- use one of {METHODS}, or counts mode.")
    if method == "minkowski":
        # pvclust calls dist(t(x), method) without p, so R's default p=2 applies and
        # minkowski IS euclidean. Verified against R: max difference 0.0. A p != 2
        # Minkowski is reachable in R only by passing a function as method.dist, and
        # is not Gram-derivable -- see minkowski_distance().
        method = "euclidean"

    if method == "correlation":
        return _zero_diagonal(1.0 - _corr_from_stats(st))
    if method == "abscor":
        return _zero_diagonal(1.0 - np.abs(_corr_from_stats(st)))
    if method == "uncentered":
        G = st["G"]
        q = np.diag(G)
        return _zero_diagonal(1.0 - G / np.sqrt(np.outer(q, q)))
    if method == "euclidean":
        # R's dist(): sum over complete pairs, then scale up by n/count.
        N, Q, G = st["N"], st["Q"], st["G"]
        with np.errstate(invalid="ignore", divide="ignore"):
            ssq = Q + Q.T - 2.0 * G
            return _zero_diagonal(np.sqrt(np.clip(ssq, 0.0, None) * (st["n"] / N)))

    raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")


def distance(X, method: str = "correlation") -> np.ndarray:
    """Distance between the columns of ``X`` -- the local, non-federated path.

    Goes through the same statistics the federated path uses, so there is exactly
    one implementation of each distance to keep in agreement with R.
    """
    method = method.lower()
    st = listwise_stats(X) if method == "uncentered" else pairwise_stats(X)
    return distance_from_stats(st, method)


def minkowski_distance(X, p: float = 2.0) -> np.ndarray:
    """Minkowski distance with an arbitrary exponent, computed directly.

    Only needed for ``p != 2``. pvclust itself cannot produce this -- ``dist.pvclust``
    never passes ``p`` to R's ``dist()``, so pvclust's "minkowski" is always p=2 and
    therefore euclidean; reaching another exponent in R means supplying a function as
    ``method.dist``.

    Warning:
        A ``p != 2`` Minkowski distance is NOT recoverable from sufficient statistics,
        so it cannot be federated exactly -- counts mode only. If exact federation
        matters, stay at p=2, where this is euclidean and pools exactly.
    """
    X = np.asarray(X, dtype=float)
    if p == 2:
        return distance(X, "euclidean")

    # R's dist() excludes incomplete pairs and then scales the sum up by n/count,
    # exactly as for euclidean. Zero-filling the missing values instead treats them
    # as observed zeros and gives visibly wrong distances.
    n, q = X.shape
    M = ~np.isnan(X)
    Z = np.nan_to_num(X)
    D = np.zeros((q, q))
    for j in range(q):
        both = M[:, [j]] & M                       # rows where j and each k are present
        diff = np.abs(Z[:, [j]] - Z) ** p
        ssum = (diff * both).sum(axis=0)
        count = both.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            D[j] = np.where(count > 0, (ssum * (n / count)) ** (1.0 / p), np.nan)
    np.fill_diagonal(D, 0.0)
    return D
