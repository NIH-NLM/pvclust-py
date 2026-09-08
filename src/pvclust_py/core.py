"""
The multiscale bootstrap loop -- pvclust proper.

Ties the other modules together: build the tree once, then resample rows at each
scale, recluster, and tally which of the original clusters reappeared. The tally
goes to :mod:`pvclust_py.msfit`, which turns it into si/au/bp per cluster.

    distance.py --> hclust.py --> [ this module: resample, recluster, tally ] --> msfit.py

WHY THIS IS THE USEFUL PART
---------------------------
k-means and plain hclust hand you a partition with no measure of whether it is real.
This gives every cluster an AU p-value, so :func:`pvpick` can cut the tree at
AU >= alpha rather than making you choose k. AU reads as: "resampling patients from
the same population, this cluster would reappear this often." Note that BP -- the
ordinary bootstrap number -- is biased DOWNWARD, so cutting on BP discards real
structure; that bias is exactly what msfit removes.

FAITHFULNESS
------------
Two behaviours are reproduced deliberately and must not be "fixed":

  * Replicates whose distance matrix is not all-finite are SKIPPED, but still count
    toward nboot (pvclust-internal.R:255-262). BP is therefore deflated by bad
    replicates rather than the draw being retried. ``na_flag`` reports it, as R warns.
  * Scales come from :func:`pvclust_py.scales.effective_scales` -- ``unique(floor(n*r))/n``
    -- never the nominal r. See that module for why this matters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .distance import distance
from .hclust import edge_id, edge_members, edge_table, linkage
from .msfit import MsFit, msfit
from .scales import effective_scales


@dataclass
class PvclustResult:
    """A clustered dendrogram with a p-value on every internal node."""

    linkage: np.ndarray
    labels: List[str]
    edges: List[Dict]                      # edge_id, members, height, si/au/bp, ...
    count: np.ndarray                      # (n_edges, n_scales) integer tallies
    r: np.ndarray                          # EFFECTIVE scales actually resampled
    nboot: np.ndarray                      # replicates attempted per scale
    method_dist: str
    method_hclust: str
    cluster: str = "columns"               # which axis was clustered
    objects: Optional[List[str]] = None    # for k-means: the original object names,
                                           # since `labels` then names the CLUSTERS
    na_flag: bool = False                  # some replicates gave non-finite distances
    msfits: List[MsFit] = field(default_factory=list, repr=False)

    @property
    def is_kmeans(self) -> bool:
        """True for a k-means result.

        Ask this rather than testing ``linkage.size``. A k-means result now DOES carry
        a linkage -- a dendrogram over its centroids -- but that tree's leaves are the
        CLUSTERS, not the objects. Code wanting the object-level tree must not pick it
        up by accident.
        """
        return self.method_hclust == "kmeans"

    def edges_frame(self):
        """The per-edge table as a pandas DataFrame (pandas imported lazily)."""
        import pandas as pd
        return pd.DataFrame([{**e, "members": ";".join(e["members"])} for e in self.edges])

    def counts_frame(self):
        """Long-form counts -- exactly what a project ships in counts-mode federation."""
        import pandas as pd
        rows = [
            {"edge_id": e["edge_id"], "r": float(self.r[j]),
             "nboot": int(self.nboot[j]), "count": int(self.count[i, j])}
            for i, e in enumerate(self.edges) for j in range(len(self.r))
        ]
        return pd.DataFrame(rows)


def orient(X, labels=None, cluster: str = "columns"):
    """Put the objects to be clustered on the COLUMNS, whatever the caller passed.

    pvclust always clusters columns and resamples rows. Expression matrices get used
    both ways round -- genes across the top and samples down the side, or the reverse
    -- so rather than making callers transpose by hand (and mislabel the result), say
    which axis you want clustered and this sorts it out.

    Accepts a DataFrame, in which case labels are taken from the appropriate axis.

    Returns:
        ``(A, labels, resampling_units)`` -- ``A`` with objects as columns, the object
        labels, and a description of what the rows now are, for warnings.
    """
    if cluster not in ("columns", "rows"):
        raise ValueError(f"cluster must be 'columns' or 'rows', got {cluster!r}")

    if hasattr(X, "columns"):                       # a DataFrame
        obj_axis = X.index if cluster == "rows" else X.columns
        other = X.columns if cluster == "rows" else X.index
        A = np.asarray(X, dtype=float)
        if cluster == "rows":
            A = A.T
        return A, [str(v) for v in (labels if labels is not None else obj_axis)], \
               f"{len(other)} {'columns' if cluster == 'rows' else 'rows'}"

    A = np.asarray(X, dtype=float)
    if cluster == "rows":
        A = A.T
    labels = [str(c) for c in (labels if labels is not None else range(A.shape[1]))]
    return A, labels, f"{A.shape[0]} rows"


def _tally(X, ids, cluster_fn, sizes, nboot, seed, quiet, r_eff):
    """Resample rows at each scale, recluster, and count which candidate clusters
    reappeared.

    Shared by :func:`pvclust` and :func:`kmeans_pv` -- only ``cluster_fn`` differs,
    which is the whole point: msfit does not care how the clusters were produced,
    only whether each one is present or absent in a replicate. ``cluster_fn``
    returns a list of member-label lists, or None to skip the replicate.

    A skipped replicate still counts toward nboot, matching
    pvclust-internal.R:255-262 -- BP is deflated by bad replicates rather than the
    draw being retried.
    """
    n = X.shape[0]
    index = {eid: i for i, eid in enumerate(ids)}
    count = np.zeros((len(ids), len(sizes)), dtype=int)
    rng = np.random.default_rng(seed)
    na_flag = False

    for j, size in enumerate(sizes):
        if not quiet:
            print(f"Bootstrap (r = {r_eff[j]:.3f}, size = {size})...", flush=True)
        for _ in range(nboot):
            members = cluster_fn(X[rng.integers(0, n, size)])
            if members is None:
                na_flag = True
                continue
            for m in members:
                i = index.get(edge_id(m))
                if i is not None:
                    count[i, j] += 1
    return count, na_flag


def pvclust(X, labels: Optional[Sequence[str]] = None, *,
            cluster: str = "columns",
            method_dist: str = "correlation", method_hclust: str = "average",
            nboot: int = 1000, r: Optional[Sequence[float]] = None,
            seed: int = 42, quiet: bool = True) -> PvclustResult:
    """Hierarchical clustering of the COLUMNS of ``X`` with AU p-values.

    Args:
        X: array or DataFrame. By default rows are the resampling units (e.g.
            samples) and columns are the objects clustered (e.g. proteins).
        cluster: ``"columns"`` (default) or ``"rows"``. Clustering rows transposes
            internally, so the samples become the objects and the analytes become the
            resampling units -- the other half of a two-way expression figure. See the
            warning below; this is not a symmetric choice statistically.
        labels: names for the clustered objects. Taken from the DataFrame axis when
            not given.
        method_dist: correlation | abscor | uncentered | euclidean.
        method_hclust: an R hclust method name (average, complete, ward.D2, ...).
        nboot: replicates per scale. R's default is 1000.
        r: nominal relative sample sizes; defaults to ``seq(.5, 1.4, by=.1)``.
        seed: RNG seed.

    Returns:
        PvclustResult.

    Warning:
        The bootstrap is only meaningful if the RESAMPLING UNITS are exchangeable
        draws from a population. With ``cluster="columns"`` on a samples x analytes
        matrix those units are samples, which is the sound direction. With
        ``cluster="rows"`` they are analytes -- co-expressed, not independent -- so AU
        comes out ANTI-CONSERVATIVE, i.e. optimistic. Both dendrograms of a two-way
        figure are worth having; only one of them carries a p-value you can quote
        without qualification. A warning is emitted for the rows case.
    """
    X, labels, units = orient(X, labels, cluster)
    n, p = X.shape
    if len(labels) != p:
        raise ValueError(f"{len(labels)} labels for {p} objects to cluster")
    if cluster == "rows":
        import warnings
        warnings.warn(
            f"clustering rows means resampling the {units}, which are typically "
            f"correlated rather than independent draws; AU will be anti-conservative "
            f"(optimistic). Fine for the companion dendrogram of a two-way figure -- "
            f"just do not quote these p-values as if they were the column ones.",
            stacklevel=2)
    if n < 3:
        raise ValueError(f"need at least 3 rows to resample, got {n}")
    if n < 30:
        import warnings
        warnings.warn(
            f"only {n} resampling units: the multiscale bootstrap has little to work "
            f"with, and AU will be poorly determined. If these are genes being "
            f"resampled to assess patient clusters, AU is also anti-conservative.",
            stacklevel=2)

    # --- the original tree, whose edges everything is counted against ---------
    Z = linkage(distance(X, method_dist), method_hclust)
    edges = edge_table(Z, labels)
    ids = [e["edge_id"] for e in edges]
    index = {eid: i for i, eid in enumerate(ids)}

    sizes, r_eff = effective_scales(n, r)

    def cluster_replicate(M):
        D = distance(M, method_dist)
        if not np.isfinite(D).all():
            return None                      # R skips, but still counts toward nboot
        return edge_members(linkage(D, method_hclust), labels)

    count, na_flag = _tally(X, ids, cluster_replicate, sizes, int(nboot), seed, quiet, r_eff)
    nboot_vec = np.full(len(sizes), int(nboot))
    fits = [msfit(count[i] / nboot_vec, r_eff, nboot_vec) for i in range(len(edges))]

    for e, f in zip(edges, fits):
        e.update(si=f.si, au=f.au, bp=f.bp, se_si=f.se_si, se_au=f.se_au,
                 se_bp=f.se_bp, v=f.v, c=f.c, df=f.df, rss=f.rss, pchi=f.pchi)

    return PvclustResult(linkage=Z, labels=labels, edges=edges, count=count,
                         r=r_eff, nboot=nboot_vec, method_dist=method_dist,
                         method_hclust=method_hclust, na_flag=na_flag, msfits=fits,
                         cluster=cluster)


def pvpick(result: PvclustResult, alpha: float = 0.95, *, use: str = "au",
           max_only: bool = True) -> List[Dict]:
    """The clusters worth reporting: significant clusters, largest first.

    This is the answer to "how do I choose k" -- you do not. Clusters clearing the
    threshold are returned and the tree cuts itself.

    Ported from R's ``pvpick``, including two behaviours that are easy to miss:

      * THE ROOT IS EXCLUDED. R loops ``for(i in (len-1):1)``, skipping the last
        edge, which always contains every leaf and always has AU = 1. Returning it
        would be vacuous -- an earlier draft of this function did exactly that.
      * ``max_only`` is greedy top-down, not a subset test. Walking from the largest
        cluster downward, a cluster is kept only if none of its members were already
        claimed. Members of a passing cluster are marked claimed whether or not it
        was itself kept.

    Cutting on ``use="bp"`` discards real structure, since BP is biased downward.
    """
    if use not in ("au", "bp", "si"):
        raise ValueError(f"use must be one of au|bp|si, got {use!r}")

    edges = result.edges
    claimed: set = set()
    picked: List[Dict] = []

    # R: for(i in (len-1):1) -- skip the root, walk from largest cluster downward.
    for e in reversed(edges[:-1]):
        if e[use] < alpha:
            continue
        members = set(e["members"])
        if not max_only or not (members & claimed):
            picked.append(e)
        claimed |= members          # claimed even when not kept, as in R

    picked.reverse()                # R restores ascending edge order
    return picked


# --------------------------------------------------------------------- k-means
def kmeans_pv(X, labels: Optional[Sequence[str]] = None, *, k: int = 3,
              cluster: str = "columns",
              nboot: int = 1000, r: Optional[Sequence[float]] = None,
              seed: int = 42, n_init: int = 10, jaccard: Optional[float] = None,
              quiet: bool = True) -> PvclustResult:
    """k-means on the COLUMNS of ``X``, with AU p-values from the same bootstrap.

    msfit never asks where a cluster came from -- it needs only a cluster that is
    present or absent in each replicate. A k-means cluster qualifies, and identity by
    member set (:func:`~pvclust_py.hclust.edge_id`) neatly solves k-means' awkward
    part: cluster labels are arbitrarily permuted between runs, which is irrelevant
    once a cluster is named by its contents.

    So this gives k-means what it normally lacks -- a statement that a cluster would
    reappear in a fresh sample -- using exactly the machinery pvclust uses.

    Args:
        k: number of clusters, held fixed across replicates (it must be, or the
            candidates are not comparable).
        n_init: k-means restarts per fit. Do not lower this: k-means' own
            stochasticity would otherwise be counted as sampling variability.
        jaccard: if given (e.g. 0.75), a candidate counts as recovered when some
            replicate cluster reaches this Jaccard similarity, instead of matching
            exactly. See the warning below.

    Warning:
        Exact member-set matching is faithful to Shimodaira's framework -- the event
        is "this exact cluster is the one inferred" -- but for k-means it COLLAPSES AS
        THE NUMBER OF OBJECTS GROWS, because a large cluster rarely reappears
        identically. Measured on lung expression, k=3, nboot=100::

            objects clustered   exact: max BP    jaccard=0.75: max BP
                    20               0.825               0.884
                    50               0.202               0.891
                   150               0.000               0.757

        At a full SomaScan panel or transcriptome, exact matching yields nothing at
        all. ``jaccard`` (cf. Hennig's clusterboot) rescues it, but CHANGES THE
        ESTIMAND -- the result is a cluster-stability measure, not the AU p-value as
        published. Say which you used.

        The default stays exact so the published quantity is what you get unless you
        ask otherwise; a warning fires when it degenerates.

        Hierarchical clustering does not have this problem: its edges are nested, so
        small clusters recur often and the tally stays informative.
    """
    from sklearn.cluster import KMeans

    X, labels, _units = orient(X, labels, cluster)
    n, p = X.shape
    if not 2 <= k <= p:
        raise ValueError(f"k must be between 2 and p={p}, got {k}")

    def fit(M):
        """k-means over the columns of M -- sklearn clusters rows, hence the .T."""
        A = np.nan_to_num(M.T)
        km = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(A)
        return [[labels[i] for i in np.where(km.labels_ == c)[0]] for c in range(k)]

    candidates = [sorted(m) for m in fit(X)]
    ids = [edge_id(m) for m in candidates]

    sizes, r_eff = effective_scales(n, r)

    if jaccard is None:
        cluster_fn = lambda M: [sorted(m) for m in fit(M)]
        count, na_flag = _tally(X, ids, cluster_fn, sizes, int(nboot), seed, quiet, r_eff)
    else:
        # Relaxed matching cannot go through edge_id, so tally directly.
        sets = [set(m) for m in candidates]
        count = np.zeros((k, len(sizes)), dtype=int)
        rng = np.random.default_rng(seed)
        na_flag = False
        for j, size in enumerate(sizes):
            for _ in range(int(nboot)):
                rep = [set(m) for m in fit(X[rng.integers(0, n, size)])]
                for i, cand in enumerate(sets):
                    if any(len(cand & q) / len(cand | q) >= jaccard for q in rep):
                        count[i, j] += 1

    nboot_vec = np.full(len(sizes), int(nboot))
    fits = [msfit(count[i] / nboot_vec, r_eff, nboot_vec) for i in range(k)]

    if jaccard is None and all(f.df == 0 for f in fits):
        import warnings
        warnings.warn(
            f"exact member-set matching recovered no cluster at any scale over "
            f"{p} objects, so every AU is degenerate and meaningless. This is "
            f"expected once the object count grows: pass jaccard=0.75 for a "
            f"stability measure instead -- but report it as such, not as an AU "
            f"p-value.", stacklevel=2)

    edges = [
        {"edge_id": ids[i], "members": candidates[i], "n_members": len(candidates[i]),
         "height": float("nan"), "merge_order": i + 1,
         "si": f.si, "au": f.au, "bp": f.bp, "se_si": f.se_si, "se_au": f.se_au,
         "se_bp": f.se_bp, "v": f.v, "c": f.c, "df": f.df, "rss": f.rss, "pchi": f.pchi}
        for i, f in enumerate(fits)
    ]
    # A dendrogram over the k CENTROIDS. k-means gives a flat partition, but the
    # clusters themselves have profiles, and hierarchically clustering those shows how
    # the clusters relate -- the usual way a k-means result is drawn as a tree. Its
    # leaves are the clusters, not the original objects, so its merge heights describe
    # between-cluster structure. The AU values above belong to the CLUSTERS; this tree
    # is a picture of how they sit relative to one another, and its own merges carry no
    # p-value (nothing resampled them).
    Z = np.empty((0, 4))
    if k >= 2:
        from .distance import distance as _distance
        from .hclust import linkage as _linkage
        # Mean over each cluster's members, per row. A row with NO measured member of
        # a cluster genuinely has no centroid value there, so NaN is the right answer
        # -- but np.nanmean warns about it. Compute it explicitly instead of emitting
        # a RuntimeWarning for an expected case; the distance layer handles NaN
        # pairwise from here.
        cols = []
        for members in candidates:
            sel = X[:, [labels.index(m) for m in members]] if members else X[:, :0]
            seen = ~np.isnan(sel)
            n_seen = seen.sum(axis=1)
            cols.append(np.where(n_seen > 0,
                                 np.nansum(sel, axis=1) / np.maximum(n_seen, 1),
                                 np.nan))
        centroids = np.column_stack(cols)
        Z = _linkage(_distance(centroids, "correlation" if k > 2 else "euclidean"),
                     "average")

    return PvclustResult(linkage=Z, labels=[f"cluster{i + 1}" for i in range(k)] if Z.size
                         else labels,
                         edges=edges, count=count, r=r_eff, nboot=nboot_vec,
                         method_dist=f"kmeans(k={k})", method_hclust="kmeans",
                         na_flag=na_flag, msfits=fits, cluster=cluster,
                         objects=labels)


def count_edges(X, candidates: Sequence[Sequence[str]],
                labels: Optional[Sequence[str]] = None, *,
                cluster: str = "columns",
                method_dist: str = "correlation", method_hclust: str = "average",
                method: str = "hclust", k: Optional[int] = None,
                jaccard: Optional[float] = None,
                nboot: int = 1000, r: Optional[Sequence[float]] = None,
                seed: int = 42, quiet: bool = True):
    """Count how often GIVEN clusters reappear in this project's bootstrap.

    The second pass of counts-mode federation. :func:`pvclust` counts against the
    clusters of its own tree, so two projects tally different things and their counts
    cannot be added. Here the candidate set is supplied -- normally the aggregator's
    federated catalogue -- so every project answers the same question and the counts
    become poolable.

    A project can count a cluster its own tree never produced; that is the point.

    Args:
        method: how each replicate is clustered -- ``hclust`` (default) or ``kmeans``.
            It must match how the catalogue was produced: counting hierarchical
            replicates against a k-means catalogue asks a different question at every
            project and the tallies would not be comparable.
        k: clusters per replicate, required for ``method="kmeans"``.
        jaccard: for ``method="kmeans"``, count a candidate as recovered when a
            replicate cluster reaches this Jaccard similarity instead of matching
            exactly. Effectively required above a few dozen objects -- exact k-means
            matching collapses to zero. It changes the estimand: the result is cluster
            stability, not the published AU p-value.

    Returns:
        ``(counts, r_eff, nboot_vec, na_flag)`` with ``counts`` shaped
        ``(len(candidates), len(r_eff))``.
    """
    X, labels, _units = orient(X, labels, cluster)
    n, p = X.shape
    ids = [edge_id(sorted(c)) for c in candidates]
    sizes, r_eff = effective_scales(n, r)

    if method not in ("hclust", "kmeans"):
        raise ValueError(f"method must be hclust or kmeans, got {method!r}")
    if method == "kmeans" and not k:
        raise ValueError("method='kmeans' needs k")

    if method == "hclust":
        def cluster_replicate(M):
            D = distance(M, method_dist)
            if not np.isfinite(D).all():
                return None
            return edge_members(linkage(D, method_hclust), labels)
    else:
        from sklearn.cluster import KMeans

        def cluster_replicate(M):
            A = np.nan_to_num(M.T)
            km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(A)
            return [sorted(labels[i] for i in np.where(km.labels_ == c)[0])
                    for c in range(k)]

    if jaccard is None:
        counts, na_flag = _tally(X, ids, cluster_replicate, sizes, int(nboot), seed,
                                 quiet, r_eff)
    else:
        # Relaxed matching cannot go through edge_id, so tally memberships directly.
        want = [set(c) for c in candidates]
        counts = np.zeros((len(candidates), len(sizes)), dtype=int)
        rng = np.random.default_rng(seed)
        na_flag = False
        for j, size in enumerate(sizes):
            for _ in range(int(nboot)):
                rep = cluster_replicate(X[rng.integers(0, n, size)])
                if rep is None:
                    na_flag = True
                    continue
                reps = [set(m) for m in rep]
                for i, cand in enumerate(want):
                    if any(len(cand & q) / len(cand | q) >= jaccard for q in reps):
                        counts[i, j] += 1

    return counts, r_eff, np.full(len(sizes), int(nboot)), na_flag
