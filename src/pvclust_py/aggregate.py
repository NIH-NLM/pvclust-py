"""
Aggregator steps: combine what projects ship into a federated result.

Mirrors ``oadr_cpep.aggregate``. Two things pool, by two different rules, and they
are independent -- you can do either without the other.

  shared_features   the vocabulary every project has, as oadr-cpep intersects
                    per-site feature selections.

  federated_tree    POOL SUFFICIENT STATISTICS.  The p x p distance matrix is this
                    method's analogue of a coefficient vector, and unlike FedAvg on
                    coefficients it is EXACT: adding per-project statistics gives
                    bit-for-bit the distance matrix of the pooled raw data, so the
                    federated dendrogram *is* the centralised dendrogram.

  federated_edges   POOL BOOTSTRAP COUNTS.  A forest cannot be averaged, so oadr-cpep
                    unions forests and pools their votes; the multiscale bootstrap is
                    likewise an ensemble, and its votes are the count matrix. msfit
                    consumes only counts, so the federated AU comes out of the
                    identical function -- no new statistics, no approximation.

Counts from projects at the same scale are SUMMED; projects of different sizes have
different effective scales, so theirs are CONCATENATED into one scatter and the same
two-parameter curve is fitted through all of it. msfit already accepts a per-scale
nboot vector, so this needs no change to the fitting code.

Nothing here reads subject-level data.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from .distance import distance_from_stats, pool_stats
from .hclust import compatible, edge_id, edge_table, linkage
from .msfit import msfit


def shared_features(vocabularies: Iterable[Sequence[str]]) -> List[str]:
    """The intersection of the per-project feature vocabularies.

    Clustering can only be federated over objects every project measures; anything
    else has no shared meaning. Returns them sorted, for a stable column order.
    """
    vocabs = [set(v) for v in vocabularies]
    if not vocabs:
        raise ValueError("no vocabularies given")
    return sorted(set.intersection(*vocabs))


def federated_tree(stats: Iterable[Mapping[str, np.ndarray]], labels: Sequence[str], *,
                   method_dist: str = "correlation", method_hclust: str = "average"):
    """Pool sufficient statistics into the federated distance matrix and tree.

    Exact: the result equals what you would get with all the raw data in one place.

    Returns:
        ``(D, Z, edges)`` -- pooled distance, linkage, and the edge table that becomes
        the catalogue projects count against.
    """
    pooled = pool_stats(stats)
    D = distance_from_stats(pooled, method_dist)
    Z = linkage(D, method_hclust)
    return D, Z, edge_table(Z, labels)


def pool_counts(frames: Iterable, *, rescale: bool = True) -> "object":
    """Combine per-project long-form counts into one table.

    Each frame has columns ``edge_id, r, nboot, count`` and, when ``rescale`` is on,
    ``n`` -- the project's own row count.

    RESCALING IS NOT OPTIONAL FOR CORRECTNESS. Each project's ``r`` is relative to
    ITS OWN n, so "r = 1.0" means 60 samples at one project and 15 at another. Those
    are different scales and summing them as if they matched is simply wrong. With
    ``rescale``, each measurement is re-expressed against the pooled n:

        size      = r_project * n_project          (absolute rows drawn)
        r_pooled  = size / sum(n_project)

    Rows agreeing on ``(edge_id, r_pooled)`` are then summed; the rest survive side by
    side as separate measurements of the same curve.

    Note:
        Rescaling means no project may reach ``r_pooled = 1``, since none holds all
        the rows. The pooled scatter can therefore sit entirely below 1, making the
        extrapolation to sigma^2 = -1 longer than in a centralised run. Whether that
        is a net gain -- more scales and wider spread -- or a net loss is an open
        question this implementation does not settle.
    """
    import pandas as pd
    frames = list(frames)
    if not frames:
        raise ValueError("no count frames to pool")
    allc = pd.concat(frames, ignore_index=True)
    missing = {"edge_id", "r", "nboot", "count"} - set(allc.columns)
    if missing:
        raise ValueError(f"count frames missing columns {sorted(missing)}")

    if rescale:
        if "n" not in allc.columns:
            raise ValueError(
                "pool_counts(rescale=True) needs an 'n' column giving each project's "
                "row count, so scales can be expressed against the pooled n. Pass "
                "rescale=False only when every project has the same n.")
        n_total = sum(f["n"].iloc[0] for f in frames)
        allc = allc.assign(r=(allc["r"] * allc["n"]).round().astype(int) / n_total)
        allc = allc.drop(columns="n")

    grouped = (allc.groupby(["edge_id", "r"], as_index=False)[["nboot", "count"]].sum()
                   .sort_values(["edge_id", "r"], ignore_index=True))
    if "project" in allc.columns:
        # Carry project coverage through so federated_edges can warn when clusters
        # were not measured by everyone.
        cover = allc.groupby("edge_id")["project"].agg(lambda x: ";".join(sorted(set(x))))
        grouped["project"] = grouped["edge_id"].map(cover)
    return grouped


def federated_edges(pooled_counts, catalogue: Optional[Sequence[Mapping]] = None):
    """Run msfit on the pooled counts -- the federated AU per cluster.

    Args:
        pooled_counts: output of :func:`pool_counts`.
        catalogue: optional edge records (with ``members``) to attach names to ids.

    Returns:
        DataFrame with si/au/bp, their standard errors, v, c, pchi, and the number of
        scale measurements the fit used.
    """
    import pandas as pd
    members = {e["edge_id"]: e["members"] for e in (catalogue or [])}

    # Counts are only addable when every project counted the SAME clusters. If each
    # ran pvclust on its own data, their trees differ and most edges are measured by
    # one project alone -- pooling those produces a number with no meaning. Catch it
    # here rather than letting a silently wrong federated AU through.
    coverage = {}
    if "project" in pooled_counts.columns:
        # pool_counts joins contributing projects into "A;B", so nunique() on the
        # column counts STRINGS, not projects. Split them back apart.
        for eid, joined in pooled_counts.groupby("edge_id")["project"].agg("first").items():
            coverage[eid] = {p for part in str(joined).split(";") for p in [part] if part}
        all_projects = set().union(*coverage.values()) if coverage else set()
        n_projects = len(all_projects)
        partial = sum(1 for v in coverage.values() if len(v) < n_projects)
        if n_projects > 1 and partial:
            import warnings
            warnings.warn(
                f"{partial} of {len(coverage)} clusters were measured by fewer than "
                f"all {n_projects} projects. Counts are only poolable when every "
                f"project counts the SAME candidate clusters -- run the two-pass "
                f"protocol: build the catalogue with aggregate-trees --stats, then "
                f"have each project run count-edges against it. Pooling per-project "
                f"trees gives a federated AU that means nothing.", stacklevel=2)

    rows = []
    for eid, g in pooled_counts.groupby("edge_id"):
        g = g.sort_values("r")
        nboot = g["nboot"].to_numpy(float)
        f = msfit(g["count"].to_numpy(float) / nboot, g["r"].to_numpy(float), nboot)
        rows.append({
            "edge_id": eid,
            "members": ";".join(members.get(eid, [])),
            "n_members": len(members.get(eid, [])) or None,
            "si": f.si, "au": f.au, "bp": f.bp,
            "se_si": f.se_si, "se_au": f.se_au, "se_bp": f.se_bp,
            "v": f.v, "c": f.c, "df": f.df, "pchi": f.pchi,
            "n_scales": len(g), "total_nboot": int(nboot.sum()),
            "n_projects": len(coverage[eid]) if eid in coverage else None,
        })
    return pd.DataFrame(rows).sort_values("au", ascending=False, ignore_index=True)


def consensus_clusters(edges, alpha: float = 0.95, use: str = "au") -> List[Dict]:
    """Assemble a consistent set of clusters from a catalogue that came from many trees.

    Used in counts mode, where there is no single pooled dendrogram: candidate edges
    are contributed by different projects and may conflict. Accept them in descending
    support, keeping only those compatible (nested or disjoint) with everything
    already accepted -- a majority-rule style consensus.

    Yields a consensus FOREST, not necessarily a full binary tree. Exact mode gives a
    real pooled dendrogram; this does not, and the write-up should say so.
    """
    records = edges.to_dict("records") if hasattr(edges, "to_dict") else list(edges)

    def _mem(e):
        m = e["members"]
        return m.split(";") if isinstance(m, str) else list(m)

    # Exclude the root. It contains every object, always scores AU = 1, and reporting
    # it makes "the consensus clusters" mean "everything" -- the same trap pvpick has.
    universe = set().union(*(set(_mem(e)) for e in records)) if records else set()
    passing = [e for e in records
               if e.get(use, 0) >= alpha and e.get("members")
               and set(_mem(e)) != universe]
    passing.sort(key=lambda e: -e[use])

    accepted: List[Dict] = []
    for e in passing:
        m = _mem(e)
        if all(compatible(m, a["_members"]) for a in accepted):
            accepted.append({**e, "_members": m})
    return accepted

