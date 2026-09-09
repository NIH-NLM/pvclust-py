"""
Hierarchical clustering, edge extraction, and edge identity.

Ports R's ``hclust`` usage and ``hc2split`` (pvclust-internal.R). An *edge* is an
internal node of the dendrogram, and what identifies it is its SET OF MEMBER LEAVES
-- not its position in the tree.

WHY EDGE IDENTITY IS THE LOAD-BEARING DECISION
----------------------------------------------
R identifies edges by merge-order index, which is fine inside one run. It is useless
across projects: project A's edge 7 and project B's edge 7 are unrelated, and two
runs on the same data can order tied merges differently. Federation requires adding
counts for "the same cluster" across projects, so an edge needs a name derived from
its content:

    edge_id = sha1(",".join(sorted(members)))[:12]

Everything downstream depends on this. It is what lets a project count occurrences of
a cluster its own tree never produced, and it is why the aggregator can sum count
matrices at all.

R METHOD NAMES
--------------
R's ``ward.D2`` is scipy's ``ward``; R's ``mcquitty`` is scipy's ``weighted``. R's
``ward.D`` (the pre-3.1 behaviour, applying the Ward update to unsquared distances)
has no scipy equivalent and is refused rather than silently mapped to something else.
"""
from __future__ import annotations

import hashlib
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.cluster.hierarchy import linkage as _scipy_linkage
from scipy.spatial.distance import squareform

#: R hclust method -> scipy linkage method.
METHOD_MAP = {
    "average": "average",
    "complete": "complete",
    "single": "single",
    "centroid": "centroid",
    "median": "median",
    "mcquitty": "weighted",
    "ward.D2": "ward",
    "ward": "ward",          # R >= 3.1.0 maps bare "ward" to ward.D; see below
}

#: Refused rather than silently approximated.
UNSUPPORTED = {
    "ward.D": "R's ward.D applies the Ward update to unsquared distances and has no "
              "scipy equivalent; use ward.D2 (scipy's 'ward') and say so in the write-up",
}


def linkage(D: np.ndarray, method: str = "average") -> np.ndarray:
    """SciPy linkage matrix from a square ``p x p`` distance matrix.

    Args:
        D: symmetric distance matrix with zero diagonal, as produced by
            :mod:`pvclust_py.distance`.
        method: an R hclust method name (see :data:`METHOD_MAP`).

    Note:
        ``centroid``, ``median`` and ``ward`` are only meaningful for Euclidean
        distances -- as in R, nothing stops you passing a correlation distance, and
        as in R the result is hard to interpret if you do.
    """
    if method in UNSUPPORTED:
        raise ValueError(f"{method!r}: {UNSUPPORTED[method]}")
    if method not in METHOD_MAP:
        raise ValueError(f"unknown hclust method {method!r}; "
                         f"expected one of {sorted(METHOD_MAP)}")

    D = np.asarray(D, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"expected a square distance matrix, got shape {D.shape}")

    # Symmetrise and zero the diagonal before condensing: squareform is strict about
    # both, and our distances can differ in the last bit across the diagonal.
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)
    if not np.isfinite(D).all():
        raise ValueError("distance matrix contains non-finite entries")

    return _scipy_linkage(squareform(D, checks=False), method=METHOD_MAP[method])


def edge_members(Z: np.ndarray, labels: Sequence[str]) -> List[List[str]]:
    """Member labels of each internal node, in merge order.

    The direct analogue of R's ``hc2split``: row ``i`` of the linkage yields the set
    of leaves beneath it, sorted. Members are returned as labels rather than indices
    so that they mean the same thing in every project.
    """
    labels = list(labels)
    n = len(labels)
    if Z.shape[0] != n - 1:
        raise ValueError(f"linkage has {Z.shape[0]} merges but {n} labels given")

    members: List[List[str]] = []
    for a, b, *_ in Z:
        acc: List[str] = []
        for idx in (int(a), int(b)):
            acc.extend([labels[idx]] if idx < n else members[idx - n])
        members.append(sorted(acc))
    return members


def edge_id(members: Sequence[str]) -> str:
    """Stable content-derived name for a cluster.

    Two projects assign the same id to the same set of members regardless of tree
    shape, merge order, or label ordering -- which is what makes counts addable.
    """
    key = ",".join(sorted(members)).encode("utf-8")
    return hashlib.sha1(key).hexdigest()[:12]


def edge_pattern(members: Sequence[str], labels: Sequence[str]) -> str:
    """R's ``hc2split`` 0/1 membership string, over the original label order.

    Kept for cross-checking against R fixtures. Not used as an identity: it depends
    on column order, so it is not comparable across projects.
    """
    member_set = set(members)
    return "".join("1" if lab in member_set else "0" for lab in labels)


def edge_table(Z: np.ndarray, labels: Sequence[str]) -> List[Dict]:
    """One record per internal node: id, members, height, size, merge order."""
    members = edge_members(Z, labels)
    return [
        {
            "edge_id": edge_id(m),
            "members": m,
            "n_members": len(m),
            "height": float(Z[i, 2]),
            "merge_order": i + 1,          # 1-based, matching R
        }
        for i, m in enumerate(members)
    ]


def compatible(a: Sequence[str], b: Sequence[str]) -> bool:
    """True when two clusters can coexist in one tree: nested or disjoint.

    The test the aggregator uses to assemble a consensus from a catalogue of edges
    that came from different projects' trees.
    """
    sa, sb = set(a), set(b)
    return not (sa & sb) or sa <= sb or sb <= sa


def subtree_support(Z, edges, key: str = "au", how: str = "max"):
    """Support of the best (or average) cluster within each subtree.

    A leaf contains no cluster, so it gets ``-inf`` -- it should never win a
    comparison against a subtree that actually holds one.

    Returns a dict keyed as scipy indexes nodes: ``0..n-1`` are leaves, ``n+i`` is the
    cluster formed by merge ``i``.
    """
    Z = np.asarray(Z, dtype=float)
    n = Z.shape[0] + 1
    own = {n + i: float(e.get(key, 0.0) or 0.0) for i, e in enumerate(edges)}

    best, total, count = {}, {}, {}
    for i in range(Z.shape[0]):
        node = n + i
        vals = [own[node]]
        sums, cnts = own[node], 1
        for child in (int(Z[i, 0]), int(Z[i, 1])):
            if child >= n:
                vals.append(best[child])
                sums += total[child]
                cnts += count[child]
        best[node] = max(vals)
        total[node], count[node] = sums, cnts

    out = {i: float("-inf") for i in range(n)}          # leaves hold no cluster
    for node in best:
        out[node] = best[node] if how == "max" else total[node] / count[node]
    return out


def rotate_by_support(Z, edges, key: str = "au", how: str = "max"):
    """Rotate each merge so the better-supported side is drawn first.

    Swapping a merge's two children is a **rotation**, not a reordering: the tree is
    unchanged and every cluster keeps its members. Only the left-to-right layout
    moves, so a heatmap drawn in this order reads strongest-first without altering
    any result.

    The comparison is on the best cluster ANYWHERE in each subtree, not on the two
    children's own values. Comparing children directly does almost nothing: a leaf has
    no support of its own, so leaf-versus-leaf always ties and leaf-versus-cluster
    compares a real number against a fallback. Looking down the whole subtree means
    the branch containing the single best-supported cluster is the one that goes
    first, all the way from the root.

    Args:
        Z: linkage matrix.
        edges: per-edge records in merge order, carrying ``key``.
        key: which support value to sort on -- ``au``, ``bp`` or ``si``.
        how: ``max`` puts the branch holding the single best cluster first;
            ``mean`` favours the branch that is well supported throughout.

    Returns:
        A new linkage matrix. scipy's ``dendrogram`` draws a merge's first child on
        the left.
    """
    Z = np.asarray(Z, dtype=float).copy()
    strength = subtree_support(Z, edges, key=key, how=how)

    for i in range(Z.shape[0]):
        a, b = int(Z[i, 0]), int(Z[i, 1])
        if strength[b] > strength[a]:
            Z[i, 0], Z[i, 1] = Z[i, 1], Z[i, 0]
    return Z
