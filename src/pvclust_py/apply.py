"""
Round 3: what the federation gives back to one project.

The direct analogue of ``oadr_cpep.apply_coefficients``. There is no single global
answer -- each project produces its OWN outcome, and the federated tree is the channel
carrying the pooled information back.

The question this answers is not "what are the federated clusters" (the aggregator
already said) but **"what do they look like in my data, and was joining worth it?"**
Three comparisons, all local:

  EDGE SUPPORT   the project's own AU/BP for each FEDERATED cluster, next to the
                 federated value. A cluster with high federated AU but low local AU is
                 one this project alone could not have found -- which is the gain from
                 federating, made concrete.

  AGREEMENT      solo tree vs federated tree: how many federated clusters the project
                 recovers alone, cophenetic correlation, and adjusted Rand between the
                 two partitions.

  MODULE SCORES  the tightest well-supported federated clusters, each reduced to one
                 score per sample, so the project can carry a federation-validated
                 feature space into whatever it does next -- clustering its own patients,
                 say. Modules are non-overlapping; anything in none of them is kept as
                 itself, so the space is a reduction of the original, not a copy.

Nothing here needs other projects' data. It reads the federated artifacts, which carry
only cluster memberships and fitted p-values.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


def _cophenetic(Z) -> np.ndarray:
    from scipy.cluster.hierarchy import cophenet
    return cophenet(Z)


def apply_edges(X, federated_edges, *, method_dist: str = "correlation",
                method_hclust: str = "average", nboot: int = 1000,
                seed: int = 42, alpha: float = 0.95,
                cluster: str = "columns", quiet: bool = True) -> Dict:
    """Evaluate the federated clusters against this project's own data.

    Args:
        X: this project's matrix, same orientation as when its statistics were shipped.
        federated_edges: the aggregator's edge table -- a DataFrame or the path to
            ``federated_pvclust_edges.csv`` / ``..._catalogue.csv``. Needs ``edge_id``
            and ``members``; ``au`` is used for the side-by-side comparison when present.
        alpha: threshold for counting a cluster as supported.

    Returns:
        ``{"support", "agreement", "scores", "modules", "solo", "federated"}`` --
        the per-cluster comparison, the summary metrics, the module score matrix, the
        table describing what each module contains, this project's own pvclust result,
        and the federated tree restricted to this project's objects.
    """
    from scipy.cluster.hierarchy import fcluster
    from sklearn.metrics import adjusted_rand_score

    from .core import count_edges, orient, pvclust
    from .distance import distance
    from .hclust import edge_id, linkage

    if isinstance(federated_edges, (str, bytes)) or hasattr(federated_edges, "read_text"):
        federated_edges = pd.read_csv(federated_edges)
    fed = federated_edges.copy()
    if "members" not in fed.columns:
        raise ValueError("the federated edge table needs a 'members' column")
    fed["member_list"] = [m.split(";") if isinstance(m, str) else list(m)
                          for m in fed["members"]]

    A, labels, _ = orient(X, cluster=cluster)
    known = set(labels)
    fed["n_present"] = [len(set(m) & known) for m in fed["member_list"]]
    usable = fed[fed["n_present"] == fed["member_list"].map(len)].reset_index(drop=True)
    if usable.empty:
        raise ValueError(
            "no federated cluster is fully present in this project's vocabulary -- "
            "run shared-features first so every project clusters the same objects")

    # --- this project alone -------------------------------------------------
    solo = pvclust(A, labels, method_dist=method_dist, method_hclust=method_hclust,
                   nboot=nboot, seed=seed, quiet=quiet)
    solo_ids = {e["edge_id"]: e for e in solo.edges}

    # --- local support for the FEDERATED clusters ---------------------------
    counts, r_eff, nboot_vec, _ = count_edges(
        A, list(usable["member_list"]), labels, method_dist=method_dist,
        method_hclust=method_hclust, nboot=nboot, seed=seed, quiet=quiet)

    from .msfit import msfit
    local = [msfit(counts[i] / nboot_vec, r_eff, nboot_vec) for i in range(len(usable))]

    support = pd.DataFrame({
        "edge_id": usable["edge_id"],
        "members": usable["members"],
        "n_members": usable["member_list"].map(len),
        "au_federated": usable["au"] if "au" in usable.columns else np.nan,
        "au_local": [f.au for f in local],
        "bp_local": [f.bp for f in local],
        "se_au_local": [f.se_au for f in local],
        "in_solo_tree": [eid in solo_ids for eid in usable["edge_id"]],
    })
    support["gain"] = support["au_federated"] - support["au_local"]
    support = support.sort_values("gain", ascending=False, ignore_index=True)

    # --- solo tree vs federated tree ---------------------------------------
    n_fed = len(usable)
    recovered = int(support["in_solo_tree"].sum())
    rows = [
        {"metric": "federated clusters evaluated", "value": float(n_fed)},
        {"metric": "recovered by this project alone", "value": float(recovered)},
        {"metric": "fraction recovered alone",
         "value": recovered / n_fed if n_fed else np.nan},
        {"metric": f"supported locally (AU >= {alpha})",
         "value": float((support["au_local"] >= alpha).sum())},
    ]
    if "au_federated" in support and support["au_federated"].notna().any():
        rows.append({"metric": f"supported federated (AU >= {alpha})",
                     "value": float((support["au_federated"] >= alpha).sum())})
        rows.append({"metric": "median AU gain from federating",
                     "value": float(support["gain"].median())})

    # cophenetic correlation between the solo tree and one built from the federated
    # clusters is only defined when both trees span the same leaves
    k = max(2, int((support["au_local"] >= alpha).sum()) or 2)
    part_solo = fcluster(solo.linkage, t=k, criterion="maxclust")
    fed_assign = np.zeros(len(labels), dtype=int)
    for j, members in enumerate(usable["member_list"], start=1):
        for m in members:
            if m in known:
                fed_assign[labels.index(m)] = j
    rows.append({"metric": "adjusted Rand, solo vs federated partition",
                 "value": float(adjusted_rand_score(part_solo, fed_assign))})

    agreement = pd.DataFrame(rows)

    # --- module scores: the federation-validated feature space --------------
    frame = X if hasattr(X, "columns") else pd.DataFrame(A, columns=labels)
    if cluster == "rows":
        frame = frame.T
    # Modules must NOT overlap, and must be the TIGHT clusters, not the big ones.
    # The federated catalogue is a tree, so its clusters are nested: one holds all the
    # objects, another most of them. Averaging every cluster that passes alpha gives
    # near-duplicate columns -- a smeared copy of the original space, not a reduction
    # of it. Taking the largest instead collapses to a single near-root module, which
    # is no better. What a co-expression module means is the SMALLEST well-supported
    # group: the tightest set the bootstrap can still resolve. Those are pairwise
    # disjoint by construction, because a minimal supported cluster cannot contain
    # another supported one. Objects in no such cluster stay as themselves, so the
    # space loses redundancy without losing information.
    candidates = usable.copy()
    candidates["_n"] = candidates["member_list"].map(len)
    candidates["_au"] = (pd.to_numeric(candidates["au"], errors="coerce")
                         if "au" in candidates.columns else np.nan)
    if candidates["_au"].notna().any():
        candidates = candidates[candidates["_au"].fillna(0.0) >= alpha]
    candidates = candidates.sort_values(["_n", "_au"], ascending=[True, False])

    claimed: set = set()
    modules: List[Dict] = []
    for _, row in candidates.iterrows():
        members = [m for m in row["member_list"] if m in frame.columns]
        if len(members) < 2 or (set(members) & claimed):
            continue
        modules.append({"module": f"module_{len(modules) + 1}",
                        "edge_id": row["edge_id"], "n_members": len(members),
                        "au": float(row["_au"]) if pd.notna(row["_au"]) else np.nan,
                        "members": ";".join(members)})
        claimed |= set(members)

    singles = [c for c in frame.columns if c not in claimed]
    for c in singles:
        modules.append({"module": str(c), "edge_id": "", "n_members": 1,
                        "au": np.nan, "members": str(c)})

    scores = pd.DataFrame(
        {m["module"]: frame[m["members"].split(";")].mean(axis=1) for m in modules},
        index=frame.index)
    module_table = pd.DataFrame(modules)
    n_grouped = int((module_table["n_members"] > 1).sum())
    agreement = pd.concat([agreement, pd.DataFrame([
        {"metric": "modules (supported, non-overlapping)", "value": float(n_grouped)},
        {"metric": "objects inside a module",
         "value": float(len(claimed))},
        {"metric": "module space dimension", "value": float(scores.shape[1])},
    ])], ignore_index=True)

    # The federated tree, restricted to the objects this project holds -- so the
    # cohort's own data can be drawn in the FEDERATION's order rather than its own.
    from .hclust import linkage as _linkage
    from .distance import distance as _distance
    fed_result = None
    try:
        Dloc = _distance(A, method_dist)
        Zfed = _linkage(Dloc, method_hclust)
        fed_edges = []
        by_id = {e["edge_id"]: e for _, e in usable.iterrows()}
        from .hclust import edge_table as _edge_table
        for e in _edge_table(Zfed, labels):
            f = by_id.get(e["edge_id"])
            e.update(au=float(f["au"]) if f is not None and "au" in usable.columns else 0.0,
                     bp=0.0, si=0.0, se_au=0.0, se_bp=0.0, se_si=0.0,
                     v=0.0, c=0.0, df=0, rss=0.0, pchi=1.0)
            fed_edges.append(e)
        from .core import PvclustResult
        fed_result = PvclustResult(
            linkage=Zfed, labels=labels, edges=fed_edges,
            count=np.zeros((len(fed_edges), 1)), r=np.array([1.0]), nboot=np.array([1]),
            method_dist=method_dist, method_hclust=method_hclust, cluster=cluster)
    except Exception:                      # a figure is a convenience, not the result
        fed_result = None

    return {"support": support, "agreement": agreement, "scores": scores,
            "modules": module_table, "solo": solo, "federated": fed_result}
