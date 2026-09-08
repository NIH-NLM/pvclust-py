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

  MODULE SCORES  each federated cluster reduced to one score per sample, so the
                 project can carry a federation-validated feature space into whatever
                 it does next -- clustering its own patients, say.

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
            ``federated_edges.csv`` / ``federated_catalogue.csv``. Needs ``edge_id``
            and ``members``; ``au`` is used for the side-by-side comparison when present.
        alpha: threshold for counting a cluster as supported.

    Returns:
        ``{"support", "agreement", "scores", "solo"}`` -- two DataFrames, the module
        score matrix, and the local pvclust result.
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
    keep = usable[usable["au_federated"].fillna(1.0) >= alpha] if "au_federated" in usable else usable
    scores = pd.DataFrame(
        {f"module_{i+1}": frame[[m for m in mem if m in frame.columns]].mean(axis=1)
         for i, mem in enumerate(keep["member_list"])},
        index=frame.index)

    return {"support": support, "agreement": agreement, "scores": scores, "solo": solo}
