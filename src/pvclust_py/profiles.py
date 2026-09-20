"""
Federating a clustering of PATIENTS, where the exact path does not exist.

Everything in :mod:`pvclust_py.aggregate` rests on one fact: the four sufficient
statistics are sums over rows, so summing them across projects reproduces exactly what
pooled rows would give. That works because the **objects** are features, shared by
every project, and the **rows** are subjects, disjoint between them.

Flip the orientation and it collapses. Clustering patients makes the patients the
objects, and projects hold disjoint patients, so there is no shared object vocabulary
at all. A cluster is then a set of subject ids, which is subject-level data by
definition, and the distance matrix is subject by subject, which is also subject-level.
The additive direction now runs over features, which would need the same subjects
measured on different features at each site -- vertical partitioning, not the
horizontal partitioning a multi-cohort study actually has.

So there is no pooled patient tree and there cannot be one. This module does the thing
that *is* possible.

Each project clusters its own patients and ships, per cluster, the **mean profile over
its members** and the **number of members**. That is an average over patients, the same
object as pseudobulk, and it carries no subject id. The aggregator matches those
profiles across projects and reports which clusters correspond.

What federation buys here is **replication, not a pooled p-value**. An endotype found
independently at two projects is more credible than one found at one. With a handful of
projects you cannot bootstrap projects, so no approximately-unbiased value is available
at this level and none is invented.

**Two release rules, both enforced by** :func:`project_profiles`.

Anything a project releases from one dataset is ONE release. Ship the partition you
chose, not every k you tried: two partitions of the same patients that differ by one
member reveal that member's whole profile by subtraction, however large the clusters
are. Cluster size has never been the guarantee -- Homer et al. (2008) recovered
participants from allele frequencies pooled over roughly a thousand people.

A partition with any cluster below the floor is suppressed **whole**. Dropping just the
small cluster announces that a cluster below the floor existed, and the remaining
profiles plus the total count reconstruct its mean.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


class SuppressedError(RuntimeError):
    """Raised when a partition may not be released."""


def project_profiles(X, assignment, *, project: str, min_cluster: int = 10,
                     annotations: Optional[pd.DataFrame] = None) -> Dict:
    """One project's releasable summary of its own patient clusters.

    Args:
        X: this project's matrix, patients as rows, features as columns.
        assignment: cluster id per patient, indexed like ``X``.
        project: project id, stamped into the output.
        min_cluster: refuse to release a partition containing any cluster smaller
            than this. See the module docstring for why the whole partition goes
            rather than just the offending cluster.
        annotations: optional per-patient metadata. Only counts per level are
            summarised, never a value that could belong to one patient.

    Returns:
        ``{"profiles", "counts", "demographics"}``. ``profiles`` is
        ``clusters x features``; nothing in it is a subject.

    Raises:
        SuppressedError: if any cluster is below ``min_cluster``.
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    assignment = pd.Series(assignment).reindex(X.index)
    if assignment.isna().any():
        raise ValueError("every patient needs a cluster assignment")

    sizes = assignment.value_counts().sort_index()
    small = sizes[sizes < min_cluster]
    if len(small):
        raise SuppressedError(
            f"{project}: clusters {list(small.index)} hold {list(small.values)} "
            f"patients, below the floor of {min_cluster}. The WHOLE partition is "
            f"suppressed, not just those clusters -- releasing the rest would "
            f"announce that a cluster below the floor existed, and the remaining "
            f"profiles plus the total count reconstruct its mean. Use a smaller k.")

    profiles = X.groupby(assignment.to_numpy()).mean()
    profiles.index = [f"{project}:c{c}" for c in profiles.index]

    counts = pd.DataFrame({"cluster": profiles.index,
                           "n_patients": sizes.to_numpy(),
                           "project": project})

    demo = None
    if annotations is not None:
        rows = []
        ann = annotations.reindex(X.index)
        for cid, idx in assignment.groupby(assignment).groups.items():
            sub = ann.loc[idx]
            for col in ann.columns:
                for level, n in sub[col].astype(str).value_counts().items():
                    rows.append({"cluster": f"{project}:c{cid}", "variable": col,
                                 "level": level, "n": int(n), "project": project})
        demo = pd.DataFrame(rows)
        # A level held by fewer than the floor is itself a small cell.
        demo = demo[demo["n"] >= min_cluster]

    return {"profiles": profiles, "counts": counts, "demographics": demo}


def _centred(profiles: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Stack every project's profiles, each centred on ITS OWN per-feature mean.

    Centring is not cosmetic, it is what makes the comparison valid. A profile over
    thousands of proteins is dominated by which proteins are abundant, and abundance
    spans about five orders of magnitude, so any two profiles correlate at 0.98
    whether or not they are the same endotype -- measured on the SLE cohorts, all
    sixteen cross-cohort pairs fell between 0.91 and 0.99, a spread of 0.08. Going
    non-parametric does not help: Spearman on raw profiles gave 0.07, because the rank
    order of abundance is just as shared. Centring takes the spread to 1.86.

    Each project is centred on its own mean rather than a common one. Here cohort and
    batch are the same thing, so a project's per-feature mean IS its batch offset, and
    subtracting it removes the batch shift at the one point where projects are
    compared.
    """
    shared = list(profiles[0].columns)
    for q in profiles[1:]:
        shared = [c for c in shared if c in q.columns]
    if not shared:
        raise ValueError("the projects share no features, so their cluster profiles "
                         "are not comparable")
    return pd.concat([q[shared] - q[shared].mean(axis=0) for q in profiles])


def match_profiles(profiles: Sequence[pd.DataFrame], *,
                   method_dist: str = "minkowski",
                   method_hclust: str = "ward.D2",
                   nboot: int = 1000, seed: int = 42, quiet: bool = True):
    """Cluster every project's patient profiles together, with AU p-values.

    The profiles are the objects and the features are the resampling units, so this is
    :func:`~pvclust_py.core.pvclust` with ``cluster="rows"``, using the same distance
    and linkage as the rest of the pipeline rather than a second mechanism invented
    for this step.

    A correspondence is then a cluster of the resulting tree that holds profiles from
    more than one project, and its AU says how reliably the two sit together.

    Warning:
        These AU values are **anti-conservative**. The resampling units are features,
        and features are co-expressed rather than independent draws, so resampling them
        as if independent understates the variability and AU comes out too high. It is
        a real number computed correctly, and it is optimistic. Report it as such, not
        as the published quantity.

    Returns:
        ``(result, similarity)`` -- the pvclust result over the pooled centred
        profiles, and the cross-project correlation matrix kept as a diagnostic.
    """
    from .core import pvclust

    if len(profiles) < 2:
        raise ValueError("matching needs at least two projects")

    P = _centred(profiles)
    result = pvclust(P, cluster="rows", method_dist=method_dist,
                     method_hclust=method_hclust, nboot=nboot, seed=seed, quiet=quiet)

    n0 = len(profiles[0])
    S = np.corrcoef(P.to_numpy(float))[:n0, n0:]
    similarity = pd.DataFrame(S, index=P.index[:n0], columns=P.index[n0:])
    return result, similarity


def replication(result, counts: pd.DataFrame, alpha: float = 0.95) -> pd.DataFrame:
    """Which clusters replicated across projects, and how well supported that is.

    A cluster replicates when the tree puts it together with a cluster from another
    project. The AU on that grouping is reported beside it, with the caveat from
    :func:`match_profiles` attached.

    This is the federated statement, and it is deliberately **not** a pooled p-value
    over patients. Projects hold disjoint patients; there is no pooled patient tree.
    """
    n = dict(zip(counts["cluster"], counts["n_patients"]))

    def project_of(label):
        return str(label).split(":")[0]

    # the smallest edge that puts this profile with a profile from another project
    best = {}
    for e in sorted(result.edges, key=lambda x: x["n_members"]):
        members = list(e["members"])
        if len({project_of(m) for m in members}) < 2:
            continue
        for m in members:
            best.setdefault(m, e)

    rows = []
    for cluster in counts["cluster"]:
        e = best.get(cluster)
        partners = [m for m in (e["members"] if e else []) if m != cluster]
        rows.append({
            "cluster": cluster,
            "n_patients": n.get(cluster, 0),
            "replicated": bool(e is not None and e["au"] >= alpha),
            "matched_with": ";".join(partners),
            "group_size": int(e["n_members"]) if e else 0,
            "au": float(e["au"]) if e else np.nan,
        })
    return pd.DataFrame(rows)
