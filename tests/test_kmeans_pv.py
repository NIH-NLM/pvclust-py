"""
Unit tests for kmeans_pv -- k-means clusters carrying AU p-values.

msfit never asks where a cluster came from, only whether it is present or absent in
each replicate, so k-means clusters can go through the same machinery. What needs
testing is that the k-means specifics behave: clusters partition the objects, the
permutation of k-means labels is genuinely irrelevant, and k is validated.
"""
from __future__ import annotations

import numpy as np
import pytest

from pvclust_py.core import kmeans_pv
from pvclust_py.datasets import load_lung


def _lung():
    df = load_lung()
    return df.to_numpy(float), list(df.columns)


@pytest.mark.parametrize("k", [2, 3, 5])
def test_clusters_partition_the_objects(k):
    """k-means assigns every object to exactly one cluster, so the reported clusters
    must tile the label set without overlap."""
    X, labels = _lung()
    res = kmeans_pv(X, labels, k=k, nboot=10, seed=1)

    assert len(res.edges) == k
    members = [set(e["members"]) for e in res.edges]
    assert set().union(*members) == set(labels)
    assert sum(len(m) for m in members) == len(labels), "clusters must not overlap"


def test_every_cluster_gets_the_full_set_of_pvalues():
    X, labels = _lung()
    res = kmeans_pv(X, labels, k=3, nboot=20, seed=1)
    for e in res.edges:
        for field in ("si", "au", "bp", "se_au", "v", "c", "pchi"):
            assert field in e
        assert 0.0 <= e["au"] <= 1.0 and 0.0 <= e["bp"] <= 1.0


def test_counts_are_bounded_by_nboot():
    X, labels = _lung()
    res = kmeans_pv(X, labels, k=3, nboot=15, seed=1)
    assert res.count.shape == (3, len(res.r))
    assert (res.count >= 0).all() and (res.count <= 15).all()


def test_label_permutation_between_runs_is_irrelevant():
    """k-means numbers its clusters arbitrarily. Identity by member set means that
    permutation never reaches the counts -- which is what makes k-means poolable
    across projects at all."""
    X, labels = _lung()
    a = kmeans_pv(X, labels, k=3, nboot=20, seed=1)
    b = kmeans_pv(X, labels, k=3, nboot=20, seed=1)
    assert ({frozenset(e["members"]) for e in a.edges}
            == {frozenset(e["members"]) for e in b.edges})
    assert {e["edge_id"] for e in a.edges} == {e["edge_id"] for e in b.edges}


def test_reproducible_for_a_fixed_seed():
    X, labels = _lung()
    a = kmeans_pv(X, labels, k=3, nboot=20, seed=42)
    b = kmeans_pv(X, labels, k=3, nboot=20, seed=42)
    np.testing.assert_array_equal(a.count, b.count)


def test_k_is_validated():
    X, labels = _lung()
    for bad in (1, 0, len(labels) + 1):
        with pytest.raises(ValueError, match="k must be between"):
            kmeans_pv(X, labels, k=bad, nboot=5, seed=1)


def test_jaccard_matching_is_at_least_as_permissive_as_exact():
    """Relaxed matching can only recognise more replicate clusters, never fewer, so
    BP cannot go down. (AU can move either way -- it depends on the SHAPE of the
    BP-vs-r curve, not its level, which is worth remembering before reading a rise
    in BP as a stronger result.)"""
    X, labels = _lung()
    exact = kmeans_pv(X, labels, k=3, nboot=40, seed=3)
    loose = kmeans_pv(X, labels, k=3, nboot=40, seed=3, jaccard=0.75)

    by_id = {e["edge_id"]: e for e in loose.edges}
    for e in exact.edges:
        assert by_id[e["edge_id"]]["bp"] >= e["bp"] - 1e-12


def test_exact_matching_degrades_as_the_object_count_grows():
    """Exact member-set matching collapses with more objects -- measured, not assumed.

    A larger cluster rarely reappears identically, so BP falls to zero and every AU
    with it. This is the practical limit of the exact-matching k-means path and the
    reason the jaccard option exists.
    """
    X, labels = _lung()
    small = kmeans_pv(X[:, :20], labels[:20], k=3, nboot=100, seed=1)
    large = kmeans_pv(X, labels, k=3, nboot=100, seed=1)

    assert max(e["bp"] for e in small.edges) > 0.5, "20 objects should still match"
    assert max(e["bp"] for e in large.edges) < 0.1, \
        f"{len(labels)} objects should degenerate under exact matching"


def test_degenerate_exact_matching_warns_with_the_remedy():
    """Silently returning AU = 0 for everything would be the worst outcome."""
    X, labels = _lung()
    with pytest.warns(UserWarning, match="jaccard"):
        kmeans_pv(X, labels, k=3, nboot=20, seed=1)


def test_jaccard_rescues_what_exact_matching_cannot_reach():
    """Where exact matching finds nothing, relaxed matching still fits a curve --
    a different estimand, but a usable one."""
    X, labels = _lung()
    loose = kmeans_pv(X, labels, k=3, nboot=100, seed=1, jaccard=0.75)
    assert all(e["df"] > 0 for e in loose.edges), "every cluster should fit"
    assert max(e["bp"] for e in loose.edges) > 0.5
