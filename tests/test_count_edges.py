"""
Unit tests for count_edges -- the second pass of counts-mode federation.

pvclust() counts against the clusters of its OWN tree, so two projects tally
different things and their counts cannot be added. count_edges takes the candidate
set as an argument so every project answers the same question. The tests below pin
that property, since losing it breaks pooling silently rather than loudly.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from pvclust_py.core import count_edges, pvclust
from pvclust_py.datasets import load_cohorts, load_lung
from pvclust_py.hclust import edge_id
from pvclust_py.scales import effective_scales


def _lung():
    df = load_lung()
    return df.to_numpy(float), list(df.columns)


def test_counts_have_one_row_per_candidate_and_scale():
    X, labels = _lung()
    candidates = [labels[:3], labels[3:7]]
    counts, r_eff, nboot, _ = count_edges(X, candidates, labels, nboot=20, seed=1)

    assert counts.shape == (len(candidates), len(r_eff))
    assert len(nboot) == len(r_eff)
    assert (counts >= 0).all() and (counts <= 20).all()


def test_scales_come_from_effective_scales_not_the_nominal_r():
    X, labels = _lung()
    _, r_eff, _, _ = count_edges(X, [labels[:2]], labels, nboot=5, seed=1)
    expected_sizes, expected_r = effective_scales(len(X))
    np.testing.assert_allclose(r_eff, expected_r, rtol=0, atol=0)


def test_a_project_can_count_a_cluster_its_own_tree_never_produced():
    """The whole point of the two-pass protocol: candidates come from the federated
    catalogue, so a project reports on clusters it would not have found alone. Such a
    cluster must appear in the output (with whatever count, possibly 0) rather than
    being dropped."""
    X, labels = _lung()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        own = pvclust(X, labels, nboot=20, seed=1)
    own_ids = {e["edge_id"] for e in own.edges}

    # a deliberately arbitrary grouping, essentially certain not to be in the tree
    foreign = [labels[0], labels[5], labels[11], labels[17]]
    assert edge_id(sorted(foreign)) not in own_ids

    counts, r_eff, _, _ = count_edges(X, [foreign], labels, nboot=20, seed=1)
    assert counts.shape == (1, len(r_eff))


def test_candidate_order_does_not_change_a_cluster_s_counts():
    """Members are hashed as a sorted set, so how the caller happens to order them
    is irrelevant -- otherwise projects could disagree about identity."""
    X, labels = _lung()
    c1, _, _, _ = count_edges(X, [labels[:4]], labels, nboot=30, seed=7)
    c2, _, _, _ = count_edges(X, [list(reversed(labels[:4]))], labels, nboot=30, seed=7)
    np.testing.assert_array_equal(c1, c2)


def test_counts_are_reproducible_for_a_fixed_seed():
    X, labels = _lung()
    kw = dict(labels=labels, nboot=25, seed=99)
    a, *_ = count_edges(X, [labels[:3]], **kw)
    b, *_ = count_edges(X, [labels[:3]], **kw)
    np.testing.assert_array_equal(a, b)


def test_counts_agree_with_pvclust_on_a_shared_candidate():
    """Given the same seed and the same cluster, counting against a supplied
    catalogue must reproduce what pvclust tallied for its own edge."""
    X, labels = _lung()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = pvclust(X, labels, nboot=40, seed=5)

    target = res.edges[0]
    counts, _, _, _ = count_edges(X, [target["members"]], labels, nboot=40, seed=5)
    np.testing.assert_array_equal(counts[0], res.count[0])
