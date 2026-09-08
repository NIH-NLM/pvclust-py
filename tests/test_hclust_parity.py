"""
Tree parity with R's hclust/hc2split, and the properties edge identity must have.

The second half tests edge_id rather than R agreement, because edge_id has no R
counterpart -- R names edges by merge order, which is meaningless across projects.
Those properties are what federation rests on.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pvclust_py.distance import distance
from pvclust_py.hclust import (compatible, edge_id, edge_members, edge_pattern,
                               edge_table, linkage)

FIXTURES = Path(__file__).parent / "fixtures"


def _lung_tree():
    X = pd.read_csv(FIXTURES / "lung_subset.csv", index_col=0,
                    float_precision="round_trip")
    labels = list(X.columns)
    Z = linkage(distance(X.to_numpy(float), "correlation"), "average")
    return Z, labels


def test_edge_sets_match_r():
    """The clusters R found and the clusters we find are the same clusters."""
    Z, labels = _lung_tree()
    mine = {edge_pattern(m, labels) for m in edge_members(Z, labels)}
    theirs = set((FIXTURES / "run_patterns.txt").read_text().split())
    assert mine == theirs


def test_merge_order_matches_r():
    """Stronger than set equality: scipy and R agree on the order of merges here,
    including tie-breaking. If this ever fails while test_edge_sets_match_r passes,
    the trees are the same but ordered differently -- content-hashed edge ids make
    that harmless, so relax this test rather than chasing scipy."""
    Z, labels = _lung_tree()
    mine = [edge_pattern(m, labels) for m in edge_members(Z, labels)]
    assert mine == (FIXTURES / "run_patterns.txt").read_text().split()


def test_merge_heights_match_r():
    Z, _ = _lung_tree()
    expected = pd.read_csv(FIXTURES / "run_height.csv",
                           float_precision="round_trip")["height"].to_numpy()
    np.testing.assert_allclose(Z[:, 2], expected, rtol=0, atol=1e-12)


def test_edge_id_depends_only_on_membership():
    """The federation property: the same cluster gets the same name everywhere,
    whatever order its members arrive in and whatever tree produced it."""
    a = ["gene_c", "gene_a", "gene_b"]
    assert edge_id(a) == edge_id(sorted(a)) == edge_id(list(reversed(a)))
    assert edge_id(a) != edge_id(a + ["gene_d"])
    assert edge_id(["x", "y"]) != edge_id(["x", "z"])


def test_edge_id_is_stable_across_label_orderings():
    """A project whose columns are in a different order must still name the shared
    clusters identically, or counts cannot be pooled."""
    X = pd.read_csv(FIXTURES / "lung_subset.csv", index_col=0,
                    float_precision="round_trip")
    Z1 = linkage(distance(X.to_numpy(float), "correlation"), "average")
    ids1 = {e["edge_id"] for e in edge_table(Z1, list(X.columns))}

    shuffled = X[list(X.columns[::-1])]
    Z2 = linkage(distance(shuffled.to_numpy(float), "correlation"), "average")
    ids2 = {e["edge_id"] for e in edge_table(Z2, list(shuffled.columns))}

    assert ids1 == ids2, "edge ids must not depend on column order"


def test_edge_table_shape():
    Z, labels = _lung_tree()
    table = edge_table(Z, labels)
    assert len(table) == len(labels) - 1
    assert table[-1]["n_members"] == len(labels), "the root contains every leaf"
    assert [e["merge_order"] for e in table] == list(range(1, len(labels)))


def test_compatible_means_nested_or_disjoint():
    """The rule the aggregator uses to assemble a consensus tree from edges
    contributed by different projects."""
    assert compatible(["a", "b"], ["c", "d"])          # disjoint
    assert compatible(["a", "b"], ["a", "b", "c"])     # nested
    assert compatible(["a", "b"], ["a", "b"])          # identical
    assert not compatible(["a", "b"], ["b", "c"])      # overlapping, incompatible


def test_ward_d_is_refused_not_approximated():
    Z, _ = _lung_tree()
    with pytest.raises(ValueError, match="ward.D2"):
        linkage(np.eye(3) * 0, "ward.D")


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="unknown hclust method"):
        linkage(np.zeros((3, 3)), "banana")
