"""
Unit tests for the aggregator: what projects ship, and what pooling does with it.

The pooling rules carry real weight -- one of them (scale rescaling) is the
difference between counts-mode federation working and finding nothing at all -- so
they are tested directly rather than only through the notebooks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pvclust_py.aggregate import (consensus_clusters, federated_edges,
                                  federated_tree, pool_counts, shared_features)
from pvclust_py.datasets import load_cohorts, load_lung
from pvclust_py.distance import distance, pairwise_stats
from pvclust_py.hclust import edge_table, linkage


# ------------------------------------------------------------ shared_features
def test_shared_features_is_the_intersection():
    assert shared_features([["a", "b", "c"], ["b", "c", "d"], ["c", "b"]]) == ["b", "c"]


def test_shared_features_is_sorted_for_a_stable_column_order():
    """Projects must agree on column ORDER as well as content, or their statistic
    matrices are not addable element-wise."""
    assert shared_features([["z", "a", "m"], ["m", "z", "a"]]) == ["a", "m", "z"]


def test_shared_features_rejects_nothing_to_intersect():
    with pytest.raises(ValueError, match="no vocabularies"):
        shared_features([])


def test_disjoint_vocabularies_give_nothing():
    assert shared_features([["a"], ["b"]]) == []


# -------------------------------------------------------------- federated_tree
def test_federated_tree_equals_the_centralised_tree():
    """The exactness claim, at the level the aggregator actually produces."""
    full = load_lung()
    labels = list(full.columns)
    cohorts = load_cohorts(3)

    D_fed, Z_fed, cat = federated_tree(
        [pairwise_stats(df.to_numpy(float)) for df in cohorts.values()], labels)

    D_central = distance(full.to_numpy(float), "correlation")
    Z_central = linkage(D_central, "average")

    np.testing.assert_allclose(D_fed, D_central, rtol=0, atol=1e-12)
    assert ({e["edge_id"] for e in cat}
            == {e["edge_id"] for e in edge_table(Z_central, labels)})


# ----------------------------------------------------------------- pool_counts
def _frame(edge, n, rs, counts, nboot=100):
    return pd.DataFrame({"edge_id": edge, "r": rs, "n": n,
                         "nboot": nboot, "count": counts})


def test_pool_counts_rescales_r_to_the_pooled_n():
    """Each project's r is relative to ITS OWN n, so r=1.0 means 80 rows at one
    project and 20 at another. Pooling must put them on a common footing:
    r_pooled = round(r*n) / sum(n). Adding raw r values would be meaningless."""
    a = _frame("e1", 80, [0.5, 1.0], [40, 70])
    b = _frame("e1", 20, [0.5, 1.0], [10, 18])

    pooled = pool_counts([a, b])          # n_total = 100
    got = sorted(pooled["r"].round(6).tolist())
    # 0.5*80=40 -> .40 ; 1.0*80=80 -> .80 ; 0.5*20=10 -> .10 ; 1.0*20=20 -> .20
    assert got == [0.10, 0.20, 0.40, 0.80]


def test_pool_counts_sums_only_measurements_at_the_same_pooled_scale():
    """Equal-sized projects land on identical scales, so their replicates add --
    nboot as well as count."""
    a = _frame("e1", 50, [0.5, 1.0], [30, 45])
    b = _frame("e1", 50, [0.5, 1.0], [20, 40])

    pooled = pool_counts([a, b]).sort_values("r", ignore_index=True)
    assert len(pooled) == 2, "same-sized projects must collapse onto shared scales"
    assert pooled["count"].tolist() == [50, 85]
    assert pooled["nboot"].tolist() == [200, 200]


def test_pool_counts_keeps_distinct_scales_side_by_side():
    """Unequal projects contribute measurements at different scales; they must
    survive as separate points on the curve, not be merged."""
    pooled = pool_counts([_frame("e1", 80, [0.5], [40]), _frame("e1", 20, [0.5], [10])])
    assert len(pooled) == 2


def test_pool_counts_requires_n_when_rescaling():
    bad = _frame("e1", 50, [0.5], [30]).drop(columns="n")
    with pytest.raises(ValueError, match="needs an 'n' column"):
        pool_counts([bad])


def test_pool_counts_without_rescaling_leaves_r_alone():
    """rescale=False is only valid when every project has the same n."""
    a = _frame("e1", 50, [0.5, 1.0], [30, 45])
    b = _frame("e1", 50, [0.5, 1.0], [20, 40])
    pooled = pool_counts([a, b], rescale=False)
    assert sorted(pooled["r"].tolist()) == [0.5, 1.0]


def test_pool_counts_rejects_malformed_frames():
    with pytest.raises(ValueError, match="missing columns"):
        pool_counts([pd.DataFrame({"edge_id": ["e"], "r": [0.5]})])
    with pytest.raises(ValueError, match="no count frames"):
        pool_counts([])


# -------------------------------------------------------------- federated_edges
def test_federated_edges_produces_one_row_per_cluster_with_pvalues():
    frames = [_frame("e1", 50, [0.5, 0.8, 1.0, 1.2], [20, 25, 30, 34]),
              _frame("e2", 50, [0.5, 0.8, 1.0, 1.2], [80, 85, 90, 95], nboot=100)]
    out = federated_edges(pool_counts(frames))

    assert set(out["edge_id"]) == {"e1", "e2"}
    for col in ("si", "au", "bp", "se_au", "v", "c", "pchi", "n_scales"):
        assert col in out.columns
    assert ((out["au"] >= 0) & (out["au"] <= 1)).all()
    assert out["au"].is_monotonic_decreasing, "rows should come back sorted by AU"


def test_federated_edges_attaches_members_from_the_catalogue():
    frame = _frame("e1", 50, [0.5, 0.8, 1.0], [20, 25, 30])
    out = federated_edges(pool_counts([frame]),
                          catalogue=[{"edge_id": "e1", "members": ["a", "b"]}])
    assert out.loc[0, "members"] == "a;b"
    assert out.loc[0, "n_members"] == 2


# ---------------------------------------------------------- consensus_clusters
def _edges(rows):
    return pd.DataFrame([{"edge_id": e, "members": ";".join(m), "n_members": len(m),
                          "au": au} for e, m, au in rows])


def test_consensus_excludes_the_root():
    """The root contains every object and always scores 1. Returning it makes
    'the consensus clusters' mean 'everything' -- the same trap pvpick has."""
    edges = _edges([("root", ["a", "b", "c", "d"], 1.0),
                    ("e1", ["a", "b"], 0.99)])
    got = consensus_clusters(edges, alpha=0.95)
    assert [e["edge_id"] for e in got] == ["e1"]


def test_consensus_keeps_only_compatible_clusters():
    """Candidates come from different projects' trees and can conflict. Overlapping
    but non-nested clusters cannot coexist in one tree, so the weaker is dropped."""
    edges = _edges([("universe", ["a", "b", "c", "d"], 1.0),
                    ("strong", ["a", "b"], 0.99),
                    ("conflict", ["b", "c"], 0.97),      # overlaps 'strong', not nested
                    ("nested", ["a", "b", "c"], 0.96)])  # contains 'strong' -> allowed
    got = [e["edge_id"] for e in consensus_clusters(edges, alpha=0.95)]
    assert "strong" in got and "nested" in got
    assert "conflict" not in got


def test_consensus_respects_alpha():
    edges = _edges([("universe", ["a", "b", "c"], 1.0), ("weak", ["a", "b"], 0.80)])
    assert consensus_clusters(edges, alpha=0.95) == []
    assert [e["edge_id"] for e in consensus_clusters(edges, alpha=0.75)] == ["weak"]


def test_consensus_accepts_records_as_well_as_a_frame():
    rows = _edges([("universe", ["a", "b", "c"], 1.0),
                   ("e1", ["a", "b"], 0.99)]).to_dict("records")
    assert [e["edge_id"] for e in consensus_clusters(rows, alpha=0.95)] == ["e1"]
