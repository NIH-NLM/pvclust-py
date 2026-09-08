"""
Clustering either axis: cluster="columns" (the default) and cluster="rows".

A two-way expression figure wants dendrograms on both axes, so both must work. They
are not statistically equivalent, though, and these tests pin that the difference is
surfaced rather than buried.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pvclust_py.core import kmeans_pv, orient, pvclust
from pvclust_py.datasets import load_lung


def _df():
    return load_lung().iloc[:, :12]        # 73 samples x 12 genes, small and quick


# ------------------------------------------------------------------- orient()
def test_orient_columns_leaves_the_matrix_alone():
    df = _df()
    A, labels, _ = orient(df, cluster="columns")
    assert A.shape == df.shape
    assert labels == [str(c) for c in df.columns]


def test_orient_rows_transposes_and_takes_labels_from_the_index():
    df = _df()
    A, labels, _ = orient(df, cluster="rows")
    assert A.shape == (df.shape[1], df.shape[0]), "objects must end up on the columns"
    assert labels == [str(i) for i in df.index]


def test_orient_handles_plain_arrays():
    X = np.arange(12, dtype=float).reshape(4, 3)
    A, labels, _ = orient(X, cluster="rows")
    assert A.shape == (3, 4) and labels == ["0", "1", "2", "3"]


def test_orient_rejects_a_bad_axis():
    with pytest.raises(ValueError, match="cluster must be"):
        orient(_df(), cluster="diagonal")


# ------------------------------------------------------------------- pvclust
def test_clustering_columns_gives_one_edge_per_internal_node_over_genes():
    df = _df()
    res = pvclust(df, nboot=20, seed=1)
    assert res.cluster == "columns"
    assert len(res.edges) == df.shape[1] - 1
    assert set(res.labels) == {str(c) for c in df.columns}


def test_clustering_rows_gives_a_dendrogram_over_samples():
    df = _df()
    with pytest.warns(UserWarning, match="anti-conservative"):
        res = pvclust(df, cluster="rows", nboot=20, seed=1)
    assert res.cluster == "rows"
    assert len(res.edges) == df.shape[0] - 1
    assert set(res.labels) == {str(i) for i in df.index}


def test_the_two_axes_cluster_different_things():
    """Guard against a transpose that silently does nothing."""
    df = _df()
    cols = pvclust(df, nboot=20, seed=1)
    with pytest.warns(UserWarning):
        rows = pvclust(df, cluster="rows", nboot=20, seed=1)
    assert not (set(cols.labels) & set(rows.labels))


def test_clustering_rows_warns_about_the_resampling_units():
    """Resampling analytes to assess sample clusters is anti-conservative. It is a
    legitimate half of a two-way figure, but it must not pass silently -- the AU
    values are optimistic and should not be quoted like the column ones."""
    with pytest.warns(UserWarning, match="correlated rather than independent"):
        pvclust(_df(), cluster="rows", nboot=10, seed=1)


def test_clustering_columns_does_not_warn_about_orientation():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        pvclust(_df(), nboot=10, seed=1)      # the sound direction: no complaint


def test_cluster_rows_matches_passing_the_transpose_by_hand():
    """The convenience must be exactly equivalent to transposing yourself."""
    df = _df()
    with pytest.warns(UserWarning):
        auto = pvclust(df, cluster="rows", nboot=20, seed=1)
    manual = pvclust(df.T, nboot=20, seed=1)

    assert ({e["edge_id"] for e in auto.edges} == {e["edge_id"] for e in manual.edges})
    np.testing.assert_array_equal(auto.count, manual.count)


# ------------------------------------------------------------------ kmeans_pv
def test_kmeans_respects_the_axis():
    df = _df()
    cols = kmeans_pv(df, k=3, nboot=10, seed=1)
    rows = kmeans_pv(df, k=3, cluster="rows", nboot=10, seed=1)

    assert set().union(*(set(e["members"]) for e in cols.edges)) == {str(c) for c in df.columns}
    assert set().union(*(set(e["members"]) for e in rows.edges)) == {str(i) for i in df.index}
    assert cols.cluster == "columns" and rows.cluster == "rows"
