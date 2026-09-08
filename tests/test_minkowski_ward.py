"""
Parity for the configuration used in practice here: Minkowski distance with ward.D2
linkage, the pairing carried over from microarray and RNAseq work.

The headline fact this file pins: in pvclust, **minkowski IS euclidean**. R's
``dist.pvclust`` calls ``dist(t(x), method)`` and never passes ``p``, so R's default
``p = 2`` stands. Verified against R with a maximum difference of exactly 0.0.
That is good news for federation -- it means the preferred metric pools exactly.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pvclust_py.distance import (METHODS, NOT_GRAM_DERIVABLE, distance,
                                 distance_from_stats, minkowski_distance,
                                 pairwise_stats, pool_stats)
from pvclust_py.hclust import edge_members, edge_pattern, linkage

FIXTURES = Path(__file__).parent / "fixtures"


def _lung():
    df = pd.read_csv(FIXTURES / "lung_subset.csv", index_col=0,
                     float_precision="round_trip")
    return df.to_numpy(float), list(df.columns)


def _expected(name):
    return pd.read_csv(FIXTURES / name, index_col=0,
                       float_precision="round_trip").to_numpy(float)


def test_minkowski_matches_r():
    X, _ = _lung()
    np.testing.assert_allclose(distance(X, "minkowski"), _expected("dist_minkowski.csv"),
                               rtol=0, atol=1e-12)


def test_minkowski_is_euclidean_in_pvclust():
    """Not an approximation -- the same numbers. pvclust cannot reach another
    exponent, because it never forwards p to dist()."""
    X, _ = _lung()
    np.testing.assert_array_equal(distance(X, "minkowski"), distance(X, "euclidean"))


def test_minkowski_is_therefore_federatable():
    """Because it is euclidean, the preferred metric pools exactly -- it must not be
    listed among the methods exact mode refuses."""
    assert "minkowski" in METHODS
    assert "minkowski" not in NOT_GRAM_DERIVABLE

    X, _ = _lung()
    parts = [X[:37], X[37:65], X[65:]]
    pooled = pool_stats([pairwise_stats(p) for p in parts])
    np.testing.assert_allclose(distance_from_stats(pooled, "minkowski"),
                               distance(X, "minkowski"), rtol=0, atol=1e-12)


def test_minkowski_with_other_exponents_matches_r_dist():
    """p != 2 is reachable only outside pvclust. R's dist() drops incomplete pairs
    and scales the sum up by n/count before taking the p-th root; zero-filling the
    missing values instead gives visibly wrong distances."""
    X, _ = _lung()
    np.testing.assert_allclose(minkowski_distance(X, 3),
                               _expected("dist_minkowski_p3.csv"), rtol=0, atol=1e-12)


def test_minkowski_p2_routes_to_the_exact_path():
    X, _ = _lung()
    np.testing.assert_array_equal(minkowski_distance(X, 2), distance(X, "euclidean"))


def test_ward_d2_tree_matches_r():
    """ward.D2 is scipy's 'ward'. Edge sets, merge order and heights all agree."""
    X, labels = _lung()
    Z = linkage(distance(X, "minkowski"), "ward.D2")
    mine = [edge_pattern(m, labels) for m in edge_members(Z, labels)]
    assert mine == (FIXTURES / "ward_patterns.txt").read_text().split()

    heights = pd.read_csv(FIXTURES / "ward_height.csv",
                          float_precision="round_trip")["height"].to_numpy()
    np.testing.assert_allclose(Z[:, 2], heights, rtol=0, atol=1e-11)


def test_minkowski_ward_pvalues_agree_with_r(): 
    """The full loop in this configuration, within Monte-Carlo error."""
    from pvclust_py.core import pvclust
    X, labels = _lung()
    res = pvclust(X, labels, method_dist="minkowski", method_hclust="ward.D2",
                  nboot=1000, seed=42)
    exp = pd.read_csv(FIXTURES / "ward_edges_expected.csv", index_col=0,
                      float_precision="round_trip")

    got = np.array([e["au"] for e in res.edges])
    got_se = np.array([e["se_au"] for e in res.edges])
    bound = np.maximum(4.0 * np.hypot(got_se, exp["se.au"].to_numpy()), 0.05)
    bad = np.abs(got - exp["au"].to_numpy()) > bound
    assert not bad.any(), f"AU diverges on edges {np.where(bad)[0].tolist()}"
