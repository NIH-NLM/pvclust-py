"""
Distance parity with R's dist.pvclust, and the exactness of federated pooling.

The second half of this file is the justification for the whole exact-federation
design: pooling per-project sufficient statistics must give bit-for-bit the same
distance matrix as pooling the raw data. If that ever stops holding, exact mode is
not exact and the claim has to come out of the paper.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pvclust_py.distance import (METHODS, NOT_GRAM_DERIVABLE, distance,
                                 distance_from_stats, listwise_stats,
                                 pairwise_stats, pool_stats)

FIXTURES = Path(__file__).parent / "fixtures"
TOL = 1e-12


def _lung():
    return pd.read_csv(FIXTURES / "lung_subset.csv", index_col=0,
                       float_precision="round_trip").to_numpy(float)


@pytest.mark.parametrize("method", METHODS)
def test_distance_matches_r(method):
    """Every supported distance, against R's dist.pvclust on the same matrix.

    The fixture matrix deliberately contains missing values, because the three
    methods handle them by three different rules and a complete matrix would hide
    all three.
    """
    expected = pd.read_csv(FIXTURES / f"dist_{method}.csv", index_col=0,
                           float_precision="round_trip").to_numpy(float)
    got = distance(_lung(), method)
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12,
                               err_msg=f"{method} diverges from R")


def test_fixture_actually_has_missing_values():
    """Guard the guard: without NAs the three NA rules are indistinguishable and
    test_distance_matches_r stops testing them."""
    assert np.isnan(_lung()).sum() > 0


def test_uncentered_is_listwise_not_pairwise():
    """R applies na.omit() before the crossproduct for `uncentered`, unlike the
    pairwise rule used for `correlation`. Reproducing this is not optional --
    computing it pairwise gives visibly different numbers."""
    X = _lung()
    listwise = distance_from_stats(listwise_stats(X), "uncentered")
    pairwise = distance_from_stats(pairwise_stats(X), "uncentered")
    assert not np.allclose(listwise, pairwise, atol=1e-8), \
        "listwise and pairwise must differ here, else this test proves nothing"

    expected = pd.read_csv(FIXTURES / "dist_uncentered.csv", index_col=0,
                           float_precision="round_trip").to_numpy(float)
    np.testing.assert_allclose(listwise, expected, rtol=0, atol=1e-12)


def test_euclidean_scales_up_for_dropped_pairs():
    """R's dist() scales the squared sum by n/count when pairs are dropped.
    Omitting the scaling silently shrinks distances wherever data is missing."""
    X = _lung()
    st = pairwise_stats(X)
    scaled = distance_from_stats(st, "euclidean")

    unscaled = np.sqrt(np.clip(st["Q"] + st["Q"].T - 2 * st["G"], 0, None))
    np.fill_diagonal(unscaled, 0.0)
    assert not np.allclose(scaled, unscaled, atol=1e-8), \
        "scaling must matter on this fixture, else the test is vacuous"
    assert (scaled >= unscaled - 1e-12).all(), "scaling can only increase distances"


@pytest.mark.parametrize("method", METHODS)
def test_pooled_statistics_reproduce_the_pooled_raw_distance(method):
    """THE exact-federation claim.

    Three projects hold disjoint rows and share the column vocabulary. Each ships
    only its sufficient statistics; the aggregator adds them. The resulting distance
    matrix must equal the one computed from the pooled raw data -- not approximately,
    to machine precision.
    """
    X = _lung()
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(X))
    parts = [X[perm[0:37]], X[perm[37:65]], X[perm[65:]]]
    assert sum(len(p) for p in parts) == len(X) and len(parts) == 3

    stats_fn = listwise_stats if method == "uncentered" else pairwise_stats
    pooled = pool_stats([stats_fn(p) for p in parts])

    federated = distance_from_stats(pooled, method)
    centralised = distance(X, method)

    np.testing.assert_allclose(
        federated, centralised, rtol=0, atol=1e-12,
        err_msg=f"{method}: federated pooling is not exact")


def test_pooling_rejects_a_different_number_of_objects():
    """Projects must agree on the column set before their statistics can be added."""
    X = _lung()
    with pytest.raises(ValueError, match="disagree on the number of objects"):
        pool_stats([pairwise_stats(X), pairwise_stats(X[:, :5])])


def test_pooling_rejects_same_shape_but_different_objects():
    """The dangerous case, and the reason labels travel with the numbers.

    Two projects that each selected their own most-variable features produce matrices
    of the SAME shape describing DIFFERENT proteins. Shape agreement is not enough:
    adding them would pair one project's protein with another's and yield a pooled
    Gram that is meaningless, without any error. Seen for real -- two cohorts each
    took their top 50 by variance and only 46 overlapped.
    """
    X = _lung()
    a = pairwise_stats(X[:, :10]); a["labels"] = [f"p{i}" for i in range(10)]
    b = pairwise_stats(X[:, 5:15]); b["labels"] = [f"p{i}" for i in range(5, 15)]

    with pytest.raises(ValueError, match="different objects"):
        pool_stats([a, b])


def test_pooling_accepts_matching_labels():
    X = _lung()
    labels = [f"p{i}" for i in range(10)]
    a = pairwise_stats(X[:20, :10]); a["labels"] = labels
    b = pairwise_stats(X[20:, :10]); b["labels"] = labels
    pooled = pool_stats([a, b])
    assert pooled["labels"] == labels
    np.testing.assert_allclose(pooled["G"], pairwise_stats(X[:, :10])["G"], atol=1e-10)


def test_unlabelled_statistics_still_pool():
    """Statistics written before labels were carried must keep working."""
    X = _lung()
    pooled = pool_stats([pairwise_stats(X[:20]), pairwise_stats(X[20:])])
    np.testing.assert_allclose(pooled["G"], pairwise_stats(X)["G"], atol=1e-10)


def test_partially_labelled_statistics_are_refused():
    """Mixing labelled and unlabelled means the check cannot be made -- say so."""
    X = _lung()
    a = pairwise_stats(X[:20, :10]); a["labels"] = [f"p{i}" for i in range(10)]
    with pytest.raises(ValueError, match="some statistics carry labels"):
        pool_stats([a, pairwise_stats(X[20:, :10])])


@pytest.mark.parametrize("method", NOT_GRAM_DERIVABLE)
def test_non_derivable_methods_are_refused(method):
    """Exact mode must refuse what it cannot compute, with a message that points
    at the alternative rather than failing obscurely."""
    with pytest.raises(ValueError, match="not recoverable|counts mode"):
        distance_from_stats(pairwise_stats(_lung()), method)
