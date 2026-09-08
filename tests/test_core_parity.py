"""
Parity of the full pvclust loop and pvpick against R.

Unlike msfit/scales/distance, this cannot be an exact comparison: the bootstrap
draws from a different RNG than R's, so the counts differ by Monte-Carlo noise. The
tests below separate what MUST match exactly (the tree, the edge sets, pvpick's
structural rules) from what can only match statistically (AU/BP), and bound the
latter using the standard errors msfit itself reports rather than an arbitrary
tolerance.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pvclust_py.core import pvclust, pvpick
from pvclust_py.hclust import edge_pattern

FIXTURES = Path(__file__).parent / "fixtures"
NBOOT = 1000          # matches scripts/make_fixtures.R section 6
SIGMA = 4.0           # how many combined standard errors we allow
FLOOR = 0.05          # absolute floor, for edges pinned at 0 or 1


@pytest.fixture(scope="module")
def run():
    X = pd.read_csv(FIXTURES / "lung_subset.csv", index_col=0,
                    float_precision="round_trip")
    return pvclust(X.to_numpy(float), list(X.columns),
                   method_dist="correlation", method_hclust="average",
                   nboot=NBOOT, seed=42), X


@pytest.fixture(scope="module")
def expected():
    return pd.read_csv(FIXTURES / "full_edges_expected.csv", index_col=0,
                       float_precision="round_trip")


def test_edge_set_matches_r_exactly(run):
    """The tree is deterministic even though the bootstrap is not, so the clusters
    themselves must match R exactly -- only their p-values are noisy."""
    res, _ = run
    mine = [edge_pattern(e["members"], res.labels) for e in res.edges]
    assert mine == (FIXTURES / "full_patterns.txt").read_text().split()


def test_counts_are_well_formed(run):
    res, _ = run
    assert res.count.shape == (len(res.edges), len(res.r))
    assert (res.count >= 0).all() and (res.count <= NBOOT).all()
    assert res.count[-1].min() == NBOOT, "the root contains every leaf at every scale"


@pytest.mark.parametrize("field", ["au", "bp"])
def test_pvalues_agree_with_r_within_monte_carlo_error(run, expected, field):
    """AU and BP must agree with R to within the noise of the bootstrap.

    The bound is 4 combined standard errors -- msfit reports se_au/se_bp, so the
    test uses the estimator's own uncertainty rather than a number picked to make it
    pass. FLOOR covers edges pinned at 0 or 1, where se is exactly 0 and a single
    count differing between runs flips an edge between degenerate and fitted.
    """
    res, _ = run
    got = np.array([e[field] for e in res.edges])
    got_se = np.array([e[f"se_{field}"] for e in res.edges])
    theirs = expected[field].to_numpy()
    theirs_se = expected[f"se.{field}"].to_numpy()

    bound = np.maximum(SIGMA * np.hypot(got_se, theirs_se), FLOOR)
    bad = np.abs(got - theirs) > bound
    assert not bad.any(), (
        f"{field} diverges beyond Monte-Carlo error on edges {np.where(bad)[0].tolist()}: "
        f"python={got[bad]} R={theirs[bad]} bound={bound[bad]}")


def test_most_edges_are_much_closer_than_the_bound(run, expected):
    """Guard against a bound so loose it would pass anything: the median AU
    discrepancy should be far below the tolerance we allow."""
    res, _ = run
    got = np.array([e["au"] for e in res.edges])
    theirs = expected["au"].to_numpy()
    assert np.median(np.abs(got - theirs)) < 0.02


def test_pvpick_matches_r(run):
    """Same clusters R picks at au >= 0.95, as sets of member labels."""
    res, _ = run
    fx = pd.read_csv(FIXTURES / "full_pvpick_au95.csv")
    theirs = {frozenset(m.split(";")) for m in fx["members"]}
    mine = {frozenset(e["members"]) for e in pvpick(res, 0.95, use="au")}
    assert mine == theirs, f"picked {len(mine)} clusters, R picked {len(theirs)}"


def test_pvpick_excludes_the_root(run):
    """R loops (len-1):1, skipping the root. Returning it would make 'the
    significant clusters' mean 'everything' -- an earlier draft did exactly that."""
    res, _ = run
    for alpha in (0.0, 0.5, 0.95):
        picked = pvpick(res, alpha)
        assert all(e["n_members"] < len(res.labels) for e in picked), \
            f"root leaked into pvpick at alpha={alpha}"


def test_pvpick_returns_non_overlapping_clusters(run):
    """max_only is greedy top-down: kept clusters never share a leaf."""
    res, _ = run
    picked = pvpick(res, 0.90)
    seen: set = set()
    for e in picked:
        members = set(e["members"])
        assert not (members & seen), f"cluster {e['edge_id']} overlaps an earlier pick"
        seen |= members


def test_au_exceeds_bp_for_the_supported_clusters(run):
    """The reason the method exists: BP is biased downward, so real clusters score
    higher under AU. On this dataset the picked clusters sit near AU 0.95 while BP
    calls them ~0.58 -- cutting on BP would discard them."""
    res, _ = run
    picked = pvpick(res, 0.95)
    assert picked, "expected at least one supported cluster on the lung data"
    for e in picked:
        assert e["au"] > e["bp"], f"{e['edge_id']}: au {e['au']} !> bp {e['bp']}"


def test_short_resampling_axis_warns(run):
    """Clustering patients by resampling a handful of genes must not pass silently."""
    _, X = run
    with pytest.warns(UserWarning, match="resampling units"):
        pvclust(X.to_numpy(float)[:12], list(X.columns), nboot=10, seed=1)
