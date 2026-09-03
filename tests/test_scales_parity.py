"""
Parity of the bootstrap scale machinery against R.

Separate from the msfit tests because these two rules are what feed msfit its
inputs -- if they drift, msfit is fed the wrong r and every downstream p-value is
subtly wrong while every msfit unit test still passes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pvclust_py.scales import default_scales, effective_scales, seq_by

FIXTURES = Path(__file__).parent / "fixtures"


def test_seq_by_matches_r_bit_for_bit():
    """R's seq(from, to, by), to the last bit.

    Not a tolerance test on purpose. The values are floored downstream, so a
    one-ulp difference becomes an off-by-one in the resample size:
    np.arange(0.5, 1.45, 0.1) gives 0.9999999999999999 where R gives 1.0, and
    floor(63 * 0.9999999999999999) is 62 rather than 63.
    """
    # Read the value column as TEXT: pandas' default C float parser is fast but not
    # correctly rounded, and silently shifts these fixtures by one ulp on read.
    fx = pd.read_csv(FIXTURES / "seq_by.csv", dtype={"value": str})
    for (start, stop, by), group in fx.groupby(["start", "stop", "by"], sort=False):
        group = group.sort_values("index")
        expected = np.array([float(v) for v in group["value"]])
        got = seq_by(start, stop, by)

        assert len(got) == len(expected), f"seq({start},{stop},by={by}): wrong length"
        for i, (g, e) in enumerate(zip(got, expected)):
            assert g.hex() == float(e).hex(), (
                f"seq({start},{stop},by={by})[{i}]: {g!r} != R's {e!r} "
                f"({g.hex()} vs {float(e).hex()})")


def test_arange_would_have_been_wrong():
    """Guard the guard: prove np.arange really does diverge, so that
    test_seq_by_matches_r_bit_for_bit is testing something real."""
    naive = np.arange(0.5, 1.45, 0.1)
    good = default_scales()
    assert len(naive) == len(good)
    assert not all(a.hex() == b.hex() for a, b in zip(naive, good))
    assert int(np.floor(63 * naive[5])) == 62, "the off-by-one this test exists for"
    assert int(np.floor(63 * good[5])) == 63


@pytest.mark.parametrize("n", [8, 12, 63, 100, 916])
def test_effective_scales_match_r(n):
    """size = unique(floor(n*r)) and effective r = size/n, per pvclust-internal.R:24."""
    fx = pd.read_csv(FIXTURES / "effective_r.csv", float_precision="round_trip")
    group = fx[fx["n"] == n].sort_values("scale_index")
    size, eff = effective_scales(n)

    np.testing.assert_array_equal(size, group["size"].to_numpy(),
                                  err_msg=f"n={n}: sizes must be unique(floor(n*r))")
    np.testing.assert_allclose(eff, group["effective_r"].to_numpy(), rtol=1e-12, atol=0)


def test_unique_collapses_scales_at_small_n():
    """At n=8 the ten nominal scales collapse to eight. A caller that assumes it
    gets back as many scales as it passed in is wrong."""
    size, eff = effective_scales(8)
    assert len(default_scales()) == 10
    assert len(size) == 8


def test_all_scales_flooring_to_zero_is_an_error():
    """R stops with 'invalid scale parameter(r)' when size == 0."""
    with pytest.raises(ValueError, match="invalid scale parameter"):
        effective_scales(1, r=[0.1, 0.2, 0.3])
