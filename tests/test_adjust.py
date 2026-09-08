"""
Batch adjustment, tested on synthetic data where the truth is known.

Real data cannot tell you whether a correction removed the right thing, because you
do not know the answer. These fixtures plant a batch effect and a group effect of
known size, then check that the batch one goes and the group one stays.

The test that matters most is
:func:`test_protect_preserves_the_group_effect_under_strong_confounding` -- it is the
whole reason ``protect=`` exists.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pvclust_py.adjust import _dummies, adjust_batch, confounding


def make_data(n=200, p=40, group_effect=2.0, batch_effect=5.0, confound=0.5, seed=0):
    """Samples with a known group effect and a known batch effect.

    ``confound`` is the probability that a group-1 sample lands in batch B, against
    0.5 for group 0. At 0.5 the design is balanced; at 0.95 batch and group are
    nearly the same variable.
    """
    rng = np.random.default_rng(seed)
    group = pd.Series(rng.integers(0, 2, n), index=[f"s{i}" for i in range(n)], name="group")
    p_b = np.where(group == 1, confound, 0.5)
    batch = pd.Series(np.where(rng.random(n) < p_b, "B", "A"), index=group.index, name="batch")

    X = pd.DataFrame(rng.normal(0, 1, (n, p)), index=group.index,
                     columns=[f"f{j}" for j in range(p)])
    X.loc[group == 1] += group_effect          # biology, on every feature
    X.loc[batch == "B"] += batch_effect        # technical, on every feature
    return X, group, batch


def group_gap(X, group):
    """Mean difference between the groups, averaged over features -- the signal."""
    return float((X[group == 1].mean() - X[group == 0].mean()).mean())


def batch_gap(X, batch):
    return float((X[batch == "B"].mean() - X[batch == "A"].mean()).mean())


# ------------------------------------------------------------------- _dummies
def test_dummies_drops_a_reference_level():
    f = pd.Series(list("aabbcc"), name="f")
    assert _dummies(f, drop_first=True).shape[1] == 2
    assert _dummies(f, drop_first=False).shape[1] == 3


def test_dummies_keeps_a_single_level_column():
    """Dropping the only level would leave nothing to fit."""
    assert _dummies(pd.Series(["a", "a"], name="f"), drop_first=True).shape[1] == 1


# --------------------------------------------------------------- adjust_batch
def test_batch_effect_is_removed():
    X, group, batch = make_data(batch_effect=5.0, confound=0.5)
    assert batch_gap(X, batch) == pytest.approx(5.0, abs=0.5)
    assert batch_gap(adjust_batch(X, batch), batch) == pytest.approx(0.0, abs=1e-8)


def test_a_balanced_design_keeps_the_group_effect_without_protection():
    """When batch is balanced across groups there is nothing to protect against."""
    X, group, batch = make_data(group_effect=2.0, confound=0.5)
    out = adjust_batch(X, batch)
    assert group_gap(out, group) == pytest.approx(2.0, abs=0.3)


def test_protect_preserves_the_group_effect_under_confounding():
    """THE reason protect= exists.

    Note what the truth is here. The group difference OBSERVED in the unadjusted data
    is not the group effect -- under confounding it is inflated by the batch effect,
    because group-1 samples sit mostly in the shifted batch (4.3 observed for a
    planted 2.0 at confound=0.95). The target is the value that was PLANTED.

    Asserted as a trend rather than fixed thresholds: naive adjustment degrades
    monotonically as confounding grows, while protection recovers the planted effect
    at every level.
    """
    planted = 2.0
    naive, protected = [], []
    for confound in (0.5, 0.8, 0.9, 0.99):
        X, group, batch = make_data(group_effect=planted, batch_effect=5.0,
                                    confound=confound)
        naive.append(group_gap(adjust_batch(X, batch), group))
        protected.append(group_gap(adjust_batch(X, batch, protect=group.to_frame()), group))

    for value in protected:
        assert value == pytest.approx(planted, abs=0.25), \
            f"protection should recover {planted}, got {protected}"

    assert naive[-1] < naive[0] - 0.4, \
        f"naive adjustment should degrade as confounding grows, got {naive}"
    assert protected[-1] > naive[-1] + 0.4, \
        "at high confounding, protection should clearly beat naive"


def test_the_observed_group_gap_is_inflated_by_a_confounded_batch():
    """Guard on the reasoning above: if this stopped being true, the test that
    protection helps would be comparing against the wrong number."""
    X, group, batch = make_data(group_effect=2.0, batch_effect=5.0, confound=0.95)
    assert group_gap(X, group) > 3.5, "confounded data should overstate the group effect"


def test_protection_still_removes_the_batch_effect():
    """Preserving biology must not mean leaving the batch effect in place."""
    X, group, batch = make_data(confound=0.95)
    out = adjust_batch(X, batch, protect=group.to_frame())
    assert abs(batch_gap(out, batch)) < abs(batch_gap(X, batch)) * 0.5


def test_complete_confounding_is_refused_with_an_explanation():
    """When every batch holds one group, no method can separate them -- saying so is
    more useful than returning a confident, meaningless correction."""
    X, group, _ = make_data(n=60)
    batch = pd.Series(np.where(group == 1, "B", "A"), index=X.index, name="batch")
    with pytest.raises(ValueError, match="completely confounded"):
        adjust_batch(X, batch, protect=group.to_frame())


def test_tiny_batches_are_refused():
    X, group, batch = make_data(n=60)
    batch.iloc[:2] = "C"
    with pytest.raises(ValueError, match="too small"):
        adjust_batch(X, batch)


def test_missing_batch_labels_are_refused():
    X, group, batch = make_data(n=60)
    batch.iloc[0] = np.nan
    with pytest.raises(ValueError, match="no batch label"):
        adjust_batch(X, batch)


def test_shape_and_labels_survive():
    X, group, batch = make_data()
    out = adjust_batch(X, batch, protect=group.to_frame())
    assert out.shape == X.shape
    assert list(out.index) == list(X.index) and list(out.columns) == list(X.columns)


def test_overall_level_is_preserved():
    """The correction removes batch DIFFERENCES, not the overall signal level --
    otherwise downstream scales shift for no reason."""
    X, group, batch = make_data()
    out = adjust_batch(X, batch)
    np.testing.assert_allclose(out.mean().mean(), X.mean().mean(), atol=1e-8)


def test_scale_option_equalises_per_batch_variance():
    X, group, batch = make_data(n=300)
    X.loc[batch == "B"] *= 3.0                       # inflate one batch's spread
    before = X[batch == "B"].std().mean() / X[batch == "A"].std().mean()
    out = adjust_batch(X, batch, scale=True)
    after = out[batch == "B"].std().mean() / out[batch == "A"].std().mean()
    assert before > 2.0 and after == pytest.approx(1.0, abs=0.25)


# ---------------------------------------------------------------- confounding
def test_confounding_calls_a_balanced_design_balanced():
    X, group, batch = make_data(n=400, confound=0.5)
    out = confounding(batch, group.to_frame()).set_index("variable")
    assert out.loc["group", "verdict"] == "balanced"


def test_confounding_calls_a_partial_design_partial():
    X, group, batch = make_data(n=400, confound=0.9)
    out = confounding(batch, group.to_frame()).set_index("variable")
    assert out.loc["group", "verdict"] == "partial"
    assert "protect" in out.loc["group", "action"]


def test_confounding_calls_a_complete_design_complete():
    X, group, _ = make_data(n=200)
    batch = pd.Series(np.where(group == 1, "B", "A"), index=X.index, name="batch")
    out = confounding(batch, group.to_frame()).set_index("variable")
    assert out.loc["group", "verdict"] == "complete"
    assert "cannot adjust" in out.loc["group", "action"]


def test_confounding_skips_constant_columns():
    X, group, batch = make_data(n=60)
    ann = pd.DataFrame({"constant": ["x"] * len(X)}, index=X.index)
    assert confounding(batch, ann).set_index("variable").loc["constant", "verdict"] == "skipped"


# --------------------------------------------------------------------- combat
def test_combat_matches_the_linear_correction_on_large_batches():
    """ComBat adds empirical-Bayes shrinkage on top of the linear correction. With
    large batches there is little to shrink, so the two should agree closely -- which
    is what makes them a genuine sensitivity analysis rather than one method twice."""
    inmoose = pytest.importorskip("inmoose", reason="ComBat is an optional extra")
    from pvclust_py.adjust import combat

    X, group, batch = make_data(n=300, p=50, confound=0.5)
    linear = adjust_batch(X, batch)
    cb = combat(X, batch)
    r = np.corrcoef(linear.to_numpy().ravel(), cb.to_numpy().ravel())[0, 1]
    assert r > 0.99


def test_combat_removes_the_batch_effect():
    pytest.importorskip("inmoose", reason="ComBat is an optional extra")
    from pvclust_py.adjust import combat

    X, group, batch = make_data(batch_effect=5.0, confound=0.5)
    assert abs(batch_gap(combat(X, batch), batch)) < 0.5
