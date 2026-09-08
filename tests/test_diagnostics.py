"""
Association diagnostics: is the clustering biology, or the plate it was run on?

The bias correction is the part worth testing hardest. An uncorrected Cramer's V
scored a 96-level plate position at 0.56 ("strong") on data whose chi-square p-value
was 0.46 -- i.e. pure noise -- which would have raised a false batch alarm. A
diagnostic that cries wolf is worse than none.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pvclust_py.core import kmeans_pv, pvclust
from pvclust_py.datasets import load_lung
from pvclust_py.diagnostics import _cramers_v, association, cluster_labels, report


def _tiny():
    return load_lung().iloc[:40, :12]


# ------------------------------------------------------------ cluster_labels
def test_cluster_labels_cuts_a_dendrogram_into_k_groups():
    res = pvclust(_tiny(), nboot=10, seed=1)
    labels = cluster_labels(res, k=3)
    assert labels.nunique() == 3
    assert set(labels.index) == set(res.labels)


def test_cluster_labels_uses_kmeans_groups_directly():
    """k-means has no tree to cut -- its clusters are already the assignment."""
    res = kmeans_pv(_tiny(), k=4, nboot=10, seed=1)
    labels = cluster_labels(res)
    assert labels.nunique() == 4
    assert len(labels) == _tiny().shape[1]


def test_cluster_labels_defaults_to_the_clusters_you_would_report():
    """Without k, cut at however many clusters pvpick keeps -- so the test is run on
    the clusters you would actually publish, not an arbitrary number."""
    res = pvclust(_tiny(), nboot=50, seed=1)
    assert cluster_labels(res, alpha=0.95).nunique() >= 2


# --------------------------------------------------------- Cramer's V correction
def test_cramers_v_correction_removes_many_level_inflation():
    """A variable with many levels relative to n is inflated by the uncorrected
    formula. With chi-square at its expected value under independence, the corrected
    V must be ~0 while the uncorrected one is substantial."""
    n, r, c = 300, 4, 96
    chi2_expected = (r - 1) * (c - 1)            # E[chi2] under independence
    uncorrected = np.sqrt(chi2_expected / (n * (min(r, c) - 1)))
    corrected = _cramers_v(chi2_expected, n, r, c)

    assert uncorrected > 0.3, "the inflation this corrects should be visible"
    assert corrected == 0.0, "no association means no effect"


def test_cramers_v_still_detects_a_real_association():
    n, r, c = 300, 3, 3
    assert _cramers_v(chi2=250.0, n=n, r=r, c=c) > 0.5


def test_cramers_v_is_never_negative():
    assert _cramers_v(0.0, 100, 5, 5) == 0.0


# ------------------------------------------------------------------ association
def _fixture(n=200, seed=0):
    rng = np.random.default_rng(seed)
    clusters = pd.Series(rng.integers(0, 3, n), index=[f"s{i}" for i in range(n)])
    ann = pd.DataFrame({
        "perfect": clusters.map({0: "a", 1: "b", 2: "c"}),       # IS the clustering
        "unrelated": rng.choice(list("xyz"), n),                 # independent
        "numeric_related": clusters * 10 + rng.normal(0, 1, n),  # tracks the clustering
        "numeric_noise": rng.normal(0, 1, n),
        "constant": ["same"] * n,
    }, index=clusters.index)
    return clusters, ann


def test_a_variable_identical_to_the_clustering_scores_at_the_top():
    clusters, ann = _fixture()
    out = association(clusters, ann).set_index("variable")
    assert out.loc["perfect", "effect"] > 0.9
    assert out.loc["perfect", "adjusted_rand"] > 0.95
    assert out.loc["unrelated", "strength"] == "negligible"


def test_numeric_variables_use_kruskal_wallis():
    clusters, ann = _fixture()
    out = association(clusters, ann).set_index("variable")
    assert out.loc["numeric_related", "test"] == "Kruskal-Wallis"
    assert out.loc["numeric_related", "p_value"] < 1e-10
    assert out.loc["numeric_noise", "strength"] == "negligible"


def test_results_are_sorted_by_effect_not_by_p_value():
    """Effect size is what matters -- p shrinks with n, so on a few hundred samples
    almost anything is 'significant'."""
    clusters, ann = _fixture()
    out = association(clusters, ann)
    eff = out["effect"].dropna().to_numpy()
    assert (np.diff(eff) <= 1e-12).all()


def test_constant_columns_are_skipped_not_crashed_on():
    clusters, ann = _fixture()
    out = association(clusters, ann).set_index("variable")
    assert out.loc["constant", "test"] == "skipped"


def test_mismatched_indexes_are_refused():
    clusters, ann = _fixture()
    with pytest.raises(ValueError, match="in common"):
        association(clusters, ann.set_index(pd.Index([f"other{i}" for i in range(len(ann))])))


# ---------------------------------------------------------------------- report
def test_report_flags_a_batch_variable_that_is_both_strong_and_significant():
    assoc = pd.DataFrame([{"variable": "PlateId", "effect": 0.7, "p_value": 1e-9,
                           "effect_name": "Cramers V", "strength": "strong"}])
    assert "WARNING" in report(assoc) and "PlateId" in report(assoc)


def test_report_does_not_cry_wolf_on_a_large_but_insignificant_effect():
    """The exact false alarm that prompted the bias correction: V = 0.56 with
    p = 0.46 is small-sample noise, not a batch effect."""
    assoc = pd.DataFrame([{"variable": "PlatePosition", "effect": 0.56, "p_value": 0.46,
                           "effect_name": "Cramers V", "strength": "strong"}])
    assert "WARNING" not in report(assoc)


def test_report_ignores_a_significant_but_trivial_effect():
    assoc = pd.DataFrame([{"variable": "PlateId", "effect": 0.05, "p_value": 1e-12,
                           "effect_name": "Cramers V", "strength": "negligible"}])
    assert "WARNING" not in report(assoc)


def test_report_separates_biological_from_technical_variables():
    assoc = pd.DataFrame([
        {"variable": "PlateId", "effect": 0.10, "p_value": 0.4, "effect_name": "Cramers V", "strength": "negligible"},
        {"variable": "Group", "effect": 0.55, "p_value": 1e-15, "effect_name": "Cramers V", "strength": "strong"},
    ])
    out = report(assoc)
    assert "No batch-like variable" in out and "Group" in out
