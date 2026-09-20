"""Federating a clustering of patients, where the exact path does not exist.

The two release rules are the point of these tests. A partition with any cluster below
the floor is suppressed WHOLE, because dropping just the small cluster announces it
existed and the remainder reconstructs its mean.
"""
import numpy as np
import pandas as pd
import pytest

from pvclust_py.profiles import (SuppressedError, _centred, match_profiles,
                                 project_profiles, replication)


@pytest.fixture
def two_sites():
    """Two sites, disjoint patients, the SAME three underlying endotypes."""
    rng = np.random.default_rng(0)
    p = 40
    signatures = rng.normal(size=(3, p)) * 2.0

    def site(name, sizes):
        rows, labels, ids = [], [], []
        for c, n in enumerate(sizes):
            rows.append(signatures[c] + rng.normal(scale=0.4, size=(n, p)))
            labels += [c] * n
            ids += [f"{name}_p{len(ids) + i}" for i in range(n)]
        X = pd.DataFrame(np.vstack(rows), index=ids,
                         columns=[f"prot{j}" for j in range(p)])
        return X, pd.Series(labels, index=ids)

    return site("A", [30, 25, 20]), site("B", [15, 18, 22])


def test_profiles_carry_no_patient(two_sites):
    (X, a), _ = two_sites
    out = project_profiles(X, a, project="A", min_cluster=10)
    assert len(out["profiles"]) == 3
    assert list(out["profiles"].columns) == list(X.columns)
    # no row of the release may be a patient id
    assert not set(out["profiles"].index) & set(X.index)
    assert not set(out["counts"]["cluster"]) & set(X.index)


def test_profile_is_the_mean_of_its_members(two_sites):
    (X, a), _ = two_sites
    out = project_profiles(X, a, project="A", min_cluster=10)
    for c in sorted(a.unique()):
        expect = X[a == c].mean()
        assert np.allclose(out["profiles"].loc[f"A:c{c}"], expect)


def test_counts_sum_to_the_cohort(two_sites):
    (X, a), _ = two_sites
    out = project_profiles(X, a, project="A", min_cluster=10)
    assert out["counts"]["n_patients"].sum() == len(X)


def test_small_cluster_suppresses_the_WHOLE_partition(two_sites):
    """Not just the offending cluster -- that would announce it existed."""
    (X, a), _ = two_sites
    a = a.copy()
    a.iloc[:len(a) - 3] = 0          # leaves a cluster of 3
    with pytest.raises(SuppressedError, match="WHOLE partition"):
        project_profiles(X, a, project="A", min_cluster=10)


def test_floor_is_respected_exactly(two_sites):
    (X, a), _ = two_sites
    smallest = int(a.value_counts().min())
    project_profiles(X, a, project="A", min_cluster=smallest)        # ok
    with pytest.raises(SuppressedError):
        project_profiles(X, a, project="A", min_cluster=smallest + 1)


def test_matching_pairs_the_shared_endotypes(two_sites):
    (XA, a), (XB, b) = two_sites
    pa = project_profiles(XA, a, project="A", min_cluster=10)["profiles"]
    pb = project_profiles(XB, b, project="B", min_cluster=10)["profiles"]
    res, _sim = match_profiles([pa, pb], nboot=30)
    # cut into three and each group should hold one profile from each site
    from scipy.cluster.hierarchy import fcluster
    lab = fcluster(res.linkage, t=3, criterion="maxclust")
    for g in set(lab):
        members = [res.labels[i] for i, x in enumerate(lab) if x == g]
        assert len({m.split(":")[0] for m in members}) == 2, members
        assert len({m.split("c")[-1] for m in members}) == 1, members


def test_centring_is_what_makes_it_discriminate(two_sites):
    """Raw profiles are dominated by which features are abundant, so everything
    correlates with everything. This is the bug the centring exists to prevent."""
    (XA, a), (XB, b) = two_sites
    # give the features wildly different baselines, as real abundances have
    base = np.linspace(1, 500, XA.shape[1])
    XA, XB = XA + base, XB + base
    pa = project_profiles(XA, a, project="A", min_cluster=10)["profiles"]
    pb = project_profiles(XB, b, project="B", min_cluster=10)["profiles"]

    raw = np.corrcoef(pa.to_numpy(float), pb.to_numpy(float))[:3, 3:]
    P = _centred([pa, pb])
    cen = np.corrcoef(P.to_numpy(float))[:3, 3:]
    assert raw.max() - raw.min() < 0.05, "raw profiles should NOT discriminate"
    assert cen.max() - cen.min() > 0.5, "centred profiles should discriminate"


def test_centring_removes_a_project_wide_offset(two_sites):
    """Cohort and batch are the same thing here, so a project's own mean is its batch
    offset. Adding one must not change the answer."""
    (XA, a), (XB, b) = two_sites
    pa = project_profiles(XA, a, project="A", min_cluster=10)["profiles"]
    pb = project_profiles(XB, b, project="B", min_cluster=10)["profiles"]
    before = _centred([pa, pb])
    after = _centred([pa, pb + 17.5])
    assert np.allclose(before.to_numpy(float), after.to_numpy(float))


def test_matching_needs_a_shared_feature_space(two_sites):
    (XA, a), (XB, b) = two_sites
    pa = project_profiles(XA, a, project="A", min_cluster=10)["profiles"]
    pb = project_profiles(XB, b, project="B", min_cluster=10)["profiles"]
    pb.columns = [f"other{i}" for i in range(pb.shape[1])]
    with pytest.raises(ValueError, match="share no features"):
        match_profiles([pa, pb], nboot=10)


def test_matching_needs_two_projects(two_sites):
    (XA, a), _ = two_sites
    pa = project_profiles(XA, a, project="A", min_cluster=10)["profiles"]
    with pytest.raises(ValueError, match="at least two"):
        match_profiles([pa], nboot=10)


def test_replication_reports_every_cluster(two_sites):
    (XA, a), (XB, b) = two_sites
    oa = project_profiles(XA, a, project="A", min_cluster=10)
    ob = project_profiles(XB, b, project="B", min_cluster=10)
    res, _ = match_profiles([oa["profiles"], ob["profiles"]], nboot=30)
    counts = pd.concat([oa["counts"], ob["counts"]], ignore_index=True)
    rep = replication(res, counts, alpha=0.95)
    assert len(rep) == 6
    assert set(rep["cluster"]) == set(counts["cluster"])
    assert rep["matched_with"].str.len().gt(0).all()


def test_demographics_suppress_small_cells(two_sites):
    (X, a), _ = two_sites
    ann = pd.DataFrame({"Group": ["SLE"] * (len(X) - 2) + ["HV", "HV"]}, index=X.index)
    out = project_profiles(X, a, project="A", min_cluster=10, annotations=ann)
    assert (out["demographics"]["n"] >= 10).all()
