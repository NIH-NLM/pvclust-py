"""apply_edges: the project's own outcome from the federated catalogue.

The module space is the part worth pinning down. It is built from a TREE, so the
clusters that feed it are nested by construction -- and an earlier version averaged
every cluster that passed alpha, which produced 54 mutually-redundant columns out of
55 proteins. These tests hold the invariants that stop that returning.
"""
import numpy as np
import pandas as pd
import pytest

from pvclust_py.apply import apply_edges


@pytest.fixture
def data():
    """Three tight blocks of four objects, so the tree has an obvious nesting."""
    rng = np.random.default_rng(0)
    n = 60
    blocks = {}
    for b in range(3):
        driver = rng.normal(size=n)
        for j in range(4):
            blocks[f"b{b}_{j}"] = driver + rng.normal(scale=0.25, size=n)
    return pd.DataFrame(blocks, index=[f"s{i}" for i in range(n)])


@pytest.fixture
def catalogue(data):
    """A nested catalogue: every block, every pair inside it, and the root."""
    cols = list(data.columns)
    rows = []
    for b in range(3):
        members = [c for c in cols if c.startswith(f"b{b}_")]
        rows.append({"members": ";".join(sorted(members)), "au": 0.99})
        rows.append({"members": ";".join(sorted(members[:2])), "au": 0.98})
    rows.append({"members": ";".join(sorted(cols)), "au": 0.999})
    from pvclust_py.hclust import edge_id
    for r in rows:
        r["edge_id"] = edge_id(r["members"].split(";"))
    return pd.DataFrame(rows)


def _modules(out):
    return out["modules"][out["modules"]["n_members"] > 1]


def test_modules_do_not_overlap(data, catalogue):
    out = apply_edges(data, catalogue, nboot=100, alpha=0.95)
    seen: set = set()
    for m in out["modules"]["members"]:
        members = set(m.split(";"))
        assert not (members & seen), f"{m} overlaps an earlier module"
        seen |= members


def test_every_object_appears_exactly_once(data, catalogue):
    out = apply_edges(data, catalogue, nboot=100, alpha=0.95)
    flat = [x for m in out["modules"]["members"] for x in m.split(";")]
    assert sorted(flat) == sorted(data.columns)


def test_module_space_is_smaller_than_the_original(data, catalogue):
    out = apply_edges(data, catalogue, nboot=100, alpha=0.95)
    assert out["scores"].shape[1] < data.shape[1]
    assert out["scores"].shape[0] == data.shape[0]


def test_prefers_the_tight_cluster_over_the_root(data, catalogue):
    """The root passes alpha too, and taking it would collapse everything to one
    column. Modules must come from the small end of the tree."""
    out = apply_edges(data, catalogue, nboot=100, alpha=0.95)
    mods = _modules(out)
    assert len(mods) >= 2
    assert mods["n_members"].max() < data.shape[1]


def test_scores_are_the_mean_of_their_members(data, catalogue):
    out = apply_edges(data, catalogue, nboot=100, alpha=0.95)
    for _, m in out["modules"].iterrows():
        expect = data[m["members"].split(";")].mean(axis=1)
        assert np.allclose(out["scores"][m["module"]], expect)


def test_unsupported_clusters_are_not_made_into_modules(data, catalogue):
    weak = catalogue.copy()
    weak["au"] = 0.10
    out = apply_edges(data, weak, nboot=100, alpha=0.95)
    assert _modules(out).empty
    assert out["scores"].shape[1] == data.shape[1]


def test_refuses_a_catalogue_this_project_cannot_match(data):
    other = pd.DataFrame([{"edge_id": "x", "members": "zz1;zz2", "au": 0.99}])
    with pytest.raises(ValueError, match="no federated cluster"):
        apply_edges(data, other, nboot=50)
