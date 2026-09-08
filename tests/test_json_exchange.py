"""
The JSON federation payload: what one project sends the aggregator.

Two properties matter more than the field list. It must carry **no subject-level
data**, and it must be self-describing enough that an aggregator refuses to pool
things that are not comparable -- silently averaging a correlation/average run with a
minkowski/ward.D2 run would produce a number, and that number would be meaningless.
"""
from __future__ import annotations

import json

import pytest

from pvclust_py.core import pvclust
from pvclust_py.datasets import load_lung
from pvclust_py.io import (SCHEMA_PROJECT, counts_from_json, read_project_json,
                           write_project_json)


@pytest.fixture(scope="module")
def payload(tmp_path_factory):
    X = load_lung().iloc[:40, :10]
    res = pvclust(X, nboot=20, seed=1)
    path = tmp_path_factory.mktemp("j") / "P1.json"
    write_project_json(res, path, "P1", n=X.shape[0])
    return path, res, X


def test_round_trips(payload):
    path, res, _ = payload
    got = read_project_json(path)
    assert got["project"] == "P1"
    assert got["schema"] == SCHEMA_PROJECT
    assert len(got["edges"]) == len(res.edges)
    assert got["labels"] == list(res.labels)


def test_carries_no_subject_level_data(payload):
    """The whole privacy claim. The payload holds cluster memberships over FEATURE
    names, integer tallies and fitted p-values -- nothing about individual rows."""
    path, res, X = payload
    text = json.loads(path.read_text())
    blob = json.dumps(text)

    for sample_id in X.index:
        assert str(sample_id) not in blob, f"sample id {sample_id} leaked into the payload"
    assert "labels" in text and set(text["labels"]) <= {str(c) for c in X.columns}
    for value in X.to_numpy().ravel()[:200]:
        assert f"{value:.6f}" not in blob


def test_records_n_so_scales_can_be_reconciled(payload):
    """Each project's r is relative to its own row count, so the aggregator cannot
    put scales on a common footing without n."""
    path, res, X = payload
    assert read_project_json(path)["data"]["n_resampling_units"] == X.shape[0]


def test_records_how_the_clustering_was_produced(payload):
    path, res, _ = payload
    c = read_project_json(path)["clustering"]
    assert c["method_dist"] == res.method_dist
    assert c["method_hclust"] == res.method_hclust
    assert c["cluster"] == res.cluster


def test_counts_survive_the_round_trip(payload):
    path, res, _ = payload
    df = counts_from_json([path])
    assert len(df) == len(res.edges) * len(res.r)
    assert set(df.columns) >= {"project", "edge_id", "r", "n", "nboot", "count"}
    assert df["count"].sum() == int(res.count.sum())


def test_a_foreign_schema_is_refused(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"schema": "something-else/1", "edges": []}))
    with pytest.raises(ValueError, match="not a pvclust-py project payload"):
        read_project_json(p)


def test_a_future_schema_version_is_refused_rather_than_guessed(tmp_path):
    p = tmp_path / "future.json"
    p.write_text(json.dumps({"schema": "pvclust-py/project/99", "edges": []}))
    with pytest.raises(ValueError, match="Refusing rather than guessing"):
        read_project_json(p)


def test_pooling_incompatible_clusterings_is_refused(tmp_path):
    """Counts from different distance or linkage settings are not comparable. Pooling
    them would give a number that means nothing, so it must fail loudly."""
    X = load_lung().iloc[:40, :10]
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    write_project_json(pvclust(X, method_dist="correlation", nboot=10, seed=1), a, "A")
    write_project_json(pvclust(X, method_dist="euclidean", nboot=10, seed=1), b, "B")

    with pytest.raises(ValueError, match="not comparable"):
        counts_from_json([a, b])


def test_pooling_matching_clusterings_is_allowed(tmp_path):
    X = load_lung().iloc[:40, :10]
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    write_project_json(pvclust(X.iloc[:20], nboot=10, seed=1), a, "A", n=20)
    write_project_json(pvclust(X.iloc[20:], nboot=10, seed=2), b, "B", n=20)
    df = counts_from_json([a, b])
    assert set(df["project"]) == {"A", "B"}
