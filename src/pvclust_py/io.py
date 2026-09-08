"""
Reading matrices in, and writing artifacts out. Format-agnostic.

Everything here works on a plain labelled matrix: **rows are the resampling units**
(samples), **columns are the objects clustered** (genes, proteins, whatever you
measured). One CSV or TSV, first column the row index.

That is the entire input contract. A vendor-specific layout is a matter of getting
your data into this shape -- :mod:`pvclust_py.somascan` is one such adapter -- and
nothing downstream knows or cares where a matrix came from.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _unique_labels(values, fallback, source: str):
    """Readable labels that stay unique.

    Human-readable names are rarely unique -- several probes, reagents or transcripts
    routinely map to one gene -- and duplicate labels would be worse than ugly: two
    distinct measurements sharing a name become indistinguishable, so the
    content-hashed edge_id can no longer tell their clusters apart. Collisions get the
    original id appended; missing names fall back to the id entirely.
    """
    import pandas as pd
    values = pd.Series(values).astype("string")
    fallback = pd.Series(fallback).astype(str)

    out = values.fillna(pd.Series(fallback, index=values.index))
    dup = out.duplicated(keep=False)
    out = out.astype(str)
    out[dup] = out[dup] + " (" + fallback[dup].to_numpy() + ")"

    if out.duplicated().any():
        raise ValueError(f"labels from {source!r} are still not unique after "
                         f"disambiguation -- fall back to the original ids")
    return out.to_numpy()


def _read_any(path, index_col=None):
    """Read a CSV or TSV, inferring the separator and stripping a UTF-8 BOM.

    The BOM matters: a file exported from Excel begins with \ufeff, which silently
    becomes part of the first column's name, so a lookup on that column finds nothing.
    """
    path = Path(path)
    sep = "\t" if path.suffix.lower() in (".txt", ".tsv") else ","
    return pd.read_csv(path, sep=sep, index_col=index_col,
                       encoding="utf-8-sig", float_precision="round_trip")


def read_matrix(path, index_col: int = 0, transpose: bool = False,
                log2: bool = False, feature_map=None,
                feature_key: Optional[str] = None,
                feature_label: Optional[str] = None) -> pd.DataFrame:
    """Read a plain CSV/TSV matrix, optionally relabelling and log-transforming.

    Args:
        path: the matrix. Rows are resampling units, columns the objects clustered.
        transpose: flip it, if yours is the other way round.
        log2: take log2. Abundance data is usually strongly right-skewed, and
            correlation on the raw scale is then dominated by a few abundant
            features. Off by default because a matrix may already be logged --
            check the value range before deciding.
        feature_map: lookup table for renaming columns -- probe ids to gene symbols,
            accessions to names, whatever you have. Any CSV/TSV with a key column and
            a label column.
        feature_key: column in ``feature_map`` matching the matrix column names.
            Defaults to its FIRST column.
        feature_label: column to rename them to. Defaults to its SECOND column.
            Collisions get the original id appended, since duplicate labels would make
            two distinct measurements indistinguishable and corrupt the
            content-hashed edge ids.
    """
    df = _read_any(path, index_col=index_col)
    if transpose:
        df = df.T

    if feature_map is not None:
        fm = _read_any(feature_map)
        if fm.shape[1] < 2:
            raise ValueError("a feature map needs at least two columns: an id and a label")
        feature_key = feature_key or fm.columns[0]
        feature_label = feature_label or fm.columns[1]
        for col in (feature_key, feature_label):
            if col not in fm.columns:
                raise ValueError(f"feature map has no {col!r} column; "
                                 f"available: {list(fm.columns)}")
        lookup = dict(zip(fm[feature_key].astype(str), fm[feature_label].astype(str)))
        missing = [c for c in df.columns if str(c) not in lookup]
        if len(missing) == len(df.columns):
            raise ValueError(
                f"no matrix column matched {feature_key!r} in the feature map. "
                f"Matrix columns look like {list(df.columns[:3])}, map keys like "
                f"{list(lookup)[:3]}")
        names = pd.Series([lookup.get(str(c)) for c in df.columns])
        df.columns = _unique_labels(names, [str(c) for c in df.columns], feature_label)

    if log2:
        if (df <= 0).any().any():
            raise ValueError("non-positive values cannot be log2-transformed; "
                             "drop --log2 and handle them explicitly")
        df = np.log2(df)
    return df


def read_metadata(path, index_col: int = 0) -> pd.DataFrame:
    """Sample annotations, indexed by sample id -- for the association diagnostics."""
    return _read_any(path, index_col=index_col)


def write_edges(result, path) -> None:
    """Write the per-cluster table: id, members, si/au/bp, standard errors, pchi."""
    result.edges_frame().to_csv(path, index=False)


def write_counts(result, path, project: str, n: Optional[int] = None) -> None:
    """Write the long-form counts -- exactly what a project ships to the aggregator.

    The ``n`` column is required by :func:`pvclust_py.aggregate.pool_counts`: each
    project's ``r`` is relative to its own row count, so the aggregator needs ``n`` to
    put the scales on a common footing.
    """
    df = result.counts_frame()
    df.insert(0, "project", project)
    df["n"] = n if n is not None else int(round(result.nboot[0] * 0)) or len(result.labels)
    df.to_csv(path, index=False)


def write_stats(stats, path) -> None:
    """Write sufficient statistics as compressed npz -- what exact mode ships."""
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in stats.items()})


def read_stats(path) -> dict:
    """Read sufficient statistics written by :func:`write_stats`."""
    with np.load(path) as z:
        out = {k: z[k] for k in z.files}
    if "n" in out:
        out["n"] = int(out["n"])
    return out


# ---------------------------------------------------------------- JSON exchange
#: Bumped when the payload shape changes incompatibly, so an aggregator can refuse a
#: file it does not understand rather than misreading it.
SCHEMA_PROJECT = "pvclust-py/project/1"
SCHEMA_FEDERATED = "pvclust-py/federated/1"


def write_project_json(result, path, project: str, *, n: Optional[int] = None,
                       extra: Optional[dict] = None) -> None:
    """Write one project's result as JSON -- the federation exchange format.

    Self-describing on purpose: the aggregator has to know how a payload was produced
    before it can decide whether pooling is meaningful. Distance, linkage, orientation
    and the effective scales all travel with the counts, so mismatched methods are
    caught rather than silently averaged.

    Contains **no subject-level data** -- cluster memberships over feature names,
    per-scale integer tallies, and fitted p-values. Nothing that identifies a sample.

    The ``n`` field is not decoration. Each project's ``r`` is relative to its own row
    count, so ``r = 1.0`` means 300 samples at one project and 40 at another; the
    aggregator needs ``n`` to put the scales on a common footing before summing.
    """
    import json

    from . import __version__

    n = n if n is not None else len(result.labels)
    payload = {
        "schema": SCHEMA_PROJECT,
        "pvclust_py_version": __version__,
        "project": project,
        "clustering": {
            "cluster": result.cluster,
            "method_dist": result.method_dist,
            "method_hclust": result.method_hclust,
            "na_replicates_skipped": bool(result.na_flag),
        },
        "data": {"n_resampling_units": int(n), "n_objects": len(result.labels)},
        "scales": {"r": [float(x) for x in result.r],
                   "nboot": [int(x) for x in result.nboot]},
        "labels": list(result.labels),
        "edges": [
            {"edge_id": e["edge_id"], "members": list(e["members"]),
             "n_members": int(e["n_members"]),
             "si": e["si"], "au": e["au"], "bp": e["bp"],
             "se_si": e["se_si"], "se_au": e["se_au"], "se_bp": e["se_bp"],
             "v": e["v"], "c": e["c"], "df": int(e["df"]), "pchi": e["pchi"],
             "counts": [int(c) for c in result.count[i]]}
            for i, e in enumerate(result.edges)
        ],
    }
    if extra:
        payload["extra"] = extra
    Path(path).write_text(json.dumps(payload, indent=2))


def read_project_json(path) -> dict:
    """Read a project payload, refusing a schema this version cannot interpret."""
    import json

    payload = json.loads(Path(path).read_text())
    schema = payload.get("schema", "")
    if not schema.startswith("pvclust-py/project/"):
        raise ValueError(f"{path} is not a pvclust-py project payload (schema={schema!r})")
    if schema != SCHEMA_PROJECT:
        raise ValueError(f"{path} uses schema {schema!r}; this version reads "
                         f"{SCHEMA_PROJECT!r}. Refusing rather than guessing.")
    return payload


def counts_from_json(paths) -> "pd.DataFrame":
    """Long-form counts from project payloads, ready for ``aggregate.pool_counts``.

    Refuses to mix clustering methods or orientations: pooling counts from a
    correlation/average run with a minkowski/ward.D2 run would produce a number, and
    that number would be meaningless.
    """
    rows, methods = [], set()
    for p in paths:
        payload = read_project_json(p)
        c = payload["clustering"]
        methods.add((c["cluster"], c["method_dist"], c["method_hclust"]))
        r, nboot = payload["scales"]["r"], payload["scales"]["nboot"]
        n = payload["data"]["n_resampling_units"]
        for e in payload["edges"]:
            for j, count in enumerate(e["counts"]):
                rows.append({"project": payload["project"], "edge_id": e["edge_id"],
                             "r": float(r[j]), "n": int(n), "nboot": int(nboot[j]),
                             "count": int(count)})
    if len(methods) > 1:
        raise ValueError(f"payloads use different clustering settings {sorted(methods)} "
                         f"-- their counts are not comparable and must not be pooled")
    return pd.DataFrame(rows)


def write_federated_json(edges, path, *, projects, mode: str,
                         catalogue=None, extra: Optional[dict] = None) -> None:
    """Write the aggregator's result -- what goes back to each project."""
    import json

    from . import __version__

    members = {e["edge_id"]: list(e["members"]) for e in (catalogue or [])}
    recs = edges.to_dict("records") if hasattr(edges, "to_dict") else list(edges)
    payload = {
        "schema": SCHEMA_FEDERATED,
        "pvclust_py_version": __version__,
        "projects": list(projects),
        "mode": mode,
        "edges": [
            {**{k: (v.item() if hasattr(v, "item") else v)
                for k, v in e.items() if k != "members"},
             "members": members.get(e["edge_id"],
                                    e.get("members", "").split(";") if e.get("members") else [])}
            for e in recs
        ],
    }
    if extra:
        payload["extra"] = extra
    Path(path).write_text(json.dumps(payload, indent=2))
