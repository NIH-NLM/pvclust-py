"""
Command-line interface for pvclust-py.

A single typer app, one thin wrapper per single-function step, mirroring
``oadr_cpep.cli``::

  project    : cluster                 (hierarchical clustering with AU p-values)
               kmeans                  (k-means, AU from the same bootstrap)
               project-features        (this project's vocabulary + summary)
               project-stats           (sufficient statistics -- exact-mode payload)
               count-edges             (counts against a shared catalogue -- pass 2)
  aggregator : shared-features, aggregate-trees

Every step takes its inputs as EXPLICIT files and writes outputs to the current
working directory (no output-dir option; under Nextflow that is the process work dir,
published by publishDir).

INPUT IS A MATRIX. One CSV or TSV, rows = resampling units (samples), columns = the
objects clustered. ``--metadata`` adds sample annotations, ``--feature-map`` renames
columns from ids to readable names. That is the whole contract, and it is what every
command speaks.

``--rfu/--samples/--somamers`` is a convenience adapter for SomaLogic's positional
three-file layout (see :mod:`pvclust_py.somascan`), which produces exactly the same
matrix. Public SomaScan depositions are usually plain CSVs and need ``--matrix``.

Orientation: rows are the resampling units, columns are the objects clustered. Use
``--cluster rows`` to cluster the other axis -- see the warning it prints.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer

app = typer.Typer(
    add_completion=False,
    help="Hierarchical clustering with AU p-values via multiscale bootstrap, and its federated form.",
)

# --- shared input options (typer needs them declared inline per command) -------
_MATRIX = typer.Option(None, "--matrix", help="Matrix CSV/TSV: rows = resampling units, columns = objects")
_LOG2 = typer.Option(False, "--log2", help="log2-transform the matrix (abundance data is usually right-skewed)")
_FEATMAP = typer.Option(None, "--feature-map", help="Lookup table renaming matrix columns, e.g. SOMAmer ids to gene symbols")
_FEATKEY = typer.Option(None, "--feature-key", help="Feature-map column matching the matrix column names (default: its first column)")
_FEATLABEL = typer.Option(None, "--feature-label", help="Feature-map column to rename columns to (default: its second column). For the SomaScan adapter, the somamer-file column to label with, e.g. EntrezGeneSymbol")
_METADATA = typer.Option(None, "--metadata", help="Sample annotations CSV/TSV, indexed by the matrix row ids")
_ADJUST = typer.Option(
    "none", "--adjust",
    help="Batch correction before clustering: none | linear | combat. 'combat' is "
         "the standard (Johnson 2007) and needs `pip install inmoose`; 'linear' is "
         "the same correction without empirical-Bayes shrinkage, no dependency.")
_BATCHCOL = typer.Option(None, "--batch-col", help="Metadata column holding the batch label")
_PROTECT = typer.Option(
    None, "--protect",
    help="Comma-separated metadata columns whose effect must be PRESERVED. Think "
         "twice in an unsupervised analysis: protecting the outcome conditions the "
         "data on what you are trying to discover. Check --adjust-report first.")
_TECHNICAL = typer.Option(
    None, "--technical",
    help="Comma-separated annotations that are TECHNICAL rather than biological, e.g. "
         "'Batch,PlateId,ScannerID'. An association with one of these is a batch "
         "effect, not a finding, and is reported as a warning.")
_ADJREPORT = typer.Option(
    False, "--adjust-report",
    help="Print how entangled batch is with each annotation, and stop. Run this "
         "BEFORE adjusting -- a completely confounded design cannot be corrected.")
# SomaScan adapter -- one vendor's positional layout. Use --matrix for anything else.
_RFU = typer.Option(None, "--rfu", help="[SomaScan adapter] headerless RFU matrix (samples x SOMAmers)")
_SAMPLES = typer.Option(None, "--samples", help="[SomaScan adapter] sample annotation, one row per RFU row")
_SOMAMERS = typer.Option(None, "--somamers", help="[SomaScan adapter] SOMAmer annotation, one row per RFU column")
_CLUSTER = typer.Option("columns", "--cluster", help="Which axis to cluster: columns | rows")
_DIST = typer.Option("correlation", "--dist", help="correlation | abscor | uncentered | euclidean | minkowski")
_LINK = typer.Option("average", "--linkage", help="hclust method: average | complete | ward.D2 | ...")
_NBOOT = typer.Option(1000, "--n-boot", help="Bootstrap replicates per scale")
_SEED = typer.Option(42, "--seed", help="Random seed")
_PLOT = typer.Option(False, "--plot", help="Also draw the figure as .png, .svg and interactive .html")
_ANNOTATE = typer.Option(
    None, "--annotate",
    help="Comma-separated sample columns to draw as annotation strips, e.g. "
         "'Batch,PlateId,ScannerID'. This is the batch-effect check: if the "
         "clustering follows the strips rather than biology, you are seeing batch.")


_TOPVAR = typer.Option(
    None, "--top-variable",
    help="Keep only the N most variable objects. Cost grows with the SQUARE of the "
         "object count, so a full panel is impractical: ~7s for 100 objects at "
         "nboot=1000, ~4min for 1000, hours for 7000.")


def _guess_technical(columns):
    """Best guess at which annotations are technical, when --technical is not given.

    A guess, not a rule: it matches common assay-metadata names. Pass --technical
    explicitly for anything else, because a mislabelled variable turns a real finding
    into a batch warning or, worse, the reverse.
    """
    from .somascan import TECHNICAL_COLUMNS
    known = {c.lower() for c in TECHNICAL_COLUMNS}
    hints = ("batch", "plate", "scanner", "run", "slide", "well", "position",
             "lane", "flowcell", "site", "operator", "instrument")
    return [c for c in columns
            if c.lower() in known or any(h in c.lower() for h in hints)]


def _apply_adjust(X, ann, adjust, batch_col, protect, report):
    """Batch-correct the matrix, or report the confounding and stop."""
    from .adjust import adjust_batch, combat, confounding

    if report:
        if ann is None or batch_col is None:
            raise typer.BadParameter("--adjust-report needs --metadata and --batch-col")
        common = X.index.intersection(ann.index)
        out = confounding(ann.loc[common, batch_col], ann.loc[common].drop(columns=[batch_col]))
        typer.echo(out.to_string(index=False))
        typer.echo("\n  balanced -> adjust freely | partial -> consider --protect | "
                   "complete -> the design cannot separate them")
        raise typer.Exit()

    if adjust == "none":
        return X
    if adjust not in ("linear", "combat"):
        raise typer.BadParameter("--adjust must be none, linear or combat")
    if ann is None or batch_col is None:
        raise typer.BadParameter("--adjust needs --metadata (or --samples) and --batch-col")
    if batch_col not in ann.columns:
        raise typer.BadParameter(f"no {batch_col!r} column; have {list(ann.columns)[:8]}")

    common = X.index.intersection(ann.index)
    if len(common) == 0:
        raise typer.BadParameter("no sample ids shared between the matrix and the metadata")
    X, ann = X.loc[common], ann.loc[common]

    prot = None
    if protect:
        cols = [c.strip() for c in protect.split(",")]
        missing = [c for c in cols if c not in ann.columns]
        if missing:
            raise typer.BadParameter(f"no such --protect columns: {missing}")
        prot = ann[cols]

    fn = combat if adjust == "combat" else adjust_batch
    try:
        out = fn(X, ann[batch_col], protect=prot)
    except ImportError as exc:
        # ComBat's dependency is optional; say how to get it rather than
        # dumping a traceback at someone running a command-line tool.
        raise typer.BadParameter(
            f"{exc}\n\n  Install it:   pip install -e \".[combat]\"\n"
            f"  Or use:       --adjust linear   (no dependency; the same linear "
            f"correction without empirical-Bayes shrinkage)") from None
    typer.echo(f"  batch-corrected with {adjust}"
               + (f", protecting {protect}" if protect else "")
               + f" ({ann[batch_col].nunique()} batches, {len(X)} samples)")
    return out


def _load(matrix, rfu, samples, somamers, top_variable=None, cluster="columns",
          log2=False, feature_map=None, feature_key=None, feature_label=None):
    """Resolve whichever input form was given into one labelled DataFrame."""
    from .io import read_matrix
    from .somascan import read_somascan

    if matrix and rfu:
        raise typer.BadParameter("give either --matrix or the SomaScan trio, not both")
    if matrix:
        X = read_matrix(matrix, log2=log2, feature_map=feature_map,
                        feature_key=feature_key, feature_label=feature_label)
    elif rfu and samples and somamers:
        X = read_somascan(rfu, samples, somamers,
                          somamer_id=feature_label or "SeqId")[0]
    else:
        raise typer.BadParameter(
            "no input given. Use --matrix <file> (rows = samples, columns = "
            "objects), or the SomaScan adapter --rfu --samples --somamers (all "
            "three, since they are aligned by position)")

    if top_variable:
        # Filter the axis being CLUSTERED -- filtering the resampling axis instead
        # would silently throw away samples.
        axis = X.T if cluster == "rows" else X
        if top_variable >= axis.shape[1]:
            typer.echo(f"  --top-variable {top_variable} >= {axis.shape[1]} objects; keeping all")
        else:
            keep = axis.var().nlargest(top_variable).index
            X = X.loc[keep] if cluster == "rows" else X[keep]
            typer.echo(f"  kept the {top_variable} most variable of {axis.shape[1]} objects")
    return X


# ------------------------------------------------------------------ project
@app.command("cluster")
def cluster_command(
    project: str = typer.Option(..., "--project", help="Project/cohort id, e.g. BLSA_B1"),
    matrix: Optional[Path] = _MATRIX,
    rfu: Optional[Path] = _RFU,
    samples: Optional[Path] = _SAMPLES,
    somamers: Optional[Path] = _SOMAMERS,
    cluster: str = _CLUSTER,
    top_variable: Optional[int] = _TOPVAR,
    metadata: Optional[Path] = _METADATA,
    log2: bool = _LOG2,
    feature_map: Optional[Path] = _FEATMAP,
    feature_key: str = _FEATKEY,
    feature_label: str = _FEATLABEL,
    adjust: str = _ADJUST,
    batch_col: Optional[str] = _BATCHCOL,
    protect: Optional[str] = _PROTECT,
    adjust_report: bool = _ADJREPORT,
    dist: str = _DIST,
    linkage: str = _LINK,
    n_boot: int = _NBOOT,
    seed: int = _SEED,
    alpha: float = typer.Option(0.95, "--alpha", help="AU threshold for the picked clusters"),
    plot: bool = _PLOT,
):
    """Hierarchical clustering with AU p-values (pvclust)."""
    from .core import pvclust, pvpick
    from .io import write_counts, write_edges, write_project_json

    X = _load(matrix, rfu, samples, somamers, top_variable, cluster,
              log2, feature_map, feature_key, feature_label)
    _ann = None
    if metadata:
        from .io import read_metadata as _rm
        _ann = _rm(metadata)
    elif samples:
        from .somascan import read_somascan as _rs
        _ann = _rs(rfu, samples, somamers, somamer_id=feature_label or "SeqId")[1]
    X = _apply_adjust(X, _ann, adjust, batch_col, protect, adjust_report)
    res = pvclust(X, cluster=cluster, method_dist=dist, method_hclust=linkage,
                  nboot=n_boot, seed=seed, quiet=False)

    n_units = X.shape[0] if cluster == "columns" else X.shape[1]
    write_edges(res, f"{project}_edges.csv")
    write_counts(res, f"{project}_counts.csv", project, n=n_units)
    write_project_json(res, f"{project}.json", project, n=n_units)
    picked = pvpick(res, alpha)
    typer.echo(f"{project}: {len(res.edges)} clusters evaluated, {len(picked)} at AU >= {alpha}")
    typer.echo(f"  wrote {project}_edges.csv, {project}_counts.csv and "
               f"{project}.json (the federation payload)")
    if plot:
        from .plot import dendrogram
        dendrogram(res, f"{project}_dendrogram", alpha=alpha,
                   title=f"{project} — {dist} / {linkage}, nboot={n_boot}")
        typer.echo(f"  wrote {project}_dendrogram.(png|svg|html)")


@app.command("kmeans")
def kmeans_command(
    project: str = typer.Option(..., "--project", help="Project/cohort id"),
    k: int = typer.Option(..., "--k", help="Number of clusters"),
    matrix: Optional[Path] = _MATRIX,
    rfu: Optional[Path] = _RFU,
    samples: Optional[Path] = _SAMPLES,
    somamers: Optional[Path] = _SOMAMERS,
    cluster: str = _CLUSTER,
    top_variable: Optional[int] = _TOPVAR,
    metadata: Optional[Path] = _METADATA,
    log2: bool = _LOG2,
    feature_map: Optional[Path] = _FEATMAP,
    feature_key: str = _FEATKEY,
    feature_label: str = _FEATLABEL,
    adjust: str = _ADJUST,
    batch_col: Optional[str] = _BATCHCOL,
    protect: Optional[str] = _PROTECT,
    adjust_report: bool = _ADJREPORT,
    n_boot: int = _NBOOT,
    seed: int = _SEED,
    alpha: float = typer.Option(0.95, "--alpha", help="AU threshold"),
    plot: bool = _PLOT,
    jaccard: Optional[float] = typer.Option(
        None, "--jaccard",
        help="Relax exact matching to this Jaccard similarity. Needed above a few "
             "hundred objects, but changes the estimand -- report it as stability, "
             "not as an AU p-value."),
):
    """k-means clustering, with AU p-values from the same multiscale bootstrap."""
    from .core import kmeans_pv
    from .io import write_counts, write_edges, write_project_json

    X = _load(matrix, rfu, samples, somamers, top_variable, cluster,
              log2, feature_map, feature_key, feature_label)
    _ann = None
    if metadata:
        from .io import read_metadata as _rm
        _ann = _rm(metadata)
    elif samples:
        from .somascan import read_somascan as _rs
        _ann = _rs(rfu, samples, somamers, somamer_id=feature_label or "SeqId")[1]
    X = _apply_adjust(X, _ann, adjust, batch_col, protect, adjust_report)
    res = kmeans_pv(X, k=k, cluster=cluster, nboot=n_boot, seed=seed, jaccard=jaccard)

    write_edges(res, f"{project}_kmeans{k}_edges.csv")
    write_counts(res, f"{project}_kmeans{k}_counts.csv", project,
                 n=X.shape[0] if cluster == "columns" else X.shape[1])
    if plot:
        from .plot import dendrogram
        # The tree is over the k CENTROIDS -- it shows how the clusters relate to one
        # another. Its leaves are clusters, not the original objects.
        dendrogram(res, f"{project}_kmeans{k}_dendrogram", alpha=alpha,
                   label_nodes=False,
                   title=f"{project} — k-means k={k}, dendrogram over cluster centroids")
        typer.echo(f"  wrote {project}_kmeans{k}_dendrogram.(png|svg|html)")

    typer.echo(f"{project}: k={k}, AU per cluster -> {project}_kmeans{k}_edges.csv")
    for e in sorted(res.edges, key=lambda e: -e["au"]):
        typer.echo(f"    AU={e['au']:.3f} BP={e['bp']:.3f} n={e['n_members']}")


@app.command("project-features")
def project_features_command(
    project: str = typer.Option(..., "--project", help="Project/cohort id"),
    matrix: Optional[Path] = _MATRIX,
    rfu: Optional[Path] = _RFU,
    samples: Optional[Path] = _SAMPLES,
    somamers: Optional[Path] = _SOMAMERS,
    cluster: str = _CLUSTER,
    top_variable: Optional[int] = _TOPVAR,
    log2: bool = _LOG2,
    feature_map: Optional[Path] = _FEATMAP,
    feature_key: str = _FEATKEY,
    feature_label: str = _FEATLABEL,
    metadata: Optional[Path] = _METADATA,
    adjust: str = _ADJUST,
    batch_col: Optional[str] = _BATCHCOL,
    protect: Optional[str] = _PROTECT,
    adjust_report: bool = _ADJREPORT,
):
    """This project's clustering vocabulary, with a per-feature summary."""
    import pandas as pd
    from .core import orient

    X = _load(matrix, rfu, samples, somamers, top_variable, cluster,
              log2, feature_map, feature_key, feature_label)
    _ann = None
    if metadata:
        from .io import read_metadata as _rm
        _ann = _rm(metadata)
    elif samples:
        from .somascan import read_somascan as _rs
        _ann = _rs(rfu, samples, somamers, somamer_id=feature_label or "SeqId")[1]
    X = _apply_adjust(X, _ann, adjust, batch_col, protect, adjust_report)
    A, labels, _ = orient(X, cluster=cluster)
    df = pd.DataFrame({
        "feature": labels,
        "n_present": (~pd.isna(A)).sum(axis=0),
        "mean": pd.DataFrame(A).mean().to_numpy(),
        "sd": pd.DataFrame(A).std().to_numpy(),
    })
    df["project"] = project
    df.to_csv(f"{project}_features.csv", index=False)
    typer.echo(f"{project}: {len(df)} features -> {project}_features.csv")


@app.command("project-stats")
def project_stats_command(
    project: str = typer.Option(..., "--project", help="Project/cohort id"),
    matrix: Optional[Path] = _MATRIX,
    rfu: Optional[Path] = _RFU,
    samples: Optional[Path] = _SAMPLES,
    somamers: Optional[Path] = _SOMAMERS,
    cluster: str = _CLUSTER,
    top_variable: Optional[int] = _TOPVAR,
    log2: bool = _LOG2,
    feature_map: Optional[Path] = _FEATMAP,
    feature_key: str = _FEATKEY,
    feature_label: str = _FEATLABEL,
    metadata: Optional[Path] = _METADATA,
    adjust: str = _ADJUST,
    batch_col: Optional[str] = _BATCHCOL,
    protect: Optional[str] = _PROTECT,
    adjust_report: bool = _ADJREPORT,
    dist: str = _DIST,
    min_n: int = typer.Option(20, "--min-n", help="Refuse to emit below this many rows"),
):
    """Sufficient statistics -- the exact-mode federation payload.

    Four p x p matrices whose entries are sums over rows, so the aggregator can add
    them and recover the pooled distance matrix exactly. No row leaves the project.
    """
    from .core import orient
    from .distance import listwise_stats, pairwise_stats
    from .io import write_stats

    X = _load(matrix, rfu, samples, somamers, top_variable, cluster,
              log2, feature_map, feature_key, feature_label)
    _ann = None
    if metadata:
        from .io import read_metadata as _rm
        _ann = _rm(metadata)
    elif samples:
        from .somascan import read_somascan as _rs
        _ann = _rs(rfu, samples, somamers, somamer_id=feature_label or "SeqId")[1]
    X = _apply_adjust(X, _ann, adjust, batch_col, protect, adjust_report)
    A, labels, _ = orient(X, cluster=cluster)
    n_rows, n_obj = A.shape
    if n_rows < min_n:
        raise typer.BadParameter(
            f"only {n_rows} rows, below --min-n={min_n}. At n=1 the Gram matrix is "
            f"rank-1 and returns the individual exactly; at any small n it exposes "
            f"the subspace they occupy. Lower --min-n only deliberately.")
    if n_rows <= n_obj:
        typer.echo(
            f"  WARNING: {n_rows} rows but {n_obj} objects. rank(G) = min(n, p) = "
            f"{min(n_rows, n_obj)}, so the statistics expose the exact "
            f"{n_rows}-dimensional subspace these rows occupy -- individuals are "
            f"hidden only by a rotation within it.\n"
            f"           Keep objects BELOW rows for the exact path (--top-variable "
            f"{max(1, n_rows - 1)} or fewer here), or use counts mode, whose payload "
            f"is integer tallies and carries no subspace.")

    stats = listwise_stats(A) if dist == "uncentered" else pairwise_stats(A)
    write_stats(stats, f"{project}_stats.npz")
    Path(f"{project}_labels.txt").write_text("\n".join(labels))
    typer.echo(f"{project}: {A.shape[0]} rows, {A.shape[1]} objects "
               f"-> {project}_stats.npz ({4 * A.shape[1] ** 2:,} numbers)")


@app.command("count-edges")
def count_edges_command(
    project: str = typer.Option(..., "--project", help="Project/cohort id"),
    catalogue: Path = typer.Option(..., "--catalogue", help="Federated edge catalogue CSV (edge_id, members)"),
    matrix: Optional[Path] = _MATRIX,
    rfu: Optional[Path] = _RFU,
    samples: Optional[Path] = _SAMPLES,
    somamers: Optional[Path] = _SOMAMERS,
    cluster: str = _CLUSTER,
    top_variable: Optional[int] = _TOPVAR,
    log2: bool = _LOG2,
    feature_map: Optional[Path] = _FEATMAP,
    feature_key: str = _FEATKEY,
    feature_label: str = _FEATLABEL,
    metadata: Optional[Path] = _METADATA,
    adjust: str = _ADJUST,
    batch_col: Optional[str] = _BATCHCOL,
    protect: Optional[str] = _PROTECT,
    adjust_report: bool = _ADJREPORT,
    dist: str = _DIST,
    linkage: str = _LINK,
    method: str = typer.Option("hclust", "--method",
        help="How each replicate is clustered: hclust | kmeans. MUST match how the "
             "catalogue was built, or every project is answering a different question."),
    k: Optional[int] = typer.Option(None, "--k", help="Clusters per replicate, for --method kmeans"),
    jaccard: Optional[float] = typer.Option(None, "--jaccard",
        help="Relaxed matching for kmeans (0.75 is conventional). Effectively required "
             "there: exact k-means matching collapses to zero above a few dozen objects."),
    n_boot: int = _NBOOT,
    seed: int = _SEED,
):
    """Count how often the aggregator's clusters reappear in this project's bootstrap.

    Pass 2 of counts-mode federation: every project counts the SAME clusters, which is
    what makes the tallies addable.
    """
    import pandas as pd
    from .core import count_edges, orient

    X = _load(matrix, rfu, samples, somamers, top_variable, cluster,
              log2, feature_map, feature_key, feature_label)
    _ann = None
    if metadata:
        from .io import read_metadata as _rm
        _ann = _rm(metadata)
    elif samples:
        from .somascan import read_somascan as _rs
        _ann = _rs(rfu, samples, somamers, somamer_id=feature_label or "SeqId")[1]
    X = _apply_adjust(X, _ann, adjust, batch_col, protect, adjust_report)
    A, labels, _ = orient(X, cluster=cluster)
    cat = pd.read_csv(catalogue)
    members = [m.split(";") for m in cat["members"]]

    counts, r_eff, nboot_vec, na = count_edges(
        A, members, labels, method_dist=dist, method_hclust=linkage,
        method=method, k=k, jaccard=jaccard, nboot=n_boot, seed=seed)

    pd.DataFrame([
        {"project": project, "edge_id": cat["edge_id"].iloc[i], "r": float(r_eff[j]),
         "n": A.shape[0], "nboot": int(nboot_vec[j]), "count": int(counts[i, j])}
        for i in range(len(members)) for j in range(len(r_eff))
    ]).to_csv(f"{project}_counts.csv", index=False)
    typer.echo(f"{project}: counted {len(members)} catalogue clusters -> {project}_counts.csv"
               + ("  (some replicates skipped: non-finite distances)" if na else ""))


@app.command("heatmap")
def heatmap_command(
    project: str = typer.Option(..., "--project", help="Project/cohort id -- names the output files"),
    matrix: Optional[Path] = _MATRIX,
    rfu: Optional[Path] = _RFU,
    samples: Optional[Path] = _SAMPLES,
    somamers: Optional[Path] = _SOMAMERS,
    top_variable: Optional[int] = _TOPVAR,
    log2: bool = _LOG2,
    feature_map: Optional[Path] = _FEATMAP,
    feature_key: str = _FEATKEY,
    feature_label: str = _FEATLABEL,
    adjust: str = _ADJUST,
    batch_col: Optional[str] = _BATCHCOL,
    protect: Optional[str] = _PROTECT,
    adjust_report: bool = _ADJREPORT,
    method: str = typer.Option("pvclust", "--method", help="pvclust | kmeans"),
    k: Optional[int] = typer.Option(None, "--k", help="Clusters per axis, for --method kmeans"),
    dist: str = _DIST,
    linkage: str = _LINK,
    n_boot: int = _NBOOT,
    seed: int = _SEED,
    alpha: float = typer.Option(0.95, "--alpha", help="AU threshold for the red boxes"),
    jaccard: Optional[float] = typer.Option(None, "--jaccard", help="Relaxed matching for k-means"),
    annotate: Optional[str] = _ANNOTATE,
    metadata: Optional[Path] = _METADATA,
    max_rows: int = typer.Option(80, "--max-rows", help="Subsample rows above this, to keep the figure legible"),
):
    """Two-way clustered heatmap: dendrograms on BOTH axes, with the AU boxes.

    Clusters the columns and the rows, then draws the matrix in that order. Writes
    .png, .svg and an interactive .html.

    Note that the row dendrogram (samples) resamples analytes, which are correlated
    rather than independent, so its AU values are anti-conservative. It is the right
    figure to look at; just do not quote the row p-values like the column ones.
    """
    from .core import kmeans_pv, pvclust
    from .io import read_metadata
    from .somascan import read_somascan
    from .plot import heatmap

    if method not in ("pvclust", "kmeans"):
        raise typer.BadParameter("--method must be pvclust or kmeans")
    if method == "kmeans" and not k:
        raise typer.BadParameter("--method kmeans needs --k")

    X = _load(matrix, rfu, samples, somamers, top_variable, "columns",
              log2, feature_map, feature_key, feature_label)

    # Annotations first: both the adjustment and the annotation strips need them.
    ann = read_metadata(metadata) if metadata else None
    X = _apply_adjust(X, ann, adjust, batch_col, protect, adjust_report)

    if annotate:
        if metadata:
            smp = ann
        elif samples:
            _, smp, _ = read_somascan(rfu, samples, somamers, somamer_id=label_by)
        else:
            raise typer.BadParameter("--annotate needs --metadata (any matrix) or "
                                     "--samples (the SomaScan layout)")
        cols = [c.strip() for c in annotate.split(",")]
        missing = [c for c in cols if c not in smp.columns]
        if missing:
            raise typer.BadParameter(f"no such sample columns: {missing}")
        ann = smp[cols]

    if len(X) > max_rows:
        X = X.sample(max_rows, random_state=seed).sort_index()
        typer.echo(f"  subsampled to {max_rows} rows for legibility (--max-rows)")
    if ann is not None:
        ann = ann.loc[X.index]

    if method == "pvclust":
        kw = dict(method_dist=dist, method_hclust=linkage, nboot=n_boot, seed=seed)
        cols_res = pvclust(X, **kw)
        rows_res = pvclust(X, cluster="rows", **kw)
    else:
        cols_res = kmeans_pv(X, k=k, nboot=n_boot, seed=seed, jaccard=jaccard)
        rows_res = kmeans_pv(X, k=k, cluster="rows", nboot=n_boot, seed=seed, jaccard=jaccard)

    base = f"{project}_heatmap_{method}"
    heatmap(X, base, row_result=rows_res, col_result=cols_res, alpha=alpha,
            row_annotations=ann,
            title=f"{project} — {method} ({X.shape[0]} rows x {X.shape[1]} columns)")
    typer.echo(f"{project}: wrote {base}.(png|svg|html)")


@app.command("diagnose")
def diagnose_command(
    project: str = typer.Option(..., "--project", help="Project/cohort id"),
    matrix: Optional[Path] = _MATRIX,
    rfu: Optional[Path] = _RFU,
    samples: Optional[Path] = _SAMPLES,
    somamers: Optional[Path] = _SOMAMERS,
    metadata: Optional[Path] = _METADATA,
    annotate: Optional[str] = typer.Option(
        None, "--annotate", help="Comma-separated annotation columns; default is all of them"),
    top_variable: Optional[int] = _TOPVAR,
    log2: bool = _LOG2,
    feature_map: Optional[Path] = _FEATMAP,
    feature_key: str = _FEATKEY,
    feature_label: str = _FEATLABEL,
    adjust: str = _ADJUST,
    batch_col: Optional[str] = _BATCHCOL,
    protect: Optional[str] = _PROTECT,
    adjust_report: bool = _ADJREPORT,
    technical: Optional[str] = _TECHNICAL,
    k: int = typer.Option(3, "--k", help="Cut the sample dendrogram into this many clusters"),
    dist: str = _DIST,
    linkage: str = _LINK,
    n_boot: int = _NBOOT,
    seed: int = _SEED,
):
    """Is the clustering driven by biology, or by the plate it was run on?

    Clusters the SAMPLES, then tests the assignment against every annotation:
    chi-square + Cramer's V for categorical, Kruskal-Wallis + eta-squared for numeric,
    and adjusted Rand for how far a categorical variable simply IS the clustering.

    AU tells you a cluster is reproducible, not that it is meaningful -- a batch
    effect is perfectly reproducible. This is the check that separates the two.
    """
    from .core import pvclust
    from .diagnostics import association, cluster_labels, report
    from .io import read_metadata
    from .somascan import read_somascan

    X = _load(matrix, rfu, samples, somamers, top_variable, "columns",
              log2, feature_map, feature_key, feature_label)

    # Annotations first: adjustment needs them, and so does the association test.
    if metadata:
        ann = read_metadata(metadata)
    elif samples:
        ann = read_somascan(rfu, samples, somamers, somamer_id=feature_label or "SeqId")[1]
    else:
        raise typer.BadParameter("diagnose needs --metadata, or the SomaScan --samples file")

    X = _apply_adjust(X, ann, adjust, batch_col, protect, adjust_report)

    if annotate:
        cols = [c.strip() for c in annotate.split(",")]
        missing = [c for c in cols if c not in ann.columns]
        if missing:
            raise typer.BadParameter(f"no such annotation columns: {missing}")
        ann = ann[cols]

    res = pvclust(X, cluster="rows", method_dist=dist, method_hclust=linkage,
                  nboot=n_boot, seed=seed)
    labels = cluster_labels(res, k=k)
    common = labels.index.intersection(ann.index)
    if len(common) == 0:
        raise typer.BadParameter(
            f"no sample ids in common. Matrix rows look like {list(labels.index[:3])}, "
            f"annotation index like {list(ann.index[:3])}")

    assoc = association(labels, ann.loc[common])
    assoc.to_csv(f"{project}_diagnostics.csv", index=False)

    typer.echo(f"\n{project}: {k} clusters over {len(common)} samples\n")
    show = [c for c in ("variable", "test", "p_value", "effect", "effect_name",
                        "adjusted_rand", "strength") if c in assoc.columns]
    typer.echo(assoc[show].to_string(index=False))
    tech = ([c.strip() for c in technical.split(",")] if technical
            else _guess_technical(ann.columns))
    typer.echo("\n" + report(assoc, batch_like=tech))
    typer.echo(f"\nwrote {project}_diagnostics.csv")


@app.command("apply-edges")
def apply_edges_command(
    project: str = typer.Option(..., "--project", help="Project/cohort id"),
    federated_edges: Path = typer.Option(..., "--federated-edges",
        help="The aggregator's federated_edges.csv (or federated_catalogue.csv)"),
    matrix: Optional[Path] = _MATRIX,
    rfu: Optional[Path] = _RFU,
    samples: Optional[Path] = _SAMPLES,
    somamers: Optional[Path] = _SOMAMERS,
    top_variable: Optional[int] = _TOPVAR,
    feature_map: Optional[Path] = _FEATMAP,
    feature_key: Optional[str] = _FEATKEY,
    feature_label: Optional[str] = _FEATLABEL,
    log2: bool = _LOG2,
    cluster: str = _CLUSTER,
    dist: str = _DIST,
    linkage: str = _LINK,
    n_boot: int = _NBOOT,
    seed: int = _SEED,
    alpha: float = typer.Option(0.95, "--alpha", help="AU threshold"),
):
    """Round 3: what the federation gives back to THIS project.

    Takes the aggregator's clusters and evaluates them against local data. Reports the
    project's own AU beside the federated AU, how many federated clusters it would have
    found alone, and writes one module score per sample -- a federation-validated
    feature space to carry into whatever comes next.

    No other project's data is needed; the federated artifacts carry only cluster
    memberships and fitted p-values.
    """
    from .apply import apply_edges

    X = _load(matrix, rfu, samples, somamers, top_variable, cluster,
              log2, feature_map, feature_key, feature_label)
    out = apply_edges(X, federated_edges, method_dist=dist, method_hclust=linkage,
                      nboot=n_boot, seed=seed, alpha=alpha, cluster=cluster)

    out["support"].to_csv(f"{project}_from-federated_edge_support.csv", index=False)
    out["agreement"].to_csv(f"{project}_from-federated_agreement.csv", index=False)
    out["scores"].to_csv(f"{project}_from-federated_module_scores.csv")

    typer.echo(f"\n{project} — this project's outcome using the federated clusters\n")
    typer.echo(out["agreement"].to_string(index=False))
    gained = out["support"][out["support"]["gain"] > 0.05]
    if len(gained):
        typer.echo(f"\n{len(gained)} clusters where the federation helped most:")
        for _, r in gained.head(5).iterrows():
            typer.echo(f"  AU {r['au_local']:.3f} alone -> {r['au_federated']:.3f} federated"
                       f"  (n={int(r['n_members'])})  {r['members'][:60]}")
    typer.echo(f"\nwrote {project}_from-federated_"
               f"{{edge_support,agreement,module_scores}}.csv")


# --------------------------------------------------------------- aggregator
@app.command("shared-features")
def shared_features_command(
    features: List[Path] = typer.Option(..., "--features", help="Per-project features CSV (repeat)"),
):
    """The shared vocabulary = the intersection of the per-project feature lists."""
    import pandas as pd
    from .aggregate import shared_features

    shared = shared_features(pd.read_csv(f)["feature"].astype(str) for f in features)
    pd.DataFrame({"feature": shared}).to_csv("shared_features.csv", index=False)
    typer.echo(f"{len(shared)} shared features from {len(features)} projects "
               f"-> shared_features.csv")


@app.command("aggregate-trees")
def aggregate_trees_command(
    labels: Path = typer.Option(..., "--labels", help="Object labels, one per line (from project-stats)"),
    stats: List[Path] = typer.Option(None, "--stats", help="Per-project stats npz (repeat) -- exact mode"),
    counts: List[Path] = typer.Option(None, "--counts", help="Per-project counts, .json payload or .csv (repeat) -- counts mode"),
    dist: str = _DIST,
    linkage: str = _LINK,
    alpha: float = typer.Option(0.95, "--alpha", help="AU threshold for the consensus set"),
    partition: str = typer.Option("hclust", "--partition",
        help="How to build the catalogue from the pooled distance matrix: hclust "
             "(a dendrogram) or kmeans (a flat k-medoids partition, needs --k)"),
    k: Optional[int] = typer.Option(None, "--k", help="Clusters, for --partition kmeans"),
):
    """Combine per-project artifacts into the federated tree and AU p-values.

    ``--stats`` pools sufficient statistics: the federated distance matrix, and hence
    the tree, is exactly the one you would get from pooled raw data.
    ``--counts`` pools bootstrap tallies to give federated AU. Give both for the full
    result; only fitted parameters are ever read, never subject-level data.
    """
    import pandas as pd
    from .aggregate import (consensus_clusters, federated_edges, federated_tree,
                            pool_counts)
    from .io import read_stats

    if not stats and not counts:
        raise typer.BadParameter("give --stats (exact mode) and/or --counts (counts mode)")

    names = [l for l in Path(labels).read_text().splitlines() if l]
    catalogue = None

    if stats:
        D, Z, catalogue = federated_tree([read_stats(s) for s in stats], names,
                                         method_dist=dist, method_hclust=linkage)
        if partition == "kmeans":
            if not k:
                raise typer.BadParameter("--partition kmeans needs --k")
            from .aggregate import federated_kmeans
            catalogue = federated_kmeans(D, names, k=k)
            typer.echo(f"  k-medoids partition on the pooled distance matrix, k={k}")
        elif partition != "hclust":
            raise typer.BadParameter("--partition must be hclust or kmeans")
        pd.DataFrame(D, index=names, columns=names).to_csv("federated_distance.csv")
        pd.DataFrame([{**e, "members": ";".join(e["members"])} for e in catalogue]
                     ).to_csv("federated_catalogue.csv", index=False)
        typer.echo(f"pooled {len(stats)} projects -> federated_distance.csv, "
                   f"federated_catalogue.csv ({len(catalogue)} clusters)")

    if counts:
        from .io import counts_from_json
        js = [c for c in counts if str(c).endswith(".json")]
        cs = [c for c in counts if not str(c).endswith(".json")]
        frames = ([counts_from_json(js)] if js else []) + [pd.read_csv(c) for c in cs]
        pooled = pool_counts(frames)
        edges = federated_edges(pooled, catalogue)
        edges.to_csv("federated_edges.csv", index=False)
        cons = consensus_clusters(edges, alpha=alpha)
        typer.echo(f"pooled counts from {len(counts)} projects across "
                   f"{pooled['r'].nunique()} scales -> federated_edges.csv")
        typer.echo(f"  {len(cons)} clusters at AU >= {alpha}")


def main():
    app()


if __name__ == "__main__":
    main()
