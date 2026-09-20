"""
The biclustered view: a cluster is a SQUARE, not an axis.

A protein dendrogram down one side and a patient dendrogram down the other answers two
separate questions and shows neither as a block. An **endotype** is a set of patients
carrying a coordinated shift across a set of proteins, which is a rectangle in the
matrix, and it only becomes visible when both axes are grouped and sorted so that
rectangle lines up.

So this module does three things, in order:

1. **Apply the federated protein blocks back.** The column grouping is not recomputed
   at the cohort. It comes from the aggregator's pooled tree, so cohort A and cohort B
   are cut the same way and a square in one can be compared to the same square in the
   other. That is the whole point of federating the protein axis.

2. **Group the patients against those blocks.** Each patient is reduced to one mean
   per protein block -- a short profile in block space rather than in 7288-protein
   space -- and patients are clustered on that. This is what makes the rows line up
   with the columns: patients are grouped by how they behave in the blocks the
   federation found, not by some independent structure.

3. **Assess afterwards.** AU and the dendrograms are computed on the result and drawn
   beside it. They report on the blocks; they do not decide them.

No k anywhere. ``pvpick`` returns whatever clears the AU threshold on each axis and
the trees cut themselves.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


def cut_blocks(result, n_blocks: int, labels: Optional[Sequence[str]] = None,
               prefix: str = "block") -> pd.Series:
    """Cut a tree into exactly ``n_blocks`` groups, for the LAYOUT.

    This is a display choice and carries no statistical claim. AU is applied
    afterwards, by :func:`assess`, which reports the support of each block once the
    blocks exist.

    Selecting blocks by AU instead does not work for a biclustered figure, and it is
    worth recording why. Taking the maximal clusters clearing 0.95 gave one block
    holding 46 of 53 proteins -- the whole figure, not a square. Taking the minimal
    ones gave nine blocks of two to five proteins and left 32 unassigned. Neither is
    a usable column grouping. A cut gives blocks of comparable size, and the
    assessment then says which of them the data supports.
    """
    from scipy.cluster.hierarchy import fcluster

    names = list(labels if labels is not None else result.labels)
    lab = fcluster(result.linkage, t=n_blocks, criterion="maxclust")
    return pd.Series([f"{prefix}{v}" for v in lab], index=names, dtype=object)


def assess(result, assign: pd.Series) -> pd.DataFrame:
    """AU for each block, applied AFTER the blocks are decided.

    A block is looked up among the tree's clusters by its exact member set. A block
    the tree never formed has no AU -- reported as missing rather than as zero, which
    would read as evidence against it.
    """
    by_members = {frozenset(e["members"]): e for e in result.edges}
    rows = []
    for b, members in assign.groupby(assign):
        e = by_members.get(frozenset(members.index))
        rows.append({"block": b, "n": len(members),
                     "au": float(e["au"]) if e else np.nan,
                     "bp": float(e["bp"]) if e else np.nan,
                     "df": int(e["df"]) if e else -1,
                     "in_tree": e is not None,
                     "members": ";".join(sorted(members.index))})
    return pd.DataFrame(rows).sort_values("n", ascending=False, ignore_index=True)


def patient_blocks(X: pd.DataFrame, blocks: pd.Series, n_groups: int, *,
                   method_dist: str = "minkowski", method_hclust: str = "ward.D2",
                   nboot: int = 1000, seed: int = 42, quiet: bool = True):
    """Group patients by how they behave IN the protein blocks.

    Each patient is reduced to one mean per protein block -- a short profile in block
    space -- and clustered on that. Clustering on all proteins instead would group
    patients by whatever dominates the panel, which need not align with the blocks and
    so would not produce squares.

    Returns ``(assignment, result)``; the result is what :func:`assess` reads.
    """
    from .core import pvclust

    scores = pd.DataFrame(
        {b: X[[c for c in members.index if c in X.columns]].mean(axis=1)
         for b, members in blocks.groupby(blocks)},
        index=X.index)
    res = pvclust(scores, cluster="rows", method_dist=method_dist,
                  method_hclust=method_hclust, nboot=nboot, seed=seed, quiet=quiet)
    return cut_blocks(res, n_groups, labels=list(X.index), prefix="group"), res


def _ordered(M: pd.DataFrame, axis: str, assign: pd.Series,
             method_dist: str, method_hclust: str):
    """Block order, and within each block the order its own tree implies."""
    from scipy.cluster.hierarchy import dendrogram as sd

    from .distance import distance
    from .hclust import linkage

    names = list(assign.value_counts().index)      # biggest block first

    order, blocks = [], []
    for b in names:
        members = list(assign.index[assign == b])
        Z = None
        if len(members) > 2:
            sub = M.loc[members] if axis == "rows" else M[members]
            A = sub.to_numpy(float).T if axis == "rows" else sub.to_numpy(float)
            D = distance(A, method_dist)
            if np.isfinite(D).all():
                Z = linkage(D, method_hclust)
                members = [members[i] for i in sd(Z, no_plot=True)["leaves"]]
        order += members
        blocks.append((b, members, Z))
    return order, blocks


def bicluster_heatmap(X: pd.DataFrame, base: str, *, proteins: pd.Series,
                      patients: pd.Series,
                      annotations: Optional[pd.DataFrame] = None,
                      method_dist: str = "minkowski",
                      method_hclust: str = "ward.D2",
                      title: Optional[str] = None, max_labels: int = 160) -> Dict:
    """Draw the biclustered heatmap -> png, svg, pdf, html.

    Both axes are grouped and sorted, with white rules between blocks, so a cluster
    reads as a **square**. The dendrograms drawn beside each axis are local to a block
    and are there to assess the block, not to order the figure.

    ``X`` must already be scaled -- log2 then z-score -- because the same matrix has
    to feed the statistics, the federated tree and this figure. Scaling here instead
    would order the blocks by one matrix and support them with another.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    from .plot import _annotation_key, _annotation_strip

    # X arrives already scaled -- log2 then z-score, the same matrix the sufficient
    # statistics and the federated tree were built from. Re-scaling here would make
    # the layout disagree with the support drawn beside it.
    Z = X
    row_order, row_blocks = _ordered(Z, "rows", patients.reindex(Z.index),
                                     method_dist, method_hclust)
    col_order, col_blocks = _ordered(Z, "columns", proteins.reindex(Z.columns),
                                     method_dist, method_hclust)
    M = Z.loc[row_order, col_order]

    n_ann = 0 if annotations is None else annotations.shape[1]
    fig = plt.figure(figsize=(max(10, min(30, 0.17 * M.shape[1] + 6)),
                              max(8, min(26, 0.13 * M.shape[0] + 5))),
                     constrained_layout=True)
    gs = GridSpec(2, 2 + (1 if n_ann else 0), figure=fig,
                  height_ratios=[1.0, 7],
                  width_ratios=([0.12 * n_ann] if n_ann else []) + [7, 0.25])
    c_main = 1 if n_ann else 0

    # block means, drawn above the columns: what each square actually is
    ax = fig.add_subplot(gs[0, c_main])
    means = []
    for _b, rmem, _z in row_blocks:
        row = []
        for _c, cmem, _z2 in col_blocks:
            row += [float(M.loc[rmem, cmem].to_numpy().mean())] * len(cmem)
        means.append(row)
    A = np.repeat(np.array(means), 1, axis=0)
    lim0 = float(np.abs(A).max()) or 1.0
    ax.imshow(A, aspect="auto", cmap="RdBu_r", vmin=-lim0, vmax=lim0,
              interpolation="nearest")
    ax.set_yticks(range(len(row_blocks)))
    ax.set_yticklabels([b for b, _m, _z in row_blocks], fontsize=7)
    ax.set_xticks(range(M.shape[1]))
    ax.set_xticklabels(col_order, rotation=90, fontsize=6)
    ax.xaxis.set_ticks_position("top")
    x = 0
    for _b, members, _z in col_blocks[:-1]:
        x += len(members)
        ax.axvline(x - 0.5, color="white", lw=2.5)
    ax.set_ylabel("block mean", fontsize=8)

    key = []
    if n_ann:
        key = _annotation_strip(fig.add_subplot(gs[1, 0]),
                                annotations.reindex(M.index), horizontal=False)

    ax = fig.add_subplot(gs[1, c_main])
    lim = float(np.nanpercentile(np.abs(M.to_numpy()), 99)) or 1.0
    im = ax.imshow(M.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim,
                   interpolation="nearest")
    x = 0
    for _b, members, _z in col_blocks[:-1]:
        x += len(members)
        ax.axvline(x - 0.5, color="white", lw=2.5)
    y = 0
    for _b, members, _z in row_blocks[:-1]:
        y += len(members)
        ax.axhline(y - 0.5, color="white", lw=2.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(f"{M.shape[1]} proteins in {len(col_blocks)} federated blocks")
    ax.set_ylabel(f"{M.shape[0]} patients in {len(row_blocks)} groups")

    fig.colorbar(im, cax=fig.add_subplot(gs[1, c_main + 1]), label="z-score")
    if key:
        _annotation_key(fig, key)
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")
    for ext in (".png", ".svg", ".pdf"):
        fig.savefig(base + ext, dpi=200 if ext == ".png" else None,
                    bbox_inches="tight")
    plt.close(fig)

    table = pd.DataFrame(
        [{"patient_group": rb, "protein_block": cb,
          "n_patients": len(rm), "n_proteins": len(cm),
          "mean_z": float(M.loc[rm, cm].to_numpy().mean())}
         for rb, rm, _z in row_blocks for cb, cm, _z2 in col_blocks])
    table.to_csv(base + "_squares.csv", index=False)
    return {"squares": table, "row_order": row_order, "col_order": col_order}
