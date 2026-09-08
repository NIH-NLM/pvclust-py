"""
Plotting for pvclust-py. All graphics live here (matplotlib for PNG/SVG, plotly for
the self-contained interactive HTML) so the compute modules stay free of rendering
code -- the same split as ``oadr_cpep.plot``.

The dendrogram reproduces what R's ``plot.pvclust`` + ``pvrect`` show, because that
figure is how most people read a pvclust result:

  * **AU** printed in red on the left of each node, as a percentage
  * **BP** printed in green on the right
  * red rectangles around the clusters that clear the AU threshold

Reading it: a node with a high red number and a much lower green one is a cluster the
ordinary bootstrap would have discarded and AU rescues -- which is the entire reason
for using pvclust rather than a bootstrapped tree.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

AU_COLOUR = "#c0392b"      # red, as in R
BP_COLOUR = "#27ae60"      # green, as in R
EDGE_COLOUR = "#2980b9"    # blue edge numbers, as in R


def _leaf_positions(dd) -> dict:
    """Map each leaf label to its x coordinate in scipy's dendrogram space."""
    return {lab: 5.0 + 10.0 * i for i, lab in enumerate(dd["ivl"])}


def _node_xy(Z, labels, dd):
    """(x, y) of every internal node, indexed by merge order.

    scipy draws each merge as a bracket; the node sits at the midpoint of the two
    joined subtrees, at the merge height.
    """
    n = len(labels)
    pos = _leaf_positions(dd)
    centre, out = {}, []
    for i, (a, b, h, _) in enumerate(Z):
        xa = pos[labels[int(a)]] if int(a) < n else centre[int(a) - n]
        xb = pos[labels[int(b)]] if int(b) < n else centre[int(b) - n]
        x = (xa + xb) / 2.0
        centre[i] = x
        out.append((x, float(h)))
    return out


def dendrogram(result, base: str, alpha: float = 0.95, *, use: str = "au",
               title: Optional[str] = None, figsize=None, label_nodes: bool = True,
               sort_by_support: Optional[str] = "au",
               max_leaf_labels: int = 80) -> None:
    """Draw the pvclust dendrogram -> ``<base>.png``, ``.svg`` and interactive ``.html``.

    Args:
        result: a :class:`~pvclust_py.core.PvclustResult`.
        base: output path stem, without extension.
        alpha: threshold for the red rectangles.
        use: which p-value the rectangles use -- ``au`` (default), ``bp`` or ``si``.
        label_nodes: print AU/BP at each node. Turn off for large trees, where the
            numbers overlap into illegibility.
        max_leaf_labels: above this many leaves, leaf names are suppressed rather than
            drawn as an unreadable smear.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram

    from .core import pvpick

    Z, labels = result.linkage, result.labels
    if Z.size == 0:
        raise ValueError("this result has no dendrogram to draw")

    n = len(labels)
    show_leaves = n <= max_leaf_labels
    figsize = figsize or (max(8.0, min(28.0, 0.22 * n + 4)), 6.5)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    dd = scipy_dendrogram(_support_ordered(result, sort_by_support),
                          labels=labels, ax=ax, color_threshold=0,
                          above_threshold_color="#555555",
                          no_labels=not show_leaves)

    nodes = _node_xy(Z, labels, dd)
    if label_nodes and n <= max_leaf_labels:
        for i, (x, y) in enumerate(nodes[:-1]):        # skip the root: always 100
            e = result.edges[i]
            ax.text(x - 1.5, y, f"{e['au'] * 100:.0f}", color=AU_COLOUR,
                    fontsize=7, ha="right", va="bottom", fontweight="bold")
            ax.text(x + 1.5, y, f"{e['bp'] * 100:.0f}", color=BP_COLOUR,
                    fontsize=7, ha="left", va="bottom")

    # red rectangles round the significant clusters (R's pvrect)
    pos = _leaf_positions(dd)
    picked = pvpick(result, alpha, use=use)
    for e in picked:
        xs = [pos[m] for m in e["members"] if m in pos]
        if not xs:
            continue
        i = e["merge_order"] - 1
        height = nodes[i][1] if i < len(nodes) else float(Z[:, 2].max())
        ax.add_patch(Rectangle((min(xs) - 4, 0), (max(xs) - min(xs)) + 8, height * 1.02,
                               fill=False, edgecolor=AU_COLOUR, linewidth=1.4))

    ax.set_title(title or (f"{result.method_dist} / {result.method_hclust} — "
                           f"{n} objects ({result.cluster}), nboot={int(result.nboot[0])}\n"
                           f"{len(picked)} clusters at {use.upper()} ≥ {alpha}"),
                 fontweight="bold", fontsize=11)
    ax.set_ylabel("height")
    if show_leaves:
        plt.setp(ax.get_xticklabels(), rotation=90, fontsize=7)
    else:
        ax.set_xlabel(f"{n} objects (labels suppressed above {max_leaf_labels})")

    handles = [plt.Line2D([], [], color=AU_COLOUR, lw=2, label="AU (approximately unbiased)"),
               plt.Line2D([], [], color=BP_COLOUR, lw=2, label="BP (ordinary bootstrap)")]
    ax.legend(handles=handles, loc="upper right", fontsize=8, frameon=False)

    fig.savefig(base + ".png", dpi=220)
    fig.savefig(base + ".svg")
    plt.close(fig)

    _interactive(result, dd, nodes, picked, base, alpha, use)


def _interactive(result, dd, nodes, picked, base, alpha, use) -> None:
    """Self-contained plotly HTML: hover a node for its full statistics."""
    try:
        import plotly.graph_objects as go
    except Exception:                       # plotly optional; PNG/SVG still produced
        return

    fig = go.Figure()
    for xs, ys in zip(dd["icoord"], dd["dcoord"]):
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", showlegend=False,
                                 line=dict(color="#555555", width=1),
                                 hoverinfo="skip"))

    picked_ids = {e["edge_id"] for e in picked}
    xs, ys, text, colours = [], [], [], []
    for i, (x, y) in enumerate(nodes[:-1]):
        e = result.edges[i]
        xs.append(x); ys.append(y)
        colours.append(AU_COLOUR if e["edge_id"] in picked_ids else "#95a5a6")
        text.append(
            f"<b>{e['n_members']} members</b><br>"
            f"AU {e['au']:.3f} ± {e['se_au']:.3f}<br>"
            f"BP {e['bp']:.3f} ± {e['se_bp']:.3f}<br>"
            f"SI {e['si']:.3f}<br>"
            f"pchi {e['pchi']:.3f} (fit quality)<br>"
            f"<extra></extra>{', '.join(e['members'][:8])}"
            + ("…" if e["n_members"] > 8 else ""))

    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", showlegend=False,
                             marker=dict(size=7, color=colours),
                             hovertemplate="%{text}", text=text))
    fig.update_layout(
        title=f"{result.method_dist} / {result.method_hclust} — "
              f"{len(result.labels)} objects, {len(picked)} at {use.upper()} ≥ {alpha}",
        xaxis=dict(showticklabels=False), yaxis_title="height",
        width=min(1500, 260 + 22 * len(result.labels)), height=560)
    fig.write_html(base + ".html")


# --------------------------------------------------------------------- heatmap
def _support_ordered(result, sort_by_support):
    """The linkage to draw: rotated by support, or as computed."""
    if not sort_by_support or result.is_kmeans or not result.linkage.size:
        return result.linkage
    from .hclust import rotate_by_support
    return rotate_by_support(result.linkage, result.edges, key=sort_by_support)


def _order_and_blocks(result, labels, sort_by_support=None):
    """Leaf order and cluster blocks for one axis of the heatmap.

    A pvclust result orders by its dendrogram, optionally rotated so better-supported
    subtrees come first. A k-means result orders by cluster, laid out in the order its
    CENTROID dendrogram implies, so neighbouring blocks are the most similar clusters
    rather than merely the largest.
    """
    if result is None:
        return list(labels), None, None
    if not result.is_kmeans:                                   # hierarchical
        from scipy.cluster.hierarchy import dendrogram as sd
        dd = sd(_support_ordered(result, sort_by_support),
                labels=result.labels, no_plot=True)
        return list(dd["ivl"]), result, dd
    # k-means: lay the clusters out in the order its CENTROID dendrogram implies, so
    # neighbouring blocks are the most similar clusters rather than merely the largest.
    from scipy.cluster.hierarchy import dendrogram as sd
    if result.linkage.size:
        leaf_order = sd(result.linkage, no_plot=True)["leaves"]
    else:
        leaf_order = sorted(range(len(result.edges)),
                            key=lambda i: -result.edges[i]["n_members"])
    order, blocks = [], []
    for idx in leaf_order:
        e = result.edges[idx]
        blocks.append((len(order), len(order) + e["n_members"], e))
        order.extend(e["members"])
    return order, result, blocks


def heatmap(matrix, base: str, *, row_result=None, col_result=None, alpha: float = 0.95,
            row_annotations=None, col_annotations=None, title: Optional[str] = None,
            cmap: str = "RdBu_r", z_score: Optional[str] = "columns",
            sort_by_support: Optional[str] = "au", max_labels: int = 60) -> None:
    """Clustered heatmap with dendrograms -> ``<base>.png``, ``.svg`` and ``.html``.

    Args:
        matrix: DataFrame, rows x columns, as clustered.
        row_result, col_result: results for each axis. Either may be a
            :func:`~pvclust_py.core.pvclust` result (drawn as a dendrogram with red
            boxes on the significant clusters) or a
            :func:`~pvclust_py.core.kmeans_pv` result (drawn as a coloured cluster
            bar, since k-means has no tree). Omit one to leave that axis unordered.
        row_annotations: DataFrame indexed like the rows -- one coloured strip per
            column. **This is how you check for batch effects**: annotate samples with
            PlateId / Batch / ScannerID and look at whether the clustering follows
            them rather than biology.
        z_score: standardise ``"rows"``, ``"columns"`` or ``None``. Without it a few
            high-abundance analytes dominate the colour scale and everything else is
            flat.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Rectangle
    from scipy.cluster.hierarchy import dendrogram as sd

    from .core import pvpick

    row_order, rres, rextra = _order_and_blocks(row_result, matrix.index, sort_by_support)
    col_order, cres, cextra = _order_and_blocks(col_result, matrix.columns, sort_by_support)
    M = matrix.loc[row_order, col_order]

    if z_score == "columns":
        M = (M - M.mean()) / M.std().replace(0, np.nan)
    elif z_score == "rows":
        M = M.sub(M.mean(axis=1), axis=0).div(M.std(axis=1).replace(0, np.nan), axis=0)
    M = M.fillna(0.0)

    n_rann = 0 if row_annotations is None else row_annotations.shape[1]
    n_cann = 0 if col_annotations is None else col_annotations.shape[1]

    fig = plt.figure(figsize=(max(9, min(26, 0.16 * M.shape[1] + 5)),
                              max(7, min(24, 0.14 * M.shape[0] + 4))),
                     constrained_layout=True)
    gs = GridSpec(3 + (1 if n_cann else 0), 3 + (1 if n_rann else 0), figure=fig,
                  height_ratios=([1.6] + ([0.12 * n_cann] if n_cann else []) + [6, 0.35]),
                  width_ratios=([1.4] + ([0.12 * n_rann] if n_rann else []) + [6, 0.25]))
    r_main = 1 + (1 if n_cann else 0)
    c_main = 1 + (1 if n_rann else 0)

    # --- column dendrogram / cluster bar, across the top ----------------------
    if col_result is not None:
        ax = fig.add_subplot(gs[0, c_main])
        _axis_marker(ax, col_result, cextra, alpha, len(col_order), True, sort_by_support)

    # --- row dendrogram / cluster bar, down the left --------------------------
    if row_result is not None:
        ax = fig.add_subplot(gs[r_main, 0])
        _axis_marker(ax, row_result, rextra, alpha, len(row_order), False, sort_by_support)

    # --- annotation strips ----------------------------------------------------
    if n_cann:
        ax = fig.add_subplot(gs[1, c_main])
        _annotation_strip(ax, col_annotations.loc[col_order], horizontal=True)
    if n_rann:
        ax = fig.add_subplot(gs[r_main, 1])
        _annotation_strip(ax, row_annotations.loc[row_order], horizontal=False)

    # --- the heatmap itself ---------------------------------------------------
    ax = fig.add_subplot(gs[r_main, c_main])
    lim = float(np.nanpercentile(np.abs(M.to_numpy()), 99)) or 1.0
    im = ax.imshow(M.to_numpy(), aspect="auto", cmap=cmap, vmin=-lim, vmax=lim,
                   interpolation="nearest")
    ax.set_xticks(range(M.shape[1])) if M.shape[1] <= max_labels else ax.set_xticks([])
    ax.set_yticks(range(M.shape[0])) if M.shape[0] <= max_labels else ax.set_yticks([])
    if M.shape[1] <= max_labels:
        ax.set_xticklabels(col_order, rotation=90, fontsize=6)
    if M.shape[0] <= max_labels:
        ax.set_yticklabels(row_order, fontsize=6)
    ax.set_xlabel(f"{M.shape[1]} columns" + ("" if M.shape[1] <= max_labels else " (labels suppressed)"))
    ax.set_ylabel(f"{M.shape[0]} rows" + ("" if M.shape[0] <= max_labels else " (labels suppressed)"))

    cax = fig.add_subplot(gs[r_main, c_main + 1])
    fig.colorbar(im, cax=cax, label="z-score" if z_score else "value")

    fig.suptitle(title or "clustered heatmap", fontweight="bold")
    fig.savefig(base + ".png", dpi=200)
    fig.savefig(base + ".svg")
    plt.close(fig)

    _interactive_heatmap(M, base, title, row_annotations, col_annotations,
                         row_order, col_order)


def _axis_marker(ax, result, extra, alpha, n, horizontal, sort_by_support=None):
    """Dendrogram (hierarchical) or coloured cluster bar (k-means) beside the heatmap."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from scipy.cluster.hierarchy import dendrogram as sd

    from .core import pvpick

    ax.set_xticks([]); ax.set_yticks([])
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)

    if not result.is_kmeans:                      # hierarchical: draw the tree
        sd(_support_ordered(result, sort_by_support), ax=ax, color_threshold=0,
           above_threshold_color="#555555", no_labels=True,
           orientation="top" if horizontal else "left")
        picked = pvpick(result, alpha)
        pos = {lab: 5.0 + 10.0 * i for i, lab in enumerate(
            sd(_support_ordered(result, sort_by_support), labels=result.labels,
               no_plot=True)["ivl"])}
        lim = ax.get_ylim()[1] if horizontal else ax.get_xlim()[0]
        for e in picked:
            xs = [pos[m] for m in e["members"] if m in pos]
            if not xs:
                continue
            lo, hi = min(xs) - 4, max(xs) + 4
            if horizontal:
                ax.add_patch(Rectangle((lo, 0), hi - lo, lim, fill=False,
                                       edgecolor=AU_COLOUR, lw=1.2))
            else:
                ax.add_patch(Rectangle((lim, lo), -lim, hi - lo, fill=False,
                                       edgecolor=AU_COLOUR, lw=1.2))
        ax.set_title(f"{len(picked)} clusters AU≥{alpha}", fontsize=8) if horizontal else None
    else:                                         # k-means: blocks + centroid tree
        cmap = plt.get_cmap("tab20")
        if horizontal and result.linkage.size:
            # The centroid dendrogram, stretched so each leaf sits over its block.
            from scipy.cluster.hierarchy import dendrogram as sd
            dd = sd(result.linkage, no_plot=True)
            centres = {leaf: (lo + hi) / 2 for leaf, (lo, hi, _) in
                       zip(dd["leaves"], extra)}
            hmax = float(result.linkage[:, 2].max()) or 1.0
            for xs, ys in zip(dd["icoord"], dd["dcoord"]):
                # scipy lays leaves at 5, 15, 25...; map those onto block centres
                mapped = [centres.get(dd["leaves"][int((x - 5) // 10)], x)
                          if (x - 5) % 10 == 0 else None for x in xs]
                if any(m is None for m in mapped):
                    lo_i, hi_i = sorted([xs[0], xs[-1]])
                    mapped = [np.interp(x, [lo_i, hi_i],
                                        [mapped[0] or lo_i, mapped[-1] or hi_i])
                              for x in xs]
                ax.plot(mapped, [1.15 + 0.5 * y / hmax for y in ys],
                        color="#555555", lw=1)
        for j, (lo, hi, e) in enumerate(extra):
            colour = cmap(j % 20)
            if horizontal:
                ax.add_patch(Rectangle((lo, 0), hi - lo, 1, color=colour))
                ax.text((lo + hi) / 2, 1.15, f"AU {e['au']:.2f}", fontsize=6,
                        ha="center", color=AU_COLOUR)
            else:
                ax.add_patch(Rectangle((0, lo), 1, hi - lo, color=colour))
        if horizontal:
            ax.set_xlim(0, n); ax.set_ylim(0, 1.8 if result.linkage.size else 1.4)
            ax.set_title(f"k-means, k={len(extra)} (tree over centroids)", fontsize=8)
        else:
            ax.set_ylim(n, 0); ax.set_xlim(0, 1)


def _annotation_strip(ax, ann, horizontal):
    """Categorical annotation bars -- the batch-effect check."""
    import matplotlib.pyplot as plt
    import pandas as pd

    codes = []
    for col in ann.columns:
        cat = pd.Categorical(ann[col].astype(str))
        codes.append(cat.codes / max(1, len(cat.categories) - 1))
    A = np.array(codes)
    ax.imshow(A if horizontal else A.T, aspect="auto", cmap="tab20",
              interpolation="nearest")
    if horizontal:
        ax.set_yticks(range(len(ann.columns)))
        ax.set_yticklabels(ann.columns, fontsize=6)
        ax.set_xticks([])
    else:
        ax.set_xticks(range(len(ann.columns)))
        ax.set_xticklabels(ann.columns, rotation=90, fontsize=6)
        ax.set_yticks([])


def _interactive_heatmap(M, base, title, row_ann, col_ann, row_order, col_order):
    """Self-contained plotly heatmap in the clustered order, with annotations on hover."""
    try:
        import plotly.graph_objects as go
    except Exception:
        return

    hover = np.empty(M.shape, dtype=object)
    rtxt = ["" for _ in row_order]
    ctxt = ["" for _ in col_order]
    if row_ann is not None:
        a = row_ann.loc[row_order]
        rtxt = ["<br>".join(f"{c}: {a[c].iloc[i]}" for c in a.columns) for i in range(len(a))]
    if col_ann is not None:
        a = col_ann.loc[col_order]
        ctxt = ["<br>".join(f"{c}: {a[c].iloc[j]}" for c in a.columns) for j in range(len(a))]
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            hover[i, j] = (f"<b>{row_order[i]}</b><br>{rtxt[i]}<br>"
                           f"<b>{col_order[j]}</b><br>{ctxt[j]}<br>"
                           f"value {M.iat[i, j]:.2f}")

    lim = float(np.nanpercentile(np.abs(M.to_numpy()), 99)) or 1.0
    fig = go.Figure(go.Heatmap(z=M.to_numpy(), x=list(col_order), y=list(row_order),
                               colorscale="RdBu_r", zmid=0, zmin=-lim, zmax=lim,
                               text=hover, hovertemplate="%{text}<extra></extra>"))
    fig.update_layout(title=(title or "clustered heatmap") + " (clustered order)",
                      width=min(1600, 320 + 14 * M.shape[1]),
                      height=min(1400, 300 + 12 * M.shape[0]),
                      xaxis=dict(showticklabels=M.shape[1] <= 80, tickangle=90),
                      yaxis=dict(showticklabels=M.shape[0] <= 80, autorange="reversed"))
    fig.write_html(base + ".html")
