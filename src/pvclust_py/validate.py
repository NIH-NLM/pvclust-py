"""
Do the clusters recover things we already know to be related?

AU says a cluster is reproducible. It cannot say the cluster is *right*. The check
here is external: take groups of objects that must belong together for reasons outside
the data, and ask whether the clustering puts them together more often than chance.

TWO SOURCES OF GROUND TRUTH
---------------------------
**Replicate reagents.** Assay panels routinely measure one protein with several
reagents -- in a SomaScan 7K panel, 780 proteins are covered by 2 to 11 SOMAmers.

BUT SHARING A GENE SYMBOL IS NOT ENOUGH. Reagents against one UniProt accession may
target genuinely different molecules, and then they SHOULD NOT cluster:

    FN1     P02751   "Fibronectin", "Fragment 2", "Fragment 3", "Fragment 4"
    PILRA   Q9UKJ1   plain, "isoform FDF03-M14", "isoform FDF03-deltaTM"

Fibronectin is cleaved in circulation and its fragments vary independently; PILRA's
deltaTM is the soluble form, regulated separately from the membrane form. Treating
those as failed replicates would be forcing a clustering the biology does not support.

So :func:`reagent_groups` splits on the descriptive name as well as the symbol.
Reagents whose ``TargetFullName`` matches exactly are TRUE REPLICATES and must
co-cluster; reagents differing in name are DISTINCT FORMS and are reported
separately, as information rather than as a failure.

**Any external grouping** -- protein complexes, pathways, interaction partners,
subcellular compartments. Pass them the same way and the same statistics apply.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

#: Depositors mark a repeated reagent by appending its id to the gene symbol.
REAGENT_SUFFIX = re.compile(r"^(?P<symbol>.+?)[_ ]\(?(?P<reagent>seq[.\d]+|\d+-\d+)\)?$")


def reagent_groups(labels: Sequence[str], min_size: int = 2,
                   annotation: Optional[pd.DataFrame] = None,
                   label_col: str = "label", name_col: str = "TargetFullName",
                   replicates_only: bool = True) -> Dict[str, List[str]]:
    """Group labels naming the same protein measured by different reagents.

    Recognises both conventions seen in the wild: the depositors' ``SYMBOL_seq.N.N``
    and this package's own collision suffix ``SYMBOL (seqid)``.

    Args:
        annotation: reagent annotation carrying a descriptive name. Without it, all
            reagents sharing a symbol are grouped -- which mixes true replicates with
            fragments and isoforms that have no reason to agree.
        name_col: the descriptive-name column, e.g. ``TargetFullName``.
        replicates_only: keep only groups whose members share that name exactly, i.e.
            true replicates. Set False to get the distinct forms too.

    Returns:
        ``{protein: [labels]}`` for groups of at least ``min_size``.
    """
    groups: Dict[str, List[str]] = {}
    for lab in labels:
        m = REAGENT_SUFFIX.match(str(lab))
        symbol = m.group("symbol").strip() if m else str(lab)
        groups.setdefault(symbol, []).append(str(lab))

    if annotation is not None and name_col in annotation.columns:
        by_label = dict(zip(annotation[label_col].astype(str),
                            annotation[name_col].astype(str))) \
            if label_col in annotation.columns \
            else dict(zip(annotation.index.astype(str), annotation[name_col].astype(str)))
        split: Dict[str, List[str]] = {}
        for symbol, members in groups.items():
            byname: Dict[str, List[str]] = {}
            for lab in members:
                byname.setdefault(by_label.get(lab, "?"), []).append(lab)
            if replicates_only:
                for name, labs in byname.items():
                    if len(labs) >= min_size:
                        split[symbol if len(byname) == 1 else f"{symbol} [{name}]"] = labs
            else:
                split[symbol] = members
        groups = split

    return {k: v for k, v in groups.items() if len(v) >= min_size}


def cocluster(result, groups: Dict[str, List[str]], *, alpha: float = 0.95,
              n_permutations: int = 1000, seed: int = 42) -> pd.DataFrame:
    """For each group, the smallest cluster containing all of it, and its support.

    A group that lands inside one tight, well-supported cluster is recovered. One
    scattered across the tree is not, and the row says how large a cluster you would
    need to gather it.

    Args:
        result: a pvclust or kmeans_pv result.
        groups: from :func:`reagent_groups`, or any external grouping.
        n_permutations: null comparison -- how tight would a random group of the same
            size be? Reported as ``p_value``.

    Returns:
        One row per group: size, the smallest enclosing cluster, its AU, and how that
        tightness compares with random groups of equal size.
    """
    labels = set(result.labels)
    edges = [e for e in result.edges]
    rng = np.random.default_rng(seed)
    n_obj = len(result.labels)

    def enclosing(members):
        """Smallest cluster containing every member, and its AU."""
        best, best_au = None, np.nan
        for e in edges:
            if members <= set(e["members"]):
                if best is None or e["n_members"] < best:
                    best, best_au = e["n_members"], e["au"]
        return (best if best is not None else n_obj), best_au

    # null: smallest enclosing cluster for random groups of each size
    null: Dict[int, np.ndarray] = {}
    sizes = sorted({len(v) for v in groups.values()})
    all_labels = list(result.labels)
    for k in sizes:
        draws = [enclosing(set(rng.choice(all_labels, k, replace=False)))[0]
                 for _ in range(n_permutations)]
        null[k] = np.asarray(draws, dtype=float)

    rows = []
    for protein, members in sorted(groups.items()):
        present = {m for m in members if m in labels}
        if len(present) < 2:
            continue
        size, au = enclosing(present)
        p = float((null[len(present)] <= size).mean()) if len(present) in null else np.nan
        rows.append({
            "group": protein,
            "n_reagents": len(present),
            "enclosing_cluster": size,
            "enclosing_au": au,
            "tight": size == len(present),      # nothing else had to be included
            "p_value": p,
            "members": ";".join(sorted(present)),
        })
    out = pd.DataFrame(rows)
    return out.sort_values(["tight", "enclosing_cluster"],
                           ascending=[False, True], ignore_index=True) if len(out) else out


def summary(table: pd.DataFrame, alpha: float = 0.05) -> str:
    """A verdict on whether the clustering recovered the known groupings."""
    if table.empty:
        return "No groups with two or more members were found."
    n = len(table)
    tight = int(table["tight"].sum())
    sig = int((table["p_value"] < alpha).sum())
    lines = [
        f"{n} groups tested.",
        f"  {tight} ({100*tight/n:.0f}%) sit in a cluster of exactly their own members.",
        f"  {sig} ({100*sig/n:.0f}%) are tighter than random groups of the same size "
        f"(p < {alpha}).",
    ]
    loose = table[~table["tight"]].nlargest(3, "enclosing_cluster")
    if len(loose):
        lines.append("  Least well recovered:")
        for _, r in loose.iterrows():
            lines.append(f"    {r['group']}: {r['n_reagents']} reagents need a "
                         f"{int(r['enclosing_cluster'])}-member cluster to gather them")
    if tight / n < 0.5:
        lines.append("  Fewer than half sit in a cluster of their own members. Check "
                     "whether these are TRUE replicates before treating it as a "
                     "failure: reagents sharing a gene symbol may target different "
                     "fragments or isoforms, which have no reason to agree. Pass an "
                     "annotation to reagent_groups() to separate the two.")
    return "\n".join(lines)
