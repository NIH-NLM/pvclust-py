# pvclust-py

Hierarchical clustering with AU *p*-values via multiscale bootstrap resampling — a
Python port of the R package [pvclust](https://cran.r-project.org/package=pvclust) —
and its **federated** form, in which several projects contribute to one clustering
without any of them sharing subject-level data.

> **Status: early. Not yet usable.** The multiscale curve fit (`msfit`) is ported and
> matches R exactly. The clustering, bootstrap, CLI, and federation layers are not
> written yet. See [Status](#status).

## Why this exists

The ordinary bootstrap probability (BP) — *"this cluster appeared in 87% of bootstrap
trees"* — is a **biased** measure of support, and biased in the dangerous direction: it
understates support for clusters that are real. Shimodaira showed the bias is
first-order and removable if you bootstrap at several *scales* — resample sizes
`n' = r·n` for a range of `r` — and fit a two-parameter curve through the results.

Writing `σ² = 1/r`, the normalised bootstrap z-value follows

```
z_r = -Φ⁻¹(BP_r) = v·√r + c/√r
```

so `v` and `c` are separable from BP measured at three or more scales. The two
*p*-values are two points on that fitted curve:

| | scale | formula |
|---|---|---|
| **BP** | `σ² = +1` (r = 1, the actual sample size) | `Φ(-(v + c))` |
| **AU** | `σ² = −1` | `Φ(-(v − c))` |

**AU sits at a negative variance.** No bootstrap can sample that point — it is reached
only by fitting across the scales you *can* observe and analytically continuing past
zero. That is why the multiscale bootstrap needs several sample sizes at all, and why
`msfit` is the heart of the package rather than a detail of it.

pvclust 2.2-0 also returns **SI**, the selective-inference *p*-value of Terada &
Shimodaira, built on the selection probability `d0 = Φ(-c)`. This port carries all three.

## What gets clustered, and what gets resampled

pvclust's dendrogram is over the **columns**; the bootstrap resamples the **rows**. The
asymmetry is the design, not an accident:

- objects clustered = columns (distance between columns, computed across rows)
- resampling units = rows
- AU answers: *"drawing another sample of n rows from the same population, would this
  cluster of columns reappear?"*

Clustering the other orientation means transposing, which is supported — but it is not
statistically free. The bootstrap is only meaningful if the resampling units are
exchangeable draws:

| matrix | cluster | resampling units | validity |
|---|---|---|---|
| patients × genes | genes | patients | ✅ the textbook use; patients are iid |
| patients × genes | patients (subtypes) | genes | ⚠️ anti-conservative — genes are co-expressed, not iid |
| cell clusters × genes | cell types | genes | ⚠️ same caveat |
| cell clusters × genes | genes | ~30 cell clusters | ⚠️ too few resampling units |

The package warns rather than silently returning an `AU = 0.98` that means nothing.

## Fidelity to R

Numerical agreement with R is the entire value proposition, so this is a **line-by-line
port against the R source**, not a reimplementation from the papers. `msfit` is
deterministic given `(bp, r, nboot)`, so agreement is *exact* rather than statistical:
the parity suite asserts every field to `rtol=1e-8` against fixtures generated from
pvclust 2.2-0.

Three details cost real debugging and are worth knowing before touching this code:

1. **The design matrix is `cbind(sqrt(r), 1/sqrt(r))` — `v` multiplies `√r`.**
   Swapping `v` and `c` inverts AU to `1-AU`.
2. **Scales are quantised by `floor`, and deduplicated.** `pvclust-internal.R:24` is
   `size <- unique(floor(n*r))`, then `r <- size/n`. So the effective `r` is not the
   nominal `r` (at n=63, `0.5` becomes `0.492063…`), and `unique()` can *collapse*
   scales (at n=8, ten nominal scales become eight). The `round()` at line 227 acts on
   an already-effective `r` and is a no-op — don't mistake it for the rule.
3. **The `r` sequence must be built R's way.** `np.arange(0.5, 1.45, 0.1)` yields
   `0.9999999999999999` where R yields exactly `1.0`. Since the next step floors,
   that one ulp becomes an off-by-one in resample size: `floor(63 × 0.9999999999999999)`
   is 62, not 63. Use `scales.seq_by`, never `np.arange` or `np.linspace`.

Fixtures are committed so CI never needs R. Regenerate them with Docker:

```bash
docker run --rm -v "$PWD/tests/fixtures:/fixtures" -v "$PWD/scripts:/scripts" \
  rocker/r-base:latest bash -c \
  "Rscript -e 'install.packages(\"pvclust\", repos=\"https://cloud.r-project.org\")' \
   && Rscript /scripts/make_fixtures.R"
```

The upstream R source is the specification. Fetch it (gitignored, not vendored) with
`./scripts/fetch_reference.sh`.

## Input contract

**One matrix.** Rows are the resampling units (samples), columns are the objects
clustered (genes, proteins, whatever you measured). CSV or TSV, first column the row
index. That is the whole contract, and every command speaks it:

```bash
pvclust-py cluster --project mine --matrix mine.csv --log2
```

Two optional companions, both generic:

| flag | what it does |
|---|---|
| `--metadata` | sample annotations, indexed by the matrix row ids |
| `--feature-map` | renames columns from ids to readable names (defaults to the map's first two columns) |

`--rfu/--samples/--somamers` is a **convenience adapter** for SomaLogic's positional
three-file layout (`pvclust_py.somascan`), producing exactly the same matrix. It is
the only vendor-specific code in the package; public SomaScan depositions are usually
plain CSVs and want `--matrix`.

## Federation

Several projects contribute to one clustering without sharing subject-level data.
Two things pool, by different rules, and they are independent:

| what leaves a project | rule | gives |
|---|---|---|
| sufficient statistics (4 `p×p` matrices) | **add** | the federated **tree** — exact |
| bootstrap counts (integers per cluster per scale) | **sum** | the federated **AU** |

The statistics are sums over rows, so adding them reproduces the distance matrix of
the pooled raw data *bit for bit* — the federated dendrogram **is** the centralised
dendrogram. The payload is `4p²` numbers regardless of how many samples a project
holds, and no row survives individually.

### The two-pass protocol — and why it is not optional

Counts are only addable when **every project counts the same candidate clusters**.
Running `cluster` independently at each project gives each its own tree, so the edge
sets barely overlap and pooling them is meaningless. On a two-cohort test, only 12 of
106 clusters were shared, and the pooled result found **zero** significant clusters
where each cohort alone found eight.

```bash
# PASS 1 — each project ships sufficient statistics; no rows leave
pvclust-py project-stats --project cohort_A --matrix cohort_A.csv --dist minkowski

# aggregator builds the shared catalogue (and the exact pooled tree)
pvclust-py aggregate-trees --labels cohort_A_labels.txt \
    --stats cohort_A_stats.npz --stats cohort_B_stats.npz \
    --dist minkowski --linkage ward.D2

# PASS 2 — each project counts against THAT catalogue
pvclust-py count-edges --project cohort_A --matrix cohort_A.csv \
    --catalogue federated_catalogue.csv --dist minkowski --linkage ward.D2

# aggregator pools
pvclust-py aggregate-trees --labels cohort_A_labels.txt \
    --stats cohort_A_stats.npz --stats cohort_B_stats.npz \
    --counts cohort_A_counts.csv --counts cohort_B_counts.csv \
    --dist minkowski --linkage ward.D2
```

Done correctly the same test gives 13 significant clusters, every one measured by
both cohorts, across 19 scales rather than the 10 either sees alone. `federated_edges.csv`
carries an `n_projects` column, and the aggregator **warns** if any cluster was
measured by fewer than all projects.

### Scales must be reconciled before counts are summed

Each project's `r` is relative to **its own** row count, so `r = 1.0` means 262
samples at one cohort and 94 at another. `pool_counts` re-expresses every measurement
against the pooled `n` before summing. Skipping this is not a refinement — without it
the federated AU found nothing at all on real data.

An open question worth knowing: after rescaling, no project can reach `r_pooled = 1`,
since none holds all the rows. The pooled scatter is wider and denser but sits below
1, making the extrapolation to σ² = −1 longer than a centralised run needs. Whether
that is a net gain is not settled.

### JSON exchange

`cluster` also writes `<project>.json` — a self-describing payload carrying the
clustering settings, effective scales, cluster memberships, per-scale counts and
fitted *p*-values. It holds **no subject-level data**, and `aggregate-trees --counts`
accepts it directly. Payloads produced with different distance or linkage settings
are refused rather than pooled.

## Batch effects

Clustering will happily reproduce the plate a sample was run on, and AU will call
that cluster highly supported — **because a batch effect is perfectly reproducible.**
AU says a cluster recurs, not that it means anything. Two modules address this.

`pvclust_py.diagnostics` tests the cluster assignment against every annotation:
chi-square with bias-corrected Cramér's V for categorical, Kruskal-Wallis with
eta-squared for numeric, adjusted Rand for how far a variable simply *is* the
clustering. Results are sorted by **effect size, not p-value** — p shrinks with n, so
at a few hundred samples nearly everything is "significant".

`pvclust_py.adjust` removes batch effects. **Check the confounding first**:

```bash
pvclust-py diagnose --project mine --matrix mine.csv --metadata meta.csv \
    --batch-col Batch --adjust-report
```

which reports each annotation as `balanced` (adjust freely), `partial` (consider
`--protect`), or `complete` (the design cannot separate them, and no method fixes
that). Then correct and re-check:

```bash
pvclust-py diagnose --project mine --matrix mine.csv --metadata meta.csv \
    --adjust combat --batch-col Batch --technical Batch,PlateId --annotate Group,Batch
```

`--adjust combat` is ComBat (Johnson, Li & Rabinovic 2007), the standard; needs
`pip install -e ".[combat]"`. `--adjust linear` is the same correction without
empirical-Bayes shrinkage and needs nothing.

**On `--protect` in an unsupervised analysis.** Protecting the outcome conditions the
data on the very thing the clustering is meant to discover, which weakens any claim
that the structure was found without supervision. It is standard for *differential
expression*, where nothing is being discovered. For unsupervised clustering, prefer
plain correction and report the confounding openly instead.

## Try it — a complete walkthrough

Everything below runs on an **openly licensed public dataset**, so anyone can
reproduce it. No study data is committed to this repository (`data/` is gitignored).

### 1. Install

```bash
pip install -e ".[test,combat]"
pytest -q
```

The `combat` extra pulls in `inmoose` for ComBat batch correction. Without it, two
tests skip and `--adjust combat` is unavailable (`--adjust linear` still works).

### 2. Get the data

A SomaScan 7K study of systemic lupus erythematosus, with the phenotype needed to
check that clustering recovers biology rather than batch:

**https://doi.org/10.5281/zenodo.20342569**

Download the record, unzip it, and put the three files here:

```
data/SLE/
├── abundance.csv          369 samples x 7,288 SOMAmers, RFU
├── sample-metadata.csv    Group (283 SLE / 86 healthy), Sex, Age_group, Batch,
│                          Disease_activity, SLEDAI_2K, autoantibody status
└── feature_metadata.txt   SeqId -> Target / UniProt / GeneSymbol
```

```bash
mkdir -p data/SLE
# download and unzip the Zenodo record into data/SLE, then check:
ls data/SLE
```

Every command below assumes that layout. Substitute your own matrix freely — the
input contract is one CSV with samples as rows and analytes as columns.

### 3. Check for batch effects FIRST

Do this before drawing anything. Clustering will happily reproduce the plate a sample
was run on, and AU will call that cluster highly supported — **a batch effect is
perfectly reproducible.** On this dataset, correcting afterwards changes 61% of the
protein tree and **4 of 5 k-means clusters**, so figures drawn first are figures of
the wrong thing.

The order is necessarily iterative: you need a clustering before you can test it.

**3a. Is the design even correctable?**

```bash
pvclust-py diagnose --project SLE \
    --matrix data/SLE/abundance.csv --log2 \
    --metadata data/SLE/sample-metadata.csv \
    --batch-col Batch --adjust-report
```

Each annotation comes back `balanced` (adjust freely), `partial` (consider
`--protect`), or `complete` — meaning every batch holds one level, no method can
separate them, and that is a design limitation rather than something to correct
around. On this data `Group`, `Sex` and `LN_group` are partial; the rest balanced.

**3b. Does the clustering track batch or biology?**

```bash
pvclust-py diagnose --project SLE \
    --matrix data/SLE/abundance.csv --log2 \
    --metadata data/SLE/sample-metadata.csv \
    --annotate Group,Batch,Sex,Age_group --technical Batch \
    --top-variable 60 --k 3 --n-boot 200
```

Here `Batch` scores Cramer's V **0.63** against the disease's 0.58 — batch is the
stronger signal, and the raw clustering is not safe to present.

**3c. Correct, and confirm it worked.**

```bash
pvclust-py diagnose --project SLE \
    --matrix data/SLE/abundance.csv --log2 \
    --metadata data/SLE/sample-metadata.csv \
    --annotate Group,Batch,Sex,Age_group --technical Batch \
    --adjust combat --batch-col Batch \
    --top-variable 60 --k 3 --n-boot 200
```

Batch falls to **0.07** while the disease signal largely survives. Both halves matter:
batch down, biology intact. A correction that flattens everything is not a success.

Report the before/after pair — it is the evidence that the correction was warranted
and did not manufacture the result. `--adjust linear` is the no-dependency
alternative; on this data the two agree to r = 0.9996.

> **On `--protect` in an unsupervised analysis.** Protecting the outcome conditions
> the data on the very thing the clustering is meant to discover, which weakens any
> claim the structure was found without supervision. It is standard for *differential
> expression*, where nothing is being discovered. Prefer plain correction and report
> the confounding openly.

### 4. Hierarchical clustering with AU p-values

Now draw the figures, **from the corrected data** — every command below takes the
same `--adjust combat --batch-col Batch`.

```bash
pvclust-py cluster --project SLE \
    --matrix data/SLE/abundance.csv --log2 \
    --feature-map data/SLE/feature_metadata.txt --feature-label GeneSymbol \
    --metadata data/SLE/sample-metadata.csv \
    --adjust combat --batch-col Batch \
    --top-variable 60 \
    --dist minkowski --linkage ward.D2 --n-boot 1000 --plot
```

Writes `SLE_edges.csv` (one row per cluster, si/au/bp), `SLE_counts.csv`, `SLE.json`
(the federation payload) and `SLE_dendrogram.(png|svg|html)` — the R pvclust layout:
AU in red, BP in green, red boxes on clusters at AU >= 0.95.

`--top-variable 60` keeps it quick. Cost grows with the SQUARE of the analyte count:
~7s for 100 analytes at `--n-boot 1000`, ~4 min for 1,000, hours for a full panel.

### 5. k-means, with AU from the same bootstrap

```bash
pvclust-py kmeans --project SLE --k 5 \
    --matrix data/SLE/abundance.csv --log2 \
    --feature-map data/SLE/feature_metadata.txt --feature-label GeneSymbol \
    --metadata data/SLE/sample-metadata.csv \
    --adjust combat --batch-col Batch \
    --top-variable 60 --n-boot 1000 --jaccard 0.75 --plot
```

The dendrogram here is over the k **centroids** — it shows how the clusters relate to
one another; its leaves are clusters, not analytes.

**`--jaccard 0.75` is effectively required.** A cluster counts as recovered when a
bootstrap replicate reproduces it exactly; for k-means that essentially never happens
above a few dozen objects (measured: max BP 0.83 at 20 objects, 0.20 at 50,
**0.00 at 150**). Jaccard counts it when the overlap is close enough. It changes the
estimand — the result is cluster stability, not the published AU p-value — so say
which you used.

### 6. Heatmaps, both methods

```bash
pvclust-py heatmap --project SLE --method pvclust \
    --matrix data/SLE/abundance.csv --log2 \
    --feature-map data/SLE/feature_metadata.txt --feature-label GeneSymbol \
    --metadata data/SLE/sample-metadata.csv --annotate Group,Batch,Sex \
    --adjust combat --batch-col Batch \
    --top-variable 60 --max-rows 60 \
    --dist minkowski --linkage ward.D2 --n-boot 200
```

For k-means, swap `--method pvclust` for `--method kmeans --k 5 --jaccard 0.75`.

Dendrograms on both axes with the AU boxes, z-scored, annotation strips down the side.
Keep `--annotate Batch` even after correcting: the strips should now look *mixed*
rather than blocked, which is the visual confirmation of what step 3c measured.

### 7. Federation, end to end

Split the data into two cohorts to simulate two projects, then run the **two-pass
protocol** (see [Federation](#federation) for why the second pass is not optional):

```bash
# PASS 1 -- each project ships sufficient statistics; no sample rows leave
pvclust-py project-stats --project cohort_A --matrix cohort_A.csv --dist minkowski

# the aggregator builds the shared catalogue and the exact pooled tree
pvclust-py aggregate-trees --labels cohort_A_labels.txt \
    --stats cohort_A_stats.npz --stats cohort_B_stats.npz \
    --dist minkowski --linkage ward.D2

# PASS 2 -- each project counts against THAT catalogue
pvclust-py count-edges --project cohort_A --matrix cohort_A.csv \
    --catalogue federated_catalogue.csv --dist minkowski --linkage ward.D2

# pool
pvclust-py aggregate-trees --labels cohort_A_labels.txt \
    --stats cohort_A_stats.npz --stats cohort_B_stats.npz \
    --counts cohort_A_counts.csv --counts cohort_B_counts.csv \
    --dist minkowski --linkage ward.D2

# and what the federation gives back to one project
pvclust-py apply-edges --project cohort_B --matrix cohort_B.csv \
    --federated-edges federated_edges.csv --dist minkowski --linkage ward.D2
```

For a k-medoids partition instead of a dendrogram, add `--partition kmeans --k 4` to
the catalogue step and `--method kmeans --k 4 --jaccard 0.75` to `count-edges`.

On the SLE cohorts, `apply-edges` reports that the smaller cohort recovers only
**13 of 59** federated clusters on its own, and that clusters it scored at AU 0.00
alone come back at 0.92-1.00 once pooled. That is the argument for federating, in
numbers.

### A note on privacy

Disclosure risk is governed by **n relative to p**, not by n alone. `rank(G) = min(n, p)`,
so when analytes outnumber samples the statistics expose the exact subspace those
samples occupy. Keep the analyte count **below** the sample count for the exact path
— which `--top-variable` does for you — or use counts mode, whose payload is integer
tallies. See Homer et al. (2008) for why aggregate summaries are not automatically
safe.

## Why there are R scripts in a Python package

**pvclust-py is pure Python.** It never calls R — not to install, not to run, not to
test, not to build the container. The image has no R in it, and the full suite passes
inside that image. The imports are numpy, pandas, scipy, matplotlib, plotly, typer.

R appears in two places, neither at runtime:

1. **Comments in `src/`** citing line numbers in the R source (`pvclust-internal.R:24`),
   so a reader can check the port against the original. Documentation, not code.
2. **Three scripts in `scripts/`** that regenerate the test fixtures.

Those three exist for one reason: **this is a port, and a port is only worth having if
it produces the same numbers as the original.** Proving that requires the original's
numbers to compare against. The scripts run real R pvclust 2.2-0 *inside a Docker
container* and write what it produces into `tests/fixtures/` as CSVs; the Python tests
assert against those CSVs.

**The fixtures are committed, so CI never needs R and neither do you.** You would run
those scripts only if you were changing what is tested.

They are kept rather than deleted because without them the fixtures are unexplained
magic numbers that nobody could regenerate or audit — which would undermine the one
claim this package rests on. See `scripts/README.md`.

## Status

| component | state |
|---|---|
| `msfit.py` — the si/au/bp curve fit | ✅ ported, exact parity with R (34 tests) |
| `scales.py` — scale sequence and quantisation | ✅ ported, bit-for-bit parity |
| `distance.py`, `hclust.py`, `core.py` | ⬜ not started |
| `cli.py` (Typer), `plot.py` | ⬜ not started |
| federation (`stats.py`, `aggregate.py`, `apply.py`) | ⬜ not started |

```bash
pip install -e ".[test]"
pytest
```

## Related repositories

| repo | role |
|---|---|
| `pvclust-py` | this package — the port, the CLI, the container |
| `pvclust-fed-project-nf` | the per-project (institution) Nextflow workflow |
| `pvclust-fed-aggregator-nf` | the aggregation Nextflow workflow |

The pattern follows [`oadr-cpep`](https://github.com/NIH-NLM/oadr-cpep) and its two
workflow repos. **Term mapping:** what `oadr-cpep` calls a *site* (`--site`), these
repos call a *project* (`--project`); they are the same thing — one institution holding
data that does not leave it.

## License

**GPL-3.0-or-later.** Upstream pvclust is `GPL (>= 2)`, and a line-by-line port is a
derivative work, so this package carries a compatible copyleft license. Note this
differs from `oadr-cpep`, which is MIT — the fidelity that makes this port worth having
is what requires the license.

pvclust is by **Ryota Suzuki, Yoshikazu Terada and Hidetoshi Shimodaira**.
<https://cran.r-project.org/package=pvclust>

### References

- Suzuki, R. & Shimodaira, H. (2006). *pvclust: an R package for assessing the
  uncertainty in hierarchical clustering.* Bioinformatics 22(12):1540–1542.
- Shimodaira, H. (2002). *An approximately unbiased test of phylogenetic tree
  selection.* Systematic Biology 51(3):492–508.
- Shimodaira, H. (2004). *Approximately unbiased tests of regions using
  multistep-multiscale bootstrap resampling.* Annals of Statistics 32(6):2616–2641.
- Terada, Y. & Shimodaira, H. (2017). *Selective inference for the problem of regions
  via multiscale bootstrap.* arXiv:1711.00949.
