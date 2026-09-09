[![Build and Push Docker Image](https://github.com/NIH-NLM/pvclust-py/actions/workflows/docker-build.yml/badge.svg)](https://github.com/NIH-NLM/pvclust-py/actions/workflows/docker-build.yml)
# pvclust-py

Hierarchical clustering with **AU *p*-values** via multiscale bootstrap resampling —
a Python port of the R package [pvclust](https://cran.r-project.org/package=pvclust)
by Suzuki, Terada & Shimodaira — plus k-means with the same *p*-values, batch
diagnostics, and a **federated** mode in which several projects contribute to one
clustering without sharing subject-level data.

Any clustering method will hand you clusters. The question this answers is **which of
them are real.**

---

## Quickstart

Runs on an openly licensed public dataset, so anyone can reproduce it.

### Install

```bash
pip install -e ".[test,combat]"
pytest -q
```

The `combat` extra pulls in `inmoose` for ComBat batch correction; without it two
tests skip and `--adjust combat` is unavailable (`--adjust linear` still works).

### Get the data

A SomaScan 7K study of systemic lupus erythematosus, with the phenotype needed to
check that clustering finds biology rather than batch:

**https://doi.org/10.5281/zenodo.20342569**

Download the record and unzip it into `data/SLE/`:

```
data/SLE/
├── abundance.csv          369 samples x 7,288 SOMAmers, RFU
├── sample-metadata.csv    Group (283 SLE / 86 healthy), Sex, Age_group, Batch,
│                          Disease_activity, SLEDAI_2K, autoantibody status
└── feature_metadata.txt   SeqId -> Target / UniProt / GeneSymbol
```

`data/` is gitignored — no study data is committed to this repository.

### 1. Check for batch effects first

Clustering will happily reproduce the plate a sample was run on, and AU will call
that cluster highly supported — **a batch effect is perfectly reproducible.** On this
dataset, correcting afterwards changes 61% of the protein tree and 4 of 5 k-means
clusters, so figures drawn first are figures of the wrong thing.

```bash
pvclust-py diagnose --project SLE \
    --matrix data/SLE/abundance.csv --log2 \
    --metadata data/SLE/sample-metadata.csv \
    --batch-col Batch --adjust-report
```

Each annotation comes back `balanced`, `partial` (use `--protect`), or `complete` —
every batch holds one level, no method can separate them, and that is a design
limitation rather than something to correct around.

Then test the clustering against the annotations, once raw and once corrected:

```bash
pvclust-py diagnose --project SLE \
    --matrix data/SLE/abundance.csv --log2 \
    --metadata data/SLE/sample-metadata.csv \
    --annotate Group,Batch,Sex,Age_group --technical Batch \
    --top-variable 60 --k 3 --n-boot 200
```

```bash
# the same, corrected
pvclust-py diagnose --project SLE \
    --matrix data/SLE/abundance.csv --log2 \
    --metadata data/SLE/sample-metadata.csv \
    --annotate Group,Batch,Sex,Age_group --technical Batch \
    --adjust combat --batch-col Batch \
    --top-variable 60 --k 3 --n-boot 200
```

Raw, `Batch` scores Cramer's V **0.63** against the disease's 0.58 — batch is the
stronger signal. After ComBat, batch drops to **0.07** and the disease signal largely
survives. Report the pair: it is the evidence that correction was warranted and did
not manufacture the result.

### 2. Hierarchical clustering with AU

Every figure below takes the same `--adjust combat --batch-col Batch`.

```bash
pvclust-py cluster --project SLE \
    --matrix data/SLE/abundance.csv --log2 \
    --feature-map data/SLE/feature_metadata.txt --feature-label GeneSymbol \
    --metadata data/SLE/sample-metadata.csv \
    --adjust combat --batch-col Batch --top-variable 60 \
    --dist minkowski --linkage ward.D2 --n-boot 1000 --plot
```

Writes `SLE_edges.csv` (one row per cluster with si/au/bp), `SLE_counts.csv`,
`SLE.json` (the federation payload) and `SLE_dendrogram.(png|svg|html)` — R's pvclust
layout: AU in red, BP in green, red boxes on clusters at AU >= 0.95.

Cost grows with the **square** of the analyte count: ~7s for 100 analytes at
`--n-boot 1000`, ~4 min for 1,000, hours for a full panel. `--top-variable` is the
knob.

### 3. k-means, same p-values

```bash
pvclust-py kmeans --project SLE --k 5 \
    --matrix data/SLE/abundance.csv --log2 \
    --feature-map data/SLE/feature_metadata.txt --feature-label GeneSymbol \
    --metadata data/SLE/sample-metadata.csv \
    --adjust combat --batch-col Batch --top-variable 60 \
    --n-boot 1000 --jaccard 0.75 --plot
```

`msfit` never asks where a cluster came from, only whether it is present or absent in
each bootstrap replicate — so k-means gets AU from identical machinery. Its dendrogram
is over the k **centroids**: it shows how the clusters relate, and its leaves are
clusters, not analytes.

**`--jaccard 0.75` is effectively required.** Exact member-set matching essentially
never recovers a k-means cluster above a few dozen objects (measured: max BP 0.83 at
20 objects, 0.20 at 50, **0.00 at 150**). Jaccard counts a cluster as recovered when
the overlap is close enough. It changes the estimand — the result is cluster
stability, not the published AU *p*-value — so say which you used.

### 4. Heatmaps

```bash
pvclust-py heatmap --project SLE --method pvclust \
    --matrix data/SLE/abundance.csv --log2 \
    --feature-map data/SLE/feature_metadata.txt --feature-label GeneSymbol \
    --metadata data/SLE/sample-metadata.csv --annotate Group,Batch,Sex \
    --adjust combat --batch-col Batch --top-variable 60 --max-rows 60 \
    --dist minkowski --linkage ward.D2 --n-boot 200
```

For k-means, swap in `--method kmeans --k 5 --jaccard 0.75`.

Dendrograms on both axes with the AU boxes, z-scored, annotation strips down the side.
Keep `--annotate Batch` after correcting: the strips should now look mixed rather than
blocked, which is the visual confirmation of what step 1 measured.

### 5. Federation

Two things pool, by different rules:

| what leaves a project | rule | gives |
|---|---|---|
| sufficient statistics (4 `p x p` matrices) | **add** | the federated **tree** — exact |
| bootstrap counts (integers) | **sum** | the federated **AU** |

The statistics are sums over rows, so adding them reproduces the distance matrix of
the pooled raw data *bit for bit*. The payload is `4p²` numbers regardless of how many
samples a project holds, and no row leaves.

**The second pass is not optional.** Counts are only addable when every project counts
the *same* candidate clusters. Running `cluster` independently at each project gives
each its own tree: on a two-cohort test only 12 of 106 clusters overlapped, and pooling
found **zero** significant clusters where each cohort alone found eight.

```bash
# PASS 1 -- each project ships sufficient statistics
pvclust-py project-stats --project cohortA --matrix cohortA.csv --log2 --dist minkowski

# the aggregator builds the shared catalogue and the exact pooled tree
pvclust-py aggregate-trees --labels cohortA_labels.txt \
    --stats cohortA_stats.npz --stats cohortB_stats.npz \
    --dist minkowski --linkage ward.D2

# PASS 2 -- each project counts against THAT catalogue
pvclust-py count-edges --project cohortA --matrix cohortA.csv --log2 \
    --catalogue federated_catalogue.csv --dist minkowski --linkage ward.D2

# pool
pvclust-py aggregate-trees --labels cohortA_labels.txt \
    --stats cohortA_stats.npz --stats cohortB_stats.npz \
    --counts cohortA_counts.csv --counts cohortB_counts.csv \
    --dist minkowski --linkage ward.D2

# what the federation gives back to one project
pvclust-py apply-edges --project cohortB --matrix cohortB.csv --log2 \
    --federated-edges federated_edges.csv --dist minkowski --linkage ward.D2
```

Done correctly, every cluster is measured by every project and `federated_edges.csv`
carries an `n_projects` column; the aggregator **warns** when any cluster was measured
by fewer than all. For a flat k-medoids partition instead of a dendrogram, add
`--partition kmeans --k 5` to the catalogue step and `--method kmeans --k 5
--jaccard 0.75` to `count-edges`.

`apply-edges` reports what joining bought: on the SLE cohorts the smaller one recovers
only **13 of 59** federated clusters alone, and clusters it scored at AU 0.00 by itself
come back at 0.92–1.00 once pooled.

---

## How it works

### AU versus BP

The ordinary bootstrap probability (BP) — *"this cluster appeared in 87% of bootstrap
trees"* — is a **biased** measure of support, biased in the dangerous direction: it
understates clusters that are real. Shimodaira showed the bias is first-order and
removable by bootstrapping at several *scales*, resample sizes `n' = r·n`.

Writing `σ² = 1/r`, the normalised bootstrap z-value follows

```
z_r = -Φ⁻¹(BP_r) = v·√r + c/√r
```

so `v` and `c` separate from BP measured at three or more scales. The two *p*-values
are two points on that fitted curve:

| | scale | formula |
|---|---|---|
| **BP** | `σ² = +1` (your actual sample size) | `Φ(-(v+c))` |
| **AU** | `σ² = −1` | `Φ(-(v−c))` |

**AU sits at a negative variance** — no bootstrap can sample it. It is reached only by
fitting across the scales you *can* observe and continuing the curve past zero. That is
why the multiscale bootstrap needs several sample sizes at all.

pvclust 2.2-0 also returns **SI**, the selective-inference *p*-value of Terada &
Shimodaira. This port carries all three.

### What gets clustered, what gets resampled

The dendrogram is over the **columns**; the bootstrap resamples the **rows**. AU
answers: *"drawing another sample of n rows from the same population, would this
cluster of columns reappear?"*

`--cluster rows` transposes internally. It is not statistically free — the bootstrap
is only meaningful if the resampling units are exchangeable draws:

| matrix | cluster | resampling units | validity |
|---|---|---|---|
| samples × analytes | analytes | samples | ✅ the sound direction |
| samples × analytes | samples | analytes | ⚠️ anti-conservative — analytes are co-expressed, not independent |

The package warns rather than returning an `AU = 0.98` that means nothing.

### Input contract

**One matrix.** Rows are the resampling units, columns are the objects clustered.
CSV or TSV, first column the index. Two optional companions: `--metadata` for sample
annotations, `--feature-map` to rename columns from ids to readable names.

`--rfu/--samples/--somamers` is a convenience adapter for SomaLogic's positional
three-file layout (`pvclust_py.somascan`) — the only vendor-specific code here.

### Privacy

Disclosure risk is governed by **n relative to p**, not n alone. `rank(G) = min(n, p)`,
so when analytes outnumber samples the statistics expose the exact subspace those
samples occupy; at n=1 the Gram is rank-1 and returns the row exactly. Keep the analyte
count **below** the sample count for the exact path — which `--top-variable` does — or
use counts mode, whose payload is integer tallies. See Homer et al. (2008) for why
aggregate summaries are not automatically safe.

### Fidelity to R

Numerical agreement with R is the value proposition, so this is a **line-by-line port
against the R source**, not a reimplementation from the papers. `msfit` is
deterministic given `(bp, r, nboot)`, so agreement is *exact*: the parity suite asserts
every field to `rtol=1e-8` against fixtures from pvclust 2.2-0.

Three details cost real debugging and are worth knowing before touching this code:

1. **The design matrix is `cbind(sqrt(r), 1/sqrt(r))`** — `v` multiplies `√r`. Swapping
   `v` and `c` inverts AU to `1-AU`.
2. **Scales are quantised by `floor`, and deduplicated.** `pvclust-internal.R:24` is
   `size <- unique(floor(n*r))`, then `r <- size/n`. So the effective `r` is not the
   nominal `r` (at n=63, `0.5` becomes `0.492063…`), and `unique()` can *collapse*
   scales. The `round()` at line 227 acts on an already-effective `r` and is a no-op.
3. **The `r` sequence must be built R's way.** `np.arange(0.5, 1.45, 0.1)` yields
   `0.9999999999999999` where R yields exactly `1.0`; since the next step floors, that
   one ulp becomes an off-by-one in resample size. Use `scales.seq_by`.

Fixtures are committed, so CI never needs R. Regenerate with Docker:

```bash
docker run --rm -v "$PWD/tests/fixtures:/fixtures" -v "$PWD/scripts:/scripts" \
  rocker/r-base:latest bash -c \
  "Rscript -e 'install.packages(\"pvclust\", repos=\"https://cloud.r-project.org\")' \
   && Rscript /scripts/make_fixtures.R"
```

### Why there are R scripts in a Python package

**pvclust-py is pure Python** — it never calls R, and the container has no R in it.
The three `.R` files in `scripts/` regenerate the parity fixtures by running real
pvclust 2.2-0 inside Docker. They are kept because without them the fixtures would be
unexplained numbers nobody could regenerate or audit. See `scripts/README.md`.

---

## Status

| component | state |
|---|---|
| `msfit`, `scales` | ✅ exact R parity, asserted to 1e-8 / bit-for-bit |
| `distance`, `hclust` | ✅ R parity; federated pooling proven exact |
| `core` — bootstrap, pvclust, k-means, pvpick | ✅ parity within Monte-Carlo error |
| `aggregate`, `apply` — federation | ✅ working; `apply` has no unit tests yet |
| `adjust` — ComBat and linear correction | ✅ tested; no CLI-level test |
| `diagnostics` — batch association | ✅ tested |
| `plot` — dendrograms, heatmaps | ✅ png / svg / interactive html |
| `cli` — 10 commands | ✅ |
| `pvclust-fed-project-nf`, `pvclust-fed-aggregator-nf` | ⬜ not started |

`pytest -q` → **170 passed, 2 skipped** (the skips need the `combat` extra).

**Known open question.** After rescaling counts to the pooled n, no project can reach
`r_pooled = 1` — none holds all the rows — so the extrapolation to σ² = −1 is longer
than a centralised run needs. Whether the wider, denser scale coverage compensates is
not settled.

## Related repositories

| repo | role |
|---|---|
| `pvclust-py` | this package — the port, the CLI, the container |
| `pvclust-fed-project-nf` | the per-project Nextflow workflow |
| `pvclust-fed-aggregator-nf` | the aggregation Nextflow workflow |

The pattern follows [`oadr-cpep`](https://github.com/NIH-NLM/oadr-cpep). **Term
mapping:** what `oadr-cpep` calls a *site* (`--site`), these repos call a *project*
(`--project`).

## License

**GPL-3.0-or-later.** Upstream pvclust is `GPL (>= 2)` and a line-by-line port is a
derivative work, so this package carries a compatible copyleft licence — a deliberate
divergence from `oadr-cpep`, which is MIT.

pvclust is by **Ryota Suzuki, Yoshikazu Terada and Hidetoshi Shimodaira**.

### References

- Suzuki & Shimodaira (2006). *pvclust: an R package for assessing the uncertainty in
  hierarchical clustering.* Bioinformatics 22(12):1540–1542.
- Shimodaira (2002). *An approximately unbiased test of phylogenetic tree selection.*
  Systematic Biology 51(3):492–508.
- Shimodaira (2004). *Approximately unbiased tests of regions using multistep-multiscale
  bootstrap resampling.* Annals of Statistics 32(6):2616–2641.
- Terada & Shimodaira (2017). *Selective inference for the problem of regions via
  multiscale bootstrap.* arXiv:1711.00949.
- Johnson, Li & Rabinovic (2007). *Adjusting batch effects in microarray expression data
  using empirical Bayes methods.* Biostatistics 8(1):118–127.
- Gower (1966). *Some distance properties of latent root and vector methods used in
  multivariate analysis.* Biometrika 53:325–338.
- Homer et al. (2008). *Resolving individuals contributing trace amounts of DNA to
  highly complex mixtures using high-density SNP genotyping microarrays.* PLoS Genetics
  4(8):e1000167.
