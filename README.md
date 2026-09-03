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
