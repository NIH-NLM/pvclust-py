# Generate R-pvclust reference fixtures for the pvclust-py parity tests.
# Run inside a container that has pvclust installed; writes into /fixtures.
#
#   docker run --rm -v "$PWD/tests/fixtures:/fixtures" <image> Rscript /scripts/make_fixtures.R
#
# Everything written here is the ground truth the Python port is asserted against.

library(pvclust)
outdir <- "/fixtures"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

cat("pvclust version:", as.character(packageVersion("pvclust")), "\n")
writeLines(as.character(packageVersion("pvclust")), file.path(outdir, "pvclust_version.txt"))

## ---------------------------------------------------------------- 1. msfit, isolated
## Deterministic given (bp, r, nboot). These synthetic cases pin every branch of the
## function, including the degenerate short-circuit that a real run rarely reaches.
r10 <- seq(0.5, 1.4, by = 0.1)

cases <- list(
  strong        = c(.60,.66,.72,.78,.84,.88,.91,.94,.96,.975),
  weak          = c(.30,.33,.36,.39,.42,.45,.48,.51,.54,.57),
  midlow        = c(.05,.07,.09,.12,.15,.18,.22,.26,.30,.35),
  near_one      = c(.90,.93,.95,.965,.978,.985,.990,.994,.996,.998),
  all_zero      = rep(0, 10),
  all_one       = rep(1, 10),
  below_eps     = rep(0.0005, 10),
  above_1meps   = rep(0.9995, 10),
  exactly_two   = c(0, 0, 0, 0, 0, 0, 0, 0, .4, .6),          # < min.use -> degenerate
  exactly_three = c(0, 0, 0, 0, 0, 0, 0, .3, .4, .6),          # == min.use -> fits
  mixed_edges   = c(0, .001, .01, .2, .5, .8, .95, .999, 1, 1) # eps boundaries both ends
)

rows <- list()
for (nm in names(cases)) {
  for (nb in c(100, 1000)) {
    bp <- cases[[nm]]
    m  <- pvclust:::msfit(bp, r10, nb)
    rows[[length(rows) + 1]] <- data.frame(
      case = nm, nboot = nb,
      si = m$p["si"], au = m$p["au"], bp = m$p["bp"],
      se.si = m$se["si"], se.au = m$se["au"], se.bp = m$se["bp"],
      v = m$coef["v"], c = m$coef["c"],
      df = m$df, rss = m$rss, pchi = m$pchi,
      stringsAsFactors = FALSE)
  }
}
write.csv(do.call(rbind, rows), file.path(outdir, "msfit_cases_expected.csv"), row.names = FALSE)

inp <- do.call(rbind, lapply(names(cases), function(nm)
  data.frame(case = nm, scale_index = seq_along(r10), r = r10, bp = cases[[nm]],
             stringsAsFactors = FALSE)))
write.csv(inp, file.path(outdir, "msfit_cases_input.csv"), row.names = FALSE)

## ---------------------------------------------------------------- 2. a full pvclust run
## Small and low-nboot on purpose: it produces plenty of degenerate edges, which is
## exactly where the port is most likely to diverge.
data(lung)
X <- lung[1:100, 1:20]
write.csv(X, file.path(outdir, "lung_subset.csv"))

set.seed(42)
res <- pvclust(X, method.dist = "correlation", method.hclust = "average",
               nboot = 100, quiet = TRUE)

# msfit inputs: the counts, and the EFFECTIVE r (recomputed as round(n*r)/n)
write.csv(res$count, file.path(outdir, "run_count.csv"))
write.csv(data.frame(r = res$r, nboot = res$nboot), file.path(outdir, "run_scales.csv"),
          row.names = FALSE)

# msfit outputs, per edge
extra <- do.call(rbind, lapply(res$msfit, function(m)
  data.frame(df = m$df, rss = m$rss)))
write.csv(cbind(res$edges, extra), file.path(outdir, "run_edges_expected.csv"))

# the tree, for the hclust/edge-extraction parity test
write.csv(res$hclust$merge,  file.path(outdir, "run_merge.csv"),  row.names = FALSE)
write.csv(data.frame(height = res$hclust$height), file.path(outdir, "run_height.csv"), row.names = FALSE)
write.csv(data.frame(order  = res$hclust$order),  file.path(outdir, "run_order.csv"),  row.names = FALSE)
writeLines(res$hclust$labels, file.path(outdir, "run_labels.txt"))

# edge member sets, in pvclust's own order -- this is what edge_id must reproduce
writeLines(pvclust:::hc2split(res$hclust)$pattern, file.path(outdir, "run_patterns.txt"))

# the distance matrix, for the distance parity test
d <- pvclust:::dist.pvclust(X, method = "correlation", use.cor = "pairwise.complete.obs")
write.csv(as.matrix(d), file.path(outdir, "run_distance.csv"))

## ---------------------------------------------------------------- 3. all distances
for (meth in c("correlation", "abscor", "uncentered", "euclidean")) {
  dm <- pvclust:::dist.pvclust(X, method = meth, use.cor = "pairwise.complete.obs")
  write.csv(as.matrix(dm), file.path(outdir, paste0("dist_", meth, ".csv")))
}

cat("fixtures written to", outdir, "\n")
print(list.files(outdir))

## ---------------------------------------------------------------- 4. effective r
## pvclust-internal.R:24 is  size <- unique(floor(n*r))  and then  r <- size/n.
## So the scales are quantised by FLOOR (not round), and unique() can COLLAPSE two
## nominal scales into one, shrinking the scale list. msfit therefore receives the
## effective r, never the nominal one. The round() at line 227 acts on an already
## effective r, where round(n * size/n) == size, so it is a no-op in this path.
## This is the rule core.py's bootstrap loop must reproduce.
nominal <- seq(0.5, 1.4, by = 0.1)
rows <- list()
for (n in c(8, 12, 63, 100, 916)) {
  size <- unique(floor(n * nominal))
  rows[[length(rows) + 1]] <- data.frame(
    n = n, scale_index = seq_along(size), size = size, effective_r = size / n,
    n_nominal = length(nominal), n_effective = length(size),
    stringsAsFactors = FALSE)
}
write.csv(do.call(rbind, rows), file.path(outdir, "effective_r.csv"), row.names = FALSE)

## A real run at n=63, where nominal and effective r differ at every scale.
X63 <- lung[1:63, 1:12]
write.csv(X63, file.path(outdir, "lung_subset_n63.csv"))

set.seed(7)
res63 <- pvclust(X63, method.dist = "correlation", method.hclust = "average",
                 nboot = 100, quiet = TRUE)
write.csv(data.frame(effective_r = unlist(res63$r), nboot = res63$nboot),
          file.path(outdir, "run63_scales.csv"), row.names = FALSE)
write.csv(res63$count, file.path(outdir, "run63_count.csv"))
extra63 <- do.call(rbind, lapply(res63$msfit, function(m) data.frame(df = m$df, rss = m$rss)))
write.csv(cbind(res63$edges, extra63), file.path(outdir, "run63_edges_expected.csv"))

cat("n=8 collapses", length(nominal), "nominal scales to",
    length(unique(floor(8 * nominal))), "effective\n")

## ---------------------------------------------------------------- 5. seq bit-for-bit
## The nominal r sequence must be reproduced exactly: it is floored downstream, so a
## one-ulp difference becomes an off-by-one in the resample size. %.17g round-trips
## an IEEE754 double exactly.
seqs <- list(c(0.5, 1.4, 0.1), c(0.1, 1.0, 0.1), c(0.25, 2.0, 0.25), c(0.5, 3.0, 0.5))
rows <- list()
for (s in seqs) {
  v <- seq(s[1], s[2], by = s[3])
  rows[[length(rows) + 1]] <- data.frame(
    start = s[1], stop = s[2], by = s[3],
    index = seq_along(v), value = sprintf("%.17g", v), stringsAsFactors = FALSE)
}
write.csv(do.call(rbind, rows), file.path(outdir, "seq_by.csv"), row.names = FALSE)
cat("seq fixtures written\n")
