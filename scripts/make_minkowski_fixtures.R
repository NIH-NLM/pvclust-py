library(pvclust); data(lung)
X <- lung[1:100, 1:20]
o <- "/fixtures"

# Does pvclust's "minkowski" differ from "euclidean"?  dist.pvclust calls
# dist(t(x), method) and never passes p, so R's default p=2 stands.
dm <- pvclust:::dist.pvclust(X, method = "minkowski", use.cor = "pairwise.complete.obs")
de <- pvclust:::dist.pvclust(X, method = "euclidean",  use.cor = "pairwise.complete.obs")
cat("minkowski identical to euclidean in pvclust:", isTRUE(all.equal(dm, de)), "\n")
cat("max abs difference:", max(abs(as.matrix(dm) - as.matrix(de))), "\n")
write.csv(as.matrix(dm), file.path(o, "dist_minkowski.csv"))

# p != 2 is reachable only by passing a function as method.dist
d3 <- dist(t(X), method = "minkowski", p = 3)
write.csv(as.matrix(d3), file.path(o, "dist_minkowski_p3.csv"))

# A full run in the user's preferred configuration: minkowski + ward.D2
set.seed(321)
res <- pvclust(X, method.dist = "minkowski", method.hclust = "ward.D2",
               nboot = 1000, quiet = TRUE)
ex <- do.call(rbind, lapply(res$msfit, function(m) data.frame(df = m$df, rss = m$rss)))
write.csv(cbind(res$edges, ex), file.path(o, "ward_edges_expected.csv"))
write.csv(res$count, file.path(o, "ward_count.csv"))
write.csv(data.frame(r = res$r, nboot = res$nboot), file.path(o, "ward_scales.csv"), row.names = FALSE)
writeLines(pvclust:::hc2split(res$hclust)$pattern, file.path(o, "ward_patterns.txt"))
write.csv(data.frame(height = res$hclust$height), file.path(o, "ward_height.csv"), row.names = FALSE)
cat("ward.D2 fixtures written\n")
