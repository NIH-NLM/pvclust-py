library(pvclust); data(lung)
cat("full lung:", nrow(lung), "genes x", ncol(lung), "samples\n")
# Correct orientation for the demos: rows = samples (resampling units),
# columns = genes (the objects clustered). Keep the most variable genes.
v <- apply(lung, 1, var, na.rm = TRUE)
keep <- head(order(v, decreasing = TRUE), 150)
demo <- t(lung[keep, ])
cat("demo:", nrow(demo), "samples x", ncol(demo), "genes\n")
write.csv(demo, "/fixtures/lung_expression.csv")
