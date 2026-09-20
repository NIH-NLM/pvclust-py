# archive

Work kept for reference, not part of the current pipeline. Nothing here is run by
`scripts/run_notebooks.py` or by continuous integration.

## `ipynb/` — the batch-split cohorts

Twelve numbered notebooks that treated **batch A as cohort A and batch B as cohort B**,
262 samples against 94.

**Why it was built.** A random split makes federation look easier than it is: the
cohorts are exchangeable draws from one population, so pooling cannot really go wrong.
Splitting on batch gave two genuinely unequal cohorts, 17% healthy volunteers against
44%, with the technical and biological differences entangled. That is what a real
multi-site study faces.

**Why it was set aside.** Two reasons, and both are worth remembering.

*ComBat became a no-op.* With one batch per cohort there is nothing to correct against
locally, so the correction step could only be demonstrated in a separate lesson rather
than in the pipeline itself.

*Feature selection was wrong, and the batch split made it obvious.* Each cohort ranked
its own most-variable proteins and the intersection was taken afterwards. That lets a
cohort's own view of what varies decide the vocabulary, which presumes the cohorts
already agree about what matters -- the thing the analysis is supposed to discover. On
this data it discarded 27 of 80 features, and which 27 depended on the cohorts rather
than on biology.

**What it nonetheless established.** Four patient endotypes found independently in each
cohort corresponded one-to-one across them, and the correspondence reproduced under
resampling down to about 1% of the protein panel. The interferon signature -- STAT1,
ISG15, LAP3 -- separated SLE from healthy without being asked to.

**One negative result worth keeping.** AU *p*-values on the cross-cohort correspondence
were all degenerate: eight profiles in 7288 dimensions are too well separated for the
bootstrap to disturb, so every fit had zero usable scales and returned 1.000. That is
not a defect in pvclust; it is a statement about eight objects. A block bootstrap would
not fix it, which was checked rather than assumed.
