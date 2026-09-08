# scripts/

**Nothing here runs when you use pvclust-py.** The package is pure Python — numpy,
pandas, scipy, matplotlib, typer — and never calls R. These are development tools you
run by hand, rarely, and mostly never.

## Do I need R?

**No.** Not to install pvclust-py, not to run it, not to run its tests, not to build
the container. The Docker image has no R in it, and the full test suite passes inside
that image.

R appears in exactly two places, neither of them at runtime:

1. **Comments in `src/`** citing line numbers in the R source (`pvclust-internal.R:24`),
   so a reader can check the port against the original. Documentation, not code.
2. **The three `.R` scripts here**, which regenerate test fixtures.

## Why any R at all?

pvclust-py is a port, and the whole point of a port is that it produces the same
numbers as the original. Proving that needs the original's numbers to compare against.

So: these scripts run the real R pvclust 2.2-0 **inside a Docker container**, and write
what it produces into `tests/fixtures/` as CSVs. The Python tests then assert against
those CSVs. **The fixtures are committed**, so CI — and you — never need R.

Without these scripts the fixtures would be unexplained magic numbers that nobody
could regenerate or audit. That is why they are kept rather than deleted.

## What each file is

| file | language | when you run it |
|---|---|---|
| `build_notebooks.py` | Python | after editing the notebooks — regenerates `ipynb/*.ipynb` |
| `run_notebooks.py` | Python | executes every notebook cell; used by CI |
| `fetch_reference.sh` | shell | downloads the upstream R source into `reference/` (gitignored) for reading alongside the port |
| `make_fixtures.R` | **R** | rarely — regenerates the main parity fixtures |
| `make_minkowski_fixtures.R` | **R** | rarely — fixtures for minkowski + ward.D2 |
| `make_demo_data.R` | **R** | once — extracted the shipped demo dataset |

## Running the R ones (only if you are changing what is tested)

Requires Docker. R is installed inside the container, not on your machine:

```bash
docker run --rm -v "$PWD/tests/fixtures:/fixtures" -v "$PWD/scripts:/scripts" \
  rocker/r-base:latest bash -c \
  "Rscript -e 'install.packages(\"pvclust\", repos=\"https://cloud.r-project.org\")' \
   && Rscript /scripts/make_fixtures.R"
```

Then `pytest` to confirm the Python still matches.

## Reading the upstream source

The R source is the specification for this port. It is fetched, not vendored — a
committed copy of someone else's GPL code goes stale and the version it came from
stops being obvious:

```bash
./scripts/fetch_reference.sh    # -> reference/, gitignored
```
