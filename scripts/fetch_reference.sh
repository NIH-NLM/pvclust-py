#!/usr/bin/env bash
# Fetch the upstream R pvclust source into reference/ (gitignored).
#
# pvclust-py is a line-by-line port, so the R source is the specification. It is
# fetched rather than vendored: a committed copy of someone else's GPL source goes
# stale silently, and the version it was cut from stops being obvious.
#
#   ./scripts/fetch_reference.sh
#
# Then read reference/pvclust.R and reference/pvclust-internal.R alongside the port.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p reference

for f in R/pvclust.R R/pvclust-internal.R DESCRIPTION; do
  out="reference/$(basename "$f")"
  gh api "repos/cran/pvclust/contents/$f" --jq '.content' | base64 -d > "$out"
  echo "fetched $out"
done

grep -E '^(Package|Version|License):' reference/DESCRIPTION
