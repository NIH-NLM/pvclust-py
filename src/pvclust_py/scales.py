"""
Bootstrap scales: the relative sample sizes r, and their quantisation.

Small module, outsized importance. Two rules here decide whether the whole port
agrees with R, and both are easy to get wrong in ways that show up at some n and
not others:

1. THE SEQUENCE MUST BE BUILT R's WAY.  ``np.arange(0.5, 1.45, 0.1)`` produces
   0.9999999999999999 where R's ``seq(.5, 1.4, by=.1)`` produces exactly 1.0.
   Since the next step takes a FLOOR, that one-ulp difference becomes an
   off-by-one in the resample size -- floor(63 * 0.9999999999999999) is 62, not
   63 -- which changes the effective r, which changes msfit's input. Use
   :func:`seq_by`, never ``np.arange`` or ``np.linspace``.

2. THE SCALES ARE QUANTISED BY FLOOR, AND DEDUPLICATED.  pvclust-internal.R:24 is
   ``size <- unique(floor(n*r))`` followed by ``r <- size/n``. Two consequences:
   the effective r is not the nominal r (at n=63, 0.5 becomes 0.492063...), and
   unique() can COLLAPSE scales (at n=8, ten nominal scales become eight).
   The ``round()`` at pvclust-internal.R:227 acts on an already-effective r, where
   round(n * size/n) == size, so it is a no-op -- do not mistake it for the rule.

msfit always receives the EFFECTIVE r.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

#: pvclust's default nominal scales, ``seq(.5, 1.4, by=.1)``.
DEFAULT_R = (0.5, 1.4, 0.1)


def seq_by(start: float, stop: float, by: float) -> np.ndarray:
    """R's ``seq(from, to, by)``, reproduced bit-for-bit.

    R computes ``from + (0:n)*by`` and then clamps to ``to`` with pmin/pmax, which
    is why its last element is exactly ``to`` rather than drifting past it. The
    ``1e-10`` fudge in the length is R's own, guarding against a length that is
    one short when ``(to-from)/by`` lands just under an integer.
    """
    n = (stop - start) / by
    length = int(np.floor(n + 1e-10)) + 1
    x = start + np.arange(length) * by
    return np.minimum(x, stop) if by > 0 else np.maximum(x, stop)


def default_scales() -> np.ndarray:
    """pvclust's default nominal r: 0.5, 0.6, ... 1.4."""
    return seq_by(*DEFAULT_R)


def effective_scales(n: int, r: Sequence[float] | None = None):
    """Quantise nominal scales to what the bootstrap can actually draw.

    Args:
        n: number of rows (resampling units) available.
        r: nominal relative sample sizes; defaults to :func:`default_scales`.

    Returns:
        ``(size, effective_r)`` -- the integer resample sizes, deduplicated in
        first-occurrence order exactly as R's ``unique()`` does, and ``size / n``.

    Raises:
        ValueError: if every scale floors to zero (R: "invalid scale parameter(r)").

    A caller that passes ten nominal scales may get back fewer; msfit needs at
    least three usable scales, so heavy collapse at small n is worth warning about
    upstream.
    """
    r = default_scales() if r is None else np.asarray(r, dtype=float)
    sizes = np.floor(n * r).astype(int)

    # R's unique() keeps first occurrence; np.unique sorts, which would silently
    # reorder for a non-monotonic r.
    _, first_idx = np.unique(sizes, return_index=True)
    size = sizes[np.sort(first_idx)]

    size = size[size > 0]
    if size.size == 0:
        raise ValueError(
            f"invalid scale parameter(r): every scale floors to 0 rows at n={n}")

    return size, size / n
