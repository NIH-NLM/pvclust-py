"""
msfit -- the multiscale bootstrap curve fit.

A faithful port of ``pvclust:::msfit`` (pvclust 2.2-0, ``R/pvclust.R`` lines 350-407)
by Ryota Suzuki, Yoshikazu Terada and Hidetoshi Shimodaira. See LICENSE: this module
is a derivative work of GPL-2 | GPL-3 code and is distributed under the same terms.

WHAT THIS COMPUTES
------------------
The ordinary bootstrap probability BP -- "this cluster appeared in 87% of bootstrap
trees" -- is a *biased* measure of support, and the bias is driven by the curvature of
the boundary of the region in data space where the cluster would be inferred.
Shimodaira (2002, 2004) showed the bias is first-order and removable if you bootstrap
at several *scales*, i.e. with resample sizes n' = r*n for a range of r.

Writing sigma^2 = 1/r for the scale, the normalised bootstrap z-value follows

    z_r = -Phi^-1(BP_r) = v * sqrt(r) + c / sqrt(r)

so v and c are separable from BP measured at >= 3 scales. The two p-values are two
points on that fitted curve:

    BP = Phi(-(v + c))   at sigma^2 = +1   (r = 1: the actual sample size)
    AU = Phi(-(v - c))   at sigma^2 = -1

AU sits at a *negative variance*. No bootstrap can sample that point -- it is reached
only by fitting across the observable scales and analytically continuing past zero.
That is why the multiscale bootstrap needs several sample sizes at all.

pvclust 2.2-0 additionally returns SI, the selective-inference p-value of Terada &
Shimodaira, built on the selection probability d0 = Phi(-c).

NOTE ON FIDELITY
----------------
Numerical agreement with R is the entire value proposition of this port, so this
function is written against the R source line by line rather than reconstructed from
the papers. Two details are easy to get wrong and are called out here because earlier
descriptions of pvclust (including our own first draft) got them backwards:

  * the design matrix is ``cbind(sqrt(r), 1/sqrt(r))`` -- **v multiplies sqrt(r)**
    and c multiplies 1/sqrt(r), not the reverse. Swapping them inverts AU to 1-AU.
  * degenerate scales are **excluded, not clamped**: a scale is usable only when
    eps < BP_r < 1-eps, and fewer than three usable scales short-circuits the fit.

Do not "simplify" either without re-running tests/test_msfit_parity.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy.stats import chi2, norm

# --- constants, named exactly as in the R source -----------------------------
MIN_USE = 3        # R: min.use -- minimum usable scales for a fit (must be >= 2)
EPS = 0.001        # R: eps     -- BP must lie strictly inside (eps, 1-eps)


@dataclass
class MsFit:
    """The result of one edge's multiscale curve fit.

    Mirrors the fields of R's ``msfit`` S3 object. ``si``/``au``/``bp`` are the
    p-values, ``se_*`` their standard errors, ``v``/``c`` the fitted coefficients,
    and ``rss``/``df``/``pchi`` the goodness-of-fit diagnostic: a small ``pchi``
    means the two-parameter curve did not describe the observed BP values, and the
    AU on that edge should not be trusted.
    """

    si: float = 0.0
    au: float = 0.0
    bp: float = 0.0
    se_si: float = 0.0
    se_au: float = 0.0
    se_bp: float = 0.0
    v: float = 0.0
    c: float = 0.0
    df: int = 0
    rss: float = 0.0
    pchi: float = 0.0
    # diagnostics for plotting the fitted curve (R stores use/r/zz on the object)
    use: Optional[np.ndarray] = field(default=None, repr=False)
    r: Optional[np.ndarray] = field(default=None, repr=False)
    zz: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def fitted(self) -> bool:
        """False when the edge was degenerate and short-circuited to 0/1."""
        return self.use is not None


def msfit(bp: Sequence[float], r: Sequence[float], nboot) -> MsFit:
    """Fit the multiscale curve for ONE edge and return its si/au/bp p-values.

    Args:
        bp: bootstrap probability at each scale (count / nboot), same length as ``r``.
        r: relative sample sizes, e.g. ``np.arange(0.5, 1.5, 0.1)``.
        nboot: replicates per scale -- a scalar, or one value per scale.

    Returns:
        MsFit. Degenerate edges (fewer than ``MIN_USE`` usable scales) come back with
        every p-value set to 0.0 or 1.0 and ``fitted`` False, exactly as R does.

    This function touches only counts, never the data -- which is what makes
    counts-mode federation possible: sum the per-project count matrices and fit once.
    """
    bp = np.asarray(bp, dtype=float)
    r = np.asarray(r, dtype=float)
    if bp.shape != r.shape:
        raise ValueError("bp and r should have the same length")

    # R: nboot <- rep(nboot, length=length(bp)) -- recycle a scalar across scales
    nboot = np.resize(np.asarray(nboot, dtype=float), bp.shape)

    use = (bp > EPS) & (bp < 1.0 - EPS)

    # Degenerate: too few usable scales to fit two parameters. R decides the
    # direction from the mean of the FULL bp vector, before subsetting, and leaves
    # se / coef / df / rss / pchi at zero.
    if int(use.sum()) < MIN_USE:
        p = 0.0 if bp.mean() < 0.5 else 1.0
        return MsFit(si=p, au=p, bp=p)

    bp_u, r_u, nboot_u = bp[use], r[use], nboot[use]

    zz = -norm.ppf(bp_u)
    # Binomial variance of BP carried through the probit transform (delta method).
    vv = ((1.0 - bp_u) * bp_u) / (norm.pdf(zz) ** 2 * nboot_u)

    # v multiplies sqrt(r); c multiplies 1/sqrt(r). No intercept.
    X = np.column_stack([np.sqrt(r_u), 1.0 / np.sqrt(r_u)])

    # R: lsfit(X, zz, 1/vv, intercept=FALSE) minimises sum(resid^2 / vv).
    w = 1.0 / vv
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(sw[:, None] * X, sw * zz, rcond=None)
    v, c = float(coef[0]), float(coef[1])

    h_au = np.array([1.0, -1.0])
    h_bp = np.array([1.0, 1.0])
    z_au = v - c
    z_bp = v + c
    p_au = float(norm.cdf(-z_au))
    p_bp = float(norm.cdf(-z_bp))

    # Selective inference (Terada & Shimodaira): d0 is the selection probability.
    d0 = float(norm.cdf(-c))
    p_iau = float(norm.cdf(z_au))          # == 1 - p_au
    p_si = 1.0 - p_iau / d0
    p_si = 0.0 if p_si < 0.0 else (1.0 if p_si > 1.0 else p_si)

    # R: V <- solve(crossprod(X, X/vv))  ==  inv(X' W X) with W = diag(1/vv)
    V = np.linalg.inv(X.T @ (X * w[:, None]))
    vz_au = float(h_au @ V @ h_au)
    vz_bp = float(h_bp @ V @ h_bp)

    if 0.0 < p_si < 1.0:
        d1 = norm.pdf(z_au) / d0
        d2 = p_iau * norm.pdf(c) / d0 ** 2
        h_si = np.array([d1, -d1 + d2])
        v_si = float(h_si @ V @ h_si)
    else:
        v_si = 0.0

    resid = zz - X @ coef
    rss = float(np.sum(resid ** 2 / vv))
    df = int(use.sum()) - 2
    pchi = float(chi2.sf(rss, df)) if df > 0 else 1.0

    return MsFit(
        si=p_si, au=p_au, bp=p_bp,
        se_si=float(np.sqrt(v_si)),
        se_au=float(norm.pdf(z_au) * np.sqrt(vz_au)),
        se_bp=float(norm.pdf(z_bp) * np.sqrt(vz_bp)),
        v=v, c=c, df=df, rss=rss, pchi=pchi,
        use=use, r=r_u, zz=zz,
    )
