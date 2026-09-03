"""
pvclust-py: hierarchical clustering with AU p-values via multiscale bootstrap,
and its federated form.

A port of the R package pvclust (Suzuki, Terada & Shimodaira). See LICENSE --
this is a derivative work of GPL code and carries the same terms.

  msfit.py   : the multiscale curve fit (si/au/bp)  -- the R-parity core
"""

__version__ = "0.1.0"

from .msfit import MsFit, msfit

__all__ = ["MsFit", "msfit", "__version__"]
