pvclust-py
==========

Hierarchical clustering with AU *p*-values via multiscale bootstrap resampling --
a Python port of the R package `pvclust <https://cran.r-project.org/package=pvclust>`_
by Suzuki, Terada & Shimodaira -- and its federated form, in which several projects
contribute to one clustering without sharing subject-level data.

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api/msfit
   api/scales
   api/distance
   api/hclust
   api/core
   api/aggregate
   api/apply
   api/diagnostics
   api/validate
   api/adjust
   api/io
   api/somascan
   api/plot
   api/datasets
   api/cli

Quick start
-----------

.. code-block:: bash

   pip install -e ".[test]"
   pvclust-py --help

.. code-block:: python

   from pvclust_py.core import pvclust, pvpick
   from pvclust_py.datasets import load_lung

   res = pvclust(load_lung(), method_dist="correlation", nboot=1000)
   for e in pvpick(res, alpha=0.95):
       print(e["au"], e["members"])

Why AU rather than BP
---------------------

The ordinary bootstrap probability (BP) is a **biased** measure of support, and
biased in the dangerous direction: it understates clusters that are real. Shimodaira
showed the bias is first-order and removable by bootstrapping at several *scales*.

Writing :math:`\\sigma^2 = 1/r` for a resample of :math:`n' = rn` rows,

.. math::

   z_r = -\\Phi^{-1}(BP_r) = v\\sqrt{r} + c/\\sqrt{r}

BP is that curve read at :math:`\\sigma^2 = +1`; **AU is it read at**
:math:`\\sigma^2 = -1` -- a negative variance, which no bootstrap can sample. AU is
reached only by fitting across the observable scales and continuing the curve past
zero. That is why the multiscale bootstrap needs several sample sizes at all.

Indices
=======

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
