import os
import sys

# Point at the package's parent, so modules are documented as `pvclust_py.msfit`
# rather than `src.pvclust_py.msfit`.
sys.path.insert(0, os.path.abspath("../../src"))

project = "pvclust-py"
copyright = "2026, NIH-NLM"
author = "NIH-NLM"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

# Heavy dependencies are mocked so the docs build without installing them.
autodoc_mock_imports = [
    "numpy", "pandas", "scipy", "sklearn", "matplotlib", "plotly", "typer",
]

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": True,
    "show-inheritance": True,
    "exclude-members": "__weakref__",
}

# Keep signatures readable rather than spelling out every default.
autodoc_typehints = "description"
autodoc_preserve_defaults = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
