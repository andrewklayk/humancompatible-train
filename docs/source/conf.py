# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

extensions = ["myst_nb", "sphinx.ext.autodoc"]


import os


project = 'humancompatible-train'
copyright = '2026, Andrii Kliachkin, Gilles Bareillies, Jana Lepsova, Jakub Marecek'
author = 'Andrii Kliachkin, Gilles Bareillies, Jana Lepsova, Jakub Marecek'
release = '0.3.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


templates_path = ['_templates']
exclude_patterns = []

nb_toctree = False
nb_number_headings = False
nb_execution_show_tb = False
nb_execution_mode = "cache"
nb_execution_timeout = 180

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'furo'
html_theme = 'sphinx_rtd_theme'

html_static_path = ['_static']

html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "/")

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]
myst_url_schemes = ("http", "https", "mailto")

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

nb_execution_mode = "cache"

import sys 

sys.path.insert(0, os.path.abspath('./../..'))