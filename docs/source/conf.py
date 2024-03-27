# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

# Adding library path to the compilation for the autodoc documentation
import sys
import os
import shutil
from pathlib import Path


def all_but_ipynb(dir, contents):
    result = []
    for c in contents:
        if os.path.isfile(os.path.join(dir, c)) and (not c.endswith(".py")):
            result += [c]
    return result


_lib_path = Path(__file__).parents[2]/'src'
_doc_folder = Path(__file__).parents[2]/'docs/source'
_examples_path = Path(__file__).parents[2]/'examples'
sys.path.append(_lib_path.as_posix())
sys.path.append(_examples_path.as_posix())


# -- Project information -----------------------------------------------------

project = 'lime'
copyright = '2021, Vital-Fernandez'
author = 'Vital-Fernandez'

# The full version, including alpha/beta/rc tags
release = '0.9.99.1'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.imgmath',
    'matplotlib.sphinxext.plot_directive',
    'sphinx.ext.imgmath',
    'nbsphinx']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates', '_build']

autodoc_member_order = 'bysource'
autodoc_default_options = {"imported-members": True}

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

imgmath_latex_preamble = r'\usepackage[active]{preview}' # + other custom stuff for inline math, such as non-default math fonts etc.
imgmath_use_preview = True

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

shutil.rmtree(_doc_folder/'images', ignore_errors=True)
shutil.rmtree(_doc_folder/'inputs', ignore_errors=True)
shutil.rmtree(_doc_folder/'outputs', ignore_errors=True)
shutil.rmtree(_doc_folder/'sample_data', ignore_errors=True)
shutil.rmtree(_doc_folder/'tutorials', ignore_errors=True)
shutil.copytree(_examples_path, _doc_folder, dirs_exist_ok=True)
