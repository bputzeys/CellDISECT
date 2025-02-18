import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# Project information
project = 'CellDISECT'
copyright = '2024, Arian Amani, Stathis Megas'
author = 'Arian Amani, Stathis Megas'

# The full version, including alpha/beta/rc tags
release = '0.1.1'

# Add any Sphinx extension module names here
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'nbsphinx',
    'nbsphinx_link',
    'sphinx_gallery.load_style',
    'sphinxcontrib.bibtex',
]

# Bibtex files
bibtex_bibfiles = ['references.bib']
bibtex_default_style = 'plain'

# Add any paths that contain templates here
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The theme to use for HTML and HTML Help pages
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files
html_static_path = ['_static']

# Intersphinx configuration
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'anndata': ('https://anndata.readthedocs.io/en/stable/', None),
    'scanpy': ('https://scanpy.readthedocs.io/en/stable/', None),
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autodoc settings
autodoc_member_order = 'bysource'
autodoc_docstring_signature = True

# NBSphinx settings
nbsphinx_execute = 'never' 