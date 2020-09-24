import sphinx_rtd_theme
# -- Project information -----------------------------------------------------

project = 'grates'
copyright = '2020, Andreas Kvas'
author = 'Andreas Kvas'

version = '0.1'
release = '0.1.1'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'numpydoc',
    'sphinx_rtd_theme'
]

autosummary_generate = True
templates_path = ['_templates']


source_suffix = '.rst'

master_doc = 'index'

language = None

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

pygments_style = None

html_theme = 'sphinx_rtd_theme'

html_static_path = []

htmlhelp_basename = 'gratesdoc'
