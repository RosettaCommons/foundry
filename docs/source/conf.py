# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'foundry'
copyright = '2025, Institute for Protein Design, University of Washington'
author = 'Institute for Protein Design, University of Washington'
release = '0.1.7'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["myst_parser",
              "sphinx_copybutton"
              ]

templates_path = ['_templates']
exclude_patterns = ["readme.md", "readmelink.md", "readme_link.rst"]



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static', '../../models/rfd3/.assets']

html_theme_options = {
    "sidebar_hide_name":False,
    #"announcement": "<em>THIS DOCUMENTATION IS CURRENTLY UNDER CONSTRUCTION</em>",
    "light_css_variables": {
        "color-brand-primary": "#F68A33", # Rosetta Teal
        "color-brand-content": "#37939B", # Rosetta Orange
        #"color-admonition-background": "#CCE8E8", # Rosetta light orange
        "font-stack": "Open Sans, sans-serif",
        "font-stack--headings": "Open Sans, sans-serif",
        "color-background-hover": "#DCE8E8ff",
        "color-announcement-background" : "#F68A33dd",
        "color-announcement-text": "#070707",
        "color-brand-visited": "#37939B",
        },
    "dark_css_variables": {
        "color-brand-primary": "#37939B", # Rosetta teal
        "color-brand-content": "#F68A33", # Rosetta orange
        #"color-admonition-background": "#20565B", # Rosetta light orange
        "font-stack": "Open Sans, sans-serif",
        "font-stack--headings": "Open Sans, sans-serif",
        "color-brand-visited": "#37939B",
        }
    }

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
    }
