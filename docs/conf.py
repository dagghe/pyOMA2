# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

from pydantic import BaseModel
from sphinx.ext.autodoc import AttributeDocumenter, ClassDocumenter

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../"))
sys.path.insert(0, os.path.abspath("../src"))
sys.path.insert(0, os.path.abspath("../src/pyoma2"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "pyoma2"
copyright = "2024, Dag Pasca"
author = "Dag Pasca"
release = "1.1.1"


### Code to exclude the docs from pydantic - start ###
class PydanticModelDocumenter(ClassDocumenter):
    objtype = "pydantic_model"
    directivetype = "class"

    @classmethod
    def can_document_member(cls, member, membername, isattr, parent):
        return isinstance(member, type) and issubclass(member, BaseModel)


#    def add_directive_header(self, sig):
#        super().add_directive_header(sig)
#        self.add_line('   :show-inheritance:', self.get_sourcename())


class PydanticAttributeDocumenter(AttributeDocumenter):
    objtype = "pydantic_attribute"
    directivetype = "attribute"

    @classmethod
    def can_document_member(cls, member, membername, isattr, parent):
        return (
            isattr
            and isinstance(parent, type)
            and issubclass(parent, BaseModel)
            and membername in parent.__fields__
        )


def setup(app):
    app.add_autodocumenter(PydanticModelDocumenter)
    app.add_autodocumenter(PydanticAttributeDocumenter)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }


### Code to exclude the docs from pydantic - end ###

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "nbsphinx",
    "sphinx_rtd_theme",
    #    'sphinxcontrib.bibtex',
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# -- Options for EPUB output
epub_show_urls = "footnote"

# Napoleon settings
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True

# Autodoc configuration
autodoc_default_options = {
    "special-members": "__init__",
    "exclude-members": "model_computed_fields, model_config, model_fields",
}
