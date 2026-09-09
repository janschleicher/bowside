# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import re
from functools import partial

from docutils import nodes
from sphinx.application import Sphinx

import bowside

project = "bowside"
copyright = "%Y, Jan T. Schleicher"
author = "Jan T. Schleicher"
release = ".".join(bowside.__version__.split(".")[:3])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_copybutton",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "scanpydoc.definition_list_typed_field",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = []

source_suffix = {".rst": "restructuredtext", ".txt": "markdown", ".md": "markdown"}

suppress_warnings = ["myst.header"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_theme_options = dict(
    use_repository_button=True,
    repository_url="https://github.com/janschleicher/bowside",
    repository_branch="main",
)
html_title = "bowside"

# Automatic generation of API documentation
autosummary_generate = True
autodoc_default_options = {
    "member-order": "alphabetical",
    "show-inheritance": True,
}
autodoc_typehints = "both"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True
napoleon_use_param = True

intersphinx_mapping = dict(
    numpy=("https://numpy.org/doc/stable/", None),
    python=("https://docs.python.org/3", None),
    scipy=("https://docs.scipy.org/doc/scipy/", None),
    torch=("https://docs.pytorch.org/docs/stable/", None),
)


def preprocess_includes(content, base_path) -> str:
    """Preprocess MyST {include} directives by replacing their content."""

    def replace_include(match):
        # Extract the file path and optional start/end markers
        include_path = match.group(1).strip()
        start_after = match.group(2)
        end_before = match.group(3)

        # Resolve the full path of the included file
        full_path = os.path.abspath(os.path.join(base_path, include_path))
        if not os.path.isfile(full_path):
            raise FileNotFoundError(f"Included file not found: {full_path}")

        # Read the included file
        with open(full_path, "r") as f:
            included_content = f.read()

        # Apply start-after and end-before markers
        if start_after:
            included_content = included_content.split(start_after, 1)[-1]
        if end_before:
            included_content = included_content.split(end_before, 1)[0]

        # Return the processed content
        return included_content.strip()

    # Regex to match MyST {include} directives with optional start/end markers,
    # including surrounding backticks
    include_pattern = re.compile(
        r"(?:```{include}\s+([^\s]+)"
        r"(?:\s*:start-after:\s*\"([^\"]+)\")?"
        r"(?:\s*:end-before:\s*\"([^\"]+)\")?"
        r"\s*```)"
    )

    # Replace all {include} directives in the content, including backticks
    return include_pattern.sub(replace_include, content)


def replace_directives(app, docname, source):
    """Replace Markdown directives with MyST directives."""
    base_path = os.path.dirname(app.env.doc2path(docname))
    content = source[0]

    # Preprocess included content
    content = preprocess_includes(content, base_path)

    # Replace GitHub-style admonitions with MyST-compatible directives
    # Ensure this only applies to actual admonitions, not included content
    content = re.sub(
        r"^> \[!(NOTE|IMPORTANT|WARNING|TIP)\]\n> (.+)",
        r"```{\1}\n\2\n```",
        content,
        flags=re.MULTILINE,
    )

    source[0] = content


def setup(app: Sphinx) -> None:
    """App setup hook."""

    def link_replace(line):
        if line is not None:
            line = line.replace("List[", "~typing.List[")
            line = line.replace("Tuple[", "~typing.Tuple[")
            line = line.replace("Dict[", "~typing.Dict[")
            line = line.replace("Literal[", "~typing.Literal[")
            line = line.replace("Union[", "~typing.Union[")
            line = line.replace("Optional[", "~typing.Optional[")
        return line

    def autodoc_process_docstring(app, what, name, obj, options, lines):
        for i in range(len(lines)):
            lines[i] = link_replace(lines[i])

    def autodoc_process_signature(
        app, what, name, obj, options, signature, return_annotation
    ):
        return link_replace(signature), link_replace(return_annotation)

    app.connect("autodoc-process-docstring", autodoc_process_docstring)
    app.connect("autodoc-process-signature", autodoc_process_signature)

    app.add_generic_role("small", partial(nodes.inline, classes=["small"]))  # type: ignore
    app.connect("source-read", replace_directives)
