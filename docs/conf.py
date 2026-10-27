"""Sphinx configuration. Build with ``uv run --group docs sphinx-build -W docs docs/_build/html``."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

project = "innards"
author = "Erik Voigt"
copyright = "2026, Erik Voigt"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
exclude_patterns = ["_build"]
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath"]
myst_heading_anchors = 3

# API reference from the (Google-style) docstrings
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented_params"
autodoc_default_options = {"members": True, "show-inheritance": True}
autodoc_mock_imports = ["umap", "wandb", "mlflow"]
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_rtype = False
napoleon_attr_annotations = True
napoleon_use_ivar = True  # attributes as fields, so dataclass members are not documented twice

html_theme = "furo"
html_title = "innards"
html_theme_options = {
    "source_repository": "https://github.com/VoErik/innards",
    "source_branch": "main",
    "source_directory": "docs/",
    "light_css_variables": {"color-brand-primary": "#c2410c", "color-brand-content": "#c2410c"},
    "dark_css_variables": {"color-brand-primary": "#fb923c", "color-brand-content": "#fb923c"},
}


def _references_as_bullets(app, what, name, obj, options, lines):
    """Turns ``References:`` sections (hanging-indent citations) into bullet lists before napoleon runs."""
    out, i = [], 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        i += 1
        if line.strip() != "References:":
            continue
        base = len(line) - len(line.lstrip())
        entries: list[str] = []
        entry_indent = None
        while i < len(lines) and (not lines[i].strip() or len(lines[i]) - len(lines[i].lstrip()) > base):
            text = lines[i]
            i += 1
            if not text.strip():
                continue
            indent = len(text) - len(text.lstrip())
            if entry_indent is None:
                entry_indent = indent
            if indent == entry_indent:
                entries.append(text.strip())
            else:
                entries[-1] += " " + text.strip()
        out.extend(" " * (base + 4) + "- " + e for e in entries)
        out.append("")
    lines[:] = out


def setup(app):
    app.connect("autodoc-process-docstring", _references_as_bullets, priority=100)  # before napoleon
