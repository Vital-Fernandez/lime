# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


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

def create_rst_from_changelog(input_file, output_file):

    with open(input_file, 'r') as file:
        lines = file.readlines()

    # Start the rst file content
    rst_content = ["Changelog\n", "=========\n\n"]

    # Detect version lines (assuming LiMe start)
    for line in lines:
        if line.strip().startswith("LiMe") and "LiMe" in line:
            version_type, version_number, version_date = line.split('-')
            version_info = f'{version_number.strip()} {version_type.strip()} ({version_date.strip()})\n'
            rst_content.append(version_info)
            rst_content.append(f"{'-' * len(version_info)}\n\n")

        # Process bullet points with indentation
        elif line.strip().startswith("-"):
            rst_content.append(f"* {line.strip()[1:].strip()}\n")

        # Detect the date format and append it after the version
        elif line.strip():
            rst_content.append(f"**{line.strip()}**\n")

        # Empty lines or additional text
        else:
            rst_content.append(f"{line.strip()}\n")

    # Write the content to the output rst file
    with open(output_file, 'w') as file:
        file.writelines(rst_content)

def create_md_from_changelog(input_file, output_file):

    with open(input_file, 'r') as file:
        lines = file.readlines()

    # Start the Markdown file content
    md_content = ["# Changelog\n\n"]

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("LiMe") and "LiMe" in stripped:
            # Expect format like: LiMe - v2.0 - 2025-08-01
            version_type, version_number, version_date = stripped.split('-')
            version_header = f"## {version_number.strip()} {version_type.strip()} ({version_date.strip()})\n"
            md_content.append(version_header)

        elif stripped.startswith("-"):
            # Bullet point
            md_content.append(f"- {stripped[1:].strip()}\n")

        elif stripped:
            # Any other non-empty line: highlight as bold
            md_content.append(f"**{stripped}**\n")

        else:
            # Preserve empty lines
            md_content.append("\n")

    # Write to output markdown file
    with open(output_file, 'w') as file:
        file.writelines(md_content)

    return



# -- Project information -----------------------------------------------------

project = 'lime'
copyright = '2021, Vital-Fernandez'
author = 'Vital-Fernandez'

# The full version, including alpha/beta/rc tags
release = "2.0.dev14"


# -- General configuration ---------------------------------------------------

extensions = [# 'myst_parser',  # Markdown support
                'myst_nb',
                'sphinx_togglebutton',
                'sphinx.ext.duration',
                'sphinx.ext.doctest',
                'sphinx.ext.autodoc',
                'sphinx.ext.viewcode',
                'sphinx.ext.autosummary',
                'sphinx.ext.intersphinx',
                'sphinx.ext.mathjax',
                'matplotlib.sphinxext.plot_directive',
                # 'nbsphinx',
                ]

# Markdown configuration
myst_enable_extensions = [
    "amsmath",     # for \begin{equation} … \end{equation} support
    "dollarmath",  # for inline math with $…$ or $$…$$
    "colon_fence",
    "html_admonition",
    "html_image",
    "deflist",
    "smartquotes",
]

# Auto-generate heading anchors up to this level
myst_heading_anchors = 2

source_suffix = {'.rst': 'restructuredtext',
                 '.ipynb': 'myst-nb',
                 '.myst': 'myst-nb',
}

# Jupyter notebooks settings
nbsphinx_markdown = True
nbsphinx_allow_errors = True
nbsphinx_execute = 'always'

# Autodoc options
autodoc_member_order = 'bysource'
autodoc_default_options = {"imported-members": True}

# Template paths
templates_path = ['_templates', '_build']

# Exclude patterns
exclude_patterns = []

# Cell execution in case of documentation testing
nb_execution_mode = "off"

# -- MathJax configuration ---------------------------------------------------

mathjax3_config = {
    "tex": {
        "macros": {
            "textsc": ["{\\small \\uppercase{#1}}", 1]
        }
    }
}

imgmath_latex_preamble = r'\usepackage[active]{preview}'
imgmath_use_preview = True


# -- HTML output configuration -----------------------------------------------

# html_theme = 'sphinx_rtd_theme'
html_theme = 'sphinx_book_theme'
# html_static_path = ['_static']
html_static_path = []

html_theme_options = {"logo": {"image_light": "0_resources/images/LiMe2_logo_white_transparent.png",
                               "image_dark":  "0_resources/images/LiMe2_logo_dark_transparent.png",
                               "alt_text":    "LiMe Documentation",
                                # "text":        "Documentation-",
                              },

                      "secondary_sidebar_items": [], # Hide right contents sidebar
                     }

# -- Folder cleanup and example copy -----------------------------------------

# Paths to the documentation and the notebooks (examples) folder
_lib_path = Path(__file__).parents[2]/'src'
_doc_folder = Path(__file__).parents[2]/'docs/source'
_examples_path = Path(__file__).parents[2]/'examples'

sys.path.append(_lib_path.as_posix())
sys.path.append(_examples_path.as_posix())

# Delete existing files and copy the new versions
list_folders = ['0_resources', '1_introduction', '2_guides', '3_explanations', '4_references']
for sub_folder in list_folders: shutil.rmtree(_doc_folder / sub_folder, ignore_errors=True)
for sub_folder in list_folders: shutil.copytree(_examples_path/sub_folder, _doc_folder/sub_folder, dirs_exist_ok=True)

# -- Changelog compiler ------------------------------------------------------

# Compile the changelog page
input_txt_changelog = _lib_path/'lime/changelog.txt'  # Path to the uploaded changelog file
output_rst_changelog = _doc_folder/'4_references/0_changelog.md'  # Output md file
create_md_from_changelog(input_txt_changelog, output_rst_changelog)