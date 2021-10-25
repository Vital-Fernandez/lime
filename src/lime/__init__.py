"""
LiMe - A python package for Measuring emission lines to study ionized gas chemical and dynamical properties
"""

import os
import sys
import configparser

from .treatment import Spectrum
from .io import loadConfData, lineslogFile_to_DF, import_fits_data
from .tools import label_decomposition, continuum_remover, line_finder, match_lines

# Get python version being used
__python_version__ = sys.version_info

# Read lime configuration
_dir_path = os.path.dirname(os.path.realpath(__file__))
_setup_cfg = configparser.ConfigParser()
_setup_cfg.optionxform = str
_setup_cfg.read(os.path.join(_dir_path, 'lime.cfg'))

__version__ = _setup_cfg['metadata']['version']
