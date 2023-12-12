"""
LiMe - A python package for measuring lines in astronomical spectra
"""

import os
import sys
import configparser
import logging

# Creating the lime logger
_logger = logging.getLogger("LiMe")
_logger.setLevel(logging.INFO)

# Outputting format
consoleHandle = logging.StreamHandler()
consoleHandle.setFormatter(logging.Formatter('%(name)s %(levelname)s: %(message)s'))
_logger.addHandler(consoleHandle)

# Get python version being used
__python_version__ = sys.version_info

# Read lime configuration
_dir_path = os.path.dirname(os.path.realpath(__file__))
_setup_cfg = configparser.ConfigParser()
_setup_cfg.optionxform = str
_setup_cfg.read(os.path.join(_dir_path, 'config.cfg'))

__version__ = _setup_cfg['metadata']['version']
_lines_database_path = (os.path.join(_dir_path, 'resources/parent_bands.txt'))

# Logging configuration
_logger.debug(f'Launching LiMe {__version__} in Python {__python_version__}')


class Error(Exception):
    """LiMe exception function"""


from .observations import Spectrum, Sample, Cube, line_bands
from .io import *
from .tools import *
from .transitions import Line, label_decomposition
from .read_fits import OpenFits


