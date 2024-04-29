"""
LiMe - A python package for measuring lines in astronomical spectra
"""

import os
import sys
import logging
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

# Creating the lime logger
_logger = logging.getLogger("LiMe")
_logger.setLevel(logging.INFO)

# Outputting format
consoleHandle = logging.StreamHandler()
consoleHandle.setFormatter(logging.Formatter('%(name)s %(levelname)s: %(message)s'))
_logger.addHandler(consoleHandle)

# Get python version being used
__python_version__ = sys.version_info

# Read lime configuration .toml
_inst_dir = Path(__file__).parent
_conf_path = _inst_dir/'config.toml'
with open(_conf_path, mode="rb") as fp:
    _setup_cfg = tomllib.load(fp)

__version__ = _setup_cfg['metadata']['version']
_lines_database_path = (os.path.join(_inst_dir, 'resources/parent_bands.txt'))

# Logging configuration
_logger.debug(f'Launching LiMe {__version__} in Python {__python_version__}')


class Error(Exception):
    """LiMe exception function"""

from .observations import Spectrum, Sample, Cube, line_bands
from .io import *
from .tools import *
from .transitions import Line, label_decomposition
from .read_fits import OpenFits, show_instrument_cfg
from .recognition import detection_function
from .plots import theme