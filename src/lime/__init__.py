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
_conf_path = _inst_dir/'lime.toml'
with open(_conf_path, mode="rb") as fp:
    _setup_cfg = tomllib.load(fp)

__version__ = _setup_cfg['metadata']['version']

# Logging configuration
_logger.debug(f'Launching LiMe {__version__} in Python {__python_version__}')


# class Error(Exception):
#     """LiMe exception function"""

from lime.observations import Spectrum, Sample, Cube
from lime.io import *
from lime.tools import *
from lime.transitions import Line, label_decomposition, bands_from_frame
from lime.archives.read_fits import OpenFits, show_instrument_cfg
from lime.plotting.plots import theme
from lime.workflow import line_bands
