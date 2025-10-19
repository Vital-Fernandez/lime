"""
LiMe - A python package for measuring lines in astronomical spectra
"""

import sys
import logging

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

from lime.observations import Spectrum, Sample, Cube
from lime.io import *
from lime.tools import *
from lime.plotting.plots import theme
from lime.archives.read_fits import OpenFits, show_instrument_cfg
from lime.transitions import label_decomposition, lines_frame, bands_from_measurements, Line, Particle
from lime.rsrc_manager import lineDB
from lime.fitting.lines import show_profile_parameters

# Get python version being used
__python_version__ = sys.version_info

# Library version
__version__ = lime_cfg['metadata']['version']

# Logging configuration
_logger.debug(f'Launching LiMe {__version__} in Python {__python_version__}')

