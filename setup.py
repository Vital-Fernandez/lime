import os
import pathlib
import configparser
from setuptools import setup
from setuptools import find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# README
README = (HERE/"README.rst").read_text()

# Read lime configuration
_dir_path = os.path.dirname(os.path.realpath(__file__))
_setup_cfg = configparser.ConfigParser()
_setup_cfg.optionxform = str
_setup_cfg.read(HERE/'setup.cfg')

# Setup
setup(
    name=_setup_cfg['metadata']['name'],
    version=_setup_cfg['metadata']['version'],
    author=_setup_cfg['metadata']['author'],
    author_email=_setup_cfg['metadata']['author_email'],
    description=_setup_cfg['metadata']['description'],
    long_description=README,
    long_description_content_type=_setup_cfg['metadata']['long_description_content_type'],
    url=_setup_cfg['metadata']['url'],
    license=_setup_cfg['metadata']['licence'],
    packages=find_packages('src'),
    package_dir={'': 'src'},
    package_data={'': ['config.toml', 'resources/*']},
    include_package_data=True,
    install_requires=['numpy', 'matplotlib', 'pandas', 'astropy', 'lmfit', 'scipy', 'pylatex', 'openpyxl',
                      'joblib'],
    )
