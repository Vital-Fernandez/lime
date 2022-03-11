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
    classifiers=[
                "License :: OSI Approved :: MIT License",
                "Programming Language :: Python :: 3",
                "Programming Language :: Python :: 3.7",
                ],
    packages=find_packages('src'),
    package_dir={'': 'src'},
    package_data={'': ['lime.cfg', 'types_params.txt']},
    include_package_data=True,
    install_requires=['numpy', 'matplotlib', 'pandas', 'astropy', 'lmfit', 'scipy', 'specutils', 'pylatex', 'openpyxl'],
    )
