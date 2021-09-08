import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# README
README = (HERE/"README.rst").read_text()

# Setup
setup(
    name='lime',
    version='0.1.3',
    description="Line Measurer for ionized gas analysis",
    long_description=README,
    long_description_content_type='text/x-rst',
    url='https://github.com/Vital-Fernandez/lime',
    author="Vital Fernandez",
    author_email="vital.fernandez@userena.cl",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=['lime'],
    include_package_data=True,
    install_requires=['numpy', 'matplotlib', 'pandas', 'astropy'],
)
