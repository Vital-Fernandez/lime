#########
Line Measurer
#########

This project provides a set of tools to fit emission lines from ionized spectra for a posterior chemical and kinematic analysis

Getting Started
==================

The current beta version has been developed for the ULS students and researchers.

Installation
=============

The current version does not include a package installation. The library source folder (../lime/src/) needs to be
added to the python path manually. For example you may add the following lines at the begining of your scripts to import the library: ::

    import sys
    src_lime_folder = '\folder\to\lime\src'
    sys.path.append(src_lime_folder)
    import lime


