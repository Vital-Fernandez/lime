# Installation

$\mathrm{LiMe}$ can be installed via [pip](https://pypi.org/project/lime-stable/). The following commands install $\mathrm{LiMe}$ 
either on its own or along with all its dependencies:

```bash
pip install lime-stable  
pip install lime-stable[full]
```

## Core Dependencies

The following packages are essential for $\mathrm{LiMe}$'s operation:

- [Numpy](https://numpy.org/install/) – Array operations  
- [Pandas](https://pandas.pydata.org/docs/getting_started/install.html) – Table management and reading tabulated files  
- [Matplotlib](https://matplotlib.org/stable/users/installing/index.html) – Default plotting library  
- [LmFit](https://lmfit.github.io/lmfit-py/installation.html) – Fitting library  
- [Astropy](https://docs.astropy.org/en/stable/install.html) – Loading and saving `.fits` files  
- [scipy](https://scipy.org/beginner-install/) – Scientific algorithms  
- [tomli](https://pypi.org/project/tomli/#installation) – Reading `.toml` files for Python < 3.11  

## Optional Dependencies

The following packages enable optional features but are not required:

- [asdf](https://asdf.readthedocs.io/en/stable/asdf/install.html) – Save measurements as `.asdf` files  
- [bokeh](https://docs.bokeh.org/en/latest/docs/first_steps/installation.html) – Plots using the Bokeh library  
- [mplcursors](https://mplcursors.readthedocs.io/en/stable/index.html) – Interactive pop-ups in plots  
- [openpyxl](https://pypi.org/project/openpyxl/) – Save measurements as `.xlsx` files  
- [PyLaTeX](https://jeltef.github.io/PyLaTeX/current/) – Save measurements as `.pdf` files  
- [toml](https://toml.io/en/) – Save configuration files as `.toml`  

## Updating LiMe

To upgrade to the latest version:

```bash
pip install lime-stable --upgrade
```

## Uninstalling or Changing Versions

To uninstall $\mathrm{LiMe}$:

```bash
pip uninstall lime-stable
```

To install a specific version:

```bash
pip install lime-stable==1.4
```

## For Developers

The following command installs all dependencies required to compile the documentation and run tests:

```bash
pip install lime-stable[full,docs,tests]
```

**LiMe v2.0** has been tested with **Python 3.12**. See the [pyproject.toml](https://github.com/Vital-Fernandez/lime/blob/4319afec0920d6bb5bcb0b7304e7fd51604d2099/pyproject.toml) on GitHub for the most up-to-date build and dependency information.