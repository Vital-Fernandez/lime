Installation
============

:math:`\mathrm{LiMe}` can be installed from pip_. The following commands let the user choose the number of dependencies
installed along side it:

.. code-block:: console

   pip install lime-stable
   pip install lime-stable[full]

The first command only installs the core dependencies:

* Numpy_
* Pandas_
* Matplotlib_
* LmFit_ (fitting library)
* Astropy_ (loading and saving *.fits* files)
* scipy_ (:math:`\mathrm{LmFit}` and :math:`\mathrm{AstroPy}` dependency)
* tomli_ (read configuration files with toml_ files for python < 3.11)

The second command also installs the optional dependencies:

* asdf_ (Saving the measurements as *.asdf* files)
* bokeh_ (Plots using the bokeh libraryy)
* mplcursors_ (Interactive pop-ups in plots)
* openpyxl_ (Saving the measurements as *.xlsx* files)
* PyLatex_ (Saving the measurements as *.pdf* files)
* toml_ (Saving configuration files as *.toml* files)

To update the library to its latest version you can run this command:

.. code-block:: console

   pip install lime-stable --upgrade

Finally, for :math:`\mathrm{LiMe}` developers the following command installs the necessary dependencies to compile the
documentation and run the tests.

.. code-block:: console

   pip install lime-stable[full,docs,tests]

:math:`\mathrm{LiMe\,v2.0}` has been tested successfully with :math:`\mathrm{python\,v3.12}`. The github pyproject.toml_
contains the current building information including its dependencies version.

.. _pip: https://pypi.org/project/lime-stable/
.. _github: https://github.com/Vital-Fernandez/lime
.. _Numpy: https://numpy.org/install/
.. _Pandas: https://pandas.pydata.org/docs/getting_started/install.html
.. _scipy: https://scipy.org/beginner-install/
.. _Matplotlib: https://matplotlib.org/stable/users/installing/index.html
.. _LmFit: https://lmfit.github.io/lmfit-py/installation.html
.. _Astropy: https://docs.astropy.org/en/stable/install.html
.. _tomli: https://pypi.org/project/tomli/#installation

.. _asdf: https://asdf.readthedocs.io/en/stable/asdf/install.html
.. _bokeh: https://docs.bokeh.org/en/latest/docs/first_steps/installation.html
.. _mplcursors: https://mplcursors.readthedocs.io/en/stable/index.html
.. _openpyxl: https://pypi.org/project/openpyxl/
.. _PyLatex: https://jeltef.github.io/PyLaTeX/current/
.. _toml: https://toml.io/en/

.. _pyproject.toml: https://github.com/Vital-Fernandez/lime/blob/4319afec0920d6bb5bcb0b7304e7fd51604d2099/pyproject.toml


