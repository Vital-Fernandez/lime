# API reference

$\mathrm{LiMe}$ features a composite software design, utilizing
instances of other classes to implement the target functionality. This
approach is akin to that of IRAF: Functions are organized into
multi-level packages, which users access to perform the corresponding
task. The diagram below outlines this workflow:

```{image} ../0_resources/images/LiMe_structure.png
:width: 500px
:align: center
```

At the highest level, $\mathrm{LiMe}$ provides of observational classes:
spectrum, cube, and sample. The first two are essentially 2D and 3D data
containers, respectively. The third class functions as a dictionary-like
container for multiple spectrum or cube objects. Moreover, as
illustrated in the figure above, various tools can be invoked via the
$\mathrm{LiMe}$ import for tasks, such as loading and saving data. Many
of these functions are also within the observations.

At an intermediate level, each observational class includes the *.fit*,
*.plot*, and *.check* objects. The first provides functions to launch
the measurements from the observation data. The second organizes
functions to plot the observations and/or measurements, while the
emph{.check} object facilitates interactive plots, allowing users to
select or adjust data through mouse clicks or widgets. In these
functions, users must specify an output file to store these user inputs.

Finally, at the lowest level, we find the functions that execute the
measurements or plots. Beyond the aforementioned functionality, the main
distinction between these commands lies in the extent of the data they
handle.

For instance, the [Spectrum.fit.bands](0_API.md) and [Spectrum.fit.frame](0_API.md) commands fit a single line and a list of
lines in a spectrum, respectively. Conversely, the [Cube.fit.spatial_mask](0_API.md) command fits a list of lines 
within a spatial region of an IFS cube.

The next sections detail the functions attributes and their outputs:

## Information

```{eval-rst}
.. autofunction:: lime.show_instrument_cfg
    :noindex:
```

```{eval-rst}
.. autofunction:: lime.show_profile_parameters
    :noindex:
```

## Inputs/outputs

```{eval-rst}
.. autofunction:: lime.load_cfg
    :noindex:
```

```{eval-rst}
.. autofunction:: lime.load_frame
    :noindex:
```

```{eval-rst}
.. autofunction:: lime.save_frame
    :noindex:
```
### Loading long-slit *.fits* files

```{eval-rst}
.. autofunction:: lime.OpenFits.osiris
    :noindex:
```

### Loading long-slit *.text* files

```{eval-rst}
.. autofunction:: lime.OpenFits.text
    :noindex:
```

### Loading long-slit *.fits* files

```{eval-rst}
.. autofunction:: lime.OpenFits.text
    :noindex:
```

```{eval-rst}
.. autofunction:: lime.show_instrument_cfg
    :noindex:
```

### Transitions and lines

```{eval-rst}
.. autofunction:: lime.lines_frame
    :noindex:
```

```{eval-rst}
.. autoclass:: lime.Line
   :members: 
   :exclude-members: __init__, label, wavelength, particle
```

```{eval-rst}
.. autoclass:: lime.Particle
   :members:
   :no-index:
   :exclude-members: __init__   
```

```{eval-rst}
.. autofunction:: lime.label_decomposition
    :noindex:
```


## Spectrum

```{eval-rst}
.. autoclass:: lime.Spectrum
   :members:
   :undoc-members:
   :no-show-inheritance:
   :exclude-members: from_survey, line_detection

.. rubric:: Spectrum fitting functions (.fit)

.. autoattribute:: lime.Spectrum.fit
   :noindex:
   :annotation:

.. automethod:: lime.workflow.SpecTreatment.bands
   :noindex:
   
.. automethod:: lime.workflow.SpecTreatment.frame
   :noindex:

.. automethod:: lime.workflow.SpecTreatment.continuum
   :noindex:
   
.. rubric:: Spectrum plotting functions (.plot) (matplotlib)

.. autoattribute:: lime.Spectrum.plot
   :noindex:
   :annotation:

.. automethod:: lime.plotting.plots.SpectrumFigures.spectrum
   :noindex:

.. automethod:: lime.plotting.plots.SpectrumFigures.grid
   :noindex:
   
.. automethod:: lime.plotting.plots.SpectrumFigures.velocity_profile
   :noindex:

.. automethod:: lime.plotting.plots.SpectrumFigures.show
   :noindex:
   
.. rubric:: Spectrum review functions (Cube.check.)

.. automethod:: lime.plotting.plots_interactive.BandsInspection.bands
   :noindex:   

```
## Cube

```{eval-rst}
.. autoclass:: lime.Cube
   :members:
   :undoc-members:
   :no-show-inheritance:

.. rubric:: Cube fitting functions (Cube.fit.)

.. autoattribute:: lime.Cube.fit
   :noindex:
   :annotation:

.. automethod:: lime.workflow.SpecTreatment.bands
   :noindex:

.. rubric:: Cube plotting functions (.plot) (matplotlib)

.. autoattribute:: lime.Cube.plot
   :noindex:
   :annotation:

.. automethod:: lime.plotting.plots.CubeFigures.cube
   :noindex:

.. rubric:: Cube review functions (Cube.check.)

.. automethod:: lime.plotting.plots_interactive.CubeInspection.cube
   :noindex:

```

## Tools

```{eval-rst}
.. autofunction:: lime.unit_conversion
    :noindex:
```

```{eval-rst}
.. autofunction:: lime.save_parameter_maps
    :noindex:
```