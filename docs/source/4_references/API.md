# API

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

For instance, the [Spectrum.fit.bands](API.md) and [Spectrum.fit.frame](API.md) commands fit a single line and a list of
lines in a spectrum, respectively. Conversely, the [Cube.fit.spatial_mask](API.md) command fits a list of lines 
within a spatial region of an IFS cube.

The next sections detail the functions attributes and their outputs:

# Inputs/outputs

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