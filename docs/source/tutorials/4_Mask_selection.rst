4) Mask selection
=================

An important step in the spectra analysis is the proper definition of the wavelength masks. :math:`\textsc{LiMe}` follows
a mask design based on the Lick indices `(see Zhang et al. 2005) <https://arxiv.org/abs/astro-ph/0508634v1>`_. These spectral
line incides are based on the data from the `Lick Observatory <https://www.lickobservatory.org/>`_ and they were proposed to study stellar age and metallicity.

The Lick indices consist in three wavelength intervals: One interval covers the line (emission and absorption) while the
other two cover a blue and red interval from the adcjacent continua. The first region is necessary to measure the line
properties while the other to are used to fit the local continuum. This can be seen in the following image:

.. image:: ../_static/mask_selection.jpg

The plot above shows the output `matplotlib <https://matplotlib.org/>`_ figure from the ``lime.Spectrum.display_results``
function. The upper plot shows the object spectrum where the input ``.flux`` is normalized by the ``.norm_flux``. The
mask wavelength limits (:math:`(w1, w2, ..., w6 )`) have been annotated over the plot. However, it may be appreciated that
there are three shaded regions below the spectrum shape. The green region corresponds to the line band while the orange
regions correspond to the continua bands.
<
The lower plot displays the observed spectrum minus the fitted profile (the solid black and dashed orange lines
respectively from the upper plot, respectively). In this plot, however, the regions outside the mask are not subtracted.
This is why there are flat regions (most visibly for the blue band) of the stepped line.

The mask file used by :math:`\textsc{LiMe}` consist in a seven columns table: One for the emission line label and six for
the line bands. The columns in this file are organized from lower to higher wavelengths.

.. image:: ../_static/4_mask_file.png

In the recommended workflow, the user will move through three mask files:

1. A master mask: This file contains all the lines a researcher is interested.
2. An instrument mask: This file contains all the candidate lines, which might be found in a spectra sample given the
   instrument wavelength range and the observation with the largest signal-to-noise.
3. An object mask: This file only contains the lines observed in an object spectrum. The band limits have been individually
   inspected and adjusted to match the line width and to avoid uneven features on the adjacent continua.

In large observations, however, a compromise might be necessary for the last step. To make the task easier, lime includes
the ``.MaskInspector`` class.

This tutorial can also be found as a python script in the `github 4th example <https://github.com/Vital-Fernandez/lime/blob/master/examples/example4_interactive_mask_plots.py>`_.

Let's start by importing the data files from the Green Pea sample:

.. code-block:: python

    import numpy as np
    from astropy.io import fits
    import lime

    # Input files
    obsFitsFile = './sample_data/gp121903_BR.fits'
    instrMaskFile = './sample_data/gp121903_BR_mask.txt'
    cfgFile = './sample_data/config_file.cfg'

    # Load configuration
    sample_cfg = lime.load_cfg(cfgFile, obj_section={'sample_data': 'object_list'})

    # Load mask
    maskDF = lime.load_lines_log(instrMaskFile)

    # Load spectrum
    ext = 0
    with fits.open('./sample_data/gp121903_BR.fits') as hdul:
        flux, header = hdul[ext].data, hdul[ext].header
    w_min, dw, n_pix = header['CRVAL1'], header['CD1_1'], header['NAXIS1']
    w_max = w_min + dw * n_pix

    wave = np.linspace(w_min, w_max, n_pix, endpoint=False)

Now we can prepare the data for the ``.MaskInspector`` class:

.. code-block:: python

    # Object properties
    z_obj = sample_cfg['sample_data']['z_array'][2]
    norm_flux = sample_cfg['sample_data']['norm_flux']

    # Run the interative plot
    objMaskFile = './sample_data/gp121903_BR_mask_corrected.txt'
    lime.MaskInspector(lines_log_address=objMaskFile, lines_DF=maskDF,
                       input_wave=wave, input_flux=flux, redshift=z_obj, norm_flux=norm_flux)

This class generates an interactive grid plot with the line masks provided by the ``log`` parameter:

.. image:: ../_static/4_mask_selection_grid.png

Clicking and dragging the mouse within a line plot cell will update the line band region, both in the plot and the output
file in the ``lines_log_address`` parameter. There are some caveat in the window selection:

* The plot wavelength range is always 5 pixels beyond the mask bands. Therefore dragging the mouse beyond the mask limits
  (below :math:`w1` or above :math:`w6`) will change the displayed range. This can be used to move beyond the original
  mask limits.
* Selections between the :math:`w2` and :math:`w5` wavelength bands are always assigned to the line region mask as the new
  math:`w3` and :math:`w4` values.
* Due to the previous point, to increase the :math:`w2` value or to decrease :math:`w5` value the user must select a region
  between :math:`w1` and :math:`w3` or :math:`w4` and :math:`w6` respectively.
* The text file is updated with each new selection but only for the corrected line.

In the case where the input spectrum has many lines or the user display is not sufficiently big. The user can apply the
``.MaskInspector`` mask in several steps. Each time the function will update the output object mask with the corrected
values.

.. code-block:: python

    lines_log_section = maskDF[:5]
    lime.MaskInspector(lines_log_address=objMaskFile, log=lines_log_section,
                       input_wave=wave, input_flux=flux, redshift=z_obj, norm_flux=norm_flux)


.. image:: ../_static/4_mask_selection_grid_Detail.png
