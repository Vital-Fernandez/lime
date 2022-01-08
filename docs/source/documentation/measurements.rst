========================
Measurements description
========================

This section describes the parameters measured by LiMe. Unless otherwise noted, these parameters have the same notation
in the output measurements log as the attributes in the programing objects generated with the ``lime.Spectrum`` class.
These parameter references are also the column names of the ``pandas.DataFrame`` lines log (``lime.Spectrum.linesDF``).

Inputs
++++++

This section includes 3 parameters which are actually provided by the user inputs. However, they are also included in
the output log for consistency.

* **lineLabel** (``.lineLabel``, ``str``): This attribute provides the LiMe label for an emission line flux. This label has the
  following format:

  .. image:: ../_static/line_notation.png
    :scale: 40%
    :align: center

  Single lines are those whose profile can be characterised with a single Gaussian component (default). In contrast,
  blended and merged lines need at least to Gaussian components. The discrepancy between both categories is that the
  blended lines the spectral resolution is high enough for a mathematical fitting of the individual components. Merged lines,
  can still be used in the chemical analysis, and hence, it is importat to preserve the components into the output measurements
  log.

  .. note::
     Even with the blended **'_b'** and merged **'_m'** suffices the user still needs to include the line components in
     the fit configuration. Otherwise, the lines shall be treated as single lines during the analysis.


* **lineWaves** (``.lineWaves``, ``np.array()``): This attribute consists in a six-value vector with the emission line
  location and adjacent continua:

  .. image:: ../_static/mask_selection.jpg
    :align: center

  This mask values must be supplied in increasing order. The units must be the same as the spectrum wavelength array.
  Finally, in the output measurements log these wavelengths are stored as  ``w1``, ``w2``, ``w3``, ``w4``, ``w5``, ``w6``.

* **blended_label** (``.blended_label``, ``str``): This attribute consists in a dash separated string with the line components
  in a blended or merged line. The individual components labels have the same notation as in the ``.lineLabel``. For example,
  in the configuration file the blended labels are defined as:

  .. code-block::

        H1_6563A_b = H1_6563A_b-N2_6584A-N2_6548A
        O2_3727A_m = O2_3727A-O2_3729A


Identification
++++++++++++++

These parameters are not attributes of the ``lime.Spectrum`` class. Nonetheless, they are stored in the ``lime.Spectrum.linesDF``
``pandas.DataFrame`` and the output measuring logs for chemical analysis of the emission fluxes.

  **wave**: This parameter contains the theoretical, rest-frame, wavelength for the emission line. This value is derived
  from the ``.lineLabel`` provided by the user.

  **ion**: This parameter contains the ion responsible for the emission line photons. This value is derived from the
  ``.lineLabel`` provided by the user.

  **latexLabel**: This parameter contains the transition classical notation in latex format. This string includes the
  blended and merged components if they were provided during the fitting.


Integrated properties
+++++++++++++++++++++

These attributes are calculated by the ``lime.Spectrum.line_properties`` without any assumption of the emission line profile.

.. attention::
    In the output measurements log and the ``lime.Spectrum.linesDF``, these parameters have the same flux units as the
    input spectrum. However, the attributes of the ``lime.Spectrum`` are normalized by the constant provided by the user
    ``lime.Spectrum.normFLux``

* **peak_wave** (``.peak_wave``, ``float``): This is the wavelength of the highest pixel value in the line region.

* **peak_flux** (``.peak_flux``, ``float``): This is the flux of the highest pixel value in the line region.

* **m_cont**  (``.m_cont``, ``float``): Using the line adjacent continua regions LiMe fits a linear continuum.
  This parameter represents is the gradient. :code:`y = m*x + n`

* **n_cont** (``.n_cont``, ``float``): Using the line adjacent continua regions LiMe fits a linear continuum.
  This parameter represents is the interception. :code:`y = m*x + n`

* **cont** (``.cont``, ``float``): This parameter is the flux of the linear continuum at the ``.peak_wave``.

* **std_cont**  (``.std_cont``, ``float``): This parameter is standard deviation of the adjacent continua flux. It is
  calculated from the observed continuum minus the linear model for both continua masks.

* **intg_flux** (``.intg_flux``, ``float``): This attribute contains measurement of the integrated flux.
  This value is calculated via a Monte Carlo algorithm:

  * If the pixel error spectrum is not provided by the user the algorithm takes the line ``.std_cont`` for all the pixels in the
    line region.

  * The pixels error spectrum (or ``.std_cont``) is added stochastically to each pixel in the line region mask.

  * The flux in the line region is summed up taking into consideration the line region averaged pixel width and removing
    the contribution of the linear continuum.

  * The previous two steps are repeated in a 500 loop. The mean flux value from the resulting array is taken as the integrated
    flux value.


* **intg_err** (``.intg_err``, ``float``): This attribute contains the integrated flux uncertainty. This
  value is derived from the standard deviation of the Monte Carlo algorithm steps described above.

.. attention::
    Blended lines have the same ``.intg_flux`` and ``.intg_err`` values.

* **eqw** (``.eqw``, ``float`` or ``np.array()``): This parameter is the equivalent of the emission line. It is calculated
  using the expression below:

    .. math::

        Eqw = \frac{F_{\lambda}}{F_{cont}}



  In blended lines the ``.gauss_flux`` is used otherwise the ``.intg_flux`` is used. In all cases the ``.cont`` is used
  as denominator.

* **eqw_err** (``.eqw``, ``float`` or ``np.array()``): This parameter is the uncertainty in the equivalent width. It is
  calculated from a Monte Carlo vector of the  ``.cont`` and its ``.std_cont`` and the uncertainty of the line flux.

* **z_line** (``.z_line``, ``float``): This parameter is the emission line redshift:

  .. math::

        z_{\lambda} = \frac{\lambda_{obs}}{\lambda_{theo}} - 1

  where :math:`\lambda_{obs}` is the ``.peak_wave`` for non-blended lines. Otherwise the gaussian profile ``.center`` is
  used. In all cases :math:`\lambda_{theo}` is the theoretical transition wavelength obtained from the input ``.lineLabel``

* **FWHM_int** (``.FWHM_int``, ``float``): This parameter is the Full Width Half-Measure in :math:`km/s` computed from
  the integrated profile: The algorithm finds the pixel coordinates which are above half the line peak flux. The blue and and red
  edge :math:`km/s` are subtracted (blue is negative). This operation is only available for lines whose width is above 15 pixels.

* **snr_line**  (``.FWHM_int``, ``float``): This parameter is the signal to noise ratio of the emission line region using the
  `IRAF splot definition <https://github.com/joequant/iraf/blob/master/noao/onedspec/splot/avgsnr.x>`_:

   .. math::

      SNR = \frac{avg}{rms} = \frac{{\frac {1}{n}}\sum _{i=1}^{n}y_{i}}{\sqrt{(\frac{1}{n})\sum_{i=1}^{n}(y_{i} - y_{avg})^{2}}}

* **snr_cont** This parameter is the signal to noise ratio of the emission line region using the `IRAF splot definition <https://github.com/joequant/iraf/blob/master/noao/onedspec/splot/avgsnr.x>`_
  as in the equation above.

* **v_med** (``.v_med``, ``float``)

* **v_50** (``.v_50``, ``float``)

* **v_5** (``.v_5``, ``float``)

* **v_10** (``.v_10``, ``float``)

* **v_90** (``.v_90``, ``float``)

* **v_95** (``.v_95``, ``float``)


Gaussian properties
+++++++++++++++++++

* **amp** (``.amp``, ``float`` or ``np.array()``)
* **amp_err** (``.amp_err``, ``float`` or ``np.array()``)

* **center** (``.center``, ``float`` or ``np.array()``)
* **center_err** (``.center_err``, ``float`` or ``np.array()``)

* **sigma** (``.sigma``, ``float`` or ``np.array()``)
* **sigma_err** (``.sigma_err``, ``float`` or ``np.array()``)

* **v_r** (``.v_r``, ``float`` or ``np.array()``)
* **v_r_err** (``.v_r_err``, ``float`` or ``np.array()``)

* **sigma_vel** (``.sigma_vel``, ``float`` or ``np.array()``)
* **sigma_vel_err** (``sigma_vel_err``, ``float`` or ``np.array()``)

* **FWHM_g** (``.FWHM_g``, ``float`` or ``np.array()``)

* **gauss_flux** (``.gauss_flux``, ``float`` or ``np.array()``)

* **gauss_err** (``.gauss_err``, ``float`` or ``np.array()``)


Measurement diagnostics
+++++++++++++++++++++++

* **chisqr** (``.chisqr``, ``float``)

* **redchi** (``.redchi``, ``float``)

* **aic** (``.aic``, ``float``)

* **bic** (``.bic``, ``float``)

* **observation** (``.observation``, ``str``)

* **comments** (``.comments``, ``str``)
