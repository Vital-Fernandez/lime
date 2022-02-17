.. _measurements_page:

========================
Measurements description
========================

This section describes the parameters measured by :math:`\textsc{LiMe}`. Unless otherwise noted, these parameters have the same notation
in the output measurements log as the attributes in the programing objects generated with the ``lime.Spectrum`` class.
These parameter references are also the column names of the ``pandas.DataFrame`` lines log (``lime.Spectrum.log``).

Inputs
++++++

This section includes 3 parameters which are actually provided by the user inputs. However, they are also included in
the output log for consistency.

* **line** (``.line``, ``str``): This attribute provides the :math:`\textsc{LiMe}` label for an emission line. This label has the
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
     the fit configuration. Otherwise, the lines will be treated with a single component during the fitting.


* **lineWaves** (``.lineWaves``, ``np.array()``): This attribute consists in a six-value vector with the emission line
  location and adjacent continua:

  .. image:: ../_static/mask_selection.jpg
    :align: center

  This mask values must be supplied in increasing order. The units must be the same as the spectrum wavelength array.
  Finally, in the output measurements log these wavelengths are stored as  ``w1``, ``w2``, ``w3``, ``w4``, ``w5``, ``w6``.

* **blended_label** (``.blended_label``, ``str``): This attribute consists in a dash separated string with the line components
  in a blended or merged line. The individual components labels have the same notation as in the ``.line``. For example,
  in the configuration file the blended labels are defined as:

  .. code-block::

        H1_6563A_b = H1_6563A_b-N2_6584A-N2_6548A
        O2_3727A_m = O2_3727A-O2_3729A


Identification
++++++++++++++

These parameters are not attributes of the ``lime.Spectrum`` class. Nonetheless, they are stored in the ``lime.Spectrum.log``
``pandas.DataFrame`` and the output measuring logs for chemical analysis of the emission fluxes.

  **wave**: This parameter contains the theoretical, rest-frame, wavelength for the emission line. This value is derived
  from the ``.line`` provided by the user.

  **ion**: This parameter contains the ion responsible for the emission line photons. This value is derived from the
  ``.line`` provided by the user.

  **latexLabel**: This parameter contains the transition classical notation in latex format. This string includes the
  blended and merged components if they were provided during the fitting.


Integrated properties
+++++++++++++++++++++

These attributes are calculated by the ``lime.Spectrum.line_properties`` function. In these calculations, there is no
assumption on the emission line profile shape.

.. attention::
    In the output measurements log and the ``lime.Spectrum.log``, these parameters have the same flux units as the
    input spectrum. However, the attributes of the ``lime.Spectrum`` are normalized by the constant provided by the user
    ``lime.Spectrum.normFLux``

* **peak_wave** (``.peak_wave``, ``float``): This variable is the wavelength of the highest pixel value in the line region.

* **peak_flux** (``.peak_flux``, ``float``): This variable is the flux of the highest pixel value in the line region.

* **m_cont**  (``.m_cont``, ``float``): Using the line adjacent continua regions :math:`\textsc{LiMe}` fits a linear continuum.
  This variable represents is the gradient. :code:`y = m*x + n`

* **n_cont** (``.n_cont``, ``float``): Using the line adjacent continua regions :math:`\textsc{LiMe}` fits a linear continuum.
  This variable represents is the interception. :code:`y = m*x + n`

* **cont** (``.cont``, ``float``): This variable is the flux of the linear continuum at the ``.peak_wave``.

* **std_cont**  (``.std_cont``, ``float``): This variable is standard deviation of the adjacent continua flux. It is
  calculated from the observed continuum minus the linear model for both continua masks.

* **intg_flux** (``.intg_flux``, ``float``): This variable contains measurement of the integrated flux.
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
    Blended components have the same ``.intg_flux`` and ``.intg_err`` values.

* **eqw** (``.eqw``, ``float`` or ``np.array()``): This parameter is the equivalent of the emission line. It is calculated
  using the expression below:

    .. math::

        Eqw = \frac{F_{\lambda}}{F_{cont}}



  In blended lines the ``.gauss_flux`` is used otherwise the ``.intg_flux`` is used. In all cases the ``.cont`` is used
  as denominator.

* **eqw_err** (``.eqw``, ``float`` or ``np.array()``): This parameter is the uncertainty in the equivalent width. It is
  calculated from a Monte Carlo vector of the  ``.cont`` and its ``.std_cont`` and the uncertainty of the line flux.

* **z_line** (``.z_line``, ``float``): This variable is the emission line redshift:

  .. math::

        z_{\lambda} = \frac{\lambda_{obs}}{\lambda_{theo}} - 1

  where :math:`\lambda_{obs}` is the ``.peak_wave`` for non-blended lines. Otherwise the gaussian profile ``.center`` is
  used. In all cases :math:`\lambda_{theo}` is the theoretical transition wavelength obtained from the input ``.line``

* **FWHM_int** (``.FWHM_int``, ``float``): This variable is the Full Width Half-Measure in :math:`km/s` computed from
  the integrated profile: The algorithm finds the pixel coordinates which are above half the line peak flux. The blue and and red
  edge :math:`km/s` are subtracted (blue is negative).

  .. attention::
     This operation is only available for lines whose width is above 15 pixels.

* **snr_line**  (``.FWHM_int``, ``float``): This variable is the signal to noise ratio of the emission line region using the
  `IRAF splot definition <https://github.com/joequant/iraf/blob/master/noao/onedspec/splot/avgsnr.x>`_:

   .. math::

      SNR = \frac{avg}{rms} = \frac{{\frac {1}{n}}\sum _{i=1}^{n}y_{i}}{\sqrt{(\frac{1}{n})\sum_{i=1}^{n}(y_{i} - y_{avg})^{2}}}

* **snr_cont** (``.snr_cont``, ``float``): This variable is the signal to noise ratio of the emission line region using the `IRAF splot definition <https://github.com/joequant/iraf/blob/master/noao/onedspec/splot/avgsnr.x>`_
  as in the equation above.

* **v_med** (``.v_med``, ``float``): This variable is the median velocity of the emission line. The emission line wavelength
  is converted to velocity units using the formula:

  .. math::

        V (Km/s) = c \cdot \frac{\lambda_{obs}}{\lambda_{peak}} - 1

  where :math:`c = 299792.458 km/s` is the speed of light, :math:`\lambda_{obs}` is the wavelength mask array selection
  between :math:`w3` and :math:`w4` points and :math:`\lambda_{peak}` is the ``.peak_wave`` of the emission line.

* **v_50** (``.v_50``, ``float``): This variable is velocity corresponding to the 50th percentile of the emission line
  flux in :math:`km/s`. A cumulative sum is performed in the line flux array.  Afterwards, this array is multiplied by the
  ``.pixelWidth`` and divided by the ``.intg_flux``. The resulting vector quantifies the flux percentage corresponding to
  each pixel in the :math:`w3` and :math:`w4` mask selection. Afterwards, this vector is interpolated with respect to the
  velocity array (whose calculation is provided at ``.v_med``).  in order to compute velocity at the 50th flux percentile.

    .. attention::
       This operation is only available for lines whose width is above 15 pixels.

* **v_5** (``.v_5``, ``float``): This variable is the velocity corresponding to the 5th percentile of the emission line
  flux in :math:`km/s`. The calculation procedure is described at ``.v_50``.

* **v_10** (``.v_10``, ``float``): This variable is the velocity corresponding to the 10th percentile of the emission line
  flux in :math:`km/s`. The calculation procedure is described at ``.v_50``.

* **v_90** (``.v_90``, ``float``): This variable is the velocity corresponding to the 90th percentile of the emission line
  flux in :math:`km/s`. The calculation procedure is described at ``.v_50``.

* **v_95** (``.v_95``, ``float``): This variable is the velocity corresponding to the 95th percentile of the emission line
  flux in :math:`km/s`. The calculation procedure is described at ``.v_50``.


Gaussian properties
+++++++++++++++++++

These attributes are calculated by the ``lime.Spectrum.gauss_lmfit`` function. These calculations assume a Gaussian or
multi-Gaussian profile:

  .. math::

        F_{\lambda}=\sum_{i}A_{i}e^{-\left(\frac{\lambda-\mu_{i}}{2\sigma_{i}}\right)^{2}}

where :math:`F_{\lambda}` is the combined flux profile of the emission line for the line wavelength range :math:`\lambda`.
:math:`A_{i}` is the height of a gaussian component with respect to the line continuum (``.cont``), :math:`\mu_{i}` is the center
of the of gaussian component and :math:`\sigma_{i}` is the standard deviation. The first parameters has the input
flux units (``lime.Spectrum.flux``), while the later two have the input wavelength units (``lime.Spectrum.wave``).

The output uncertainty in these parameters corresponds to the `1σ error <https://lmfit.github.io/lmfit-py/fitting.html#uncertainties-in-variable-parameters-and-their-correlations>`_:
This is the standard error which increases the magnitude of the :math:`\chi^2` calculated by the least squares algorithm.

.. note::
   The Gaussian built-in model in `LmFit <https://lmfit.github.io/lmfit-py/builtin_models.html#lmfit.models.GaussianModel>`__
   defines the amplitude :math:`(A_{i})` as the flux under the gaussian profile. :math:`\textsc{LiMe}` defines its own model where the
   amplitude is defined as the height of the line with respect to the adjacent continuum.

* **amp** (``.amp``, ``np.array()``): This array contains the amplitude of the Gaussian components. The parameter units
  are those of the input spectrum flux (``lime.Spectrum.flux``).
* **amp_err** (``.amp_err``, ``np.array()``): This array contains the uncertainty on the Gaussian profiles amplitude.
  The parameter units are those of the input flux (``lime.Spectrum.flux``).

* **center** (``.center``, ``np.array()``): This array contains the Gaussian components central wavelength. The parameter units
  are those of the input spectrum wavelength (``lime.Spectrum.wave``).
* **center_err** (``.center_err``, ``np.array()``): This array contains the uncertainty on the Gaussian profiles central
  wavelength.

* **sigma** (``.sigma``, ``np.array()``): This array contains the Gaussian components standard deviation. The parameter units
  are those of the input spectrum wavelength.
* **sigma_err** (``.sigma_err``, ``np.array()``): This array contains the uncertainty on the Gaussian profiles standard deviation.

* **v_r** (``.v_r``, ``np.array()``): This array contains the Gaussian components radial velocity in :math:`km/s`. This
  parameter is calculated using the expression:

  .. math::

        v_{r} = c \cdot \frac{\lambda_{center}}{\lambda_{ref}} - 1

  where :math:`c = 299792.458 km/s` is the speed of light, :math:`\lambda_{center}` is the Gaussian profile central wavelength
  (``.center``) and :math:`\lambda_{ref}` is the reference wavelength. In non-blended lines :math:`\lambda_{ref}` is the
  observed peak wavelength (``.peak_wave``). In blended lines, :math:`\lambda_{ref}` is the theoretical wavelength (``.wave``) of the
  emission line transition (redshifted by the value provided by in the ``lime.Spectrum`` definition).

* **v_r_err** (``.v_r_err``, ``np.array()``): This array contains the uncertainty of the Gaussian components radial velocity
  in :math:`km/s`.

* **sigma_vel** (``.sigma_vel``, ``np.array()``): This array contains the Gaussian components standard deviation in :math:`km/s`.
  This parameter is calculated using the expression:

  .. math::

        \sigma_{v} (km/s) = c \cdot \frac{\sigma}{\lambda_{ref}}

  where c :math:`c = 299792.458 km/s` is the speed of light, :math:`\sigma` is the Gaussian profile standard deviation
  (``.sigma``) and :math:`\lambda_{ref}` is the reference wavelength. In non-blended lines :math:`\lambda_{ref}` is the
  observed peak wavelength (``.peak_wave``). In blended lines, :math:`\lambda_{ref}` is the theoretical wavelength
  (``.wave``) of the emission line transition (redshifted by the value provided by in the ``lime.Spectrum`` definition)

* **sigma_vel_err** (``sigma_vel_err``, ``float`` or ``np.array()``) This array contains the uncertainty of the Gaussian
  components standard deviation in :math:`km/s`.

* **FWHM_g** (``.FWHM_g``, ``np.array()``): This array contains the Full Width Half Maximum of the Gaussian components in
  in :math:`km/s`. This parameter is calculated as:

  .. math::

        FWHM_{g}=2\sqrt{2\,ln2}\sigma_{v}

  where :math:`\sigma` is the velocity dispersion of the Gaussian components (``.sigma_vel``).

* **gauss_flux** (``.gauss_flux``, ``np.array()``): This array contains the flux of the Gaussian components. It is calculated
  using the expression:

  .. math::
        F_{i, g} = A_i \cdot 2.50663 \cdot \sigma_i

  where :math:`A_i` is Gaussian component amplitude (``.amp``) and :math:`\sigma_{i}` gaussian component standard deviation (``.sigma``)

* **gauss_err** (``.gauss_err``, ``np.array()``): This array contains the uncertainty of the Gaussian components flux.


Diagnostics
+++++++++++

These section contains the parameters which provide a qualitative or quantitative diagnostic on the line measurement.

* **chisqr** (``.chisqr``, ``float``): This variable contains the :math:`\chi^2` diagnostic `calculated by LmFit <https://lmfit.github.io/lmfit-py/fitting.html#goodness-of-fit-statistics>`_

* **redchi** (``.redchi``, ``float``): This variable contains the reduced :math:`\chi^2` diagnostic
  `calculated by LmFit <https://lmfit.github.io/lmfit-py/fitting.html#goodness-of-fit-statistics>`_:

  .. math::
        \chi_{\nu}^2 = \frac{\chi^2}{N-N_{varys}}

  where the :math:`\chi^2` diagnostic is divided by the number of data points, :math:`N`, minus the number of dimensions
  :math:`N_{varys}`

* **aic** (``.aic``, ``float``): This variable contains the `Akaike information criteria <https://en.wikipedia.org/wiki/Akaike_information_criterion>`_
  calculated by `LmFit <https://lmfit.github.io/lmfit-py/fitting.html#goodness-of-fit-statistics>`_

* **bic** (``.bic``, ``float``): This variable contains the `Bayesian information criteria <https://en.wikipedia.org/wiki/Bayesian_information_criterion>`_
  calculated by  `LmFit <https://lmfit.github.io/lmfit-py/fitting.html#goodness-of-fit-statistics>`_

* **observation** (``.observation``, ``str``): This variable contains errors or warnings generated during the fitting of the line (not implemented).

* **comments** (``.comments``, ``str``): This variable is left empty for the user to store comments.
