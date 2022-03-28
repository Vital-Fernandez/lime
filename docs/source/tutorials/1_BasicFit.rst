1) Simple fit
=============

In this example we perform single line fits on the emission spectrum of the Green Pea galaxy GP121903 which was observed
with the GTC (Gran Telescopio de Canarias). You can download this spectrum from the `github examples folder <https://github.com/Vital-Fernandez/lime/tree/master/examples>`_.
You can read more about this data set in `Fernandez et al (2021) <https://arxiv.org/abs/2110.07741>`_.

This tutorial can also be found as a python script in the `github 1st example <https://github.com/Vital-Fernandez/lime/blob/master/examples/example1_simple_fit.py>`_.

Loading the spectrum
------------------------

Once you have installed the library, you may import it into your scripts

.. code-block:: python

    import lime


Please check the version you are version you are using, whenever you have a question/issue/comment you want to share at
github.

.. code-block:: python

    print(lime.__version__)


We can start opening a spectrum. We can use `Astropy fits module <https://docs.astropy.org/en/stable/io/fits/index.html>`_:

.. code-block:: python

    import numpy as np
    from astropy.io import fits

    ext = 0
    with fits.open('./sample_data/gp121903_BR.fits') as hdul:
        flux, header = hdul[ext].data, hdul[ext].header

The spectrum wavelength can be reconstructed from the header:

.. code-block:: python

    w_min = header['CRVAL1']
    dw = header['CD1_1']
    pixels = header['NAXIS1']
    w_max = w_min + dw * pixels
    wave = np.linspace(w_min, w_max, pixels, endpoint=False)

Most of :math:`\textsc{LiMe}` functions are performed by the Spectrum class: This object stores your spectrum and performs the line
fitting functions. Its obligatory inputs are the spectrum wavelength and flux. However, in order to identify and
labeling lines a redshift value is necessary. Finally, many line fitting functions will fail in non-normalized CGS units
commonly used in spectra. Consequently, it is recommended to introduce a normalization value. In the case of GP121903:

.. code-block:: python

    z_gp = 0.19531
    normFlux_gp = 1e-14

.. note::

    Despite these inputs the measurement will be performed in the observed framed. Moreover, they are stored
    in the input flux units without the normalization.

Using this information, the Spectrum object is defined as:

.. code-block:: python

    gp_spec = lime.Spectrum(wave, flux, redshift=z_gp, norm_flux=normFlux_gp)




As additional inputs, you may provide the sigma (uncertainty) spectrum and a two value array to crop the spectrum
wavelength range (in the same frame as the input wavelength).

To display the input spectrum you can use the function:

.. code-block:: python

    gp_spec.plot_spectrum()

.. image:: ../_static/plot_spectrum.png

To fit a line we need to provide its location: Two wavelengths marking the spectrum region where the line is located.
Additionally, you need to define two continuum regions on the left and right hand side of the line. Therefore, you need
to provide a six value array ordered from lower to higher wavelengths:

.. image:: ../_static/mask_selection.jpg

For this Green Pea spectrum the Hα mask is:

.. code-block:: python

   lineWaves = np.array([6438.03, 6508.66, 6535.10, 6600.95, 6627.70, 6661.82])

In this array the first two values correspond to the left continuum, the third and fourth values correspond to the line
region and the the final two values correspond to the right hand side continuum. These values must be in the rest frame.

Let's fit the Hα line using the function fit_from_wavelengths

.. code-block:: python

    gp_spec.fit_from_wavelengths('H1_6563A', lineWaves)


You can plot the fit using:

.. code-block:: python

    gp_spec.display_results()

.. image:: ../_static/1_firstFitAttemp.png

You can see that the fitting is not very good. Let's increase the complexity by including the [NII] lines:

.. code-block:: python

    Halpha_conf = {'H1_6563A_b':     'H1_6563A-N2_6584A-N2_6548A',
                   'N2_6548A_amp':   {'expr': 'N2_6584A_amp / 2.94'},
                   'N2_6548A_kinem': 'N2_6584A'}

The dictionary above has three elements:

* First: The line labelled as 'H1_6563A_b' consists in three components: H1_6563A, N2_6584A and N2_6548A
* Second: The line labelled as 'N2_6548A' has an amplitude value fixed by the amplitude fitted in the line "N2_6584A"
* Three: The line labelled as 'N2_6548A' has its kinematics (both radial and dispersion velocity) imported from those fit in the line N2_6548A.

Now we include this information in the fitting:

.. code-block:: python

    gp_spec.fit_from_wavelengths(line, lineWaves, fit_conf)
    gp_spec.display_results()

.. image:: ../_static/1_secondFitAttemp.png

This time the fitted profile better represents the observation.


.. code-block:: python

    import numpy as np
    import lime

    # Load the spectrum data
    wave, flux = np.loadtxt('GP121903', unpack=True)
    z_gp = 0.19531
    normFlux_gp = 1e-14

    # Declare lime Spectrum object
    gp_spec = lime.Spectrum(wave, flux, redshift=z_gp, norm_flux=normFlux_gp)
    gp_spec.plot_spectrum(frame='rest', spec_label='GP121903')


.. code-block:: python

    # Perform the fitting
    line = 'H1_6563A'
    mask = np.array([6438.03, 6508.66, 6535.10, 6600.95, 6627.70, 6661.82])
    gp_spec.fit_from_wavelengths(line, mask)
    gp_spec.display_results()


.. code-block:: python

    # Perform the fitting
    line = 'H1_6563A_b'

    mask = np.array([6438.03, 6508.66, 6535.10, 6600.95, 6627.70, 6661.82])

    fit_conf = {'H1_6563A_b': 'H1_6563A-N2_6584A-N2_6548A',
                'N2_6548A_amp': {'expr': 'N2_6584A_amp / 2.94'},
                'N2_6548A_kinem': 'N2_6584A'}

    gp_spec.fit_from_wavelengths(line, mask, fit_conf)
    gp_spec.display_results()

.. code-block:: python

    # Measurements be saved according to the output file extension
    lime.save_line_log(gp_spec.log, './sample_data/example1_linelog.txt')
    lime.save_line_log(gp_spec.log, './sample_data/example1_linelog.fits', ext='GP121903')
    lime.save_line_log(gp_spec.log, './sample_data/example1_linelog.pdf')
    lime.save_line_log(gp_spec.log, './sample_data/example1_linelog.xlsx', ext='GP121903')

.. code-block:: python

    # Load measurements
    log = lime.load_lines_log('./sample_data/example1_linelog.fits', ext='GP121903')



.. code-block:: python

    # Load configuration
    obs_cfg = lime.load_cfg(cfgFile)

    gp_spec.fit_from_wavelengths(line, mask, fit_conf=obs_cfg['SHOC579_region0_line_fitting'])

.. code-block::

    H1_4341A_b = H1_4341A-O3_4363A
    O3_4363A_sigma = expr:H1_4341A_sigma if H1_4341A_amp/2. > 100 else 1.25

.. code-block::

    O3_5007A_b = O3_5007A-O3_5007A_W1-He1_5016A
    O3_5007A_w1_sigma = expr:>2.0*O3_5007A_sigma
    O3_5007A_w1_amp = expr:<10.0*O3_5007A_amp
    He1_5016A_center = min:5014,max:5018
    He1_5016A_sigma = min:1.0,max:2.0

.. code-block::

    O2_3726A_b = O2_3726A-O2_3729A-H1_3721A-H1_3734A
    O2_3726A_kinem = O2_3729A
    H1_3721A_kinem = H1_6563A
    H1_3734A_kinem = H1_6563A

    O2_3726A_cont_slope = vary:False
    O2_3726A_cont_intercept = vary:False

.. code-block::

    H1_4861A_b =  H1_4861A-H1_4861A_abs
    H1_4861A_abs_amp = value:-1,min:-inf,max:0
    H1_4861A_abs_sigma = expr:>2*H1_4861A_sigma

.. code-block:: python

    gp_spec.fit_from_wavelengths(line,
                                 mask,
                                 fit_conf=obs_cfg['gp121903'],
                                 fit_method='least_squares')

.. code-block:: python

    # Save some log results as ImageHDU
    param_list = ['intg_flux', 'intg_err', 'gauss_flux',
                  'gauss_err', 'v_r', 'v_r_err']
    lines_list = ['H1_4861A', 'H1_6563A', 'O3_4363A', 'O3_4959A',
                  'O3_5007A', 'S3_6312A', 'S3_9069A', 'S3_9531A']
    lime.save_param_maps(log_file, param_list, lines_list,
                         output_folder='./sample_data/',
                         spatial_mask_file=spatial_mask,
                         output_files_prefix='SHOC579_',
                         page_hdr=hdr_coords)