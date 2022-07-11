1) Single line fitting
======================

In this example, we perform single line fitting on the spectrum of the Green Pea galaxy GP121903 which was observed
with the GTC (Gran Telescopio de Canarias). You can download this spectrum from the `github examples folder <https://github.com/Vital-Fernandez/lime/tree/master/examples>`_.
You can read more about this data set in `Fernandez et al (2021) <https://arxiv.org/abs/2110.07741>`_.

This tutorial can also be found as a python script in the `github 1st example <https://github.com/Vital-Fernandez/lime/blob/master/examples/example1_single_line_fit.py>`_.

Loading the spectrum
------------------------

Once you have installed the library, you can load it into your scripts via:

.. code-block:: python

    import numpy as np
    from astropy.io import fits
    import lime


Please check the version you are using, whenever you have a question/issue/comment you want to share.

.. code-block:: python

    print(lime.__version__)


We can start by opening the .fits file using the `Astropy fits module <https://docs.astropy.org/en/stable/io/fits/index.html>`_ to
reconstruct the spectrum:

.. code-block:: python

    def import_osiris_fits(file_address, ext=0):

        # Open the fits file
        with fits.open(file_address) as hdul:
            data, header = hdul[ext].data, hdul[ext].header

        # Reconstruct the wavelength array from the header data
        w_min, dw, n_pix = header['CRVAL1'],  header['CD1_1'], header['NAXIS1']
        w_max = w_min + dw * n_pix
        wavelength = np.linspace(w_min, w_max, n_pix, endpoint=False)

        return wavelength, data, header


    # Address of the Green Pea galaxy spectrum
    gp_fits = './sample_data/gp121903_BR.fits'

    # Load spectrum
    wave, flux, hdr = import_osiris_fits(gp_fits)

Most of :math:`\textsc{LiMe}` functions are performed by the Spectrum class: This object stores your spectrum and
performs the line fitting functions. Its compulsory inputs are the spectrum wavelength and flux arras. However, in order
to identify and label the lines, a redshift input is necessary (unless z = 0). Finally, many line fitting functions will
fail in the non-normalized CGS units commonly used in spectra. Consequently, it is recommended to introduce a normalization
value. In the case of GP121903:

.. code-block:: python

    # Galaxy redshift and the flux normalization
    z_gp = 0.19531
    normFlux_gp = 1e-18

.. note::

    Even if a redshift is introduced the measurement will be performed in the observed framed. Moreover, they results are
    stored without the normalization.

Using this information, the Spectrum object is defined as:

.. code-block:: python

    gp_spec = lime.Spectrum(wave, flux, redshift=z_gp, norm_flux=normFlux_gp)


As additional inputs, you may provide the sigma (uncertainty) spectrum, a two value array to crop the spectrum
wavelength range and the wavelength and flux units (the default values are angstroms (A) and :math:`erg/s^2/cm/\AA`
(`erg/cm^2/s/A`).

To display the input spectrum you can use the function:

.. code-block:: python

    gp_spec.plot_spectrum(spec_label='GP121903')

.. image:: ../_static/plot_spectrum.png

To fit a line, we need to provide its location: First, two wavelengths marking the spectrum band, where the line is located.
Additionally, you need to define two continua regions on the left and right hand side of the line. This is a sorted six
value array:

.. image:: ../_static/mask_selection.jpg

For this galaxy, the Hα mask is:

.. code-block:: python

   line = 'H1_6563A'
   lineWaves = np.array([6438.03, 6508.66, 6535.10, 6600.95, 6627.70, 6661.82])


Let's fit the Hα line using the function ``fit_from_wavelengths``.

.. code-block:: python

    gp_spec.fit_from_wavelengths(line, lineWaves)


You can plot the fit using:

.. code-block:: python

    gp_spec.display_results()

.. image:: ../_static/1_firstFitAttemp.png

You can see that the result was not very good. Let's increase the complexity by including the [NII] lines:

.. code-block:: python

    line = 'H1_6563A_b'
    Halpha_conf = {'H1_6563A_b':     'H1_6563A-N2_6584A-N2_6548A',
                   'N2_6548A_amp':   {'expr': 'N2_6584A_amp/2.94'},
                   'N2_6548A_kinem': 'N2_6584A'}

The dictionary above has three elements:

* First: The line labelled as 'H1_6563A_b' consists in three components: H1_6563A, N2_6584A and N2_6548A
* Second: The line labelled as 'N2_6548A' has an amplitude fixed by the amplitude of "N2_6584A"
* Three: The line labelled as 'N2_6548A' has its kinematics (both radial and dispersion velocity) tied to those of "N2_6584A".

Now we include this information in the fitting:

.. code-block:: python

    gp_spec.fit_from_wavelengths(line, lineWaves, fit_conf)
    gp_spec.display_results()

.. image:: ../_static/1_secondFitAttemp.png

This time the profile is closer to the observational data.

Finally, the results can be saved as a table using the ``lime.save_line_log`` function. The log output format is
determined from the user address extension. Moreover, the user can also provide a sheet name for multi-page files (excel, fits
and asdf). This way the each new log will append a new sheet to the output file or update the one already there. Finally,
user can constrain the output measurements with a list of parameters. You can find the parameters keywords in the
:ref:`measurements documentation <measurements_page>`.

.. code-block:: python

    lime.save_line_log(gp_spec.log, './sample_data/example1_linelog.txt')
    lime.save_line_log(gp_spec.log, './sample_data/example1_linelog.fits', ext='GP121903')
    lime.save_line_log(gp_spec.log, './sample_data/example1_linelog.pdf', parameters=['eqw', 'gauss_flux', 'gauss_err'])
    lime.save_line_log(gp_spec.log, './sample_data/example1_linelog.xlsx', ext='GP121903')
    lime.save_line_log(gp_spec.log, './sample_data/example1_linelog.asdf', ext='GP121903')
