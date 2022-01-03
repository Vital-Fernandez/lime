Simple fit
==========

In this example we perform single line fits on the emission spectrum of the Green Pea galaxy GP121903 which was observed
with the GTC (Gran Telescopio de Canarias). You can download this spectrum from the `github examples folder <https://github.com/Vital-Fernandez/lime/tree/master/examples>`_.
You can read more about this data set in `Fernandez et al (2021) <https://arxiv.org/abs/2110.07741>`_.

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

Most of LiMe functions are performed by the Spectrum class: This object stores your spectrum and performs the line
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

    gp_spec = lime.Spectrum(wave, flux, redshift=z_gp, normFlux=normFlux_gp)




As additional inputs, you may provide the sigma (uncertainty) spectrum and a two value array to crop the spectrum
wavelength range (in the same frame as the input wavelength).

To display the spectrum you can use the function:

.. code-block:: python

    gp_spec.plot_spectrum()

To fit a line we need to provide its location: Two wavelengths marking the spectrum region where the line is located.
Additionally, you need to define two continuum regions on the left and right hand side of the line. Therefore, you need
to provide a six value array ordered from lower to higher wavelengths:
