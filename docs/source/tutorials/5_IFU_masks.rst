5) IFU treatment: Spatial masks
===============================

In this tutorial, we perform a pre-analysis of an IFU (Integral Field Unit) cube. These data sets provide both spatial
and spectroscopic information of the observed light of astronomical bodies. You can find this tutorial as a python script
in the `github 5th example <https://github.com/Vital-Fernandez/lime/blob/master/examples/example5_IFU_Cube_masking.py>`_.

The :math:`\textsc{LiMe}` treatment we have seen in the previous tutorials remains the same for IFU data sets. However,
in order to preserve the spatial information and to maximise the quality of the measurements, it is recommended to use
spatial masks. These provide two advantages:

* In many cases, the scientific data does not cover the complete IFU field of view. For a better use of the astronomer's
  time and the computational resources, the non-scientific data should be excluded from the workflow.
* In most astronomical bodies, the phenomena responsible for the observed photons do not remain constant. For example,
  the gas ionization within a galaxy can change in a few IFU spaxels from very high to non-existent. Consequently, as the
  number and profile of the emission features changes, so should your analysis adapt. Spatial masks provide the means
  to personalise the :math:`\textsc{LiMe}` treatment.

Let's start by downloading one IFU data cube from the `MANGA survey <https://www.sdss.org/surveys/manga/>`_. In this
tutorial we will analyze `SHOC579 <https://dr17.sdss.org/marvin/galaxy/8626-12704/>`_, a compact galaxy with an intense
star forming region:

.. image:: ../_static/5_SHOC579_marvin.png
    :align: center

You can download the cube covering the field in the image above from the `MARVIN explorer <https://dr17.sdss.org/marvin/galaxy/8626-12704/>`_
website. However, you should be able to download the data cube with the commands:

.. code-block:: python

    import lime
    import wget
    import gzip
    import shutil
    import numpy as np
    from astropy.io import fits
    from pathlib import Path


    # Function to download the cube if not done already
    def fetch_spec(save_address, cube_url):
        if not Path(save_address).is_file():
            wget.download(cube_url, save_address)
        return


    # Function to extract the compressed cube if not done already
    def extract_gz_file(input_file_address, output_file_address):
        print(output_file_address)
        if not Path(output_file_address).is_file():
            with gzip.open(input_file_address, 'rb') as f_in:
                with open(output_file_address, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)


    # Web link and saving location
    SHOC579_url = 'https://data.sdss.org/sas/dr17/manga/spectro/redux/v3_1_1/8626/stack/manga-8626-12704-LOGCUBE.fits.gz'
    SHOC579_gz_address = './sample_data/manga-8626-12704-LOGCUBE.fits.gz'

    # Download the data (it may take some time)
    fetch_spec(SHOC579_gz_address, SHOC579_url)

    # Extract the gz file
    SHOC579_cube_address = './sample_data/manga-8626-12704-LOGCUBE.fits'
    extract_gz_file(SHOC579_gz_address, SHOC579_cube_address)


The second step is visualizing the data. While, the are many (and better) tools in the literature to visualize an
IFU cube, :math:`\textsc{LiMe}` provides the ``CubeFitsInspector`` class. This command opens an interactive `matplotlib <https://matplotlib.org/>`_.
window to display spectrum of the clicked spaxel. Before running this task, however, we need to load the data:

.. code-block:: python

    # Open the cube fits file
    with fits.open(SHOC579_cube_address) as hdul:
        hdr = hdul['FLUX'].header
        wave = hdul['WAVE'].data
        flux = hdul['FLUX'].data

Now we need to define a 2D image of our galaxy for the plot. Since this object luminosity comes from the ionized gas,
it is a good idea to use the emission lines bands. We can get this data from the masks files and the redshift
of the galaxy:

.. code-block:: python

    # Load the configuration file and the line masks:
    cfgFile = './sample_data/config_file.cfg'
    obs_cfg = lime.load_cfg(cfgFile)
    z_SHOC579 = obs_cfg['SHOC579_data']['redshift']

    # and the masks file
    mask_file = './sample_data/osiris_mask.txt'
    mask_log = lime.load_lines_log(mask_file)

Now we are going to generate a :math:`H\alpha` image summing all the pixels in the ``H1_6563A_b`` mask (actually, this
band also includes the :math:`[NII]6548,6584\AA` emission). To do this, we do some `fancy indexing <https://numpy.org/doc/stable/user/basics.indexing.html>`_
on the data cube:

.. code-block:: python

    # Establish the band image for the plot background using Halpha
    Halpha_band = mask_log.loc['H1_6563A_b', 'w3':'w4'].values * (1 + z_SHOC579)
    idcs_Halpha = np.searchsorted(wave, Halpha_band)
    Halpha_image = flux[idcs_Halpha[0]:idcs_Halpha[1], :, :].sum(axis=0)

Finally, we have the option to include some contours in the plot. These can be for the background image or we can use
another one. For example, let's use the percentile intensity of the :math:`[SII]6716,6731\AA` band for the image contours

.. code-block:: python

    # Use SII lines as the foreground image contours
    SII_band = mask_log.loc['S2_6716A_b', 'w3':'w4'].values * (1 + z_SHOC579)
    idcs_SII = np.searchsorted(wave, SII_band)
    SII_image = flux[idcs_SII[0]:idcs_SII[1], :, :].sum(axis=0)

    # Establishing the contours intensity using percentiles
    percentile_array = np.array([70, 80, 90, 95, 99, 99.9])
    SII_contourLevels = np.nanpercentile(SII_image, percentile_array)

Now we can run the ``CubeFitsInspector`` class:

.. code-block:: python

    # Labels for the axes
    ax_conf = {'image': {'xlabel': r'RA', 'ylabel': r'DEC', 'title': f'MANGA SHOC579'}}


    # Color normalization for the flux band:
    min_flux = np.nanpercentile(Halpha_image, 60)
    log_norm_bg = colors.SymLogNorm(linthresh=min_flux, vmin=min_flux, base=10)

    # Interactive plotter for IFU data cubes
    lime.CubeFitsInspector(wave, flux, Halpha_image, SII_image, SII_contourLevels,
                           fits_header=hdr, axes_conf=ax_conf, color_norm=log_norm_bg)



.. image:: ../_static/5_SHOC579_CubeFitsInspector.png
    :align: center

.. note::

    The interpretation of the data can be heavily affected by the selected flux band, as well as the color palette
    and normalization. The current phase: data visualization and mask selection is arguably the most important and the user
    is encouraged to attempt many strategies.

As we can see SHOC579 is very compact with a rich set of emission lines. At this point we can use the ``spatial_mask_generator``
to generate spatial masks based on the [SII] percentiles as in the plot above:

.. code-block:: python

    # Output masks file address
    mask_file = './sample_data/SHOC579_mask.fits'

    # Create a dictionary with the coordinate entries for the header
    hdr_coords = {}
    for key in lime.COORD_ENTRIES:
        if key in hdr:
            hdr_coords[key] = hdr[key]

    # Run the task
    lime.spatial_mask_generator(SII_image, 'percentile', percentile_array, mask_ref='S2_6716A_b', output_address=mask_file,
                                show_plot=True, fits_header=hdr_coords)

