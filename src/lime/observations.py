import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from astropy.io import fits
from collections import UserDict

from lime.tools import extract_fluxes, normalize_fluxes, ProgressBar, check_units, extract_wcs_header, \
    parse_unit_convertion

from lime.inference.detection import FeatureDetection
from lime.plotting.plots import SpectrumFigures, SampleFigures, CubeFigures
from lime.plotting.plots_interactive import SpectrumCheck, CubeCheck, SampleCheck
from lime.plotting.bokeh_plots import BokehFigures
from lime.io import _LOG_EXPORT_RECARR, save_frame, LiMe_Error, check_file_dataframe, check_file_array_mask, load_frame, lime_cfg

from lime.transitions import Line
from lime.archives.read_fits import OpenFits
from lime.workflow import SpecTreatment, CubeTreatment, SpecRetriever

# Log variable
_logger = logging.getLogger('LiMe')

try:
    import mplcursors
    mplcursors_check = True
except ImportError:
    mplcursors_check = False

if mplcursors_check:
    from mplcursors._mplcursors import _default_annotation_kwargs as popupProps
    popupProps['bbox']['alpha'] = 0.9


def review_sample_levels(log, id_name, file_name, id_level="id", file_level="file", line_level="line"):

    # If single index we make 2 [id, file, line] or 3 [id, line]
    # If multi-index, we don't do anything

    # Check if multi-index dataframe
    if not isinstance(log.index, pd.MultiIndex):

        # Name line level
        log.index.name = line_level

        # Add id level
        log[id_level] = id_name
        log.set_index(id_level, append=True, inplace=True)

        # Add file level
        if file_name is not None:
            log[file_level] = file_name
            log.set_index(file_level, append=True, inplace=True)

        # Sort to the default order
        user_levels = [id_level, line_level] if file_name is None else [id_level, file_level, line_level]
        log = log.reorder_levels(user_levels)

    return log


def mask_bad_entries(name_arr, value_arr, mask_check, pixel_mask, pixel_target_value, output_pixel_mask):

    mask_arr = np.ma.masked_array(value_arr, mask=pixel_mask) if mask_check else value_arr

    if np.isnan(pixel_target_value):
        idcs_target = np.isnan(mask_arr)
    elif np.isinf(pixel_target_value):
        idcs_target = np.isinf(mask_arr)
    else:
        raise LiMe_Error(f'Pixel target value "{pixel_target_value}" is not recognized')

    target_check = np.any(idcs_target)

    if target_check:
        message = f'The input {name_arr} array contains "{pixel_target_value}" entries'
        if name_arr == 'wave':
            message += f' you need to manually remove the invalid wavelength values'
            raise LiMe_Error(message)
        else:
            message += f' not included on the input pixel_mask array' if mask_check else  ' a pixel_mask will be generated automatically'
            _logger.info(message)

        # Create the mask
        if mask_check is False:

            # First initiation
            output_pixel_mask = np.zeros(value_arr.shape).astype(bool) if output_pixel_mask is None else output_pixel_mask

            # Assign values
            output_pixel_mask[idcs_target] = True

    return output_pixel_mask


def check_inputs_arrays(wave, flux, err_flux, pixel_mask, lime_object):

    # Check there is a mask
    mask_check = True if pixel_mask is not None else False

    # Automatic pixel mask
    output_pixel_mask = None

    # Loop through the observation properties
    for i, items in enumerate(locals().items()):
        name_arr, value_arr = items

        if i < 4:
            if value_arr is not None:

                # Check numpy array dimensions
                if isinstance(value_arr, np.ndarray):

                    dimensions = len(value_arr.shape)
                    spec_check = dimensions == 1 and (isinstance(lime_object, Spectrum) or i == 0)
                    cube_type = dimensions == 3 and isinstance(lime_object, Cube)

                    if not spec_check and not cube_type:
                        raise LiMe_Error(f'The dimensions of the input {name_arr} are {dimensions}.\n'
                                         f'LiMe only recognizes 1D arrays for the wavelength array, \n'
                                         f'1D flux arrays for the Spectrum objects \n'
                                         f'and 3D flux arrays Cube objects.')
                else:
                    raise LiMe_Error(f'The input {name_arr} array must be numpy array. '
                                     f'The input variable type is a {type(value_arr)}:{value_arr}')

                # Check for unmasked nan and inf entries
                if i < 3:

                    # Ignore the wavelength array for cube data)
                    if not (i == 0 and isinstance(lime_object, Cube)):
                        output_pixel_mask = mask_bad_entries(name_arr, value_arr, mask_check, pixel_mask, np.nan, output_pixel_mask)
                        output_pixel_mask = mask_bad_entries(name_arr, value_arr, mask_check, pixel_mask, np.inf, output_pixel_mask)

            else:
                if i <= 1:
                    _logger.warning(f'No input array for the observation {name_arr} was provided.')

    # Assign the output mask
    output_pixel_mask = pixel_mask if mask_check else output_pixel_mask

    return output_pixel_mask


def check_spectra_arrays(observation):

    if np.all(observation.flux.mask):
        _logger.critical(f'All the input {observation.__class__.__name__} pixels are masked. Please check that only bad '
                         f'pixels entries are masked (in numpy arrays flux_arr[pixel_mask] = bad_entries)')

    if observation.err_flux is not None:
        if observation.err_flux.data.sum() == 0:
            _logger.warning(f'All the {observation.__class__.__name__} flux uncertainty entries are 0.'
                            f' This can cause some measurements to fail.')

    return

def check_redshift_norm(redshift, norm_flux, flux_array, units_flux, norm_factor=1, min_flux_scale=0.001, max_flux_scale=1e50):

    if (redshift is None) or np.isnan(redshift) or np.isinf(redshift):
        _logger.warning(f'No redshift provided for the spectrum. Assuming local universe observation (z = 0)')
        redshift = 0

    if redshift < 0:
        _logger.warning(f'Input spectrum redshift has a negative value: z = {redshift}')

    if norm_flux is None:
        if units_flux.scale == 1:
            mean_flux = np.nanmean(flux_array)
            if mean_flux < min_flux_scale:
                if mean_flux <= 0:
                    norm_flux = np.power(10, np.floor(np.log10(mean_flux-np.nanmin(flux_array))) - norm_factor)
                else:
                    norm_flux = np.power(10, np.floor(np.log10(mean_flux)) - norm_factor)
                _logger.info(f'The observation does not include a normalization but the mean flux value is '
                             f'below {min_flux_scale}. The flux will be automatically normalized by {norm_flux}.')
            else:
                norm_flux = 1
        else:
            norm_flux = 1

    return redshift, norm_flux


def check_sample_input_files(log_list, file_list, page_list, id_list):

    # Confirm all the files have the same address
    for key, value in {'log_list': log_list, 'file_list': file_list, 'page_list': page_list}.items():
        if value is not None:
            if not (len(id_list) == len(value)):
                raise LiMe_Error(f'The length of the {key} must be the same of the input "id_list".')

    if log_list is None and file_list is None:
        raise LiMe_Error(f'To define a sample, the user must provide alongside an "id_list" a "log_list" and/or a '
                         f'"file_list".')

    return


def check_sample_file_parser(source, fits_reader, load_function, default_load_function):

    # Assign input load function
    if load_function is not None:
        output = None, load_function

    # Assign the
    elif (source is not None) and (fits_reader is not None):
        output = fits_reader, None

    # No load function nor instrument
    else:
        raise LiMe_Error(f'To create a Sample object you need to provide "load_function" or provide a "instrument" '
                         f'supported by LiMe')

    return output


def check_sample_levels(levels, necessary_levels=("id", "file")):

    for comp_level in necessary_levels:
        if comp_level not in levels:
            _logger.warning(f'Input log levels do not include a "{comp_level}". This can cause issues with LiMe functions')

    return


def cropping_spectrum(crop_waves, input_wave, input_flux, input_err, pixel_mask):

    if crop_waves is not None:

        idx_min = np.searchsorted(input_wave, crop_waves[0]) if crop_waves[0] != 0 else 0
        idx_max = np.searchsorted(input_wave, crop_waves[1]) if crop_waves[1] != -1 else None

        idcs_crop = (idx_min, idx_max)
        input_wave = input_wave[idcs_crop[0]:idcs_crop[1]]

        # Spectrum
        if len(input_flux.shape) == 1:
            input_flux = input_flux[idcs_crop[0]:idcs_crop[1]]
            if input_err is not None:
                input_err = input_err[idcs_crop[0]:idcs_crop[1]]

        # Cube
        elif len(input_flux.shape) == 3:
            input_flux = input_flux[idcs_crop[0]:idcs_crop[1], :, :]
            if input_err is not None:
                input_err = input_err[idcs_crop[0]:idcs_crop[1], :, :]

        # Not recognized
        else:
            raise LiMe_Error(f'The dimensions of the input flux are {input_flux.shape}. LiMe only recognized flux 1D '
                             f'arrays for Spectrum objects and 3D arrays for Cube objects')

        if pixel_mask is not None:
            pixel_mask = pixel_mask[idcs_crop[0]:idcs_crop[1]]

    return input_wave, input_flux, input_err, pixel_mask


def spec_normalization_masking(input_wave, input_flux, input_err, pixel_mask, redshift, norm_flux):

    # Wavelength
    wave = input_wave
    wave_rest = None if wave is None else input_wave / (1 + redshift)

    # Flux
    flux = input_flux
    err_flux = None if input_err is None else input_err

    # Normalization
    flux = flux if flux is None else flux/norm_flux
    err_flux = err_flux if err_flux is None else err_flux/norm_flux

    # Add all False mask if none provided
    pixel_mask = pixel_mask if pixel_mask is not None else np.zeros(flux.shape).astype(bool)

    # Wavelength masking 1D arrays
    if len(pixel_mask.shape) == 1:
        wave = np.ma.masked_array(wave, pixel_mask)
        wave_rest = np.ma.masked_array(wave_rest, pixel_mask)

    # Cube wavelength masking (all flux entries in one plane must be masked)
    else:
        wave_mask = np.all(pixel_mask, axis=(1, 2))
        wave = np.ma.masked_array(wave, wave_mask)
        wave_rest = np.ma.masked_array(wave_rest, wave_mask)

    # Spectrum or Cube spectral masking
    flux = np.ma.masked_array(flux, pixel_mask)
    err_flux = None if err_flux is None else np.ma.masked_array(err_flux, pixel_mask)

    return wave, wave_rest, flux, err_flux


class Spectrum:

    """
    Long-slit spectrum container with utilities for fitting, plotting, and retrieving measurements.

    A :class:`Spectrum` holds wavelength, flux, and optional uncertainty, arrays for a
    single long-slit observation.

    The user can provide a flux normalization (otherwise the algorithm will compute one if the input flux median <0.0001),
    pixel masking (to exclude bad pixels), and the observation redshift (if none is provided, it is assumed z = 0),
    wavelength/flux units, and instrumental resolving power.

    Parameters
    ----------
    input_wave : numpy.ndarray, optional
        Observed frame wavelength array.
    input_flux : numpy.ndarray, optional
        Flux array aligned with ``input_wave``.
    input_err : numpy.ndarray, optional
        1σ flux uncertainty array (same shape and units as ``input_flux``).
    redshift : float, optional
        Observation redshift ``z``.
    norm_flux : float, optional
        Flux normalization factor. Useful when flux magnitudes are very small; applied
        internally for fitting and removed in reported measurements.
    crop_waves : tuple or numpy.ndarray, optional
        Two-element ``(min, max)`` wavelength range used to crop the input arrays.
    res_power : float or numpy.ndarray, optional
        Instrument resolving power :math:`R = \\lambda/\\Delta\\lambda`. If provided,
        it can be used to compute and apply an instrumental broadening correction
        (``sigma_instr``) during analysis.
    units_wave : str, optional
        Wavelength units. Accepts any valid `Astropy units string
        <https://docs.astropy.org/en/stable/units/#module-astropy.units>`_,
        such as ``"Angstrom"``, ``"nm"``, ``"um"``, ``"mm"``, ``"cm"``, or ``"Hz"``.
        Default is ``"Angstrom"`` (equivalent to ``"AA"``, ``"A"``).
    units_flux : str, optional
        Flux units. Accepts any valid `Astropy units string
        <https://docs.astropy.org/en/stable/units/#module-astropy.units>`_,
        such as ``"erg / (s cm2 Angstrom)"``, ``"erg / (s cm2 Hz)"``, ``"Jy"``,
        ``"mJy"``, or ``"nJy"``.
        Default is ``"erg / (s cm2 Angstrom)"`` (equivalent to ``"FLAM"``).
    pixel_mask : numpy.ndarray of bool, optional
        Boolean mask with **True for pixels to exclude** from measurements (same length
        as ``input_wave``).
    id_label : str, optional
        Identifier for this spectrum (e.g., object name).
    review_inputs : bool, optional
        If ``True`` (default), validate and assign inputs via ``_set_attributes`` on init.

    Attributes
    ----------
    label : str or None
        Canonical internal name for the spectrum (may be derived from ``id_label``).
    wave : numpy.ndarray
        Observed‐frame wavelength array (after any cropping).
    wave_rest : numpy.ndarray
        Rest-frame wavelength array, if ``redshift`` is set.
    flux : numpy.ndarray
        Flux array (after any normalization handling).
    err_flux : numpy.ndarray or None
        1σ flux uncertainty.
    cont, cont_std : numpy.ndarray or None
        Continuum and its scatter (filled by downstream steps if applicable).
    frame : pandas.DataFrame or None
        Internal per-pixel frame, if/when constructed.
    redshift, norm_flux, res_power : float or None
        Stored metadata as described in *Parameters*.
    units_wave, units_flux : str
        Stored unit labels.

    Analysis helpers
    ----------------
    fit : :class:`lime.workflow.SpecTreatment`
        Fitting interface (line/profile fitting, corrections, etc.).
    infer : :class:`lime.workflow.FeatureDetection`
        Feature detection utilities.
    retrieve : :class:`lime.workflow.SpecRetriever`
        Tools for retreiven spectrum data.

    Plotting
    --------
    plot : :class:`lime.plots.SpectrumFigures`
        Matplotlib figures.
    check : :class:`lime.plots.SpectrumCheck`
        Interactive figures to review/modify the input data.
    bokeh : :class:`lime.plots.BokehFigures`
        Bokeh figures.

    Notes
    -----
    - If flux magnitudes are extremely small, set ``norm_flux`` to rescale the spectrum
      and ensure numerical stability during fitting. LiMe automatically removes this
      normalization from the final reported measurements. Similarly, if the wavelength
      array varies only by a few decimal places, the fitting routines may fail due to
      insufficient numerical precision.
    - ``pixel_mask`` follows NumPy’s masking convention: ``True`` entries mark pixels
      **to be excluded** from fitting and measurements.
    - Default units are ``"AA"`` (Angstrom) for wavelength and ``"FLAM"`` (erg s⁻¹ cm⁻² Å⁻¹)
      for flux. Ensure that your input arrays are consistent with the declared units.


    Examples
    --------
    Create a spectrum with uncertainties and a pixel mask:

    >>> spec = Spectrum(input_wave=wave, input_flux=flux, input_err=err,
    ...                 redshift=0.0132, units_wave="AA", units_flux="FLAM")

    Apply a normalization for very small fluxes:

    >>> spec = Spectrum(input_wave=wave, input_flux=flux, redshift=0, norm_flux=1e-16)

    Provide instrument resolving power for sigma correction:

    >>> spec = Spectrum(input_wave=wave, input_flux=flux, redshift=0, res_power=6000)
    """

    # File manager for a Cube created from an observation file
    _fitsMgr = None

    def __init__(self, input_wave=None, input_flux=None, input_err=None, redshift=None, norm_flux=None, crop_waves=None,
                 res_power=None, units_wave='AA', units_flux='FLAM', pixel_mask=None, id_label=None, review_inputs=True):

        # Class attributes
        self.label = None
        self.wave = None
        self.wave_rest = None
        self.flux = None
        self.err_flux = None
        self.cont = None
        self.cont_std = None

        self.frame = None

        self.redshift = None
        self.norm_flux = None
        self.res_power = None
        self.units_wave = None
        self.units_flux = None

        # Treatments objects
        self.fit = SpecTreatment(self)
        self.infer = FeatureDetection(self)
        self.retrieve = SpecRetriever(self)

        # Plotting objects
        self.plot = SpectrumFigures(self)
        self.check = SpectrumCheck(self)
        self.bokeh = BokehFigures(self)

        # Review and assign the attibutes data
        if review_inputs:
            self._set_attributes(input_wave, input_flux, input_err, redshift, norm_flux, crop_waves, res_power,
                                 units_wave, units_flux, pixel_mask, id_label)

        return

    @classmethod
    def from_cube(cls, cube, idx_j, idx_i, label=None):

        """
        Create a :class:`~lime.Spectrum` from a :class:`~lime.Cube` spaxel

        This class method extracts a one-dimensional spectrum (wavelength, flux, and
        associated metadata) from a LiMe cube at the specified pixel
        coordinates (these are the numpy arry coordiantes).

        Parameters
        ----------
        cube : lime.Cube
            Parent LiMe cube object containing 3D arrays of flux and wavelength data.
        idx_j : int
            Spatial pixel index along the cube’s Y-axis (row).
        idx_i : int
            Spatial pixel index along the cube’s X-axis (column).
        label : str, optional
            Identifier label for the extracted spectrum (e.g., ``"spaxel_45_32"``).

        Returns
        -------
        Spectrum
            A :class:`~lime.transitions.Spectrum` instance representing the 1D spectrum
            at the specified spatial position.

        Notes
        -----
        - The extracted spectrum inherits:
          * ``flux`` and optional ``err_flux`` from ``cube.flux`` and ``cube.err_flux``.
          * ``wave`` and ``wave_rest`` from the cube’s wavelength arrays, applying the same flux mask.
          * ``redshift``, ``norm_flux``, ``res_power``, ``units_wave``, and ``units_flux`` directly from the parent cube metadata.
        - The resulting object is initialized with ``review_inputs=False`` to avoid re-validation of already-processed arrays.
        - The method returns a new :class:`Spectrum` that can be analyzed or fitted independently of the cube.
        - For more information about masked arrays, see `numpy.ma.MaskedArray <https://numpy.org/doc/stable/reference/maskedarray.generic.html>`_.

        Examples
        --------
        Extract a single-spaxel spectrum from a cube:

        >>> spec = Spectrum.from_cube(cube, idx_j=45, idx_i=32, label="spaxel_45_32")
        >>> spec.plot.show_spectrum()
        """

        # Load parent classes
        spec = cls(review_inputs=False)

        # Class attributes
        spec.label = label
        spec.flux = cube.flux[:, idx_j, idx_i]
        # from matplotlib import pyplot as plt
        # fig, ax = plt.subplots()
        # ax.imshow(cube.err_flux[500, :, :].data)
        # plt.show()

        spec.err_flux = None if cube.err_flux is None else cube.err_flux[:, idx_j, idx_i]
        spec.norm_flux = cube.norm_flux
        spec.redshift = cube.redshift
        spec.frame = pd.DataFrame(np.empty(0, dtype=_LOG_EXPORT_RECARR))
        spec.res_power = cube.res_power
        spec.units_wave = cube.units_wave
        spec.units_flux = cube.units_flux

        # Use the flux mask for the wavelength
        spec.wave = np.ma.MaskedArray(cube.wave.data, spec.flux.mask)
        spec.wave_rest = np.ma.MaskedArray(cube.wave_rest.data, spec.flux.mask)

        # Check the input arrays
        check_spectra_arrays(spec)

        return spec

    @classmethod
    def from_file(cls, fname, instrument, redshift=None, norm_flux=None, crop_waves=None, res_power=None,
                  units_wave=None, units_flux=None, pixel_mask=None, id_label=None, wcs=None, **kwargs):

        """
        Create a :class:`~lime.transitions.Spectrum` instance from an observational FITS file or .txt file.

        This constructor reads a 1D spectroscopic observation from a supported instrument or survey
        and converts it into a fully initialized :class:`Spectrum` object. The function automatically
        interprets the file structure from the instrument template.

        To view the list of supported instruments and their configurations, use :func:`show_instrument_cfg`.

        Parameters
        ----------
        fname : str or pathlib.Path
            Path to the observational FITS file to read.
        instrument : str
            Name of the instrument or survey. The value is case-insensitive and
            is automatically converted to lowercase.
            Currently supported options include:
            ``"nirspec"``, ``"isis"``, ``"osiris"``, and ``"sdss"``.
        redshift : float, optional
            Source or observation redshift.
        norm_flux : float, optional
            Flux normalization factor. If provided, the spectrum is scaled internally
            for fitting stability, and the normalization is removed from the output
            measurements.
        crop_waves : tuple or numpy.ndarray, optional
            Two-element ``(min, max)`` wavelength range to crop the extracted spectrum.
        res_power : float or numpy.ndarray, optional
            Instrument resolving power (R = λ/Δλ). Used to compute the instrumental
            broadening correction in subsequent analysis.
        units_wave : str, optional
            Wavelength units. Accepts any valid `Astropy units string
            <https://docs.astropy.org/en/stable/units/#module-astropy.units>`_,
            such as ``"Angstrom"``, ``"nm"``, or ``"um"``.
            Default is determined automatically from the instrument configuration.
        units_flux : str, optional
            Flux units. Accepts any valid `Astropy units string
            <https://docs.astropy.org/en/stable/units/#module-astropy.units>`_,
            such as ``"erg / (s cm2 Angstrom)"``, ``"Jy"``, or ``"mJy"``.
            Default is determined automatically from the instrument configuration.
        pixel_mask : numpy.ndarray of bool or list, optional
            Boolean mask or a list of flux criteria used to exclude pixels from
            measurements.
            For example, ``[np.nan, "negative"]`` masks pixels with NaN values or
            negative fluxes in the input file.
        id_label : str, optional
            Identifier label for the resulting :class:`Spectrum` object.
        wcs : object, optional
            Optional WCS information for spatially-calibrated FITS files.
        **kwargs
            Additional keyword arguments passed directly to the :class:`Spectrum`
            initializer.

        Returns
        -------
        Spectrum
            A fully initialized :class:`Spectrum` object containing
            wavelength, flux, uncertainty (if available), and metadata from the FITS file.

        Notes
        -----
        - The method automatically detects the appropriate *.fits* file parser based on the
          ``instrument`` keyword.
        - For text files (``instrument="text"``), the ``**kwargs`` arguments are passed directly to
          the `numpy.loadtxt <https://numpy.org/doc/stable/reference/generated/numpy.loadtxt.html>`_
          function.
        - Flux units and wavelength calibration are derived from the instrument
          configuration and FITS. This assumes a standard calibration pipeline. Please check the units match your
          observation.
        - The user can override any automatically instrument parameter by explicitly passing the corresponding argument.

        Examples
        --------
        Load an OSIRIS FITS spectrum and set the observation redshift:

        >>> spec = Spectrum.from_file("osiris_obs.fits", instrument="osiris", redshift=0.013)

        Load an SDSS spectrum and mask invalid pixels:

        >>> spec = Spectrum.from_file("spec-12345.fits", instrument="sdss",
        ...                            pixel_mask=[np.nan, "negative"])

        Apply normalization and restrict the wavelength range:

        >>> spec = Spectrum.from_file("isis_star.fits", instrument="isis",
        ...                            norm_flux=1e-16, crop_waves=(4000, 7000))
        """

        # Create file manager object to administrate the file source and observation properties
        cls._fitsMgr = OpenFits(fname, instrument, cls.__name__)

        # Load the scientific data from the file
        fits_args = cls._fitsMgr.parse_data_from_file(cls._fitsMgr.file_address, pixel_mask, **kwargs)

        # Update the file parameters with the user parameters
        input_args = dict(redshift=redshift, norm_flux=norm_flux, crop_waves=crop_waves, res_power=res_power,
                            units_wave=units_wave, units_flux=units_flux, id_label=id_label, wcs=wcs)

        if cls._fitsMgr.spectrum_check:
            input_args.pop('wcs')

        input_args = {**fits_args, **{k: v for k, v in input_args.items() if v is not None}}

        # Create the LiMe object
        return cls(**input_args)

    @classmethod
    def from_survey(cls, target_id, survey, mask_flux_entries=None, **kwargs):

        """

        This method creates a lime.Spectrum object from a survey observational (.fits) file. The user needs to provide an
        object ID alongside the calague organization labels to identify the file.

        Currently, this method supports the DESI survey. This method will lower case the input survey name.

        The user can include list of pixel values to generate a mask from the input file flux entries. For example, if the
        user introduces [np.nan, 'negative'] the output spectrum will mask np.nan entries and negative fluxes.

        This method provides the arguments necesary to create the LiMe.Spectrum object. However, the user should provide
        the indexation values to locate the file on the survey. For example, for the DESI survey these would be the
        catalogue (i.e. healpix), program (i.e. dark) and release (fuji).

        :param file_address: Input object ID label.
        :type file_address: str

        :param survey: Input object survey name
        :type survey: str

        :param mask_flux_entries: List of pixel values to mask from flux array
        :type mask_flux_entries: list

        :param kwargs: Survey indexation arguments for the object

        :return: lime.Spectrum

        """

        # Create file manager object to administrate the file source and observation properties
        cls._fitsMgr = OpenFits(target_id, survey, cls.__name__)

        # Load the scientific data from the file
        fits_args = cls._fitsMgr.parse_data_from_url(cls._fitsMgr.file_address, mask_flux_entries, **kwargs)

        # Create the LiMe object
        return cls(**fits_args)

    def _set_attributes(self, input_wave, input_flux, input_err, redshift, norm_flux, crop_waves, res_power, units_wave,
                        units_flux, pixel_mask, label):

        # Class attributes
        self.label = label

        # Review the arrays data
        pixel_mask = check_inputs_arrays(input_wave, input_flux, input_err, pixel_mask, self)

        # Checks units
        self.units_wave, self.units_flux = check_units(units_wave, units_flux)

        # Check redshift and normalization
        self.redshift, self.norm_flux = check_redshift_norm(redshift, norm_flux, input_flux, self.units_flux)

        # Crop the input spectrum if necessary
        input_wave, input_flux, input_err, pixel_mask = cropping_spectrum(crop_waves, input_wave, input_flux, input_err,
                                                                          pixel_mask)

        # Normalization and masking
        self.wave, self.wave_rest, self.flux, self.err_flux = spec_normalization_masking(input_wave, input_flux,
                                                                                         input_err, pixel_mask,
                                                                                         self.redshift, self.norm_flux)

        # Generate empty dataframe to store measurement use cwd as default storing folder # TODO we are not using this
        self.frame = pd.DataFrame(np.empty(0, dtype=_LOG_EXPORT_RECARR))

        # Set the instrumental sigma correction
        self.res_power = res_power

        # Review the final data
        check_spectra_arrays(self)

        return

    def unit_conversion(self, wave_units_out=None, flux_units_out=None, norm_flux=None):

        """
        Convert the spectrum dispersion, energy density, and energy uncertainty units of the .

        This method updates the internal data arrays of the :class:`~lime.Spectrum` instance to new physical units.
        Conversions are handled by the `Astropy Units module
        <https://docs.astropy.org/en/stable/units/>`_. The function will remove the existing normalization and apply
        the new ``norm_flux`` if provided.

        Parameters
        ----------
        wave_units_out : str or astropy.units.Unit, optional
            Target wavelength units. Accepts any valid `Astropy unit string
            <https://docs.astropy.org/en/stable/units/#module-astropy.units>`_ (e.g.,
            ``"Angstrom"``, ``"nm"``, ``"um"``, ``"Hz"``) or an ``astropy.units.Unit``
            object. If ``None``, the current wavelength units are preserved.
        flux_units_out : str or astropy.units.Unit, optional
            Target flux units. Accepts any valid Astropy unit string or
            ``astropy.units.Unit`` object. Common shortcuts include:
            ``"FLAM"`` (erg s⁻¹ cm⁻² Å⁻¹), ``"FNU"`` (erg s⁻¹ cm⁻² Hz⁻¹),
            ``"PHOTLAM"`` (photon s⁻¹ cm⁻² Å⁻¹), and ``"PHOTNU"`` (photon s⁻¹ cm⁻² Hz⁻¹).
            Lowercase equivalents (``"flam"``, ``"fnu"``, etc.) are also accepted.
            If ``None``, the flux units are preserved.
        norm_flux : float, optional
            Flux normalization factor to apply after conversion. If provided,
            the flux and uncertainty arrays are scaled accordingly, and the new
            normalization is stored in ``self.norm_flux``.

        Returns
        -------
        None
            The method modifies the current :class:`Spectrum` instance in place,
            updating the arrays:
            ``wave``, ``wave_rest``, ``flux``, ``err_flux``,
            and the attributes ``units_wave``, ``units_flux``, and ``norm_flux``.

        Examples
        --------
        Convert wavelength from (current) Angstrom to nanometers:

        >>> spec.unit_conversion(wave_units_out="nm")

        Convert flux from Fλ to Fν and apply normalization:

        >>> spec.unit_conversion(flux_units_out="FNU", norm_flux=1e-16)

        Using Astropy unit objects directly:

        >>> import astropy.units as u
        >>> spec.unit_conversion(wave_units_out=u.um, flux_units_out=u.Jy)
        """

        # Extract the new values
        wave_units_out, flux_units_out, output_wave, output_flux, output_err, pixel_mask = parse_unit_convertion(self,
                                                                                                                 wave_units_out, flux_units_out)

        # Reassign the units and normalization
        self.units_wave, self.units_flux = check_units(wave_units_out, flux_units_out)
        self.redshift, self.norm_flux = check_redshift_norm(self.redshift, norm_flux, output_flux, self.units_flux)
        self.wave, self.wave_rest, self.flux, self.err_flux = spec_normalization_masking(output_wave, output_flux,
                                                                                         output_err, pixel_mask,
                                                                                         self.redshift, self.norm_flux)

        return

    def save_frame(self, fname, page='FRAME', param_list='all', header=None, column_dtypes=None,
                   safe_version=True, skip_failed=False):

        """
        Save the spectrum line measurements to disk in one of several supported formats.

        This method exports the current spectrum’s measurements to a table file. The output format is inferred
        from the filename extension. Supported formats include plain text tables, .fits, ASDF trees, and Excel sheets and
        pdf files.

        Parameters
        ----------
        fname : str or pathlib.Path
            Destination file path. The extension determines the output format.
            Supported extensions are ``.txt``, ``.fits``, ``.asdf``, ``.pdf``, and ``.xlsx``.
        page : str, optional
            Name of the HDU (for FITS) or sheet (for Excel) where the data will be
            written. Default is ``"FRAME"``.
        param_list : list or {"all"}, optional
            List of parameter columns to include in the output. If ``"all"``, all available measurements are written. Default is ``"all"``.
        header : dict, optional
            Optional metadata dictionary to include in the output file. For FITS and ASDF formats, this is added to the primary header.
        column_dtypes : str, type, or dict, optional
            Conversion rule for the output record array used in FITS files.
            - If a string or type, the specified type is applied to all columns.
            - If a dictionary, keys or indices define column-specific types. See the `pandas.DataFrame.to_records` documentation for details: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_records.html
        safe_version : bool, optional
            If ``True`` (default), append the current LiMe version number as a footnote
            or header annotation in the exported log.
        skip_failed : bool, optional
            If ``True``, skip over measurements that failed or could not be saved
            (e.g., missing flux values). Default is ``False``.

        Returns
        -------
        None
            The file is written to disk. No value is returned.

        Examples
        --------
        Save the current spectrum’s line measurements to a FITS table:

        >>> spec.save_frame("lines.fits", header={"OBSERVER": "V. Pérez"})

        Save selected columns to an Excel file:

        >>> spec.save_frame("lines.xlsx", page="FRAME",
        ...                 param_list=["id", "line", "flux", "err_flux"])

        Export to a text file and skip failed lines:

        >>> spec.save_frame("lines.txt", skip_failed=True)
        """

        # Meta parameters from the observations
        meta_params = {'LiMe':       lime_cfg['metadata']["version"],
                       'u_wave':     self.units_wave.to_string(),
                       'u_flux':     self.units_flux.to_string(),
                       'redshift':   self.redshift,
                       'id':         self.label}

        # Exclude the failed fittings from the output log
        out_put_frame = self.frame if skip_failed is False else self.frame.loc[self.frame['observations'] != 'No_errorbars']

        # Save the dataframe
        save_frame(fname, out_put_frame, page, param_list, header, column_dtypes=column_dtypes, safe_version=safe_version,
                   **meta_params)

        return

    def load_frame(self, fname, page='LINESFRAME'):

        """
        Load a lines frame into the spectrum.

        This method reads a previously saved line–measurement table (e.g., produced by
        :meth:`~lime.Spectrum.save_frame`) and attaches it to the current
        :class:`Spectrum` instance as the ``frame`` attribute. Any flux-dependent
        quantities are renormalized according to the spectrum’s current
        normalization (``norm_flux``).

        Parameters
        ----------
        fname : str or pathlib.Path
            Path to the input file containing the line measurements log.
            Supported formats include ``.txt``, ``.fits``, ``.asdf``, and ``.xlsx``.
        page : str, optional
            Name of the HDU (for FITS) or sheet (for Excel) from which to load the
            data. Default is ``"LINESFRAME"``.

        Returns
        -------
        None
            The method updates the current :class:`Spectrum` instance in place,
            assigning the loaded data to ``self.frame``.

        Notes
        -----
        - The file format is inferred automatically from its extension.
        - The table is normalized using the current spectrum’s ``norm_flux`` so that
          the loaded measurements remain consistent with the flux scaling in memory.
        - For FITS and Excel files, the ``page`` argument selects the HDU/sheet name
          to read from.
        - This method is complementary to :meth:`~lime.transitions.Spectrum.save_frame`.

        Examples
        --------
        Load a previously saved text line log:

        >>> spec.load_frame("lines.txt")

        Load from an Excel sheet named ``LINESFRAME``:

        >>> spec.load_frame("lines.fits", page="LINESFRAME")

        After loading, the line measurements are accessible via:

        >>> spec.frame.head()
        """

        # Load the log file if it is a log file
        log_df = check_file_dataframe(fname, ext=page)

        # Security checks:
        if log_df is not None and log_df.index.size > 0:
            line_list = log_df.index.values

            # Get the first line in the log
            line_0 = Line.from_transition(line_list[0], data_frame=log_df, norm_flux=self.norm_flux)

            # Confirm the lines in the log match the one of the spectrum
            if line_0.units_wave != self.units_wave:
                _logger.warning(f'Different units in the spectrum dispersion ({self.units_wave}) axis and the lines log'
                                f' in {line_0.units_wave[0]}')

            # Confirm all the log lines have the same units
            au_str = 'A' if line_0.units_wave == 'Angstrom' else str(line_0.units_wave)
            same_units_check = np.flatnonzero(np.core.defchararray.find(line_list.astype(str), au_str) != -1).size == line_list.size
            if not same_units_check:
                _logger.warning(f'The log has lines with different units')

            # Assign the log
            self.frame = log_df

        else:
            _logger.info(f'Log file with 0 entries ({fname})')

        return

    def update_redshift(self, redshift):

        """
        Update the spectrum's redshift and recompute its rest-frame wavelength.

        The normalization is preserved, and masked pixels remain unchanged.

        Parameters
        ----------
        redshift : float
            New redshift value to assign to the spectrum. The value should represent
            the observed-to-rest-frame wavelength scaling factor
            (:math:`1 + z = \\lambda_\\mathrm{obs} / \\lambda_\\mathrm{rest}`).

        Returns
        -------
        None
            The method updates the spectrum in place, modifying the attributes:
            ``redshift``, ``wave_rest``.

        Notes
        -----
        - Internally, this method reuses the existing data arrays stored in
          ``self.wave``, ``self.flux``, and ``self.err_flux``.
        - The pixel mask (``flux.mask``) is preserved and applied consistently
          to the updated arrays.
        - The update is performed through :func:`lime.spec_normalization_masking`,
          which handles normalization, masking, and wavelength conversion.
        - The normalization factor is fixed to unity (``norm_flux=1``) for this
          operation.

        Examples
        --------
        Update a spectrum to a new redshift:

        >>> spec.update_redshift(0.0145)
        >>> spec.wave_rest[:5]
        array([4932.1, 4932.9, 4933.8, 4934.6, 4935.5])

        The observed-frame wavelength array (``wave``) remains unchanged,
        while the rest-frame array (``wave_rest``) is updated accordingly.
        """

        # Check if it is a masked array
        input_wave = self.wave.data
        input_flux = self.flux.data
        input_err = self.err_flux.data
        pixel_mask = self.flux.mask

        # Normalization and masking
        self.redshift = redshift
        self.wave, self.wave_rest, self.flux, self.err_flux = spec_normalization_masking(input_wave, input_flux,
                                                                                         input_err, pixel_mask,
                                                                                         self.redshift, 1)

        return

    def line_detection(self, *args, **kwargs):

        raise LiMe_Error(f'The line_detection functionality has been moved an rebranded. Please use:\n'
                         f'Spectrum.infer.peaks_troughs()')
    
    def clear_data(self):

        """
        Clear the spectrum’s measurements frame.

        This method removes all entries from the internal ``frame`` attribute,
        effectively resetting the stored measurements while preserving the
        dataframe structure (columns and metadata).

        Returns
        -------
        None
            The method modifies the current :class:`Spectrum` instance in place.

        Notes
        -----
        - The operation is equivalent to reassigning ``self.frame = self.frame[0:0]``,
          which clears all rows but keeps column definitions intact.
        - Use this method to reset the spectrum’s measurement results before
          reprocessing or refitting without recreating the object.

        Examples
        --------
        >>> spec.frame.shape
        (25, 10)
        >>> spec.clear_data()
        >>> spec.frame.shape
        (0, 10)
        """

        self.frame = self.frame[0:0]

        return


class Cube:

    """
    Create a data cube for an integral-field spectroscopic observation.

    The `Cube` class represents a three-dimensional spectroscopic dataset, typically
    from an **integral field spectrograph (IFS)** observation. The cube combines a 1D
    wavelength axis with 2D spatial dimensions (x, y), producing a 3D flux array
    (wavelength × y × x). Optionally, a matching 3D uncertainty array and pixel mask
    can be provided.

    Parameters
    ----------
    input_wave : numpy.ndarray
        One-dimensional wavelength array in physical units.
    input_flux : numpy.ndarray
        Three-dimensional flux array (wavelength × y × x).
    input_err : numpy.ndarray, optional
        Three-dimensional uncertainty array (same shape as `input_flux`),
        in the same flux units.
    redshift : float, optional
        Redshift of the observed target. Used to compute the rest-frame wavelength axis.
    norm_flux : float, optional
        Flux normalization factor. Useful when flux magnitudes are very small to
        ensure numerical stability during line profile fitting. The normalization
        is removed from final reported measurements.
    crop_waves : array-like or tuple, optional
        Minimum and maximum wavelength limits to crop the cube.
    res_power : float or numpy.ndarray, optional
        Instrument resolving power (:math:`R = \\lambda / \\Delta\\lambda`), used
        to compute instrumental broadening corrections.
    units_wave : str or `astropy.units.Unit`, optional
        Wavelength units (default: ``"AA"`` for Angstroms). Accepts
        `Astropy unit strings <https://docs.astropy.org/en/stable/units/index.html>`_,
        e.g. ``"nm"``, ``"um"``, ``"Hz"``, ``"cm"``, ``"mm"``.
    units_flux : str or `astropy.units.Unit`, optional
        Flux units (default: ``"FLAM"`` → erg s⁻¹ cm⁻² Å⁻¹). Accepts any valid
        Astropy unit string, such as ``"FNU"``, ``"Jy"``, ``"mJy"``, or ``"nJy"``.
    pixel_mask : numpy.ndarray, optional
        Boolean 3D array (same shape as `input_flux`) marking pixels **to be excluded**
        from analysis. `True` values indicate excluded pixels.
    id_label : str, optional
        Identifier or label for the cube object.
    wcs : astropy.wcs.WCS, optional
        World Coordinate System object describing the spatial coordinates of the cube.
        See the `Astropy WCS documentation <https://docs.astropy.org/en/stable/wcs/index.html>`_.

    Attributes
    ----------
    fit : lime.cube.CubeTreatment
        Handler for fitting and measurement procedures.
    plot : lime.plots.CubeFigures
        Plotting interface for visualization.
    check : lime.check.CubeCheck
        Quality control and validation utilities.

    Notes
    -----
    - A pixel mask can be used to exclude bad or flagged spaxels from analysis.
    - The resolving power (`res_power`) enables automatic line-width corrections
      when fitting emission lines.
    - Unit conversions are managed via the `Astropy Units` framework.
    - If the flux values are very small (< 0.0001), consider setting a `norm_flux` value
      to improve numerical stability the fitting minimization.

    Examples
    --------
    Create a cube with a wavelength axis, flux data, uncertainty, redshift and WCS:

    >>> from astropy.wcs import WCS
    >>> cube = Cube(input_wave=wave, input_flux=flux, input_err=err_flux,
    ...             redshift=0.01, norm_flux=1e-17, wcs=WCS(header))

    Crop the wavelength range and specify custom units:

    >>> cube = Cube(input_wave=wave, input_flux=flux, redshift=0.01, crop_waves=(4800, 5100),
    ...             units_wave="nm", units_flux="Jy")
    """

    # File manager for a Cube created from an observation file
    _fitsMgr = None

    def __init__(self, input_wave=None, input_flux=None, input_err=None, redshift=None, norm_flux=None, crop_waves=None,
                 res_power=None, units_wave='AA', units_flux='FLAM', pixel_mask=None, id_label=None, wcs=None):

        # Review the 2_guides
        pixel_mask = check_inputs_arrays(input_wave, input_flux, input_err, pixel_mask, self)

        # Class attributes
        self.obj_name = id_label
        self.wave = None
        self.wave_rest = None
        self.flux = None
        self.err_flux = None
        self.res_power = res_power
        self.wcs = wcs

        # Treatments objects
        self.fit = CubeTreatment(self)

        # Plotting objects
        self.plot = CubeFigures(self)
        self.check = CubeCheck(self)

        # Checks units
        self.units_wave, self.units_flux = check_units(units_wave, units_flux)

        # Check redshift and normalization
        self.redshift, self.norm_flux = check_redshift_norm(redshift, norm_flux, input_flux, self.units_flux)

        # Start cropping the input spectrum if necessary
        input_wave, input_flux, input_err, pixel_mask = cropping_spectrum(crop_waves, input_wave, input_flux, input_err,
                                                                          pixel_mask)

        # Spectrum normalization, redshift and mask calculation
        self.wave, self.wave_rest, self.flux, self.err_flux = spec_normalization_masking(input_wave, input_flux,
                                                                                         input_err, pixel_mask,
                                                                                         self.redshift, self.norm_flux)
        # Review the final data
        check_spectra_arrays(self)

        return

    @classmethod
    def from_file(cls, file_address, instrument, redshift=None, norm_flux=None, crop_waves=None, res_power=None,
                  units_wave=None, units_flux=None, pixel_mask=None, wcs=None, id_label=None, **kwargs):

        """
        Create a :class:`lime.Cube` instance from an observational FITS file.

        This class method reads a 3D spectroscopic data cube from a FITS file and constructs
        a :class:`lime.Cube` object. The user must provide the file path and the name of the
        **instrument** or survey used for the observation.

        You can check the list of supported instruments and their configurations using:
        ``lime.show_instrument_cfg()``.

        The method automatically retrieves wavelength, flux, uncertainty (if available),
        normalization, and WCS information from the FITS file based on the instrument’s
        configuration. These default values can be overridden by passing explicit arguments
        such as ``redshift``, ``norm_flux``, or ``units_wave``.

        Users can also specify a 3D boolean ``pixel_mask`` to exclude unwanted pixels or
        spaxels from subsequent analysis.


        Parameters
        ----------
        file_address : str or pathlib.Path
            Path to the observational FITS file.
        instrument : str
            Instrument or survey name (e.g., ``"MUSE"``, ``"MANGA"``). The name is
            case-insensitive.
        redshift : float, optional
            Redshift of the observed target. Used to compute the rest-frame wavelength axis.
        norm_flux : float, optional
            Flux normalization factor. Useful when flux magnitudes are very small to
            improve numerical stability during line fitting. The normalization is
            removed in output measurements.
        crop_waves : array-like or tuple, optional
            Minimum and maximum wavelength limits to crop the cube before loading.
        res_power : float or numpy.ndarray, optional
            Instrument resolving power (:math:`R = \\lambda / \\Delta\\lambda`), used
            to compute instrumental broadening corrections.
        units_wave : str or `astropy.units.Unit`, optional
            Wavelength units. Default is None. Accepts
            `Astropy-compatible unit strings
            <https://docs.astropy.org/en/stable/units/index.html>`_
        units_flux : str or `astropy.units.Unit`, optional
            Energy density units. Default is None. Accepts `Astropy-compatible unit strings
            <https://docs.astropy.org/en/stable/units/index.html>`_
        pixel_mask : numpy.ndarray, optional
            Boolean 3D array (same shape as the flux cube) marking pixels **to be excluded**
            from analysis. ``True`` entries correspond to rejected pixels.
        wcs : astropy.wcs.WCS, optional
            World Coordinate System describing the spatial axes of the cube.
            See the `Astropy WCS documentation
            <https://docs.astropy.org/en/stable/wcs/index.html>`_.
        id_label : str, optional
            Identifier or label for the cube object.
        **kwargs
            Additional keyword arguments passed to the :class:`OpenFits.parse_data_from_file` function.

        Returns
        -------
        lime.Cube
            A fully constructed :class:`lime.Cube` object containing the wavelength array,
            flux cube, uncertainty data (if available), and WCS metadata.

        Notes
        -----
        - The method uses the :class:`lime.io.OpenFits` manager to interpret the FITS
          structure and extract instrument-specific metadata.
        - The instrument configuration defines which FITS extensions and data formats
          are read for wavelength, flux, error, and WCS.
        - The FITS file is read in memory; the original file is not modified.

        Examples
        --------
        Load a MUSE data cube from file:

        >>> cube = Cube.from_file("muse_cube.fits", instrument="MUSE")

        Load a MaNGA cube and apply a custom normalization and redshift:

        >>> cube = Cube.from_file("manga_cube.fits", instrument="MANGA",
        ...                       redshift=0.015, norm_flux=1e-16)

        Crop the wavelength range and set custom output units:

        >>> cube = Cube.from_file("muse_cube.fits", instrument="MUSE",
        ...                       crop_waves=(4800, 5100), units_wave="nm", units_flux="Jy")
        """

        # Create file manager object to administrate the file source and observation properties
        cls._fitsMgr = OpenFits(file_address, instrument, cls.__name__)

        # Load the scientific data from the file
        fits_args = cls._fitsMgr.parse_data_from_file(cls._fitsMgr.file_address, pixel_mask, **kwargs)

        # Update the file parameters with the user parameters
        input_args = dict(redshift=redshift, norm_flux=norm_flux, crop_waves=crop_waves, res_power=res_power,
                            units_wave=units_wave, units_flux=units_flux, id_label=id_label, wcs=wcs)

        # Update the parameters file parameters with the user parameters
        input_args = {**fits_args, **{k: v for k, v in input_args.items() if v is not None}}

        # Create the LiMe object
        return cls(**input_args)

    def spatial_masking(self, line, fname=None, bands=None, param='flux', contour_pctls=(90, 95, 99),
                        mask_label_prefix=None, header=None):

        """
        Generate a spatial binary mask for a given line.

        This method creates one or more **binary spatial masks** based on the flux or
        signal-to-noise distribution of a specified spectral line across the data cube.
        Each mask corresponds to a percentile contour level in the input parameter
        (e.g., total flux or S/N).

        The ``line`` argument identifies the target line. If no custom ``bands``
        are provided, the default bands database is used to locate the line and continuum
        wavelength intervals.

        The parameter used for the mask calculation is controlled by ``param``:
        - ``"flux"`` → integrates the flux over the line region.
        - ``"SN_line"`` → computes the line signal-to-noise ratio using the definition from `Rola et al. (1994) <https://ui.adsabs.harvard.edu/abs/1994A%26A...287..676R/abstract>`_.
        - ``"SN_cont"`` → computes the continuum signal-to-noise ratio from the adjacent continuum bands.

        The method generates multiple masks based on percentile thresholds defined in
        ``contour_pctls`` (e.g., 90, 95, 99). Higher percentiles correspond to smaller,
        brighter regions of emission. Masks can be saved as a multi-extension FITS file
        or returned as an ``astropy.io.fits.HDUList`` object.

        Parameters
        ----------
        line : str
            Label of the emission line to analyze (in `LiMe notation <https://lime-stable.readthedocs.io/en/latest/inputs/n_inputs2_line_labels.html>`_).
        bands : pandas.DataFrame, str, or pathlib.Path, optional
            Bands dataframe or file path defining the wavelength intervals for the line and continua. If not provided, the default LiMe bands database is used.
        param : {'flux', 'SN_line', 'SN_cont'}, optional
            Parameter for mask generation. Determines the property used to compute percentile contours:
              * ``'flux'`` — integrated flux over the emission-line band.
              * ``'SN_line'`` — emission-line signal-to-noise ratio (Rola et al. 1994).
              * ``'SN_cont'`` — continuum signal-to-noise ratio.

            Default is ``'flux'``.
        contour_pctls : array-like, optional
            Sorted percentile values for mask generation (in increasing order). Default is ``(90, 95, 99)``.
        fname : str or pathlib.Path, optional
            File path to save the output FITS file. If not provided, the function returns an :class:`astropy.io.fits.HDUList` object instead.
        mask_label_prefix : str, optional
            Prefix added to the FITS extension names for each mask. By default, masks are labeled as ``MASK_0``, ``MASK_1``, etc.
        header : dict, optional
            Dictionary of header metadata to include in the output FITS extensions.
            Keys correspond to mask names (e.g., ``'mask_0'``) or a global header.

        Returns
        -------
        astropy.io.fits.HDUList or None
            - Returns an HDUList containing the generated masks if ``fname=None``.
            - Writes the masks to a FITS file and returns ``None`` otherwise.

        Notes
        -----
        - Each mask corresponds to a percentile contour level of the input ``param`` image.
        - Pixels with all-NaN flux values across the wavelength dimension are automatically
          excluded from the masks.
        - If the FITS file output path is invalid, the function raises a :class:`LiMe_Error`.
        - The FITS header for each mask have the following keys:
            * ``PARAM`` — the parameter used (e.g., flux or SN_line)
            * ``PARAMIDX`` — percentile level
            * ``PARAMVAL`` — threshold parameter value
            * ``NUMSPAXE`` — number of unmasked spaxels

        - The output FITS file retains the cube’s spatial WCS coordinates using
          :func:`lime.io.extract_wcs_header`.

        Examples
        --------
        Generate flux-based spatial masks for Hα at 90th, 95th, and 99th percentiles:

        >>> hdul = cube.spatial_masking("H1_6563A", bands="default_bands.xlsx")

        Save the masks as a FITS file:

        >>> cube.spatial_masking("O3_5007A", fname="O3_masks.fits")

        Compute masks based on the line signal-to-noise ratio:

        >>> cube.spatial_masking("H1_4861A", param="SN_line", contour_pctls=[80, 90, 95])
        """

        # Check the function 2_guides
        contour_pctls = np.atleast_1d(contour_pctls)
        if not np.all(np.diff(contour_pctls) > 0):
            raise LiMe_Error(f'The mask percentiles ({contour_pctls}) must be in increasing order')
        inver_percentiles = np.flip(contour_pctls)

        if not param in ['flux', 'SN_line', 'SN_cont']:
            raise LiMe_Error(f'The mask calculation parameter ({param}) is not recognised. Please use "flux", "SN_line", "SN_cont"')

        # Line for the background image
        line_bg = Line.from_transition(line, data_frame=bands)

        # Get the band indexes
        idcsEmis, idcsCont = line_bg.index_bands(self.wave, self.redshift)
        signal_slice = self.flux[idcsEmis, :, :]
        signal_slice = signal_slice.data

        # Get indeces all nan entries to exclude them from the analysis
        idcs_all_nan = np.all(np.isnan(signal_slice.data), axis=0)

        # If not mask parameter provided we use the flux percentiles
        if param == 'flux':
            param = self.units_flux
            param_image = signal_slice.sum(axis=0)

        # S/N cont
        elif param == 'SN_cont':
            param_image = np.nanmean(signal_slice, axis=0) / np.nanstd(signal_slice, axis=0)

        # S/N line
        elif param == 'SN_line':
            n_pixels = np.sum(idcsCont)
            cont_slice = self.flux[idcsCont, :, :].data
            cont_slice = cont_slice.data

            Amp_image = np.nanmax(signal_slice, axis=0) - np.nanmean(cont_slice, axis=0)
            std_image = np.nanstd(cont_slice, axis=0)
            param_image = (np.sqrt(2 * n_pixels * np.pi) / 6) * (Amp_image / std_image)

        else:
            raise LiMe_Error(f'Parameter {param} is not recognized please use: "flux", "SN_line" or "SN_cont"')

        # Percentiles vector for the target parameter
        param_array = np.nanpercentile(param_image, inver_percentiles)

        # If minimum level not provided by user use lowest contour_level
        min_level = param_array[-1]

        # Containers for the mask parameters
        mask_dict = {}
        param_level = {}
        boundary_dict = {}

        # Loop throught the counter levels and compute the
        for i, n_levels in enumerate(param_array):

            # # Operation every element
            if i == 0:
                maParamImage = np.ma.masked_where((param_image >= param_array[i]) &
                                                  (param_image >= min_level),
                                                  param_image)

            else:
                maParamImage = np.ma.masked_where((param_image >= param_array[i]) &
                                                  (param_image < param_array[i - 1]) &
                                                  (param_image >= min_level),
                                                  param_image)

            if np.sum(maParamImage.mask) > 0:
                mask_dict[f'mask_{i}'] = maParamImage.mask & ~idcs_all_nan
                boundary_dict[f'mask_{i}'] = inver_percentiles[i]
                param_level[f'mask_{i}'] = param_array[i]

        # Use as HDU as container for the mask
        hdul = fits.HDUList([fits.PrimaryHDU()])

        # Recover coordinates from the wcs to store in the headers:
        hdr_coords = extract_wcs_header(self.wcs, drop_axis='spectral')

        for idx_region, region_items in enumerate(mask_dict.items()):
            region_label, region_mask = region_items

            # Metadata for the fits page
            hdr_i = fits.Header({'PARAM': param,
                                 'PARAMIDX': boundary_dict[region_label],
                                 'PARAMVAL': param_level[region_label],
                                 'NUMSPAXE': np.sum(region_mask)})

            # Add WCS information
            if hdr_coords is not None:
                hdr_i.update(hdr_coords)

            # Add user information
            if header is not None:
                page_hdr = header.get(f'{mask_label_prefix}{region_label}', None)
                page_hdr = header if page_hdr is None else page_hdr
                hdr_i.update(page_hdr)

            # Extension for the mask
            mask_ext = region_label if mask_label_prefix is None else f'{mask_label_prefix}{region_label}'

            # Mask HDU
            mask_hdu = fits.ImageHDU(name=mask_ext, data=region_mask.astype(int), ver=1, header=hdr_i)
            hdul.append(mask_hdu)

        # Output folder computed from the output address
        fname = Path(fname) if fname is not None else None

        # Return an array with the masks
        if fname is not None:
            if fname.parent.is_dir():
                hdul.writeto(fname, overwrite=True, output_verify='fix')
                output_func = None
            else:
                raise LiMe_Error(f'Mask could not be saved. Folder not found: {fname.parent.as_posix()}')

        # Return the hdul
        else:
            output_func = hdul

        return output_func

    def unit_conversion(self, wave_units_out=None, flux_units_out=None, norm_flux=None):

        """
        Convert the cube’s wavelength and/or flux units.

        This method updates the units of the cube’s wavelength axis and flux cube using
        the `Astropy Units module <https://docs.astropy.org/en/stable/units/index.html>`_.
        Both string representations (e.g., ``"AA"``, ``"Jy"``) and `astropy.units.Unit`
        objects are accepted.

        The conversion applies consistently across all spaxels in the data cube and
        automatically updates the internal attributes ``units_wave`` and ``units_flux``.
        The user may also specify a new flux normalization factor with ``norm_flux``,
        which rescales the flux and uncertainty arrays accordingly.

        Parameters
        ----------
        wave_units_out : str or astropy.units.Unit, optional
            Target wavelength units. Accepts any valid `Astropy unit string
            <https://docs.astropy.org/en/stable/units/#module-astropy.units>`_ (e.g.,
            ``"Angstrom"``, ``"nm"``, ``"um"``, ``"Hz"``) or an ``astropy.units.Unit``
            object. If ``None``, the current wavelength units are preserved.
        flux_units_out : str or astropy.units.Unit, optional
            Target flux units. Accepts any valid Astropy unit string or
            ``astropy.units.Unit`` object. Common shortcuts include:
            ``"FLAM"`` (erg s⁻¹ cm⁻² Å⁻¹), ``"FNU"`` (erg s⁻¹ cm⁻² Hz⁻¹),
            ``"PHOTLAM"`` (photon s⁻¹ cm⁻² Å⁻¹), and ``"PHOTNU"`` (photon s⁻¹ cm⁻² Hz⁻¹).
            Lowercase equivalents (``"flam"``, ``"fnu"``, etc.) are also accepted.
            If ``None``, the flux units are preserved.
        norm_flux : float, optional
            Flux normalization factor to apply after conversion. If provided,
            the flux and uncertainty arrays are scaled accordingly, and the new
            normalization is stored in ``self.norm_flux``.

        Notes
        -----
        - The function removes any normalization present on the energy density units, althought it will apply one if ``norm_flux=None`` and the mean flux is < 0.0001.
        - The conversion is handled using `Astropy’s Unit equivalencies <https://docs.astropy.org/en/stable/units/equivalencies.html>`_, ensuring
          consistent transformations between flux–density and wavelength/frequency units.
        - The conversion preserves the cube’s redshift and mask structure.

        Examples
        --------
        Convert the cube’s wavelength to nanometers:

        >>> cube.unit_conversion(wave_units_out="nm")

        Convert both wavelength and flux to frequency-based units:

        >>> cube.unit_conversion(wave_units_out="Hz", flux_units_out="FNU")

        Apply a normalization after the unit conversion

        >>> cube.unit_conversion(flux_units_out="FLAM", norm_flux=1e-16)
        """

        # Extract the new values
        wave_units_out, flux_units_out, output_wave, output_flux, output_err, pixel_mask = parse_unit_convertion(self,
                                                                                                                 wave_units_out, flux_units_out)

        # Reassign the units and normalization
        self.units_wave, self.units_flux = check_units(wave_units_out, flux_units_out)
        self.redshift, self.norm_flux = check_redshift_norm(self.redshift, norm_flux, output_flux, self.units_flux)
        self.wave, self.wave_rest, self.flux, self.err_flux = spec_normalization_masking(output_wave, output_flux,
                                                                                         output_err, pixel_mask,
                                                                                         self.redshift, self.norm_flux)

        return

    def get_spectrum(self, idx_j, idx_i, id_label=None):

        """
        Extract a single spaxel spectrum from the data cube.

        This method returns a :class:`lime.Spectrum` object corresponding to the
        spaxel located at the given spatial indices ``(idx_j, idx_i)`` in the cube.
        The extracted spectrum preserves the cube’s wavelength, flux normalization,
        redshift, instrumental resolution, and physical units.

        Parameters
        ----------
        idx_j : int
            Spatial index along the cube’s **y-axis** (array row coordinate).
        idx_i : int
            Spatial index along the cube’s **x-axis** (array column coordinate).
        id_label : str, optional
            Identifier for the extracted spectrum. If not provided, a default label
            is assigned based on the spaxel indices.

        Returns
        -------
        lime.Spectrum

        Notes
        -----
        - The coordiantes correspond to the indexes in the numpy array.
        - This method is a convenience wrapper around
          :meth:`lime.Spectrum.from_cube`.

        Examples
        --------
        Extract a single spaxel spectrum from a cube:

        >>> spec = cube.spectrum_from_indices(25, 30)

        """

        return Spectrum.from_cube(self, idx_j, idx_i, id_label)

    def export_spaxels(self, fname, mask_file, mask_list=None, log_ext_suffix='_SPEC', progress_output='bar'):

        """
        Export individual spaxel spectra to a ``fname`` from a cube based on spatial masks ``mask_file`` calculated
        in advance.

        This method extracts and saves the flux (and uncertainty if available) spectra
        for all spaxels included in one or more spatial masks. Each selected spaxel
        is stored as an individual table extension within a single multi-extension
        FITS file, allowing direct integration with subsequent LiMe analysis routines.

        The input masks can be provided either as:
          * A **FITS file** produced by :meth:`~lime.Cube.spatial_masking`, or
          * A **dictionary or array** containing binary mask data.

        Each FITS extension in the output corresponds to one spaxel, labeled by its
        2D position in the cube (e.g., ``'25-40_SPEC'``). The exported spectra
        retain the cube’s flux normalization, units, and coordinate metadata.

        Parameters
        ----------
        fname : str or pathlib.Path
            Path to the output FITS file where all spaxel spectra will be saved.
            The parent directory must already exist.
        mask_file : str, pathlib.Path, numpy.ndarray, or dict
            Input mask(s) specifying which spaxels to export. Supported inputs:
              * Path to a FITS file containing binary masks (e.g., from
                :meth:`~lime.Cube.spatial_masking`).
              * Numpy array of boolean masks.
              * Dictionary mapping mask labels to mask data arrays.
        mask_list : list of str, optional
            Subset of mask names to process from ``mask_file``. If not provided,
            all masks on ``mask_file`` are used.
        log_ext_suffix : str, optional
            Suffix appended to each FITS extension name. Default is ``"_LINELOG"``.
        progress_output : {'bar', 'counter', None}, optional
            Controls how progress is displayed during export:
              * ``'bar'`` — shows a progress bar (default).
              * ``'counter'`` — prints textual progress updates.
              * ``None`` — disables all progress messages.

        Returns
        -------
        None
            The function writes the extracted spaxel spectra to a FITS file at
            ``output_address``.

        Notes
        -----
        - Each exported spaxel is saved as a binary table HDU with fields:
            * ``flux`` — observed flux values (normalized).
            * ``flux_err`` — flux uncertainty values (if available).
        - The WCS spatial coordinates are added to each FITS extension header if
          the cube includes valid WCS metadata.
        - The FITS file can be directly reopened using :mod:`astropy.io.fits` for
          subsequent analysis or visualization.

        Examples
        --------
        Export all spaxel spectra from a given spatial mask file:

        >>> cube.export_spaxels("outputs/O3_spaxels.fits", mask_file="O3_masks.fits")

        Export only specific masks (e.g., ``mask_0`` and ``mask_1``):

        >>> cube.export_spaxels("outputs/O3_selected_spaxels.fits",
        ...                     mask_file="O3_masks.fits",
        ...                     mask_list=["mask_0", "mask_1"])

        Disable progress display during export:

        >>> cube.export_spaxels("outputs/O3_spaxels.fits", "O3_masks.fits",
        ...                     progress_output=None)
        """

        # Check if the mask variable is a file or an array
        mask_dict = check_file_array_mask(mask_file, mask_list)

        # Unpack mask dictionary
        mask_list = np.array(list(mask_dict.keys()))
        mask_data_list = list(mask_dict.values())

        # Checks for the data type
        err_check = False if self.err_flux is None else True
        # masked_check = False if np.ma.isMaskedArray(self.flux) is False else True

        # Check if the output log folder exists
        output_address = Path(fname)
        if not output_address.parent.is_dir():
            raise LiMe_Error(f'The folder of the output log file does not exist at {output_address}')

        # Determine the spaxels to treat at each mask
        total_spaxels, spaxels_dict = 0, {}
        for idx_mask, mask_data in enumerate(mask_data_list):
            spa_mask, hdr_mask = mask_data
            idcs_spaxels = np.argwhere(spa_mask)

            total_spaxels += len(idcs_spaxels)
            spaxels_dict[idx_mask] = idcs_spaxels

        # Spaxel counter to save the data everytime n_save is reached
        spax_counter = 0

        # HDU_container
        hdul = fits.HDUList([fits.PrimaryHDU()])

        # Header data
        if self.wcs is not None:
            hdr_coords = extract_wcs_header(self.wcs, drop_axis='spatial')
        else:
            hdr_coords = None

        # Loop through the masks
        n_masks = len(mask_list)
        for i in np.arange(n_masks):

            # Mask progress indexing
            mask_name = mask_list[i]
            idcs_spaxels = spaxels_dict[i]

            # Loop through the spaxels
            n_spaxels = idcs_spaxels.shape[0]
            pbar = ProgressBar(progress_output, f'{n_spaxels} spaxels')
            print(f'\n\nSpatial mask {i + 1}/{n_masks}) {mask_name} ({n_spaxels} spaxels)')
            for j in np.arange(n_spaxels):

                idx_j, idx_i = idcs_spaxels[j]
                spaxel_label = f'{idx_j}-{idx_i}'
                ext_label = f'{spaxel_label}{log_ext_suffix}'

                # Spaxel progress message
                pbar.output_message(j, n_spaxels, pre_text="", post_text=f'(coordinate {spaxel_label})')

                # Recover the spectrum
                spec_flux = self.flux[:, idx_j, idx_i] * self.norm_flux
                spec_err_flux = self.err_flux[:, idx_j, idx_i] * self.norm_flux if err_check else None

                # Remove mask
                spec_flux = spec_flux.data
                spec_err_flux = spec_err_flux.data if err_check else None

                # Convert to table-HDU format
                if err_check:
                    data_array = np.rec.fromarrays([spec_flux, spec_err_flux], dtype=[('flux', '>f8'), ('flux_err', '>f8')])
                else:
                    data_array = np.rec.fromarrays([spec_flux], dtype=[('flux', '<f8')])

                # Create spaxel_page
                table_hdu_i = fits.TableHDU(data_array, header=hdr_coords, name=ext_label)
                hdul.append(table_hdu_i)

        hdul.writeto(output_address, overwrite=True)
        hdul.close()

        return


class Sample(UserDict, OpenFits):

    """

    This class creates a dictionary-like variable to store LiMe observations, by the fault it is assumed that these are
    ``Spectrum`` objects.

    The sample is indexed via the input ``log`` parameter, a pandas dataframe, whose levels must be declared via
    the ``levels`` parameter. By default, three levels are assumed: an "id" column and a "file" column specifying the object
    ID and observation file address respectively. The "line" level refers to the label measurements in the corresponding
    The user can specify more levels via the ``levels`` parameter. However, it is recommended to keep this structure: "id"
    and "file" levels first and the "line" column last.

    To create the LiMe observation variables (``Spectrum`` or ``Cube``) the user needs to specify a ``load_function``.
    This is a python method which declares how the observational files are read and parsed and returns a LiMe object.
    This ``load_function`` must have 4 parameters: ``log_df``, ``obs_idx``, ``folder_obs`` and ``**kwargs``.

    The first and second variable represent the sample ``log`` and a single pandas multi-index entry for the requested
    observation. The ``folder_obs`` and ``**kwargs`` are provided at the ``Sample`` creation:

    The ``folder_obs`` parameter specifies the root file location for the targeted observation file. This root address
    is combined with the corresponding log level ``file`` value. If a ``folder_obs`` is not specified, it is assumed that
    the ``file`` log column contains the absolute file address.

    The ``**kwargs`` argument specifies keyword arguments used in the creation of the ``Spectrum`` or ``Cube`` objects
    such as the ```redshift`` or ``norm_flux`` for example.

    The user may also specify the instrument used for the observation. In this case LiMe will use the inbuilt functions
    to read the supported instruments. This, however, may not contain all the necessary information to create the LiMe
    variable (such as the redshift). In this case, the user can include a load_function which returns a dictionary with
    observation parameters not found on the ".fits" file.

    :param sample_log: multi-index dataframe with the parameter properties belonging to the ``Sample``.
    :type sample_log: pd.Dataframe

    :param levels: levels for the sample log dataframe. By default, these levels are "id", "file", "line".
    :type levels: list

    :param load_function: python method with the instructions to convert the observation file into a LiMe observation.
    :type load_function: python method

    :param instrument: instrument name responsible for the sample observations.
    :type instrument: string, optional.

    :param folder_obs: Root address for the observations' location. This address is combined with the "file" log column value.
    :type folder_obs: string, optional.

    :param kwargs: Additional keyword arguments for the creation of the LiMe observation variables.

    """

    def __init__(self, sample_log, levels=('id', 'file', 'line'), load_function=None, instrument=None, folder_obs=None,
                 units_wave='AA', units_flux='FLAM', **kwargs):

        # Initiate the user dictionary with a dictionary of observations if provided
        super().__init__()

        # Load parent classes
        OpenFits.__init__(self, folder_obs, instrument, load_function, 'Sample')

        # Function attributes
        self.label_list = None
        self.objects = None
        self.group_list = None
        self.levels = list(levels)

        # Check the levels on combined labels target log
        check_sample_levels(self.levels)

        # Checks units
        self.units_wave, self.units_flux = check_units(units_wave, units_flux)

        self.frame = check_file_dataframe(sample_log, sample_levels=self.levels)
        self._load_function = load_function
        self.load_params = kwargs

        # Functionality objects
        self.plot = SampleFigures(self)
        self.check = SampleCheck(self)
        self.instrument = instrument

        # Check if there is not a log
        if self.frame is None:
            _logger.warning(f'Sample was created with a null log')

        return

    @classmethod
    def from_file(cls, id_list, log_list=None, file_list=None, page_list=None, levels=('id', 'file', "line"),
                  load_function=None, instrument=None, folder_obs=None, **kwargs):

        """
        This class creates a dictionary-like variable to store LiMe observations taking a list of observations IDs, line
        logs and a list of files.

        The sample is indexed via the input ``log`` parameter, a pandas dataframe, whose levels must are declared via
        the ``levels`` parameter. By default, three levels are assumed: an "id" column and a "file" column specifying the object
        ID and observation file address respectively. The "line" level refers to the label measurements in the corresponding
        The user can specify more levels via the ``levels`` parameter. However, it is recommended to keep this structure: "id"
        and "file" levels first and the "line" column last.

        The sample log levels are created from the input values for the ``id_list``, ``log_list`` and ``file_list`` while
        the individual logs from each observation are combined where the line labels in the "line" level. If the input logs
        are ".fits" files the user must specify extension name or number via the ``page_list`` parameter.

        To create the LiMe observation variables (``Spectrum`` or ``Cube``) the user needs to specify a ``load_function``.
        This is a python method which declares how the observational files are read and parsed and returns a LiMe object.
        This ``load_function`` must have four parameters: ``log_df``, ``obs_idx``, ``folder_obs`` and ``**kwargs``.

        The first and second variable represent the sample ``log`` and a single pandas multi-index entry for the requested
        observation. The ``folder_obs`` and ``**kwargs`` are provided at the ``Sample`` creation:

        The ``folder_obs`` parameter specifies the root file location for the targeted observation file. This root address
        is combined with the corresponding log level ``file`` value. If a ``folder_obs`` is not specified, it is assumed
        that the ``file`` log column contains the absolute file address. This is

        The ``**kwargs`` argument specifies keyword arguments used in the creation of the ``Spectrum`` or ``Cube`` objects
        such as the ```redshift`` or ``norm_flux`` for example.

        :param id_list: List of observation names
        :type id_list: list

        :param log_list: List of observation log data frames or files or pandas data frames
        :type log_list: list

        :param file_list: List of observation files.
        :type file_list: list

        :param page_list: List of extension files or names for the observation ".fits" files
        :type page_list: list

        :param levels: levels for the sample log dataframe. By default, these levels are "id", "file", "line".
        :type levels: list

        :param load_function: python method with the instructions to convert the observation file into a LiMe observation.
        :type load_function: python method

        :param instrument: instrument name responsible for the sample observations.
        :type instrument: string, optional.

        :param folder_obs: Root address for the observations' location. This address is combined with the "file" log
                            column value.
        :type folder_obs: string, optional.

        :param kwargs: Additional keyword arguments for the creation of the LiMe observation variables.

        """

        # Confirm matching length of entries
        check_sample_input_files(log_list, file_list, page_list, id_list)

        # Check the levels on combined labels target log
        check_sample_levels(levels)

        # Loop through observations and combine the log
        df_list = []
        for i, id_spec in enumerate(id_list):

            # Page and spec index
            file_spec = None if file_list is None else file_list[i]
            page_name = page_list[i] if page_list is not None else 'LINESFRAME'

            # Load the log and check the levels
            if log_list is not None:
                log_i = load_frame(log_list[i], page_name, levels)
                df_list.append(review_sample_levels(log_i, id_spec, file_spec))
            else:
                log_i = pd.DataFrame(columns=["id", "file"], data=[[id_spec, file_spec]])
                log_i.set_index(["id", "file"], inplace=True)
                df_list.append(log_i)

        sample_log = pd.concat(df_list)

        return cls(sample_log, levels, load_function, instrument, folder_obs, **kwargs)

    def load_function(self, log_df, obs_idx, root_address, instrument=None, **kwargs):

        # User load function
        if self._load_function is not None:
            load_function_output = self._load_function(log_df, obs_idx, root_address, **kwargs)
            if not isinstance(load_function_output, dict):
                return load_function_output
            else:
                return Spectrum(**load_function_output) if self.spectrum_check else Cube(**load_function_output)

            # if isinstance(load_function_output, (Spectrum, Cube)):
            #     return load_function_output
            # else:
            #     return Spectrum(**load_function_output) if self.spectrum_check else Cube(**load_function_output)

        # Instrument load function
        else:
            fname_obj = root_address / obs_idx[log_df.index.names.index('file')]
            class_obj = Spectrum if self.spectrum_check else Cube
            return class_obj.from_file(fname_obj, instrument=instrument, **kwargs)


        # # Proceed to create the LiMe object if necessary
        # if isinstance(load_function_output, dict):
        #
        #     # Get address of observation
        #     file_spec = root_address / obs_idx[log_df.index.names.index('file')]
        #
        #     # User provides a data parser
        #     if self.fits_reader is not None:
        #         spec_data = self.fits_reader(file_spec)
        #         fits_args = {'input_wave': spec_data[0], 'input_flux': spec_data[1], 'input_err': spec_data[2],
        #                      **spec_data[4]}
        #     else:
        #         fits_args = {}
        #
        #     # Create observation
        #     obs_args = {**fits_args, **load_function_output}
        #     obs = Spectrum(**obs_args) if self.spectrum_check else Cube(**obs_args)
        #
        # else:
        #     obs = load_function_output
        #
        #
        # return obs

    def __getitem__(self, id_key):

        output = None
        valid_check = self._review_df_indexes()

        # Proceed to selection
        if valid_check:

            # Check if Pandas indeces, numpy boolean or scalar key
            if isinstance(id_key, pd.Index) or isinstance(id_key, pd.MultiIndex) or isinstance(id_key, pd.Series):
                idcs = id_key
            elif isinstance(id_key, (np.ndarray, np.bool_)):
                idcs = self.frame.index[id_key]
            else:
                idcs = self.frame.index.get_level_values('id').isin([id_key])

            # Not entry found
            if np.all(idcs is False):
                raise KeyError(id_key)

            # Crop sample
            output = Sample(self.frame.loc[idcs], self.levels, self.load_function, self.source, self.file_address,
                            **self.load_params)

        return output

    def get_observation(self, input_index, default_none=False):

        output = None
        valid_check = self._review_df_indexes()

        if valid_check:

            # Case only ID string
            if isinstance(input_index, str):
                idcs = self.frame.index.get_level_values('id').isin(np.atleast_1d(input_index))

                # Not entry found
                if np.all(idcs is False):
                    raise KeyError(input_index)

                # Check for logs without lines
                if 'line' not in self.frame.index.names:
                    obj_idcs = self.frame.loc[idcs].index.unique()
                else:
                    obj_idcs = self.frame.loc[idcs].iloc[0].name
            else:
                obj_idcs = input_index

            # # Not entry found
            # if len(obj_idcs) > 1:
            #     raise LiMe_Error(f'Multiple observations match the input id: {obj_idcs}')

            # Load the LiMe object
            output = self.load_function(self.frame, obj_idcs, self.file_address, **self.load_params)

        return output

    def get_spectrum(self, idx, **kwargs):

        # TODO add trick to convert tupple to multi-index MultiIndex.from_tuples([idx_obs], names=sample_files.frame.index.names)
        if isinstance(idx, pd.Series):
            idx_true = self.frame.loc[idx].index

            if idx_true.size > 1:
                raise LiMe_Error(f'Input sample spectrum extraction has more than one existing entry')

            idx_in = self.frame.loc[idx_true].index.values[0]

        else:
            idx_in = idx

        # Combine local load_params with the current ones if provided
        load_params = {**self.load_params, **kwargs}

        return self.load_function(self.frame, idx_in, self.file_address, **load_params)

    @property
    def index(self):
        return self.frame.index

    @property
    def loc(self):
        return self.frame.loc

    @property
    def ids(self):
        return self.frame.index.get_level_values('id')

    @property
    def files(self):
        return self.frame.index.get_level_values('file')

    @property
    def lines(self):
        return self.frame.index.get_level_values('line')

    @property
    def size(self):
        return self.frame.index.size

    def load_frame(self, dataframe, ext='LINESFRAME', sample_levels=['id', 'line']):

        # Load the log file if it is a log file
        log_df = check_file_dataframe(dataframe, ext=ext, sample_levels=sample_levels)

        # Security checks:
        if log_df.index.size > 0:

            if self.units_wave is not None:
                line_list = log_df.index.values

                # Get the first line in the log
                line_0 = Line.from_transition(line_list[0], data_frame=log_df, norm_flux=self.norm_flux)

                # Confirm the lines in the log match the one of the spectrum
                # TODO we need something more advance for the line_0 units
                # if line_0.units_wave != self.units_wave:
                    # _logger.warning(f'Different units in the spectrum dispersion ({self.units_wave}) axis and the '
                    #                 f' lines log in {line_0.units_wave}')

                # Confirm all the log lines have the same units
                same_units_check = np.flatnonzero(np.core.defchararray.find(line_list.astype(str), line_0.units_wave) != -1).size == line_list.size
                if not same_units_check:
                    _logger.warning(f'The log has lines with different units')

        else:
            _logger.info(f'Log file with 0 entries ({dataframe})')

        # Assign the log
        self.frame = log_df

        return

    def save_frame(self, fname, ext='LINESFRAME', param_list='all', fits_header=None):

        # Save the file
        save_frame(fname, self.frame, ext, param_list, fits_header)

        return

    def extract_fluxes(self, flux_type='mixture', sample_level='line', column_names='line_flux', column_positions=1):

        return extract_fluxes(self.frame, flux_type, sample_level, column_names, column_positions)

    def normalize_fluxes(self, normalization_line, flux_entries=['line_flux', 'line_flux_err'], column_names=None,
                         column_positions=[1, 2]):

        return normalize_fluxes(self.frame, normalization_line, flux_entries, column_names, column_positions)

    def _review_df_indexes(self):

        # Check there is a log
        check = False

        if self.frame is None:
            _logger.info(f'Sample does not contain observations')

        # Check there is load function
        elif self.load_function is None:
            _logger.info(f'The sample does not contain a load_function')

        # Check there is a 'file' index
        elif 'id' not in self.frame.index.names:
            _logger.info(f'The sample log does not contain an "id" index column the observation label')

        # Check there is a 'file' index
        elif 'file' not in self.frame.index.names:
            _logger.info(f'The sample log does not contain a "file" index column with the observation file')

        else:
            check = True

        return check



