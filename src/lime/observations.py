import logging
import numpy as np
import pandas as pd
from pathlib import Path
from astropy.io import fits
from collections import UserDict

from .tools import unit_conversion, extract_fluxes, normalize_fluxes, ProgressBar, check_units, au, extract_wcs_header

from .recognition import LineFinder, DetectionInference
from .plots import SpectrumFigures, SampleFigures, CubeFigures
from .plots_interactive import SpectrumCheck, CubeCheck, SampleCheck
from .io import _LOG_EXPORT_RECARR, save_frame, LiMe_Error, check_file_dataframe, _PARENT_BANDS, \
    check_file_array_mask, load_frame

from .read_fits import OpenFits, SPECTRUM_FITS_PARAMS
from .transitions import Line, latex_from_label, air_to_vacuum_function
from .workflow import SpecTreatment, CubeTreatment
from . import Error, __version__

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


def check_inputs_arrays(wave, flux, err_flux, lime_object):

    for i, items in enumerate(locals().items()):

        if i < 3:

            key, value = items
            if value is not None:

                # Confirm numpy array inputs
                if isinstance(value, np.ndarray):

                    # Confirm dimensions
                    dimensions = len(value.shape)
                    spec_check = dimensions == 1 and (isinstance(lime_object, Spectrum) or key == 'wave')
                    cube_type = dimensions == 3 and isinstance(lime_object, Cube)
                    if not spec_check and not cube_type:
                        raise LiMe_Error(f'The dimensions of the input {key} are {dimensions}.\n'
                                         f'LiMe only recognizes 1D arrays for the wavelength array, \n'
                                         f'1D flux arrays for the Spectrum objects \n'
                                         f'and 3D flux arrays Cube objects.')
                else:
                    raise LiMe_Error(f'The input {key} array must be numpy array. The input variable type is a {type(value)}')
            else:
                if key in ['wave', 'flux']:
                    _logger.info(f'No value has been provided for {key}.')

    return


def check_redshift_norm(redshift, norm_flux, flux_array, units_flux, norm_factor=100):

    if redshift is None:
        _logger.info(f'No redshift provided for the spectrum. Assuming local universe observation (z = 0)')
        redshift = 0

    if redshift < 0:
        _logger.warning(f'Input spectrum redshift has a negative value: z = {redshift}')

    if norm_flux is None:
        if units_flux.scale == 1:
            norm_flux = np.nanmean(flux_array) / norm_factor
            _logger.info(f'Normalizing input flux by {norm_flux}')
        else:
            norm_flux = 1

    return redshift, norm_flux


def check_spectrum_axes(lime_object):

    # Check for masked arrays
    array_labels = ['wave', 'wave_rest', 'flux']
    check_mask = np.zeros(3).astype(bool)
    for i, arg in enumerate(array_labels):
        if np.ma.isMaskedArray(lime_object.__getattribute__(arg)):
            check_mask[i] = True

    # TODO this one should go at the begining and review inputs
    if np.any(check_mask) and isinstance(lime_object, Spectrum):
        if ~np.all(check_mask):
            for i, arg in enumerate(array_labels):
                if not check_mask[i]:
                    _logger.warning(f'Your {arg} array does not include a pixel mask this can caused issues on the fittings')

    # Check that the flux and wavelength normalization #
    # if not isinstance(lime_object, Cube):
    #     if np.nanmedian(lime_object.flux) < 0.0001:
    #         _logger.info(f'The input flux has a median value of {np.nanmedian(lime_object.flux):.2e} '
    #                         f'{UNITS_LATEX_DICT[lime_object.units_flux]}. This can cause issues in the fitting. '
    #                         f'Try changing the flux normalization')

    return


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

        # min_limit = crop_waves[0] if crop_waves[0] != 0 else input_wave[0]
        # max_limit = crop_waves[1] if crop_waves[1] != -1 else input_wave[-1]
        #
        # idcs_crop = np.searchsorted(input_wave, (min_limit, max_limit))
        # input_wave = input_wave[idcs_crop[0]:idcs_crop[1]]
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

    # Apply the redshift correction
    if input_wave is not None:
        wave_rest = input_wave / (1 + redshift)
        if (input_wave is not None) and (input_flux is not None):
            wave = input_wave
            flux = input_flux  # * (1 + self.redshift)
            if input_err is not None:
                err_flux = input_err  # * (1 + self.redshift)
            else:
                err_flux = None

    # Normalize the spectrum
    if input_flux is not None:
        flux = flux / norm_flux
        if input_err is not None:
            err_flux = err_flux / norm_flux

    # Masked the arrays if requested
    if pixel_mask is not None:

        # Confirm boolean mask
        bool_mask = pixel_mask.astype(bool)

        # Check for non-1D arrays
        if len(pixel_mask.shape) == 1:
            wave = np.ma.masked_array(wave, bool_mask)
            wave_rest = np.ma.masked_array(wave_rest, bool_mask)

        # Spectrum or Cube spectral masking
        flux = np.ma.masked_array(flux, bool_mask)

        # if len(input_flux.shape) == 1:
        #     mask_array = pixel_mask
        # else:
        #     mask_array = np.ones(flux.shape).astype(bool)
        #     mask_array[pixel_mask, :, :] = pixel_mask

        if err_flux is not None:
            err_flux = np.ma.masked_array(err_flux, bool_mask)

    return wave, wave_rest, flux, err_flux


def line_bands(wave_intvl=None, lines_list=None, particle_list=None, z_intvl=None, units_wave='Angstrom', decimals=None,
               vacuum=False, ref_bands=None):
    """

    This function returns `LiMe bands database <https://lime-stable.readthedocs.io/en/latest/inputs/n_inputs3_line_bands.html>`_
    as a pandas dataframe.

    If the user provides a wavelength array (``wave_inter``), a lime.Spectrum or lime.Cube the output dataframe will be
    limited to the lines within this wavelength interval.

    Similarly, the user provides a ``lines_list`` or a ``particle_list`` the output bands will be limited to the these
    lists. These inputs must follow `LiMe notation style <https://lime-stable.readthedocs.io/en/latest/inputs/n_inputs2_line_labels.html>`_

    If the user provides a redshift interval (``z_intvl``) alongside the wavelength interval (``wave_intvl``) the output
    bands will be limited to the transitions which can be observed given the two parameters.

    The default line labels and bands ``units_wave`` are angstroms (A), additional options are: um, nm, Hz, cm, mm.

    The argument ``decimals`` determines the number of decimal figures for the line labels.

    The user can request the output line labels and bands wavelengths in vacuum setting ``vacuum=True``. This conversion
    is done using the relation from `Greisen et al. (2006) <https://www.aanda.org/articles/aa/abs/2006/05/aa3818-05/aa3818-05.html>`_.

    Instead of the default LiMe database, the user can provide a ``ref_bands`` dataframe (or the dataframe file address)
    to use as the reference database.

    :param wave_intvl: Wavelength interval for output line transitions.
    :type wave_intvl: list, numpy.array, lime.Spectrum, lime.Cube, optional

    :param lines_list: Line list for output line bands.
    :type lines_list: list, numpy.array, optional

    :param particle_list: Particle list for output line bands.
    :type particle_list: list, numpy.array, optional

    :param z_intvl: Redshift interval for output line bands.
    :type z_intvl: list, numpy.array, optional

    :param units_wave: Labels and bands wavelength units. The default value is "A".
    :type units_wave: str, optional

    :param decimals: Number of decimal figures for the line labels.
    :type decimals: int, optional

    :param vacuum: Set to True for vacuum wavelength values. The default value is False.
    :type vacuum: bool, optional

    :param ref_bands: Reference bands dataframe. The default value is None.
    :type ref_bands: pandas.Dataframe, str, pathlib.Path, optional

    :return:
    """

    # Use the default lime mask if none provided
    if ref_bands is None:
        ref_bands = _PARENT_BANDS

    # Load the reference bands
    mask_df = check_file_dataframe(ref_bands, pd.DataFrame)

    # Recover line label components
    idcs_rows = np.ones(mask_df.index.size).astype(bool)

    # Convert to vacuum wavelengths if requested
    if vacuum:

        # First the table data
        air_columns = ['wavelength', 'w1', 'w2', 'w3', 'w4', 'w5', 'w6']
        mask_df[air_columns] = mask_df[air_columns].apply(air_to_vacuum_function, raw=True)

    # Convert to requested units
    units_wave = au.Unit(units_wave)
    if units_wave != 'Angstrom':
        conversion_factor = unit_conversion(au.Unit('Angstrom'), units_wave, wave_array=1, dispersion_units='dispersion axis')
        mask_df.loc[:, 'wavelength':'w6'] = mask_df.loc[:, 'wavelength':'w6'] * conversion_factor

    # Reconstruct the latex label
    n_bands = mask_df.index.size
    mask_df['latex_label'] = latex_from_label(None, mask_df['particle'], mask_df['wavelength'],
                                              np.array([units_wave] * n_bands), np.zeros(n_bands),
                                              mask_df['transition'], decimals=decimals)

    # Re-write the line band
    particle_array = mask_df['particle'].to_numpy().astype(str)

    wave_array = mask_df['wavelength'].to_numpy()
    wave_array = np.round(wave_array, decimals) if decimals is not None else np.round(wave_array, 0).astype(int)
    wave_array = wave_array.astype(str)

    unit_string = 'A' if units_wave == 'Angstrom' else str(units_wave)

    labels_array = np.core.defchararray.add(particle_array, '_')
    labels_array = np.core.defchararray.add(labels_array, wave_array)
    labels_array = np.core.defchararray.add(labels_array, unit_string)

    mask_df.rename(index=dict(zip(mask_df.index.values, labels_array)), inplace=True)

    # First slice by wavelength and redshift
    if wave_intvl is not None:

        # In case the input is a spectrum
        if isinstance(wave_intvl, (Spectrum, Cube)):
            wave_intvl = wave_intvl.wave_rest

        # Establish the lower and upper wavelength limits
        if np.ma.isMaskedArray(wave_intvl):
            w_min, w_max = wave_intvl.data[0], wave_intvl.data[-1]
        else:
            w_min, w_max = wave_intvl[0], wave_intvl[-1]

        if z_intvl is not None:
            z_intvl = np.array(z_intvl, ndmin=1)
            w_min, w_max = w_min * (1 + z_intvl[0]), w_max * (1 + z_intvl[-1])

        wavelength_array = mask_df['wavelength']
        idcs_rows = idcs_rows & (wavelength_array >= w_min) & (wavelength_array <= w_max)

    # Second slice by particle
    if particle_list is not None:
        idcs_rows = idcs_rows & mask_df.particle.isin(particle_list)

    # Finally slice by the name of the lines
    if lines_list is not None:
        idcs_rows = idcs_rows & mask_df.index.isin(lines_list)

    return mask_df.loc[idcs_rows]


class Spectrum(LineFinder):

    """
    This class creates an astronomical cube variable for an integral field spectrograph observation.

    The user needs to provide wavelength and flux arrays. Additionally, the user can include a flux uncertainty
    array. This uncertainty must be in the same units as the flux. The cube should include its ``redshift``.

    If the flux units result in very small magnitudes, the user should also provide a normalization to make the flux
    magnitude well above zero. Otherwise, the profile fittings are likely to fail. This normalization is removed in the
    output measurements.

    The user can provide a ``pixel_mask`` boolean array with the pixels **to be excluded** from the measurements.

    The default ``units_wave`` are angtroms (Å), additional options are: um, nm, Hz, cm, mm

    The default ``units_flux`` are Flam (erg s^-1 cm^-2 Å^-1), additional options are: Fnu, Jy, mJy, nJy

    The user can also specify an instrument FWHM (``inst_FWHM``), so it can be taken into account during the measurements.

    The user can provide a ``pixel_mask`` boolean array with the pixels **to be excluded** from the measurements.

    :cvar fit: Fitting function instance from  :class:`lime.workflow.SpecTreatment`.

    :cvar plot: Plotting function instance from :class:`lime.plots.SpectrumFigures`.

    :param input_wave: wavelength array.
    :type input_wave: numpy.array

    :param input_flux: flux array.
    :type input_flux: numpy.array

    :param input_err: flux sigma uncertainty array.
    :type input_err: numpy.array, optional

    :param redshift: observation redshift.
    :type redshift: float, optional

    :param norm_flux: spectrum flux normalization.
    :type norm_flux: float, optional

    :param crop_waves: spectrum (minimum, maximum) values
    :type crop_waves: np.array, tuple, optional

    :param inst_FWHM: Instrumental FWHM.
    :type inst_FWHM: float, optional

    :param units_wave: Wavelength array units. The default value is "A".
    :type units_wave: str, optional

    :param units_flux: Flux array physical units. The default value is "Flam".
    :type units_flux: str, optional

    :param pixel_mask: Boolean array with True values for rejected pixels.
    :type pixel_mask: np.array, optional

    :param id_label: identity label for the spectrum object
    :type id_label: str, optional

    """

    # File manager for a Cube created from an observation file
    _fitsMgr = None

    def __init__(self, input_wave=None, input_flux=None, input_err=None, redshift=None, norm_flux=None, crop_waves=None,
                 inst_FWHM=None, units_wave='AA', units_flux='FLAM', pixel_mask=None, id_label=None, review_inputs=True):

        # Load parent classes
        LineFinder.__init__(self)

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
        self.inst_FWHM = None
        self.units_wave = None
        self.units_flux = None

        # Treatments objects
        self.fit = SpecTreatment(self)
        self.infer = DetectionInference(self)

        # Plotting objects
        self.plot = SpectrumFigures(self)
        self.check = SpectrumCheck(self)

        # Review and assign the attibutes data
        if review_inputs:
            self._set_attributes(input_wave, input_flux, input_err, redshift, norm_flux, crop_waves, inst_FWHM,
                                 units_wave, units_flux, pixel_mask, id_label)

        return

    @classmethod
    def from_cube(cls, cube, idx_j, idx_i, label=None):

        # Load parent classes
        spec = cls(review_inputs=False)

        # Class attributes
        spec.label = label
        spec.wave = cube.wave
        spec.wave_rest = cube.wave_rest
        spec.flux = cube.flux[:, idx_j, idx_i]
        spec.err_flux = None if cube.err_flux is None else cube.err_flux[:, idx_j, idx_i]
        spec.norm_flux = cube.norm_flux
        spec.redshift = cube.redshift
        spec.frame = pd.DataFrame(np.empty(0, dtype=_LOG_EXPORT_RECARR))
        spec.inst_FWHM = cube.inst_FWHM
        spec.units_wave = cube.units_wave
        spec.units_flux = cube.units_flux

        # Check if masked array
        if np.ma.isMaskedArray(spec.flux):
            spec.wave = np.ma.masked_array(spec.wave, cube.flux[:, idx_j, idx_i].mask)
            spec.wave_rest = np.ma.masked_array(cube.wave_rest, cube.flux[:, idx_j, idx_i].mask)

        return spec

    @classmethod
    def from_file(cls, file_address, instrument, mask_flux_entries=None, **kwargs):

        """

        This method creates a lime.Spectrum object from an observational (.fits) file. The user needs to introduce the
        file address location and the name of the instrument of survey.

        Currently, this method supports NIRSPEC, ISIS, OSIRIS and SDSS as input instrument sources. This method will
        lower case the input instrument or survey name.

        The user can include list of pixel values to generate a mask from the input file flux entries. For example, if the
        user introduces [np.nan, 'negative'] the output spectrum will mask np.nan entries and negative fluxes.

        This method provides the instrument observational units and normalization but the user should introduce
        the additional LiMe.Spectrum arguments (such as the observation redshift).

        :param file_address: Input file location address.
        :type file_address: Path, string

        :param instrument: Input file instrument or survey name
        :type instrument: str

        :param mask_flux_entries: List of pixel values to mask from flux array
        :type mask_flux_entries: list

        :param kwargs: lime.Spectrum arguments.

        :return: lime.Spectrum

        """

        # Create file manager object to administrate the file source and observation properties
        cls._fitsMgr = OpenFits(file_address, instrument, cls.__name__)

        # Load the scientific data from the file
        fits_args = cls._fitsMgr.parse_data_from_file(cls._fitsMgr.file_address, mask_flux_entries)

        # Update the parameters file parameters with the user parameters
        obs_args = {**fits_args, **kwargs}

        # Create the LiMe object
        return cls(**obs_args)

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

    def _set_attributes(self, input_wave, input_flux, input_err, redshift, norm_flux, crop_waves, inst_FWHM, units_wave,
                        units_flux, pixel_mask, label):

        # Class attributes
        self.label = label
        self.inst_FWHM = np.nan if inst_FWHM is None else inst_FWHM

        # Review the inputs
        check_inputs_arrays(input_wave, input_flux, input_err, self)

        # Checks units
        self.units_wave, self.units_flux = check_units(units_wave, units_flux)

        # Check redshift and normalization
        self.redshift, self.norm_flux = check_redshift_norm(redshift, norm_flux, input_flux, self.units_flux)

        # Start cropping the input spectrum if necessary
        input_wave, input_flux, input_err, pixel_mask = cropping_spectrum(crop_waves, input_wave, input_flux, input_err,
                                                                          pixel_mask)

        # Normalization and masking
        self.wave, self.wave_rest, self.flux, self.err_flux = spec_normalization_masking(input_wave, input_flux,
                                                                                         input_err, pixel_mask,
                                                                                         self.redshift, self.norm_flux)

        # Check nan entries and mask quality
        check_spectrum_axes(self)

        # Generate empty dataframe to store measurement use cwd as default storing folder # TODO we are not using this
        self.frame = pd.DataFrame(np.empty(0, dtype=_LOG_EXPORT_RECARR))

        return

    def unit_conversion(self, wave_units_out=None, flux_units_out=None, norm_flux=None):

        """

        This function converts spectrum wavelength array, the flux array or both arrays units.

        The user can also provide a flux normalization for the spectrum flux array.

        The wavelength units available are AA (angstroms), um, nm, Hz, cm, mm

        The flux units available are Flam (erg s^-1 cm^-2 Å^-1), Fnu (erg s^-1 cm^-2 Hz^-1), Jy, mJy, nJy

        :param wave_units_out: Wavelength array units
        :type wave_units_out: str, optional

        :param flux_units_out: Flux array units
        :type flux_units_out: str, optional

        :param norm_flux: Flux normalization
        :type norm_flux: float, optional

        """

        # Remove existing normalization
        if (self.norm_flux != 1) and (self.norm_flux is not None):

            # Remove mask and normalization
            input_mask = self.flux.mask if np.ma.isMaskedArray(self.flux) else None
            flux_arr = self.flux.data * self.norm_flux if input_mask is None else self.flux * self.norm_flux
            err_arr = self.err_flux.data * self.norm_flux if input_mask is None else self.err_flux * self.norm_flux

            # Re-apply mask
            self.flux = flux_arr if input_mask is None else np.ma.masked_array(flux_arr, self.flux.mask)
            self.err_flux = err_arr if input_mask is None else np.ma.masked_array(err_arr, self.err_flux.mask)
            self.norm_flux = None

        # Convert the requested units to astropy unit object
        wave_units_out = au.Unit(wave_units_out) if wave_units_out is not None else None
        flux_units_out = au.Unit(flux_units_out) if flux_units_out is not None else None

        # Dispersion axes conversion
        if wave_units_out is not None:

            # Convert the data
            output_wave = unit_conversion(self.units_wave, wave_units_out, wave_array=self.wave)

            # Assign the new values
            self.wave = output_wave
            self.wave_rest =  np.ma.masked_array(self.wave.data / (1+self.redshift), self.wave.mask)
            self.units_wave = wave_units_out

        # Flux axis conversion
        if flux_units_out is not None:

            # Flux conversion
            output_flux = unit_conversion(self.units_flux, flux_units_out, wave_array=self.wave,
                                          flux_array=self.flux, dispersion_units=self.units_wave)

            # Flux uncertainty conversion
            output_err = None if self.err_flux is None else unit_conversion(self.units_flux, flux_units_out,
                                                                            wave_array=self.wave,
                                                                            flux_array=self.err_flux,
                                                                            dispersion_units=self.units_wave)

            # Assign new values
            self.flux = output_flux
            self.err_flux = None if self.err_flux is None else output_err
            self.units_flux = flux_units_out

        # Switch the normalization
        if norm_flux is not None:

            # Remove mask and then apply normalization
            input_mask = self.flux.mask if np.ma.isMaskedArray(self.flux) else None
            flux_arr = self.flux.data / norm_flux if input_mask is None else self.flux / norm_flux
            err_arr = self.err_flux.data / norm_flux if input_mask is None else self.err_flux / norm_flux

            # Re-apply mask
            self.flux = flux_arr if input_mask is None else np.ma.masked_array(flux_arr, self.flux.mask)
            self.err_flux = err_arr if input_mask is None else np.ma.masked_array(err_arr, self.err_flux.mask)
            self.norm_flux = norm_flux

        return

    def save_frame(self, fname, page='FRAME', param_list='all', header=None, column_dtypes=None,
                   safe_version=True):


        """

        This function saves the spectrum measurements at the ``file_address`` provided by the user.

        The accepted extensions  are ".txt", ".pdf", ".fits", ".asdf" and ".xlsx".

        For ".fits" and ".xlsx" files the user can provide a page name for the HDU/sheet with the ``ext`` argument.
        The default name is "LINESFRAME".

        The user can specify the ``parameters`` to be saved in the output file.

        For ".fits" files the user can provide a dictionary to add to the ``fits_header``. The user can provide a ``column_dtypes``
        string or dictionary for the output fits file record array. This overwrites LiMe deafult formatting and it must have the
        same columns as the file names.


        :param fname: Output log address.
        :type fname: str, Path

        :param param_list: Output parameters list. The default value is "all"
        :type param_list: list

        :param page: Name for the HDU/sheet for ".fits"/".xlsx" files.
        :type page: str, optional

        :param header: Dictionary for ".fits" and ".asdf" files.
        :type header: dict, optional

        :param column_dtypes: Conversion variable for the `records array <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_records.html>`.
                          for the output fits file. If a string or type, the data type to store all columns. If a dictionary, a mapping of column
                          names and indices (zero-indexed) to specific data types.
        :type column_dtypes: str, dict, optional

        :param safe_version: Save LiMe version as footnote or page header on the output log. The default value is True.
        :type safe_version: bool, optional

        """

        # Meta parameters from the observations
        meta_params = {'LiMe':       __version__,
                       'u_wave':     self.units_wave.to_string(),
                       'u_flux':     self.units_flux.to_string(),
                       'redshift':   self.redshift,
                       'id':         self.label}

        # Save the file
        save_frame(fname, self.frame, page, param_list, header, column_dtypes=column_dtypes,
                   safe_version=safe_version, **meta_params)

        return

    def load_frame(self, fname, page='LINESFRAME'):

        """

        This function loads a lines measurements log as a lime.Spectrum.log variable.

        The appropriate variables are normalized by the current spectrum flux normalization.

        :param fname: Input log address.
        :type fname: str, Path

        :param page: Name of the HDU/sheet for ".fits"/".xlsx" files.
        :type page: str, optional

        """

        # Load the log file if it is a log file
        log_df = check_file_dataframe(fname, pd.DataFrame, ext=page)

        # Security checks:
        if log_df.index.size > 0:
            line_list = log_df.index.values

            # Get the first line in the log
            line_0 = Line.from_log(line_list[0], log_df, norm_flux=self.norm_flux)

            # Confirm the lines in the log match the one of the spectrum
            if line_0.units_wave[0] != self.units_wave:
                _logger.warning(f'Different units in the spectrum dispersion ({self.units_wave}) axis and the lines log'
                                f' in {line_0.units_wave[0]}')

            # Confirm all the log lines have the same units
            au_str = 'A' if line_0.units_wave[0] == 'Angstrom' else str(line_0.units_wave)
            same_units_check = np.flatnonzero(np.core.defchararray.find(line_list.astype(str), au_str) != -1).size == line_list.size
            if not same_units_check:
                _logger.warning(f'The log has lines with different units')

            # Assign the log
            self.frame = log_df

        else:
            _logger.info(f'Log file with 0 entries ({fname})')

        return

    def update_redshift(self, redshift):

        # Check if it is a masked array
        if np.ma.isMaskedArray(self.wave):
            input_wave = self.wave.data
            input_flux = self.flux.data
            input_err = self.err_flux.data
            pixel_mask = self.wave.mask
        else:
            input_wave = self.wave
            input_flux = self.flux
            input_err = self.err_flux
            pixel_mask = None

        # Normalization and masking
        self.redshift = redshift
        self.wave, self.wave_rest, self.flux, self.err_flux = spec_normalization_masking(input_wave, input_flux,
                                                                                         input_err, pixel_mask,
                                                                                         self.redshift, 1)

        return


class Cube:

    """

    This class creates an astronomical cube for an integral field spectrograph observation.

    The user needs to provide 1D wavelength and 3D flux arrays. Additionally, the user can include a 3D flux uncertainty
    array. This uncertainty must be in the same units as the flux. The cube should include its ``redshift``.

    If the flux units result in very small magnitudes, the user should also provide a normalization to make the flux
    magnitude well above zero. Otherwise, the profile fittings are likely to fail. This normalization is removed in the
    output measurements.

    The default ``units_wave`` are angtroms (Å), additional options are: um, nm, Hz, cm, mm

    The default ``units_flux`` are Flam (erg s^-1 cm^-2 Å^-1), additional options are: Fnu, Jy, mJy, nJy

    The user can also specify an instrument FWHM (``inst_FWHM``), so it can be taken into account during the measurements.

    The user can provide a ``pixel_mask`` boolean 3D array with the pixels **to be excluded** from the measurements.

    The observation object should include an astropy World Coordinate System (``wcs``) to export the spatial coordinate
    system to the measurement files.

    :param input_wave: wavelength 1D array
    :type input_wave: numpy.array

    :param input_flux: flux 3D array
    :type input_flux: numpy.array

    :param input_err: flux sigma uncertainty 3D array.
    :type input_err: numpy.array, optional

    :param redshift: observation redshift.
    :type redshift: float, optional

    :param norm_flux: spectrum flux normalization
    :type norm_flux: float, optional

    :param crop_waves: spectrum (minimum, maximum) values
    :type crop_waves: np.array, tuple, optional

    :param inst_FWHM: Instrumental FWHM.
    :type inst_FWHM: float, optional

    :param units_wave: Wavelength units. The default value is "A"
    :type units_wave: str, optional

    :param units_flux: Flux array physical units. The default value is "Flam"
    :type units_flux: str, optional

    :param pixel_mask: Boolean 3D array with True values for rejected pixels.
    :type pixel_mask: np.array, optional

    :param id_label: identity label for the spectrum object
    :type id_label: str, optional

    :param wcs: Observation `world coordinate system <https://docs.astropy.org/en/stable/wcs/index.html>`_.
    :type wcs: astropy WCS, optional

    """

    # File manager for a Cube created from an observation file
    _fitsMgr = None

    def __init__(self, input_wave=None, input_flux=None, input_err=None, redshift=None, norm_flux=None, crop_waves=None,
                 inst_FWHM=None, units_wave='AA', units_flux='FLAM', pixel_mask=None, id_label=None, wcs=None):

        # Review the inputs
        check_inputs_arrays(input_wave, input_flux, input_err, self)

        # Class attributes
        self.obj_name = id_label
        self.wave = None
        self.wave_rest = None
        self.flux = None
        self.err_flux = None
        self.inst_FWHM = np.nan if inst_FWHM is None else inst_FWHM
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

        # Check nan entries and mask quality
        check_spectrum_axes(self)

        return

    @classmethod
    def from_file(cls, file_address, instrument, mask_flux_entries=None, **kwargs):

        """

        This method creates a lime.Cube object from an observational (.fits) file.  The user needs to introduce the
        file address location and the name of the instrument of survey.

        Currently, this method supports MANGA and MUSE input instrument sources. This method will lower case the input
        instrument or survey name.

        The user can include list of pixel values to generate a mask from the input file flux entries. For example, if the
        user introduces ["nan", "negative"] the output spectrum will mask np.nan entries and negative fluxes.

        This method procures the instrument observations units, normalization and wcs but the user should introduce the
        LiMe.Spectrum arguments (such as the observation redshift).

        :param file_address: Input file location address.
        :type file_address: Path, string

        :param instrument: Input file instrument or survey name
        :type instrument: str

        :param mask_flux_entries: List of pixel values to mask from flux array
        :type mask_flux_entries: list

        :param kwargs: lime.Cube arguments.

        :return: lime.Cube

        """

        # TODO kwargs are not passing
        # Create file manager object to administrate the file source and observation properties
        cls._fitsMgr = OpenFits(file_address, instrument, cls.__name__)

        # Load the scientific data from the file
        fits_args = cls._fitsMgr.parse_data_from_file(cls._fitsMgr.file_address, mask_flux_entries)

        # Update the parameters file parameters with the user parameters
        obs_args = {**fits_args, **kwargs}

        # Create the LiMe object
        return cls(**obs_args)

    def spatial_masking(self, line, bands=None, param='flux', contour_pctls=(90, 95, 99), output_address=None,
                        mask_label_prefix=None, header=None):

        """

        This function generates a spatial binary mask for an input ``line``.

        The ``line`` argument provides the label for the mask spatial image. The bands are read from the ``bands``
        dataframe argument.

        The mask calculation can be done as a function of three parameters as a function of the ``param`` argument: "flux"
        is the sum of the flux on input band, "SN_line" is the signal-to-noise ratio for an emission line and "SN_cont"
        is the signal-to-noise of the continuum. The latter two parameters use the `Rola et al. (1994)
        <https://ui.adsabs.harvard.edu/abs/1994A%26A...287..676R/abstract>`_ definition.

        The number and spread of the binary masks is determined from percentile levels in the ``contour_pctls`` argument.

        If the user provides an ``output_address`` this function will be saved as a ".fits" file. If none is provided the
        function will return and HDUL variable.

        By default, the masks are saved in a ".fits" file with the extension name "MASK_0", "MASK_1"... The user can add a
        prefix to these names witht he ```mask_label_prefix`` argument.

        :param line: Line label for the spatial image.
        :param type: str

        :param bands: Bands dataframe (or file address to the dataframe).
        :type bands: pandas.Dataframe, str, path.Pathlib, optional

        :param param: Parameter label for mask calculation. The default value is 'flux'.
        :type param: str

        :param contour_pctls: Sorted percentile values for the binary mask calculation.
        :type contour_pctls: np.array

        :param mask_label_prefix: Prefix for the mask page name in output file
        :type mask_label_prefix: str, optional

        :param output_address: File location to store the mask.
        :type output_address: str, optional

        :param header: Dictionary for mask ".fits" file header
        :type header: dict, optional

        :return:

        """

        # Check the function inputs
        contour_pctls = np.atleast_1d(contour_pctls)
        if not np.all(np.diff(contour_pctls) > 0):
            raise Error(f'The mask percentiles ({contour_pctls}) must be in increasing order')
        inver_percentiles = np.flip(contour_pctls)

        if not param in ['flux', 'SN_line', 'SN_cont']:
            raise Error(f'The mask calculation parameter ({param}) is not recognised. Please use "flux", "SN_line", "SN_cont"')

        # Line for the background image
        line_bg = Line(line, bands)

        # Get the band indexes
        idcsEmis, idcsCont = line_bg.index_bands(self.wave, self.redshift)
        signal_slice = self.flux[idcsEmis, :, :]
        signal_slice = signal_slice if not np.ma.isMaskedArray(signal_slice) else signal_slice.data

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
            cont_slice = self.flux[idcsCont, :, :]
            cont_slice = cont_slice if not np.ma.isMaskedArray(cont_slice) else cont_slice.data

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
            signal_slice = signal_slice if not np.ma.isMaskedArray(signal_slice) else signal_slice.data

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
        output_address = Path(output_address) if output_address is not None else None

        # Return an array with the masks
        if output_address is not None:
            if output_address.parent.is_dir():
                hdul.writeto(output_address, overwrite=True, output_verify='fix')
                output_func = None
            else:
                raise LiMe_Error(f'Mask could not be saved. Folder not found: {output_address.parent.as_posix()}')

        # Return the hdul
        else:
            output_func = hdul

        return output_func

    def unit_conversion(self, units_wave=None, units_flux=None, norm_flux=None):

        """

        This function converts cube wavelength array and/or the flux array units.

        The user can also provide a flux normalization for the spectrum flux array.

        The wavelength units available are A (angstroms), um, nm, Hz, cm, mm

        The flux units available are Flam (erg s^-1 cm^-2 Å^-1), Fnu (erg s^-1 cm^-2 Hz^-1), Jy, mJy, nJy

        :param units_wave: Wavelength array units
        :type units_wave: str, optional

        :param units_flux: Flux array units
        :type units_flux: str, optional

        :param norm_flux: Flux normalization
        :type norm_flux: float, optional

        """

        # Dispersion axes conversion
        if units_wave is not None:

            # Remove the masks for the conversion
            input_wave = self.wave.data if np.ma.isMaskedArray(self.wave) else self.wave

            output_wave = unit_conversion(self.units_wave, units_wave, wave_array=input_wave)

            # Reflect the new units
            if np.ma.isMaskedArray(self.wave):
                self.wave = np.ma.masked_array(output_wave, self.wave.mask)
                self.wave_rest = np.ma.masked_array(output_wave/(1+self.redshift), self.wave.mask)
            else:
                self.wave = output_wave
                self.wave_rest = output_wave/(1+self.redshift)
            self.units_wave = au.Unit(units_wave)

        # Flux axis conversion
        if units_flux is not None:

            # Remove the masks for the conversion
            input_wave = self.wave.data if np.ma.isMaskedArray(self.wave) else self.wave
            input_flux = self.flux.data if np.ma.isMaskedArray(self.flux) else self.flux
            input_err = self.err_flux.data if np.ma.isMaskedArray(self.err_flux) else self.err_flux

            # TODO this is slow
            flux_shape = input_flux.shape
            y_range, x_range = np.arange(flux_shape[1]), np.arange(flux_shape[2])
            if len(flux_shape) == 3:
                output_flux = np.empty(flux_shape)
                for j in y_range:
                    for i in x_range:
                        output_flux[:, j, i] = unit_conversion(self.units_flux, units_flux, wave_array=self.wave,
                                              flux_array=input_flux[:, j, i], dispersion_units=self.units_wave)
            else:
                output_flux = unit_conversion(self.units_flux, units_flux, wave_array=self.wave,
                                              flux_array=input_flux, dispersion_units=self.units_wave)

            if input_err is not None:
                output_err = unit_conversion(self.units_flux, units_flux, wave_array=input_wave,
                                             flux_array=input_err, dispersion_units=self.units_wave)

            # Reflect the new units
            if np.ma.isMaskedArray(self.flux):
                self.flux = np.ma.masked_array(output_flux, self.flux.mask)
            else:
                self.flux = output_flux
            if input_err is not None:
                self.err_flux = np.ma.masked_array(output_err, self.err_flux.mask) if np.ma.isMaskedArray(self.err_flux) else output_err
            self.units_flux = au.Unit(units_flux)

        # Switch the normalization
        if norm_flux is not None:

            mask_check = np.ma.isMaskedArray(self.flux)

            # Remove old
            if mask_check:
                new_flux = self.flux.data * self.norm_flux / norm_flux
                new_err = None if self.err_flux is None else self.err_flux.data * self.norm_flux / norm_flux

                self.flux = np.ma.masked_array(new_flux, self.flux.mask)
                self.err_flux = None if self.err_flux is None else np.ma.masked_array(new_err, self.err_flux.mask)
            else:
                self.flux = self.flux * self.norm_flux / norm_flux
                self.err_flux = None if self.err_flux is None else self.err_flux * self.norm_flux / norm_flux
            self.norm_flux = norm_flux

        return

    def get_spectrum(self, idx_j, idx_i, id_label=None):

        """

        This function returns a lime.Spectrum object from the input array coordinates

        :param idx_j: y-axis array coordinate
        :type idx_j: int

        :param idx_i: x-axis array coordinate
        :type idx_i: int

        :param id_label: Identity label for spectrum object
        :type id_label: str, optional

        """

        return Spectrum.from_cube(self, idx_j, idx_i, id_label)

    def export_spaxels(self, output_address, mask_file, mask_list=None, log_ext_suffix='_LINELOG', progress_output='bar'):

        # Check if the mask variable is a file or an array
        mask_dict = check_file_array_mask(mask_file, mask_list)

        # Unpack mask dictionary
        mask_list = np.array(list(mask_dict.keys()))
        mask_data_list = list(mask_dict.values())

        # Checks for the data type
        err_check = False if self.err_flux is None else True
        masked_check = False if np.ma.isMaskedArray(self.flux) is False else True

        # Check if the output log folder exists
        output_address = Path(output_address)
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
                if masked_check:
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

        self.frame = check_file_dataframe(sample_log, pd.DataFrame, sample_levels=self.levels)
        self._load_function = load_function
        self.load_params = kwargs

        # Functionality objects
        self.plot = SampleFigures(self)
        self.check = SampleCheck(self)

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
                log_i = pd.DataFrame(columns=["id", "file"], data=(id_spec, file_spec))
                log_i.set_index(["id", "file"], inplace=True)

        sample_log = pd.concat(df_list)

        return cls(sample_log, levels, load_function, instrument, folder_obs, **kwargs)

    def load_function(self, log_df, obs_idx, root_address, **kwargs):

        # Use loading function
        if self._load_function is not None:
            load_function_output = self._load_function(log_df, obs_idx, root_address, **kwargs)
        else:
            load_function_output = {}

        # Proceed to create the LiMe object if necessary
        if isinstance(load_function_output, dict):

            # Get address of observation
            file_spec = root_address / obs_idx[log_df.index.names.index('file')]

            # User provides a data parser
            if self.fits_reader is not None:
                spec_data = self.fits_reader(file_spec)
                fits_args = {'input_wave': spec_data[0], 'input_flux': spec_data[1], 'input_err': spec_data[2],
                             **spec_data[4]}
            else:
                fits_args = {}

            # Create observation
            obs_args = {**fits_args, **load_function_output}
            obs = Spectrum(**obs_args) if self.spectrum_check else Cube(**obs_args)

        else:
            obs = load_function_output

        return obs

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

    def get_spectrum(self, idx):

        if isinstance(idx, pd.Series):
            idx_true = self.frame.loc[idx].index

            if idx_true.size > 1:
                raise LiMe_Error(f'Input sample spectrum extraction has more than one existing entry')

            idx_in = self.frame.loc[idx_true].index.values[0]

        else:
            idx_in = idx

        return self.load_function(self.frame, idx_in, self.file_address, **self.load_params)

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
        log_df = check_file_dataframe(dataframe, pd.DataFrame, ext=ext, sample_levels=sample_levels)

        # Security checks:
        if log_df.index.size > 0:

            if self.units_wave is not None:
                line_list = log_df.index.values

                # Get the first line in the log
                line_0 = Line.from_log(line_list[0], log_df, norm_flux=self.norm_flux)

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


class ObsManager(OpenFits):

    def __init__(self, file_address, file_source, lime_object, load_function=None, **kwargs):

        # Initialize the .fits reading class
        OpenFits.__init__(self, file_address, file_source, lime_object, load_function)

        # Define attribute
        self.spectrum_check = False
        self.load_function = None
        self.user_params = None

        # Store the user arguments for the spectra
        self.user_params = kwargs

        # State the type of spectra
        if file_source is None:
            self.spectrum_check = True
        else:
            self.spectrum_check = True if self.source in list(SPECTRUM_FITS_PARAMS.keys()) else False

        # Assign input load function
        if load_function is not None:
            self.load_function = load_function

        # Assign the
        elif (file_source is not None) and (self.fits_reader is not None):
            self.load_function = self.default_file_parser

        # No load function nor instrument
        else:
            raise LiMe_Error(f'To create a Sample object you need to provide "load_function" or provide a "instrument" '
                             f'supported by LiMe')

        return

    def load_function(self, file_spec, log_df=None, id_spec=None, **kwargs):

        default_args = self.fits_reader(file_spec) if self.fits_reader is not None else {}
        user_args = self.user_params if self.user_params is not None else {}

        # Recover the user params
        user_args = user_args if user_args is not None else self.user_params

        # Update with the default arguments
        default_args = {**default_args, **user_args}

        # Run the load function
        load_function_output = self.load_function(file_spec, log_df, id_spec, **default_args)

        # Proceed to create LiMe object if necessary
        if isinstance(load_function_output, dict):
            obs_args = {**default_args, **load_function_output}
            obs = Spectrum(**obs_args) if self.spectrum_check else Cube(**obs_args)
        else:
            obs = load_function_output

        return obs

    def default_file_parser(self, log_df=None, id_spec=None, **kwargs):

        file_spec = self.id_spec[log_df.index.names.index('file')]

        fits_args = self.fits_reader(file_spec)

        return fits_args

