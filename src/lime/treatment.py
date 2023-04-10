import logging
import numpy as np
import pandas as pd
from pathlib import Path
from astropy.io import fits

import lime
from .tools import LineFinder, UNITS_LATEX_DICT, DISPERSION_UNITS, label_decomposition, air_to_vacuum_function, \
                   FLUX_DENSITY_UNITS, unit_convertor, define_masks, extract_fluxes, relative_fluxes, compute_line_ratios

from .plots import SpectrumFigures, SampleFigures, CubeFigures
from .plots_interactive import SpectrumCheck, CubeCheck, SampleCheck
from .io import _LOG_DTYPES_REC, save_log, LiMe_Error, check_file_dataframe
from .transitions import Line
from .workflow import SpecTreatment, CubeTreatment
from . import Error

from matplotlib import pyplot as plt, colors, cm, gridspec, rc_context
from matplotlib.widgets import RadioButtons

_logger = logging.getLogger('LiMe')

try:
    import mplcursors
    mplcursors_check = True
except ImportError:
    mplcursors_check = False

if mplcursors_check:
    from mplcursors._mplcursors import _default_annotation_kwargs as popupProps
    popupProps['bbox']['alpha'] = 0.9


def check_inputs(wave, flux, err_flux, lime_object):

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


def check_units_norm_redshift(units_wave, units_flux, norm_flux, redshift):

    # Checks SMACS_v2.0 units
    for arg in ['units_wave', 'units_flux']:
        arg_value = locals()[arg]
        if arg_value not in UNITS_LATEX_DICT:
            _logger.warning(f'Input {arg} = {arg_value} is not recognized.\nPlease try to convert it to the accepted'
                            f'units: {list(UNITS_LATEX_DICT.keys())}')

    # Check if spectrum redshift and flux normalization flux are provided
    for arg in ['norm_flux', 'redshift']:
        arg_value = locals()[arg]
        if arg_value is None:
            _logger.debug(f'No value provided for the {arg}')


    return


def check_spectrum_axes(lime_object):

    # Check for masked arrays
    check_mask = np.zeros(3).astype(bool)
    for i, arg in enumerate(['wave', 'wave_rest', 'flux']):
        if np.ma.is_masked(lime_object.__getattribute__(arg)):
            check_mask[i] = True

    if np.any(check_mask):
        lime_object._masked_inputs = True
        if ~np.all(check_mask):
            _logger.warning(f'Make sure *all* your input wavelength, flux and uncertainty are masked arrays')

    # Check that the flux and wavelength normalization #TODO ONLY DO THIS IN REVISION
    # if not isinstance(lime_object, Cube):
    #     if np.nanmedian(lime_object.flux) < 0.0001:
    #         _logger.info(f'The input flux has a median value of {np.nanmedian(lime_object.flux):.2e} '
    #                         f'{UNITS_LATEX_DICT[lime_object.units_flux]}. This can cause issues in the fitting. '
    #                         f'Try changing the flux normalization')

    return


def cropping_spectrum(crop_waves, input_wave, input_flux, input_err, pixel_mask):

    if crop_waves is not None:

        min_limit = crop_waves[0] if crop_waves[0] != 0 else input_wave[0]
        max_limit = crop_waves[1] if crop_waves[1] != -1 else input_wave[-1]

        idcs_crop = np.searchsorted(input_wave, (min_limit, max_limit))
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
        wave = np.ma.masked_array(wave, pixel_mask)
        wave_rest = np.ma.masked_array(wave_rest, pixel_mask)

        # Spectrum or Cube spectral masking
        if len(input_flux.shape) == 1:
            mask_array = pixel_mask
        else:
            mask_array = np.ones(flux.shape).astype(bool)
            mask_array[pixel_mask, :, :] = pixel_mask
        flux = np.ma.masked_array(flux, mask_array)

        if err_flux is not None:
            err_flux = np.ma.masked_array(err_flux, mask_array)

    return wave, wave_rest, flux, err_flux


def spectral_bands(wave_inter=None, lines_list=None, ion_list=None, z_range=None,
                   parent_mask_address=None, units_wave='A', sig_fig=4, vacuum_conversion=False):

    # Use the default lime mask if none provided
    if parent_mask_address is None:
        parent_mask_address = Path(__file__).parent/'resources'/'parent_mask.txt'


    # Check that the file exists: #TODO remove blended, merged extensions from parent mask (or recover them here)
    if not Path(parent_mask_address).is_file():
        _logger.warning(f'Parent mask file not found. The input user address was {parent_mask_address}')

    # Load the parent mask
    mask_df = pd.read_csv(parent_mask_address, delim_whitespace=True, header=0, index_col=0)

    # Recover line label components
    ion_array, wave_array, latex_array = label_decomposition(mask_df.index.values)
    idcs_rows = np.ones(mask_df.index.size).astype(bool)

    # Convert to vacuum wavelengths if requested
    if vacuum_conversion:

        # First the table data
        mask_df = mask_df.apply(air_to_vacuum_function)

        # The line headers
        wave_array = air_to_vacuum_function(wave_array)
        if sig_fig is not None:
            wave_array = np.round(wave_array, sig_fig) if sig_fig != 0 else np.round(wave_array, sig_fig).astype(int)

        # Reconstruct the line labels for the new units
        labels_array = np.core.defchararray.add(ion_array, '_')
        labels_array = np.core.defchararray.add(labels_array, wave_array.astype(str))
        labels_array = np.core.defchararray.add(labels_array, units_wave)

        # Rename the indeces
        mask_df.rename(index=dict(zip(mask_df.index.values, labels_array)), inplace=True)

    # Momentarily add columns with the transition wavelength, ion and latex notation
    mask_df['wave'], mask_df['ion'], mask_df['latex'] = wave_array, ion_array, latex_array

    # Convert the output mask to the units requested by the user
    if units_wave != 'A':

        conversion_factor = unit_convertor('A', units_wave,  wave_array=1, dispersion_units='dispersion axis', sig_fig=sig_fig)
        mask_df.loc[:, 'w1':'w6'] = mask_df.loc[:, 'w1':'w6'] * conversion_factor
        wave_array = wave_array * conversion_factor

        if sig_fig is not None:
            wave_array = np.round(wave_array, sig_fig) if sig_fig != 0 else np.round(wave_array, sig_fig).astype(int)

        # Reconstruct the line labels for the new units
        labels_array = np.core.defchararray.add(ion_array, '_')
        labels_array = np.core.defchararray.add(labels_array, wave_array.astype(str))
        labels_array = np.core.defchararray.add(labels_array, units_wave)

        # Rename the indeces
        mask_df.rename(index=dict(zip(mask_df.index.values, labels_array)), inplace=True)

    # First slice by wavelength and redshift
    if wave_inter is not None:

        # In case the input is an spectrum
        if isinstance(wave_inter, Spectrum):
            wave_inter = wave_inter.wave_rest

        # Establish the lower and upper wavelenght limits
        if np.ma.isMaskedArray(wave_inter):
            w_min, w_max = wave_inter.data[0], wave_inter.data[-1]
        else:
            w_min, w_max = wave_inter[0], wave_inter[-1]

        if z_range is not None:
            z_range = np.array(z_range, ndmin=1)
            w_min, w_max = w_min * (1 + z_range[0]), w_max * (1 + z_range[-1])

        idcs_rows = idcs_rows & (wave_array >= w_min) & (wave_array <= w_max)

    # Second slice by ion
    if ion_list is not None:
        idcs_rows = idcs_rows & mask_df.ion.isin(ion_list)

    # Finally slice by the name of the lines
    if lines_list is not None:
        idcs_rows = idcs_rows & mask_df.index.isin(lines_list)

    # Return the sliced the parent mask
    output_mask = mask_df.loc[idcs_rows, 'w1':'w6']

    return output_mask


class Spectrum(LineFinder):

    """
    This class defines a spectrum object from which the lines can be measured. The required inputs for the spectrum definition
    are the observation wavelength and flux arrays.

    Optionally, the user can provide the sigma spectrum with the pixel uncertainty. This array must be in the same
    units as the ``input_flux``.

    It is recommended to provide the object redshift and a flux normalization. This guarantees the functionality of
    the class functions.

    The user can also provide a two value array with the same wavelength limits. This array must be in the
    same units and _frame of reference as the ``.wave``.

    The user can also include the spectrum instrumental FWHM so it can be taken into account during the measurements.

    :param input_wave: wavelength array
    :type input_wave: numpy.array

    :param input_flux: flux array
    :type input_flux: numpy.array

    :param input_err: sigma array of the ``input_flux``
    :type input_err: numpy.array, optional

    :param redshift: spectrum redshift
    :type redshift: float, optional

    :param norm_flux: spectrum flux normalization
    :type norm_flux: float, optional

    :param crop_waves: wavelength array crop values
    :type norm_flux: np.array, optional

    :param inst_FWHM: Spectrum instrument FWHM
    :type inst_FWHM: float, optional

    :param units_wave: Wavelength array physical units. The default value is angstrom (A)
    :type units_wave: str, optional

    :param units_flux: Flux array physical units. The default value is F_lamda (erg/cm^2/s/A)
    :type units_flux: str, optional

    :param label: Name for the spectrum object
    :type label: str, optional

    """

    def __init__(self, input_wave=None, input_flux=None, input_err=None, redshift=0, norm_flux=1.0, crop_waves=None,
                 inst_FWHM = np.nan, units_wave='A', units_flux='Flam', pixel_mask=None, label=None, review_inputs=True):

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

        self.log = None

        self.redshift = None
        self.norm_flux = None
        self.inst_FWHM = None
        self.units_wave = None
        self.units_flux = None
        self._masked_inputs = False

        # Treatments objects
        self.fit = SpecTreatment(self)

        # Plotting objects
        self.plot = SpectrumFigures(self)
        self.check = SpectrumCheck(self)

        # Review and assign the attibutes data
        if review_inputs:
            self._set_attributes(input_wave, input_flux, input_err, redshift, norm_flux, crop_waves, inst_FWHM,
                                 units_wave, units_flux, pixel_mask, label)

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
        spec.log = pd.DataFrame(np.empty(0, dtype=_LOG_DTYPES_REC))
        spec.inst_FWHM = cube.inst_FWHM
        spec.units_wave = cube.units_wave
        spec.units_flux = cube.units_flux
        spec._masked_inputs = False

        return spec

    def _set_attributes(self, input_wave, input_flux, input_err, redshift, norm_flux, crop_waves, inst_FWHM, units_wave,
                        units_flux, pixel_mask, label):

        # Class attributes
        self.label = label
        self.norm_flux = norm_flux
        self.redshift = redshift
        self.inst_FWHM = inst_FWHM
        self.units_wave = units_wave
        self.units_flux = units_flux

        # Review the inputs
        check_inputs(input_wave, input_flux, input_err, self)

        # Checks units
        check_units_norm_redshift(self.units_wave, self.units_flux, self.norm_flux, self.redshift)

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
        self.log = pd.DataFrame(np.empty(0, dtype=_LOG_DTYPES_REC))

        return

    def convert_units(self, units_wave=None, units_flux=None, norm_flux=None):

        # Dispersion axes conversion
        if units_wave is not None:

            # Remove the masks for the conversion
            input_wave = self.wave.data if np.ma.is_masked(self.wave) else self.wave

            # Convert the data
            if units_wave in DISPERSION_UNITS:
                output_wave = unit_convertor(self.units_wave, units_wave, wave_array=input_wave)
            else:
                _logger.warning(f'- Dispersion units {units_wave} not recognized for conversion. '
                                f'Please use {DISPERSION_UNITS} to convert from {self.units_wave}')

            # Reflect the new units
            if np.ma.is_masked(self.wave):
                self.wave = np.ma.masked_array(output_wave, self.wave.mask)
                self.wave_rest = np.ma.masked_array(output_wave/(1+self.redshift), self.wave.mask)
            else:
                self.wave = output_wave
                self.wave_rest = output_wave/(1+self.redshift)
            self.units_wave = units_wave

        # Flux axis conversion
        if units_flux is not None:

            # Remove the masks for the conversion
            input_wave = self.wave.data if np.ma.is_masked(self.wave) else self.wave
            input_flux = self.flux.data if np.ma.is_masked(self.flux) else self.flux
            input_err = self.err_flux.data if np.ma.is_masked(self.err_flux) else self.err_flux

            if units_flux in FLUX_DENSITY_UNITS:
                output_flux = unit_convertor(self.units_flux, units_flux, wave_array=self.wave,
                                             flux_array=input_flux, dispersion_units=self.units_wave)

                if input_err is not None:
                    output_err = unit_convertor(self.units_flux, units_flux, wave_array=input_wave,
                                                flux_array=input_err, dispersion_units=self.units_wave)

            else:
                _logger.warning(f'- Dispersion units {units_flux} not recognized for conversion. '
                                f'Please use {FLUX_DENSITY_UNITS} to convert from {self.units_flux}')

            # Reflect the new units
            if np.ma.is_masked(self.flux):
                self.flux = np.ma.masked_array(output_flux, self.flux.mask)
            else:
                self.flux = output_flux
            if input_err is not None:
                self.err_flux = np.ma.masked_array(output_err, self.err_flux.mask) if np.ma.is_masked(self.err_flux) else output_err
            self.units_flux = units_flux

        # Switch the normalization
        if norm_flux is not None:
            # TODO isMaskedArray checks individually?
            mask_check = np.ma.is_masked(self.flux)

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

    def save_log(self, file_address, ext='LINESLOG', param_list='all', fits_header=None):

        # Save the file
        save_log(self.log, file_address, ext, param_list, fits_header)

        return

    def load_log(self, log_var, ext='LINESLOG'):

        # Load the log file if it is a log file
        log_df = check_file_dataframe(log_var, pd.DataFrame, ext=ext)

        # Security checks:
        if log_df.index.size > 0:
            line_list = log_df.index.values

            # Get the first line in the log
            line_0 = Line.from_log(line_list[0], log_df, norm_flux=self.norm_flux)

            # Confirm the lines in the log match the one of the spectrum
            if line_0.units_wave != self.units_wave:
                _logger.warning(f'Different units in the spectrum dispersion ({self.units_wave}) axis and the lines log'
                                f' in {line_0.units_wave}')

            # Confirm all the log lines have the same units
            same_units_check = np.flatnonzero(np.core.defchararray.find(line_list.astype(str), line_0.units_wave) != -1).size == line_list.size
            if not same_units_check:
                _logger.warning(f'The log has lines with different units')

            # Assign the log
            self.log = log_df

        else:
            _logger.info(f'Log file with 0 entries ({log_var})')

        return

    def udpate_redshift(self, redshift):

        # Check if it is a masked array
        if np.ma.is_masked(self.wave):
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

    def __init__(self, input_wave=None, input_flux=None, input_err=None, redshift=0, norm_flux=1.0, crop_waves=None,
                 inst_FWHM = np.nan, units_wave='A', units_flux='Flam', spatial_mask=None, pixel_mask=None, obj_name=None):

        # Review the inputs
        check_inputs(input_wave, input_flux, input_err, self)

        # Class attributes
        self.obj_name = obj_name
        self.wave = None
        self.wave_rest = None
        self.flux = None
        self.err_flux = None
        self.norm_flux = norm_flux
        self.redshift = redshift
        self.log = None
        self.inst_FWHM = inst_FWHM
        self.units_wave = units_wave
        self.units_flux = units_flux
        self._masked_inputs = False

        # Treatments objects
        self.fit = CubeTreatment(self)

        # Plotting objects
        self.plot = CubeFigures(self)
        self.check = CubeCheck(self)

        # Checks SMACS_v2.0 units
        check_units_norm_redshift(self.units_wave, self.units_flux, self.norm_flux, self.redshift)

        # Start cropping the input spectrum if necessary
        input_wave, input_flux, input_err, pixel_mask = cropping_spectrum(crop_waves, input_wave, input_flux, input_err,
                                                                          pixel_mask)

        # SMACS_v2.0 normalization, redshift and mask calculation
        self.wave, self.wave_rest, self.flux, self.err_flux = spec_normalization_masking(input_wave, input_flux,
                                                                                         input_err, pixel_mask,
                                                                                         self.redshift, self.norm_flux)

        # Check nan entries and mask quality
        check_spectrum_axes(self)

        return

    def spatial_masker(self, line, band=None, param=None, percentiles=[85, 90, 95], mask_ref=None, min_percentil=None,
                       output_address=None, header_dict={}, bands_frame=None):

        """
        This function computes a spatial mask for an input flux image given an array of limits for a certain intensity parameter.
        Currently, the only one implemented is the percentile intensity. If an output address is provided, the mask is saved as a fits file
        where each intensity level mask is stored in its corresponding page. The parameter calculation method, its intensity and mask
        index are saved in the corresponding HDU header as PARAM, PARAMIDX and PARAMVAL.

        :param image_flux: Matrix with the image flux to be spatially masked.
        :type image_flux: np.array()

        :param mask_param: Flux intensity model from which the masks are calculated. The options available are 'flux',
               'SN_line' and 'SN_cont'.
        :type mask_param: str, optional

        :param contour_levels: Vector in decreasing order with the parameter values for the mask_param chosen.
        :type contour_levels: np.array()

        :param mask_ref: String label for the mask. If none provided the masks will be named in cardinal order.
        :type mask_ref: str, optional

        :param output_address: Output address for the mask fits file.
        :type output_address: str, optional

        :param min_level: Minimum level for the masks calculation. If none is provided the minimum value from the contour_levels
                          vector will be used.
        :type min_level: float, optional

        :param show_plot: If true a plot will be displayed with the mask calculation. Additionally, if an output_address is
                          provided the plot will be saved in the parent folder as image taking into consideration the
                          mask_ref value.
        :type show_plot: bool, optional

        :param fits_header: Dictionary with key-values to be included in the output .fits file header.
        :type fits_header: dict, optional

        :return:
        """

        # Check the function inputs
        if not np.all(np.diff(percentiles) > 0):
            raise Error(f'The mask percentiles ({percentiles}) must be in increasing order')
        inver_percentiles = np.flip(percentiles)

        if not param in [None, 'SN_line', 'SN_cont']:
            raise Error(f'The mask calculation parameter ({param}) is not recognised. Please use "flux", "SN_line", "SN_cont"')


        # TODO overwrite spatial mask file not update
        # Line for the background image
        line_bg = Line(line, band, ref_log=bands_frame)

        # Get the band indexes
        idcsEmis, idcsCont = define_masks(self.wave, line_bg.mask * (1 + self.redshift), line_bg.pixel_mask)
        signal_slice = self.flux[idcsEmis, :, :]

        # If not mask parameter provided we use the flux percentiles
        if param is None:
            default_title = 'Flux percentiles masks'
            param = self.units_flux
            param_image = signal_slice.sum(axis=0)

        # S/N cont
        elif param == 'SN_cont':
            default_title = 'Continuum S/N percentile masks'

            param_image = np.nanmean(signal_slice, axis=0) / np.nanstd(signal_slice, axis=0)

        # S/N line
        else:
            default_title = 'Emission line S/N percentile masks'

            n_pixels = np.sum(idcsCont)
            cont_slice = self.flux[idcsCont, :, :]
            Amp_image = np.nanmax(signal_slice, axis=0) - np.nanmean(cont_slice, axis=0)
            std_image = np.nanstd(cont_slice, axis=0)
            param_image = (np.sqrt(2 * n_pixels * np.pi) / 6) * (Amp_image / std_image)

        # Percentiles vector for the target parameter
        param_array = np.nanpercentile(param_image, inver_percentiles)

        # If minimum level not provided by user use lowest contour_level
        min_level = param_array[-1] if min_percentil is None else min_percentil

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
                mask_dict[f'mask_{i}'] = maParamImage.mask
                boundary_dict[f'mask_{i}'] = inver_percentiles[i]
                param_level[f'mask_{i}'] = param_array[i]

        # Use as HDU as container for the mask
        hdul = fits.HDUList([fits.PrimaryHDU()])

        for idx_region, region_items in enumerate(mask_dict.items()):
            region_label, region_mask = region_items

            # Metadata for the fits page
            header_lime = {'PARAM': param,
                           'PARAMIDX': boundary_dict[region_label],
                           'PARAMVAL': param_level[region_label],
                           'NUMSPAXE': np.sum(region_mask)}
            fits_hdr = fits.Header(header_lime)

            if header_dict is not None:
                fits_hdr.update(header_dict)

            # Extension for the mask
            mask_ext = region_label if mask_ref is None else f'{mask_ref}_{region_label}'

            # Mask HDU
            mask_hdu = fits.ImageHDU(name=mask_ext, data=region_mask.astype(int), ver=1, header=fits_hdr)

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

    def get_spaxel(self, idx_j, idx_i, label=None):

        spaxel = Spectrum.from_cube(self, idx_j, idx_i, label)

        return spaxel


class Sample(dict):

    def __init__(self, label_list=None, observation_list=None, log_list=None, level_names=['id', 'line'], ext_list='LINESLOG'):

        '''
        First entry of mutli index is the object name
        '''

        # Inherit the default dictionary properties
        obj_dict = {}
        super().__init__(obj_dict)

        # Attributes
        self.label_list = None
        self.obj_list = None
        self.group_list = None

        self.log = None

        self.norm_flux = None
        self.units_wave = None
        self.units_flux = None

        # Functionality objects
        self.plot = SampleFigures(self)
        self.check = SampleCheck(self)

        # Check labels are provided
        if label_list is not None:
            label_list = np.array([label_list], ndmin=1).squeeze()
            self.obj_list = label_list if len(label_list.shape) == 1 else label_list[:, 0]

            # Get the sub-groups array
            if len(label_list.shape) > 1:
                self.group_list = label_list[:, 1:]

            # Generate label list
            self.label_list = list(self.obj_list.astype(str))
            if self.group_list is not None:
                for i, label in enumerate(self.label_list):
                    self.label_list[i] = f'{label},{",".join(self.group_list[i])}'
            self.label_list = np.array(self.label_list)

            # Store the observations
            if observation_list is not None:
                for i, obs in enumerate(observation_list):

                    # Check for non-LiMe objects
                    if not isinstance(obs, (lime.Spectrum, lime.Cube)):
                        raise LiMe_Error(f'Object {label} of type {type(obs)} is not a LiMe object')

                    # Check for the SMACS_v2.0 units and normalization
                    for attr in ['norm_flux', 'units_wave', 'units_flux']:
                        attr_value_sample = self.__getattribute__(attr)
                        attr_value_obj = obs.__getattribute__(attr)
                        if attr_value_sample is None:
                            self.__setattr__(attr, obs.__getattribute__(attr))
                        else:
                            if attr_value_sample != attr_value_obj:
                                _logger.warning(f'Observation { self.label_list[i]} {attr} value ({attr_value_obj}) '
                                                   f'does not sample value ({attr_value_sample})')

            # Store the SMACS_v2.0
            log_dict = {}
            if log_list is not None:

                ext_list = np.array([ext_list], ndmin=1)
                for i, log in enumerate(log_list):

                    ext = ext_list[0] if len(ext_list) == 1 else ext_list[i]
                    log_df = check_file_dataframe(log, pd.DataFrame, ext=ext)

                    log_dict[self.label_list[i]] = log_df

                    # Add to object if it is there
                    if self.label_list[i] in self:
                        self.label_list[i].load_log(log_df)

                # Concact the panel
                obj_list, log_list = list(log_dict.keys()), list(log_dict.values())
                self.log = pd.concat(list(log_dict.values()), axis=0, keys=list(log_dict.keys()))
                self.log.rename_axis(index=level_names, inplace=True)

        return

    def add_object(self, label, obs_type='spectrum', **kwargs):

        # Establish the type of observations
        if obs_type == 'spectrum':
            lime_obj = Spectrum(label=label, **kwargs)

        # Add object to the container
        self[label] = lime_obj

        # Renew the list of objects
        self.obj_list = np.array(list(self.keys()))

        # Check if the units and normalizations match
        if len(self.keys()) == 1:
            self.norm_flux = lime_obj.norm_flux
            self.units_wave, self.units_flux = lime_obj.units_wave, lime_obj.units_flux

        else:
            for prop in ['norm_flux', 'units_wave', 'units_flux']:
                if self.__getattribute__(prop) != lime_obj.__getattribute__(prop):
                    _logger.warning(f'The {prop} of object {label} do not match those in the sample:'
                                    f' "{lime_obj.__getattribute__(prop)}" in object versus "{self.__getattribute__(prop)}" in sample')

    def load_log(self, log_var, ext='LINESLOG', sample_levels=['id', 'line']):

        # Load the log file if it is a log file
        log_df = check_file_dataframe(log_var, pd.DataFrame, ext=ext, sample_levels=sample_levels)

        # Security checks:
        if log_df.index.size > 0:

            if self.units_wave is not None:
                line_list = log_df.index.values

                # Get the first line in the log
                line_0 = Line.from_log(line_list[0], log_df, norm_flux=self.norm_flux)

                # Confirm the lines in the log match the one of the spectrum
                if line_0.units_wave != self.units_wave:
                    _logger.warning(f'Different units in the spectrum dispersion ({self.units_wave}) axis and the '
                                    f' lines log in {line_0.units_wave}')

                # Confirm all the log lines have the same units
                same_units_check = np.flatnonzero(np.core.defchararray.find(line_list.astype(str), line_0.units_wave) != -1).size == line_list.size
                if not same_units_check:
                    _logger.warning(f'The log has lines with different units')

        else:
            _logger.info(f'Log file with 0 entries ({log_var})')

        # Assign the log
        self.log = log_df

        return

    def save_log(self, file_address, ext='LINESLOG', param_list='all', fits_header=None):

        # Save the file
        save_log(self.log, file_address, ext, param_list, fits_header)

        return

    def extract_fluxes(self, flux_type='mixture', sample_level='line', column_names=['line_flux', 'line_flux_err'],
                       column_positions=[1, 2]):

        return extract_fluxes(self.log, flux_type, sample_level, column_names, column_positions)

    def relative_fluxes(self, normalization_line, flux_entries=['line_flux', 'line_flux_err'], column_names=None,
                        column_positions=[1, 2]):

        return relative_fluxes(self.log, normalization_line, flux_entries, column_names, column_positions)

    def compute_line_ratios(self, line_ratios=None, flux_headers=['line_flux', 'line_flux_err'],
                            sample_levels=['id', 'line'], keep_empty_columns=True):

        return compute_line_ratios(self.log, line_ratios, flux_headers, sample_levels, object_id=None,
                                   keep_empty_columns=keep_empty_columns)