import logging
import numpy as np
import pandas as pd
from pathlib import Path
from lmfit import fit_report as lmfit_fit_report
from sys import exit
from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS

from .model import EmissionFitting
from .tools import label_decomposition, LineFinder, UNITS_LATEX_DICT, latex_science_float, DISPERSION_UNITS,\
                   FLUX_DENSITY_UNITS, unit_convertor, define_masks

from .plots import LiMePlots, STANDARD_PLOT, STANDARD_AXES, colorDict, save_close_fig_swicth, frame_mask_switch, SpectrumFigures, SampleFigures, CubeFigures
from .plots_interactive import SpectrumCheck, CubeCheck
from .io import _LOG_DTYPES_REC, _LOG_EXPORT, _LOG_COLUMNS, load_log, save_log, LiMe_Error, check_file_dataframe
from .model import gaussian_profiles_computation, linear_continuum_computation
from .transitions import Line
from .workflow import SpecTreatment, CubeTreatment
from . import Error

from matplotlib import pyplot as plt, colors, cm, gridspec, rc_context
from matplotlib.widgets import SpanSelector
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

    # Checks spectra units
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

    # Check that the flux and wavelength normalization
    if not isinstance(lime_object, Cube):
        if np.nanmedian(lime_object.flux) < 0.0001:
            _logger.info(f'The input flux has a median value of {np.nanmedian(lime_object.flux):.2e} '
                            f'{UNITS_LATEX_DICT[lime_object.units_flux]}. This can cause issues in the fitting. '
                            f'Try changing the flux normalization')

    return


def cropping_spectrum(crop_waves, input_wave, input_flux, input_err, pixel_mask):

    if crop_waves is not None:

        idcs_crop = np.searchsorted(input_wave, crop_waves)
        input_wave = input_wave[idcs_crop[0]:idcs_crop[1]]

        # Spectrum
        if input_flux.shape == 1:
            input_flux = input_flux[idcs_crop[0]:idcs_crop[1]]
            if input_err is not None:
                input_err = input_err[idcs_crop[0]:idcs_crop[1]]

        # Cube
        elif input_flux.shape == 3:
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
        self.norm_flux = None
        self.redshift = None
        self.log = None
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

        # Checks spectra units
        check_units_norm_redshift(self.units_wave, self.units_flux, self.norm_flux, self.redshift)

        # Start cropping the input spectrum if necessary
        input_wave, input_flux, input_err, pixel_mask = cropping_spectrum(crop_waves, input_wave, input_flux, input_err,
                                                                          pixel_mask)

        # Spectra normalization and masking
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
                self.wave_rest = output_wave/1+self.redshift
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

        # Checks spectra units
        check_units_norm_redshift(self.units_wave, self.units_flux, self.norm_flux, self.redshift)

        # Start cropping the input spectrum if necessary
        input_wave, input_flux, input_err, pixel_mask = cropping_spectrum(crop_waves, input_wave, input_flux, input_err,
                                                                          pixel_mask)

        # Spectra normalization, redshift and mask calculation
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

    def __init__(self, spec_dict={}):

        # Inherit the default dictionary properties
        super().__init__(spec_dict)

        # Attributes
        self.obj_list = np.array([])
        self.plot = SampleFigures(self)
        self.norm_flux = None
        self.units_wave = None
        self.units_flux = None

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
            print('ALOMOJOR')
            self.norm_flux = lime_obj.norm_flux
            self.units_wave, self.units_flux = lime_obj.units_wave, lime_obj.units_flux

        else:
            for prop in ['norm_flux', 'units_wave', 'units_flux']:
                if self.__getattribute__(prop) != lime_obj.__getattribute__(prop):
                    _logger.warning(f'The {prop} of object {label} do not match those in the sample:'
                                    f' "{lime_obj.__getattribute__(prop)}" in object versus "{self.__getattribute__(prop)}" in sample')


# class MaskInspector(Spectrum):
#
#     """
#     This class plots the masks from the ``_log_address`` as a grid for the input spectrum. Clicking and
#     dragging the mouse within a line cell will update the line band region, both in the plot and the ``_log_address``
#     file provided.
#
#     Assuming that the band wavelengths `w1` and `w2` specify the adjacent blue (left continuum), the `w3` and `w4`
#     wavelengths specify the line band and the `w5` and `w6` wavelengths specify the adjacent red (right continuum)
#     the interactive selection has the following rules:
#
#     * The plot wavelength range is always 5 pixels beyond the mask bands. Therefore dragging the mouse beyond the
#       mask limits (below `w1` or above `w6`) will change the displayed range. This can be used to move beyond the
#       original mask limits.
#
#     * Selections between the `w2` and `w5` wavelength bands are always assigned to the line region mask as the new
#       `w3` and `w4` values.
#
#     * Due to the previous point, to increase the `w2` value or to decrease `w5` value the user must select a region
#       between `w1` and `w3` or `w4` and `w6` respectively.
#
#     The user can limit the number of lines displayed on the screen using the ``lines_interval`` parameter. This
#     parameter can be an array of strings with the labels of the target lines or a two value integer array with the
#     interval of lines to plot.
#
#     Lines in the mask file outside the spectral wavelength range will be excluded from the plot: w2 and w5 smaller
#     and greater than the blue and red wavelegnth values respectively.
#
#     :param log_address: Address for the lines log mask file.
#     :type log_address: str
#
#     :param input_wave: Wavelength array of the input spectrum.
#     :type input_wave: numpy.array
#
#     :param input_flux: Flux array for the input spectrum.
#     :type input_flux: numpy.array
#
#     :param input_err: Sigma array of the `input_flux`
#     :type input_err: numpy.array, optional
#
#     :param redshift: Spectrum redshift
#     :type redshift: float, optional
#
#     :param norm_flux: Spectrum flux normalization
#     :type norm_flux: float, optional
#
#     :param crop_waves: Wavelength limits in a two value array
#     :type crop_waves: np.array, optional
#
#     :param n_cols: Number of columns of the grid plot
#     :type n_cols: integer
#
#     :param n_rows: Number of columns of the grid plot
#     :type n_rows: integer
#
#     :param lines_interval: List of lines or mask file line interval to display on the grid plot. In the later case
#                            this interval must be a two value array.
#     :type lines_interval: list
#
#     :param y_scale: Y axis scale. The default value (auto) will switch between between linear and logarithmic scale
#                     strong and weak lines respectively. Use ``linear`` and ``log`` for a fixed scale for all lines.
#     :type y_scale: str, optional
#
#     """
#
#     def __init__(self, log_address, input_wave=None, input_flux=None, input_err=None, redshift=0,
#                  norm_flux=1.0, crop_waves=None, n_cols=10, n_rows=None, lines_interval=None, y_scale='auto'):
#
#         # Output file address
#         self.linesLogAddress = Path(log_address)
#         self.y_scale = y_scale
#
#         # Assign attributes to the parent class
#         super().__init__(input_wave, input_flux, input_err, redshift, norm_flux, crop_waves)
#
#         # Lines log address is provided and we read the DF from it
#         if Path(self.linesLogAddress).is_file():
#             self.log = load_lines_log(self.linesLogAddress)
#
#         # Lines log not provide code ends
#         else:
#             _logger.warning(f'No lines log file found at {log_address}. Leaving the script')
#             exit()
#
#         # Only plotting the lines in the lines interval
#         self.line_inter = lines_interval
#         if lines_interval is None:
#             n_lines = len(self.log.index)
#             self.target_lines = self.log.index.values
#
#         else:
#             # Array of strings
#             if isinstance(lines_interval[0], str):
#                 n_lines = len(lines_interval)
#                 self.target_lines = np.array(lines_interval, ndmin=1)
#             # Array of integers
#             else:
#                 n_lines = lines_interval[1] - lines_interval[0]
#                 self.target_lines = self.log[lines_interval[0]:lines_interval[1]].index.values
#
#         # Establish the grid shape
#         if n_lines > n_cols:
#             if n_rows is None:
#                 n_rows = int(np.ceil(n_lines / n_cols))
#         else:
#             n_cols = n_lines
#             n_rows = 1
#
#         # Adjust the plot theme
#         PLOT_CONF = STANDARD_PLOT.copy()
#         AXES_CONF = STANDARD_AXES.copy()
#         PLOT_CONF['figure.figsize'] = (n_rows * 2, 8)
#         AXES_CONF.pop('xlabel')
#
#         with rc_context(PLOT_CONF):
#
#             self.fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols)
#             self.ax = ax.flatten() if n_lines > 1 else [ax]
#             self.in_ax = None
#             self.dict_spanSelec = {}
#             self.axConf = {}
#
#             # Plot function
#             self.plot_line_mask_selection(logscale=self.y_scale, grid_size=n_rows * n_cols, n_lines=n_lines)
#             plt.gca().axes.yaxis.set_ticklabels([])
#
#             try:
#                 manager = plt.get_current_fig_manager()
#                 manager.window.showMaximized()
#             except:
#                 print()
#
#             # Show the image
#             save_close_fig_swicth(None, 'tight', self.fig)
#
#
#         return
#
#     def plot_line_mask_selection(self, logscale='auto', grid_size=None, n_lines=None):
#
#         # Generate plot
#         for i in range(grid_size):
#             if i < n_lines:
#                 line = self.target_lines[i]
#                 if line in self.log.index:
#                     self.mask = self.log.loc[line, 'w1':'w6'].values
#                     self.plot_line_region_i(self.ax[i], line, logscale=logscale)
#                     self.dict_spanSelec[f'spanner_{i}'] = SpanSelector(self.ax[i],
#                                                                        self.on_select,
#                                                                        'horizontal',
#                                                                        useblit=True,
#                                                                        rectprops=dict(alpha=0.5, facecolor='tab:blue'))
#                 else:
#                     print(f'- WARNING: line {line} not found in the input mask')
#
#             # Clear not filled axes
#             else:
#                 self.fig.delaxes(self.ax[i])
#
#         bpe = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
#         aee = self.fig.canvas.mpl_connect('axes_enter_event', self.on_enter_axes)
#
#         return
#
#     def plot_line_region_i(self, ax, lineLabel, limitPeak=5, logscale='auto'):
#
#         # Plot line region:
#         ion, lineWave, latexLabel = label_decomposition(lineLabel, scalar_output=True, units_wave=self.units_wave)
#
#         # Decide type of plot
#         non_nan = (~pd.isnull(self.mask)).sum()
#
#         # Incomplete selections
#         if non_nan < 6:  # selections
#
#             # Peak region
#             idcsLinePeak = (lineWave - limitPeak <= self.wave_rest) & (self.wave_rest <= lineWave + limitPeak)
#             wavePeak, fluxPeak = self.wave_rest[idcsLinePeak], self.flux[idcsLinePeak]
#
#             # Plot region
#             idcsLineArea = (lineWave - limitPeak * 2 <= self.wave_rest) & (
#                         lineWave - limitPeak * 2 <= self.mask[3])
#             waveLine, fluxLine = self.wave_rest[idcsLineArea], self.flux[idcsLineArea]
#
#             # Plot the line region
#             ax.step(waveLine, fluxLine, where='mid')
#
#             # Fill the user selections
#             if non_nan == 2:
#                 idx1, idx2 = np.searchsorted(self.wave_rest, self.mask[0:2])
#                 ax.fill_between(self.wave_rest[idx1:idx2], 0.0, self.flux[idx1:idx2], facecolor=self._color_dict['cont_band'],
#                                 step='mid', alpha=0.5)
#
#             if non_nan == 4:
#                 idx1, idx2, idx3, idx4 = np.searchsorted(self.wave_rest, self.mask[0:4])
#                 ax.fill_between(self.wave_rest[idx1:idx2], 0.0, self.flux[idx1:idx2], facecolor=self._color_dict['cont_band'],
#                                 step='mid', alpha=0.5)
#                 ax.fill_between(self.wave_rest[idx3:idx4], 0.0, self.flux[idx3:idx4], facecolor=self._color_dict['cont_band'],
#                                 step='mid', alpha=0.5)
#
#         # Complete selections
#         else:
#
#             # Get line regions
#             idcsContLeft = (self.mask[0] <= self.wave_rest) & (self.wave_rest <= self.mask[1])
#             idcsContRight = (self.mask[4] <= self.wave_rest) & (self.wave_rest <= self.mask[5])
#
#             idcsLinePeak = (lineWave - limitPeak <= self.wave_rest) & (self.wave_rest <= lineWave + limitPeak)
#             idcsLineArea = (self.mask[2] <= self.wave_rest) & (self.wave_rest <= self.mask[3])
#
#             waveCentral, fluxCentral = self.wave_rest[idcsLineArea], self.flux[idcsLineArea]
#             wavePeak, fluxPeak = self.wave_rest[idcsLinePeak], self.flux[idcsLinePeak]
#
#             idcsLinePlot = (self.mask[0] - 5 <= self.wave_rest) & (self.wave_rest <= self.mask[5] + 5)
#             waveLine, fluxLine = self.wave_rest[idcsLinePlot], self.flux[idcsLinePlot]
#
#             # Plot the line
#             ax.step(waveLine, fluxLine, color=self._color_dict['fg'], where='mid')
#
#             # Fill the user selections
#             ax.fill_between(waveCentral, 0, fluxCentral, step="mid", alpha=0.4, facecolor=self._color_dict['line_band'])
#             ax.fill_between(self.wave_rest[idcsContLeft], 0, self.flux[idcsContLeft], facecolor=self._color_dict['cont_band'],
#                             step="mid", alpha=0.2)
#             ax.fill_between(self.wave_rest[idcsContRight], 0, self.flux[idcsContRight], facecolor=self._color_dict['cont_band'],
#                             step="mid", alpha=0.2)
#
#         # Plot format
#         ax.yaxis.set_major_locator(plt.NullLocator())
#         ax.xaxis.set_major_locator(plt.NullLocator())
#
#         ax.update({'title': lineLabel})
#         ax.yaxis.set_ticklabels([])
#         ax.axes.yaxis.set_visible(False)
#         try:
#             idxPeakFlux = np.argmax(fluxPeak)
#             ax.set_ylim(ymin=np.min(fluxLine) / 5, ymax=fluxPeak[idxPeakFlux] * 1.25)
#
#             if logscale == 'auto':
#                 if fluxPeak[idxPeakFlux] > 5 * np.median(fluxLine):
#                     ax.set_yscale('log')
#             else:
#                 if logscale == 'log':
#                     ax.set_yscale('log')
#         except:
#             print(f'fail at {self.line}')
#
#
#         return
#
#     def on_select(self, w_low, w_high):
#
#         # Check we are not just clicking on the plot
#         if w_low != w_high:
#
#             # Count number of empty entries to determine next step
#             non_nans = (~pd.isnull(self.mask)).sum()
#
#             # Case selecting 1/3 region
#             if non_nans == 0:
#                 self.mask[0] = w_low
#                 self.mask[1] = w_high
#
#             # Case selecting 2/3 region
#             elif non_nans == 2:
#                 self.mask[2] = w_low
#                 self.mask[3] = w_high
#                 self.mask = np.sort(self.mask)
#
#             # Case selecting 3/3 region
#             elif non_nans == 4:
#                 self.mask[4] = w_low
#                 self.mask[5] = w_high
#                 self.mask = np.sort(self.mask)
#
#             elif non_nans == 6:
#                 self.mask = np.sort(self.mask)
#
#                 # Caso que se corrija la region de la linea
#                 if w_low > self.mask[1] and w_high < self.mask[4]:
#                     self.mask[2] = w_low
#                     self.mask[3] = w_high
#
#                 # Caso que se corrija el continuum izquierdo
#                 elif w_low < self.mask[2] and w_high < self.mask[2]:
#                     self.mask[0] = w_low
#                     self.mask[1] = w_high
#
#                 # Caso que se corrija el continuum derecho
#                 elif w_low > self.mask[3] and w_high > self.mask[3]:
#                     self.mask[4] = w_low
#                     self.mask[5] = w_high
#
#                 # Case we want to select the complete region
#                 elif w_low < self.mask[0] and w_high > self.mask[5]:
#
#                     # # Remove line from dataframe and save it
#                     # self.remove_lines_df(self.current_df, self.Current_Label)
#                     #
#                     # # Save lines log df
#                     # self.save_lineslog_dataframe(self.current_df, self.lineslog_df_address)
#
#                     # Clear the selections
#                     # self.mask = np.array([np.nan] * 6)
#
#                     print(f'\n-- The line {self.line} mask has been removed')
#
#                 else:
#                     print('- WARNING: Unsucessful line selection:')
#                     print(f'-- {self.line}: w_low: {w_low}, w_high: {w_high}')
#
#             # Check number of measurements after selection
#             non_nans = (~pd.isnull(self.mask)).sum()
#
#             # Proceed to re-measurement if possible:
#             if non_nans == 6:
#
#                 # TODO add option to perform the measurement a new
#                 # self.clear_fit()
#                 # self.fit_from_wavelengths(self.line, self.mask, user_cfg={})
#
#                 # Parse the line regions to the dataframe
#                 self.results_to_database(self.line, self.log, fit_conf={}, export_params=[])
#
#                 # Save the corrected mask to a file
#                 self.store_measurement()
#
#             # Redraw the line measurement
#             self.in_ax.clear()
#             self.plot_line_region_i(self.in_ax, self.line, logscale=self.y_scale)
#             self.in_fig.canvas.draw()
#
#         return
#
#     def on_enter_axes(self, event):
#
#         # Assign new axis
#         self.in_fig = event.canvas.figure
#         self.in_ax = event.inaxes
#
#         # TODO we need a better way to index than the latex label
#         # Recognise line line
#         idx_line = self.log.index == self.in_ax.get_title()
#         self.line = self.log.loc[idx_line].index.values[0]
#         self.mask = self.log.loc[idx_line, 'w1':'w6'].values[0]
#
#         # Restore measurements from log
#         # self.database_to_attr()
#
#         # event.inaxes.patch.set_edgecolor('red')
#         # event.canvas.draw()
#
#     def on_click(self, event):
#
#         if event.dblclick:
#             print(self.line)
#             print(f'{event.button}, {event.x}, {event.y}, {event.xdata}, {event.ydata}')
#
#     def store_measurement(self):
#
#         # Read file in the stored address
#         if self.linesLogAddress.is_file():
#             file_DF = load_lines_log(self.linesLogAddress)
#
#             # Add new line to the DF and sort it if it was new
#             if self.line in file_DF.index:
#                 file_DF.loc[self.line, 'w1':'w6'] = self.mask
#             else:
#                 file_DF.loc[self.line, 'w1':'w6'] = self.mask
#
#                 # Sort the lines by theoretical wavelength
#                 lineLabels = file_DF.index.values
#                 ion_array, wavelength_array, latexLabel_array = label_decomposition(lineLabels, units_wave=self.units_wave)
#                 file_DF = file_DF.iloc[wavelength_array.argsort()]
#
#         # If the file does not exist (or it is the first time)
#         else:
#             file_DF = self.log
#
#         # Save to a file
#         save_line_log(file_DF, self.linesLogAddress)
#
#         return


class CubeInspector(Spectrum):

    def __init__(self, wave, cube_flux, image_bg, image_fg=None, contour_levels=None, color_norm=None,
                 redshift=0, lines_log_address=None, fits_header=None, plt_cfg={}, ax_cfg={},
                 ext_suffix='_LINESLOG', mask_file=None, units_wave='A', units_flux='Flam'):

        """
        This class provides an interactive plot for IFU (Integra Field Units) data cubes consisting in two figures:
        On the left-hand side, a 2D image of the cube read from the ``image_bg`` parameter. This plot can include
        intensity contours from the ``contour_levels`` parameter. By default, the intensity contours are calculated from
        the ``image_bg`` matrix array unless an optional foreground ``image_fg`` array is provided. The spaxel selection
        is accomplished with a mouse right click.

        On the right-hand side, the selected spaxel spectrum is plotted. The user can select either window region using
        the matplotlib window *zoom* or *span* tools. As a new spaxel is selected the plotting limits in either figure
        should not change. To restore the plot axes limits you can click the *reset* (house icon).

        The user can provide a ``lines_log_address`` with the line measurements of the plotted object. In this case,
        the plot will include the fitted line profiles. The *.fits* file HDUs will be queried by the spaxel coordinates.
        The default format is ``{idx_j}-{idx_i}_LINESLOG)`` where idx_j and idx_i are the spaxel Y and X coordinates
        respectively

        :param wave: One dimensional array with the spectra wavelength range.
        :type wave: numpy.array

        :param cube_flux: Three dimensional array with the IFU cube flux
        :type cube_flux: numpy.array

        :param image_bg Two dimensional array with the flux band image for the plot background
        :type image_bg: numpy.array

        :param image_fg: Two dimensional array with the flux band image to plot foreground contours
        :type image_fg: numpy.array, optional

        :param contour_levels: One dimensional array with the flux contour levels in increasing order.
        :type contour_levels: numpy.array, optional

        :param color_norm: `Color normalization <https://matplotlib.org/stable/tutorials/colors/colormapnorms.html#sphx-glr-tutorials-colors-colormapnorms-py>`_
                            form the galaxy image plot
        :type color_norm: matplotlib.colors.Normalize, optional

        :param redshift: Object astronomical redshift
        :type redshift: float, optional

        :param lines_log_address: Address of the *.fits* file with the object line measurements.
        :type lines_log_address: str, optional

        :param fits_header: *.fits* header with the entries for the astronomical coordinates plot conversion.
        :type fits_header: dict, optional

        :param plt_cfg: Dictionary with the configuration for the matplotlib rcParams style.
        :type plt_cfg: dict, optional

        :param ax_cfg: Dictionary with the configuration for the matplotlib axes style.
        :type ax_cfg: dict, optional

        :param ext_suffix: Suffix of the line logs extensions. The default value is “_LINESLOG”.
        :type ext_suffix: str, optional

        :param units_wave: Wavelength array physical units. The default value is introduced as "A"
        :type units_wave: str, optional

        :param units_flux: Flux array physical units. The default value is erg/cm^2/s/A
        :type units_flux: str, optional

        """

        #TODO add _frame argument

        # Assign attributes to the parent class
        super().__init__(input_wave=np.zeros(1), input_flux=np.zeros(1), redshift=redshift, norm_flux=1,
                         units_wave=units_wave, units_flux=units_flux)

        # Data attributes
        self.grid_mesh = None
        self.cube_flux = cube_flux
        self.wave = wave
        self.header = fits_header
        self.image_bg = image_bg
        self.image_fg = image_fg
        self.contour_levels_fg = contour_levels
        self.hdul_linelog = None
        self.ext_log = ext_suffix

        # Mask correction attributes
        self.mask_file = None
        self.mask_ext = None
        self.mask_dict = {}
        self.mask_color = None
        self.mask_array = None

        # Plot attributes
        self.fig = None
        self.ax0, self.ax1, self.in_ax = None, None, None
        self.fig_conf = None
        self.axes_conf = {}
        self.axlim_dict = {}
        self.color_norm = color_norm
        self.mask_color_i = None
        self.key_coords = None
        self.marker = None
        self._color_dict = colorDict
        # Scenario we use the background image also for the contours
        if (image_fg is None) and (contour_levels is not None):
            self.image_fg = image_bg

        # Update the axes labels to the units
        AXES_CONF = STANDARD_AXES.copy()
        norm_label = r' $\,/\,{}$'.format(latex_science_float(self.norm_flux)) if self.norm_flux != 1.0 else ''
        AXES_CONF['ylabel'] = f'Flux $({UNITS_LATEX_DICT[self.units_flux]})$' + norm_label
        AXES_CONF['xlabel'] = f'Wavelength $({UNITS_LATEX_DICT[self.units_wave]})$'

        # State the figure and axis format
        self.fig_conf = STANDARD_PLOT.copy()
        self.axes_conf = {'image': {'xlabel': r'RA', 'ylabel': r'DEC', 'title': f'Cube flux slice'},
                          'spectrum': AXES_CONF}

        # Adjust the default theme
        self.fig_conf['figure.figsize'] = (18, 6)

        # Update to the user configuration
        self.fig_conf = {**self.fig_conf, **plt_cfg}

        for plot_type in ('image', 'spectrum'):
            if plot_type in ax_cfg:
                self.axes_conf[plot_type] = {**self.axes_conf[plot_type], **ax_cfg[plot_type]}

        # Prepare the mask correction attributes
        if mask_file is not None:

            assert Path(mask_file).is_file(), f'- ERROR: mask file at {mask_file} not found'

            self.mask_file = mask_file
            with fits.open(self.mask_file) as maskHDUs:

                # Save the fits data to restore later
                for HDU in maskHDUs:
                    if HDU.name != 'PRIMARY':
                        self.mask_dict[HDU.name] = (HDU.data.astype('bool'), HDU.header)
                self.mask_array = np.array(list(self.mask_dict.keys()))

                # Get the target mask
                self.mask_ext = self.mask_array[0]

        # Generate the figure
        with rc_context(self.fig_conf):

            # Figure structure
            self.fig = plt.figure()
            gs = gridspec.GridSpec(nrows=1, ncols=2, figure=self.fig, width_ratios=[1, 2], height_ratios=[1])

            # Spectrum plot
            self.ax1 = self.fig.add_subplot(gs[1])

            # Create subgrid for buttons if mask file provided
            if self.mask_ext is not None:
                gs_image = gridspec.GridSpecFromSubplotSpec(nrows=2, ncols=1, subplot_spec=gs[0], height_ratios=[0.8, 0.2])
            else:
                gs_image = gs

            # Image axes Astronomical coordinates if provided
            if self.header is None:
                self.ax0 = self.fig.add_subplot(gs_image[0])
            else:
                self.ax0 = self.fig.add_subplot(gs_image[0], projection=WCS(self.header), slices=('x', 'y', 1))

            # Buttons axis if provided
            if self.mask_ext is not None:
                self.ax2 = self.fig.add_subplot(gs_image[1])
                radio = RadioButtons(self.ax2, list(self.mask_array))

            # Image mesh grid
            frame_size = self.cube_flux.shape
            y, x = np.arange(0, frame_size[1]), np.arange(0, frame_size[2])
            self.grid_mesh = np.meshgrid(x, y)

            # Use central voxel as initial coordinate
            self.key_coords = int(self.cube_flux.shape[1] / 2), int(self.cube_flux.shape[2] / 2)

            # Load the complete fits lines log if input
            if lines_log_address is not None:
                self.hdul_linelog = fits.open(lines_log_address, lazy_load_hdus=False)

            # Generate the plot
            self.plot_map_voxel(self.image_bg, self.key_coords, self.image_fg, self.contour_levels_fg)

            # Connect the widgets
            self.fig.canvas.mpl_connect('button_press_event', self.on_click)
            self.fig.canvas.mpl_connect('axes_enter_event', self.on_enter_axes)
            if self.mask_file is not None:
                radio.on_clicked(self.mask_selection)

            # Display the figure
            save_close_fig_swicth()

            # Close the lines log if it has been opened
            if isinstance(self.hdul_linelog, fits.hdu.HDUList):
                self.hdul_linelog.close()

        return

    def plot_map_voxel(self, image_bg, voxel_coord=None, image_fg=None, flux_levels=None, frame='observed'):

        # Background image
        self.im = self.ax0.imshow(image_bg, cmap=cm.gray, norm=self.color_norm)

        # Emphasize input coordinate
        idx_j, idx_i = voxel_coord
        if voxel_coord is not None:

            # Delete previous if it is there
            if self.marker is not None:
                self.marker.remove()
                self.marker = None

            self.marker, = self.ax0.plot(idx_i, idx_j, '+', color='red')

        # Plot contours image
        if image_fg is not None:
            self.ax0.contour(self.grid_mesh[0], self.grid_mesh[1], image_fg, cmap='viridis', levels=flux_levels,
                             norm=colors.LogNorm())

        # Voxel spectrum
        if voxel_coord is not None:
            flux_voxel = self.cube_flux[:, idx_j, idx_i]
            self.ax1.step(self.wave, flux_voxel, label='', where='mid', color=self._color_dict['fg'])

        # Plot the emission line fittings:
        if self.hdul_linelog is not None:

            ext_name = f'{idx_j}-{idx_i}{self.ext_log}'

            # Better sorry than permission. Faster?
            try:
                lineslogDF = Table.read(self.hdul_linelog[ext_name]).to_pandas()
                lineslogDF.set_index('index', inplace=True)
                self.log = lineslogDF

            except KeyError:
                self.log = None

            if self.log is not None:

                # Plot all lines encountered in the voxel log
                line_list = self.log.index.values

                # Reference _frame for the plot
                wave_plot, flux_plot, z_corr, idcs_mask = frame_mask_switch(self.wave, flux_voxel, self.redshift, frame)

                # Compute the individual profiles
                wave_array, gaussian_array = gaussian_profiles_computation(line_list, self.log, (1 + self.redshift))
                wave_array, cont_array = linear_continuum_computation(line_list, self.log, (1 + self.redshift))

                # Mask with
                w3_array, w4_array = self.log.w3.values, self.log.w4.values

                # Separating blended from unblended lines
                idcs_nonBlended = (self.log.index.str.endswith('_m')) | (self.log.profile_label == 'no').values

                w3 = self.log.w3.values * (1 + self.redshift)
                w4 = self.log.w4.values * (1 + self.redshift)
                idcsLines = ((w3 - 5) <= wave_plot[:, None]) & (wave_plot[:, None] <= (w4 + 5))

                # Plot single lines
                line_list = self.log.loc[idcs_nonBlended].index
                for line in line_list:

                    i = self.log.index.get_loc(line)

                    # Plot the gauss curve elements
                    wave_i = wave_array[:, i][..., None]
                    cont_i = cont_array[:, i][..., None]
                    gauss_i = gaussian_array[:, i][..., None]

                    line_comps = [line]
                    self._gaussian_profiles_plotting(line_comps, self.log,
                                                     wave_plot[idcsLines[:, i]], flux_plot[idcsLines[:, i]], z_corr,
                                                     axis=self.ax1, cont_bands=None,
                                                     wave_array=wave_i, cont_array=cont_i,
                                                     gaussian_array=gauss_i)

                # Plot combined lines
                profile_list = self.log.loc[~idcs_nonBlended, 'profile_label'].unique()
                for profile_group in profile_list:

                    idcs_group = (self.log.profile_label == profile_group)
                    i_group = np.where(idcs_group)[0]

                    # Determine the line region
                    idcs_plot = ((w3[i_group[0]] - 1) <= wave_plot) & (wave_plot <= (w4[i_group[0]] + 1))

                    # Plot the gauss curve elements
                    wave_i = wave_array[:, i_group[0]:i_group[-1]+1]
                    cont_i = cont_array[:, i_group[0]:i_group[-1]+1]
                    gauss_i = gaussian_array[:, i_group[0]:i_group[-1]+1]

                    line_comps = profile_group.split('-')
                    self._gaussian_profiles_plotting(line_comps, self.log,
                                                     wave_plot[idcs_plot], flux_plot[idcs_plot], z_corr,
                                                     axis=self.ax1, cont_bands=None,
                                                     wave_array=wave_i, cont_array=cont_i,
                                                     gaussian_array=gauss_i)

        # Overplot the mask region
        if self.mask_file is not None:

            inv_masked_array = np.ma.masked_where(~self.mask_dict[self.mask_ext][0], self.image_bg)

            cmap_contours = cm.get_cmap(colorDict['mask_map'], self.mask_array.size)
            idx_color = np.argwhere(self.mask_array == self.mask_ext)[0][0]
            cm_i = colors.ListedColormap([cmap_contours(idx_color)])

            self.ax0.imshow(inv_masked_array, cmap=cm_i, vmin=0, vmax=1, alpha=0.5)

        # Add the mplcursors legend
        if mplcursors_check and (self.hdul_linelog is not None):
            for label, lineProfile in self._legends_dict.items():
                mplcursors.cursor(lineProfile).connect("add", lambda sel, label=label: sel.annotation.set_text(label))

        # Update the axis
        self.axes_conf['spectrum']['title'] = f'Voxel {idx_j} - {idx_i}'
        self.ax0.update(self.axes_conf['image'])
        self.ax1.update(self.axes_conf['spectrum'])

        return

    def on_click(self, event, new_voxel_button=3, mask_button='m'):

        if self.in_ax == self.ax0:

            # Save axes zoom
            self.save_zoom()

            if event.button == new_voxel_button:

                # Save clicked coordinates for next plot
                self.key_coords = np.rint(event.ydata).astype(int), np.rint(event.xdata).astype(int)

                # Remake the drawing
                self.im.remove()# self.ax0.clear()
                self.ax1.clear()
                self.plot_map_voxel(self.image_bg, self.key_coords, self.image_fg, self.contour_levels_fg)

                # Reset the image
                self.reset_zoom()
                self.fig.canvas.draw()

            if event.dblclick:

                if self.mask_file is not None:

                    # Save clicked coordinates for next plot
                    self.key_coords = np.rint(event.ydata).astype(int), np.rint(event.xdata).astype(int)

                    # Add or remove voxel from mask:
                    self.spaxel_selection()

                    # Save the new mask:
                    hdul = fits.HDUList([fits.PrimaryHDU()])
                    for mask_name, mask_attr in self.mask_dict.items():
                        hdul.append(fits.ImageHDU(name=mask_name, data=mask_attr[0].astype(int), ver=1, header=mask_attr[1]))
                    hdul.writeto(self.mask_file, overwrite=True, output_verify='fix')

                    # Remake the drawing
                    self.im.remove()# self.ax0.clear()
                    self.ax1.clear()
                    self.plot_map_voxel(self.image_bg, self.key_coords, self.image_fg, self.contour_levels_fg)

                    # Reset the image
                    self.reset_zoom()
                    self.fig.canvas.draw()

            return

    def mask_selection(self, mask_label):

        # Assign the mask
        self.mask_ext = mask_label

        # Replot the figure
        self.save_zoom()
        self.im.remove()# self.ax0.clear()
        self.ax1.clear()
        self.plot_map_voxel(self.image_bg, self.key_coords, self.image_fg, self.contour_levels_fg)

        # Reset the image
        self.reset_zoom()
        self.fig.canvas.draw()

        return

    def spaxel_selection(self):

        for mask, mask_data in self.mask_dict.items():

            mask_matrix = mask_data[0]
            if mask == self.mask_ext:
                mask_matrix[self.key_coords[0], self.key_coords[1]] = not mask_matrix[self.key_coords[0], self.key_coords[1]]
            else:
                mask_matrix[self.key_coords[0], self.key_coords[1]] = False

        return

    def on_enter_axes(self, event):
        self.in_ax = event.inaxes

    def save_zoom(self):

        self.axlim_dict['image_xlim'] = self.ax0.get_xlim()
        self.axlim_dict['image_ylim'] = self.ax0.get_ylim()
        self.axlim_dict['spec_xlim'] = self.ax1.get_xlim()
        self.axlim_dict['spec_ylim'] = self.ax1.get_ylim()

        return

    def reset_zoom(self):

        self.ax0.set_xlim(self.axlim_dict['image_xlim'])
        self.ax0.set_ylim(self.axlim_dict['image_ylim'])
        self.ax1.set_xlim(self.axlim_dict['spec_xlim'])
        self.ax1.set_ylim(self.axlim_dict['spec_ylim'])

        return