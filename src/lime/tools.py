__all__ = ['COORD_KEYS',
           'unit_convertor',
           'extract_fluxes',
           'relative_fluxes',
           'compute_line_ratios',
           'redshift_calculation']

import logging
import numpy as np
import pandas as pd

from .io import LiMe_Error
from sys import stdout
from astropy import units as au

_logger = logging.getLogger('LiMe')

VAL_LIST = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
SYB_LIST = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]

ASTRO_UNITS_KEYS = {'A': au.AA,
                    'um': au.um,
                    'nm': au.nm,
                    'Hz': au.Hz,
                    'cm': au.cm,
                    'mm': au.mm,
                    'Flam': au.erg/au.s/au.cm**2/au.AA,
                    'Fnu': au.erg/au.s/au.cm**2/au.Hz,
                    'Jy': au.Jy,
                    'mJy': au.mJy,
                    'nJy': au.nJy}

UNITS_LATEX_DICT = {'A': r'\AA',
                    'um': r'\mu\!m',
                    'nm': 'nm',
                    'Hz': 'Hz',
                    'cm': 'cm',
                    'mm': 'mm',
                    'Flam': r'erg\,cm^{-2}s^{-1}\AA^{-1}',
                    'Fnu': r'erg\,cm^{-2}s^{-1}\Hz^{-1}',
                    'Jy': 'Jy',
                    'mJy': 'mJy',
                    'nJy': 'nJy'}

DISPERSION_UNITS = ('A', 'um', 'nm', 'Hz', 'cm', 'mm')

FLUX_DENSITY_UNITS = ('Flam', 'Fnu', 'Jy', 'mJy', 'nJy')

PARAMETER_LATEX_DICT = {'Flam': r'$F_{\lambda}$',
                        'Fnu': r'$F_{\nu}$',
                        'SN_line': r'$\frac{S}{N}_{line}$',
                        'SN_cont': r'$\frac{S}{N}_{cont}$'}

# Variables with the astronomical coordinate information for the creation of new .fits files
COORD_KEYS = ['CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2', 'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2', 'CUNIT1', 'CUNIT2',
              'CTYPE1', 'CTYPE2']


# Number conversion to Roman style
def int_to_roman(num):
    i, roman_num = 0, ''
    while num > 0:
        for _ in range(num // VAL_LIST[i]):
            roman_num += SYB_LIST[i]
            num -= VAL_LIST[i]
        i += 1
    return roman_num


# Favoured method to get line fluxes according to resolution
def extract_fluxes(log, flux_type='mixture', sample_level='line', column_names=None, column_positions=None):

    # Get indeces of blended lines
    if not isinstance(log.index, pd.MultiIndex):
        idcs_blended = (log['profile_label'] != 'no') & (~log.index.str.endswith('_m'))

    else:
        if sample_level not in log.index.names:
            raise LiMe_Error(f'Input log does not have a index level with column "{sample_level}"')

        idcs_blended = (log['profile_label'] != 'no') & (~log.index.get_level_values('line').str.endswith('_m'))

    # Mixture model: Integrated fluxes for all lines except blended
    if flux_type == 'mixture' and np.any(idcs_blended):
        obsFlux = log['intg_flux'].to_numpy(copy=True)
        obsErr = log['intg_err'].to_numpy(copy=True)
        obsFlux[idcs_blended.values] = log.loc[idcs_blended.values, 'gauss_flux'].to_numpy(copy=True)
        obsErr[idcs_blended.values] = log.loc[idcs_blended.values, 'gauss_err'].to_numpy(copy=True)

    # Use the one requested by the user
    else:
        obsFlux = log[f'{flux_type}_flux'].to_numpy(copy=True)
        obsErr = log[f'{flux_type}_err'].to_numpy(copy=True)

    output_fluxes = [obsFlux, obsErr]

    # Add columns to input dataframe
    if column_names is not None:
        if column_positions is not None:
            for i, pos_i in enumerate(column_positions):
                if column_names[i] not in log.columns:
                    log.insert(loc=pos_i, column=column_names[i], value=output_fluxes[i])
                else:
                    log[column_names[i]] = output_fluxes[i]
                    # log.insert(loc=column_positions[0], column=column_names[0], value=obsFlux)

        else:
            log[column_names[0]] = obsFlux
            log[column_names[1]] = obsErr

        function_return = None

    else:
        function_return = obsFlux, obsErr

    return function_return


# Compute the fluxes
def relative_fluxes(log, normalization_line, flux_entries=['intg_flux', 'intg_err'], column_names=None,
                    column_positions=None):

    '''
    If the normalization line is not available, no operation is added.
    '''

    # If normalization_line is not none
    if len(flux_entries) != np.sum(log.columns.isin(flux_entries)):
        raise LiMe_Error(f'Input log is missing {len(flux_entries)} "flux_entries" in the column headers')

    # Container for params
    nflux_array, nErr_array = None, None

    # Single index dataframes
    if not isinstance(log.index, pd.MultiIndex):
        idcs_slice = log.index

        if normalization_line in idcs_slice:
            nflux_array = log.loc[idcs_slice, flux_entries[0]]/log.loc[normalization_line, flux_entries[0]]
            errLog_n = np.power(log.loc[idcs_slice, flux_entries[1]]/log.loc[idcs_slice, flux_entries[0]], 2)
            errNorm_n = np.power(log.loc[normalization_line, flux_entries[1]]/log.loc[normalization_line, flux_entries[0]], 2)
            nErr_array = nflux_array * np.sqrt(errLog_n + errNorm_n)

    # Multi-index dataframes
    else:
        log_slice = log.xs(normalization_line, level="line")
        idcs_slice = log_slice.index

        if len(log_slice) > 0:
            nflux_array = log.loc[idcs_slice, flux_entries[0]]/log_slice[flux_entries[0]]
            errLog_n = np.power(log.loc[idcs_slice, flux_entries[1]]/log.loc[idcs_slice, flux_entries[0]], 2)
            errNorm_n = np.power(log_slice[flux_entries[1]]/log_slice[flux_entries[0]], 2)
            nErr_array = nflux_array * np.sqrt(errLog_n + errNorm_n)

    # Confirm lines were normalized
    if nflux_array is None:
        _logger.info(f'The normalization line {normalization_line} is not found on the input log')

    # Check for column names
    if column_names is None:
        column_names = [f'n{flux_entries[0]}', f'n{flux_entries[1]}']

    # Add columns to input dataframe
    if nflux_array is not None:
        if (column_positions is not None) and (column_names[0] not in log.columns):
            log.insert(loc=column_positions[0], column=column_names[0], value=np.nan)
            log.insert(loc=column_positions[1], column=column_names[1], value=np.nan)

        log.loc[idcs_slice, column_names[0]] = nflux_array
        log.loc[idcs_slice, column_names[1]] = nErr_array

    return


# Get Weighted redshift from lines
def redshift_calculation(input_log, line_list=None, weight_parameter=None, sample_levels=['id', 'line'], obj_label='spec_0'):

    #TODO accept LiME objects as imput log

    # Check the weighted parameter presence
    if weight_parameter is not None:
        if weight_parameter not in input_log.columns:
            raise LiMe_Error(f'The parameter {weight_parameter} is not found on the input lines log headers')

    # Check input line is not a string
    line_list = np.array(line_list, ndmin=1) if isinstance(line_list, str) else line_list

    # Check if single or multi-index
    sample_check = isinstance(input_log.index, pd.MultiIndex)

    if sample_check:
        id_list = input_log.index.droplevel(sample_levels[-1]).unique()
    else:
        id_list = np.array([obj_label])

    # Container for redshifts
    z_df = pd.DataFrame(index=id_list, columns=['z_mean', 'z_std', 'lines', 'weight'])
    if sample_check:
        z_df.rename_axis(index=sample_levels[:-1], inplace=True)

    # Loop through the ids
    for idx in id_list:

        # Slice to the object log
        if not sample_check:
            df_slice = input_log
        else:
            df_slice = input_log.xs(idx, level=sample_levels[:-1])

        # Get the lines requested
        if line_list is not None:
            idcs_slice = df_slice.index.isin(line_list)
            df_slice = df_slice.loc[idcs_slice]
        else:
            df_slice = df_slice

        # Check the line has lines
        n_lines = len(df_slice.index)
        if n_lines > 0:
            z_array = (df_slice['center']/df_slice['wavelength'] - 1).to_numpy()
            obsLineList = ','.join(df_slice.index.values)

            # Just one line
            if n_lines == 1:
                z_mean = z_array[0]
                z_std = df_slice.center_err.to_numpy()[0]/df_slice.wavelength.to_numpy()[0]

            # Multiple lines
            else:

                # Not weighted parameter
                if weight_parameter is None:
                    z_mean = z_array.mean()
                    z_std = z_array.std()

                # With a weighted parameter
                else:
                    w_array = df_slice[weight_parameter]
                    z_err_array = df_slice.center_err.to_numpy()/df_slice.wavelength.to_numpy()

                    z_mean = np.sum(w_array * z_array)/np.sum(w_array)
                    z_std = np.sqrt(np.sum(np.power(w_array, 2) * np.power(z_err_array, 2)) / np.sum(np.power(w_array, 2)))

        else:
            z_mean, z_std, obsLineList = np.nan, np.nan, None

        # Add to dataframe
        z_df.loc[idx, 'z_mean':'weight'] = z_mean, z_std, obsLineList, weight_parameter

    return z_df


def compute_line_ratios(log, line_ratios=None, flux_columns=['intg_flux', 'intg_err'], sample_levels=['id', 'line'],
                        object_id='obj_0', keep_empty_columns=True):

    # If normalization_line is not none
    if len(flux_columns) != np.sum(log.columns.isin(flux_columns)):
        raise LiMe_Error(f'Input log is missing {len(flux_columns)} "flux_entries" in the column headers')

    # Check if single or multi-index
    sample_check = isinstance(log.index, pd.MultiIndex)

    if sample_check:
        idcs = log.index.droplevel(sample_levels[-1]).unique()
    else:
        idcs = np.array([object_id])

    ratio_df = pd.DataFrame(index=idcs)

    # Loop through
    if line_ratios is not None:

        # Loop through the ratios
        for ratio_str in line_ratios:
            numer, denom = ratio_str.split('/')

            # Slice the dataframe to objects having both lines
            numer_flux, denom_flux = None, None
            if not isinstance(log.index, pd.MultiIndex):
                if (numer in log.index) and (denom in log.index):
                    numer_flux = log.loc[numer, flux_columns[0]]
                    numer_err = log.loc[numer, flux_columns[1]]

                    denom_flux = log.loc[denom, flux_columns[0]]
                    denom_err = log.loc[denom, flux_columns[1]]
                    idcs_slice = object_id
                else:
                    idcs_slice = ratio_df.index

            else:

                # Slice the dataframe to objects which have both lines
                idcs_slice = log.index.get_level_values(sample_levels[-1]).isin([numer, denom])
                grouper = log.index.droplevel('line')
                idcs_slice = pd.Series(idcs_slice).groupby(grouper).transform('sum').ge(2).array
                df_slice = log.loc[idcs_slice]

                # Get fluxes
                if df_slice.size > 0:
                    numer_flux = df_slice.xs(numer, level=sample_levels[-1])[flux_columns[0]]
                    numer_err = df_slice.xs(numer, level=sample_levels[-1])[flux_columns[1]]

                    denom_flux = df_slice.xs(denom, level=sample_levels[-1])[flux_columns[0]]
                    denom_err = df_slice.xs(denom, level=sample_levels[-1])[flux_columns[1]]
                    idcs_slice = numer_flux.index
                else:
                    idcs_slice = ratio_df.index

            # Check there have been measure
            if (numer_flux is not None) and (denom_flux is not None):

                # Compute the ratios with the error propagation
                ratio_array = numer_flux/denom_flux
                errRatio_array = ratio_array * np.sqrt(np.power(numer_err/numer_flux, 2) +
                                                       np.power(denom_err/denom_flux, 2))
            else:
                ratio_array, errRatio_array = np.nan, np.nan

            # Store in dataframe (with empty columns)
            if not ((numer_flux is None) and (denom_flux is None) and (keep_empty_columns is False)):
                ratio_df.loc[idcs_slice, ratio_str] = ratio_array
                ratio_df.loc[idcs_slice, f'{ratio_str}_err'] = errRatio_array

    return ratio_df


def compute_FWHM0(idx_peak, spec_flux, delta_wave, cont_flux, emission_check=True):

    """

    :param idx_peak:
    :param spec_flux:
    :param delta_wave:
    :param cont_flux:
    :param emission_check:
    :return:
    """

    i = idx_peak
    i_final = 0 if delta_wave < 0 else spec_flux.size - 1

    if emission_check:
        while (spec_flux[i] >= cont_flux[i]) and (i != i_final):
            i += delta_wave
    else:
        while (spec_flux[i] <= cont_flux[i]) and (i != i_final):
            i += delta_wave

    return i


def blended_label_from_log(line, log):

    # Default values: single line
    blended_check = False
    profile_label = 'no'

    if line in log.index:

        if 'profile_label' in log.columns:

            if log.loc[line, 'profile_label'] == 'no':
                profile_label = 'no'
            elif line.endswith('_m'):
                profile_label = log.loc[line, 'profile_label']
            else:
                blended_check = True
                profile_label = log.loc[line, 'profile_label']
    else:
        # TODO this causes and error if we forget the '_b' componentes in the configuration file need to check input cfg
        _logger.warning(f'The line {line} was not found on the input log. If you are specifying the components of a '
                        f'blended line in the fitting configuration, make sure you are not missing the "_b" subscript')

    return blended_check, profile_label


def latex_science_float(f, dec=2):
    float_str = f'{f:.{dec}g}'
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str


def unit_convertor(in_units, out_units, wave_array=None, flux_array=None, dispersion_units=None, sig_fig=None,
                   mask_check=False):

    # Converting the wavelength array
    if (in_units in DISPERSION_UNITS) and (out_units in DISPERSION_UNITS):
        input_mask = wave_array.mask if mask_check else None
        input_array = wave_array * ASTRO_UNITS_KEYS[in_units]
        output_array = input_array.to(ASTRO_UNITS_KEYS[out_units])

    # Converting the flux array
    elif (in_units in FLUX_DENSITY_UNITS) and (out_units in FLUX_DENSITY_UNITS):
        input_mask = flux_array.mask if mask_check else None
        input_array = flux_array * ASTRO_UNITS_KEYS[in_units]
        wave_unit_array = wave_array * ASTRO_UNITS_KEYS[dispersion_units]
        output_array = input_array.to(ASTRO_UNITS_KEYS[out_units], au.spectral_density(wave_unit_array))

    # Not recognized units
    else:
        _logger.warning(f'Input units {in_units} could not be converted to {out_units}')

    # Reapply the mask if necessary
    if mask_check:
        output_array = np.ma.masked_array(output_array.value, input_mask)
    else:
        output_array = output_array.value

    if sig_fig is None:
        return output_array
    else:
        return np.round(output_array, sig_fig)


def refraction_index_air_vacuum(wavelength_array, units='A'):

    refraction_index = (1 + 1e-6 * (287.6155 + 1.62887 / np.power(wavelength_array * 0.0001, 2) + 0.01360 / np.power(wavelength_array * 0.0001, 4)))

    return refraction_index


def format_line_mask_option(entry_value, wave_array):

    # Check if several entries
    formatted_value = entry_value.split(',') if ',' in entry_value else [f'{entry_value}']

    # Check if interval or single pixel mask
    for i, element in enumerate(formatted_value):
        if '-' in element:
            formatted_value[i] = element.split('-')
        else:
            element = float(element)
            pix_width = (np.diff(wave_array).mean())/2
            formatted_value[i] = [element-pix_width, element+pix_width]

    formatted_value = np.array(formatted_value).astype(float)

    return formatted_value


def define_masks(wavelength_array, masks_array, merge_continua=True, line_mask_entry='no'):

    # Make sure it is a matrix
    # TODO warning for mask outside limimes
    masks_array = np.array(masks_array, ndmin=2)

    # Check if it is a masked array
    if np.ma.is_masked(wavelength_array):
        wave_arr = wavelength_array.data
    else:
        wave_arr = wavelength_array

    # Remove masked pixels from this function wavelength array
    if line_mask_entry != 'no':

        # Convert cfg mask string to limits
        line_mask_limits = format_line_mask_option(line_mask_entry, wave_arr)

        # Get masked indeces
        idcsMask = (wave_arr[:, None] >= line_mask_limits[:, 0]) & (wave_arr[:, None] <= line_mask_limits[:, 1])
        idcsValid = ~idcsMask.sum(axis=1).astype(bool)[:, None]

    else:
        idcsValid = np.ones(wave_arr.size).astype(bool)[:, None]

    # Find indeces for six points in spectrum
    idcsW = np.searchsorted(wave_arr, masks_array)

    # Emission region
    idcsLineRegion = ((wave_arr[idcsW[:, 2]] <= wave_arr[:, None]) & (wave_arr[:, None] <= wave_arr[idcsW[:, 3]]) & idcsValid).squeeze()
    
    # Return left and right continua merged in one array
    if merge_continua:

        idcsContRegion = (((wave_arr[idcsW[:, 0]] <= wave_arr[:, None]) &
                          (wave_arr[:, None] <= wave_arr[idcsW[:, 1]])) |
                          ((wave_arr[idcsW[:, 4]] <= wave_arr[:, None]) & (
                           wave_arr[:, None] <= wave_arr[idcsW[:, 5]])) & idcsValid).squeeze()

        return idcsLineRegion, idcsContRegion

    # Return left and right continua in separated arrays
    else:

        idcsContLeft = ((wave_arr[idcsW[:, 0]] <= wave_arr[:, None]) & (wave_arr[:, None] <= wave_arr[idcsW[:, 1]]) & idcsValid).squeeze()
        idcsContRight = ((wave_arr[idcsW[:, 4]] <= wave_arr[:, None]) & (wave_arr[:, None] <= wave_arr[idcsW[:, 5]]) & idcsValid).squeeze()

        return idcsLineRegion, idcsContLeft, idcsContRight


class ProgressBar:

    def __init__(self, message_type=None, count_type=""):

        self.output_message = None
        self.count_type = count_type

        if message_type is None:
            self.output_message = self.no_output

        if message_type == 'bar':
            self.output_message = self.progress_bar

        if message_type == 'counter':
            self.output_message = self.counter

        return

    def progress_bar(self, i, i_max, pre_text, post_text, n_bar=10):

        # TODO this might be a bit slower for IFU coords, get input coords as kwargs

        j = (i + 1) / i_max
        stdout.write('\r')
        message = f'[{"=" * int(n_bar * j):{n_bar}s}] {int(100 * j)}% of {self.count_type}'
        stdout.write(message)
        stdout.flush()

        return

    def counter(self, i, i_max, pre_text="", post_text="", n_bar=10):
        print(f'{i + 1}/{self.count_type}) {post_text}')

        return

    def no_output(self, i, i_max, pre_text, post_text, n_bar=10):

        return
