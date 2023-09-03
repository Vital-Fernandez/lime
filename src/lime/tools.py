__all__ = ['COORD_KEYS',
           'unit_conversion',
           'extract_fluxes',
           'normalize_fluxes',
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
def extract_fluxes(log, flux_type='mixture', sample_level='line', column_name='line_flux', column_positions=None):

    if flux_type not in ('mixture', 'intg', 'gauss'):
        raise LiMe_Error(f'Flux type "{flux_type}" is not recognized please one of "intg", "gauss", or "mixture" ')

    # Get indeces of blended lines
    if not isinstance(log.index, pd.MultiIndex):
        idcs_blended = (log['profile_label'].notnull()) & (~log.index.str.endswith('_m'))
    else:
        if sample_level not in log.index.names:
            raise LiMe_Error(f'Input log does not have a index level with column "{sample_level}"')

        idcs_blended = (log['profile_label'].notnull()) & (~log.index.get_level_values('line').str.endswith('_m'))

    # Mixture model: Integrated fluxes for all lines except blended
    if flux_type == 'mixture' and np.any(idcs_blended):
        obsFlux = log['intg_flux'].to_numpy(copy=True)
        obsErr = log['intg_flux_err'].to_numpy(copy=True)
        obsFlux[idcs_blended.values] = log.loc[idcs_blended.values, 'gauss_flux'].to_numpy(copy=True)
        obsErr[idcs_blended.values] = log.loc[idcs_blended.values, 'gauss_flux_err'].to_numpy(copy=True)

    # Use the one requested by the user
    else:
        obsFlux = log[f'{flux_type}_flux'].to_numpy(copy=True)
        obsErr = log[f'{flux_type}_flux_err'].to_numpy(copy=True)

    # Add columns to input dataframe
    output_fluxes = [obsFlux, obsErr]
    if column_name is not None:
        column_positions = 0 if column_positions is None else column_positions
        if column_name in log.columns:
            log.loc[f'{column_name}'] = output_fluxes[0]
            log.loc[f'{column_name}_err'] = output_fluxes[1]
        else:
            log.insert(loc=column_positions, column=f'{column_name}', value=output_fluxes[0])
            log.insert(loc=column_positions + 1, column=f'{column_name}_err', value=output_fluxes[1])
        function_return = None
    else:
        function_return = output_fluxes

    return function_return


def check_lines_normalization(input_lines, norm_line, log):

    # Single or multi-index behaviour
    single_index_check = not isinstance(log.index, pd.MultiIndex)

    # If not input lines use all of them
    if input_lines is None:
        input_lines = log.index.to_numpy() if single_index_check else log.index.get_level_values('line').unique()

    # In case there is only one input line as str
    input_lines = [input_lines] if isinstance(input_lines, str) else input_lines

    # 1) One-norm in norm_line  # 2) Multiple-norm in norm_line # 3) Multiple-norm in input_lines

    # Cases 1 and 2
    if norm_line is not None:

        # Unique normalization
        if isinstance(norm_line, str) or (len(norm_line) == 1):

            if single_index_check:
                line_list = list(log.loc[log.index.isin(input_lines)].index.to_numpy())
            else:
                line_list = list(log.loc[log.index.get_level_values('line').isin(input_lines)].index.get_level_values('line').unique())

            norm_list = [norm_line] * len(line_list)

        # Multiple normalizations
        else:
            if len(input_lines) and len(norm_line):
                line_list, norm_list = [], []
                candidate_lines = log.index if single_index_check else log.index.get_level_values('line')
                for i, line in enumerate(input_lines):
                    if (line in candidate_lines) and (norm_line[i] in candidate_lines):
                        line_list.append(line)
                        norm_list.append(norm_line[i])
            else:
                raise LiMe_Error(f'The number of normalization lines does not match the number of input lines:\n'
                                   f'- Model lines ({input_lines}): {input_lines}\n'
                                   f'- Norm lines  ({len(norm_line)}): {norm_line}')

    # Case 3
    else:

        # Split the nominators and denominators
        line_list, norm_list = [], []
        for ratio in input_lines:

            if '/' not in ratio:
                raise LiMe_Error(f'Input line list must use "/" with their normalization (for example H1_6563A/H1_4861A)\n'
                                   f'The input ratio: {ratio} does not have it. Try to specify a "norm_list" argument instead.')

            nomin_i, denom_i = ratio.replace(" ", "").split('/')
            line_list.append(nomin_i)
            norm_list.append(denom_i)

    return line_list, norm_list


# Compute the fluxes
def normalize_fluxes(log, line_list=None, norm_list=None, flux_column='gauss_flux', column_name='line_flux',
                     column_position=0, column_normalization_name='norm_line', sample_levels=['id', 'line']):

    '''
    If the normalization line is not available, no operation is added.
    '''

    # Check columns present in log
    if (flux_column not in log.columns) or (f'{flux_column}_err' not in log.columns):
        raise LiMe_Error(f'Input log is missing "{flux_column}" or "{flux_column}_err" columns')

    # Check the normalization for the lines
    line_array, norm_array = check_lines_normalization(line_list, norm_list, log)

    # Add new columns if necessary
    if column_name not in log.columns:
        log.insert(loc=column_position, column=f'{column_name}', value=np.nan)
        log.insert(loc=column_position+1, column=f'{column_name}_err', value=np.nan)

    # Add new line with normalization by default
    if column_normalization_name not in log.columns:
        log.insert(loc=column_position+2, column=column_normalization_name, value=np.nan)

    # Loop throught the lines to compute their normalization
    single_index = not isinstance(log.index, pd.MultiIndex)
    for i in np.arange(len(line_array)):

            numer, denom = line_array[i], norm_array[i]
            numer_flux, denom_flux = None, None

            # Single-index dataframe
            if single_index:
                idcs_ratios = numer
                if (numer in log.index) and (denom in log.index):
                    numer_flux = log.loc[numer, flux_column]
                    numer_err = log.loc[numer, f'{flux_column}_err']

                    denom_flux = log.loc[denom, flux_column]
                    denom_err = log.loc[denom, f'{flux_column}_err']

            #Multi-index dataframe
            else:

                if numer == denom:  # Same line normalization
                    idcs_slice = log.index.get_level_values(sample_levels[-1]) == numer
                    df_slice = log.loc[idcs_slice]
                else:  # Rest cases
                    idcs_slice = log.index.get_level_values(sample_levels[-1]).isin([numer, denom])
                    grouper = log.index.droplevel('line')
                    idcs_slice = pd.Series(idcs_slice).groupby(grouper).transform('sum').ge(2).array
                    df_slice = log.loc[idcs_slice]

                # Get fluxes
                if df_slice.size > 0:
                    num_slice = df_slice.xs(numer, level=sample_levels[-1], drop_level=False)
                    numer_flux = num_slice[flux_column].to_numpy()
                    numer_err = num_slice[f'{flux_column}_err'].to_numpy()

                    denom_slice = df_slice.xs(denom, level=sample_levels[-1], drop_level=False)
                    denom_flux = denom_slice[flux_column].to_numpy()
                    denom_err = denom_slice[f'{flux_column}_err'].to_numpy()

                    idcs_ratios = num_slice.index

            # Compute the ratios with error propagation
            ratio_array, ratio_err = None, None
            if (numer_flux is not None) and (denom_flux is not None):
                ratio_array = numer_flux / denom_flux
                ratio_err = ratio_array * np.sqrt(np.power(numer_err / numer_flux, 2) + np.power(denom_err / denom_flux, 2))

            # Store in dataframe (with empty columns)
            if (ratio_array is not None) and (ratio_err is not None):
                log.loc[idcs_ratios, f'{column_name}'] = ratio_array
                log.loc[idcs_ratios, f'{column_name}_err'] = ratio_err

                # Store normalization line
                if column_normalization_name is not None:
                    log.loc[idcs_ratios, f'{column_normalization_name}'] = denom

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
            df_slice = input_log.xs(idx, level=sample_levels[0])

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


# def compute_line_ratios(log, line_ratios=None, flux_columns=['intg_flux', 'intg_flux_err'], sample_levels=['id', 'line'],
#                         object_id='obj_0', keep_empty_columns=True):
#
#     # If normalization_line is not none
#     if len(flux_columns) != np.sum(log.columns.isin(flux_columns)):
#         raise LiMe_Error(f'Input log is missing {len(flux_columns)} "flux_entries" in the column headers')
#
#     # Check if single or multi-index
#     sample_check = isinstance(log.index, pd.MultiIndex)
#
#     if sample_check:
#         idcs = log.index.droplevel(sample_levels[-1]).unique()
#     else:
#         idcs = np.array([object_id])
#
#     ratio_df = pd.DataFrame(index=idcs)
#
#     # Loop through
#     if line_ratios is not None:
#
#         # Loop through the ratios
#         for ratio_str in line_ratios:
#             numer, denom = ratio_str.split('/')
#
#             # Slice the dataframe to objects having both lines
#             numer_flux, denom_flux = None, None
#             if not isinstance(log.index, pd.MultiIndex):
#                 if (numer in log.index) and (denom in log.index):
#                     numer_flux = log.loc[numer, flux_columns[0]]
#                     numer_err = log.loc[numer, flux_columns[1]]
#
#                     denom_flux = log.loc[denom, flux_columns[0]]
#                     denom_err = log.loc[denom, flux_columns[1]]
#                     idcs_slice = object_id
#                 else:
#                     idcs_slice = ratio_df.index
#
#             else:
#
#                 # Slice the dataframe to objects which have both lines
#                 idcs_slice = log.index.get_level_values(sample_levels[-1]).isin([numer, denom])
#                 grouper = log.index.droplevel('line')
#                 idcs_slice = pd.Series(idcs_slice).groupby(grouper).transform('sum').ge(2).array
#                 df_slice = log.loc[idcs_slice]
#
#                 # Get fluxes
#                 if df_slice.size > 0:
#                     numer_flux = df_slice.xs(numer, level=sample_levels[-1])[flux_columns[0]]
#                     numer_err = df_slice.xs(numer, level=sample_levels[-1])[flux_columns[1]]
#
#                     denom_flux = df_slice.xs(denom, level=sample_levels[-1])[flux_columns[0]]
#                     denom_err = df_slice.xs(denom, level=sample_levels[-1])[flux_columns[1]]
#                     idcs_slice = numer_flux.index
#                 else:
#                     idcs_slice = ratio_df.index
#
#             # Check there have been measure
#             if (numer_flux is not None) and (denom_flux is not None):
#
#                 # Compute the ratios with the error propagation
#                 ratio_array = numer_flux/denom_flux
#                 errRatio_array = ratio_array * np.sqrt(np.power(numer_err/numer_flux, 2) +
#                                                        np.power(denom_err/denom_flux, 2))
#             else:
#                 ratio_array, errRatio_array = np.nan, np.nan
#
#             # Store in dataframe (with empty columns)
#             if not ((numer_flux is None) and (denom_flux is None) and (keep_empty_columns is False)):
#                 ratio_df.loc[idcs_slice, ratio_str] = ratio_array
#                 ratio_df.loc[idcs_slice, f'{ratio_str}_err'] = errRatio_array
#
#     return ratio_df


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
    profile_label = np.nan

    if line in log.index:

        if 'profile_label' in log.columns:

            if log.loc[line, 'profile_label'] is np.nan:
                profile_label = np.nan
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


def unit_conversion(in_units, out_units, wave_array=None, flux_array=None, dispersion_units=None, decimals=None,
                    mask_check=False):

    """

    This function converts the input array (wavelength or flux) ``in_units`` into the requested ``out_units``.

    .. attention::
        Due to the nature of the ``flux_array``, the user also needs to include the ``wave_array`` and its units in the
        ``dispersion_units``. units

    The user can also provide the number of ``decimals`` to round the output array.

    :param in_units: Input array units
    :type in_units: str

    :param out_units: Output array untis
    :type out_units: str

    :param wave_array: Wavelength array
    :type wave_array: numpy.array

    :param flux_array: Flux array
    :type flux_array: numpy.array

    :param dispersion_units:
    :type dispersion_units:

    :param decimals: Number of decimals.
    :type decimals: int, optional

    :param mask_check: Re-apply the numpy array mask to the output array. The default value is True.
    :type mask_check: bool, optional

    """


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

    if decimals is None:
        return output_array
    else:
        return np.round(output_array, decimals)


def refraction_index_air_vacuum(wavelength_array, units='A'):

    # TODO add warnings issues with units

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
        message = f'[{"=" * int(n_bar * j):{n_bar}s}] {int(100 * j)}% of {self.count_type} ({post_text})'
        stdout.write(message)
        stdout.flush()

        return

    def counter(self, i, i_max, pre_text="", post_text="", n_bar=10):
        print(f'{i + 1}/{self.count_type}) {post_text}')

        return

    def no_output(self, i, i_max, pre_text, post_text, n_bar=10):

        return
