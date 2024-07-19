__all__ = ['unit_conversion',
           'extract_fluxes',
           'normalize_fluxes',
           'redshift_calculation',
           'save_parameter_maps']

import logging
import numpy as np
import pandas as pd

from .io import LiMe_Error, load_frame, log_to_HDU
from sys import stdout

from astropy import units as au
from astropy.units.core import CompositeUnit, IrreducibleUnit, Unit

from astropy.io import fits
from pathlib import Path

_logger = logging.getLogger('LiMe')

# Arrays for roman numerals conversion
VAL_LIST = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
SYB_LIST = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]


dict_units = {'flam': au.erg/au.s/au.cm**2/au.AA, 'FLAM': au.erg/au.s/au.cm**2/au.AA,
              'fnu': au.erg/au.s/au.cm**2/au.Hz, 'FNU': au.erg/au.s/au.cm**2/au.Hz,
              'photlam': au.photon/au.s/au.cm**2/au.AA, 'PHOTLAM': au.photon/au.s/au.cm**2/au.AA,
              'photnu': au.photon/au.s/au.cm**2/au.Hz, 'PHOTNU': au.photon/au.s/au.cm**2/au.Hz}
au.set_enabled_aliases(dict_units)

# flam = au.def_unit(['flam', 'FLAM'], au.erg/au.s/au.cm**2/au.AA,
#                     format={"latex": r"erg\,cm^{-2}s^{-1}\AA^{-1}",
#                             "generic": "FLAM", "console": "FLAM"})
#
# fnu = au.def_unit(['fnu', 'FNU'], au.erg/au.s/au.cm**2/au.Hz,
#                     format={"latex": r"erg\,cm^{-2}s^{-1}Hz^{-1}",
#                             "generic": "FNU", "console": "FNU"})
#
# photlam = au.def_unit(['photlam', 'PHOTLAM'], au.photon/au.s/au.cm**2/au.AA,
#                         format={"latex": r"photon\,cm^{-2}s^{-1}\AA^{-1}",
#                         "generic": "PHOTLAM", "console": "PHOTLAM"})
#
# photnu = au.def_unit(['photnu', 'PHOTNU'], au.photon/au.s/au.cm**2/au.Hz,
#                         format={"latex": r"photon\,cm^{-2}s^{-1}Hz^{-1}",
#                         "generic": "PHOTNU", "console": "PHOTNU"})
#
# au.add_enabled_units([flam, fnu, photlam, photnu])


PARAMETER_LATEX_DICT = {'Flam': r'$F_{\lambda}$',
                        'Fnu': r'$F_{\nu}$',
                        'SN_line': r'$\frac{S}{N}_{line}$',
                        'SN_cont': r'$\frac{S}{N}_{cont}$'}


def mult_err_propagation(nominal_array, err_array, result):

    err_result = result * np.sqrt(np.sum(np.power(err_array/nominal_array, 2)))

    return err_result

# Number conversion to Roman style
def int_to_roman(num):
    i, roman_num = 0, ''
    while num > 0:
        for _ in range(num // VAL_LIST[i]):
            roman_num += SYB_LIST[i]
            num -= VAL_LIST[i]
        i += 1
    return roman_num


def pd_get(df, row, column, default=None, transform=None):

    # Fast get from dataframe
    try:
        cell = df.at[row, column]
    except KeyError:
        cell = default

    # Transform the value from the dataframe to the default if requested
    if transform is not None:
        cell = default if cell == transform else cell

    return cell


# Favoured method to get line fluxes according to resolution
def extract_fluxes(log, flux_type='mixture', sample_level='line', column_name='line_flux', column_positions=None):

    if flux_type not in ('mixture', 'intg', 'profile'):
        raise LiMe_Error(f'Flux type "{flux_type}" is not recognized please one of "intg", "profile", or "mixture" ')

    # Get indeces of blended lines
    if not isinstance(log.index, pd.MultiIndex):
        idcs_blended = (log['group_label'] != 'none') & (~log.index.str.endswith('_m'))
    else:
        if sample_level not in log.index.names:
            raise LiMe_Error(f'Input log does not have a index level with column "{sample_level}"')

        idcs_blended = (log['group_label'] != 'none') & (~log.index.get_level_values('line').str.endswith('_m'))

    # Mixture models: Integrated fluxes for all lines except blended
    if flux_type == 'mixture' and np.any(idcs_blended):
        obsFlux = log['intg_flux'].to_numpy(copy=True)
        obsErr = log['intg_flux_err'].to_numpy(copy=True)
        obsFlux[idcs_blended.values] = log.loc[idcs_blended.values, 'profile_flux'].to_numpy(copy=True)
        obsErr[idcs_blended.values] = log.loc[idcs_blended.values, 'profile_flux_err'].to_numpy(copy=True)

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
def normalize_fluxes(log, line_list=None, norm_list=None, flux_column='profile_flux', column_name='line_flux',
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
        log.insert(loc=column_position+2, column=column_normalization_name, value='none')

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
                idcs_slice = pd.Series(idcs_slice).groupby(grouper).transform('sum').ge(2).to_numpy()
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
def redshift_calculation(input_log, line_list=None, weight_parameter=None, obj_label='spec_0'):

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
        levels = input_log.index.names
        id_list = input_log.index.droplevel(levels[-1]).unique()
    else:
        id_list = np.array([obj_label])
        levels = None

    # Container for redshifts
    z_df = pd.DataFrame(index=id_list, columns=['z_mean', 'z_std', 'lines', 'weight'])
    if sample_check:
        z_df.rename_axis(index=levels[:-1], inplace=True)

    # Loop through the ids
    for idx in id_list:

        # Slice to the object log
        if not sample_check:
            df_slice = input_log
        else:
            df_slice = input_log.xs(idx, level=levels[0])

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
    group_label = None

    if line in log.index:

        if 'group_label' in log.columns:

            if log.loc[line, 'group_label'] == 'none':
                group_label = None
            elif line.endswith('_m'):
                group_label = log.loc[line, 'group_label']
            else:
                blended_check = True
                group_label = log.loc[line, 'group_label']
    else:
        # TODO this causes and error if we forget the '_b' componentes in the configuration file need to check input cfg
        _logger.warning(f'The line {line} was not found on the input log. If you are specifying the components of a '
                        f'blended line in the fitting configuration, make sure you are not missing the "_b" subscript')

    return blended_check, group_label


def latex_science_float(f, dec=2):
    float_str = f'{f:.{dec}g}'
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str


def unit_conversion(in_units, out_units, wave_array=None, flux_array=None, dispersion_units=None, decimals=None):

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

    """

    # Converting the wavelength array
    if flux_array is None:
        input_mask = wave_array.mask if np.ma.isMaskedArray(wave_array) else None
        input_array = wave_array * in_units if input_mask is None else wave_array.data * in_units
        output_array = input_array.to(au.Unit(out_units))
        output_array = output_array.value  # Remove the units

    # Converting the flux array
    else:
        input_mask = flux_array.mask if np.ma.isMaskedArray(flux_array) else None
        input_array = flux_array * in_units if input_mask is None else flux_array.data * in_units
        w_array = wave_array.data * dispersion_units if np.ma.isMaskedArray(wave_array) else wave_array * dispersion_units
        output_array = input_array.to(au.Unit(out_units), au.spectral_density(w_array))
        output_array = output_array.value  # Remove the units

    # Reapply the mask
    output_array = output_array if input_mask is None else np.ma.masked_array(output_array, input_mask)

    # Round to decimal places
    output_array = output_array if decimals is None else np.round(output_array, decimals)

    return output_array


def refraction_index_air_vacuum(wavelength_array, units='A'):

    # TODO add warnings issues with units

    refraction_index = (1 + 1e-6 * (287.6155 + 1.62887 / np.power(wavelength_array * 0.0001, 2) + 0.01360 / np.power(wavelength_array * 0.0001, 4)))

    return refraction_index





# def define_masks(wavelength_array, masks_array, merge_continua=True, line_mask_entry='no', line=None):
#
#     # TODO we might delete this one and leave the transitions one
#     if masks_array is None:
#         raise LiMe_Error()
#
#     # Make sure it is a matrix
#     masks_array = np.atleast_2d(masks_array)
#
#     if np.any(masks_array[:, 0] < wavelength_array[0]) or np.any(masks_array[:, 5] > wavelength_array[-1]):
#         _logger.warning(f'The {line} bands do not match the spectrum wavelength range (observed):')
#         _logger.warning(f'-- The spectrum wavelength range is: ({wavelength_array[0]:0.2f}, {wavelength_array[-1]:0.2f}) (observed frame)')
#         _logger.warning(f'-- The {line} bands are: {masks_array} (rest frame * (1 + z))')
#
#     # Check if it is a masked array
#     if np.ma.isMaskedArray(wavelength_array):
#         wave_arr = wavelength_array.data
#     else:
#         wave_arr = wavelength_array
#
#     # Remove masked pixels from this function wavelength array
#     if line_mask_entry != 'no':
#
#         # Convert cfg mask string to limits
#         line_mask_limits = format_line_mask_option(line_mask_entry, wave_arr)
#
#         # Get masked indeces
#         idcsMask = (wave_arr[:, None] >= line_mask_limits[:, 0]) & (wave_arr[:, None] <= line_mask_limits[:, 1])
#         idcsValid = ~idcsMask.sum(axis=1).astype(bool)[:, None]
#
#     else:
#         idcsValid = np.ones(wave_arr.size).astype(bool)[:, None]
#
#     # Find indeces for six points in spectrum
#     idcsW = np.searchsorted(wave_arr, masks_array)
#
#     # Emission region
#     idcsLineRegion = ((wave_arr[idcsW[:, 2]] <= wave_arr[:, None]) & (wave_arr[:, None] <= wave_arr[idcsW[:, 3]]) & idcsValid).squeeze()
#
#     # Return left and right continua merged in one array
#     if merge_continua:
#         idcsContRegion = (((wave_arr[idcsW[:, 0]] <= wave_arr[:, None]) &
#                           (wave_arr[:, None] <= wave_arr[idcsW[:, 1]])) |
#                           ((wave_arr[idcsW[:, 4]] <= wave_arr[:, None]) & (
#                            wave_arr[:, None] <= wave_arr[idcsW[:, 5]])) & idcsValid).squeeze()
#
#         outputs = idcsLineRegion, idcsContRegion
#
#     # Return left and right continua in separated arrays
#     else:
#         idcsContLeft = ((wave_arr[idcsW[:, 0]] <= wave_arr[:, None]) & (wave_arr[:, None] <= wave_arr[idcsW[:, 1]]) & idcsValid).squeeze()
#         idcsContRight = ((wave_arr[idcsW[:, 4]] <= wave_arr[:, None]) & (wave_arr[:, None] <= wave_arr[idcsW[:, 5]]) & idcsValid).squeeze()
#
#         outputs = idcsLineRegion, idcsContLeft, idcsContRight
#
#     return outputs


def join_fits_files(log_file_list, output_address, delete_after_join=False, levels=['id', 'line']):

    """
    This functions combines multiple log files into single *.fits* file. The user can request to the delete the individual
    logs after the individual logs have been combined.

    If the case of individual *.fits* the function loop through the individual HDU and add them to the output file. Currently,
    this is not available to other multi-page files (such as .xlsx or .asdf)

    :param log_file_list: Input list of log files.
    :type log_file_list: list

    :param output_address: String or path for the output combined log file.
    :type output_address: str, Path, optional

    :param delete_after_join: Delete individual files after joining them. The default value is False
    :type output_address: bool, optional

    :param levels: Indexes name list for MultiIndex dataframes. The default value is ['id', 'line'].
    :type levels: list, optional

    :return:

    """

    # Confirm is a path
    output_address = Path(output_address)

    # Create new HDU for the combined file with a new PrimaryHDU
    hdulist = fits.HDUList([fits.PrimaryHDU()])

    # Progress bar
    n_log = len(log_file_list)
    pbar = ProgressBar('bar', f'log files combined')

    # Iterate through the file paths, open each FITS file, and append the non-primary HDUs to hdulist
    missing_files = []
    for i, log_path in enumerate(log_file_list):

        log_path = Path(log_path)
        pbar.output_message(i, n_log, pre_text="", post_text=None)

        if log_path.is_file():

            ext = log_path.suffix

            # Fits file
            if ext == '.fits':
                with fits.open(log_path) as hdul:
                    for j, hdu in enumerate(hdul):
                        if j == 0:
                            if not isinstance(hdu, fits.PrimaryHDU):
                                hdu_i = fits.BinTableHDU(data=hdu.data, header=hdu.header, name=hdu.name,
                                                         character_as_bytes=False)
                            else:
                                hdu_i = None

                        else:
                            hdu_i = fits.BinTableHDU(data=hdu.data, header=hdu.header, name=hdu.name,
                                                      character_as_bytes=False)
                        # Append
                        if hdu_i is not None:
                            hdulist.append(hdu_i)

            # Remaining types
            else:
                df_i = load_frame(log_path, levels=levels)
                name_i = log_path.stem
                hdu_i = log_to_HDU(df_i, ext_name=name_i)

                # Append
                if hdu_i is not None:
                    hdulist.append(hdu_i)

            # Append to the list
            # with fits.open(log_path) as hdulist_i:
            #
            #     for i, hdu in enumerate(hdulist_i):
            #         if i == 0:
            #             if not isinstance(hdu, fits.PrimaryHDU):
            #                 hdulist.append(fits.TableHDU(data=hdu.data, header=hdu.header, name=hdu.name))
            #         else:
            #             hdulist.append(fits.TableHDU(data=hdu.data, header=hdu.header, name=hdu.name))

        else:
            missing_files.append(log_path)

    # Save to a combined file
    hdulist.writeto(output_address, overwrite=True, output_verify='ignore')
    hdulist.close()

    # Warn of missing files
    if len(missing_files) > 0:
        _logger.info(f"Warning these files were missing: {missing_files}")

    # Delete individual files if requested
    if delete_after_join:
        if len(missing_files) == 0:
            for log_path in log_file_list:
                log_path.unlink()
        else:
            _logger.info("The individual masks won't be deleted")

    return


def check_units(units_wave, units_flux):

    # Check if input are already astropy units
    units_wave = au.Unit(units_wave) if not isinstance(units_wave, (IrreducibleUnit, CompositeUnit, Unit)) else units_wave
    units_flux = au.Unit(units_flux) if not isinstance(units_flux, (IrreducibleUnit, CompositeUnit, Unit)) else units_flux

    return units_wave, units_flux


def save_parameter_maps(lines_log_file, output_folder, param_list, line_list, mask_file=None, mask_list='all',
                        image_shape=None, log_ext_suffix='_LINELOG', spaxel_fill_value=np.nan, output_file_prefix=None,
                        header=None, wcs=None):

    """

    This function converts a line measurements log from an IFS cube, into a set of 2D parameter maps.

    The parameter ".fits" files are saved into the ``output_folder``. These files are named after the parameters in the
    ``param_list`` with the optional prefix from the ``output_file_prefix`` argument. These files will have one page per
    line in the ``line_list`` argument.

    The user can provide a spatial mask file address from which to recover the spaxels with line measurements. If the
    mask ``.fits`` file contains several pages, the user can provide a ``mask_list`` with the ones to explore. Otherwise,
    all mask pages will be used.

    .. attention::
        The user can provide an ``image_shape`` tuple to generate the output parameter map. However, for a large image
        size this approach may require a long time to query the log file pages.

    The expected page name in the input ``lines_log_file`` is "idx_j-idx_i_log_ext_suffix" where "idx_j" and "idx_i"
    are the y and x array coordinates of the cube coordinates, by default ``log_ext_suffix='_LINELOG'``.

    The output ``.fits`` parameter page header includes the ``PARAM`` and ``LINE`` entries with the line and parameter
    labels respectively. The user should also include a ``wcs`` argument to export the astronomical coordinates to the
    output files. The user can add additional information via the ``header`` argument.

    :param lines_log_file: Fits file with IFU cube line measurements.
    :param lines_log_file: str, pathlib.Path

    :param param_list: List of parameters to map
    :param param_list: list

    :param line_list: List of lines to map
    :param line_list: list

    :param output_folder: Output folder to save the maps
    :param output_folder: str, pathlib.Path

    :param mask_file: Address of binary spatial mask file
    :type mask_file: str, pathlib.Path

    :param mask_list: Mask name list to explore on the ``mask_file``.
    :type mask_list: list, optional

    :param image_shape: Array with the image spatial size.
    :param image_shape: list, tuple, optional

    :param spaxel_fill_value: Map filling value for empty pixels. The default value is "numpy.nan".
    :param spaxel_fill_value: float, optional

    :param log_ext_suffix: Suffix for the lines log extension. The default value is "_LINELOG"
    :param log_ext_suffix: str, optional

    :param output_file_prefix: Prefix for the output parameter ".fits" file. The default value is None.
    :param output_file_prefix: str, optional

    :param header: Dictionary for parameter ".fits" file header
    :type header: dict, optional

    :param wcs: Observation `world coordinate system <https://docs.astropy.org/en/stable/wcs/index.html>`_.
    :type wcs: astropy WCS, optional

    """

    assert Path(lines_log_file).is_file(), f'- ERROR: lines log at {lines_log_file} not found'
    assert Path(output_folder).is_dir(), f'- ERROR: Output parameter maps folder {output_folder} not found'

    # Compile the list of voxels to recover the provided masks
    if mask_file is not None:

        assert Path(mask_file).is_file(), f'- ERROR: mask file at {mask_file} not found'

        with fits.open(mask_file) as maskHDUs:

            # Get the list of mask extensions
            if mask_list == 'all':
                if ('PRIMARY' in maskHDUs) and (len(maskHDUs) > 1):
                    mask_list = []
                    for i, HDU in enumerate(maskHDUs):
                        mask_name = HDU.name
                        if mask_name != 'PRIMARY':
                            mask_list.append(mask_name)
                    mask_list = np.array(mask_list)
                else:
                    mask_list = np.array(['PRIMARY'])
            else:
                mask_list = np.array(mask_list, ndmin=1)

            # Combine all the mask voxels into one
            for i, mask_name in enumerate(mask_list):
                if i == 0:
                    mask_array = maskHDUs[mask_name].data
                    image_shape = mask_array.shape
                else:
                    assert image_shape == maskHDUs[mask_name].data.shape, '- ERROR: Input masks do not have the same dimensions'
                    mask_array += maskHDUs[mask_name].data

            # Convert to boolean
            mask_array = mask_array.astype(bool)

            # List of spaxels in list [(idx_j, idx_i), ...] format
            spaxel_list = np.argwhere(mask_array)

    # No mask file is provided and the user just defines an image size tupple (nY, nX)
    else:
        mask_array = np.ones(image_shape).astype(bool)
        spaxel_list = np.argwhere(mask_array)

    # Generate containers for the data:
    images_dict = {}
    for param in param_list:

        # Make sure is an array and loop throuh them
        for line in line_list:
            images_dict[f'{param}-{line}'] = np.full(image_shape, spaxel_fill_value)

    # Loop through the spaxels and fill the parameter images
    n_spaxels = spaxel_list.shape[0]
    spaxel_range = np.arange(n_spaxels)
    pbar = ProgressBar('bar', f'spaxels from file')

    with fits.open(lines_log_file) as logHDUs:

        for i_spaxel in spaxel_range:
            idx_j, idx_i = spaxel_list[i_spaxel]
            spaxel_ref = f'{idx_j}-{idx_i}{log_ext_suffix}'

            post_text = f'of spaxels from file ({lines_log_file}) read ({n_spaxels} total spaxels)'
            pbar.output_message(i_spaxel, n_spaxels, pre_text="", post_text=post_text)

            # progress_bar(i_spaxel, n_spaxels, post_text=f'of spaxels from file ({lines_log_file}) read ({n_spaxels} total spaxels)')

            # Confirm log extension exists
            if spaxel_ref in logHDUs:

                # Recover extension data
                log_data = logHDUs[spaxel_ref].data
                log_lines = log_data['index']

                # Loop through the parameters and the lines:
                for param in param_list:
                    idcs_log = np.argwhere(np.in1d(log_lines, line_list))
                    for i_line in idcs_log:
                        images_dict[f'{param}-{log_lines[i_line][0]}'][idx_j, idx_i] = log_data[param][i_line][0]

    # New line after the rustic progress bar
    print()

    # Recover coordinates from the wcs to store in the headers:
    hdr_coords = extract_wcs_header(wcs, drop_axis='spectral')

    # Save the parameter maps as individual fits files with one line per page
    output_file_prefix = '' if output_file_prefix is None else output_file_prefix
    for param in param_list:

        # Primary header
        paramHDUs = fits.HDUList()
        paramHDUs.append(fits.PrimaryHDU())

        # ImageHDU for the parameter maps
        for line in line_list:

            # Create page header with the default data
            hdr_i = fits.Header()
            hdr_i['LINE'] = (line, 'Line label')
            hdr_i['PARAM'] = (param, 'LiMe parameter label')

            # Add WCS information
            if hdr_coords is not None:
                hdr_i.update(hdr_coords)

            # Add user information
            if header is not None:
                page_hdr = header.get(f'{param}-{line}', None)
                page_hdr = header if page_hdr is None else page_hdr
                hdr_i.update(page_hdr)

            # Create page HDU entry
            HDU_i = fits.ImageHDU(name=line, data=images_dict[f'{param}-{line}'], header=hdr_i, ver=1)
            paramHDUs.append(HDU_i)

        # Write to new file
        output_file = Path(output_folder)/f'{output_file_prefix}{param}.fits'
        paramHDUs.writeto(output_file, overwrite=True, output_verify='fix')

    return


def extract_wcs_header(wcs, drop_axis=None):

    if wcs is not None:

        # Remove 3rd dimensional axis if present
        if drop_axis is not None:
            if drop_axis == 'spectral':
                input_wcs = wcs.dropaxis(2) if wcs.naxis == 3 else wcs
            elif drop_axis == 'spatial':
                if wcs.naxis == 3:
                    input_wcs = wcs.dropaxis(1)
                    input_wcs = wcs.dropaxis(0)
            else:
                raise LiMe_Error(f'Fits coordinates axis: "{drop_axis}" not recognized. Please use: "spectral" or'
                                 f' "spatial"')

        else:
            input_wcs = wcs

        # Convert to HDU header
        hdr_coords = input_wcs.to_fits()[0].header

    else:
        hdr_coords = None

    return hdr_coords


class ProgressBar:

    def __init__(self, message_type=None, count_type='bar'):

        self.output_message = None
        self.count_type = count_type

        if message_type is None:
            self.output_message = self.no_output

        if message_type == 'bar':
            self.output_message = self.progress_bar

        return

    def progress_bar(self, i, i_max, pre_text, post_text, n_bar=10):

        post_text = "" if post_text is None else post_text

        j = (i + 1) / i_max
        stdout.write('\r')
        message = f'[{"=" * int(n_bar * j):{n_bar}s}] {int(100 * j)}% of {self.count_type} {post_text}'
        stdout.write(message)
        stdout.flush()

        return

    def no_output(self, i, i_max, pre_text, post_text, n_bar=10):

        return
