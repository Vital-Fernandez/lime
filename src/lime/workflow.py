import logging

import numpy as np
import pandas as pd
from pathlib import Path
from astropy.io import fits
from time import time
from lmfit.models import PolynomialModel

from lime.fitting.lines import LineFitting, signal_to_noise_rola, sigma_corrections, k_gFWHM, velocity_to_wavelength_band, profiles_computation, linear_continuum_computation
from lime.tools import ProgressBar, join_fits_files, extract_wcs_header, pd_get, unit_conversion
from lime.transitions import Line, air_to_vacuum_function, label_decomposition
from lime.io import check_file_dataframe, check_file_array_mask, log_to_HDU, results_to_log, load_frame, LiMe_Error, check_fit_conf, _PARENT_BANDS
from lime.fitting.redshift import RedshiftFitting
from lime import __version__
from itertools import combinations

try:
    import aspect
    aspect_check = True
except ImportError:
    aspect_check = False


_logger = logging.getLogger('LiMe')


def review_bands(spec, line, min_line_pixels=3, min_cont_pixels=2, user_cont_from_bands=True, user_err_from_bands=False):

    # Check if the line bands are provided
    if line.mask is None:
        _logger.warning(f"Line {line} was not found on the input bands database. It won't be measured")
        return None


    # Check if the line is within the w3, w4 limits
    limit_blue, limit_red = spec.wave.compressed()[0], spec.wave.compressed()[-1]
    if ((line.mask[2] * (1 + spec.redshift)) < limit_blue) or ((line.mask[3] * (1 + spec.redshift)) > limit_red):
        _logger.warning(f"Line {line} bands are outside spectrum wavelengh range: w3 < w_min_rest ({line.mask[2]} < {limit_blue}) or"
                                                                               f" w4 > w_max_rest ({line.mask[3]} > {limit_red})"
                                                                               f" it won't be measured")

        return None

    # Check if the spectrum does not have the error arr but the user has requested it
    if spec.err_flux is None and user_err_from_bands is False:
        _logger.warning(f'The observation does not have an error spectrum but the fit command has requested not to use '
                        f'the adjacent bands to compute the uncertainty. Please set the "user_err_from_bands=True" to perform'
                        f' a measurement.')

        return None

    # Compute the line and adjacent continua indeces:
    idcsEmis, idcsCont = line.index_bands(spec.wave, spec.redshift)

    # Check if all the flux entries are masked
    emis_flux, cont_flux = spec.flux[idcsEmis], spec.flux[idcsCont]
    if np.all(emis_flux.mask):
        _logger.warning(f"Line {line} flux is fully masked. It won't be measured")
        return None
    if np.all(cont_flux.mask) and user_cont_from_bands:
        _logger.warning(f"Line {line} adjacent continua flux is fully masked. It won't be measured")
        return None

    # Check if all the flux entries are zero
    if not np.any(emis_flux):
        _logger.warning(f"Line {line} flux entries are all 0. It won't be measured")
        return None
    if not np.any(cont_flux) and user_cont_from_bands:
        _logger.warning(f"Line {line} continuum flux entries are all 0. It won't be measured")
        return None

    # Check if the line selection is too narrow
    if np.sum(~emis_flux.mask) < min_line_pixels:
        _logger.warning(f"Line {line} has only {np.sum(~emis_flux.mask)} pixels. It won't be measured")
        return None

    # Check if the continua selection is too narrow
    if (np.sum(~cont_flux.mask) < min_cont_pixels) and user_cont_from_bands:
        _logger.warning(f"Line {line} continuum bands have only {np.sum(~cont_flux.mask)} pixels. It won't be measured")
        return None

    return idcsEmis, idcsCont


def import_line_kinematics(line, z_cor, log, fit_conf):

    # Check if imported kinematics come from blended component
    for idx_child, child_label in enumerate(line.list_comps):

        # Check for kinem order
        parent_label = fit_conf.get(f'{child_label}_kinem')
        if (parent_label is not None) and line.blended_check:

            # Tied kinematics in blended profile
            if parent_label in line.list_comps:
                idx_parent = line.list_comps.index(parent_label)
                factor = f'{line.wavelength[idx_child] / line.wavelength[idx_parent]:0.8f}'
                fit_conf[f'{child_label}_center'] = {'expr': f'{factor}*{parent_label}_center'}
                fit_conf[f'{child_label}_sigma'] = {'expr': f'{factor}*{parent_label}_sigma'}

            # Import kinematics from previously measured
            elif parent_label in log.index:
                mu_parent = log.loc[parent_label, ['center', 'center_err']].to_numpy()
                sigma_parent = log.loc[parent_label, ['sigma', 'sigma_err']].to_numpy()
                wave_ratio = line.wavelength[idx_child]/log.loc[parent_label, 'wavelength']

                center_child_arr = wave_ratio * (mu_parent / z_cor)
                sigma_child_arr = wave_ratio * sigma_parent

                # Store the value on the dictionary
                fit_conf[f'{child_label}_center'] = {'value': center_child_arr[0], 'vary': False}
                fit_conf[f'{child_label}_sigma'] = {'value': sigma_child_arr[0], 'vary': False}

                # Error for the propagation
                fit_conf[f'{child_label}_center_err'] = center_child_arr[1]
                fit_conf[f'{child_label}_sigma_err'] = sigma_child_arr[1]

            # Line has not been measured before found
            else:
                _logger.info(f'\n{parent_label} has not been found on the input lines frame for {child_label} kinematics export.'
                             f'\n - Please check you are using the right line label and that the line has been measured prior '
                             f'prior to the current fitting.')

    return


def check_cube_bands(input_bands, mask_list, fit_cfg):

    if input_bands is None:

        # Recover the mask_configuration as a list
        for mask_name in mask_list:

            mask_fit_cfg = fit_cfg.get(f'{mask_name}_line_fitting')

            missing_mask = False
            if mask_fit_cfg is not None:
                if mask_fit_cfg.get('bands') is None:
                    missing_mask = True
            else:
                missing_mask = True

            if missing_mask:
                error_message = 'No input "bands" provided. In this case you need to include the \n' \
                                f'you need to specify an "bands=log_file_address" entry the ' \
                                f'"[{mask_name}_file]" of your fitting configuration file'
                raise LiMe_Error(error_message)

    return


def recover_level_conf(fit_cfg, mask_key, default_key):

    default_cfg = fit_cfg.get(f'{default_key}_line_fitting') if default_key is not None else None
    mask_cfg = fit_cfg.get(f'{mask_key}_line_fitting') if mask_key is not None else None

    # Case there are not leveled entries
    if (default_cfg is None) and (mask_cfg is None):
        output_conf = fit_cfg

    # Proceed to update the levels
    else:

        # Default configuration
        default_conf = {} if default_cfg is None else default_cfg
        default_detect = default_conf.get('line_detection')

        # Mask conf
        mask_conf = {} if mask_cfg is None else mask_cfg
        mask_detect = mask_conf.get('line_detection')

        # Update the levels
        output_conf = {**default_conf, **mask_conf}

        # If no line detection don't add it
        if mask_detect is not None:
            output_conf['line_detection'] = mask_detect
        elif default_detect is not None:
            output_conf['line_detection'] = default_detect
        else:
            pass

    return output_conf


def check_compound_line_exclusion(line, lines_df):

    # Confirm the dataframe includes the group of lines
    group_label = pd_get(lines_df, line, 'group_label', transform='none')

    # Confirm if the line is in the group of lines
    if group_label is not None:
        comp_list = group_label.split('+')
        measure_check = False if line in comp_list else True
    else:
        measure_check = True

    return measure_check


def continuum_model_fit(x_array, y_array, idcs, degree):

    poly3Mod = PolynomialModel(prefix=f'poly_{degree}', degree=degree)
    poly3Params = poly3Mod.guess(y_array[idcs], x=x_array[idcs])

    try:
        poly3Out = poly3Mod.fit(y_array[idcs], poly3Params, x=x_array[idcs])
        cont_fit = poly3Out.eval(x=x_array)

    except TypeError:
        _logger.warning(f'- The continuum fitting polynomial has more degrees ({degree}) than data points')
        cont_fit = np.full(x_array.size, np.nan)

    return cont_fit


def line_bands(wave_intvl=None, line_list=None, particle_list=None, redshift=None, units_wave='Angstrom', decimals=None,
               vacuum_waves=False, ref_bands=None, update_labels=False, update_latex=False, vacuum_label=False):
    """

    This function returns `LiMe bands database <https://lime-stable.readthedocs.io/en/latest/inputs/n_inputs3_line_bands.html>`_
    as a pandas dataframe.

    If the user provides a wavelength array (``wave_inter``) the output dataframe will be limited to the lines within
    this wavelength interval.

    Similarly, the user provides a ``lines_list`` or a ``particle_list`` the output bands will be limited to the these
    lists. These inputs must follow `LiMe notation style <https://lime-stable.readthedocs.io/en/latest/inputs/n_inputs2_line_labels.html>`_

    If the user provides a redshift value alongside the wavelength interval (``wave_intvl``) the output bands will be
    limited to the transitions at that observed range.

    The user can specify the desired wavelength units using the `astropy string format <https://docs.astropy.org/en/stable/units/ref_api.html>`_
    or introducing the `astropy unit variable  <https://docs.astropy.org/en/stable/units/index.html>`_. The default value
    unit is angstroms.

    The argument ``sig_digits`` determines the number of decimal figures for the line labels.

    The user can request the output line labels and bands wavelengths in vacuum setting ``vacuum=True``. This conversion
    is done using the relation from `Greisen et al. (2006) <https://www.aanda.org/articles/aa/abs/2006/05/aa3818-05/aa3818-05.html>`_.

    Instead of the default LiMe database, the user can provide a ``ref_bands`` dataframe (or the dataframe file address)
    to use as the reference database.

    :param wave_intvl: Wavelength interval for output line transitions.
    :type wave_intvl: list, numpy.array, lime.Spectrum, lime.Cube, optional

    :param line_list: Line list for output line bands.
    :type line_list: list, numpy.array, optional

    :param particle_list: Particle list for output line bands.
    :type particle_list: list, numpy.array, optional

    :param redshift: Redshift interval for output line bands.
    :type redshift: list, numpy.array, optional

    :param units_wave: Labels and bands wavelength units. The default value is "A".
    :type units_wave: str, optional

    :param decimals: Number of decimal figures for the line labels.
    :type decimals: int, optional

    :param vacuum_waves: Set to True for vacuum wavelength values. The default value is False.
    :type vacuum_waves: bool, optional

    :param ref_bands: Reference bands dataframe. The default value is None.
    :type ref_bands: pandas.Dataframe, str, pathlib.Path, optional

    :return:
    """

    # Use the default lime mask if none provided
    if ref_bands is None:
        ref_bands = _PARENT_BANDS

    # Load the reference bands
    bands_df = check_file_dataframe(ref_bands)

    # Convert to requested units
    if units_wave != 'Angstrom':
        wave_columns = ['wave_vac', 'wavelength', 'w1', 'w2', 'w3', 'w4', 'w5', 'w6']
        conversion_factor = unit_conversion(in_units='Angstrom', out_units=units_wave, wave_array=1)
        bands_df.loc[:, wave_columns] = bands_df.loc[:, wave_columns] * conversion_factor
        bands_df['units_wave'] = units_wave
    else:
        conversion_factor = 1

    # First slice by wavelength and redshift
    idcs_rows = np.ones(bands_df.index.size).astype(bool)
    if wave_intvl is not None:
        w_min, w_max = wave_intvl[0], wave_intvl[-1]

        # Account for redshift
        redshift = redshift if redshift is not None else 0
        if 'wavelength' in bands_df.columns:
            wave_arr = bands_df['wavelength'] * (1 + redshift)
        else:
            wave_arr = label_decomposition(bands_df.index.to_numpy(), params_list=['wavelength'])[0] * conversion_factor

        # Compare with wavelength values
        idcs_rows = idcs_rows & (wave_arr >= w_min) & (wave_arr <= w_max)

    # Second slice by particle
    if particle_list is not None:
        idcs_rows = idcs_rows & bands_df.particle.isin(particle_list)

    # Finally slice by the name of the lines
    if line_list is not None:
        idcs_rows = idcs_rows & bands_df.index.isin(line_list)

    # Final table
    bands_df = bands_df.loc[idcs_rows]

    # Update the labels to reflect new wavelengths and units if necessary and user requests it
    if update_labels or vacuum_label:
        list_labels = [None] * bands_df.index.size
        for i, label in enumerate(bands_df.index):
            line = Line(label, band=bands_df)
            line.update_label(decimals=decimals, bands_df=bands_df, update_latex=update_latex, vacuum_label=vacuum_label)
            if update_latex:
                bands_df.loc[label, 'latex_label'] = line.latex_label[0]
            list_labels[i] = line.label
        bands_df.rename(index=dict(zip(bands_df.index, list_labels)), inplace=True)

    # Convert to vacuum wavelengths if requested but after renaming the labels to keep air if requested
    if vacuum_waves:
        bands_df['wavelength'] = bands_df['wave_vac']
        bands_lim_columns = ['w1', 'w2', 'w3', 'w4', 'w5', 'w6']
        bands_df[bands_lim_columns] = air_to_vacuum_function(bands_df[bands_lim_columns].to_numpy())

    return bands_df

def res_power_approx(wavelength_arr):

    delta_lambda = np.ediff1d(wavelength_arr, to_end=0)
    delta_lambda[-1] = delta_lambda[-2]

    return wavelength_arr/delta_lambda


def get_merged_blended_lines(spec, bands, in_cfg, composite_lines):

    # Get lines on the list
    comps_dict = {} if in_cfg is False else {comp: group_label
                                             for comp, group_label in in_cfg.items()
                                             if comp.endswith(('_b', '_m'))}

    # Limit the selection to the user lines
    if composite_lines is not None:
        comps_dict = {line: comps for line, comps in comps_dict.items() if line in composite_lines}

    # # Determine the grouped lines
    # key_cfg = f'{default_cfg_prefix}_line_fitting'
    # default_dict = {} if key_cfg not in in_cfg else {comp: group_label
    #                                                 for comp, group_label in in_cfg[key_cfg].items()
    #                                                 if comp.endswith(('_b', '_m'))}
    #
    # key_cfg = f'{obj_cfg_prefix}_line_fitting'
    # obj_dict =  {} if key_cfg not in in_cfg else{comp: group_label
    #                                              for comp, group_label in in_cfg[key_cfg].items()
    #                                              if comp.endswith(('_b', '_m'))}
    #
    # if len(default_dict) == 0 and len(obj_dict) == 0:
    #     output_dict = fit_conf.copy()
    # else:
    #     output_dict = {**default_dict, **obj_dict}

    # b_m_arr = np.array(list(output_dict.keys()))
    # core_arr = np.array([s[:-2] for s in b_m_arr])
    # unique_elements, counts = np.unique(core_arr, return_counts=True)
    # repeated_lines = unique_elements[counts > 1]
    #
    # # Check each case individually
    # for line in repeated_lines:
    #     line_blended, line_merged = line + '_b', line + '_m'
    #     default_b, default_m = line_blended in default_dict, line_merged in default_dict
    #     obj_b, obj_m = line_blended in obj_dict, line_merged in obj_dict
    #
    #     # Object configuration has priority
    #     if not (obj_b == obj_m):
    #         output_dict.pop(line_blended if obj_b else line_merged)
    #
    #     # Try default configuration
    #     elif not (default_b == default_m):
    #         output_dict.pop(line_blended if default_b else line_merged)
    #
    #     # Solve the kinematics
    #     else:
    #
    #         # Get components
    #         comps = output_dict[line_blended].split('+')
    #
    #         # Check if more than one component is on the bands
    #         df_comps = bands.loc[bands.index.isin(comps)]
    #         if df_comps.index.size > 1:
    #
    #             # Set minimum number of pixels for the model solution in the bands
    #             central_band = bands.loc[bands.index.isin(comps), 'w3':'w4'].to_numpy() * (1 + spec.redshift)
    #             idcs_central = np.searchsorted(spec.wave, (central_band[:, 0].min(), central_band[:, 1].max()))
    #             blended_check = idcs_central[1] - idcs_central[0] > (3 * len(comps))
    #
    #             # Print warning
    #             _logger.info(f'The input configuration has merged and blended entries for {line}. '
    #                          f'Automatic assignment as {"blended" if blended_check else "Merged"} ({line_blended if blended_check else line_blended}'
    #                          f'={output_dict[line_blended if blended_check else line_blended]})')
    #             output_dict.pop(line_blended if blended_check else line_blended)

    # # Get candidate lines for grouping
    # line_arr = [item for v in output_dict.values() for item in v.split('+')]
    # line_arr = bands.loc[bands.index.isin(line_arr)].index
    # lambda_arr = bands.loc[line_arr, 'wavelength'].to_numpy()
    #
    # # Generate the line - wavelength matrix with the line pixels width
    # wave_matrix = np.zeros((lambda_arr.size, spec.wave_rest.data.size))
    # w3_arr = np.searchsorted(spec.wave_rest.data, bands.loc[line_arr, 'w3'].to_numpy())
    # w4_arr = np.searchsorted(spec.wave_rest.data, bands.loc[line_arr, 'w4'].to_numpy())
    # cols = np.arange(wave_matrix.shape[1])
    # wave_matrix[(cols >= w3_arr[:, None]) & (cols <= w4_arr[:, None])] = 1
    # pixels_width = wave_matrix.sum(axis=1)
    #
    # # Get the decision matrix with the matching intervals
    # decision_matrix = wave_matrix @ wave_matrix.T
    # blended_matrix = decision_matrix < np.ceil(pixels_width/3)[:, None]
    # math_dict = dict(zip(line_arr, np.arange(line_arr.size)))
    #
    # results_dict = {}
    # for label, group_label in output_dict.items():
    #     list_group = group_label.split('+')
    #     if all(key in math_dict for key in list_group):
    #         blended_group = True if label[-1] == 'b' else False
    #         relations = np.array([blended_matrix[math_dict[i], math_dict[j]] for i, j in list(combinations(list_group, 2))])
    #         if np.all(relations == blended_group):
    #             print(f'Passess: {label}={group_label}')
    #             results_dict[label] = group_label

    return comps_dict





# upper_triangle < (np.ceil(pixels_width/3)[:, None]=
#
#
#     np.fill_diagonal(decision_matrix, 0)  # optional
#
#     # Step 2: Extract the upper triangle (excluding diagonal)
#     upper_triangle = np.ma.masked_array(np.triu(decision_matrix, k=1), mask=np.tril(np.ones_like(decision_matrix), k=0).astype(bool)).astype(int)
#
#     # Step 3: Find (i, j) pairs with >1 shared steps
#     rows_i, rows_j = np.where(upper_triangle > 1)
#     edges = list(zip(line_arr[rows_i], line_arr[rows_j]))
#
#     matrix < thresholds[:, None]
#
#     # Step 3: Check if all values in each column satisfy the condition
#     idcs_type = upper_triangle < np.ceil(pixels_width/3)[:, None]
#     merged_idcs = np.all(idcs_type, axis=0)
#     blended_idcs = np.all(~idcs_type, axis=0)
#     #
    # math_dict = dict(zip(line_arr, np.arange(line_arr.size)))
    # for row in rows_j:
    #     print(line_arr[row])
    #
    # for label, group_label in output_dict.keys():
    #     list_group = group_label.split('+')
    #     if len(list_group) == 2:
    #         x_label, y_label = math_dict.get(list_group[0]), math_dict.get(list_group[1])
    #         if (x_label is not None) and (y_label is not None):
    #


    # print(w3_arr)
    # print(w4_arr)
    #
    # wave_matrix[w3_arr:w4_arr, :]
    # # box_pixels =
    # #
    # # print(idcs_lines)
    # lambda_arr = bands.index.isin(line_arr)
    # wave_arr = spec.wave.data

    # Generate matrices array

    # b_m_arr = np.array(list(output_dict.keys()))
    # core_arr = np.array([s[:-2] for s in b_m_arr])
    # unique_elements, counts = np.unique(core_arr, return_counts=True)
    # repeated_lines = unique_elements[counts > 1]
    #
    # # Check each case individually
    # for line in repeated_lines:
    #     line_blended, line_merged = line + '_b', line + '_m'
    #     default_b, default_m = line_blended in default_dict, line_merged in default_dict
    #     obj_b, obj_m = line_blended in obj_dict, line_merged in obj_dict
    #
    #     # Object configuration has priority
    #     if not (obj_b == obj_m):
    #         output_dict.pop(line_blended if obj_b else line_merged)
    #
    #     # Try default configuration
    #     elif not (default_b == default_m):
    #         output_dict.pop(line_blended if default_b else line_merged)
    #
    #     # Solve the kinematics
    #     else:
    #
    #         # Get components
    #         comps = output_dict[line_blended].split('+')
    #
    #         # Check if more than one component is on the bands
    #         df_comps = bands.loc[bands.index.isin(comps)]
    #         if df_comps.index.size > 1:
    #
    #             # Set minimum number of pixels for the model solution in the bands
    #             central_band = bands.loc[bands.index.isin(comps), 'w3':'w4'].to_numpy() * (1 + spec.redshift)
    #             idcs_central = np.searchsorted(spec.wave, (central_band[:, 0].min(), central_band[:, 1].max()))
    #             blended_check = idcs_central[1] - idcs_central[0] > (3 * len(comps))
    #
    #             # Print warning
    #             _logger.info(f'The input configuration has merged and blended entries for {line}. '
    #                          f'Automatic assignment as {"blended" if blended_check else "Merged"} ({line_blended if blended_check else line_blended}'
    #                          f'={output_dict[line_blended if blended_check else line_blended]})')
    #             output_dict.pop(line_blended if blended_check else line_blended)
    #
    #
    # return output_dict

def pars_bands_conf(spec, bands, ref_bands, fit_conf, composite_lines):

    # Get the the grouped lines
    comps_dict = {} if fit_conf is False else {comp: group_label
                                               for comp, group_label in fit_conf.items()
                                               if comp.endswith(('_b', '_m'))}

    # Limit the selection to the user lines
    if composite_lines is not None:
        comps_dict = {line: comps for line, comps in comps_dict.items() if line in composite_lines}

    # Loop through the lines on the list and review which changes are necessary
    rename_dict, exclude_list= {}, []
    group_dict, w3_dict, w4_dict = {}, {}, {}
    for new_label, group_label in comps_dict.items():

        component_list = np.unique([Line(x).core for x in group_label.split('+') if '_k-' not in x])
        old_label = component_list[0]

        # Only apply corrections if components are present
        idcs_comps = bands.index.isin(component_list)
        if np.sum(idcs_comps) == component_list.size:

            # Save the modifications
            rename_dict[old_label] = new_label
            exclude_list += list(component_list)
            w3_dict[new_label] = bands.loc[idcs_comps, 'w3'].min()
            w4_dict[new_label] = bands.loc[idcs_comps, 'w4'].max()
            group_dict[new_label] = group_label

        # Check the line or the same group is already there
        else:
            low, high = spec.wave_rest.compressed()[[0, -1]]
            check_arr = np.array([(low < Line(label).wavelength < high)[0] for label in component_list])

            # Lines outside wavelength range
            if np.all(~check_arr):
                continue

            # Check if lines outside range
            else:
                if 'group_label' in bands.columns:
                    if not np.any(comps_dict[new_label] == bands.group_label):
                        _logger.info(f'Line component "{old_label}" for configuration entry: '
                                     f'"{new_label}={comps_dict[new_label]}" not found in lines table')
                else:
                    _logger.info(f'Missing line(s) "{np.setxor1d(bands.loc[idcs_comps].index.to_numpy(), component_list)}" '
                                 f'for configuration entry: '
                                 f'"{new_label}={comps_dict[new_label]}" in reference lines table')

    # Warn in case some of the bands dont match the database:
    if not set(exclude_list).issubset(bands.index):
        _logger.info(f' The following blended or merged lines were not found on the input lines database:\n'
                     f' - {list(set(exclude_list) - set(bands.index))}\n'
                     f' - It is recommended that the merged/blended components follow the reference transitions labels.\n')

    # Change the latex labels
    for old_label, new_label in rename_dict.items():
        line = Line(new_label, band=bands, fit_conf=comps_dict, update_latex=True)
        bands.loc[old_label, 'latex_label'] = line.latex_label[0] if line.merged_check else '+'.join(line.latex_label)

    # Change the indexes
    bands.rename(index=dict(rename_dict), inplace=True)

    # Remove components columns
    bands.drop(exclude_list, errors='ignore', inplace=True)

    # Add the group_label values
    if 'group_label' not in bands.columns:
        bands['group_label'] = 'none'
    bands['group_label'] = pd.Series(bands.index.map(group_dict), index=bands.index).fillna(bands['group_label'])

    # Change velocity limits
    bands['w3'] = pd.Series(bands.index.map(w3_dict), index=bands.index).fillna(bands['w3'])
    bands['w4'] = pd.Series(bands.index.map(w4_dict), index=bands.index).fillna(bands['w4'])

    return


class SpecRetriever:

    def __init__(self, spectrum):

        self._spec = spectrum

        return

    def line_bands(self, band_vsigma=70, n_sigma=4, adjust_central_band=True, instrumental_correction=True, components_detection=False,
               composite_lines=None, fit_cfg=None, default_cfg_prefix='default', obj_cfg_prefix=None, update_default=True,
               line_list=None, particle_list=None, decimals=None, vacuum_waves=False, ref_bands=None, update_labels=False,
               update_latex=False, vacuum_label=False):

        # Remove the mask from the wavelength array if necessary
        wave_intvl = self._spec.wave.compressed()

        # Crop the bands to match the observation
        bands = line_bands(wave_intvl, line_list, particle_list, redshift=self._spec.redshift, units_wave=self._spec.units_wave,
                           decimals=decimals, vacuum_waves=vacuum_waves, ref_bands=ref_bands, update_labels=update_labels,
                           update_latex=update_latex, vacuum_label=vacuum_label)

        # Compute the resolving power if necessary
        if self._spec.res_power is not None:
            res_power = self._spec.res_power
        else:
            res_power = res_power_approx(wave_intvl) if (instrumental_correction or fit_cfg is not None) else None

        # Adjust the middle bands to match the line width
        if adjust_central_band:

            # Expected transitions in the observed frame
            lambda_obs = bands.wavelength.to_numpy() * (1 + self._spec.redshift)

            # Add correction for the instrumental broadening
            if instrumental_correction:

                # Indexes for the lines emission peak
                idcs = np.searchsorted(wave_intvl, lambda_obs)

                # Use the instrumental resolution if available
                delta_lambda_inst = lambda_obs / (res_power[idcs] * k_gFWHM)

            # Constant velocity width
            else:
                delta_lambda_inst = 0

            # Convert to spectral width
            delta_lambda = velocity_to_wavelength_band(n_sigma, band_vsigma, lambda_obs, delta_lambda_inst)

            # Add new values to database in the rest frame
            bands['w3'] = (lambda_obs - delta_lambda) / (1 + self._spec.redshift)
            bands['w4'] = (lambda_obs + delta_lambda) / (1 + self._spec.redshift)

        # Combine the blended/merged lines
        if fit_cfg is not None:
            in_cfg = check_fit_conf(fit_cfg, default_cfg_prefix, obj_cfg_prefix, update_default)
            pars_bands_conf(self._spec, bands, ref_bands, in_cfg, composite_lines)

        # Filter the table to match the line detections
        if components_detection:
            if self._spec.infer.pred_arr is not None:

                # Create masks for all intervals
                starts = bands.w3.to_numpy()[:, None] * (1 + self._spec.redshift)
                ends = bands.w4.to_numpy()[:, None] * (1 + self._spec.redshift)

                # Check if x values fall within each interval
                in_intervals = (self._spec.wave.data >= starts) & (self._spec.wave.data < ends)

                # Check where y equals the target category
                is_target_category = np.isin(self._spec.infer.pred_arr, (3, 7, 9))

                # Combine the masks to count target_category occurrences in each interval
                counts = np.sum(in_intervals & is_target_category, axis=1)

                # Check which intervals satisfy the minimum count condition
                idcs = counts >= 3
                bands = bands.loc[idcs]

            else:
                _logger.warning(f'The observations has an empty preduction array. Please run the components detection')

        return bands

    def spectrum(self, line_label=None, output_address=None, ref_frame=None, split_components=False, **kwargs):

        # Headers for the default list
        headers = np.array(["wave", "flux", "err_flux", "pixel_mask"])

        # Use the observation frame if none is provided
        frame = self._spec.frame if ref_frame is None else ref_frame

        # By default report complete spectrum
        idcs = (0, -1)

        # If a line is provided get indexes for the bands limits
        line_measured = False
        if line_label is not None:
            if frame is not None:
                if line_label in frame.index:
                    bands_limits = frame.loc[line_label, 'w1':'w6']
                    idcs_bands = np.searchsorted(self._spec.wave.data, bands_limits * (1 + self._spec.redshift))
                    idcs = (idcs_bands[0], idcs_bands[5])
                    line_measured = True
                else:
                    _logger.warning(f'Line {line_label} not found on observation frame')
            else:
                _logger.warning(f'No lines measured on object')

        # Compute the bands
        if line_measured:

            # Declare line object and the components and its components from the frame
            line = Line(line_label, frame)
            line_list = line.list_comps

            # Compute the linear components
            gaussian_arr = profiles_computation(line_list, frame, 1 + self._spec.redshift, line._p_shape,
                                 x_array=self._spec.wave.data[idcs[0]: idcs[1]])
            linear_arr = linear_continuum_computation(line_list, frame, 1 + self._spec.redshift, x_array=self._spec.wave.data[idcs[0]: idcs[1]])

            # Determine which component you want to extract:
            if split_components is False:
                gaussian_arr = gaussian_arr.sum(axis=1) + linear_arr[:, 0]
                gaussian_arr = gaussian_arr.reshape(-1, 1)
                line_hdrs = [line_label]
            else:
                gaussian_arr = gaussian_arr + linear_arr[:, 0][:, np.newaxis]
                line_hdrs = line_list

            # Add the line list to the headers
            headers = np.append(headers, line_hdrs)

        # Container for the data
        out_arr = np.full((self._spec.wave.data[idcs[0]: idcs[1]].size, len(headers)), np.nan)

        # Fill the array:
        out_arr[:, 0] = self._spec.wave.data[idcs[0]: idcs[1]]
        out_arr[:, 1] = self._spec.flux.data[idcs[0]: idcs[1]] * self._spec.norm_flux

        # Err array if it exists
        if self._spec.err_flux is not None:
            out_arr[:, 2] = self._spec.err_flux[idcs[0]: idcs[1]].data * self._spec.norm_flux

        # Pixel mask if any is invalid
        if np.any(self._spec.wave.mask):
            out_arr[:, 3] = self._spec.wave[idcs[0]: idcs[1]].mask

        # Add the components
        if line_measured:
            for i, line_comp in enumerate(line_hdrs):
                out_arr[:, 4 + i] = gaussian_arr[:, i]

        # Crop array if some columns are missing
        nan_columns = np.zeros(out_arr.shape[1]).astype(bool)
        nan_columns[:4] = np.all(np.isnan(out_arr[:, :4]), axis=0)
        out_arr = out_arr[:, ~nan_columns]

        # Headers
        headers = headers[~nan_columns]

        # Formatting for the data
        spec_hdrs_list = ['%.18e', '%.18e', '%.18e', '%d']
        spec_hdrs_list = spec_hdrs_list + ['%.18e'] * len(line_hdrs) if line_measured else spec_hdrs_list
        array_fmt = np.array(spec_hdrs_list)
        array_fmt = list(array_fmt[~nan_columns])

        # Update defaults with user-provided values
        default_kwargs = {"fmt": array_fmt, "delimiter": ' '}
        default_kwargs.update(kwargs)

        # Create header
        if default_kwargs.get('header') is None:
            default_kwargs['header'] = default_kwargs['delimiter'].join(headers)

        # Dictionary with parameters
        if 'footer' not in default_kwargs:
            footer_dict = {'LiMe': f'v{__version__}',
                            'units_wave': self._spec.units_wave, 'units_flux':  self._spec.units_flux,
                           'redshift': self._spec.redshift, 'norm_flux': self._spec.norm_flux, 'id_label': self._spec.label}
            footer_str = "\n".join(f"{key}:{value}" for key, value in footer_dict.items())
            default_kwargs['footer'] = footer_str

        # Return a recarray with the spectrum data
        if output_address is None:
            return np.core.records.fromarrays([out_arr[:, i] for i in range(out_arr.shape[1])], names=list(headers))

        # Save to a file
        else:
            return np.savetxt(output_address, out_arr, **default_kwargs)


class SpecTreatment(LineFitting, RedshiftFitting):

    def __init__(self, spectrum):

        # Instantiate the dependencies
        LineFitting.__init__(self)

        # Lime spectrum object with the scientific data
        self._spec = spectrum
        self.line = None
        self._i_line = 0
        self._n_lines = 0

    def bands(self, label, bands=None, fit_cfg=None, min_method='least_squares', profile='g-emi', cont_from_bands=True,
              err_from_bands=None, temp=10000.0, default_cfg_prefix='default', obj_cfg_prefix=None, update_default=True):

        """

        This function fits a line on the spectrum object from a given band.

        The first input is the line ``label``. The user can provide a string with the default `LiMe notation
        <https://lime-stable.readthedocs.io/en/latest/inputs/n_inputs2_line_labels.html>`_. Otherwise, the user can
        provide the transition wavelength in the same units as the spectrum and the transition will be queried from the
        ``bands`` argument.

        The second input is the line ``bands`` this argument can be a six value array with the same units as the
        spectrum wavelength specifying the line position and continua location. Otherwise, the ``bands`` can be a pandas
        dataframe (or the frame address) and the wavelength array will be automatically query from it.

        If the ``bands`` are not provided by the user, the default bands database will be used. You can learn more on
        `the bands documentation <https://lime-stable.readthedocs.io/en/latest/inputs/n_inputs3_line_bands.html>`_.

        The third input is a dictionary the fitting configuration ``fit_conf`` attribute. You can learn more on the
        `profile fitting documentation <https://lime-stable.readthedocs.io/en/latest/inputs/n_inputs4_fit_configuration.html>`_.

        The ``min_method`` argument provides the minimization algorithm for the `LmFit functions
        <https://lmfit.github.io/lmfit-py/fitting.html#lmfit.minimizer.Minimizer.minimize>`_.

        By default, the profile fitting assumes an emission Gaussian shape, with ``profile="g-emi"``. The profile keywords
        are described on the `label documentation <https://lime-stable.readthedocs.io/en/latest/inputs/n_inputs2_line_labels.html>`_

        The ``cont_from_bands=True`` argument forces the continuum to be measured from the adjacent line bands. If
        ``cont_from_bands=False`` the continuum gradient is calculated from the first and last pixel from the line band
        (w3-w4)

        For the calculation of the thermal broadening on the emission lines the user can include the line electron
        temperature in Kelvin. The default value ``temp`` is 10000 K.

        :param label: Line label or wavelength transition to be queried on the ``bands`` dataframe.
        :type label: str, float, optional

        :param bands: Bands six-value array, bands dataframe (or file address to the dataframe).
        :type bands: np.array, pandas.dataframe, str, Path, optional

        :param fit_cfg: Fitting configuration.
        :type fit_cfg: dict, optional

        :param min_method: `Minimization algorithm <https://lmfit.github.io/lmfit-py/fitting.html#lmfit.minimizer.Minimizer.minimize>`_.
                            The default value is 'least_squares'
        :type min_method: str, optional

        :param profile: Profile type for the fitting. The default value ``g-emi`` (Gaussian-emission).
        :type profile: str, optional

        :param cont_from_bands: Check for continuum calculation from adjacent bands. The default value is True.
        :type cont_from_bands: bool, optional

        :param temp: Transition electron temperature for thermal broadening calculation. The default value is 10000K.
        :type temp: bool, optional

        :param default_cfg_prefix: Label for the default configuration section in the ```fit_conf`` variable.
        :type default_cfg_prefix: str, optional

        :param obj_cfg_prefix: Label for the object configuration section in the ```fit_conf`` variable.
        :type obj_cfg_prefix: str, optional

        """

        # Make a copy of the fitting configuration
        input_conf = check_fit_conf(fit_cfg, default_cfg_prefix, obj_cfg_prefix, update_default)

        # User inputs override default behaviour for the pixel error and the continuum calculation
        err_from_bands = True if (err_from_bands is None) and (self._spec.err_flux is None) else err_from_bands
        cont_from_bands = True if cont_from_bands is None else cont_from_bands

        # Interpret the input line
        self.line = Line(label, bands, input_conf, profile, cont_from_bands)

        # Check the line selection is valid
        idcs_selection = review_bands(self._spec, self.line, user_cont_from_bands=cont_from_bands, user_err_from_bands=err_from_bands)
        if idcs_selection is not None:

            # Unpack the line selections
            idcs_line, idcs_continua = idcs_selection

            # Compute line continuum
            self.continuum_calculation(idcs_line, idcs_continua, cont_from_bands)

            # Compute line flux error
            pixel_err_arr = self.pixel_error_calculation(idcs_continua, err_from_bands)

            # Non-parametric measurements
            self.integrated_properties(self.line, self._spec.wave[idcs_line], self._spec.flux[idcs_line], pixel_err_arr[idcs_line])

            # Import kinematics if requested
            import_line_kinematics(self.line, 1 + self._spec.redshift, self._spec.frame, input_conf)

            # Profile fitting measurements
            idcs_fitting = idcs_selection[0] + idcs_selection[1] if cont_from_bands else idcs_selection[0]
            self.profile_fitting(self.line,
                                 x_arr=self._spec.wave[idcs_fitting],
                                 y_arr=self._spec.flux[idcs_fitting],
                                 err_arr= pixel_err_arr[idcs_fitting],
                                 user_conf=input_conf, fit_method=min_method)

            # Instrumental and thermal corrections for the lines # TODO make res_power array length of wave
            sigma_corrections(self.line, idcs_line, self._spec.wave[idcs_line], self._spec.res_power, temp)

            # Recalculate the SNR with the profile parameters
            self.line.snr_line = signal_to_noise_rola(self.line.amp, self.line.cont_err, self.line.n_pixels)

            # Save the line parameters
            results_to_log(self.line, self._spec.frame, self._spec.norm_flux)

        return

    def frame(self, bands, fit_cfg=None, min_method='least_squares', profile='g-emi', cont_from_bands=None, err_from_bands=None,
              temp=10000.0, line_list=None, default_cfg_prefix='default', obj_cfg_prefix=None, update_default=True, line_detection=False,
              plot_fit=False, progress_output='bar'):

        """

        This function measures multiple lines on the spectrum object from a bands dataframe.

        The input ``bands_df`` can be a pandas.Dataframe or a link to its file.

        The argument ``fit_conf`` provides the `profile-fitting configuration <https://lime-stable.readthedocs.io/en/latest/inputs/n_inputs4_fit_configuration.html>`_.

        The ``min_method`` argument provides the minimization algorithm for the `LmFit functions
        <https://lmfit.github.io/lmfit-py/fitting.html#lmfit.minimizer.Minimizer.minimize>`_.

        By default, the profile fitting assumes an emission Gaussian shape, with ``profile="g-emi"``. The profile keywords
        are described on the `label documentation <https://lime-stable.readthedocs.io/en/latest/inputs/n_inputs2_line_labels.html>`_

        The ``cont_from_bands=True`` argument forces the continuum to be measured from the adjacent line bands. If
        ``cont_from_bands=False`` the continuum gradient is calculated from the first and last pixel from the line band
        (w3-w4).

        For the calculation of the thermal broadening on the emission lines the user can include the line electron
        temperature in Kelvin. The default value ``temp`` is 10000 K.

        The user can limit the fitting to certain bands with the ``lines_list`` argument.

        If the input ``fit_conf`` has multiple sections, this function will read the parameters from the ``default_conf_key``
        argument, whose default value is "default". If the input dictionary also has a section title with the
        ``id_conf_label`` _line_fitting the ``default_conf_key`` _line_fitting parameters will be **updated** by the
        object configuration.

        If ``line_detection=True`` the input ``bands_df`` measurements will be limited to those bands with a line detection.
        The local configuration for the line detection algorithm can be provided from the fit_conf entries.

        If ``plot_fit=True`` this function will plot profile after each fitting.

        The ``progress_output`` argument determines the progress console message. A "bar" value will show a progress bar,
        while a "counter" value will print a message with the current line being measured. Finally, a None value will not
        show any message.

        :param bands: Bands dataframe (or file address to the dataframe).
        :type bands: pandas.Dataframe, str, path.Pathlib

        :param fit_cfg: Fitting configuration.
        :type fit_cfg: dict, optional

        :param min_method: `Minimization algorithm <https://lmfit.github.io/lmfit-py/fitting.html#lmfit.minimizer.Minimizer.minimize>`_.
                            The default value is 'least_squares'
        :type min_method: str, optional

        :param profile: Profile type for the fitting. The default value ``g-emi`` (Gaussian-emission).
        :type profile: str, optional

        :param cont_from_bands: Check for continuum calculation from adjacent bands. The default value is True.
        :type cont_from_bands: bool, optional

        :param temp: Transition electron temperature for thermal broadening calculation. The default value is 10000K.
        :type temp: bool, optional

        :param line_list: Line list to measure from the bands dataframe.
        :type line_list: list, optional

        :param default_cfg_prefix: Label for the default configuration section in the ```fit_conf`` variable.
        :type default_cfg_prefix: str, optional

        :param obj_cfg_prefix: Label for the object configuration section in the ```fit_conf`` variable.
        :type obj_cfg_prefix: str, optional

        :param line_detection: Set to True to run the dectection line algorithm prior to line measurements.
        :type line_detection: bool, optional

        :param plot_fit: Set to True to plot the profile fitting at each iteration.
        :type plot_fit: bool, optional

        :param progress_output: Progress message output. The options are "bar" (default), "counter" and "None".
        :type progress_output: str, optional

        """

        # Check if the lines log is a dataframe or a file address
        bands = check_file_dataframe(bands)

        if bands is not None:

            # Crop the analysis to the target lines
            if line_list is not None:
                idcs = bands.index.isin(line_list)
                bands = bands.loc[idcs]

            # Load configuration
            input_conf = check_fit_conf(fit_cfg, default_cfg_prefix, obj_cfg_prefix, line_detection=line_detection)

            # Line detection if requested
            if line_detection:

                # Review the configuration entries
                cont_fit_conf = input_conf.get('continuum', {})
                detect_conf = input_conf.get('peaks_troughs', {})

                # Perform the line detection
                self._spec.fit.continuum(**cont_fit_conf)
                bands = self._spec.infer.peaks_troughs(bands, **detect_conf)

            # Define lines to treat through the lines
            label_list = bands.index.to_numpy()
            self._n_lines = label_list.size

            # Loop through the lines
            if self._n_lines > 0:

                # On screen progress bar
                pbar = ProgressBar(progress_output, f'{self._n_lines} lines')
                if progress_output is not None:
                    print(f'\nLine fitting progress:')

                for self._i_line in np.arange(self._n_lines):

                    # Ignore line if part of a blended/merge group
                    line = label_list[self._i_line]
                    measure_check = check_compound_line_exclusion(line, bands)

                    if measure_check:

                        # Progress message
                        pbar.output_message(self._i_line, self._n_lines, pre_text="", post_text=f'({line})')

                        # Fit the lines
                        self.bands(line, bands, input_conf, min_method, profile,
                                   cont_from_bands=cont_from_bands, err_from_bands=err_from_bands,
                                   temp=temp,
                                   obj_cfg_prefix=None, default_cfg_prefix=None)

                        if plot_fit:
                            self._spec.plot.bands()

            else:
                msg = f'No lines were measured from the input dataframe:\n - line_list: {line_list}\n - line_detection: {line_detection}'
                _logger.debug(msg)

        else:
            _logger.info(f'Not input dataframe. Lines were not measured')

        return

    def continuum(self, degree_list, emis_threshold, abs_threshold=None, smooth_length=None, plot_steps=False,
                  log_scale=False):

        """

        This function fits the spectrum continuum in an iterative process. The user specifies two parameters: the ``degree_list``
        for the fitted polynomial and the ``threshold_list``` for the multiplicative standard deviation factor. At each
        interation points beyond this flux threshold are excluded from the continuum current fittings. Consequently,
        the user should aim towards more constrictive parameter values at each iteration.

        The user can specify a window length over which the spectrum will be smoothed before fitting the continuum using
        the ``smooth_length`` parameter.

        The user can visually inspect the fitting output graphically setting the parameter ``plot_steps=True``.

        :param degree_list: Integer list with the degree of the continuum polynomial
        :type degree_list: list

        :param emis_threshold: Float list for the multiplicative continuum standard deviation flux factor
        :type emis_threshold: list

        :param smooth_length: Size of the smoothing window to convolve the spectrum. The default value is None.
        :type smooth_length: integer, optional

        :param plot_steps: Set to "True" to plot the fitted continuum at each iteration.
        :type plot_steps: bool, optional

        :return:

        """

        # Create a pre-Mask based on the original mask if available
        mask_cont = ~self._spec.flux.mask
        input_wave, input_flux = self._spec.wave.data, self._spec.flux.data

        # Smooth the spectrum
        if smooth_length is not None:
            smoothing_window = np.ones(smooth_length) / smooth_length
            input_flux_s = np.convolve(input_flux, smoothing_window, mode='same')
        else:
            input_flux_s = input_flux

        # Loop through the fitting degree
        abs_threshold = emis_threshold if abs_threshold is None else abs_threshold
        for i, degree in enumerate(degree_list):

            # First iteration use percentile limits for an initial fit
            if i == 0:
                low_lim, high_lim = np.nanpercentile(input_flux_s[mask_cont], (16, 84))
                mask_cont_0 = mask_cont & (input_flux_s >= low_lim) & (input_flux_s <= high_lim)
                cont_fit = continuum_model_fit(input_wave, input_flux_s, mask_cont_0, degree)

            # Establishing the flux limits
            std_flux = np.nanstd((input_flux_s - cont_fit)[mask_cont])
            low_lim, high_lim = cont_fit - abs_threshold[i] * std_flux, cont_fit + emis_threshold[i] * std_flux

            # Add new entries to the mask
            mask_cont = mask_cont & (input_flux_s >= low_lim) & (input_flux_s <= high_lim)

            # Fit continuum
            cont_fit = continuum_model_fit(input_wave, input_flux_s, mask_cont, degree)

            # Compute the continuum and assign replace the value outside the bands the new continuum
            if plot_steps:
                ax_cfg = {'title':f'Continuum fitting, iteration ({i+1}/{len(degree_list)})'}
                self._spec.plot._continuum_iteration(input_wave, input_flux, cont_fit, input_flux_s, mask_cont, low_lim,
                                                     high_lim, emis_threshold[i], ax_cfg, log_scale=log_scale)

        # Include the standard deviation of the spectrum for the unmasked pixels
        self._spec.cont = np.ma.masked_array(cont_fit, self._spec.flux.mask)
        self._spec.cont_std = np.std((input_flux_s - cont_fit)[mask_cont])

        return


class CubeTreatment(LineFitting):

    def __init__(self, cube):

        # Instantiate the dependencies
        LineFitting.__init__(self)

        # Lime spectrum object with the scientific data
        self._cube = cube
        self._spec = None

    def spatial_mask(self, mask_file, output_address, bands=None, fit_cfg=None, mask_list=None, line_list=None,
                     log_ext_suffix='_LINELOG', min_method='least_squares', profile='g-emi', cont_from_bands=True,
                     temp=10000.0, default_cfg_prefix='default', update_default=True, line_detection=False, progress_output='bar',
                     plot_fit=False, header=None, join_output_files=True):

        """

        This function measures lines on an IFS cube from an input binary spatial ``mask_file``.

        The results are stored in a multipage ".fits" file, each page contains a measurements and it is named after the
        spatial array coordinates and the ``log_ext_suffix`` (i.e. "idx_j-idx_i_LINELOG")

        The input ``bands`` can be a pandas.Dataframe or an address to the file. The user can specify one bands file
        per mask page on the ``mask_file``. To do this, the ``fit_conf`` argument must include a section for every mask
        on the ``mask_list`` (i.e. "Mask1_line_fitting"). This function will check for a key "bands" and load the
        corresponding bands.

        The fitting configuration in the ``fit_conf`` argument accepts a three-level configuration. At the lowest level,
        The ``default_conf_key`` points towards the default configuration for all the spaxels analyzed on the cube
        (i.e. "default_line_fitting"). At an intermediate level, the parameters from the section with a name from the
        ``mask_list`` (i.e. "Mask1_line_fitting") will be applied to the spaxels in the corresponding mask. Finally, at
        the highest level, the user can provide a spaxel fitting configuration with the spatial array coordiantes
        "50-28_LINELOG". In all these cases the higher level configurate **updates** the lower levels (only common entries
        are replaced)

        .. attention::
            In this multi-level configuration design, the higher level entries **update** the lower level entries:
            only shared entries are overwritten, the final configuration will include all the entries from the
            default mask and spaxel sections.

        If the ``line_detection`` is set to True the function proceeds to run the line detection algorithm prior to the
        fitting of the lines. The user provide the configuration parameters for the line_detection function in the
        ``fit_conf`` argument. At the default, mask or spaxel configuration the user needs to specify these entries with
        the "function name" + "." + "function argument" (i.e. "line_detection.emission_type='emission'"). The multi-level
        configuration described above will be applied to this function parameters as well.

        .. note::
            The parameters for the ``line.detection`` can be found on the documentation. The user doesn't need to specify
            a "lime_detection.bands" parameter. The input bands from the corresponding mask will be used.

        :param mask_file: Address of binary spatial mask file
        :type mask_file: str, pathlib.Path

        :param output_address: File address for the output measurements log.
        :type output_address: str, pathlib.Path

        :param bands: Bands dataframe (or file address to the dataframe).
        :type bands: pandas.Dataframe, str, path.Pathlib

        :param fit_cfg: Fitting configuration.
        :type fit_cfg: dict, optional

        :param mask_list: Masks name list to explore on the ``masks_file``.
        :type mask_list: list, optional

        :param line_list: Line list to measure from the bands dataframe.
        :type line_list: list, optional

        :param log_ext_suffix: Suffix for the measurements log pages. The default value is "_LINELOG".
        :type log_ext_suffix: str, optional.

        :param min_method: `Minimization algorithm <https://lmfit.github.io/lmfit-py/fitting.html#lmfit.minimizer.Minimizer.minimize>`_.
                            The default value is 'least_squares'
        :type min_method: str, optional

        :param profile: Profile type for the fitting. The default value ``g-emi`` (Gaussian-emission).
        :type profile: str, optional

        :param cont_from_bands: Check for continuum calculation from adjacent bands. The default value is True.
        :type cont_from_bands: bool, optional

        :param temp: Transition electron temperature for thermal broadening calculation. The default value is 10000K.
        :type temp: bool, optional

        :param default_cfg_prefix: Label for the default configuration section in the ```fit_conf`` variable.
        :type default_cfg_prefix: str, optional

        :param line_detection: Set to True to run the dectection line algorithm prior to line measurements.
        :type line_detection: bool, optional

        :param plot_fit: Set to True to plot the spectrum lines fitting at each iteration.
        :type plot_fit: bool, optional

        :param progress_output: Progress message output. The options are "bar" (default), "counter" and "None".
        :type progress_output: str, optional

        :param header: Dictionary for parameter ".fits" file headers.
        :type header: dict, optional

        :param join_output_files: In the case of multiple masks, join the individual output ".fits" files into a single
                                  one. If set to False there will be one output file named per mask named after it. The
                                  default value is True.
        :type join_output_files: bool, optional

        """

        if bands is not None:
            bands = check_file_dataframe(bands)

        # Check if the mask variable is a file or an array
        mask_maps = check_file_array_mask(mask_file, mask_list)
        mask_list = np.array(list(mask_maps.keys()))
        mask_data_list = list(mask_maps.values())

        # Check the mask configuration is included if there are no masks
        input_masks = mask_list if bands is None else None
        input_conf = check_fit_conf(fit_cfg, default_key=None, obj_key=None, update_default=update_default,
                                    group_list=input_masks)

        # Check if the output log folder exists
        output_address = Path(output_address)
        address_dir = output_address.parent
        if not address_dir.is_dir():
            raise LiMe_Error(f'The folder of the output log file does not exist at {output_address}')
        address_stem = output_address.stem

        # Determine the spaxels to treat at each mask
        spax_counter, total_spaxels, spaxels_dict = 0, 0, {}
        for idx_mask, mask_data in enumerate(mask_data_list):
            spa_mask, hdr_mask = mask_data
            idcs_spaxels = np.argwhere(spa_mask)

            total_spaxels += len(idcs_spaxels)
            spaxels_dict[idx_mask] = idcs_spaxels

        # Header data
        hdr_coords = extract_wcs_header(self._cube.wcs, drop_axis='spectral') if self._cube.wcs is not None else None

        # Loop through the masks
        n_masks = len(mask_list)
        mask_log_files_list = [address_dir/f'{address_stem}_MASK-{mask_name}.fits' for mask_name in mask_list]

        for i in np.arange(n_masks):

            # HDU_container
            hdul_log = fits.HDUList([fits.PrimaryHDU()])

            # Mask progress indexing
            mask_name = mask_list[i]
            mask_hdr = mask_data_list[i][1]
            idcs_spaxels = spaxels_dict[i]

            # Recover the fitting configuration
            mask_conf = check_fit_conf(input_conf, default_cfg_prefix, mask_name)

            # Load the mask log if provided
            if bands is None:
                bands_file = mask_conf['bands']
                bands_path = Path(bands_file).absolute() if bands_file[0] == '.' else Path(bands_file)
                bands_in = load_frame(bands_path)
            else:
                bands_in = bands

            # Loop through the spaxels
            n_spaxels = idcs_spaxels.shape[0]
            n_lines, start_time = 0, time()

            print(f'\nSpatial mask {i + 1}/{n_masks}) {mask_name} ({n_spaxels} spaxels)')
            pbar = ProgressBar(progress_output, f'mask')
            for j in np.arange(n_spaxels):

                idx_j, idx_i = idcs_spaxels[j]
                spaxel_label = f'{idx_j}-{idx_i}'

                # Get the spaxel fitting configuration
                spaxel_conf = input_conf.get(f'{spaxel_label}_line_fitting')
                spaxel_conf = mask_conf if spaxel_conf is None else {**mask_conf, **spaxel_conf}

                # Spaxel progress message
                pbar.output_message(j, n_spaxels, pre_text="", post_text=f'(spaxel coordinate. {idx_j}-{idx_i})')

                # Get spaxel data
                spaxel = self._cube.get_spectrum(idx_j, idx_i, spaxel_label)

                # Fit the lines
                spaxel.fit.frame(bands_in, spaxel_conf, line_list=line_list, min_method=min_method,
                                 line_detection=line_detection, profile=profile, cont_from_bands=cont_from_bands,
                                 temp=temp, progress_output=None, plot_fit=None, obj_cfg_prefix=None,
                                 default_cfg_prefix=None)

                # Count the number of measurements
                n_lines += spaxel.frame.index.size

                # Create page header with the default data
                hdr_i = fits.Header()

                # Add WCS information
                if hdr_coords is not None:
                    hdr_i.update(hdr_coords)

                # Add user information
                if header is not None:
                    page_hdr = header.get(f'{spaxel_label}{log_ext_suffix}', None)
                    page_hdr = header if page_hdr is None else page_hdr
                    hdr_i.update(page_hdr)

                # Save to a fits file
                linesHDU = log_to_HDU(spaxel.frame, ext_name=f'{spaxel_label}{log_ext_suffix}', header_dict=hdr_i)

                if linesHDU is not None:
                    hdul_log.append(linesHDU)

                # Plot the fittings if requested:
                if plot_fit:
                    spaxel.plot.spectrum(include_fits=True, rest_frame=True)

            # Save the log at each new mask
            hdul_log.writeto(mask_log_files_list[i], overwrite=True, output_verify='ignore')
            hdul_log.close()

            # Computation time and message
            end_time = time()
            elapsed_time = end_time - start_time
            print(f'\n{n_lines} lines measured in {elapsed_time/60:0.2f} minutes.')

        if join_output_files:
            output_comb_file = f'{address_dir/address_stem}.fits'

            # In case of only one file just rename it
            if len(mask_list) == 1:
                mask_0_path = Path(mask_log_files_list[0])
                mask_0_path.rename(Path(output_comb_file))
            else:
                print(f'\nJoining spatial log files ({",".join(mask_list)}) -> {output_comb_file}')
                join_fits_files(mask_log_files_list, output_comb_file, delete_after_join=join_output_files)

        # else:
        #     # Just one mask and Join is False
        #     if len(mask_list) == 1:
        #         output_comb_file = f'{address_dir / address_stem}.fits'
        #         mask_0_path = Path(mask_log_files_list[0])
        #         mask_0_path.rename(Path(output_comb_file))


        return


