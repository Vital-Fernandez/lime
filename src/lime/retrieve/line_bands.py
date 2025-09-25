import logging
import numpy as np
import pandas as pd
from lime.io import LiMe_Error
from lime.transitions import Line


_logger = logging.getLogger('LiMe')


def deblend_criteria(mu_arr, sigma_arr, Rayleigh_threshold):

    delta_mu = np.diff(mu_arr)
    sigma_avg = np.sqrt((sigma_arr[:-1] ** 2 + sigma_arr[1:] ** 2) / 2)
    diagnostic = delta_mu / sigma_avg
    resolvable = diagnostic > Rayleigh_threshold

    return resolvable


def pars_bands_conf(spec, bands, fit_conf, composite_lines, automatic_grouping, n_sigma, Rayleigh_threshold):

    # Use the input groups
    if automatic_grouping is False:

        # Get the the grouped lines
        groups_dict = {} if fit_conf is False else {comp: group_label
                                                   for comp, group_label in fit_conf.items()
                                                   if comp.endswith(('_b', '_m'))}

        # Limit the selection to the user lines
        if composite_lines is not None:
            groups_dict = {line: comps for line, comps in groups_dict.items() if line in composite_lines}

        group_names = list(groups_dict.keys())
        group_lines = [group.split('+') for group in groups_dict.values()]
        groups_dict = dict(zip(group_names, group_lines))

    # Automatic group review
    else:

        # Check the input dataframe is sorted
        if not np.all(np.diff(bands.wavelength) >= 0):
            _logger.warning(f'The input bands table is not sorted. This can cause issues in the bands generation:'
                            f'\n{bands["wavelength"]}')


        # Get list all the line groups and their lines
        line_list, group_lines = [], []
        group_names, group_blended_check = [], []
        for comp, group_label in fit_conf.items():
            if comp.endswith(('_b', '_m')):

                # Homogeneous group
                if '_m' not in group_label:
                    lines_i = group_label.split('+')
                    groups_i = [True] * (len(lines_i) - 1) if comp[-2:] == '_b' else [False] * (len(lines_i) - 1)

                # Mixed group (merged child line in the blended parent group)
                else:
                    lines_i, groups_i = [], []
                    for i, line in enumerate(group_label.split('+')):
                        if line[-2:] != '_m':       # Single line
                            lines_i.append(line)
                            groups_i.append(0)
                        else:                       # Merged line
                            sub_group_label = fit_conf.get(line)
                            if sub_group_label:
                                items = sub_group_label.split('+')
                                lines_i += items
                                groups_i += [i] * len(items)
                            else:
                                raise LiMe_Error(f'The merged line: "{line}" in grouped line: "{comp}={group_label}" '
                                                 f'is not specified.\nPlease define a "{line}=LineA+LineB" '
                                                 f'in your configuration file.')

                    # Convert the sub_group_type to the relation
                    groups_i = np.diff(groups_i).astype(bool)

                # Add the group is all lines (sorted) in current wavelength range
                idcs_i = bands.index.get_indexer(lines_i)
                if np.all(idcs_i > -1):
                    group_names.append(comp)
                    group_lines.append(bands.loc[bands.index.isin(lines_i)].index.to_numpy())
                    group_blended_check.append(groups_i)
                    line_list += lines_i

        # Sort the input lines using line banbs table
        line_list = bands.loc[bands.index.isin(line_list)].index
        sub_bands = bands.loc[line_list]
        lambda_arr = sub_bands['wavelength'].to_numpy()

        # Compare the observed line groups
        groups_dict = {}
        if line_list.size > 1:

            # Get limits of the bands on the spectrum wavelength range
            idx3_arr = np.searchsorted(spec.wave_rest.data, bands.loc[line_list, 'w3'].to_numpy())
            idx4_arr = np.searchsorted(spec.wave_rest.data, bands.loc[line_list, 'w4'].to_numpy())

            # Generate binary matrix with the line bands location
            wave_matrix = np.zeros((lambda_arr.size, spec.wave_rest.data.size))
            cols = np.arange(wave_matrix.shape[1])
            wave_matrix[(cols >= idx3_arr[:, None]) & (cols <= idx4_arr[:, None])] = 1

            # Compute the decision matrix with the common pixels
            decision_matrix = wave_matrix @ wave_matrix.T

            # Loop through the input groups to confirm the best match
            assigned_lines = np.zeros(line_list.size).astype(bool)
            for i, group in enumerate(group_names):

                # Check if lines have been assigned before and if they are not separations
                idcs_i = sub_bands.index.get_indexer(group_lines[i])
                if np.all(assigned_lines[idcs_i] == False):

                    # Check the observed group of lines is the same size as the input one
                    obs_group_size = (np.diagonal(decision_matrix[idcs_i[0]:, idcs_i], offset=1) > 0).sum() + 1
                    if obs_group_size == idcs_i.size:

                        # Establish the observed grouping of the lines
                        w3_arr = idx3_arr[idcs_i]
                        w4_arr = idx4_arr[idcs_i]
                        obs_group_blend_chek = deblend_criteria(mu_arr=(w3_arr + w4_arr) / 2,
                                                                sigma_arr=(w4_arr - w3_arr) / (n_sigma * 2),
                                                                Rayleigh_threshold= Rayleigh_threshold)

                        # Add the configuration entry if there is a matching between observation and user grouping
                        if np.all(obs_group_blend_chek == group_blended_check[i]):
                            groups_dict[group] = fit_conf[group]
                            assigned_lines[idcs_i] = True

    # Apply the requested group changes
    rename_dict, exclude_list = {}, []
    param_dict = dict(w3 = {}, w4 = {}, wavelength = {}, latex_label = {}, group_label = {})
    for new_label, group_label in groups_dict.items():
        print(new_label)
        # The grouped line replaces the reference entry
        line = Line.from_transition(new_label, fit_conf)
        old_label = line.list_comps[line.ref_idx].core

        # Extract the components includes sub-transitions
        component_list = []
        for trans in line.list_comps:
            if trans.kinem == 0:
                component_list += list(trans.param_arr('core'))

        # Only apply corrections if components are present
        idcs_comps = bands.index.isin(component_list)
        if idcs_comps.sum() == len(component_list):

            # Save the modifications
            rename_dict[old_label] = new_label
            exclude_list += component_list
            param_dict['w3'][new_label] = bands.loc[idcs_comps, 'w3'].min()
            param_dict['w4'][new_label] = bands.loc[idcs_comps, 'w4'].max()
            param_dict['wavelength'][new_label] = line.wavelength
            param_dict['latex_label'][new_label] = line.latex_label
            param_dict['group_label'][new_label] = line.group_label

        # Check the line or the same group is already there
        else:
            low, high = spec.wave_rest.compressed()[[0, -1]]
            wave_arr = line.param_arr('wavelength')

            # Lines outside wavelength range
            if np.all((wave_arr >= low) & (wave_arr <= high)):
                continue

            # Check if lines outside range
            else:
                if 'group_label' in bands.columns:
                    if not np.any(groups_dict[new_label] == bands.group_label):
                        _logger.info(f'Line component "{old_label}" for configuration entry: '
                                     f'"{new_label}={groups_dict[new_label]}" not found in lines table')
                else:
                    _logger.info(f'Missing line(s) "{np.setxor1d(bands.loc[idcs_comps].index.to_numpy(), component_list)}" '
                                 f'for configuration entry: '
                                 f'"{new_label}={groups_dict[new_label]}" in reference lines table')

    # Warn in case some of the bands dont match the database:
    if not set(exclude_list).issubset(bands.index):
        _logger.info(f' The following blended or merged lines were not found on the input lines database:\n'
                     f' - {list(set(exclude_list) - set(bands.index))}\n'
                     f' - It is recommended that the merged/blended components follow the reference transitions labels.\n')

    # Change the indexes
    bands.rename(index=dict(rename_dict), inplace=True)

    # Remove components columns
    bands.drop(exclude_list, errors='ignore', inplace=True)

    # Add the new values to the lines table
    bands['w3'] = pd.Series(bands.index.map(param_dict['w3']), index=bands.index).fillna(bands['w3'])
    bands['w4'] = pd.Series(bands.index.map(param_dict['w4']), index=bands.index).fillna(bands['w4'])
    bands['wavelength'] = pd.Series(bands.index.map(param_dict['wavelength']), index=bands.index).fillna(bands['wavelength'])
    bands['latex_label'] = pd.Series(bands.index.map(param_dict['latex_label']), index=bands.index).fillna(bands['latex_label'])

    bands['group_label'] = 'none'
    bands['group_label'] = pd.Series(bands.index.map(param_dict['group_label']), index=bands.index).fillna(bands['group_label'])

    return