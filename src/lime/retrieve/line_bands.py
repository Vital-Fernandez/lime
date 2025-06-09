import logging
import numpy as np
import pandas as pd
from lime.io import LiMe_Error
from lime.transitions import Line


_logger = logging.getLogger('LiMe')


def pars_bands_conf(spec, bands, fit_conf, composite_lines, automatic_grouping=True):

    # Use the input groups
    if automatic_grouping is False:

        # Get the the grouped lines
        groups_dict = {} if fit_conf is False else {comp: group_label
                                                   for comp, group_label in fit_conf.items()
                                                   if comp.endswith(('_b', '_m'))}

        # Limit the selection to the user lines
        if composite_lines is not None:
            groups_dict = {line: comps for line, comps in groups_dict.items() if line in composite_lines}

    # Automatic group review
    else:

        # Check the input dataframe is sorted
        if not np.all(np.diff(bands.wavelength) >= 0):
            _logger.warning(f'The input bands table is not sorted. This can cause issues in the bands generation:'
                            f'\n{bands["wavelength"]}')

        # Get list all the line groups and their lines
        if fit_conf:

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
                                groups_i.append(True)
                            else:                       # Merged line
                                sub_group_label = fit_conf.get(line)
                                if sub_group_label:
                                    items = sub_group_label.split('+')
                                    lines_i += items
                                    groups_i += [False] * len(items)
                                else:
                                    raise LiMe_Error(f'The merged line: "{line}" in grouped line: "{comp}={group_label}" '
                                                     f'is not specified.\nPlease define a "{line}=LineA+LineB" '
                                                     f'in your configuration file.')

                        # Convert the sub_group_type to the relation
                        groups_i = np.array(groups_i)
                        groups_i = groups_i[:-1] != groups_i[1:]

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

            # Array to keep track of lines which have been assigned:
            assigned_lines = np.zeros(line_list.size).astype(bool)

            # Compare the observed line groups
            groups_dict = {}
            if line_list.size > 1:

                # Get limits of the bands on the spectrum wavelength range
                w3_arr = np.searchsorted(spec.wave_rest.data, bands.loc[line_list, 'w3'].to_numpy())
                w4_arr = np.searchsorted(spec.wave_rest.data, bands.loc[line_list, 'w4'].to_numpy())

                # Generate binary matrix with the line bands location
                wave_matrix = np.zeros((lambda_arr.size, spec.wave_rest.data.size))
                cols = np.arange(wave_matrix.shape[1])
                wave_matrix[(cols >= w3_arr[:, None]) & (cols <= w4_arr[:, None])] = 1

                # Compute the decision matrix with the common pixels
                decision_matrix = wave_matrix @ wave_matrix.T

                # pixels_width = wave_matrix.sum(axis=1)
                # blended_matrix = decision_matrix < np.ceil(pixels_width/3)[:, None]
                # math_dict = dict(zip(line_arr, np.arange(line_arr.size)))

                # Loop through the input groups to confirm the best match
                for i, group in enumerate(group_names):

                    # Diagnostic to establish the relation between the lines
                    threshold = 2
                    w3_arr = np.searchsorted(spec.wave_rest.data, bands.loc[group_lines[i], 'w3'].to_numpy())
                    w4_arr = np.searchsorted(spec.wave_rest.data, bands.loc[group_lines[i], 'w4'].to_numpy())
                    mu = (w3_arr + w4_arr) / 2
                    sigma = (w4_arr - w3_arr) / 6
                    delta_mu = np.diff(mu)
                    sigma_avg = np.sqrt((sigma[:-1] ** 2 + sigma[1:] ** 2) / 2)
                    R = delta_mu / sigma_avg
                    resolvable = R > threshold

                    match_group = True if np.all(resolvable == group_blended_check[i]) else False

                    # Before saving group check that there are no more lines grouped in the observation
                    if match_group:
                        idcs_i = sub_bands.index.get_indexer(group_lines[i])
                        if np.all(np.sum(decision_matrix[idcs_i, :] > 0, axis=1) == idcs_i.size):
                            groups_dict[group] = fit_conf[group]
                            assigned_lines[idcs_i] = True

        # Invalid
        else:
            _logger.warning(f'The user requested automatic_grouping for the line transitions but the "fit_conf" is empty')

    # Applyt the requested group changes
    rename_dict, exclude_list= {}, []
    group_dict, w3_dict, w4_dict = {}, {}, {}
    for new_label, group_label in groups_dict.items():

        component_list = np.unique([Line(x).core for x in group_lines[group_names.index(new_label)] if '_k-' not in x])
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

    # Change the latex labels
    for old_label, new_label in rename_dict.items():
        line = Line(new_label, band=bands, fit_conf=groups_dict, update_latex=True)
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

# if np.all(idcs_i > -1):

# # Logic for single, merged and blended lines
# shared_pixels = decision_matrix[idcs_i[:-1], idcs_i[1:]]
# if np.any(shared_pixels > 0):
#     match_group =False
#     # line_pixels = np.max(pixels_width[idcs_i])
#     # diag_arr = resolvable
#     #
#     # obs_type = ['_b'] if np.all(diag_arr) else ['_m'] if np.all(~diag_arr) else None
# else:
#     match_group = False

# Compare observed group versus user group

# else:
#
#     # Lines not assigned before:
#     if np.all(assigned_lines[idcs_i] == False):
#
#         # Group consists in blended merged lines: Assigned single merged
#         if (group_name[-2:] == '_m') and np.any(diag_arr[:-1] & diag_arr[1:]):
#             output_groups[group_name] = group_label
#             assigned_lines[idcs_i] = True
# # Get groups of common entries
# from scipy.sparse import csr_matrix, csgraph
# _, auto_labels = csgraph.connected_components(csgraph=csr_matrix(decision_matrix > 1), directed=False)
# for labels, group in zip(line_arr, auto_labels):
#     print(f"{labels}   {group}")
