import numpy as np
import pandas as pd
import decimal

import lime
from lime.transitions import air_to_vacuum_function, construct_classic_notation
from lime.fitting.lines import velocity_to_wavelength_band

from pathlib import Path


def format_lines_database(df_lines, redshift=0, band_velocity_sigma=100, n_sigma=4, delta_lambda_inst=0):

    # Adjust previous database
    index_arr = df_lines.index.to_numpy()
    updated_labels = [None] * index_arr.size

    # Generate synthetic spectrum to compute the initial bands values if necessary
    wave_arr, flux_arr = np.arange(0, 500000, 1), np.random.normal(0, 1, size=500000)
    spec = lime.Spectrum(wave_arr, flux_arr, redshift=redshift)

    # Not repeated entries in the original dataset
    assert np.any(df_lines.index.duplicated()) is np.False_, print(f'Repeated lines: {df_lines.loc[df_lines.index.duplicated()]}')

    for i, idx in enumerate(index_arr):
        line = lime.Line.from_transition(idx)
        if pd.notnull(df_lines.loc[idx, 'trans']):
            line.trans = df_lines.loc[idx, 'trans']
        wave_vac = df_lines.loc[idx, 'wave_vac']
        decimals = abs(decimal.Decimal(str(wave_vac)).as_tuple().exponent)
        wave_air = np.around(air_to_vacuum_function(wave_vac), decimals)

        # Confirm the vacuum wavelength
        air_trans = True if 2000 <= wave_vac <= 10000 else False

        line.wavelength = wave_air if air_trans else wave_vac
        line.update_labels(sig_digits=int(np.log10(line.wavelength)) + 1)

        # Add a decimal if the line is already there
        if line.label not in updated_labels:
            updated_labels[i] = line.label
        else:
            message = f'{line.label} -> '
            line.update_labels(sig_digits=int(np.log10(line.wavelength)) + 2)
            updated_labels[i] = line.label
            message += updated_labels[i]

        # Update dataframe values
        df_lines.loc[idx, 'wavelength'] = wave_air if air_trans else wave_vac
        # df_lines.loc[idx, 'particle'] = line.particle.label

        # Update the label using the air/wavelength transition
        construct_classic_notation(line)
        df_lines.loc[idx, 'latex_label'] = line.latex_label

        # Add missing entries
        if pd.isnull(df_lines.loc[idx, 'units_wave']):
            df_lines.loc[idx, 'units_wave'] = line.units_wave

        if pd.isnull(df_lines.loc[idx, 'particle']):
            df_lines.loc[idx, 'particle'] = line.particle.label

        # if (line.trans == 'sem') and (line.latex_label[0][:2] == '$['):
        #     df_lines.loc[idx, 'latex_label'] = np.array(['$' + line.latex_label[0][2:]])

    # Update the new names
    df_lines['new_index'] = updated_labels
    df_lines.set_index('new_index', inplace=True)
    df_lines.index.name = None

    # Bands initial width    np.floor(np.log10(np.abs(lambda_obs))).astype(int)
    lambda_obs = df_lines['wavelength'].to_numpy()
    delta_lambda = velocity_to_wavelength_band(n_sigma, band_velocity_sigma, lambda_obs, delta_lambda_inst)
    delta_lambda = n_sigma * (band_velocity_sigma / 299792.458) * lambda_obs + (lambda_obs * 0.0004)

    # Add missing central bands
    idcs_mis_central = pd.isnull(df_lines['w3']) | pd.isnull(df_lines['w4'])
    df_lines.loc[idcs_mis_central, 'w3'] = (lambda_obs[idcs_mis_central] - delta_lambda[idcs_mis_central]) / (1 + redshift)
    df_lines.loc[idcs_mis_central, 'w4'] = (lambda_obs[idcs_mis_central] + delta_lambda[idcs_mis_central]) / (1 + redshift)

    # Add missing side bands
    idcs_side = pd.isnull(df_lines['w1']) | pd.isnull(df_lines['w2'])
    df_lines.loc[idcs_side, 'w2'] = (df_lines.loc[idcs_side, 'w3'] - delta_lambda[idcs_side]/2) / (1 + redshift)
    df_lines.loc[idcs_side, 'w1'] = (df_lines.loc[idcs_side, 'w2'] - delta_lambda[idcs_side]/2) / (1 + redshift)

    idcs_side = pd.isnull(df_lines['w5']) | pd.isnull(df_lines['w6'])
    df_lines.loc[idcs_side, 'w5'] = (df_lines.loc[idcs_side, 'w4'] + delta_lambda[idcs_side]/2) / (1 + redshift)
    df_lines.loc[idcs_side, 'w6'] = (df_lines.loc[idcs_side, 'w5'] + delta_lambda[idcs_side]/2) / (1 + redshift)

    # Check sorted order
    unsorted_mask =  df_lines['wavelength'].diff().fillna(0) < 0
    unsorted_indexes = df_lines.index[unsorted_mask].tolist()

    if len(unsorted_indexes) == 0:
        print("Column is sorted.")
    else:
        print("Column is NOT sorted. Unsorted indexes:", unsorted_indexes)
        print(df_lines.loc[unsorted_indexes])

    return df_lines


if __name__ == "__main__":

    current_file_folder = Path(__file__).resolve().parent
    PARENT_DATABASE_path = current_file_folder / 'lines_database_v2.0.0.xlsx'
    CHILD_DATABASE_path = current_file_folder / 'lines_database_v2.0.4.txt'

    parent_db = pd.read_excel(PARENT_DATABASE_path, header=0, index_col=0)
    child_db = format_lines_database(parent_db)

    lime.save_frame(CHILD_DATABASE_path, child_db)




