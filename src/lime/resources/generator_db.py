import numpy as np
import pandas as pd
import decimal

import lime
from lime.transitions import air_to_vacuum_function
from lime.fitting.lines import velocity_to_wavelength_band

from pathlib import Path

current_file_folder = Path(__file__).resolve().parent
PARENT_DATABASE = current_file_folder/'Lines_database_18_01_2025.xlsx'
CHILD_DATABASE = current_file_folder/'lines_database_v2.0.1.txt'

def format_lines_database(redshift=0, band_velocity_sigma=100, n_sigma=4, continua_center_width=200, delta_lambda_inst=0):

    # Adjust previous database
    df_lines = pd.read_excel(PARENT_DATABASE, header=0, index_col=0)
    index_arr = df_lines.index.to_numpy()
    updated_labels = [None] * index_arr.size

    # Not repeated entries in the original dataset
    assert np.any(df_lines.index.duplicated()) is np.False_

    for i, idx in enumerate(index_arr):
        line = lime.Line.from_transition(idx)
        if pd.notnull(df_lines.loc[idx, 'trans']):
            line.trans = df_lines.loc[idx, 'trans']
        wave_vac = df_lines.loc[idx, 'wave_vac']
        decimals = abs(decimal.Decimal(str(wave_vac)).as_tuple().exponent)
        wave_air = np.around(air_to_vacuum_function(wave_vac), decimals)

        if 2000 <= wave_vac <= 10000:
            air_trans = True
        else:
            air_trans = False

        line.wavelength = np.array([wave_air]) if air_trans else np.array([wave_vac])
        line.update_labels(show_optional=False)

        # Add a decimal if the line is already there
        if line.label not in updated_labels:
            updated_labels[i] = line.label
        else:
            message = f'{line.label} -> '
            line.update_labels(decimals=1)
            updated_labels[i] = line.label
            message += updated_labels[i]

        # Update dataframe values
        df_lines.loc[idx, 'wavelength'] = wave_air if air_trans else wave_vac
        df_lines.loc[idx, 'particle'] = line.particle[0]

        line.update_labels(update_latex=True)
        df_lines.loc[idx, 'latex_label'] = line.latex_label

        if (line.trans == 'sem') and (line.latex_label[0][:2] == '$['):
            df_lines.loc[idx, 'latex_label'] = np.array(['$' + line.latex_label[0][2:]])

    # Update the new names
    df_lines['new_index'] = updated_labels
    df_lines.set_index('new_index', inplace=True)
    df_lines.index.name = None

    # Convert to spectral width
    lambda_obs = df_lines['wavelength'].to_numpy()
    delta_lambda = velocity_to_wavelength_band(n_sigma, band_velocity_sigma, lambda_obs, delta_lambda_inst)

    # Exclude lines which you dont want to update
    indcs_update = ~df_lines.index.str.contains('PAH')

    # Add new values to database in the rest frame
    df_lines.loc[indcs_update, 'w3'] = (lambda_obs[indcs_update] - delta_lambda[indcs_update]) / (1 + redshift)
    df_lines.loc[indcs_update, 'w4'] = (lambda_obs[indcs_update] + delta_lambda[indcs_update]) / (1 + redshift)

    # Adjust the sidebands
    for i, idx in enumerate(df_lines.index):

        line = lime.Line(idx)
        bands = df_lines.loc[idx, 'w1':'w6'].to_numpy()
        bands = bands.astype(float)
        lambda_obs_i = df_lines.loc[idx, 'wavelength']
        continua_center_sep = velocity_to_wavelength_band(n_sigma, continua_center_width, lambda_obs_i,
                                                          delta_lambda_inst)
        blue_lambda = (lambda_obs_i - continua_center_sep)
        red_lambda = (lambda_obs_i + continua_center_sep)
        blue_width = velocity_to_wavelength_band(n_sigma, band_velocity_sigma / 2, blue_lambda, delta_lambda_inst)
        red_width = velocity_to_wavelength_band(n_sigma, band_velocity_sigma / 2, red_lambda, delta_lambda_inst)

        if np.all(np.isnan(bands[:2])):
            df_lines.loc[idx, 'w1'] = (blue_lambda - blue_width) / (1 + redshift)
            df_lines.loc[idx, 'w2'] = (blue_lambda + blue_width) / (1 + redshift)

        if np.all(np.isnan(bands[4:])):
            df_lines.loc[idx, 'w5'] = (red_lambda - red_width) / (1 + redshift)
            df_lines.loc[idx, 'w6'] = (red_lambda + red_width) / (1 + redshift)

        newBands = df_lines.loc[idx, 'w1':'w6'].to_numpy()
        if not np.all(np.diff(newBands) > 0):
            df_lines.loc[idx, 'w1'] = (blue_lambda - blue_width) / (1 + redshift)
            df_lines.loc[idx, 'w2'] = (blue_lambda + blue_width) / (1 + redshift)
            df_lines.loc[idx, 'w5'] = (red_lambda - red_width) / (1 + redshift)
            df_lines.loc[idx, 'w6'] = (red_lambda + red_width) / (1 + redshift)

        newBands = df_lines.loc[idx, 'w1':'w6'].to_numpy()
        if not np.all(np.diff(newBands) > 0):
            print(f'Not sorted: ', line)

    # Adjust to new format
    df_lines.drop('wavelength.1', axis=1, inplace=True)
    df_lines.drop('wave_vac.1', axis=1, inplace=True)
    df_lines.drop('w3_backup', axis=1, inplace=True)
    df_lines.drop('w4_backup', axis=1, inplace=True)
    df_lines.drop('shape', axis=1, inplace=True)
    df_lines.drop('profile', axis=1, inplace=True)

    return df_lines


if __name__ == "__main__":

    PARENT_DATABASE = current_file_folder / 'lines_database_v2.0.0.txt'
    CHILD_DATABASE = current_file_folder / 'lines_database_v2.0.0.txt'

    parent_db = lime.load_frame(PARENT_DATABASE)
    parent_db.drop('shape', axis=1, inplace=True)
    parent_db.drop('profile', axis=1, inplace=True)

    lime.save_frame(CHILD_DATABASE, parent_db)




