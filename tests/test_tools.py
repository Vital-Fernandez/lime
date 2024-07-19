import numpy as np
import pandas as pd
import lime
from pathlib import Path
from lime.tools import int_to_roman, refraction_index_air_vacuum, join_fits_files
from lime.io import _LOG_EXPORT_DICT, hdu_to_log_df
from astropy.io import fits

# Data for the tests
baseline_folder = Path(__file__).parent / 'baseline'
outputs_folder = Path(__file__).parent / 'outputs'

file_address = baseline_folder/'manga_spaxel.txt'
lines_log_address = baseline_folder/'manga_lines_log.txt'
lines_log = lime.load_frame(lines_log_address)
redshift = 0.0475
norm_flux = 1e-17
wave_array, flux_array, err_array = np.loadtxt(file_address, unpack=True)
pixel_mask = np.isnan(err_array)

spec = lime.Spectrum(wave_array, flux_array, err_array, redshift=redshift, norm_flux=norm_flux,
                     pixel_mask=pixel_mask)

# Sample with 3 (repeated) observations
log_dict = {}
for i in range(3):
    log_dict[f'obj_{i}'] = lines_log.copy()

obj_list = [f'obj_{i}' for i in range(3)]
log_list = [lines_log_address] * 3
obs = lime.Sample.from_file(obj_list, log_list, instrument='isis')


def test_int_to_roman():

    assert int_to_roman(1) == 'I'
    assert int_to_roman(10) == 'X'
    assert int_to_roman(1542) == 'MDXLII'

    return


def test_extract_fluxes():

    log1 = lines_log.copy()
    log2 = lines_log.copy()
    log3 = lines_log.copy()
    log4 = obs.frame.copy()

    lime.extract_fluxes(log1)
    lime.extract_fluxes(log2, flux_type='intg', column_name='integrated', column_positions=3)
    lime.extract_fluxes(log3, flux_type='profile', column_name='flux')
    lime.extract_fluxes(log4)

    # Mixture
    assert log1.loc['O3_5007A', 'line_flux'] == log1.loc['O3_5007A', 'profile_flux']
    assert log1.loc['O3_4363A', 'line_flux'] == log1.loc['O3_4363A', 'intg_flux']
    assert log1.loc['O3_5007A', 'line_flux_err'] == log1.loc['O3_5007A', 'profile_flux_err']
    assert log1.loc['O3_4363A', 'line_flux_err'] == log1.loc['O3_4363A', 'intg_flux_err']

    # Integrated
    assert log2.loc['O3_5007A', 'integrated'] == log2.loc['O3_5007A', 'intg_flux']
    assert log2.loc['O3_4363A', 'integrated'] == log2.loc['O3_4363A', 'intg_flux']
    assert log2.loc['O3_5007A', 'integrated_err'] == log2.loc['O3_5007A', 'intg_flux_err']
    assert log2.loc['O3_4363A', 'integrated_err'] == log2.loc['O3_4363A', 'intg_flux_err']

    # Gaussian
    assert log3.loc['O3_5007A', 'flux'] == log3.loc['O3_5007A', 'profile_flux']
    assert log3.loc['O3_4363A', 'flux'] == log3.loc['O3_4363A', 'profile_flux']
    assert log3.loc['O3_5007A', 'flux_err'] == log3.loc['O3_5007A', 'profile_flux_err']
    assert log3.loc['O3_4363A', 'flux_err'] == log3.loc['O3_4363A', 'profile_flux_err']

    # Multi - index
    assert np.all(log4.xs('O3_5007A', level='line')['line_flux'] == log4.xs('O3_5007A', level='line')['profile_flux'])
    assert np.all(log4.xs('O3_4363A', level='line')['line_flux'] == log4.xs('O3_4363A', level='line')['intg_flux'])
    assert np.all(log4.xs('O3_5007A', level='line')['line_flux_err'] == log4.xs('O3_5007A', level='line')['profile_flux_err'])
    assert np.all(log4.xs('O3_4363A', level='line')['line_flux_err'] == log4.xs('O3_4363A', level='line')['intg_flux_err'])

    # New columns position
    assert list(log1.columns).index('line_flux') == 0
    assert list(log1.columns).index('line_flux_err') == 1

    assert list(log2.columns).index('integrated') == 3
    assert list(log2.columns).index('integrated_err') == 4

    assert list(log3.columns).index('flux') == 0
    assert list(log3.columns).index('flux_err') == 1

    return


def test_extract_fluxes_single_index():

    log1 = lines_log.copy()
    log2 = lines_log.copy()
    log3 = lines_log.copy()

    # One line normalization
    lime.normalize_fluxes(log1, norm_list='H1_4861A')
    assert log1.loc['H1_6563A', 'line_flux'] == log1.loc['H1_6563A', 'profile_flux']/log1.loc['H1_4861A', 'profile_flux']
    assert log1.loc['O3_5007A', 'line_flux'] == log1.loc['O3_5007A', 'profile_flux']/log1.loc['H1_4861A', 'profile_flux']
    assert log1.loc['H1_4861A', 'line_flux'] == 1

    # List of normalizations
    norm_list = np.array(['H1_4861A'] * log2.index.size).astype(object)
    idcs_wide = log2.index.str.contains('k-1')
    norm_list[idcs_wide] = 'H1_4861A_k-1'
    lime.normalize_fluxes(log2, norm_list=norm_list)

    f_series = log2['profile_flux']
    assert log2.loc['H1_6563A', 'line_flux'] == f_series['H1_6563A']/f_series['H1_4861A']
    assert log2.loc['O3_5007A', 'line_flux'] == f_series['O3_5007A']/f_series['H1_4861A']
    assert log2.loc['O3_5007A_k-1', 'line_flux'] == f_series['O3_5007A_k-1']/f_series['H1_4861A_k-1']
    assert log2.loc['O3_4959A_k-1', 'line_flux'] == f_series['O3_4959A_k-1']/f_series['H1_4861A_k-1']
    assert log2.loc['H1_4861A', 'line_flux'] == 1
    assert log2.loc['H1_4861A_k-1', 'line_flux'] == 1

    # Compound of normalizations
    line_list = ['H1_6563A/H1_4861A', 'O3_5007A/H1_4861A', 'O3_5007A_k-1/H1_4861A_k-1',
                 'O3_4959A_k-1/H1_4861A_k-1', 'H1_4861A_k-1/H1_4861A']
    lime.normalize_fluxes(log3, line_list=line_list)

    f_series = log3['profile_flux']
    assert log3.loc['H1_6563A', 'line_flux'] == f_series['H1_6563A']/f_series['H1_4861A']
    assert log3.loc['O3_5007A', 'line_flux'] == f_series['O3_5007A']/f_series['H1_4861A']
    assert log3.loc['O3_5007A_k-1', 'line_flux'] == f_series['O3_5007A_k-1']/f_series['H1_4861A_k-1']
    assert log3.loc['O3_4959A_k-1', 'line_flux'] == f_series['O3_4959A_k-1']/f_series['H1_4861A_k-1']
    assert np.isnan(log3.loc['H1_4861A', 'line_flux'])
    assert log3.loc['H1_4861A_k-1', 'line_flux'] == f_series['H1_4861A_k-1']/f_series['H1_4861A']
    assert np.isnan(log3.loc['O3_4363A', 'line_flux'])

    return


def test_extract_fluxes_multi_index():

    idcs_remove = (obs.frame.index.get_level_values('id') == 'obj_1') & \
                  (~obs.frame.index.get_level_values('line').isin(['H1_6563A', 'O3_4363A', 'H1_4861A']))

    log1 = obs.frame.loc[~idcs_remove].copy()
    log2 = obs.frame.copy()
    log3 = obs.frame.copy()
    log4 = obs.frame.copy()

    # One line normalization
    lime.normalize_fluxes(log1, norm_list='H1_4861A')
    ratio = log1.xs('O3_5007A', level='line')['profile_flux'] / log1.xs('H1_4861A', level='line')['profile_flux']
    assert log1.xs('O3_5007A', level='line')['line_flux']['obj_0'] == ratio['obj_0']
    assert log1.xs('O3_5007A', level='line')['line_flux']['obj_2'] == ratio['obj_2']
    assert np.isnan(ratio['obj_1'])
    assert 'obj_1' not in log1.xs('O3_5007A', level='line')['line_flux']
    assert np.all(log1['norm_line'] == 'H1_4861A')

    # List of normalizations
    norm_list = np.array(['H1_4861A'] * log2.index.size).astype(object)
    idcs_wide = log2.index.get_level_values('line').str.contains('k-1')
    norm_list[idcs_wide] = 'H1_4861A_k-1'
    lime.normalize_fluxes(log2, norm_list=norm_list)

    ratio = log2.xs('O3_5007A', level='line')['profile_flux'] / log2.xs('H1_4861A', level='line')['profile_flux']
    assert log2.xs('O3_5007A', level='line')['line_flux']['obj_0'] == ratio['obj_0']
    assert log2.xs('O3_5007A', level='line')['line_flux']['obj_1'] == ratio['obj_1']
    assert log2.xs('O3_5007A', level='line')['line_flux']['obj_2'] == ratio['obj_2']

    assert np.all(log2.loc[~idcs_wide, 'norm_line'] == 'H1_4861A')
    assert np.all(log2.loc[idcs_wide, 'norm_line'] == 'H1_4861A_k-1')

    lime.normalize_fluxes(log3, line_list='H1_6563A', norm_list='H1_4861A')
    Halpha_norm = lines_log.loc['H1_6563A', 'profile_flux'] / lines_log.loc['H1_4861A', 'profile_flux']
    assert np.all(log3.xs('H1_6563A', level='line')['line_flux'] == Halpha_norm)
    assert pd.notnull(log3.line_flux).sum() == 3

    lime.normalize_fluxes(log4, line_list=['O3_5007A/H1_4861A', 'N2_6584A/H1_6563A'])
    O3_norm = lines_log.loc['O3_5007A', 'profile_flux'] / lines_log.loc['H1_4861A', 'profile_flux']
    N2_norm = lines_log.loc['N2_6584A', 'profile_flux'] / lines_log.loc['H1_6563A', 'profile_flux']

    assert np.all(log4.xs('O3_5007A', level='line')['line_flux'] == O3_norm)
    assert np.all(log4.xs('N2_6584A', level='line')['line_flux'] == N2_norm)

    assert np.all(log4.xs('O3_5007A', level='line')['norm_line'] == 'H1_4861A')
    assert np.all(log4.xs('N2_6584A', level='line')['norm_line'] == 'H1_6563A')

    return


def test_redshift_calculation():

    # Single index
    z_df = lime.redshift_calculation(lines_log)
    z_df_eqw = lime.redshift_calculation(lines_log, weight_parameter='eqw')
    z_df_flux_gauss = lime.redshift_calculation(lines_log, weight_parameter='profile_flux')
    z_df_strong = lime.redshift_calculation(lines_log, line_list=['O3_5007A', 'H1_6563A'])

    assert np.allclose(z_df['z_mean'][0], 0.047526, atol=0.00024, equal_nan=True)
    assert np.allclose(z_df_eqw['z_mean'][0], 0.047526, atol=0.00024, equal_nan=True)
    assert np.allclose(z_df_flux_gauss['z_mean'][0], 0.047526, atol=0.00024, equal_nan=True)
    assert np.allclose(z_df_strong['z_mean'][0], 0.047498, atol=0.000018, equal_nan=True)

    assert z_df['weight'][0] is None
    assert z_df_eqw['weight'][0] == 'eqw'
    assert z_df_flux_gauss['weight'][0] == 'profile_flux'
    assert z_df_strong['weight'][0] is None
    assert z_df_strong['lines'][0] == 'O3_5007A,H1_6563A'

    # Multi-index
    z_df = lime.redshift_calculation(obs.frame)
    z_df_eqw = lime.redshift_calculation(obs.frame, weight_parameter='eqw')
    z_df_flux_gauss = lime.redshift_calculation(obs.frame, weight_parameter='profile_flux')
    z_df_strong = lime.redshift_calculation(obs.frame, line_list=['O3_5007A', 'H1_6563A'])

    assert np.allclose(z_df['z_mean'][0], 0.047526, atol=0.00024, equal_nan=True)
    assert np.allclose(z_df_eqw['z_mean'][0], 0.047526, atol=0.00024, equal_nan=True)
    assert np.allclose(z_df_flux_gauss['z_mean'][0], 0.047526, atol=0.00024, equal_nan=True)
    assert np.allclose(z_df_strong['z_mean'][0], 0.047498, atol=0.000018, equal_nan=True)

    assert np.all(z_df['weight'].to_numpy() == None)
    assert np.all(z_df_eqw['weight'] == 'eqw')
    assert np.all(z_df_flux_gauss['weight'] == 'profile_flux')
    assert np.all(z_df_strong['weight'].to_numpy() == None)
    assert np.all(z_df_strong['lines'] == 'O3_5007A,H1_6563A')

    return


def test_unit_conversion():

    spec.unit_conversion(wave_units_out='nm', flux_units_out='Jy', norm_flux=1e-8)
    assert np.allclose(spec.wave.data, wave_array/10, equal_nan=True)
    assert np.allclose(spec.flux.data[:3], np.array([457.8036672 , 493.54866591, 493.17681153]), equal_nan=True)
    assert spec.norm_flux == 1e-8

    spec.unit_conversion(wave_units_out='AA', flux_units_out='FLAM', norm_flux=1)
    assert np.allclose(spec.wave.data, wave_array, equal_nan=True)
    assert np.allclose(spec.flux.data, flux_array, equal_nan=True)
    assert spec.norm_flux == 1

    return


def test_refraction_index_air_vacuum():

    array1 = refraction_index_air_vacuum(wave_array)
    assert np.allclose(array1[:3], np.array([1.00030083, 1.00030082, 1.00030081]))

    return


def test_logs_into_fits():

    # Load existing log
    log_orig = lime.load_frame(lines_log_address)

    # New text file
    lime.save_frame(outputs_folder / f'log_1.txt', log_orig)
    lime.save_frame(outputs_folder / f'log_2.fits', log_orig, page='LOG2')
    lime.save_frame(outputs_folder / f'log_2.fits', log_orig, page='LOG3')

    file_list = [outputs_folder/f'log_1.txt', outputs_folder/f'log_2.fits']
    output_file = outputs_folder/'joined_log.fits'

    join_fits_files(file_list, output_file, delete_after_join=True)

    # Check new and deteled files
    assert output_file.is_file()
    assert not file_list[0].is_file()
    assert not file_list[1].is_file()

    name_pages = ['LOG_1', 'LOG2', 'LOG3']

    for i, name in enumerate(name_pages):
        log_test = hdu_to_log_df(output_file, name)

        if log_test is not None:
            for line in log_test.index:
                for param in log_test.columns:

                    # String
                    if _LOG_EXPORT_DICT[param].startswith('<U'):
                        if log_orig.loc[line, param] is np.nan:
                            assert log_orig.loc[line, param] is log_test.loc[line, param]
                        else:
                            assert log_orig.loc[line, param] == log_test.loc[line, param]

                    # Float
                    else:
                        param_value = log_test.loc[line, param]
                        param_exp_value = log_orig.loc[line, param]

                        if ('_err' not in param) and (f'{param}_err' in log_orig.columns):
                            param_exp_err = log_orig.loc[line, f'{param}_err']
                            assert np.allclose(param_value, param_exp_value, atol=param_exp_err * 2, equal_nan=True)
                        else:
                            assert np.allclose(param_value, param_exp_value, rtol=0.10, equal_nan=True)


    return