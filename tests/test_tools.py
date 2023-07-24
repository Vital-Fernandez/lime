import numpy as np
import pandas as pd
import lime
from pathlib import Path

lines_log_address = Path(__file__).parent/'data_tests'/'manga_lines_log.txt'

# Data for the tests
lines_log = lime.load_log(lines_log_address)

# Sample with 3 (repeated) observations
log_dict = {}
for i in range(3):
    log_dict[f'obj_{i}'] = lines_log.copy()

obs = lime.Sample()
obs.add_log_list(list(log_dict.keys()), list(log_dict.values()))


def test_extract_fluxes():

    log1 = lines_log.copy()
    log2 = lines_log.copy()
    log3 = lines_log.copy()
    log4 = obs.log.copy()

    lime.extract_fluxes(log1)
    lime.extract_fluxes(log2, flux_type='intg', column_name='integrated', column_positions=3)
    lime.extract_fluxes(log3, flux_type='gauss', column_name='flux')
    lime.extract_fluxes(log4)

    # Mixture
    assert log1.loc['O3_5007A', 'line_flux'] == log1.loc['O3_5007A', 'gauss_flux']
    assert log1.loc['O3_4363A', 'line_flux'] == log1.loc['O3_4363A', 'intg_flux']
    assert log1.loc['O3_5007A', 'line_flux_err'] == log1.loc['O3_5007A', 'gauss_flux_err']
    assert log1.loc['O3_4363A', 'line_flux_err'] == log1.loc['O3_4363A', 'intg_flux_err']

    # Integrated
    assert log2.loc['O3_5007A', 'integrated'] == log2.loc['O3_5007A', 'intg_flux']
    assert log2.loc['O3_4363A', 'integrated'] == log2.loc['O3_4363A', 'intg_flux']
    assert log2.loc['O3_5007A', 'integrated_err'] == log2.loc['O3_5007A', 'intg_flux_err']
    assert log2.loc['O3_4363A', 'integrated_err'] == log2.loc['O3_4363A', 'intg_flux_err']

    # Gaussian
    assert log3.loc['O3_5007A', 'flux'] == log3.loc['O3_5007A', 'gauss_flux']
    assert log3.loc['O3_4363A', 'flux'] == log3.loc['O3_4363A', 'gauss_flux']
    assert log3.loc['O3_5007A', 'flux_err'] == log3.loc['O3_5007A', 'gauss_flux_err']
    assert log3.loc['O3_4363A', 'flux_err'] == log3.loc['O3_4363A', 'gauss_flux_err']

    # Multi - index
    assert np.all(log4.xs('O3_5007A', level='line')['line_flux'] == log4.xs('O3_5007A', level='line')['gauss_flux'])
    assert np.all(log4.xs('O3_4363A', level='line')['line_flux'] == log4.xs('O3_4363A', level='line')['intg_flux'])
    assert np.all(log4.xs('O3_5007A', level='line')['line_flux_err'] == log4.xs('O3_5007A', level='line')['gauss_flux_err'])
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
    assert log1.loc['H1_6563A', 'line_flux'] == log1.loc['H1_6563A', 'gauss_flux']/log1.loc['H1_4861A', 'gauss_flux']
    assert log1.loc['O3_5007A', 'line_flux'] == log1.loc['O3_5007A', 'gauss_flux']/log1.loc['H1_4861A', 'gauss_flux']
    assert log1.loc['H1_4861A', 'line_flux'] == 1

    # List of normalizations
    norm_list = np.array(['H1_4861A'] * log2.index.size).astype(object)
    idcs_wide = log2.index.str.contains('k-1')
    norm_list[idcs_wide] = 'H1_4861A_k-1'
    lime.normalize_fluxes(log2, norm_list=norm_list)

    f_series = log2['gauss_flux']
    assert log2.loc['H1_6563A', 'line_flux'] == f_series['H1_6563A']/f_series['H1_4861A']
    assert log2.loc['O3_5007A', 'line_flux'] == f_series['O3_5007A']/f_series['H1_4861A']
    assert log2.loc['O3_5007A_k-1', 'line_flux'] == f_series['O3_5007A_k-1']/f_series['H1_4861A_k-1']
    assert log2.loc['O3_4959A_k-1', 'line_flux'] == f_series['O3_4959A_k-1']/f_series['H1_4861A_k-1']
    assert log2.loc['H1_4861A', 'line_flux'] == 1
    assert log2.loc['H1_4861A_k-1', 'line_flux'] == 1

    # Compound of normalizations
    line_list = ['H1_6563A/H1_4861A', 'O3_5007A/H1_4861A', 'O3_5007A_k-1/H1_4861A_k-1',
                 'O3_4959A_k-1/H1_4861A_k-1', 'H1_4861A_k-1/H1_4861A']
    lime.normalize_fluxes(log3, lines_list=line_list)

    f_series = log3['gauss_flux']
    assert log3.loc['H1_6563A', 'line_flux'] == f_series['H1_6563A']/f_series['H1_4861A']
    assert log3.loc['O3_5007A', 'line_flux'] == f_series['O3_5007A']/f_series['H1_4861A']
    assert log3.loc['O3_5007A_k-1', 'line_flux'] == f_series['O3_5007A_k-1']/f_series['H1_4861A_k-1']
    assert log3.loc['O3_4959A_k-1', 'line_flux'] == f_series['O3_4959A_k-1']/f_series['H1_4861A_k-1']
    assert np.isnan(log3.loc['H1_4861A', 'line_flux'])
    assert log3.loc['H1_4861A_k-1', 'line_flux'] == f_series['H1_4861A_k-1']/f_series['H1_4861A']
    assert np.isnan(log3.loc['O3_4363A', 'line_flux'])

    return


def test_extract_fluxes_multi_index():

    idcs_remove = (obs.log.index.get_level_values('id') == 'obj_1') &\
                  (~obs.log.index.get_level_values('line').isin(['H1_6563A', 'O3_4363A', 'H1_4861A']))

    log1 = obs.log.loc[~idcs_remove].copy()
    log2 = obs.log.loc[~idcs_remove].copy()
    log3 = obs.log.loc[~idcs_remove].copy()

    # One line normalization
    lime.normalize_fluxes(log1, norm_list='H1_4861A')
    ratio = log1.xs('O3_5007A', level='line')['gauss_flux'] / log1.xs('H1_4861A', level='line')['gauss_flux']
    assert log1.xs('O3_5007A', level='line')['line_flux']['obj_0'] == ratio['obj_0']
    assert log1.xs('O3_5007A', level='line')['line_flux']['obj_2'] == ratio['obj_2']
    assert np.isnan(ratio['obj_1'])
    assert 'obj_1' not in log1.xs('O3_5007A', level='line')['line_flux']

    # List of normalizations
    norm_list = np.array(['H1_4861A'] * log2.index.size).astype(object)
    idcs_wide = log2.index.get_level_values('line').str.contains('k-1')
    norm_list[idcs_wide] = 'H1_4861A_k-1'
    lime.normalize_fluxes(log2, norm_list=norm_list)

    ratio = log2.xs('O3_5007A', level='line')['gauss_flux'] / log2.xs('H1_4861A', level='line')['gauss_flux']
    assert log2.xs('O3_5007A', level='line')['line_flux']['obj_0'] == ratio['obj_0']
    assert log2.xs('O3_5007A', level='line')['line_flux']['obj_2'] == ratio['obj_2']
    assert np.isnan(ratio['obj_1'])
    assert 'obj_1' not in log2.xs('O3_5007A', level='line')['line_flux']

    return



# def test_relative_fluxes

# class TestSampleClass:
#
#     def test_multindex_log(self):
#
#         # Default import
#         assert isinstance(obs.log.index, pd.MultiIndex)
#         assert obs.log.index.names == ['id', 'line']
#         assert np.all(obs.log.index.get_level_values('id').unique() == ['obj_0', 'obj_1', 'obj_2'])
#
#         return
#
#     def test_extract_fluxes(self):
#
#         lime.extract_fluxes(obs.log, )
#
#         return