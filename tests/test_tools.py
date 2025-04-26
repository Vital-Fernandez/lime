import numpy as np
import pandas as pd
import lime
from pathlib import Path
from lime.tools import int_to_roman, join_fits_files, au
from lime.io import _LOG_EXPORT_DICT, hdu_to_log_df
from lime.transitions import air_to_vacuum_function
from astropy.io import fits
import astropy.units as u
from lime.fitting.lines import c_KMpS

# Data for the tests
baseline_folder = Path(__file__).parent / 'baseline'
outputs_folder = Path(__file__).parent / 'outputs'
spectra_folder = Path(__file__).parent.parent/'examples/sample_data/spectra'

file_address = baseline_folder/'manga_spaxel.txt'
lines_log_address = baseline_folder/'manga_lines_log.txt'
conf_file_address = baseline_folder/'manga.toml'

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

    lime.normalize_fluxes(log4, line_list=['O3_5007A/H1_4861A', 'N2_6583A/H1_6563A'])
    O3_norm = lines_log.loc['O3_5007A', 'profile_flux'] / lines_log.loc['H1_4861A', 'profile_flux']
    N2_norm = lines_log.loc['N2_6583A', 'profile_flux'] / lines_log.loc['H1_6563A', 'profile_flux']

    assert np.all(log4.xs('O3_5007A', level='line')['line_flux'] == O3_norm)
    assert np.all(log4.xs('N2_6583A', level='line')['line_flux'] == N2_norm)

    assert np.all(log4.xs('O3_5007A', level='line')['norm_line'] == 'H1_4861A')
    assert np.all(log4.xs('N2_6583A', level='line')['norm_line'] == 'H1_6563A')

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

    input_wavelength = 5000
    input_unit = input_wavelength * u.Angstrom
    astropy_conversion = input_unit.to(u.nm)
    lime_conversion = lime.unit_conversion('Angstrom', 'nm', wave_array=input_wavelength)
    assert np.allclose(astropy_conversion.value, lime_conversion, equal_nan=True)
    assert np.allclose(lime_conversion, 500)

    input_flux = 1e-16
    input_unit = input_flux * u.erg / (u.s * u.cm**2 * u.Angstrom)
    astropy_conversion = input_unit.to(u.Jy, equivalencies=u.spectral_density(input_wavelength*u.Angstrom))
    lime_conversion = lime.unit_conversion('FLAM', 'Jy', flux_array=input_flux, wave_array=input_wavelength,
                                           dispersion_units='Angstrom')
    assert np.allclose(astropy_conversion.value, lime_conversion, equal_nan=True)
    assert np.allclose(lime_conversion, 8.339102379953801e-05)

    input_wavelength = 1
    input_unit = input_wavelength * u.keV
    astropy_conversion = input_unit.to(u.Angstrom, equivalencies=u.spectral())
    lime_conversion = lime.unit_conversion('keV', 'Angstrom', wave_array=input_wavelength)
    assert np.allclose(astropy_conversion.value, lime_conversion, equal_nan=True)
    assert np.allclose(lime_conversion, 12.3984)

    input_wavelength = 6e14
    input_unit = input_wavelength * u.Hz
    astropy_conversion = input_unit.to(u.Angstrom, equivalencies=u.spectral())
    lime_conversion = lime.unit_conversion('Hz', 'Angstrom', wave_array=input_wavelength)
    assert np.allclose(astropy_conversion.value, lime_conversion, equal_nan=True)
    assert np.allclose(lime_conversion, 4996.540)

    return


def test_cube_spectrum_unit_conversion(file_name='jw01039-o003_t001_miri_ch4-medium_s3d.fits'):

    NGC6552 = lime.Cube.from_file(spectra_folder / file_name, instrument='miri', redshift=0.0266)

    spaxel_A = NGC6552.get_spectrum(19, 20)
    spaxel_A.unit_conversion(wave_units_out='Angstrom', flux_units_out='FLAM')

    NGC6552.unit_conversion(wave_units_out='Angstrom', flux_units_out='FLAM')
    spaxel_B = NGC6552.get_spectrum(19, 20)

    assert np.allclose(spaxel_B.wave, spaxel_A.wave)
    assert np.allclose(spaxel_B.wave_rest, spaxel_A.wave_rest)
    assert np.allclose(spaxel_B.flux, spaxel_A.flux)
    assert np.allclose(spaxel_B.err_flux, spaxel_A.err_flux)

    return


def test_spectra_unit_conversion():

    # Astropy conversion
    wave_astropy_in =  wave_array * au.Unit('Angstrom')
    flux_astropy_in = flux_array * (au.erg / (au.s * au.cm**2 * au.Angstrom))

    wave_astropy_out = wave_astropy_in.to(au.nm)
    flux_astropy_out = flux_astropy_in.to(au.Jy, au.spectral_density(wave_array * au.Unit('Angstrom')))

    # Own conversion
    spec_lime = lime.Spectrum(wave_array, flux_array, err_array, redshift=redshift, norm_flux=None, pixel_mask=pixel_mask)
    spec_lime.unit_conversion(wave_units_out='nm', flux_units_out='Jy', norm_flux=None)

    # Test conversion
    assert np.allclose(spec_lime.wave.data, wave_astropy_out.value, equal_nan=True)
    assert np.allclose(spec_lime.flux.data*spec_lime.norm_flux, flux_astropy_out.value, equal_nan=True)
    assert spec_lime.norm_flux == 1e-7

    # Test reconversion
    wave_astropy_out = wave_astropy_out.to(au.Angstrom)
    flux_astropy_out = flux_astropy_in.to(au.erg / (au.s * au.cm**2 * au.Angstrom), au.spectral_density(wave_array * au.Unit('nm')))
    spec_lime.unit_conversion(wave_units_out='AA', flux_units_out='FLAM', norm_flux=None)

    assert np.allclose(spec_lime.wave.data, wave_astropy_out.value, equal_nan=True)
    assert np.allclose(spec_lime.flux.data*spec_lime.norm_flux, flux_astropy_out.value, equal_nan=True)
    assert spec_lime.norm_flux == 1e-19

    # Compare with original
    assert np.allclose(spec_lime.wave.data, wave_array, equal_nan=True)
    assert np.allclose(spec_lime.flux.data*spec_lime.norm_flux, flux_array, equal_nan=True)

    return


def test_refraction_index_air_vacuum():

    from specutils.utils.wcs_utils import vac_to_air, refraction_index
    from astropy.units import Unit

    input_vac_arr = np.array([3621.59598486, 3622.42998417, 3623.26417553])
    specutils_air_arr = np.array([3620.50684567, 3621.34061749, 3622.17458132])

    assert np.allclose(air_to_vacuum_function(input_vac_arr), specutils_air_arr)

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


def test_line_bands_generation():

    # Creating the bands
    n_sigma = 3
    band_vsigma = 120
    ref_bands = lime.line_bands()
    bands_untouched = spec.retrieve.line_bands(adjust_central_bands=False)
    bands = spec.retrieve.line_bands(n_sigma=n_sigma, band_vsigma = band_vsigma, adjust_central_bands=True,
                                               instrumental_correction=False)
    # Existing lines
    assert bands.iloc[0].name == 'H1_3704A'
    assert bands.iloc[-1].name == 'H1_9546A'

    # Wavelength intervals
    lambda_arr = bands.wavelength.to_numpy()
    assert np.all((spec.wave_rest.data[0] < lambda_arr) & (spec.wave_rest.data[-1] > lambda_arr))

    # Central bands are modified
    assert bands.loc['H1_3704A', 'w3'] != ref_bands.loc['H1_3704A', 'w3']
    assert bands.loc['H1_9546A', 'w4'] != ref_bands.loc['H1_9546A', 'w4']
    assert bands_untouched.loc['H1_3704A', 'w3'] == ref_bands.loc['H1_3704A', 'w3']
    assert bands_untouched.loc['H1_9546A', 'w4'] == ref_bands.loc['H1_9546A', 'w4']

    # The bands have input kinematic width
    interval_wavelength = (bands.loc['H1_3704A', 'w4'] - bands.loc['H1_3704A', 'w3'])/2
    interval_speed = c_KMpS * (interval_wavelength/n_sigma) / (bands.loc['H1_3704A', 'wavelength'])
    assert np.isclose(band_vsigma, interval_speed)

    interval_wavelength = (bands.loc['H1_9546A', 'w4'] - bands.loc['H1_9546A', 'w3'])/2
    interval_speed = c_KMpS * (interval_wavelength/n_sigma) / (bands.loc['H1_9546A', 'wavelength'])
    assert np.isclose(band_vsigma, interval_speed)

    return


def test_line_bands_merged_blended():

    # Default only
    bands = spec.retrieve.line_bands(fit_cfg=conf_file_address)
    assert np.sum(bands.index.isin(['O2_3726A_m', 'H1_3889A_m', 'Ar4_4711A_m'])) == 3

    # From certain list
    bands = spec.retrieve.line_bands(fit_cfg=conf_file_address, composite_lines=['O2_3726A_m', 'H1_3889A_m', 'H1_6500A_b'])
    assert np.sum(bands.index.isin(['O2_3726A_m', 'H1_3889A_m', 'Ar4_4711A_m'])) == 2
    assert 'H1_6500A_b' not in bands.index

    # Combination of sections
    bands = spec.retrieve.line_bands(fit_cfg=conf_file_address, obj_cfg_prefix='38-35', update_default=False)
    in_list = ['O2_3726A_b', 'H1_4861A_b', 'O3_4959A_b', 'O3_5007A_b', 'O2_7319A_b', 'S3_9530A_b']
    out_list = bands.loc[bands.index.isin(in_list)].index.to_numpy()
    assert set(in_list) == set(in_list)


    return


def test_line_bands_labels():

    lines_test = ['H1_1216A', 'He2_1640A', 'C3_1909A', 'H1_4861A', 'H1_6563A', 'S3_9530A', 'He1_10832A']

    bands = lime.line_bands(line_list=lines_test)
    assert bands.loc['H1_1216A', 'wavelength'] == 1215.67
    assert bands.loc['H1_1216A', 'wavelength'] == bands.loc['H1_1216A', 'wave_vac']
    assert bands.loc['H1_4861A', 'wavelength'] == 4861.25
    assert bands.loc['H1_4861A', 'wavelength'] != bands.loc['H1_4861A', 'wave_vac']

    bands = lime.line_bands(line_list=lines_test, vacuum_waves=True)
    assert bands.loc['H1_1216A', 'wavelength'] == 1215.67
    assert bands.loc['H1_1216A', 'wavelength'] == bands.loc['H1_1216A', 'wave_vac']
    assert bands.loc['H1_4861A', 'wavelength'] != 4861.25
    assert bands.loc['H1_4861A', 'wavelength'] == 4862.683
    assert bands.loc['H1_4861A', 'wavelength'] == bands.loc['H1_4861A', 'wave_vac']

    bands = lime.line_bands(line_list=lines_test, units_wave='um', decimals=4, update_labels=True)
    assert np.isclose(bands.loc['H1_0.1216um', 'wavelength'], 0.121567)
    assert np.isclose(bands.loc['H1_0.1216um', 'wavelength'], bands.loc['H1_0.1216um', 'wave_vac'])
    assert np.isclose(bands.loc['H1_0.4861um', 'wavelength'], 0.486125)
    assert not np.isclose(bands.loc['H1_0.4861um', 'wavelength'], bands.loc['H1_0.4861um', 'wave_vac'])

    bands = lime.line_bands(line_list=lines_test, units_wave='um', decimals=4, update_labels=False)
    assert np.isclose(bands.loc['H1_1216A', 'wavelength'], 0.121567)
    assert np.isclose(bands.loc['H1_1216A', 'wavelength'], bands.loc['H1_1216A', 'wave_vac'])
    assert np.isclose(bands.loc['H1_4861A', 'wavelength'], 0.486125)
    assert not np.isclose(bands.loc['H1_4861A', 'wavelength'], bands.loc['H1_4861A', 'wave_vac'])

    bands = lime.line_bands(line_list=lines_test, vacuum_waves=True, vacuum_label=True)
    assert np.isclose(bands.loc['H1_1216A', 'wavelength'], 1215.670)
    assert np.isclose(bands.loc['H1_1216A', 'wavelength'], bands.loc['H1_1216A', 'wave_vac'])
    assert np.isclose(bands.loc['H1_4863A', 'wavelength'], 4862.683)
    assert np.isclose(bands.loc['H1_4863A', 'wavelength'], bands.loc['H1_4863A', 'wave_vac'])

    return