import numpy as np
import lime
from pathlib import Path
import pytest
from matplotlib import pyplot as plt
from lime.io import _LOG_EXPORT_DICT
from os import remove
from copy import deepcopy
import pprint


def measurement_tolerance_test(input_spec, true_log, test_log, abs_factor=5, rel_tol=0.20):

    for line in input_spec.frame.index:
        for param in input_spec.frame.columns:

            # String
            if _LOG_EXPORT_DICT[param].startswith('<U'):
                if true_log.loc[line, param] is np.nan:
                    assert true_log.loc[line, param] is test_log.loc[line, param]
                else:
                    assert true_log.loc[line, param] == test_log.loc[line, param]

            # Float
            else:
                param_exp_value = true_log.loc[line, param]
                param_value = test_log.loc[line, param]

                if ('_err' not in param) and (f'{param}_err' in true_log.columns):
                    param_err = test_log.loc[line, f'{param}_err']
                    param_exp_err = true_log.loc[line, f'{param}_err']
                    diag = np.isclose(param_value, param_exp_value,
                                      rtol=np.maximum(0.01, np.abs(param_exp_err/param_exp_value)), equal_nan=True)
                    if not diag:
                        print(f'Error 1) {line} {param}: Measured {param_value}±{param_err} VS {param_exp_value}±{param_exp_err}')
                        print(line, param, param_value, param_exp_value, param_exp_err)
                    assert diag

                else:
                    if param.endswith('_err'):
                        diag = np.allclose(param_value, param_exp_value, rtol=1, equal_nan=True)
                        if not diag:
                            print(line, param, param_value, param_exp_value)
                        assert diag

                    else:
                        if param == 'FWZI':
                            diag = np.allclose(param_value, param_exp_value, rtol=rel_tol, equal_nan=True)
                            if not diag:
                                print(line, param, param_value, param_exp_value)
                            assert diag

                        else:
                            diag = np.allclose(param_value, param_exp_value, rtol=rel_tol, equal_nan=True)
                            if not diag:
                                print(line, param, param_value, param_exp_value)
                            assert diag

    return


def deep_equal(a, b):

    if type(a) != type(b):
        return False

    if isinstance(a, dict):
        if a.keys() != b.keys():
            return False
        return all(deep_equal(a[k], b[k]) for k in a)

    if isinstance(a, list):
        return len(a) == len(b) and all(deep_equal(x, y) for x, y in zip(a, b))

    return a == b


def read_file_contents(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read()


def compare_dictionaries(dict1, dict2):
    """
    Compare two TOML-like dictionaries section by section.
    Prints differences in keys/values for each section.
    """
    sections = set(dict1.keys()) | set(dict2.keys())

    for section in sorted(sections):
        s1 = dict1.get(section)
        s2 = dict2.get(section)

        if s1 is None:
            print(f"[{section}] ❌ missing in dict1")
            continue
        if s2 is None:
            print(f"[{section}] ❌ missing in dict2")
            continue

        # Both sections exist: compare their keys
        print(f"\n[Section: {section}]")
        keys = set(s1.keys()) | set(s2.keys())
        for key in sorted(keys):
            v1 = s1.get(key, "❌ missing")
            v2 = s2.get(key, "❌ missing")
            if v1 != v2:
                print('Dictionary 1')
                print(f"{pprint.pprint(v1)}")
                print()
                print('Dictionary 2')
                print(f"{pprint.pprint(v2)}")
                print()
                print()


# Data for the tests
baseline_folder = Path(__file__).parent / 'baseline'
file_address = baseline_folder/'SHOC579_MANGA38-35.txt'
conf_file_address = baseline_folder/'lime_tests.toml'
bands_file_address = baseline_folder/'SHOC579_MANGA38-35_bands.txt'
lines_log_address = baseline_folder/'SHOC579_MANGA38-35_log.txt'
lines_tex_address = baseline_folder/'SHOC579_MANGA38-35_log.tex'

data_folder = Path(__file__).parent.parent/'examples/doc_notebooks/0_resources'
outputs_folder = data_folder/'results'
spectra_folder = data_folder/'spectra'

redshift = 0.0475
norm_flux = 1e-17
cfg = lime.load_cfg(conf_file_address)
cfg_copy = deepcopy(cfg)
tolerance_rms = 5.5

wave_array, flux_array, err_array, pixel_mask = np.loadtxt(file_address, unpack=True)

spec = lime.Spectrum(wave_array, flux_array, err_array, redshift=redshift, norm_flux=norm_flux,
                     pixel_mask=pixel_mask, id_label='SHOC579_Manga38-35')

spec.fit.continuum(degree_list=[3, 6, 6], emis_threshold=[3, 2, 1.5])

# spec.fit.continuum(degree_list=[3, 6, 6], emis_threshold=[3, 2, 1.5])
# spec.plot.spectrum(show_cont=True, log_scale=True)

# spec.fit.frame(bands_file_address, cfg, obj_cfg_prefix='38-35', cont_from_bands=False)
spec.fit.frame(bands_file_address, cfg, obj_cfg_prefix='38-35')


class TestSpectrumClass:

    def test_read_spectrum(self):

        assert spec.norm_flux == norm_flux
        assert spec.redshift == redshift
        assert np.allclose(wave_array, spec.wave.data)
        assert np.allclose(wave_array / (1 + redshift), spec.wave_rest.data)
        assert np.allclose(flux_array, spec.flux.data * norm_flux, equal_nan=True)
        assert np.allclose(err_array, spec.err_flux.data * norm_flux, equal_nan=True)

        return

    def test_change_spectrum(self):

        # Cropping

        spec0 = lime.Spectrum(wave_array, flux_array, err_array, redshift=redshift, norm_flux=norm_flux,
                             pixel_mask=pixel_mask)
        spec1 = lime.Spectrum(wave_array, flux_array, err_array, redshift=redshift, norm_flux=norm_flux,
                             pixel_mask=pixel_mask, crop_waves=(5500, -1))
        spec2 = lime.Spectrum(wave_array, flux_array, err_array, redshift=redshift, norm_flux=norm_flux,
                             pixel_mask=pixel_mask, crop_waves=(0, 9000))
        spec3 = lime.Spectrum(wave_array, flux_array, err_array, redshift=redshift, norm_flux=norm_flux,
                             pixel_mask=pixel_mask, crop_waves=(5500, 9000))

        assert spec0.wave.data[0] == wave_array[0]
        assert spec0.wave.data[-1] == wave_array[-1]

        assert np.isclose(spec1.wave.data[0], 5500, atol=np.diff(spec1.wave.data).mean())
        assert spec1.wave.data[-1] == wave_array[-1]

        assert spec2.wave.data[0] == wave_array[0]
        assert np.isclose(spec2.wave.data[-1], 9000, atol=np.diff(spec2.wave.data).mean())

        assert np.isclose(spec3.wave.data[0], 5500, atol=np.diff(spec3.wave.data).mean())
        assert np.isclose(spec3.wave.data[-1], 9000, atol=np.diff(spec3.wave.data).mean())

        assert spec0.redshift == redshift

        # Updating redshift
        spec0.update_redshift(redshift=0.06)
        assert spec0.redshift == 0.06
        assert spec0.norm_flux == norm_flux
        assert np.allclose(wave_array, spec0.wave.data)
        assert np.allclose(wave_array / (1 + 0.06), spec0.wave_rest.data)
        assert np.allclose(flux_array, spec0.flux.data * norm_flux, equal_nan=True)
        assert np.allclose(err_array, spec0.err_flux.data * norm_flux, equal_nan=True)

        spec0.update_redshift(redshift=redshift)
        assert spec0.norm_flux == norm_flux
        assert np.allclose(wave_array, spec0.wave.data)
        assert np.allclose(wave_array / (1 + redshift), spec0.wave_rest.data)
        assert np.allclose(flux_array, spec0.flux.data * norm_flux, equal_nan=True)
        assert np.allclose(err_array, spec0.err_flux.data * norm_flux, equal_nan=True)

        return

    def test_cfg_preservation(self):

        compare_dictionaries(cfg_copy, cfg)

        assert cfg == cfg_copy
        assert deep_equal(cfg, cfg_copy)

        return

    @pytest.mark.mpl_image_compare(tolerance=tolerance_rms)
    def test_line_detection_plot(self):

        spec.fit.continuum(degree_list=[3, 6, 6], emis_threshold=[5, 3, 2])
        match_bands = spec.infer.peaks_troughs(bands_file_address)

        fig = plt.figure()
        spec.plot.spectrum(in_fig=fig, bands=match_bands)

        return fig

    @pytest.mark.mpl_image_compare(tolerance=tolerance_rms)
    def test_plot_spectrum(self):

        # fig = plt.figure()
        # spec.plot.spectrum(in_fig=fig)

        fig = plt.figure()

        spec = lime.Spectrum(wave_array, flux_array, err_array, redshift=redshift, norm_flux=norm_flux,
                             pixel_mask=pixel_mask, id_label='SHOC579_Manga38-35')

        spec.fit.frame(bands_file_address, cfg, obj_cfg_prefix='38-35', line_list=['H1_6563A_b'])
        spec.plot.spectrum(in_fig=fig)

        return fig

    @pytest.mark.mpl_image_compare(tolerance=tolerance_rms)
    def test_plot_spectrum_with_fits(self):

        fig = plt.figure()
        spec.plot.spectrum(in_fig=fig, show_profiles=True)

        return fig

    @pytest.mark.mpl_image_compare(tolerance=tolerance_rms)
    def test_check_bands_spectrum(self):

        fig = plt.figure()
        spec.check.bands(fname=bands_file_address, in_fig=fig)

        return fig

    @pytest.mark.mpl_image_compare(tolerance=tolerance_rms)
    def test_plot_spectrum_maximize(self):

        fig = plt.figure()
        spec.plot.spectrum(in_fig=fig, show_profiles=True, maximize=True)

        return fig

    @pytest.mark.mpl_image_compare(tolerance=tolerance_rms)
    def test_plot_spectrum_with_bands(self):

        fig = plt.figure()
        spec.plot.spectrum(in_fig=fig, bands=bands_file_address)

        return fig

    @pytest.mark.mpl_image_compare(tolerance=tolerance_rms)
    def test_plot_line(self):

        fig = plt.figure()
        spec.plot.bands('Fe3_4658A_s-emi',  rest_frame=True, in_fig=fig)

        return fig

    @pytest.mark.mpl_image_compare(tolerance=tolerance_rms)
    def test_plot_line_2nd(self):

        fig = plt.figure()
        spec.plot.bands('Fe3_4658A_s-emi', y_scale='log', in_fig=fig)

        return fig

    @pytest.mark.mpl_image_compare(tolerance=tolerance_rms)
    def test_plot_line_3rd(self):

        fig = plt.figure()
        spec.plot.bands('Fe3_4658A_s-emi', show_bands=True, in_fig=fig)

        return fig

    @pytest.mark.mpl_image_compare(tolerance=tolerance_rms)
    def test_plot_grid(self):

        fig = plt.figure()
        spec.plot.grid(in_fig=fig)

        return fig

    @pytest.mark.mpl_image_compare(tolerance=tolerance_rms)
    def test_plot_cinematics(self):

        fig = plt.figure()
        spec.plot.velocity_profile('O3_5007A', in_fig=fig)

        return fig

    def test_measurements_txt_file(self):

        extension = 'txt'
        spec.save_frame(outputs_folder / f'test_lines_log.{extension}')

        log_orig = lime.load_frame(lines_log_address)
        log_test = lime.load_frame(outputs_folder / f'test_lines_log.{extension}')

        measurement_tolerance_test(spec, log_orig, log_test)

        return

    def test_measurements_fits_file(self):

        extension = 'fits'
        spec.save_frame(outputs_folder / f'test_lines_log.{extension}')

        log_orig = lime.load_frame(lines_log_address)
        log_test = lime.load_frame(outputs_folder / f'test_lines_log.{extension}')

        measurement_tolerance_test(spec, log_orig, log_test)

        return

    def test_measurements_csv_file(self):

        extension = 'csv'
        spec.save_frame(outputs_folder / f'test_lines_log.{extension}')

        log_orig = lime.load_frame(lines_log_address)
        log_test = lime.load_frame(outputs_folder / f'test_lines_log.{extension}')

        measurement_tolerance_test(spec, log_orig, log_test)

        return

    def test_measurements_xlsx_file(self):

        extension = 'xlsx'
        spec.save_frame(outputs_folder / f'test_lines_log.{extension}')

        log_orig = lime.load_frame(lines_log_address)
        log_test = lime.load_frame(outputs_folder / f'test_lines_log.{extension}')

        measurement_tolerance_test(spec, log_orig, log_test)

        return

    def test_extra_pages_xlsx(self):

        file_xlsx = outputs_folder / 'test_lines_log_multi_page.xlsx'

        if file_xlsx.is_file():
            try:
                remove(file_xlsx)
                print(f"File '{file_xlsx}' has been deleted successfully.")
            except OSError as e:
                print(f"Error: {e.strerror}")

        log_orig = lime.load_frame(lines_log_address)

        for page in ['FRAME', 'FRAME2', 'FRAME']:
            spec.save_frame(outputs_folder / file_xlsx, page=page)
            log_test = lime.load_frame(file_xlsx)

            measurement_tolerance_test(spec, log_orig, log_test)

        return

    def test_measurements_latex_file(self):

        # Create tex file
        extension = 'tex'
        log_new = Path(outputs_folder / f'test_lines_log.{extension}')
        spec.save_frame(log_new, param_list=['particle', 'wavelength', 'group_label', 'latex_label'], safe_version=False)

        with open(log_new, 'r') as f1, open(lines_tex_address, 'r') as f2:
            for line1, line2 in zip(f1, f2):
                assert line1 == line2, f"Line mismatch:\n{line1}\n!=\n{line2}"

        return

    # def test_measurements_asdf_file(self):
    #
    #     extension = 'asdf'
    #     spec.save_log(outputs_folder/f'test_lines_log.{extension}')
    #
    #     log_orig = lime.load_log(lines_log_address)
    #     log_test = lime.load_log(outputs_folder/f'test_lines_log.{extension}')
    #
    #     for line in spec.log.index:
    #         for param in spec.log.columns:
    #
    #             # String
    #             if _LOG_EXPORT_DICT[param].startswith('<U'):
    #                 if log_orig.loc[line, param] is np.nan:
    #                     assert log_orig.loc[line, param] is log_test.loc[line, param]
    #                 else:
    #                     assert log_orig.loc[line, param] == log_test.loc[line, param]
    #
    #             # Float
    #             else:
    #                 if param not in ['eqw', 'eqw_err']:
    #                     print(param, log_orig.loc[line, param], log_test.loc[line, param])
    #                     assert np.allclose(log_orig.loc[line, param], log_test.loc[line, param], rtol=0.05,
    #                                           equal_nan=True)
    #                 else:
    #                     assert np.allclose(log_orig.loc[line, param], log_test.loc[line, param], rtol=0.15,
    #                                           equal_nan=True)
    #
    #     return

    def test_save_load_log(self):

        spec0 = lime.Spectrum(wave_array, flux_array, err_array, redshift=redshift, norm_flux=norm_flux,
                             pixel_mask=pixel_mask)

        spec0.load_frame(lines_log_address)

        new_log_spectrum = outputs_folder / 'manga_lines_log_from_spectrum.txt'
        spec0.save_frame(new_log_spectrum)

        assert new_log_spectrum.is_file()

        return

    def test_line_detection_implicit_explicit_params(self, file_name='sdss_dr18_0358-51818-0504.fits'):

        SHOC579_a = lime.Spectrum.from_file(spectra_folder/file_name, instrument='sdss')

        assert np.isclose(SHOC579_a.redshift, 0.047232304)

        # Measure lines explicit
        cfg_file, bands_file = baseline_folder/'sample_cfg.toml', baseline_folder/'SHOC579_bands.txt'
        sample_cfg, shoc549_df = lime.load_cfg(cfg_file),  lime.load_frame(bands_file)
        SHOC579_a.fit.frame(shoc549_df, sample_cfg, obj_cfg_prefix='SHOC579', line_detection=True)
        df_a = SHOC579_a.frame.copy()

        # Clear measurements
        SHOC579_a.frame = SHOC579_a.frame[0:0]

        # Measure lines implicit
        SHOC579_a.fit.frame(bands_file, cfg_file, obj_cfg_prefix='SHOC579', line_detection=True)
        df_b = SHOC579_a.frame.copy()

        assert np.all(df_a.index == df_b.index)
        assert np.all(np.isclose(df_a.intg_flux, df_b.intg_flux, rtol=2*df_b.intg_flux_err))
        assert np.all(np.isclose(df_a.profile_flux, df_b.profile_flux, rtol=2*df_b.profile_flux_err))

        return


class TestFluxMeasurements:

    def test_intgr_flux_comparison(self):

        line_arr = spec.frame.index.to_numpy()
        group_type = spec.frame['group_label'].to_numpy()
        intg, intg_err = (spec.frame.loc[:, ['intg_flux', 'intg_flux_err']].to_numpy()/spec.norm_flux).T
        gauss, gauss_err = (spec.frame[['profile_flux', 'profile_flux_err']].to_numpy()/spec.norm_flux).T
        sigma_quad = np.sqrt(np.square(intg_err) + np.square(gauss_err))

        diag_arr = np.isclose(intg, gauss, atol=1 * sigma_quad)

        exclude_list = ['N2_5755A', 'H1_8665A', 'H1_8750A']
        for i, line in enumerate(line_arr):
            if group_type[i] == 'none' and not diag_arr[i]:
                if line not in exclude_list:
                    if diag_arr[i] == False:
                        print(line, f"Intg = {intg[i]:0.2f}±{intg_err[i]:0.2f},",
                              f"Gauss = {gauss[i]:0.2f}±{gauss_err[i]:0.2f}, diag1 = {diag_arr[i]}")
                    assert diag_arr[i] == False


        return

    def test_bands_cont(self):

        spec2 = lime.Spectrum(wave_array, flux_array, err_array, redshift=redshift, norm_flux=norm_flux,
                             pixel_mask=pixel_mask, id_label='SHOC579_Manga38-35')
        spec2.fit.frame(bands_file_address, cfg, obj_cfg_prefix='38-35', cont_source='adjacent')
        line_arr = ["O2_3726A","O2_3729A","He1_4026A","H1_4861A","H1_4861A_k-1","Fe3_4658A_s-emi","Fe3_4658A_s-abs",
                    "He2_4686A","H1_8545A","H1_8750A","Fe3_4925A_m","O1_6300A","S3_6312A","S2_6716A","S2_6731A","He1_7065A",
                    "Ar3_7751A"]

        gauss1, gauss_err1 = np.abs(spec.frame.loc[line_arr, ['profile_flux', 'profile_flux_err']].to_numpy()/spec.norm_flux).T
        gauss2, gauss_err2 = np.abs(spec2.frame.loc[line_arr, ['profile_flux', 'profile_flux_err']].to_numpy()/spec2.norm_flux).T
        sigma_quad = np.sqrt(np.square(gauss_err1) + np.square(gauss_err2))
        diag_arr = np.isclose(gauss1, gauss2, atol=1.5 * sigma_quad)

        for i, line in enumerate(line_arr):
            if diag_arr[i] == False:
                if diag_arr[i] == False:
                    print(line, f"Without C.= {gauss1[i]:0.2f}±{gauss_err1[i]:0.2f},",
                                f"With C = {gauss2[i]:0.2f}±{gauss_err2[i]:0.2f}, diag1 = {diag_arr[i]}")
                assert diag_arr[i] == False

        return

    def test_fit_cont(self):

        spec2 = lime.Spectrum(wave_array, flux_array, err_array, redshift=redshift, norm_flux=norm_flux,
                             pixel_mask=pixel_mask, id_label='SHOC579_Manga38-35')

        spec2.fit.continuum(degree_list=[3, 6, 6], emis_threshold=[3, 2, 1.5])

        line = lime.Line.from_transition('O1_6300A_b', cfg['default_line_fitting'], spec.frame,
                                         norm_flux=spec.norm_flux)

        spec2.fit.frame(bands_file_address, cfg, obj_cfg_prefix='38-35', cont_source='central', line_list=['O1_6300A_b'])
        # spec2.plot.bands(show_cont=True)
        assert np.isclose(line.measurements.cont, spec2.fit.line.measurements.cont, rtol=0.01)

        spec2.clear_data()
        spec2.fit.frame(bands_file_address, cfg, obj_cfg_prefix='38-35', cont_source='adjacent', line_list=['O1_6300A_b'])
        # spec2.plot.bands(show_cont=True)
        assert np.isclose(line.measurements.cont, spec2.fit.line.measurements.cont, rtol=0.05)

        spec2.clear_data()
        spec2.fit.frame(bands_file_address, cfg, obj_cfg_prefix='38-35', cont_source='fit', line_list=['O1_6300A_b'])
        # spec2.plot.bands(show_cont=True)
        assert np.isclose(line.measurements.cont, spec2.fit.line.measurements.cont, rtol=0.10)

        line = lime.Line.from_transition('O2_3726A_b', cfg['default_line_fitting'], spec.frame,
                                         norm_flux=spec.norm_flux)

        spec2.fit.frame(bands_file_address, cfg, obj_cfg_prefix='38-35', cont_source='central', line_list=['O2_3726A_b'])
        assert np.isclose(line.measurements.cont, spec2.fit.line.measurements.cont, rtol=0.01)

        spec2.clear_data()
        spec2.fit.frame(bands_file_address, cfg, obj_cfg_prefix='38-35', cont_source='adjacent', line_list=['O2_3726A_b'])
        assert np.isclose(line.measurements.cont, spec2.fit.line.measurements.cont, rtol=0.30)

        spec2.clear_data()
        spec2.fit.frame(bands_file_address, cfg, obj_cfg_prefix='38-35', cont_source='fit', line_list=['O2_3726A_b'])
        assert np.isclose(line.measurements.cont, spec2.fit.line.measurements.cont, rtol=0.30)

        return

    @pytest.mark.mpl_image_compare(tolerance=tolerance_rms)
    def test_cont_in_bands(self):

        spec.fit.frame(bands_file_address, cfg, obj_cfg_prefix='38-35', cont_source='central', line_list=['O2_3726A_b'])

        fig = plt.figure()
        spec.plot.bands(show_cont=True, in_fig=fig)

        return fig