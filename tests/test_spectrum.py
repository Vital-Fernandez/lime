import numpy as np
import pandas as pd
import lime
import pytest
from pathlib import Path
from astropy.io import fits
from matplotlib import pyplot as plt
from lime.io import _LOG_EXPORT_DICT
from scipy.optimize import curve_fit
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


def linear_model(x, m, n):
    return m * x + n

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

# Fitting with default central bands
spec = lime.Spectrum(wave_array, flux_array, err_array, redshift=redshift, norm_flux=norm_flux,
                     pixel_mask=pixel_mask, id_label='SHOC579_Manga38-35')
spec.fit.continuum(degree_list=[3, 6, 6], emis_threshold=[3, 2, 1.5])
spec.fit.frame(bands_file_address, cfg, obj_cfg_prefix='38-35')
# spec2.plot.spectrum(rest_frame=True)

# Re-fit the lines using the bands continua
spec2 = lime.Spectrum(wave_array, flux_array, err_array, redshift=redshift, norm_flux=norm_flux,
                      pixel_mask=pixel_mask, id_label='SHOC579_Manga38-35')
spec2.fit.frame(bands_file_address, cfg, obj_cfg_prefix='38-35', cont_source='adjacent')
# spec2.plot.spectrum(rest_frame=True)


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
                if line1.startswith(r'\footnotesize{LiMe:2.'): continue # Ignore the file version
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


class TestMeasurements:

    def test_flux_calculation_comparison(self):

        # Unpack measurements
        idcs_single = (spec.frame['group_label'] == 'none') | (spec.frame.index.str.endswith('_m'))
        line_arr = spec.frame.loc[idcs_single].index.to_numpy()
        intg, intg_err = (spec.frame.loc[idcs_single, ['intg_flux', 'intg_flux_err']].to_numpy()/spec.norm_flux).T
        gauss, gauss_err = (spec.frame.loc[idcs_single, ['profile_flux', 'profile_flux_err']].to_numpy()/spec.norm_flux).T

        # Diagnostic for matching methodology fluxes
        sigma_quad = np.sqrt(np.square(intg_err) + np.square(gauss_err))
        diag_arr = np.abs(intg - gauss) <= 2 * sigma_quad

        exclude_list = ['N2_5755A', 'O2_7319A_m', 'O2_7330A_m']
        good_count = 0

        for i, line in enumerate(line_arr):
            if line not in exclude_list:
                if not diag_arr[i]:
                    print(line, f"Intg = {intg[i]:0.2f}±{intg_err[i]:0.2f},", f"Gauss = {gauss[i]:0.2f}±{gauss_err[i]:0.2f}, diag1 = {diag_arr[i]}")
                assert diag_arr[i]
                good_count += 1

        assert good_count == 43

        return

    def test_cont(self):

        target_lines = ['H1_4861A', 'O3_5007A', 'H1_6563A', 'He1_5876A', 'Ne3_3968A_m', 'O3_4363A', 'S3_6312A', 'He2_4686A']
        x, y, err = spec2.wave.data, spec2.flux.data, spec2.err_flux.data
        for line_label in target_lines:

            # LiMe measurements
            print(line_label)
            row_central = spec.frame.loc[line_label]
            row_bands = spec2.frame.loc[line_label]

            # Index the line bands
            w1, w2, w3, w4, w5, w6 = row_bands['w1':'w6'].to_numpy() * (1 + spec2.redshift)
            idcs_central = ((x >= w3) & (x <= w4))
            idcs_bands = ((x >= w1) & (x <= w2)) | ((x >= w5) & (x <= w6))
            idcs_line = ((x >= w1) & (x <= w6))

            # Two-point fit from linear fit
            x_central, y_central, err_central = x[idcs_central], y[idcs_central], err[idcs_central]
            x0, xN = x_central[0], x_central[-1]
            y0, yN = y_central[0], y_central[-1]
            e0, eN = (err_central[0], err_central[-1])
            dx = xN - x0

            m_tp = (yN - y0) / dx
            n_tp = y0 - m_tp * x0
            m_err_tp = np.sqrt((e0 / dx) ** 2 + (eN / dx) ** 2)
            n_err_tp = np.sqrt((e0 * xN / dx) ** 2 + (eN * x0 / dx) ** 2)

            # Bands continua using Scipy
            x_b = x[idcs_bands]
            y_b = y[idcs_bands]
            e_b = err[idcs_bands]

            sigma_central, initial_values = e_b, [m_tp, n_tp]
            popt, pcov = curve_fit(linear_model, x_b, y_b, p0=initial_values, sigma=sigma_central, absolute_sigma=True)
            perr = np.sqrt(np.diag(pcov))

            m_cf, n_cf = popt
            m_err_cf, n_err_cf = perr

            # Compare techniques measurements with LiMe
            m_cont_lime_central = row_central.m_cont / spec2.norm_flux
            m_cont_err_lime_central = row_central.m_cont_err / spec2.norm_flux
            n_cont_lime_central = row_central.n_cont / spec2.norm_flux
            n_cont_err_lime_central = row_central.n_cont_err / spec2.norm_flux

            m_cont_lime_bands = row_bands.m_cont / spec2.norm_flux
            m_cont_err_lime_bands = row_bands.m_cont_err / spec2.norm_flux
            n_cont_lime_bands = row_bands.n_cont / spec2.norm_flux
            n_cont_err_lime_bands = row_bands.n_cont_err / spec2.norm_flux

            assert np.isclose(m_cont_lime_central, m_tp, atol=0.05)
            assert np.isclose(m_cont_err_lime_central, m_err_tp, atol=0.05)
            assert np.isclose(n_cont_lime_central, n_tp, atol=0.05)
            assert np.isclose(n_cont_err_lime_central, n_err_tp, atol=0.05)

            assert np.isclose(m_cont_lime_bands, m_cf, atol=0.05)
            assert np.isclose(m_cont_err_lime_bands, m_err_cf, atol=0.05)
            assert np.isclose(n_cont_lime_bands, n_cf, atol=0.05)
            assert np.isclose(n_cont_err_lime_bands, n_err_cf, atol=0.05)

            # Compare techniques with each other
            sigma_m = np.sqrt(m_err_tp**2 + m_err_cf**2)
            sigma_n = np.sqrt(n_err_tp**2 + n_err_cf**2)

            m_match = np.abs(m_tp - m_cf) <= 2 * sigma_m
            n_match = np.abs(n_tp - n_cf) <= 2 * sigma_n
            assert m_match
            assert n_match

            # Compare continuum level calculation
            idx_peak = np.argmax(y[idcs_central])
            wave_peak = x[idcs_central][idx_peak]
            flux_peak = y[idcs_central][idx_peak]

            cont_central =  row_central.cont / spec2.norm_flux
            cont_central_err =  row_central.cont_err / spec2.norm_flux
            cont_tp = m_tp * wave_peak + n_tp
            cont_tp_err = np.sqrt(((xN - wave_peak) / dx * e0) ** 2 + ((wave_peak - x0) / dx * eN) ** 2)
            cont_tp_err_naive = np.sqrt((wave_peak * m_err_tp) ** 2 + n_err_tp ** 2)

            cont_bands = row_bands.cont / spec2.norm_flux
            cont_bands_err = row_bands.cont_err / spec2.norm_flux
            cont_cf = m_cf * wave_peak + n_cf
            cont_cf_err = np.sqrt(pcov[0, 0] * wave_peak ** 2 + pcov[1, 1] + 2 * pcov[0, 1] * wave_peak)
            cont_cf_err_nocov = np.sqrt(pcov[0, 0] * wave_peak**2 + pcov[1, 1])

            assert np.isclose(cont_central, cont_tp, atol=0.05)
            assert np.isclose(cont_central_err, cont_tp_err, atol=0.05)
            assert np.isclose(cont_bands, cont_cf, atol=0.05)
            assert np.isclose(cont_bands_err, cont_cf_err, atol=0.05)

            sigma_cont = np.sqrt(cont_tp_err**2 + cont_cf_err**2)
            # assert np.abs(cont_tp - cont_cf) <= 3 * sigma_cont

            # from matplotlib import pyplot as plt
            # fig, ax = plt.subplots()
            # ax.step(x[idcs_bands], y[idcs_bands])
            # ax.step(x[idcs_central], y[idcs_central])
            # # ax.plot(x[idcs_central], m_cont_lime_central * x[idcs_central] + n_cont_lime_central, label='LiMe central', linestyle=':')
            # # ax.plot(x[idcs_central], m_tp * x[idcs_central] + n_tp, label=' Test central', linestyle='--')
            # ax.plot(x[idcs_line], m_cont_lime_bands * x[idcs_line] + n_cont_lime_bands, label='LiMe bands', linestyle=':')
            # ax.plot(x[idcs_line], m_cf * x[idcs_line] + n_cf, label='Scipy bands', linestyle=':')
            # ax.scatter(wave_peak, flux_peak)
            # ax.legend()
            # ax.set_yscale('log')
            # plt.show()

        return

    def test_cont_on_flux_effect(self):

        # Line selection
        idcs_single = spec.frame.index
        line_arr = spec.frame.loc[idcs_single].index.to_numpy()

        # Unpack measurements
        intg1, intg_err1 = (spec.frame.loc[idcs_single, ['intg_flux', 'intg_flux_err']].to_numpy()/spec.norm_flux).T
        gauss1, gauss_err1 = (spec.frame.loc[idcs_single, ['profile_flux', 'profile_flux_err']].to_numpy()/spec.norm_flux).T

        intg2, intg_err2 = (spec2.frame.loc[idcs_single, ['intg_flux', 'intg_flux_err']].to_numpy()/spec.norm_flux).T
        gauss2, gauss_err2 = (spec2.frame.loc[idcs_single, ['profile_flux', 'profile_flux_err']].to_numpy()/spec.norm_flux).T

        # Diagnostic for matching methodology fluxes
        sigma_quad = np.sqrt(np.square(gauss_err1) + np.square(gauss_err2))
        diag_arr = np.abs(gauss1 - gauss2) <= 2 * sigma_quad

        exclude_list, good_count = ['Ar4_4711A_m', 'N2_5755A'], 0
        for i, line in enumerate(line_arr):
            if line not in exclude_list:
                if not diag_arr[i]:
                    print(line, f"First = {gauss1[i]:0.2f}±{gauss_err1[i]:0.2f},", f"Second = {gauss2[i]:0.2f}±{gauss_err2[i]:0.2f}")
                # assert diag_arr[i]
                good_count += 1

        assert good_count == 65

        return

    @pytest.mark.mpl_image_compare(tolerance=tolerance_rms)
    def test_cont_in_bands(self):

        spec.fit.frame(bands_file_address, cfg, obj_cfg_prefix='38-35', cont_source='central', line_list=['O2_3726A_b'])

        fig = plt.figure()
        spec.plot.bands(show_cont=True, in_fig=fig)

        return fig

    # def test_sloan_comparison(self):
    #
    #     # State the data files
    #     sdss_fits_fname = f'{spectra_folder}/SHOC579_SDSS_dr18.fits'
    #
    #     # Load configuration
    #     cfgFile = f'{data_folder}/long_slit.toml'
    #     obs_cfg = lime.load_cfg(cfgFile)
    #
    #     # Remove O3_4959A configuration
    #     obs_cfg['SHOC579_sdss_line_fitting'].pop('O3_4959A_b')
    #
    #     # Declare LiMe spectrum
    #     spec = lime.Spectrum.from_file(sdss_fits_fname, instrument='sdss')
    #
    #     # Measure the lines
    #     bands_df = spec.retrieve.lines_frame(fit_cfg=obs_cfg, obj_cfg_prefix='SHOC579_sdss', band_vsigma=120)
    #     spec.fit.frame(bands_df, fit_cfg=obs_cfg, obj_cfg_prefix='SHOC579_sdss', cont_source='adjacent',
    #                    line_list=['O3_4959A', 'Ar3_7136A', 'S2_6716A_b'])
    #     lime_df = spec.frame.copy()
    #
    #     # Unpack SDSS line measurements
    #     data_lines = fits.getdata(sdss_fits_fname, extname='SPZLINE')
    #     line_dict = {'Ne3_3968A': '[Ne_III] 3868',
    #                  'H1_3970A': 'H_epsilon',
    #                  'H1_4861A': 'H_beta',
    #                  'H1_4102A': 'H_delta',
    #                  'H1_4340A': 'H_gamma',
    #                  'O3_4363A': '[O_III] 4363',
    #                  'O2_3726A': '[O_II] 3725', 'O2_3729A': '[O_II] 3727',
    #                  'O3_4959A': '[O_III] 4959', 'O3_5007A': '[O_III] 5007',
    #                  'O1_6300A': '[O_I] 6300', 'S3_6312A': '[S_III] 6312',
    #                  'H1_6563A': 'H_alpha', 'N2_6548A': '[N_II] 6548', 'N2_6583A': '[N_II] 6583',
    #                  'S2_6716A': '[S_II] 6716', 'S2_6731A': '[S_II] 6730',
    #                  'Ar3_7136A': '[Ar_III] 7135'}
    #
    #     # Convert FITS_rec to DataFrame
    #     sdss_df = pd.DataFrame(data_lines.tolist(), columns=data_lines.names)
    #     sdss_df.set_index('LINENAME', inplace=True)
    #     sdss_df.index = sdss_df.index.str.strip()
    #
    #     # Rename columns using inverted dictionary
    #     inv_dict = {v: k for k, v in line_dict.items()}
    #     sdss_df.rename(index=inv_dict, inplace=True)
    #
    #     # Compare the line fluxes
    #     # spec.plot.spectrum(log_scale=True)
    #     for line in ['O3_4959A', 'Ar3_7136A', 'S2_6716A', 'S2_6731A']:
    #         sdss_flux, sdss_err = sdss_df.loc[line, ['LINEAREA', 'LINEAREA_ERR']]
    #         lime_intg, lime_intg_err = lime_df.loc[line, ['intg_flux', 'intg_flux_err']]
    #         lime_gauss, lime_gauss_err = lime_df.loc[line, ['profile_flux', 'profile_flux_err']]
    #
    #         sdss_cont, sdss_cont_err = sdss_df.loc[line, ['LINECONTLEVEL', 'LINECONTLEVEL_ERR']]
    #         lime_cont, lime_cont_err = lime_df.loc[line, ['cont', 'cont_err']]
    #
    #         # Diagnostic for matching methodology fluxes
    #         sigma_quad = np.sqrt(np.square(lime_gauss_err) + np.square(sdss_err))
    #         diag_arr = np.abs(lime_gauss - sdss_flux) <= 3 * sigma_quad
    #
    #         # print(f'\nLine Area SDSS : {line} = {sdss_flux:0.3f} ± {sdss_err:0.3f}')
    #         # print(f'Integrated LiMe :        = {lime_intg:0.3f} ± {lime_intg_err:0.3f}')
    #         # print(f'Gaussian LiMe :          = {lime_gauss:0.3f} ± {lime_gauss_err:0.3f}')
    #         # print('Gaussian flux close', diag_arr)
    #
    #         # Diagnostic for matching methodology fluxes
    #         sigma_quad = np.sqrt(np.square(lime_cont_err) + np.square(sdss_cont_err))
    #         diag_arr = np.abs(lime_cont - sdss_cont) <= 3 * sigma_quad
    #
    #         # print(f'Cont SDSS : {line} = {sdss_cont:0.3f} ± {sdss_cont_err:0.3f}')
    #         # print(f'Cont LiMe :          {lime_cont:0.3f} ± {lime_cont_err:0.3f}')
    #         # print('Continuum flux close',diag_arr)
    #
    #         # assert np.isclose(sdss_flux, lime_gauss, rtol=0.05)
    #         # assert np.isclose(sdss_cont, lime_cont, rtol=0.05)
    #
    #     return