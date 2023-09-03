import numpy as np
import lime
from pathlib import Path
import pytest
from matplotlib import pyplot as plt
from matplotlib.testing.compare import compare_images
from lime.io import _LOG_EXPORT_DICT
from os import remove

# Data for the tests
baseline_folder = Path(__file__).parent/'data_tests'
file_address = Path(__file__).parent/'data_tests'/'manga_spaxel.txt'
conf_file_address = Path(__file__).parent/'data_tests'/'manga.toml'
bands_file_address = Path(__file__).parent/'data_tests'/f'manga_line_bands.txt'
lines_log_address = Path(__file__).parent/'data_tests'/'manga_lines_log.txt'

redshift = 0.0475
norm_flux = 1e-17
cfg = lime.load_cfg(conf_file_address)

wave_array, flux_array, err_array = np.loadtxt(file_address, unpack=True)
pixel_mask = np.isnan(err_array)

spec = lime.Spectrum(wave_array, flux_array, err_array, redshift=redshift, norm_flux=norm_flux,
                     pixel_mask=pixel_mask)

spec.fit.frame(bands_file_address, cfg, id_conf_prefix='38-35')


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

    @pytest.mark.line_detection_plot
    def test_line_detection_plot(self):

        match_bands = spec.line_detection(bands_file_address, cont_fit_degree=[3, 7, 7, 7], cont_int_thres=[5, 3, 2, 1.5])

        fig = plt.figure()
        spec.plot.spectrum(in_fig=fig, line_bands=match_bands)

        return fig

    @pytest.mark.mpl_image_compare
    def test_plot_spectrum(self):

        fig = plt.figure()
        spec.plot.spectrum(in_fig=fig)

        return fig

    @pytest.mark.mpl_image_compare
    def test_plot_spectrum_with_fits(self):

        fig = plt.figure()
        spec.plot.spectrum(in_fig=fig, include_fits=True)

        return fig

    @pytest.mark.mpl_image_compare
    def test_check_bands_spectrum(self):

        fig = plt.figure()
        spec.check.bands(bands_file=bands_file_address, in_fig=fig)

        return fig

    @pytest.mark.mpl_image_compare
    def test_plot_spectrum_maximize(self):

        fig = plt.figure()
        spec.plot.spectrum(in_fig=fig, include_fits=True, maximize=True)

        return fig

    @pytest.mark.mpl_image_compare
    def test_plot_spectrum_with_bands(self):

        fig = plt.figure()
        spec.plot.spectrum(in_fig=fig, line_bands=bands_file_address)

        return fig

    @pytest.mark.mpl_image_compare
    def test_plot_line(self):

        fig = plt.figure()
        spec.plot.bands('Fe3_4658A_p-g_emis', in_fig=fig)

        return fig

    @pytest.mark.mpl_image_compare
    def test_plot_grid(self):

        fig = plt.figure()
        spec.plot.grid(in_fig=fig)

        return fig

    @pytest.mark.mpl_image_compare(tolerance=10)
    def test_plot_cinematics(self):

        fig = plt.figure()
        spec.plot.velocity_profile('O3_5007A', in_fig=fig)

        return fig

    def test_measurements_txt_file(self):

        extension = 'txt'
        spec.save_log(baseline_folder/f'test_lines_log.{extension}')

        log_orig = lime.load_log(lines_log_address)
        log_test = lime.load_log(baseline_folder/f'test_lines_log.{extension}')

        for line in spec.log.index:
            for param in spec.log.columns:

                # String
                if _LOG_EXPORT_DICT[param].startswith('<U'):
                    if log_orig.loc[line, param] is np.nan:
                        assert log_orig.loc[line, param] is log_test.loc[line, param]
                    else:
                        assert log_orig.loc[line, param] == log_test.loc[line, param]

                # Float
                else:
                    if param not in ['eqw', 'eqw_err']:
                        assert np.allclose(log_orig.loc[line, param], log_test.loc[line, param], rtol=0.05, equal_nan=True)
                    else:
                        assert np.allclose(log_orig.loc[line, param], log_test.loc[line, param], rtol=0.15,
                                              equal_nan=True)

        return

    def test_measurements_fits_file(self):

        extension = 'fits'
        spec.save_log(baseline_folder/f'test_lines_log.{extension}')

        log_orig = lime.load_log(lines_log_address)
        log_test = lime.load_log(baseline_folder/f'test_lines_log.{extension}')

        for line in spec.log.index:
            for param in spec.log.columns:

                # String
                if _LOG_EXPORT_DICT[param].startswith('<U'):
                    if log_orig.loc[line, param] is np.nan:
                        assert log_orig.loc[line, param] is log_test.loc[line, param]
                    else:
                        assert log_orig.loc[line, param] == log_test.loc[line, param]

                # Float
                else:
                    if param not in ['eqw', 'eqw_err']:
                        assert np.allclose(log_orig.loc[line, param], log_test.loc[line, param], rtol=0.05,
                                              equal_nan=True)
                    else:
                        assert np.allclose(log_orig.loc[line, param], log_test.loc[line, param], rtol=0.15,
                                              equal_nan=True)

        return

    def test_measurements_csv_file(self):

        extension = 'csv'
        spec.save_log(baseline_folder/f'test_lines_log.{extension}')

        log_orig = lime.load_log(lines_log_address)
        log_test = lime.load_log(baseline_folder/f'test_lines_log.{extension}')

        for line in spec.log.index:
            for param in spec.log.columns:

                # String
                if _LOG_EXPORT_DICT[param].startswith('<U'):
                    if log_orig.loc[line, param] is np.nan:
                        assert log_orig.loc[line, param] is log_test.loc[line, param]
                    else:
                        assert log_orig.loc[line, param] == log_test.loc[line, param]

                # Float
                else:
                    if param not in ['eqw', 'eqw_err']:
                        assert np.allclose(log_orig.loc[line, param], log_test.loc[line, param], rtol=0.05,
                                              equal_nan=True)
                    else:
                        assert np.allclose(log_orig.loc[line, param], log_test.loc[line, param], rtol=0.15,
                                              equal_nan=True)

        return

    def test_measurements_xlsx_file(self):

        extension = 'xlsx'
        spec.save_log(baseline_folder/f'test_lines_log.{extension}')

        log_orig = lime.load_log(lines_log_address)
        log_test = lime.load_log(baseline_folder/f'test_lines_log.{extension}')

        for line in spec.log.index:
            for param in spec.log.columns:

                # String
                if _LOG_EXPORT_DICT[param].startswith('<U'):
                    if log_orig.loc[line, param] is np.nan:
                        assert log_orig.loc[line, param] is log_test.loc[line, param]
                    else:
                        assert log_orig.loc[line, param] == log_test.loc[line, param]

                # Float
                else:
                    if param not in ['eqw', 'eqw_err']:
                        assert np.allclose(log_orig.loc[line, param], log_test.loc[line, param], rtol=0.05,
                                              equal_nan=True)
                    else:
                        assert np.allclose(log_orig.loc[line, param], log_test.loc[line, param], rtol=0.15,
                                              equal_nan=True)

        return

    def test_extra_pages_xlsx(self):

        file_xlsx = baseline_folder/'test_lines_log_multi_page.xlsx'

        if file_xlsx.is_file():
            try:
                remove(file_xlsx)
                print(f"File '{file_xlsx}' has been deleted successfully.")
            except OSError as e:
                print(f"Error: {e.strerror}")

        log_orig = lime.load_log(lines_log_address)

        for page in ['LINELOG', 'LINESLOG2', 'LINELOG']:
            spec.save_log(baseline_folder/file_xlsx, page=page)
            log_test = lime.load_log(file_xlsx)

            for line in spec.log.index:
                for param in spec.log.columns:

                    # String
                    if _LOG_EXPORT_DICT[param].startswith('<U'):
                        if log_orig.loc[line, param] is np.nan:
                            assert log_orig.loc[line, param] is log_test.loc[line, param]
                        else:
                            assert log_orig.loc[line, param] == log_test.loc[line, param]

                    # Float
                    else:
                        if param not in ['eqw', 'eqw_err']:
                            assert np.allclose(log_orig.loc[line, param], log_test.loc[line, param], rtol=0.05,
                                                  equal_nan=True)
                        else:
                            assert np.allclose(log_orig.loc[line, param], log_test.loc[line, param], rtol=0.15,
                                                  equal_nan=True)

        return

    # def test_measurements_asdf_file(self):
    #
    #     extension = 'asdf'
    #     spec.save_log(baseline_folder/f'test_lines_log.{extension}')
    #
    #     log_orig = lime.load_log(lines_log_address)
    #     log_test = lime.load_log(baseline_folder/f'test_lines_log.{extension}')
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

        spec0.load_log(lines_log_address)

        new_log_spectrum = baseline_folder/ 'manga_lines_log_from_spectrum.txt'
        spec0.save_log(new_log_spectrum)

        assert new_log_spectrum.is_file()

        return