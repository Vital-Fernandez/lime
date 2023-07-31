import numpy as np
import lime
from pathlib import Path
import pytest
from matplotlib import pyplot as plt
from matplotlib.testing.compare import compare_images
from lime.io import _LOG_EXPORT_DICT

# Data for the tests
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

    @pytest.mark.mpl_image_compare
    def test_plot_spectrum(self):

        fig = plt.figure()
        spec.plot.spectrum(in_fig=fig)

        return fig

    @pytest.mark.mpl_image_compare
    def test_plot_line(self):

        fig = plt.figure()
        spec.plot.bands('Fe3_4658A_p-g_emis', in_fig=fig)

        return fig

    def test_measurements_txt_file(self):

        extension = 'txt'
        # spec.fit.frame(bands_file_address, cfg, id_conf_prefix='38-35')
        spec.save_log(f'test_lines_log.{extension}')

        log_orig = lime.load_log(lines_log_address)
        log_test = lime.load_log(f'test_lines_log.{extension}')

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

    def test_measurements_fits_file(self):

        extension = 'fits'
        # spec.fit.frame(bands_file_address, cfg, id_conf_prefix='38-35')
        spec.save_log(f'test_lines_log.{extension}')

        log_orig = lime.load_log(lines_log_address)
        log_test = lime.load_log(f'test_lines_log.{extension}')

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
        # spec.fit.frame(bands_file_address, cfg, id_conf_prefix='38-35')
        spec.save_log(f'test_lines_log.{extension}')

        log_orig = lime.load_log(lines_log_address)
        log_test = lime.load_log(f'test_lines_log.{extension}')

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
                        print(param, log_orig.loc[line, param], log_test.loc[line, param])
                        assert np.allclose(log_orig.loc[line, param], log_test.loc[line, param], rtol=0.05,
                                              equal_nan=True)
                    else:
                        assert np.allclose(log_orig.loc[line, param], log_test.loc[line, param], rtol=0.15,
                                              equal_nan=True)

        return

    def test_measurements_xlsx_file(self):

        extension = 'xlsx'
        # spec.fit.frame(bands_file_address, cfg, id_conf_prefix='38-35')
        spec.save_log(f'test_lines_log.{extension}')

        log_orig = lime.load_log(lines_log_address)
        log_test = lime.load_log(f'test_lines_log.{extension}')

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
                        print(param, log_orig.loc[line, param], log_test.loc[line, param])
                        assert np.allclose(log_orig.loc[line, param], log_test.loc[line, param], rtol=0.05,
                                              equal_nan=True)
                    else:
                        assert np.allclose(log_orig.loc[line, param], log_test.loc[line, param], rtol=0.15,
                                              equal_nan=True)

        return

    def test_measurements_asdf_file(self):

        extension = 'asdf'
        # spec.fit.frame(bands_file_address, cfg, id_conf_prefix='38-35')
        spec.save_log(f'test_lines_log.{extension}')

        log_orig = lime.load_log(lines_log_address)
        log_test = lime.load_log(f'test_lines_log.{extension}')

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
                        print(param, log_orig.loc[line, param], log_test.loc[line, param])
                        assert np.allclose(log_orig.loc[line, param], log_test.loc[line, param], rtol=0.05,
                                              equal_nan=True)
                    else:
                        assert np.allclose(log_orig.loc[line, param], log_test.loc[line, param], rtol=0.15,
                                              equal_nan=True)

        return