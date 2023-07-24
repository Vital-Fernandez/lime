import numpy as np
import lime
from pathlib import Path
from unittest.mock import patch
from matplotlib import pyplot as plt
from matplotlib.testing.compare import compare_images
from lime.io import _LOG_EXPORT_DICT

# Data for the tests
file_address = Path(__file__).parent/'data_tests'/'manga_spaxel.txt'
spectrum_plot_address = Path(__file__).parent/'data_tests'/'spectrum_manga_spaxel.png'
line_plot_address = Path(__file__).parent/'data_tests'/'Fe3_4658A_manga_spaxel.png'
conf_file_address = Path(__file__).parent/'data_tests'/'manga.toml'
bands_file_address = Path(__file__).parent/'data_tests'/'manga_lines_log.txt'
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

    def test_plot_spectrum(self):

        image_address = 'test_plot_spectrum.png'
        spec.plot.spectrum(output_address=image_address)
        compare_images(spectrum_plot_address, image_address, tol=0.001, in_decorator=False)

        return

    def test_plot_bands(self):

        image_address = 'test_Fe3_4658A_manga_spaxel.png'
        spec.fit.bands('Fe3_4658A')
        spec.plot.bands(output_address=image_address)
        compare_images(line_plot_address, image_address, tol=0.1, in_decorator=False)

        return

    def test_frame_measurements(self):

        log_orig = lime.load_log(lines_log_address)
        spec.fit.frame(bands_file_address, cfg, id_conf_prefix='38-35')

        for line in spec.log.index:
            for column in spec.log.columns:
                print(_LOG_EXPORT_DICT[column])

        return


