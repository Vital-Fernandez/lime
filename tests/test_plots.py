import numpy as np
import lime
from pathlib import Path
import pytest
from matplotlib import pyplot as plt
from lime.io import _LOG_EXPORT_DICT
from os import remove
from copy import deepcopy

# Data for the tests
baseline_folder = Path(__file__).parent / 'baseline'
outputs_folder = Path(__file__).parent / '3_explanations'
spectra_folder = Path(__file__).parent.parent/'examples/0_resources/spectra'
file_address = baseline_folder/'sdss_dr18_0358-51818-0504.fits'
conf_file_address = baseline_folder/'lime_tests.toml'
bands_file_address = baseline_folder/f'manga_line_bands.txt'

redshift = 0.0475
# norm_flux = 1e-17
cfg = lime.load_cfg(conf_file_address)
tolerance_rms = 5.5


class TestSpectrumClass:

    @pytest.mark.mpl_image_compare(tolerance=tolerance_rms)
    def test_continuum_plot(self):

        fig = plt.figure()
        spec = lime.Spectrum.from_file(file_address, 'sdss', redshift=0.0475, id_label='SHOC579-SDSS')
        spec.fit.continuum(degree_list=[3, 6, 6, 7], emis_threshold=[5, 3, 2, 2])
        spec.plot.spectrum(in_fig=fig, show_cont=True, log_scale=True, label='SHOC579', rest_frame=True)

        return fig

    @pytest.mark.mpl_image_compare(tolerance=tolerance_rms)
    def test_err_plot(self):

        fig = plt.figure()
        spec = lime.Spectrum.from_file(file_address, 'sdss', redshift=0.0475, id_label='SHOC579-SDSS')
        spec.err_flux = spec.err_flux * 5
        spec.plot.spectrum(in_fig=fig, show_err=True, log_scale=True, label='SHOC579',
                           ax_cfg={'title': 'Test err * 10', 'xlabel': 'Dipersion axis'})

        return fig

