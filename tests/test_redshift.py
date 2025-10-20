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
file_address = baseline_folder/'SHOC579_MANGA38-35.txt'
conf_file_address = baseline_folder/'lime_tests.toml'
bands_file_address = baseline_folder/f'manga_line_bands.txt'
lines_log_address = baseline_folder/'SHOC579_MANGA38-35_log.txt'
lines_tex_address = baseline_folder/'manga_lines_log.tex'

redshift = 0.0475
norm_flux = 1e-17
cfg = lime.load_cfg(conf_file_address)
cfg_copy = deepcopy(cfg)
tolerance_rms = 5.5

wave_array, flux_array, err_array, pixel_mask = np.loadtxt(file_address, unpack=True)
# pixel_mask = np.isnan(err_array)

spec = lime.Spectrum(wave_array, flux_array, err_array, redshift=redshift, norm_flux=norm_flux,
                     pixel_mask=pixel_mask, id_label='SHOC579_Manga38-35')

try:
    import aspect
    spec.infer.components()
    check_aspect = True
except ImportError:
    print('No aspect')
    check_aspect = False


# if check_aspect:
#
#     @pytest.mark.mpl_image_compare(tolerance=tolerance_rms)
#     def test_plot_spectrum_with_components():
#         fig = plt.figure()
#         spec.plot.spectrum(in_fig=fig, show_components=True)
#
#         return fig
#
#     def test_redshift():
#
#         bands = spec.retrieve.lines_frame(line_list=['O2_3726A', 'H1_4861A', 'O3_4959A', 'O3_5007A', 'H1_6563A'])
#         # spec.plot.spectrum(bands=bands,show_categories=True, rest_frame=True)
#         z_infer = spec.fit.redshift(bands=bands, z_min=0, z_max=1, mode='key', components=['emission'], plot_results=False)
#         assert np.isclose(z_infer, redshift, rtol=0.10)
#
#         z_infer = spec.fit.redshift(bands=bands, mode='xor', z_min=0, z_max=1, plot_results=False)
#         assert np.isclose(z_infer, redshift, rtol=0.10)
#
#         z_infer = spec.fit.redshift(bands=bands, mode='permute') # TODO not providing good results
#         # assert np.isclose(z_infer, 0.00627, rtol=0.10)
#
#         return


@pytest.mark.mpl_image_compare(tolerance=tolerance_rms)
def test_plot_spectrum_with_no_components():
    fig = plt.figure()
    spec.plot.spectrum(in_fig=fig, show_components=False)

    return fig
