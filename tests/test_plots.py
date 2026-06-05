# import numpy as np
# import lime
# from pathlib import Path
# import pytest
# from matplotlib import pyplot as plt
# from lime.io import _LOG_EXPORT_DICT
# from os import remove
# from copy import deepcopy
#
# # Data for the tests
# baseline_folder = Path(__file__).parent / 'baseline'
# outputs_folder = Path(__file__).parent / '3_explanations'
# spectra_folder = Path(__file__).parent.parent/'examples/0_resources/spectra'
# file_address = baseline_folder/'sdss_dr18_0358-51818-0504.fits'
# conf_file_address = baseline_folder/'lime_tests.toml'
# bands_file_address = baseline_folder/f'manga_line_bands.txt'
#
# redshift = 0.0475
# # norm_flux = 1e-17
# cfg = lime.load_cfg(conf_file_address)
# tolerance_rms = 5.5
#
#
# class TestSpectrumClass:
#
#     @pytest.mark.mpl_image_compare(tolerance=tolerance_rms)
#     def test_continuum_plot(self):
#
#         fig = plt.figure()
#         spec = lime.Spectrum.from_file(file_address, 'sdss', redshift=0.0475, id_label='SHOC579-SDSS')
#         spec.fit.continuum(degree_list=[3, 6, 6, 7], emis_threshold=[5, 3, 2, 2])
#         spec.plot.spectrum(in_fig=fig, show_cont=True, log_scale=True, label='SHOC579', rest_frame=True)
#
#         return fig
#
#     @pytest.mark.mpl_image_compare(tolerance=tolerance_rms)
#     def test_err_plot(self):
#
#         fig = plt.figure()
#         spec = lime.Spectrum.from_file(file_address, 'sdss', redshift=0.0475, id_label='SHOC579-SDSS')
#         spec.err_flux = spec.err_flux * 5
#         spec.plot.spectrum(in_fig=fig, show_err=True, log_scale=True, label='SHOC579',
#                            ax_cfg={'title': 'Test err * 10', 'xlabel': 'Dipersion axis'})
#
#         return fig

import numpy as np
import lime
from pathlib import Path
import pytest
from matplotlib import pyplot as plt
from unittest.mock import patch
from lime.plotting.plots import _auto_flux_scale

# ── Paths ──────────────────────────────────────────────────────────────────────
baseline_folder = Path(__file__).parent / 'baseline'
outputs_folder  = Path(__file__).parent / '3_explanations'
file_address    = baseline_folder / 'sdss_dr18_0358-51818-0504.fits'
conf_file_address = baseline_folder / 'lime_tests.toml'
bands_file_address = baseline_folder/f'SHOC579_bands.txt'

REDSHIFT       = 0.0475
TOLERANCE_RMS  = 5.5

@pytest.fixture(scope='module')
def spec_basic():
    return lime.Spectrum.from_file(file_address, 'sdss', redshift=REDSHIFT, id_label='SHOC579-SDSS')

@pytest.fixture(scope='module')
def spec_fitted():
    """Spectrum with continuum + several line fits — reused across tests."""
    spec = lime.Spectrum.from_file(file_address, 'sdss', redshift=REDSHIFT, id_label='SHOC579-SDSS')
    spec.fit.continuum(degree_list=[3, 6, 6, 7], emis_threshold=[5, 3, 2, 2])
    for line in ['H1_6563A', 'O3_5007A', 'H1_4861A']:
        try:
            spec.fit.bands(line, bands_file_address)
        except Exception:
            pass
    return spec

# save_close_fig_swicth
class TestSaveCloseFigSwitch:


    def test_saves_to_file(self, tmp_path):
        out = tmp_path / 'spectrum.png'
        spec = lime.Spectrum.from_file(file_address, 'sdss', redshift=REDSHIFT)
        spec.plot.spectrum(fname=out)
        assert out.exists()
        plt.close('all')

    def test_display_check_false_skips_show(self, spec_basic):
        import lime.plotting.plots as lime_plot
        fig = plt.figure()
        with patch.object(lime_plot.plt, 'show') as mock_show:
            spec_basic.plot.spectrum(in_fig=fig)
            mock_show.assert_not_called()
        plt.close('all')

    def test_unrecognized_path_logs_info(self, spec_basic, caplog):
        import logging
        fig = plt.figure()
        # Pass an integer as fname — not a Path/str, triggers the else branch
        with caplog.at_level(logging.INFO, logger='LiMe'):
            spec_basic.plot.spectrum(in_fig=fig, fname=42)
        plt.close('all')

class TestSpectrumPlot:

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE_RMS)
    def test_continuum_plot(self):
        fig = plt.figure()
        spec = lime.Spectrum.from_file(file_address, 'sdss', redshift=REDSHIFT, id_label='SHOC579-SDSS')
        spec.fit.continuum(degree_list=[3, 6, 6, 7], emis_threshold=[5, 3, 2, 2])
        spec.plot.spectrum(in_fig=fig, show_cont=True, log_scale=True, label='SHOC579', rest_frame=True)
        return fig

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE_RMS)
    def test_err_plot(self):
        fig = plt.figure()
        spec = lime.Spectrum.from_file(file_address, 'sdss', redshift=REDSHIFT, id_label='SHOC579-SDSS')
        spec.err_flux = spec.err_flux * 5
        spec.plot.spectrum(in_fig=fig, show_err=True, log_scale=True, label='SHOC579',
                           ax_cfg={'title': 'Test err * 10', 'xlabel': 'Dispersion axis'})
        return fig

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE_RMS)
    def test_rest_frame_with_profiles(self, spec_fitted):
        fig = plt.figure()
        spec_fitted.plot.spectrum(in_fig=fig, rest_frame=True, show_profiles=True)
        return fig

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE_RMS)
    def test_observed_frame_no_profiles(self, spec_basic):
        fig = plt.figure()
        spec_basic.plot.spectrum(in_fig=fig, rest_frame=False, show_profiles=False)
        return fig

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE_RMS)
    def test_show_masks(self, spec_basic):
        fig = plt.figure()
        spec_basic.plot.spectrum(in_fig=fig, show_masks=True)
        return fig

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE_RMS)
    def test_show_err_no_err_flux(self):
        """show_err=True but err_flux is None — hits the _logger.info branch."""
        fig = plt.figure()
        spec = lime.Spectrum.from_file(file_address, 'sdss', redshift=REDSHIFT)
        spec._err_flux = None
        spec.plot.spectrum(in_fig=fig, show_err=True)
        return fig

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE_RMS)
    def test_with_bands_overlay(self, spec_basic):
        fig = plt.figure()
        spec_basic.plot.spectrum(in_fig=fig, bands=bands_file_address)
        return fig

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE_RMS)
    def test_log_scale(self, spec_basic):
        fig = plt.figure()
        spec_basic.plot.spectrum(in_fig=fig, log_scale=True)
        return fig

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE_RMS)
    def test_show_cont_none(self, spec_basic):
        """show_cont=True but no continuum fitted — should not crash."""
        fig = plt.figure()
        spec_basic.plot.spectrum(in_fig=fig, show_cont=True)
        return fig

class TestGridPlot:

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE_RMS)
    def test_grid_no_profiles(self, spec_fitted):
        fig = plt.figure()
        spec_fitted.plot.grid(in_fig=fig, show_profiles=False)
        return fig

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE_RMS)
    def test_grid_with_profiles(self, spec_fitted):
        fig = plt.figure()
        spec_fitted.plot.grid(in_fig=fig, show_profiles=True)
        return fig

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE_RMS)
    def test_grid_rest_frame(self, spec_fitted):
        fig = plt.figure()
        spec_fitted.plot.grid(in_fig=fig, rest_frame=True)
        return fig

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE_RMS)
    def test_grid_with_adjacent(self, spec_fitted):
        fig = plt.figure()
        spec_fitted.plot.grid(in_fig=fig, show_adjacent=True)
        return fig

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE_RMS)
    def test_grid_yscale_log(self, spec_fitted):
        fig = plt.figure()
        spec_fitted.plot.grid(in_fig=fig, y_scale='log')
        return fig

    def test_grid_empty_frame_logs(self, spec_basic, caplog):
        """No lines measured → hits the _logger.info branch."""
        import logging
        fig = plt.figure()
        with caplog.at_level(logging.INFO, logger='LiMe'):
            spec_basic.plot.grid(in_fig=fig)
        plt.close('all')

class TestBandsPlot:

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE_RMS)
    def test_bands_with_profile(self, spec_fitted):
        fig = plt.figure()
        spec_fitted.plot.bands('H1_6563A', in_fig=fig, show_profile=True)
        return fig

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE_RMS)
    def test_bands_no_profile(self, spec_fitted):
        fig = plt.figure()
        spec_fitted.plot.bands('H1_6563A', in_fig=fig, show_profile=False)
        return fig

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE_RMS)
    def test_bands_show_err(self, spec_fitted):
        fig = plt.figure()
        spec_fitted.plot.bands('H1_6563A', in_fig=fig, show_err=True)
        return fig

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE_RMS)
    def test_bands_rest_frame(self, spec_fitted):
        fig = plt.figure()
        spec_fitted.plot.bands('H1_6563A', in_fig=fig, rest_frame=True)
        return fig

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE_RMS)
    def test_bands_yscale_log(self, spec_fitted):
        fig = plt.figure()
        spec_fitted.plot.bands('H1_6563A', in_fig=fig, y_scale='log')
        return fig

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE_RMS)
    def test_bands_show_cont(self, spec_fitted):
        fig = plt.figure()
        spec_fitted.plot.bands('H1_6563A', in_fig=fig, show_cont=True)
        return fig

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE_RMS)
    def test_bands_no_adjacent(self, spec_fitted):
        fig = plt.figure()
        spec_fitted.plot.bands('H1_6563A', in_fig=fig, show_bands=False)
        return fig

    def test_bands_line_not_found_logs(self, spec_basic, caplog):
        """Line absent from frame → hits the _logger.info branch."""
        import logging
        fig = plt.figure()
        with caplog.at_level(logging.INFO, logger='LiMe'):
            spec_basic.plot.bands('H1_6563A', in_fig=fig)
        plt.close('all')

# class TestVelocityProfile:
#
#     @pytest.mark.mpl_image_compare(tolerance=TOLERANCE_RMS)
#     def test_velocity_default(self, spec_fitted):
#         fig = plt.figure()
#         spec_fitted.plot.velocity_profile('H1_6563A', in_fig=fig)
#         return fig
#
#     @pytest.mark.mpl_image_compare(tolerance=TOLERANCE_RMS)
#     def test_velocity_log_scale(self, spec_fitted):
#         fig = plt.figure()
#         spec_fitted.plot.velocity_profile('H1_6563A', in_fig=fig, y_scale='log')
#         return fig
#
#     @pytest.mark.mpl_image_compare(tolerance=TOLERANCE_RMS)
#     def test_velocity_custom_ax_cfg(self, spec_fitted):
#         fig = plt.figure()
#         spec_fitted.plot.velocity_profile('H1_6563A', in_fig=fig,
#                                           ax_cfg={'title': 'Ha velocity'})
#         return fig

class TestAutoFluxScale:

    def _make_axis(self):
        fig, ax = plt.subplots()
        return ax

    def test_linear_branch(self):
        from lime.plotting.plots import _auto_flux_scale
        ax = self._make_axis()
        y = np.linspace(1, 2, 100)          # ratio ~2 → linear
        _auto_flux_scale(ax, y, 'auto')
        plt.close('all')

    def test_log_branch(self):
        from lime.plotting.plots import _auto_flux_scale
        ax = self._make_axis()
        y = np.logspace(0, 3, 100)          # ratio 1000 → log
        _auto_flux_scale(ax, y, 'auto')
        plt.close('all')

    def test_symlog_branch_with_negatives(self):
        from lime.plotting.plots import _auto_flux_scale
        ax = self._make_axis()
        y = np.concatenate([np.linspace(-100, -1, 50), np.linspace(1, 5, 50)])
        _auto_flux_scale(ax, y, 'auto')
        plt.close('all')

    def test_explicit_scale_string(self):
        from lime.plotting.plots import _auto_flux_scale
        ax = self._make_axis()
        y = np.ones(50)
        _auto_flux_scale(ax, y, 'log')      # explicit string, not 'auto'
        plt.close('all')