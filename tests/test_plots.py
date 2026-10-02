import numpy as np
import pandas as pd
import lime
from pathlib import Path
from types import SimpleNamespace
import pytest
from astropy import units as au
from matplotlib import pyplot as plt
from matplotlib._pylab_helpers import Gcf
from unittest.mock import patch
import lime.plotting.plots as lime_plot
from lime.io import LiMe_Error
from lime.plotting.format import theme, latex_science_float, spectrum_figure_labels
from lime.plotting.plots import (_auto_flux_scale, _auto_flux_scale_backUp, line_band_scaler, save_close_fig_swicth,
                                 maximize_center_fig, check_image_size, image_plot, spatial_mask_plot, _masks_plot,
                                 label_generator, bands_filling_plot, line_band_plotter, line_profile_generator,
                                 mplcursors_legend, redshift_key_evaluation, spec_plot, spec_continuum_calculation,
                                 spec_peak_calculation, Plotter, SampleFigures)

try:
    import aspect
    aspect_check = True
except ImportError:
    aspect_check = False

aspect_required = pytest.mark.skipif(not aspect_check, reason='ASPECT is not installed')

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

@pytest.fixture
def mock_show():
    """Replace plt.show so the display branches run without opening a window."""
    with patch.object(lime_plot.plt, 'show') as mock:
        yield mock


def figure_is_open(fig):
    """Check by object, since matplotlib reuses the figure numbers after closing."""
    return any(manager.canvas.figure is fig for manager in Gcf.get_all_fig_managers())


def fake_line(profiles):
    """Minimal stand-in for a lime Line with the attributes read by line_profile_generator."""
    n = len(profiles)
    comps = [SimpleNamespace(profile=profile, label=f'comp_{i}', group='s') for i, profile in enumerate(profiles)]
    measurements = SimpleNamespace(amp=np.full(n, 10.0), center=np.full(n, 5000.0), sigma=np.full(n, 2.0),
                                   gamma=np.full(n, 2.0), frac=np.full(n, 0.5), alpha=np.full(n, 2.0),
                                   a=np.full(n, 1.0), b=np.full(n, 1.0))

    return SimpleNamespace(group='b' if n > 1 else 's', profile=profiles[0], label='fake_line', list_comps=comps,
                           measurements=measurements)


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


SCALE_FUNCTIONS = [_auto_flux_scale, _auto_flux_scale_backUp, line_band_scaler]
SCALE_CASES = [(np.linspace(1, 2, 100), 'auto', 'linear'),
               (np.logspace(0, 3, 100), 'auto', 'log'),
               (np.concatenate([np.linspace(-100, -1, 50), np.linspace(1, 5, 50)]), 'auto', 'symlog'),
               (np.linspace(-100, -1, 50), 'auto', 'symlog'),       # No positive entries for the linear threshold
               (np.ones(50), 'log', 'log'),                         # Explicit scale
               (np.ones(50), 'linear', 'linear')]


class TestFluxScalers:

    @pytest.mark.parametrize('scale_function', SCALE_FUNCTIONS)
    @pytest.mark.parametrize('y, y_scale, expected', SCALE_CASES)
    def test_scale_selection(self, scale_function, y, y_scale, expected):

        fig, ax = plt.subplots()
        scale_function(ax, y, y_scale)

        assert ax.get_yscale() == expected

        # Non-linear automatic scales are annotated in the axis
        if y_scale == 'auto':
            assert len(ax.texts) == (0 if expected == 'linear' else 1)

        plt.close('all')

    def test_band_scaler_single_pixel(self):

        # Automatic scale needs more than one pixel
        fig, ax = plt.subplots()
        line_band_scaler(ax, np.array([5.0]), 'auto')

        assert ax.get_yscale() == 'linear'
        assert len(ax.texts) == 0

        plt.close('all')


class TestFigureSwitch:

    def test_no_plot_check(self, tmp_path, mock_show):

        fig = plt.figure()
        out = tmp_path / 'no_plot.png'

        assert save_close_fig_swicth(out, 'tight', fig, False, plot_check=False) is None
        assert not out.exists()
        mock_show.assert_not_called()

        plt.close('all')

    def test_display(self, mock_show):

        fig = plt.figure()
        fig.add_subplot().plot([1, 2, 3], [1, 2, 3])

        # Tight layout and window maximizing (not available for non-interactive backends)
        save_close_fig_swicth(None, 'tight', fig, True, True)
        assert mock_show.call_count == 1

        save_close_fig_swicth(None, None, fig, False, True)
        assert mock_show.call_count == 2

        plt.close('all')

    def test_save_and_close(self, tmp_path, mock_show):

        # String address and figure closing
        fig = plt.figure()
        fig.add_subplot().plot([1, 2, 3], [1, 2, 3])
        out = tmp_path / 'figure.png'
        save_close_fig_swicth(str(out), 'tight', fig)

        assert out.is_file()
        assert not figure_is_open(fig)

        # Path address keeping the figure
        fig = plt.figure()
        fig.add_subplot().plot([1, 2, 3], [1, 2, 3])
        out = tmp_path / 'figure_open.png'
        save_close_fig_swicth(out, None, None)

        assert out.is_file()
        assert figure_is_open(fig)
        mock_show.assert_not_called()

        plt.close('all')

    def test_unrecognized_address(self, mock_show):

        fig = plt.figure()
        with patch.object(lime_plot._logger, 'info') as mock_info:
            assert save_close_fig_swicth(42, 'tight', fig) is None

        assert mock_info.call_count == 1
        mock_show.assert_not_called()

        plt.close('all')

    def test_maximize_center(self):

        # Should not fail on backends without a window
        fig = plt.figure()
        maximize_center_fig(maximize_check=True, center_check=True)
        maximize_center_fig()

        assert figure_is_open(fig)

        plt.close('all')


class TestPlotTools:

    def test_check_image_size(self):

        bg_image, fg_image = np.ones((10, 10)), np.ones((10, 10))
        mask_dict = {'MASK_0': (np.ones((10, 10), dtype=bool), {}), 'MASK_1': (np.ones((10, 10), dtype=bool), {})}

        # Matching sizes
        with patch.object(lime_plot._logger, 'warning') as mock_warning:
            check_image_size(bg_image, fg_image, mask_dict)
            check_image_size(bg_image, None, {})
        assert mock_warning.call_count == 0

        # Foreground and one mask with different size
        mask_dict['MASK_1'] = (np.ones((5, 5), dtype=bool), {})
        with patch.object(lime_plot._logger, 'warning') as mock_warning:
            check_image_size(bg_image, np.ones((8, 8)), mask_dict)
        assert mock_warning.call_count == 2

    def test_image_plot(self):

        rng = np.random.default_rng(1234)
        image_bg, image_fg = rng.uniform(1, 10, (10, 10)), rng.uniform(1, 10, (10, 10))

        # Background only
        fig, ax = plt.subplots()
        im, contours, marker = image_plot(ax, image_bg, None, None, None, None, None, 'gray', 'viridis')
        assert len(ax.images) == 1
        assert np.allclose(im.get_array(), image_bg)
        assert contours is None
        assert marker is None

        # Foreground contours and cursor
        fig, ax = plt.subplots()
        fg_mesh = np.meshgrid(np.arange(0, image_fg.shape[1]), np.arange(0, image_fg.shape[0]))
        fg_levels = np.percentile(image_fg, (50, 90))
        im, contours, marker = image_plot(ax, image_bg, image_fg, fg_levels, fg_mesh, None, None, 'gray', 'viridis',
                                          cursor_cords=(2, 3))
        assert contours is not None
        assert np.allclose(contours.levels, fg_levels)
        assert marker.get_xdata()[0] == 3
        assert marker.get_ydata()[0] == 2

        plt.close('all')

    def test_spatial_mask_plot(self):

        mask_0, mask_1 = np.zeros((10, 10), dtype=bool), np.zeros((10, 10), dtype=bool)
        mask_0[2:5, 2:5], mask_1[6:8, 6:8] = True, True
        masks_dict = {'MASK_0': (mask_0, {'PARAM': 'H1_6563A', 'PARAMIDX': 90, 'PARAMVAL': 1.5e-17, 'NUMSPAXE': 9}),
                      'MASK_1': (mask_1, {'PARAM': 'H1_6563A'})}

        # All masks (only those with the percentile information get a legend)
        fig, ax = plt.subplots()
        legend_list = spatial_mask_plot(ax, masks_dict, 'viridis_r', 0.2, 'flux_units')
        assert len(ax.images) == 2
        assert len(legend_list) == 2
        assert 'MASK_0' in legend_list[0].get_label()
        assert '(9 spaxels)' in legend_list[0].get_label()
        assert legend_list[1] is None

        # Masks selection
        fig, ax = plt.subplots()
        legend_list = spatial_mask_plot(ax, masks_dict, 'viridis_r', 0.2, 'flux_units', mask_list=['MASK_1'])
        assert len(ax.images) == 1
        assert legend_list == [None, None]

        plt.close('all')

    def test_masks_plot(self):

        x, y = np.arange(20, dtype=float), np.ones(20)
        pixel_mask = np.zeros(20, dtype=bool)
        pixel_mask[5:8] = True

        # Masked pixels
        fig, ax = plt.subplots()
        _masks_plot(ax, None, x, y, 2, None, pixel_mask)
        assert len(ax.collections) == 1
        offsets = ax.collections[0].get_offsets()
        assert np.allclose(offsets[:, 0], x[pixel_mask] / 2)
        assert np.allclose(offsets[:, 1], y[pixel_mask] * 2)

        # No masked pixels
        fig, ax = plt.subplots()
        _masks_plot(ax, None, x, y, 1, None, np.zeros(20, dtype=bool))
        assert len(ax.collections) == 0

        # Masked pixels without flux
        y_nan = y.copy()
        y_nan[pixel_mask] = np.nan
        fig, ax = plt.subplots()
        _masks_plot(ax, None, x, y_nan, 1, None, pixel_mask)
        assert len(ax.collections) == 0

        plt.close('all')

    def test_label_generator(self):

        index = pd.MultiIndex.from_tuples([('obj_0', 'file_0.fits'), ('obj_1', 'file_1.fits')], names=['id', 'file'])
        log = pd.DataFrame({'redshift': [0.1, 0.2]}, index=index)
        idx_sample = ('obj_1', 'file_1.fits')

        assert label_generator(idx_sample, log, 'levels') == 'obj_1, file_1.fits'
        assert label_generator(idx_sample, log, None) is None
        assert label_generator(idx_sample, log, 'id') == 'obj_1'
        assert label_generator(idx_sample, log, 'file') == 'file_1.fits'
        assert label_generator(idx_sample, log, 'redshift') == 0.2

        with pytest.raises(LiMe_Error):
            label_generator(idx_sample, log, 'not_a_column')

    def test_bands_filling_plot(self):

        x = np.arange(100, dtype=float)
        y = 1 + np.exp(-0.5 * np.square((x - 50) / 3))
        idcs_mask = np.array([10, 20, 40, 60, 80, 90])

        # Line band
        fig, ax = plt.subplots()
        bands_filling_plot(ax, x, y, 1, idcs_mask, 'H1_6563A')
        assert len(ax.collections) == 1

        # Line and continua bands
        fig, ax = plt.subplots()
        bands_filling_plot(ax, x, y, 1, idcs_mask, 'H1_6563A', exclude_continua=False)
        assert len(ax.collections) == 3

        # Continua bands
        fig, ax = plt.subplots()
        bands_filling_plot(ax, x, y, 1, idcs_mask, 'H1_6563A', exclude_continua=False, show_central=False)
        assert len(ax.collections) == 2

        # Line band below two pixels
        fig, ax = plt.subplots()
        with patch.object(lime_plot._logger, 'warning') as mock_warning:
            bands_filling_plot(ax, x, y, 1, np.array([10, 10, 10, 11, 11, 11]), 'H1_6563A')
        assert mock_warning.call_count == 1
        assert len(ax.collections) == 0

        plt.close('all')

    def test_line_band_plotter(self):

        x = np.arange(100, dtype=float)
        y = 1 + np.exp(-0.5 * np.square((x - 50) / 3))
        idcs_mask = np.array([10, 20, 40, 60, 80, 90])

        # Line and continua bands with a linear continuum
        fig, ax = plt.subplots()
        line_band_plotter(ax, x, y, 1, idcs_mask, 'H1_6563A', theme.colors, show_adjacent=True, m_cont=0, n_cont=1)
        assert len(ax.collections) == 3

        # Line band without a continuum
        fig, ax = plt.subplots()
        line_band_plotter(ax, x, y, 1, idcs_mask, 'H1_6563A', theme.colors, show_adjacent=False)
        assert len(ax.collections) == 1

        # Unsorted bands
        fig, ax = plt.subplots()
        with patch.object(lime_plot._logger, 'warning') as mock_warning:
            line_band_plotter(ax, x, y, 1, np.array([50, 55, 10, 20, 30, 30]), 'H1_6563A', theme.colors)
        assert mock_warning.call_count == 1
        assert 'Unsorted' in mock_warning.call_args[0][0]
        assert len(ax.collections) == 1

        # Line band below two pixels
        fig, ax = plt.subplots()
        with patch.object(lime_plot._logger, 'warning') as mock_warning:
            line_band_plotter(ax, x, y, 1, np.array([10, 10, 10, 11, 11, 11]), 'H1_6563A', theme.colors)
        assert mock_warning.call_count == 1
        assert len(ax.collections) == 0

        plt.close('all')

    @pytest.mark.parametrize('profile', ['g', 'l', 'v', 'pv', 'e', 'pp', 'p'])
    def test_line_profile_generator(self, profile):

        x = np.linspace(4980, 5020, 201)
        curve_arr = line_profile_generator(fake_line([profile]), x)

        assert curve_arr.shape == (1, x.size)

    def test_line_profile_generator_gaussian(self):

        x = np.linspace(4980, 5020, 201)

        # Single line
        curve_arr = line_profile_generator(fake_line(['g']), x)
        assert np.isclose(curve_arr.max(), 10)
        assert np.isclose(x[np.argmax(curve_arr[0])], 5000)

        # Blended line (one curve per component)
        curve_arr = line_profile_generator(fake_line(['g', 'g', 'l']), x)
        assert curve_arr.shape == (3, x.size)
        assert np.allclose(curve_arr[0], curve_arr[1])

        # Unknown profile
        with pytest.raises(LiMe_Error):
            line_profile_generator(fake_line(['not_a_profile']), x)

    def test_figure_labels(self):

        flux_units = au.erg / au.s / au.cm ** 2 / au.AA

        # Scientific notation
        assert latex_science_float(1.5) == '1.5'
        assert latex_science_float(1e-17) == r'1 \times 10^{-17}'
        assert latex_science_float(1.2345e-17, dec=3) == r'1.23 \times 10^{-17}'

        # Axes labels
        x_label, y_label = spectrum_figure_labels(au.AA, flux_units, 1)
        assert x_label.startswith('Wavelength (')
        assert y_label.startswith('Flux ')
        assert r'\times' not in y_label

        x_label, y_label = spectrum_figure_labels(au.AA, flux_units, 1e-17)
        assert r'1 \times 10^{-17}' in y_label

        x_label, y_label = spectrum_figure_labels(au.AA, au.dimensionless_unscaled, 1)
        assert y_label == 'Flux (scaled)'

        x_label, y_label = spectrum_figure_labels(au.AA, flux_units, 1, plotting_library='bokeh')
        assert '$$' in x_label

        # Figures without default labels
        assert theme.ax_defaults(None, fig_type='other') == {}
        assert theme.ax_defaults({'title': 'Test'}, fig_type='other') == {'title': 'Test'}


class TestPlotter:

    def test_plot_container(self):

        plotter = Plotter()

        # New figure
        fig, ax = plotter._plot_container(None, None, {'title': 'Test title'})
        assert fig is not None
        assert ax.get_title() == 'Test title'

        # Input figure
        fig_in, ax_in = plt.subplots()
        fig, ax = plotter._plot_container(fig_in, ax_in, {'xlabel': 'Test label'})
        assert (fig is fig_in) and (ax is ax_in)
        assert ax.get_xlabel() == 'Test label'

        # Line and residual axes
        plt.figure()
        fig, ax = plotter._plot_container(None, None, gfit_type=True)
        assert fig is None
        assert len(ax) == 2

        plt.close('all')

    def test_cont_and_peak_plot(self):

        plotter = Plotter()
        fig, ax = plt.subplots()

        x = np.linspace(4850, 4870, 20)
        plotter._cont_plot(ax, x, np.full(x.size, 4.0), 2, 2)
        assert len(ax.lines) == 1
        assert np.allclose(ax.lines[0].get_xdata(), x / 2)
        assert np.allclose(ax.lines[0].get_ydata(), 4.0)

        log = pd.DataFrame({'peak_wave': [4861.0], 'peak_flux': [10.0]}, index=['H1_4861A'])
        plotter._peak_plot(ax, log, ['H1_4861A'], 1, 2)
        assert len(ax.collections) == 1
        assert np.allclose(ax.collections[0].get_offsets(), [[4861.0, 5.0]])

        plt.close('all')

    def test_line_matching_plot(self, spec_basic):

        wave, flux = spec_basic.wave.data, spec_basic.flux.data

        # Bands inside the spectrum wavelength range
        bands = lime.load_frame(bands_file_address)
        w3, w4 = bands.w3.to_numpy() * (1 + REDSHIFT), bands.w4.to_numpy() * (1 + REDSHIFT)
        bands = bands.loc[(w3 > wave[5]) & (w4 < wave[-5])].copy()
        bands['signal_peak'] = np.searchsorted(wave, bands.w3.to_numpy() * (1 + REDSHIFT))

        fig, ax = plt.subplots()
        Plotter()._line_matching_plot(ax, bands, wave, flux, 1, REDSHIFT)

        assert len(ax.texts) == bands.index.size
        assert len(ax.collections) == 1

        plt.close('all')


class TestInspectionPlots:

    def test_mplcursors_legend(self, spec_fitted):

        line = spec_fitted.frame.index[0]
        legend = mplcursors_legend(line, spec_fitted.frame, 'Test label', spec_fitted.norm_flux,
                                   spec_fitted.units_wave, spec_fitted.units_flux)

        assert legend.startswith('Test label')
        for param in ('F_{intg}', 'F_{gauss}', 'v_{r}', r'\sigma_{g}'):
            assert param in legend

    def test_spec_plot(self, spec_fitted):

        fig, ax = plt.subplots()
        spec_plot(ax, spec_fitted, show_profiles=False)
        assert len(ax.lines) == 1

        fig, ax = plt.subplots()
        spec_plot(ax, spec_fitted, rest_frame=True, show_profiles=True)
        assert len(ax.lines) > 1

        plt.close('all')

    def test_redshift_key_evaluation(self, spec_basic):

        wave, flux = spec_basic.wave.data, spec_basic.flux.data
        z_arr = np.linspace(0, 0.1, 50)
        data_mask = np.zeros(wave.size, dtype=bool)
        data_mask[100:120] = True
        bands_arr = np.zeros(wave.size)
        bands_arr[100:120] = 1
        sum_arr = np.exp(-0.5 * np.square((z_arr - REDSHIFT) / 0.01))

        fig = plt.figure()
        redshift_key_evaluation(spec_basic, z_arr, data_mask, REDSHIFT, REDSHIFT, bands_arr, bands_arr, sum_arr,
                                sum_arr, in_fig=fig)

        # Spectrum, bands (twin axis) and redshift distribution
        assert len(fig.axes) == 3
        assert f'{REDSHIFT:0.3f}' in fig.axes[-1].get_title()
        assert fig.axes[-1].get_xlabel() == 'Redshift range'

        plt.close('all')

    def test_redshift_key_evaluation_display(self, spec_basic, mock_show):

        wave = spec_basic.wave.data
        z_arr = np.linspace(0, 0.1, 50)
        data_mask = np.zeros(wave.size, dtype=bool)
        data_mask[100:120] = True
        bands_arr = data_mask.astype(float)
        sum_arr = np.exp(-0.5 * np.square((z_arr - REDSHIFT) / 0.01))

        redshift_key_evaluation(spec_basic, z_arr, data_mask, REDSHIFT, REDSHIFT, bands_arr, bands_arr, sum_arr,
                                sum_arr, label='SHOC579')

        assert mock_show.call_count == 1

        plt.close('all')

    def test_continuum_calculation(self, spec_basic):

        wave, flux = spec_basic.wave.data, spec_basic.flux.data
        cont_fit = np.full(wave.size, np.nanmedian(flux))
        std = np.nanstd(flux)
        idcs_cont = np.abs(flux - cont_fit) < std

        # Intervals: inside the spectrum, beyond its red edge and between two pixels (not plotted)
        pixel = wave[11] - wave[10]
        exclude_intvls = np.array([[wave[100], wave[200]],
                                   [wave[-50], wave[-1] + 100],
                                   [wave[10] + 0.1 * pixel, wave[10] + 0.2 * pixel]])

        fig = plt.figure()
        spec_continuum_calculation(spec_basic, wave, flux, cont_fit, idcs_cont, cont_fit - std, cont_fit + std,
                                   None, exclude_intvls, in_fig=fig, log_scale=True)

        ax = fig.axes[0]
        labels = ax.get_legend_handles_labels()[1]
        assert 'Object spectrum' in labels
        assert 'Continuum' in labels
        assert 'Rejected peaks' in labels
        assert len(ax.patches) == 2
        assert ax.get_yscale() == 'log'

        # Smoothed spectrum label without excluded intervals
        fig = plt.figure()
        spec_continuum_calculation(spec_basic, wave, flux, cont_fit, idcs_cont, cont_fit - std, cont_fit + std,
                                   5, None, in_fig=fig)

        ax = fig.axes[0]
        assert 'Smoothed spectrum (5 pixels)' in ax.get_legend_handles_labels()[1]
        assert len(ax.patches) == 0
        assert ax.get_yscale() == 'linear'

        plt.close('all')

    def test_peak_calculation(self, spec_fitted, mock_show):

        wave = spec_fitted.wave.data

        bands = lime.load_frame(bands_file_address)
        idcs_peaks = np.searchsorted(wave, bands.w3.to_numpy() * (1 + REDSHIFT))
        bands['signal_peak'] = np.clip(idcs_peaks, 0, wave.size - 1)

        continuum = np.asarray(spec_fitted.cont)
        detect_limit = continuum + np.asarray(spec_fitted.cont_std)

        spec_peak_calculation(spec_fitted, bands, detect_limit, bands['signal_peak'].to_numpy(), continuum,
                              log_scale=True)

        ax = spec_fitted.plot.ax
        labels = ax.get_legend_handles_labels()[1]
        for label in ('Unmatched Peaks', 'Flux threshold', 'Matched lines', 'Continuum'):
            assert label in labels
        assert ax.get_yscale() == 'log'
        assert mock_show.call_count == 1

        plt.close('all')


class TestSpectrumOptions:

    def test_display(self, spec_basic, mock_show):

        spec_basic.plot.spectrum()
        assert mock_show.call_count == 1

        spec_basic.plot.spectrum(maximize=True)
        assert mock_show.call_count == 2

        plt.close('all')

    def test_new_figure_and_reset(self, spec_basic, mock_show):

        # in_fig=None generates a figure without displaying it
        spec_basic.plot.spectrum(in_fig=None, fig_cfg={'figure.figsize': (5, 3)})
        fig_0 = spec_basic.plot.fig

        assert np.allclose(fig_0.get_size_inches(), (5, 3))
        assert spec_basic.plot.ax is fig_0.axes[0]
        mock_show.assert_not_called()

        # A new plot closes the previous figure
        spec_basic.plot.spectrum(in_fig=None)
        assert spec_basic.plot.fig is not fig_0
        assert not figure_is_open(fig_0)

        # Manual reset
        fig_1 = spec_basic.plot.fig
        spec_basic.plot.reset_figure()
        assert spec_basic.plot.ax is None
        assert not figure_is_open(fig_1)

        plt.close('all')

    def test_show_and_save_fig(self, spec_basic, tmp_path, mock_show):

        spec_basic.plot.spectrum(in_fig=None)
        spec_basic.plot.show(block=False)
        mock_show.assert_called_once_with(block=False)

        out = tmp_path / 'spectrum_save_fig.png'
        spec_basic.plot.save_fig(out)
        assert out.is_file()

        plt.close('all')

    def test_rest_frame_data(self, spec_basic):

        spec_basic.plot.spectrum(in_fig=None, show_masks=False)
        x_obs, y_obs = (np.array(arr, dtype=float) for arr in spec_basic.plot.ax.lines[0].get_data())

        spec_basic.plot.spectrum(in_fig=None, show_masks=False, rest_frame=True)
        x_rest, y_rest = (np.array(arr, dtype=float) for arr in spec_basic.plot.ax.lines[0].get_data())

        # Wavelength divided and flux multiplied by (1 + z)
        with np.errstate(divide='ignore', invalid='ignore'):
            x_ratio, y_ratio = x_obs / x_rest, y_rest / y_obs

        assert np.allclose(x_ratio[np.isfinite(x_ratio)], 1 + REDSHIFT)
        assert np.allclose(y_ratio[np.isfinite(y_ratio)], 1 + REDSHIFT)

        plt.close('all')

    def test_label_and_ax_cfg(self, spec_basic):

        spec_basic.plot.spectrum(in_fig=None, label='SHOC579', ax_cfg={'title': 'Test title', 'xlabel': 'Test axis'})
        ax = spec_basic.plot.ax

        assert ax.get_title() == 'Test title'
        assert ax.get_xlabel() == 'Test axis'
        assert ax.get_legend() is not None
        assert 'SHOC579' in [text.get_text() for text in ax.get_legend().get_texts()]

        # No legend without a label
        spec_basic.plot.spectrum(in_fig=None)
        assert spec_basic.plot.ax.get_legend() is None

        plt.close('all')

    def test_line_list(self, spec_basic):

        line_list = ['H1_4861A', 'O3_5007A', 'H1_6563A']

        spec_basic.plot.spectrum(in_fig=None, line_list=line_list)
        assert [text.get_text() for text in spec_basic.plot.ax.texts] == line_list

        # Lines outside the wavelength range are excluded
        spec_basic.plot.spectrum(in_fig=None, line_list=line_list + ['H1_1216A'], rest_frame=True)
        assert [text.get_text() for text in spec_basic.plot.ax.texts] == line_list

        plt.close('all')

    def test_bands_dataframe(self, spec_basic):

        wave = spec_basic.wave.data

        # Bands table with line redshifts and peak detections
        bands = lime.load_frame(bands_file_address)
        bands['z_line'] = np.nan
        bands.loc[bands.index[0], 'z_line'] = REDSHIFT
        idcs_peaks = np.searchsorted(wave, bands.w3.to_numpy() * (1 + REDSHIFT))
        bands['signal_peak'] = np.clip(idcs_peaks, 0, wave.size - 1).astype(float)

        spec_basic.plot.spectrum(in_fig=None, bands=bands)
        ax = spec_basic.plot.ax

        assert 0 < len(ax.texts) <= bands.index.size
        assert 'Peaks' in [collection.get_label() for collection in ax.collections]

        plt.close('all')

    def test_show_components_no_detection(self, spec_basic):

        # Without a components detection only the spectrum is plotted
        spec = lime.Spectrum.from_file(file_address, 'sdss', redshift=REDSHIFT)
        spec.plot.spectrum(in_fig=None, show_components=True, show_masks=False)

        assert len(spec.plot.ax.lines) == 1

        plt.close('all')

    @aspect_required
    def test_show_components(self):

        spec = lime.Spectrum.from_file(file_address, 'sdss', redshift=REDSHIFT)
        spec.infer.components()
        spec.plot.spectrum(in_fig=None, show_components=True, show_masks=False)

        # Spectrum plus the detected components and their legend
        assert len(spec.plot.ax.lines) > 1
        assert spec.plot.ax.get_legend() is not None

        plt.close('all')

    def test_resolution(self):

        spec = lime.Spectrum.from_file(file_address, 'sdss', redshift=REDSHIFT)
        wave = spec.wave.data

        # Pixel width
        spec.res_power = None
        spec.plot.resolution(in_fig=None)
        ax = spec.plot.ax
        assert ax.get_title() == 'Pixel width vs wavelength'
        assert np.allclose(ax.lines[0].get_ydata()[:-1], np.diff(wave))

        # Scalar resolving power
        spec.res_power = 2000
        spec.plot.resolution(in_fig=None, rest_frame=True)
        ax = spec.plot.ax
        assert ax.get_title() == 'Spectral resolving power vs wavelength'
        assert np.allclose(ax.lines[0].get_ydata(), 2000)
        assert np.allclose(ax.lines[0].get_xdata(), wave / (1 + REDSHIFT))

        # Resolving power array
        spec.res_power = np.linspace(1500, 2500, wave.size)
        spec.plot.resolution(in_fig=None, ax_cfg={'xlabel': 'Test axis'})
        ax = spec.plot.ax
        assert np.allclose(ax.lines[0].get_ydata(), spec.res_power)
        assert ax.get_xlabel() == 'Test axis'

        plt.close('all')

    def test_resolution_display_and_save(self, spec_basic, tmp_path, mock_show):

        spec_basic.plot.resolution()
        assert mock_show.call_count == 1

        out = tmp_path / 'resolution.png'
        spec_basic.plot.resolution(fname=out)
        assert out.is_file()
        assert mock_show.call_count == 1

        plt.close('all')


class TestGridOptions:

    def test_grid_layout(self, spec_fitted):

        n_lines = lime_plot.unique_line_arr(spec_fitted.frame).size

        # One row by default
        fig = plt.figure()
        spec_fitted.plot.grid(in_fig=fig)
        assert len(fig.axes) == n_lines
        assert len(spec_fitted.plot.ax) == n_lines

        # More lines than columns
        fig = plt.figure()
        spec_fitted.plot.grid(in_fig=fig, n_cols=max(n_lines - 1, 1), fig_cfg={'figure.dpi': 100})
        assert len(fig.axes) == n_lines

        plt.close('all')

    def test_grid_display_and_save(self, spec_fitted, tmp_path, mock_show):

        spec_fitted.plot.grid()
        assert mock_show.call_count == 1

        out = tmp_path / 'grid.png'
        spec_fitted.plot.grid(fname=out, y_scale='linear', show_adjacent=True)
        assert out.is_file()
        assert mock_show.call_count == 1

        plt.close('all')

    def test_grid_empty_frame(self, spec_basic):

        fig = plt.figure()
        with patch.object(lime_plot._logger, 'info') as mock_info:
            spec_basic.plot.grid(in_fig=fig)

        assert mock_info.call_count == 1
        assert len(fig.axes) == 0

        plt.close('all')


class TestBandsOptions:

    def test_bands_axes(self, spec_fitted):

        # Line and residual axes
        fig = plt.figure()
        spec_fitted.plot.bands('H1_6563A', in_fig=fig)
        assert len(fig.axes) == 2
        assert fig.axes[0].get_legend() is not None

        # Line axis
        fig = plt.figure()
        spec_fitted.plot.bands('H1_6563A', in_fig=fig, show_profile=False)
        assert len(fig.axes) == 1

        plt.close('all')

    def test_bands_default_line(self, spec_fitted):

        # Without a label the last measured line is plotted
        fig = plt.figure()
        spec_fitted.plot.bands(in_fig=fig)

        assert len(fig.axes) == 2

        plt.close('all')

    def test_bands_display_and_save(self, spec_fitted, tmp_path, mock_show):

        spec_fitted.plot.bands('H1_6563A')
        assert mock_show.call_count == 1

        out = tmp_path / 'bands.png'
        spec_fitted.plot.bands('H1_6563A', fname=out, rest_frame=True, y_scale='linear')
        assert out.is_file()
        assert mock_show.call_count == 1

        plt.close('all')

    def test_bands_no_continuum_no_err(self):

        # Spectrum without a fitted continuum or an uncertainty array
        spec = lime.Spectrum.from_file(file_address, 'sdss', redshift=REDSHIFT)
        spec.fit.bands('H1_6563A', bands_file_address)
        spec._err_flux = None

        fig = plt.figure()
        with patch.object(lime_plot._logger, 'info') as mock_info:
            spec.plot.bands('H1_6563A', in_fig=fig, show_cont=True, show_err=True)

        assert mock_info.call_count == 1
        assert len(fig.axes) == 2

        plt.close('all')

    def test_bands_unmeasured_line(self, spec_basic):

        # A line without measurements is still plotted, without the profile and residual axis
        fig = plt.figure()
        with patch.object(lime_plot._logger, 'info') as mock_info:
            spec_basic.plot.bands('H1_6563A', in_fig=fig)

        assert mock_info.call_count == 0
        assert len(fig.axes) == 1

        plt.close('all')
        return


# class TestVelocityProfile:
#
#     def test_velocity_profile(self, spec_fitted):
#
#         fig = plt.figure()
#         spec_fitted.plot.velocity_profile('H1_6563A', in_fig=fig, ax_cfg={'title': 'Test title'})
#         ax = fig.axes[0]
#
#         assert ax.get_xlabel() == 'Velocity (Km/s)'
#         assert ax.get_title() == 'Test title'
#         assert ax.get_yscale() == 'linear'
#
#         # Percentiles plus the profile limits
#         texts = [text.get_text() for text in ax.texts]
#         for label in ('$v_{1}$', '$v_{50}$', '$v_{99}$', '$v_{0}$', '$v_{100}$'):
#             assert label in texts
#
#         # Width arrows
#         labels = ax.get_legend_handles_labels()[1]
#         assert any(label.startswith('$w_{80}') for label in labels)
#         assert any(label.startswith('$FWZI') for label in labels)
#
#         plt.close('all')
#
#     def test_velocity_profile_scale_and_default_line(self, spec_fitted):
#
#         fig = plt.figure()
#         spec_fitted.plot.velocity_profile('H1_6563A', in_fig=fig, y_scale='log')
#         assert fig.axes[0].get_yscale() == 'log'
#
#         # Without a label the last measured line is plotted
#         fig = plt.figure()
#         spec_fitted.plot.velocity_profile(in_fig=fig)
#         assert len(fig.axes) == 1
#
#         plt.close('all')
#
#     def test_velocity_profile_display_and_save(self, spec_fitted, tmp_path, mock_show):
#
#         spec_fitted.plot.velocity_profile('H1_6563A')
#         assert mock_show.call_count == 1
#
#         out = tmp_path / 'velocity.png'
#         spec_fitted.plot.velocity_profile('H1_6563A', fname=out)
#         assert out.is_file()
#         assert mock_show.call_count == 1
#
#         plt.close('all')


class TestSampleFigures:

    @staticmethod
    def _make_sample():
        frame = pd.DataFrame({'x': [1.0, 2.0, 3.0, 4.0], 'y': [10.0, 20.0, 30.0, 40.0],
                              'x_err': [0.1, 0.1, 0.1, 0.1], 'y_err': [1.0, 1.0, 1.0, 1.0],
                              'group': ['a', 'a', 'b', 'b']}, index=['obj_0', 'obj_1', 'obj_2', 'obj_3'])
        return SimpleNamespace(frame=frame, load_function=None)

    def test_properties_groups(self, tmp_path, mock_show):

        sample = self._make_sample()
        out = tmp_path / 'properties.png'

        fig, ax = plt.subplots()
        SampleFigures(sample).properties('x', 'y', observation_list=list(sample.frame.index), x_param_err='x_err',
                                         y_param_err='y_err', groups_variable='group', output_address=out,
                                         in_fig=fig, in_axis=ax, log_scale=True)

        assert out.is_file()
        assert ax.get_legend_handles_labels()[1] == ['a', 'b']
        assert ax.get_yscale() == 'log'
        mock_show.assert_not_called()

        plt.close('all')

    def test_properties_no_groups(self, mock_show):

        sample = self._make_sample()

        fig, ax = plt.subplots()
        SampleFigures(sample).properties('x', 'y', observation_list=['obj_0', 'obj_2'], in_fig=fig, in_axis=ax)

        assert len(ax.containers) == 1
        assert ax.get_legend() is None
        assert ax.get_yscale() == 'linear'
        assert mock_show.call_count == 1

        plt.close('all')

    def test_properties_errors(self):

        sample = self._make_sample()

        # Parameter missing from the sample frame
        with pytest.raises(LiMe_Error):
            SampleFigures(sample).properties('x', 'not_a_column', observation_list=list(sample.frame.index))

        # Empty selection
        with patch.object(lime_plot._logger, 'info') as mock_info:
            SampleFigures(sample).properties('x', 'y', observation_list=[])
        assert mock_info.call_count == 1

    def test_spectra_no_load_function(self):

        with patch.object(lime_plot._logger, 'info') as mock_info:
            SampleFigures(self._make_sample()).spectra()

        assert mock_info.call_count == 1