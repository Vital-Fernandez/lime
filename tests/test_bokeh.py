import numpy as np
import lime
import pytest
from pathlib import Path

# Skip the whole file if bokeh is not available
pytest.importorskip('bokeh')

from bokeh import models
from bokeh.plotting import figure
from bokeh.layouts import gridplot
from bokeh.models import BoxAnnotation, Legend, LogScale, LinearScale, WheelZoomTool, PanTool

from lime.plotting import bokeh_plots
from lime.plotting.bokeh_plots import (BokehFigures, ensure_list, extract_figures, update_bokeh_figure,
                                       save_close_fig_swicth, bands_filling_bokeh)

try:
    import aspect
    aspect_check = True
except ImportError:
    aspect_check = False

aspect_required = pytest.mark.skipif(not aspect_check, reason='ASPECT is not installed')


def glyph_renderers(fig, glyph_type):
    return [r for r in fig.renderers if isinstance(r.glyph, glyph_type)]


def renderer_data(renderer, column):
    return np.asarray(renderer.data_source.data[column], dtype=float)


@pytest.fixture
def show_calls(monkeypatch):

    # Replace the bokeh display (it opens a browser tab) by a counter
    calls = []
    monkeypatch.setattr(bokeh_plots, 'show', lambda fig_obj, *args, **kwargs: calls.append(fig_obj))

    return calls


# Data for the tests
baseline_folder = Path(__file__).parent / 'baseline'
file_address = baseline_folder/'SHOC579_MANGA38-35.txt'
conf_file_address = baseline_folder/'lime_tests.toml'
bands_file_address = baseline_folder/'SHOC579_MANGA38-35_bands.txt'

redshift = 0.0475
norm_flux = 1e-17
cfg = lime.load_cfg(conf_file_address)

wave_array, flux_array, err_array, pixel_mask = np.loadtxt(file_address, unpack=True)

spec = lime.Spectrum(wave_array, flux_array, err_array, redshift=redshift, norm_flux=norm_flux,
                     pixel_mask=pixel_mask, id_label='SHOC579_Manga38-35')
spec.fit.continuum(degree_list=[3, 6, 6], emis_threshold=[3, 2, 1.5])
spec.fit.frame(bands_file_address, cfg, obj_cfg_prefix='38-35', line_list=['H1_6563A_b'])

bokeh_spec = BokehFigures(spec)


class TestBokehTools:

    def test_ensure_list(self):

        assert ensure_list(1) == [1]
        assert ensure_list('abc') == ['abc']
        assert ensure_list(None) == [None]
        assert ensure_list([1, 2]) == [1, 2]

        return

    def test_extract_figures(self):

        fig1, fig2 = figure(), figure()

        assert extract_figures(fig1) == [fig1]
        assert extract_figures([fig1, None, fig2]) == [fig1, fig2]
        assert extract_figures(None) == []

        # Grid layout
        grid = gridplot([[fig1, fig2]])
        fig_list = extract_figures(grid)
        assert len(fig_list) == 2
        assert (fig1 in fig_list) and (fig2 in fig_list)

        return

    def test_update_figure_single_values(self):

        fig = figure(width=600, height=600)
        output = update_bokeh_figure(fig, {'width': 900, 'height': 400, 'tools': 'pan,reset', 'min_border': None})

        assert output is fig
        assert fig.width == 900
        assert fig.height == 400

        # None entries do not overwrite the figure values
        update_bokeh_figure(fig, {'width': None})
        assert fig.width == 900

        return

    def test_update_figure_dict_values(self):

        fig = figure()
        fig_cfg = {'xaxis': {'axis_label_text_font_size': '14pt'},
                   'yaxis': {'axis_label_text_font_size': '12pt'},
                   'title': {'text': 'Test title', 'text_font_size': '16pt'},
                   'xgrid': {'grid_line_color': 'None'},
                   'ygrid': {'grid_line_alpha': 0.3}}
        update_bokeh_figure(fig, fig_cfg)

        assert fig.xaxis[0].axis_label_text_font_size == '14pt'
        assert fig.yaxis[0].axis_label_text_font_size == '12pt'
        assert fig.title.text == 'Test title'
        assert fig.title.text_font_size == '16pt'
        assert fig.xgrid[0].grid_line_color is None
        assert fig.ygrid[0].grid_line_alpha == 0.3

        return

    def test_update_figure_active_tools(self):

        fig = figure(tools='pan,wheel_zoom,box_zoom,reset,save')
        update_bokeh_figure(fig, {'active_scroll': 'WheelZoomTool', 'active_drag': 'PanTool'})

        assert isinstance(fig.toolbar.active_scroll, WheelZoomTool)
        assert isinstance(fig.toolbar.active_drag, PanTool)

        return

    def test_update_figure_list_and_grid(self):

        fig_cfg = {'width': 300, 'xaxis': {'axis_label_text_font_size': '14pt'}, 'title': {'text': 'Panel'}}

        # List of figures (empty entries are ignored)
        fig1, fig2 = figure(), figure()
        update_bokeh_figure([fig1, None, fig2], fig_cfg)
        for fig in (fig1, fig2):
            assert fig.width == 300
            assert fig.xaxis[0].axis_label_text_font_size == '14pt'
            assert fig.title.text == 'Panel'

        # Grid layout
        fig3, fig4 = figure(), figure()
        update_bokeh_figure(gridplot([[fig3, fig4]]), fig_cfg)
        for fig in (fig3, fig4):
            assert fig.width == 300
            assert fig.xaxis[0].axis_label_text_font_size == '14pt'
            assert fig.title.text == 'Panel'

        return

    def test_save_close_switch(self, tmp_path, show_calls):

        fig = figure()
        fig.line([1, 2, 3], [1, 2, 3])
        file_html = tmp_path / 'test_switch.html'

        # No display or saving for figures provided by the user
        save_close_fig_swicth(file_html, fig, False)
        save_close_fig_swicth(None, fig, False)
        assert not file_html.is_file()
        assert len(show_calls) == 0

        # Not recognized output address
        save_close_fig_swicth(5, fig, True)
        assert len(show_calls) == 0

        # Display
        save_close_fig_swicth(None, fig, True)
        assert show_calls == [fig]

        # Save to a file (Path and string)
        save_close_fig_swicth(file_html, fig, True)
        assert file_html.is_file()
        assert '<html' in file_html.read_text()

        file_str = str(tmp_path / 'test_switch_str.html')
        save_close_fig_swicth(file_str, fig, True)
        assert Path(file_str).is_file()

        return

    def test_bands_filling(self):

        x = np.arange(100, dtype=float)
        y = 1 + np.exp(-0.5 * np.square((x - 50) / 3))
        idcs_mask = np.array([10, 20, 40, 60, 80, 90])
        z_corr = 2

        # Line band only
        fig = figure()
        bands_filling_bokeh(fig, x, y, z_corr, idcs_mask, 'H1_6563A', exclude_continua=True)
        areas = glyph_renderers(fig, models.VAreaStep)
        assert len(areas) == 1
        assert np.allclose(renderer_data(areas[0], 'x'), x[40:60] / z_corr)
        assert np.allclose(renderer_data(areas[0], 'y2'), y[40:60] * z_corr)

        # Line and continua bands
        fig = figure()
        bands_filling_bokeh(fig, x, y, z_corr, idcs_mask, 'H1_6563A', exclude_continua=False)
        areas = glyph_renderers(fig, models.VAreaStep)
        assert len(areas) == 3
        assert np.allclose(renderer_data(areas[1], 'x'), x[10:20] / z_corr)
        assert np.allclose(renderer_data(areas[2], 'x'), x[80:90] / z_corr)

        # Interval with less than two pixels
        fig = figure()
        bands_filling_bokeh(fig, x, y, z_corr, np.array([10, 10, 10, 11, 11, 11]), 'H1_6563A')
        assert len(fig.renderers) == 0

        return


class TestBokehSpectrum:

    def test_spectrum(self):

        bokeh_spec.spectrum(in_fig=None, include_fits=False)
        fig = bokeh_spec.fig

        # Only the spectrum step
        assert len(fig.renderers) == 1
        steps = glyph_renderers(fig, models.Step)
        assert len(steps) == 1
        assert steps[0].glyph.mode == 'center'
        assert renderer_data(steps[0], 'x').size > 0
        assert renderer_data(steps[0], 'x').size == renderer_data(steps[0], 'y').size

        # Axes labels and scale
        assert fig.xaxis[0].axis_label is not None
        assert fig.yaxis[0].axis_label is not None
        assert isinstance(fig.y_scale, LinearScale)

        return

    def test_spectrum_rest_frame(self):

        bokeh_spec.spectrum(in_fig=None, include_fits=False)
        step_obs = glyph_renderers(bokeh_spec.fig, models.Step)[0]
        x_obs, y_obs = renderer_data(step_obs, 'x'), renderer_data(step_obs, 'y')

        bokeh_spec.spectrum(in_fig=None, include_fits=False, rest_frame=True)
        step_rest = glyph_renderers(bokeh_spec.fig, models.Step)[0]
        x_rest, y_rest = renderer_data(step_rest, 'x'), renderer_data(step_rest, 'y')

        # Wavelength divided and flux multiplied by (1 + z)
        with np.errstate(divide='ignore', invalid='ignore'):
            x_ratio, y_ratio = x_obs / x_rest, y_rest / y_obs

        assert np.allclose(x_ratio[np.isfinite(x_ratio)], 1 + redshift)
        assert np.allclose(y_ratio[np.isfinite(y_ratio)], 1 + redshift)

        return

    def test_spectrum_log_scale(self):

        bokeh_spec.spectrum(in_fig=None, include_fits=False, log_scale=True)

        assert isinstance(bokeh_spec.fig.y_scale, LogScale)

        return

    def test_spectrum_in_fig(self, show_calls):

        # The input figure is used and not displayed
        fig = figure()
        bokeh_spec.spectrum(in_fig=fig, include_fits=False)

        assert bokeh_spec.fig is fig
        assert len(glyph_renderers(fig, models.Step)) == 1
        assert len(show_calls) == 0

        return

    def test_spectrum_fig_cfg(self):

        bokeh_spec.spectrum(in_fig=None, include_fits=False, fig_cfg={'width': 950, 'height': 420,
                                                                      'title': {'text': 'SHOC579'}})

        assert bokeh_spec.fig.width == 950
        assert bokeh_spec.fig.height == 420
        assert bokeh_spec.fig.title.text == 'SHOC579'

        return

    def test_spectrum_show_err(self):

        bokeh_spec.spectrum(in_fig=None, include_fits=False)
        assert len(glyph_renderers(bokeh_spec.fig, models.VAreaStep)) == 0

        bokeh_spec.spectrum(in_fig=None, include_fits=False, show_err=True)
        assert len(glyph_renderers(bokeh_spec.fig, models.VAreaStep)) == 1

        return

    def test_spectrum_show_cont(self):

        bokeh_spec.spectrum(in_fig=None, include_fits=False, show_cont=True)
        fig = bokeh_spec.fig

        # Continuum line and its uncertainty area
        cont_lines = glyph_renderers(fig, models.Line)
        assert len(cont_lines) == 1
        assert len(glyph_renderers(fig, models.VArea)) == 1
        assert renderer_data(cont_lines[0], 'x').size == renderer_data(glyph_renderers(fig, models.Step)[0], 'x').size

        # Legend is hidden without an input label
        assert len(fig.legend) == 1
        assert fig.legend[0].visible is False

        bokeh_spec.spectrum(in_fig=None, include_fits=False, show_cont=True, label='SHOC579')
        assert bokeh_spec.fig.legend[0].visible is True

        return

    def test_spectrum_with_fits(self):

        bokeh_spec.spectrum(in_fig=None, include_fits=False)
        assert len(glyph_renderers(bokeh_spec.fig, models.Line)) == 0

        # At least the continuum and the profile per measured line
        bokeh_spec.spectrum(in_fig=None, include_fits=True)
        assert len(glyph_renderers(bokeh_spec.fig, models.Line)) >= 2 * spec.frame.index.size

        return

    def test_spectrum_bands(self):

        bands = lime.load_frame(bands_file_address)

        bokeh_spec.spectrum(in_fig=None, include_fits=False)
        assert len([item for item in bokeh_spec.fig.center if isinstance(item, BoxAnnotation)]) == 0

        # File address and dataframe inputs
        for bands_input in (bands_file_address, bands):
            bokeh_spec.spectrum(in_fig=None, include_fits=False, bands=bands_input)
            boxes = [item for item in bokeh_spec.fig.center if isinstance(item, BoxAnnotation)]

            assert 0 < len(boxes) <= bands.index.size
            for box in boxes:
                assert box.left < box.right

        return

    def test_spectrum_line_list(self):

        line_list = ['H1_4861A', 'O3_5007A', 'H1_6563A']
        bokeh_spec.spectrum(in_fig=None, include_fits=False, line_list=line_list)
        fig = bokeh_spec.fig

        # Two segments and a label per line
        assert len(glyph_renderers(fig, models.Segment)) == 2 * len(line_list)

        texts = glyph_renderers(fig, models.Text)
        assert len(texts) == len(line_list)
        assert [str(text.data_source.data['text'][0]) for text in texts] == line_list

        # Lines outside the wavelength range are excluded
        bokeh_spec.spectrum(in_fig=None, include_fits=False, line_list=line_list + ['H1_1216A'])
        assert len(glyph_renderers(bokeh_spec.fig, models.Text)) == len(line_list)

        return

    def test_spectrum_display(self, show_calls):

        bokeh_spec.spectrum(include_fits=False)

        assert show_calls == [bokeh_spec.fig]

        return

    def test_spectrum_save(self, tmp_path, show_calls):

        file_html = tmp_path / 'test_spectrum_bokeh.html'
        bokeh_spec.spectrum(fname=file_html, include_fits=False)

        assert file_html.is_file()
        assert '<html' in file_html.read_text()

        return

    def test_bokeh_not_installed(self, monkeypatch):

        monkeypatch.setattr(bokeh_plots, 'bokeh_check', False)

        with pytest.raises(bokeh_plots.LiMe_Error):
            bokeh_spec.spectrum(in_fig=None)

        return

    @aspect_required
    def test_spectrum_show_comps(self):

        spec_comps = lime.Spectrum(wave_array, flux_array, err_array, redshift=redshift, norm_flux=norm_flux,
                                   pixel_mask=pixel_mask, id_label='SHOC579_Manga38-35')
        bokeh_comps = BokehFigures(spec_comps)

        # No components detection
        bokeh_comps.spectrum(in_fig=None, show_comps=True)
        assert len(bokeh_comps.fig.legend) == 0
        assert len(glyph_renderers(bokeh_comps.fig, models.Step)) == 1

        # Components detection
        spec_comps.infer.components()
        bokeh_comps.spectrum(in_fig=None, show_comps=True)

        # Spectrum plus the detected components and legends for the shapes and the confidence
        assert len(glyph_renderers(bokeh_comps.fig, models.Step)) > 1
        legends = bokeh_comps.fig.legend
        assert len(legends) == 2
        assert all(isinstance(legend, Legend) for legend in legends)

        return

    def test_update_figure_active_tools_duplicated(self):

        fig = figure(tools='xpan,pan,xwheel_zoom,ywheel_zoom,reset')
        update_bokeh_figure(fig, {'active_scroll': 'WheelZoomTool', 'active_drag': 'PanTool'})

        assert isinstance(fig.toolbar.active_scroll, WheelZoomTool)
        assert isinstance(fig.toolbar.active_drag, PanTool)

        # Tool not in the toolbar
        update_bokeh_figure(fig, {'active_tap': 'TapTool'})
        assert fig.toolbar.active_tap is None

        return