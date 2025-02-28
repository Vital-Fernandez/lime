from lime.plotting.format import theme
from bokeh import plotting
from bokeh import models
from bokeh.plotting import figure, output_file, save, show
from bokeh.models import HoverTool, ColumnDataSource, Legend, LegendItem, LogScale
from .plots import frame_mask_switch



def update_bokeh_figure(figure_obj, config_dict):

    # Set general figure properties
    for key, value in config_dict.items():

        # Dictionary based entries
        if isinstance(value, dict):
            match key:
                case "xaxis":
                    for axis in figure_obj.xaxis:  # Update all x-axes
                        for attr, val in value.items():
                            setattr(axis, attr, val)

                case "yaxis":
                    for axis in figure_obj.yaxis:  # Update all y-axes
                        for attr, val in value.items():
                            setattr(axis, attr, val)

                case "title":
                    for attr, val in value.items():
                        setattr(figure_obj.title, attr, val)

                case "xgrid":
                    for grid in figure_obj.xgrid:  # Update all x-grids
                        for attr, val in value.items():
                            val = None if val == 'None' else val
                            setattr(grid, attr, val)

                case "ygrid":
                    for grid in figure_obj.ygrid:  # Update all y-grids
                        for attr, val in value.items():
                            val = None if val == 'None' else val
                            setattr(grid, attr, val)

        # Single value entries
        else:
            if key != 'tools':
                setattr(figure_obj, key, value)

    return figure_obj


class BokehFigures:

    def __init__(self, spectrum):

        # Lime spectrum object with the scientific data
        self._spec = spectrum

        return

    def spectrum(self, output_address=None, label=None, bands=None, rest_frame=False, log_scale=False,
                 include_fits=True, include_cont=False, in_fig=None, fig_cfg=None, ax_cfg=None, maximize=False,
                 detection_band=None, show_masks=True, show_categories=False):

        # Display check for the user figures
        display_check = True if in_fig is None else False

        # Set figure format with the user inputs overwriting the default conf
        legend_check = True if label is not None else False

        # Adjust the default theme
        PLT_CONF = theme.fig_defaults(fig_cfg, plot_lib='bokeh')
        AXES_CONF = theme.ax_defaults(ax_cfg, self._spec.units_wave, self._spec.units_flux, self._spec.norm_flux,
                                      plotting_library='bokeh')

        # Create figure with default tools if not provided
        p = figure(tools = PLT_CONF.get('tools', 'pan,wheel_zoom,box_zoom,reset,save'))

        # p.background_fill_color = theme.colors['bg']
        # p.border_fill_color = theme.colors['bg']
        # p.outline_line_color = theme.colors['fg']
        #
        # p.xaxis.axis_label_text_color = theme.colors['fg']
        # p.yaxis.axis_label_text_color = theme.colors['fg']
        # p.xaxis.major_label_text_color = theme.colors['fg']
        # p.yaxis.major_label_text_color = theme.colors['fg']
        #
        # p.xgrid.grid_line_color = None
        # p.ygrid.grid_line_color = None
        # p.xaxis.axis_line_color = theme.colors['fg']
        # p.yaxis.axis_line_color = theme.colors['fg']
        # p.xaxis.major_tick_line_color = theme.colors['fg']
        # p.yaxis.major_tick_line_color = theme.colors['fg']
        # p.yaxis.minor_tick_line_color = theme.colors['fg']
        # p.xaxis.minor_tick_line_color = theme.colors['fg']
        #
        # p.title.text_color = theme.colors['fg']

        # p = figure(title="Inverted Spectrum", background_fill_color="black")
        #
        # # ✅ Fix axis label colors
        # for axis in p.xaxis + p.yaxis:
        #     axis.axis_label_text_color = "white"
        #     axis.major_label_text_color = "white"
        #     axis.axis_line_color = "white"
        #     axis.major_tick_line_color = "white"
        #     axis.minor_tick_line_color = "white"
        #
        # # ✅ Fix grid color (optional)
        # for grid in p.grid:
        #     grid.grid_line_color = None  # Removes grid lines
        #
        # show(p)

        ## r'Wavelength ($\mathrm{\mathring{A}}$)'


        wave_plot, flux_plot, z_corr, idcs_mask = frame_mask_switch(self._spec.wave, self._spec.flux,
                                                            self._spec.redshift, rest_frame)

        # Spectrum data source
        source = ColumnDataSource(data={"x": wave_plot / z_corr, "y": flux_plot * z_corr})
        p.step("x", "y", source=source, color=theme.colors['fg'], line_width=1)

        # Plot labels
        p.xaxis.axis_label = AXES_CONF['xlabel']
        p.yaxis.axis_label = AXES_CONF['ylabel']

        # Log scale if requested
        if log_scale:
            p.y_range = LogScale()

        # Adjust the format of the plot
        update_bokeh_figure(p, PLT_CONF)

        # Save or display the plot
        if output_address:
            # output_file(output_address)
            save(p)
        else:
            # output_notebook()
            show(p)

        return



# from bokeh.plotting import figure, output_file, save, show
# from bokeh.models import HoverTool, ColumnDataSource, Legend, LegendItem, LogScale
# from bokeh.io import output_notebook
# import numpy as np
# import pandas as pd
#
# def spectrum_bokeh(self, output_address=None, label="Observed spectrum", bands=None, rest_frame=False, log_scale=False,
#                    include_fits=True, include_cont=False, detection_band=None, show_masks=True, show_categories=False):
#
#     """
#     This function plots the spectrum flux versus wavelength using Bokeh.
#     """
#
#     # Prepare wavelength and flux values
#     wave_plot, flux_plot, z_corr, idcs_mask = frame_mask_switch(self._spec.wave, self._spec.flux,
#                                                                 self._spec.redshift, rest_frame)
#
#     # Create a Bokeh figure
#     p = figure(title="Spectrum Flux vs. Wavelength", width=900, height=500,
#                x_axis_label=f"Wavelength ({self._spec.units_wave})",
#                y_axis_label=f"Flux ({self._spec.units_flux})",
#                tools="pan,wheel_zoom,box_zoom,reset,save",
#                tooltips=[("Wavelength", "@x"), ("Flux", "@y")])
#
#     # Log scale if requested
#     if log_scale:
#         p.y_range = LogScale()
#
#     # Spectrum data source
#     source = ColumnDataSource(data={"x": wave_plot / z_corr, "y": flux_plot * z_corr})
#     p.step("x", "y", source=source, legend_label=label, color="black", line_width=1.5)
#
#     # Add bands (if provided)
#     if bands is not None:
#         bands = check_file_dataframe(bands, pd.DataFrame)
#         w3_obs, w4_obs = bands.w3.to_numpy() * (1 + self._spec.redshift), bands.w4.to_numpy() * (1 + self._spec.redshift)
#         idcs_valid = (w3_obs > wave_plot[0]) & (w4_obs < wave_plot[-1])
#
#         for _, row in bands.loc[idcs_valid].iterrows():
#             p.line([row.w3 / z_corr, row.w4 / z_corr], [np.median(flux_plot), np.median(flux_plot)],
#                    line_color="blue", line_width=2, legend_label="Bands")
#
#     # Plot fitted profiles (if requested)
#     if include_fits and self._spec.frame is not None:
#         for line_label in self._spec.frame.index.values:
#             line_i = Line.from_log(line_label, self._spec.frame)
#             fit_source = ColumnDataSource(data={"x": line_i.wave / z_corr, "y": line_i.flux * z_corr})
#             p.line("x", "y", source=fit_source, legend_label=line_label, line_color="red", line_dash="dashed")
#
#     # Plot continuum (if requested)
#     if include_cont and self._spec.cont is not None:
#         cont_source = ColumnDataSource(data={"x": wave_plot / z_corr, "y": self._spec.cont * z_corr})
#         p.line("x", "y", source=cont_source, legend_label="Continuum", line_color="green", line_dash="dotdash")
#
#     # Show masks (if requested)
#     if show_masks:
#         mask_source = ColumnDataSource(data={"x": wave_plot[idcs_mask] / z_corr, "y": flux_plot[idcs_mask] * z_corr})
#         p.scatter("x", "y", source=mask_source, legend_label="Masked pixels", color="red", marker="x")
#
#     # Show detection bands (if requested)
#     if detection_band is not None:
#         detec_obj = getattr(self._spec.inference, detection_band)
#         if detec_obj.confidence is not None:
#             for conf_level in np.arange(0.3, 1.1, 0.1):
#                 idcs = detec_obj(conf_level * 100)
#                 conf_source = ColumnDataSource(data={"x": wave_plot[idcs] / z_corr, "y": flux_plot[idcs] * z_corr})
#                 p.step("x", "y", source=conf_source, legend_label=f"> {int(conf_level*100)}% confidence",
#                        line_width=1, line_color="purple")
#
#     # Show category-based plotting (if requested)
#     if show_categories and self._spec.features.pred_arr is not None:
#         categories = np.sort(np.unique(self._spec.features.pred_arr))
#         for category in categories:
#             if category != 0:
#                 feature_name = self._spec.features.model.number_feature_dict[category]
#                 idcs_feature = self._spec.features.pred_arr == category
#                 category_source = ColumnDataSource(data={"x": wave_plot[idcs_feature] / z_corr,
#                                                          "y": flux_plot[idcs_feature] * z_corr})
#                 p.step("x", "y", source=category_source, legend_label=feature_name, line_color="orange")
#
#     # Save or display the plot
#     if output_address:
#         output_file(output_address)
#         save(p)
#     else:
#         output_notebook()
#         show(p)
#
#     return p
