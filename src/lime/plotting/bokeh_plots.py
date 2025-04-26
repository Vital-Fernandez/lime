import logging
import numpy as np

from lime.transitions import label_decomposition, Line
from lime.plotting.format import theme
from lime.plotting.plots import frame_mask_switch
from lime.plotting.utils import color_selector
from lime.io import check_file_dataframe
from lime.plotting.utils import parse_bands_arguments
from lime.fitting.lines import c_KMpS, profiles_computation, linear_continuum_computation

try:
    from bokeh import plotting
    from bokeh import models
    from bokeh.plotting import figure, output_file, save, show
    from bokeh.models import ColumnDataSource, Legend, LegendItem, LogScale
    from bokeh.models import BoxAnnotation
    from bokeh.models import WheelZoomTool, PanTool, HoverTool

    bokeh_check = True
except ImportError:
    bokeh_check = False

try:
    import aspect
    aspect_check = True
except:
    aspect_check = False



_logger = logging.getLogger('LiMe')

category_conf_styles = {0: 'dotted',
                        1: 'dashed',
                        2: 'solid'}

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

    # Set zoom and pan as active
    figure_obj.toolbar.active_scroll = figure_obj.select_one(WheelZoomTool)  # Activate zoom wheel
    figure_obj.toolbar.active_drag = figure_obj.select_one(PanTool)  # Activate pan tool

    return figure_obj


def bokeh_bands(fig, bands, x, y, z_corr, redshift):

    # Open the bands file the bands
    match_log = check_file_dataframe(bands)

    # Crop the selection for the observation wavelength range
    w3_obs = match_log.w3.to_numpy() * (1 + redshift)
    w4_obs = match_log.w4.to_numpy() * (1 + redshift)
    idcs_valid = (w3_obs > x[0]) & (w4_obs < x[-1])

    # Plot the detected line peaks
    if 'signal_peak' in match_log.columns:
        idcs_peaks = match_log.loc[idcs_valid, 'signal_peak'].values.astype(int)
        fig.scatter(x[idcs_peaks] / z_corr, y[idcs_peaks] * z_corr, size=5, line_color=theme.colors['peak'], fill_color=None)

    # Get the line labels and the bands labels for the lines
    wave_array, latex = label_decomposition(match_log.index.values, params_list=('wavelength', 'latex_label'))
    idcs_band_limits = np.searchsorted(x, np.array([w3_obs / z_corr, w4_obs / z_corr]))

    # Loop through the detections and plot the names
    for i in np.arange(latex.size):
        if idcs_band_limits[0, i] != idcs_band_limits[0, i]:  # Y limit for the label check if same pixel
            max_region = np.max(y[idcs_band_limits[0, i]:idcs_band_limits[0, i]])
        else:
            max_region = y[idcs_band_limits[0, i]]

        label = 'Matched line' if i == 0 else '_'
        fig.add_layout(BoxAnnotation(left=w3_obs[i]/z_corr, right=w4_obs[i]/z_corr, fill_alpha=0.3, fill_color=theme.colors['match_line']))
        # axis.text(wave_array[i] * (1 + redshift) / z_corr, max_region * 0.9 * z_corr, latex[i], rotation=270)

    return


def bands_filling_bokeh(fig, x, y, z_corr, idcs_mask, label, exclude_continua=False, color_dict=theme.colors):

    # Security check for low selection
    if len(y[idcs_mask[0]:idcs_mask[5]]) > 1:

        # Lower limit for the filled region
        low_lim = np.min(y[idcs_mask[0]:idcs_mask[5]])
        low_lim = 0 if np.isnan(low_lim) else low_lim

        # Central bands
        fig.varea_step(x=x[idcs_mask[2]:idcs_mask[3]]/z_corr, y1=low_lim*z_corr, y2=y[idcs_mask[2]:idcs_mask[3]]*z_corr,
                        step_mode="center", fill_alpha=0.25, color=color_dict['line_band'])

        # Continua bands exclusion
        if exclude_continua is False:
            fig.varea_step(x=x[idcs_mask[0]:idcs_mask[1]]/z_corr,
                           y1=low_lim*z_corr,
                           y2=y[idcs_mask[0]:idcs_mask[1]]*z_corr,
                           step_mode="center", fill_alpha=0.25, color=color_dict['cont_band'])

            fig.varea_step(x=x[idcs_mask[4]:idcs_mask[5]]/z_corr,
                           y1=low_lim*z_corr,
                           y2=y[idcs_mask[4]:idcs_mask[5]]*z_corr,
                           step_mode="center", fill_alpha=0.25, color=color_dict['cont_band'])
    else:
        _logger.warning(f'The {label} band plot interval contains less than 1 pixel')

    return


def profile_bokeh(fig, line, z_cor, log, redshift, norm_flux):

    # Check if blended line or Single/merged
    if line.blended_check:
        if line.list_comps:
            idx_line = line.list_comps.index(line.label)
            n_comps = len(line.list_comps)
    else:
        idx_line = 0
        n_comps = 1

    # Compute the profile(s)
    wave_array, flux_array = profiles_computation(line.list_comps, log, (1 + redshift), line._p_shape)
    wave_array, cont_array = linear_continuum_computation(line.list_comps, log, (1 + redshift))

    wave_i = wave_array[:, 0]
    cont_i = cont_array[:, 0]
    flux_i = flux_array[:, idx_line]

    # Plot a continuum (only one per line)
    if idx_line == 0:
        cont_format = dict(line_color=theme.colors['cont'], line_dash="dashed", line_width=2)
        fig.line(wave_array[:, 0]/z_cor, cont_array[:, 0] * z_cor/norm_flux, **cont_format)

    # Plot combined gaussian profile if blended
    if (idx_line == 0) and (n_comps > 1):
        comb_array = (flux_array.sum(axis=1) + cont_i) * z_cor / norm_flux
        line_format = color_selector(None, line.observations, 0, 1, scale_dict=theme.plt, colors_dict=theme.colors,
                                     library='bokeh')
        fig.line(wave_i/z_cor, comb_array, **line_format)

    # Gaussian component plot
    single_array = (flux_i + cont_i) * z_cor / norm_flux
    line_format = color_selector(line.label, line.observations, idx_line, n_comps, scale_dict=theme.plt, colors_dict=theme.colors,
                                 library='bokeh')
    line_single = fig.line(wave_i/z_cor, single_array, **line_format)

    return line_single


class BokehFigures:

    def __init__(self, spectrum):

        # Lime spectrum object with the scientific data
        self._spec = spectrum

        return

    def bands(self, label, output_address=None, ref_bands=None, include_fits=True, rest_frame=False, log_scale=True, fig_cfg=None,
              ax_cfg=None, return_fig=False):


        # Unpack variables
        log, norm_flux, redshift = self._spec.frame, self._spec.norm_flux, self._spec.redshift
        units_wave, units_flux = self._spec.units_wave, self._spec.units_flux

        # Set figure format with the user inputs overwriting the default conf
        legend_check = True if label is not None else False

        # Check which line should be plotted
        line = parse_bands_arguments(label, log, ref_bands, norm_flux)

        # Proceed to plot
        if line is not None:

            # Guess whether we need both lines
            include_fits = include_fits and (line.profile_flux is not None)

            # Adjust the default theme
            PLT_CONF = theme.fig_defaults(fig_cfg, plot_lib='bokeh')
            AXES_CONF = theme.ax_defaults(ax_cfg, self._spec.units_wave, self._spec.units_flux, self._spec.norm_flux,
                                          plotting_library='bokeh')

            # Reference _frame for the plot
            wave_plot, flux_plot, z_corr, idcs_mask = frame_mask_switch(self._spec.wave, self._spec.flux,
                                                                        redshift, rest_frame)
            err_plot = self._spec.err_flux

            # Establish the limits for the line spectrum plot
            idcs_bands = line.index_bands(self._spec.wave, self._spec.redshift, just_band_edges=True)

            # Set the scale
            scale_str = 'log' if log_scale and (PLT_CONF.get('y_axis_type') is None) else 'linear'

            # Create figure with default utils if not provided
            fig = figure(tools=PLT_CONF.get('tools', "pan,wheel_zoom,box_zoom,reset,save"), y_axis_type=scale_str)

            # # Create figure with default utils if not provided
            # fig = figure(tools=PLT_CONF.get('tools', "pan,wheel_zoom,box_zoom,reset,save"))

            # Spectrum data source
            source = ColumnDataSource(data={"x": wave_plot[idcs_bands[0]:idcs_bands[5]] / z_corr,
                                            "y": flux_plot[idcs_bands[0]:idcs_bands[5]] * z_corr})
            fig.step("x", "y", source=source, color=theme.colors['fg'], line_width=1, mode='center')

            # Fille the bands
            bands_filling_bokeh(fig, wave_plot, flux_plot, z_corr, idcs_bands, line)

            # Plot labels
            fig.xaxis.axis_label = AXES_CONF['xlabel']
            fig.yaxis.axis_label = AXES_CONF['ylabel']

            # Adjust the format of the plot
            update_bokeh_figure(fig, PLT_CONF)

            # Save or display the plot
            if return_fig:
                return fig

            elif output_address is not None:
                save(fig, filename=output_address)

            else:
                # output_notebook()
                show(fig)

        return

    def grid(self, output_address=None, rest_frame=True, log_scale=False, n_cols=6, n_rows=None, col_row_scale=(2, 1.5),
             include_fits=True, in_fig=None, fig_cfg=None, ax_cfg=None, maximize=False):

        return

    def spectrum(self, output_address=None, label=None, bands=None, rest_frame=False, log_scale=False,
                 include_fits=True, include_cont=False, include_components=False, return_fig=False, fig_cfg=None, ax_cfg=None, maximize=False,
                 detection_band=None, show_masks=True, show_categories=False, show_err=False):


        # Set figure format with the user inputs overwriting the default conf
        legend_check = True if label is not None else False

        # Adjust the default theme
        PLT_CONF = theme.fig_defaults(fig_cfg, plot_lib='bokeh')
        AXES_CONF = theme.ax_defaults(ax_cfg, self._spec.units_wave, self._spec.units_flux, self._spec.norm_flux,
                                      plotting_library='bokeh')

        # Set the scale
        scale_str = 'log' if log_scale and (PLT_CONF.get('y_axis_type') is None) else 'linear'

        # Create figure with default utils if not provided
        fig = figure(tools=PLT_CONF.get('tools', "pan,wheel_zoom,box_zoom,reset,save"), y_axis_type=scale_str)

        # Data to plot
        wave_plot, flux_plot, z_corr, idcs_mask = frame_mask_switch(self._spec.wave, self._spec.flux,
                                                            self._spec.redshift, rest_frame)

        # Spectrum data source
        fig.step( wave_plot / z_corr, flux_plot * z_corr, mode="center", line_width=1, color=theme.colors['fg'])

        # Plot the bands if provided
        if bands is not None:
            bokeh_bands(fig, bands, wave_plot, flux_plot, z_corr, self._spec.redshift)

        # Show uncertainty
        if show_err and (self._spec.err_flux is not None):
            err_plot = self._spec.err_flux.data
            fig.varea_step(x=wave_plot / z_corr,
                           y1=(flux_plot - err_plot) * z_corr,
                           y2=(flux_plot + err_plot) * z_corr,
                           step_mode="center", fill_alpha=0.2, color=theme.colors['err_area'])

        # Include the continuum
        if include_cont and self._spec.cont is not None:
            fig.line(wave_plot/z_corr, self._spec.cont*z_corr, legend_label="Continuum.",
                     line_color=theme.colors['fade_fg'], line_dash="dashed", line_width=2)

            low_limit, high_limit = self._spec.cont - self._spec.cont_std, self._spec.cont + self._spec.cont_std
            fig.varea(x=wave_plot/z_corr, y1=low_limit*z_corr, y2=high_limit*z_corr, fill_alpha=0.2,
                      color=theme.colors['fade_fg'])

        # Plot the fittings
        if include_fits and self._spec.frame.size > 0:

            # Do not include the legend as the labels are necessary for mplcursors
            legend_check = False

            # Loop through the lines and plot them
            line_list = self._spec.frame.index.values
            profile_list = [None] * line_list.size
            for i, line_label in enumerate(line_list):
                line_i = Line.from_log(line_label, self._spec.frame)
                profile_list[i] = profile_bokeh(fig, line_i, z_corr, self._spec.frame, self._spec.redshift,
                                               self._spec.norm_flux)

        if include_components:
            # Define the bins you want
            bins = [40, 60, 80, 100]

            # Use np.histogram to get the counts in each bin
            if aspect_check:

                if self._spec.infer.pred_arr is not None:
                    categories = np.sort(np.unique(self._spec.infer.pred_arr))
                    # legend_scatter = []

                    color_list, feature_list = [], []
                    for category in categories:
                        if category != 0:

                            # Get category properties
                            feature_name = self._spec.infer.model_mgr.medium.number_feature_dict[category]
                            feature_color = aspect.cfg['colors'][feature_name]
                            idcs_feature = self._spec.infer.pred_arr == category
                            # legend_scatter.append(mlines.Line2D([], [], marker='o', color='w',
                            #                                     markerfacecolor=feature_color, markersize=8,
                            #                                     label=feature_name))
                            color_list.append(feature_color)
                            feature_list.append(feature_name)

                            # Count the pixels for each category
                            counts, _ = np.histogram(self._spec.infer.conf_arr[idcs_feature], bins=bins)
                            for idx_conf, count_conf in enumerate(counts):
                                if count_conf > 0:
                                    # Get indeces matching the detections
                                    idcs_count = np.where((bins[idx_conf] < self._spec.infer.conf_arr[idcs_feature]) &
                                                          (self._spec.infer.conf_arr[idcs_feature] <= bins[
                                                              idx_conf + 1]))[0]
                                    idcs_nonnan = np.where(idcs_feature)[0][
                                        idcs_count]  # Returns indices where mask is True

                                    # Generate nan arrays with the data to avoid filling non detections
                                    wave_nan, flux_nan = np.full(wave_plot.size, np.nan), np.full(flux_plot.size,
                                                                                                  np.nan)
                                    wave_nan[idcs_nonnan] = wave_plot[idcs_nonnan] / z_corr
                                    flux_nan[idcs_nonnan] = flux_plot[idcs_nonnan] * z_corr

                                    # Plot with the corresponding colors and linestyle
                                    fig.step(wave_nan, flux_nan, mode="center", color=feature_color,
                                             line_dash=category_conf_styles[idx_conf])

                    # Add intensity label
                    rlines = []
                    for dash in category_conf_styles.values():
                        rl = fig.line(x=wave_plot[0], y=wave_plot[0], line_color="black", line_dash=dash, line_width=2)
                        rl.visible = False
                        rlines.append(rl)

                    # Create invisible scatter glyphs for the legend
                    dots = []
                    for color in color_list:
                        r = fig.scatter(x=wave_plot[0], y=wave_plot[0], size=5, fill_color=color, line_color=None)
                        r.visible = False  # hide from plot
                        dots.append(r)

                    dot_legend = Legend(items=[LegendItem(label=label, renderers=[r]) for label, r in
                                                 zip(feature_list, dots)],
                                          orientation="vertical", location="top_right")
                    fig.add_layout(dot_legend)


                    labels_int = ["> 40% conf.", "> 60% conf.", "> 80% conf."]
                    style_legend = Legend(items=[LegendItem(label=label, renderers=[r]) for label, r in
                                                 zip(labels_int, rlines)],
                                          orientation="horizontal", location="center")

                    # Add legend *below* the plot
                    fig.add_layout(style_legend, 'below')

        # Plot labels
        fig.xaxis.axis_label = AXES_CONF['xlabel']
        fig.yaxis.axis_label = AXES_CONF['ylabel']

        # Adjust the format of the plot
        update_bokeh_figure(fig, PLT_CONF)

        # Save or display the plot
        if return_fig:
            return fig

        elif output_address is not None:
            save(fig, filename=output_address)

        else:
            # output_notebook()
            show(fig)

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
#                utils="pan,wheel_zoom,box_zoom,reset,save",
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
#         bands = check_file_dataframe(bands)
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
