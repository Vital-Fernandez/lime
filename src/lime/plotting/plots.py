import logging

import numpy as np
from matplotlib import pyplot as plt, gridspec, patches, rc_context, colors, lines as mlines, figure
from pathlib import Path

from lime.io import check_file_dataframe, load_spatial_mask, LiMe_Error, _LOG_COLUMNS_LATEX
from lime.tools import PARAMETER_LATEX_DICT, unique_line_arr
from lime.transitions import format_line_mask_option, Line
from lime.fitting.lines import c_KMpS, profiles_computation, linear_continuum_computation, PROFILE_FUNCTIONS

from lime.plotting.format import theme, latex_science_float
from lime.plotting.utils import parse_bands_arguments, color_selector


_logger = logging.getLogger('LiMe')


category_conf_styles = {0: 'dotted',
                        1: 'dashed',
                        2: 'solid'}

try:
    import mplcursors
    mplcursors_check = True
except ImportError:
    mplcursors_check = False

try:
    import aspect
    aspect_check = True
except ImportError:
    aspect_check = False

if mplcursors_check:
    from mplcursors._mplcursors import _default_annotation_kwargs as popupProps
    popupProps['bbox']['alpha'] = 0.9

# Sentinel object for non input figures
_NO_FIG = object()


def mplcursors_legend(line, log, latex_label, norm_flux, units_wave, units_flux):

    legend_text = latex_label + '\n'

    units_line_flux = units_wave * units_flux
    line_flux_latex = f'{units_line_flux:latex}'
    normFlux_latex = f' $({latex_science_float(norm_flux)})$' if norm_flux != 1 else ''

    intg_flux = latex_science_float(log.loc[line, 'intg_flux']/norm_flux)
    intg_err = latex_science_float(log.loc[line, 'intg_flux_err']/norm_flux)
    legend_text += r'$F_{{intg}} = {}\pm{}\,$'.format(intg_flux, intg_err) + line_flux_latex + normFlux_latex + '\n'

    gauss_flux = latex_science_float(log.loc[line, 'profile_flux']/norm_flux)
    gauss_err = latex_science_float(log.loc[line, 'profile_flux_err']/norm_flux)
    legend_text += r'$F_{{gauss}} = {}\pm{}\,$'.format(gauss_flux, gauss_err) + line_flux_latex + normFlux_latex + '\n'

    v_r = r'{:.1f}'.format(log.loc[line, 'v_r'])
    v_r_err = r'{:.1f}'.format(log.loc[line, 'v_r_err'])
    legend_text += r'$v_{{r}} = {}\pm{}\,\frac{{km}}{{s}}$'.format(v_r, v_r_err) + '\n'

    sigma_vel = r'{:.1f}'.format(log.loc[line, 'sigma_vel'])
    sigma_vel_err = r'{:.1f}'.format(log.loc[line, 'sigma_vel_err'])
    legend_text += r'$\sigma_{{g}} = {}\pm{}\,\frac{{km}}{{s}}$'.format(sigma_vel, sigma_vel_err)

    return legend_text


def save_close_fig_swicth(file_path=None, bbox_inches=None, fig_obj=None, maximise=False, plot_check=True):

    # By default, plot on screen unless an output address is provided
    if plot_check:
        if file_path is None:

            # Tight layout
            if bbox_inches is not None:
                plt.tight_layout()

            # Window positioning and size
            maximize_center_fig(maximise)

            # Display
            plt.show()

        elif isinstance(file_path, (Path, str)):
            plt.savefig(file_path, bbox_inches=bbox_inches)

            # Close the figure in the case of printing
            if fig_obj is not None:
                plt.close(fig_obj)

        # Keep the image on the memory
        else:
            _logger.info(f'Ouput filepath is not recognized: {file_path} ({type(file_path)})')
            return

    return


def frame_mask_switch(observation, rest_frame):

    # Doppler factor for rest _frame plots
    z_corr = (1 + observation.redshift) if rest_frame else 1

    # Remove mask from plots and recover bad indexes
    idcs_mask = observation.wave.mask
    wave_plot = observation.wave.data
    flux_plot = observation.flux.data
    err_plot = None if observation.err_flux is None else observation.err_flux.data

    return wave_plot, flux_plot, err_plot, z_corr, idcs_mask


def _mplcursor_parser(line_g_list, list_comps, log, norm_flux, units_wave, units_flux):

    if mplcursors_check:
        for i, line_g in enumerate(line_g_list):
            line_label = line_g.label
            latex_label = log.loc[line_label, 'latex_label']
            label_complex = mplcursors_legend(line_label, log, latex_label, norm_flux, units_wave, units_flux)
            mplcursors.cursor(line_g).connect("add", lambda sel, label=label_complex: sel.annotation.set_text(label))

    return


def maximize_center_fig(maximize_check=False, center_check=False):

    if maximize_check:

        # Windows maximize
        mng = plt.get_current_fig_manager()

        try:
            mng.window.showMaximized()
        except:
            try:
                mng.resize(*mng.window.maxsize())
            except:
                _logger.debug(f'Unable to maximize the window')

    if center_check:

        try:
            mngr = plt.get_current_fig_manager()
            mngr.window.setGeometry(1100, 300, mngr.canvas.width(), mngr.canvas.height())
        except:
            _logger.debug(f'Unable to center plot window')

    return


def _auto_flux_scale(axis, y, y_scale, scale_dict=theme.plt):

    # If non-provided auto-decide
    if y_scale == 'auto':

        # Limits for the axes, ignore the divide by zero warning
        with np.errstate(divide='ignore', invalid='ignore'):
            neg_check = np.any(y < 0)
            y_max, y_min = np.nanmax(y), np.nanmin(y)
            ratio = np.abs(y_max/y_min)
            if (ratio > 25) or (ratio < 0.06):
                if neg_check:
                    if np.sum(y>0) > 1:
                        y_scale = {'value': 'symlog', 'linthresh': min(np.ceil(np.abs(y_min)), np.min(y[y>0]))}
                    else:
                        y_scale = {'value': 'symlog', 'linthresh': np.ceil(np.abs(y_min))}
                else:
                    y_scale = {'value': 'log'}
            else:
                y_scale = {'value': 'linear'}

        axis.set_yscale(**y_scale)

        if y_scale["value"] != 'linear':
            axis.text(0.12, 0.8, f'${y_scale["value"]}$',
                      fontsize=scale_dict['textsize_notes'], ha='center', va='center',
                      transform=axis.transAxes, alpha=0.5, color=theme.colors['fg'])
    else:
        axis.set_yscale(y_scale)

    return


def _auto_flux_scale_backUp(axis, y, y_scale, scale_dict=theme.plt):

    # If non-provided auto-decide
    if y_scale == 'auto':

        # Limits for the axes, ignore the divide by zero warning
        with np.errstate(divide='ignore', invalid='ignore'):
            neg_check = np.any(y < 0)
            y_max, y_min = np.nanmax(y), np.nanmin(y)
            ratio = np.abs(y_max/y_min)
            if (ratio > 25) or (ratio < 0.06):
                if neg_check:
                    if np.sum(y>0) > 1:
                        y_scale = {'value': 'symlog', 'linthresh': min(np.ceil(np.abs(y_min)), np.min(y[y>0]))}
                    else:
                        y_scale = {'value': 'symlog', 'linthresh': np.ceil(np.abs(y_min))}
                else:
                    y_scale = {'value': 'log'}
            else:
                y_scale = {'value': 'linear'}

        axis.set_yscale(**y_scale)

        if y_scale["value"] != 'linear':
            axis.text(0.12, 0.8, f'${y_scale["value"]}$', fontsize=scale_dict['textsize_notes'], ha='center', va='center',
                      transform=axis.transAxes, alpha=0.5, color=theme.colors['fg'])
    else:
        axis.set_yscale(y_scale)

    return


def check_image_size(bg_image, fg_image, mask_dict):

    # Confirm that the background and foreground images have the same size
    if fg_image is not None:
        if bg_image.shape != fg_image.shape:
            _logger.warning(f'The cube background ({bg_image.shape}) image and foreground image ({fg_image.shape}) have'
                            f' different size')

    # Confirm that the background and mask images have the same size
    for mask_name, mask_hdul in mask_dict.items():
        mask_data, mask_hdr = mask_hdul
        if bg_image.shape != mask_data.shape:
            _logger.warning(f'The cube background ({bg_image.shape}) image and mask {mask_name} ({mask_data.shape}) have'
                            f' different size')

    return


def determine_cube_images(cube, line, band, percentiles, color_scale, contours_check=False):

    # Generate line object
    if line is not None:

        # Determine the line of reference
        line = Line.from_transition(line, data_frame=band)

        # Compute the band map slice
        idcsEmis, idcsCont = line.index_bands(cube.wave, cube.redshift)
        image = cube.flux[idcsEmis, :, :].sum(axis=0)

        # If no scale provided compute a default one
        if color_scale is None:

            levels = np.nanpercentile(image, np.array(percentiles, ndmin=1))

            # For the background image (A minimum level)
            if not contours_check:
                color_scale = colors.SymLogNorm(linthresh=levels[0], vmin=levels[0], base=10)

            # For the foreground
            else:
                color_scale = colors.LogNorm()

    else:
        line, image, levels, color_scale = None, None, None, None

    return line, image, levels, color_scale


def image_plot(ax, image_bg, image_fg, fg_levels, fg_mesh, bg_scale, fg_scale, bg_color, fg_color, cursor_cords=None):

    # Background image plot
    im = ax.imshow(image_bg, cmap=bg_color, norm=bg_scale)

    # Foreground contours
    if image_fg is not None:
        contours = ax.contour(fg_mesh[0], fg_mesh[1], image_fg, cmap=fg_color, levels=fg_levels, norm=fg_scale,
                              linewidth=0.2)
    else:
        contours = None

    # Marker
    if cursor_cords is not None:
        marker, = ax.plot(cursor_cords[1], cursor_cords[0], '+', color='red')
    else:
        marker = None

    return im, contours, marker


def spatial_mask_plot(ax, masks_dict, mask_color, mask_alpha, units_flux, mask_list=[]):

    # Container for the legends
    legend_list = [None] * len(masks_dict)

    cmap_contours = plt.get_cmap(mask_color, len(masks_dict))

    for idx_mask, items in enumerate(masks_dict.items()):

        mask_name, hdu_mask = items
        mask_data, mask_header = hdu_mask

        if (len(mask_list) == 0) or (mask_name in mask_list):

            # Inverse the mask array for the plot
            inv_mask_array = np.ma.masked_array(mask_data, ~mask_data)

            # Plot the data
            cm_i = colors.ListedColormap([cmap_contours(idx_mask)])
            ax.imshow(inv_mask_array, cmap=cm_i, vmin=0, vmax=1, alpha=mask_alpha)

            mask_param = mask_header['PARAM']
            param_idx = mask_header.get('PARAMIDX')
            param_val = mask_header.get('PARAMVAL')
            n_spaxels = mask_header.get('NUMSPAXE')

            if param_idx and param_val and n_spaxels:

                # Create the legend label
                legend_i = f'{mask_name}, ' + PARAMETER_LATEX_DICT.get(mask_param, f'${mask_param}$')

                # Add units if using the flux
                if mask_param == units_flux:
                    units_text = f'{units_flux:latex}'
                    legend_i += r'$\left({}\right)$'.format(units_text[1:-1])

                # Add percentile number
                legend_i += r', $P_{{{}th}}$ = '.format(param_idx)

                # Add percentile value and number voxels
                legend_i += f'${latex_science_float(param_val, dec=3)}$ ({n_spaxels} spaxels)'


                legend_list[idx_mask] = patches.Patch(color=cmap_contours(idx_mask), label=legend_i)

    return legend_list


def spec_plot(ax, spec, rest_frame=False, show_profiles=True, log_scale=False):

    # Reference _frame for the plot
    wave_plot, flux_plot, err_plot, z_corr, idcs_mask = frame_mask_switch(spec, rest_frame)

    # Plot the spectrum
    ax.step(wave_plot / z_corr, flux_plot * z_corr, where='mid', color=theme.colors['fg'], linewidth=theme.plt['spectrum_width'])

    # Plot the fittings
    if show_profiles and spec.frame.size > 0:
        mplcursor_list = []
        for line_label in unique_line_arr(spec.frame):
            line = Line.from_transition(line_label, data_frame=spec.frame)
            mplcursor_list += spec_profile_plotter(ax, spec, line, z_corr)

        # Pop-ups
        mplcursor_parser(mplcursor_list, spec)

    # Y scale
    if log_scale:
        ax[1].set_yscale('log')

    # # Reference frame for the plot
    # wave_plot, flux_plot, z_corr, idcs_mask = frame_mask_switch(self._cube, rest_frame)
    #
    # # Plot the spectrum
    # ax.step(wave_plot / z_corr, flux_plot * z_corr, label=label, where='mid', color=theme.colors['fg'],
    #         linewidth=scale_dict['spectrum_width'])
    #
    # # List of lines in the log
    # line_list = []
    # if log is not None:
    #     if log.index.size > 0 and include_fits:
    #         line_list = log.index.values
    #
    #         # Loop through the lines and plot them
    #         line_g_list = [None] * line_list.size
    #         for i, line_label in enumerate(line_list):
    #
    #             line_i = Line.from_log(line_label, log)
    #             line_g_list[i] = _profile_plt(ax, line_i, z_corr, log, redshift, norm_flux)
    #
    #         # Add the interactive pop-ups
    #         _mplcursor_parser(line_g_list, line_list, log, norm_flux, units_wave, units_flux)
    #
    # # Plot the masked pixels
    # _masks_plot(ax, line_list, wave_plot, flux_plot, z_corr, log, idcs_mask)

    return


def _profile_plot(axis, x, y, label, idx_line=0, n_comps=1, observations_list='yes', scale_dict=theme.plt):

    # Color and thickness
    if observations_list == 'no':

        # If only one component or combined
        if n_comps == 1:
            width_i, style, color = scale_dict['single_width'], '-', theme.colors['profile']

        # Component
        else:
            cmap = plt.get_cmap(theme.colors['comps_map'])
            width_i, style, color = scale_dict['comp_width'], ':', cmap(idx_line/n_comps)

    # Case where the line has an error
    else:
        width_i, style, color = scale_dict['err_width'], '-', theme.colors['error']

    # Plot the profile
    line_g = axis.plot(x, y, label=label, linewidth=width_i, linestyle=style, color=color)

    return line_g


def _profile_plt(axis, line, z_cor, log, redshift, norm_flux):

    # Check if blended line or Single/merged
    if line.group == 'b':
        idx_line = line.list_comps.index(line.label)
        n_comps = len(line.list_comps)

    else:
        idx_line = 0
        n_comps = 1

    # Compute the profile(s)
    wave_array, flux_array = profiles_computation(line.param_arr('label'), log, (1 + redshift), line.param_arr('profile'))
    wave_array, cont_array = linear_continuum_computation(line.param_arr('label'), log, (1 + redshift))

    wave_i = wave_array[:, 0]
    cont_i = cont_array[:, 0]
    flux_i = flux_array[:, idx_line]

    # Plot a continuum (only one per line)
    if idx_line == 0:
        cont_format = dict(label=None, color=theme.colors['cont'], linestyle='--', linewidth=0.5)
        axis.plot(wave_array/z_cor, cont_array[:, 0] * z_cor / norm_flux, **cont_format)

    # Plot combined gaussian profile if blended
    if (idx_line == 0) and (n_comps > 1):
        comb_array = (flux_array.sum(axis=1) + cont_i) * z_cor / norm_flux
        line_format = color_selector(line.label, line.measurements.observations, 0, n_comps=n_comps, scale_dict=theme.plt,
                                     colors_dict=theme.colors)
        axis.plot(wave_i / z_cor, comb_array, **line_format)

    # Gaussian component plot
    single_array = (flux_i + cont_i) * z_cor / norm_flux
    line_format = color_selector(line.label, line.measurements.observations, idx_line, n_comps, scale_dict=theme.plt,
                                     colors_dict=theme.colors)
    line_single = axis.plot(wave_i/z_cor, single_array, **line_format)

    return line_single


def _gaussian_line_profiler(axis, line_list, wave_array, gaussian_array, cont_array, z_corr, log, norm_flux):

    # Data for the plot
    observations = log.loc[log.index.isin(line_list)].observations.values

    # Plot them
    line_g_list = [None] * len(line_list)
    for i, line in enumerate(line_list):

        # Check if blended or single/merged
        idcs_comp = None
        if (not line.endswith('_m')) and (log.loc[line, 'group_label'] != 'none') and (len(line_list) > 1):
            profile_comps = log.loc[line, 'group_label']
            if profile_comps is not None:
                profile_comps = profile_comps.split('+')
                idx_line = profile_comps.index(line)
                n_comps = len(profile_comps)
                if profile_comps.index(line) == 0:
                    idcs_comp = (log['group_label'] == log.loc[line, 'group_label']).values
            else:
                idx_line = 0
                n_comps = 1
        else:
            idx_line = 0
            n_comps = 1

        # label for th elegend
        latex_label = log.loc[line, 'latex_label']

        # Get the corresponding axis
        wave_i = wave_array[:, i]
        cont_i = cont_array[:, i]
        gauss_i = gaussian_array[:, i]

        # Continuum (only one per line)
        if idx_line == 0:
            axis.plot(wave_i / z_corr, cont_i * z_corr / norm_flux, color=theme.colors['cont'], label=None,
                      linestyle='--', linewidth=0.5)

        # Plot combined gaussian profile if blended
        if idcs_comp is not None:
            gauss_comb = gaussian_array[:, idcs_comp].sum(axis=1) + cont_i
            _profile_plot(axis, wave_i/z_corr, gauss_comb*z_corr/norm_flux, None,
                               idx_line=idx_line, n_comps=1, observations_list=observations[i])

        # Gaussian component plot
        line_g_list[i] = _profile_plot(axis, wave_i/z_corr, (gauss_i+cont_i)*z_corr/norm_flux, latex_label,
                                       idx_line=idx_line, n_comps=n_comps, observations_list=observations[i])

    return line_g_list


def _masks_plot(axis, line_list, x, y, z_corr, log, spectrum_mask, color_dict={}):

    # Spectrum mask
    if np.any(spectrum_mask):
        x_mask = x[spectrum_mask]/z_corr
        y_mask = y[spectrum_mask]*z_corr
        if not np.all(np.isnan(y_mask)):
            axis.scatter(x_mask, y_mask, marker='x', label='Masked pixels', color='red') # TODO add color dict

    # Line masks
    if log is not None:
        if 'pixel_mask' in log.columns:
            if line_list is not None:
                for i, line in enumerate(line_list):
                    if line in log.index:
                        pixel_mask = log.loc[line, 'pixel_mask']
                        if pixel_mask != 'no':
                            line_mask_limits = format_line_mask_option(pixel_mask, x)
                            idcsMask = (x[:, None] >= line_mask_limits[:, 0]) & (x[:, None] <= line_mask_limits[:, 1])
                            idcsMask = idcsMask.sum(axis=1).astype(bool)
                            if np.sum(idcsMask) >= 1:
                                axis.scatter(x[idcsMask]/z_corr, y[idcsMask]*z_corr, marker="x",
                                             color=color_dict['mask_marker'])

    return


def label_generator(idx_sample, log, legend_handle):

    if legend_handle == 'levels':
        spec_label = ", ".join(map(str, idx_sample))

    elif legend_handle is None:
        spec_label = None

    else:

        if legend_handle in log.index.names:
            idx_item = list(log.index.names).index(legend_handle)
            spec_label = idx_sample[idx_item]

        elif legend_handle in log.columns:
            spec_label = log.loc[idx_sample, legend_handle]

        else:
            raise LiMe_Error(f'The input handle "{legend_handle}" is not found on the sample log columns')

    return spec_label


def redshift_key_evaluation(spectrum, mode, z_infered, data_mask, gauss_arr, z_arr, flux_sum_arr, in_fig=None, fig_cfg=None,
                            ax_cfg=None, label=None, rest_frame=True):

    # Display check for the user figures
    display_check = True if in_fig is None else False

    # Adjust the default theme
    PLT_CONF = theme.fig_defaults(fig_cfg)
    AXES_CONF = theme.ax_defaults(ax_cfg, spectrum.units_wave, spectrum.units_flux, spectrum.norm_flux)

    # Create and fill the figure
    with (rc_context(PLT_CONF)):

        # Generate the figure object and figures
        if in_fig is None:
            in_fig = plt.figure()

        grid_ax = in_fig.add_gridspec(nrows=2, ncols=1, height_ratios=[2, 1])
        ax1 = plt.subplot(grid_ax[0])
        ax2 = ax1.twinx()
        ax3 = plt.subplot(grid_ax[1])

        # Reference _frame for the plot
        wave_plot, flux_plot, err_plot, z_corr, idcs_mask = frame_mask_switch(spectrum, rest_frame)

        # Plot spectrum
        ax1.step(wave_plot / z_corr, flux_plot * z_corr, label=label, where='mid', color=theme.colors['fg'],
                   linewidth=theme.plt['spectrum_width'])

        # Plot the bands
        ax2.step(wave_plot / z_corr, gauss_arr, label=label, where='mid', color='yellow', linewidth=theme.plt['spectrum_width'])

        # Plot the data used for the masks
        y_arr = np.full(flux_plot.size, np.nan)
        y_arr[data_mask] = flux_plot[data_mask]
        ax1.step(wave_plot / z_corr, y_arr*z_corr, label=label, where='mid', color='red', linewidth=theme.plt['spectrum_width'])

        # Plot the spectrum sum
        ax3.step(z_arr, flux_sum_arr/np.max(flux_sum_arr), color=theme.colors['fg'], where='mid', linewidth=theme.plt['spectrum_width'])

        # Plot peack
        ax3.scatter(z_infered, 1, marker='o', color='red')

        # Wording and formatting
        ax1.set(**AXES_CONF)

        ax2.set_ylim(0, 1)
        ax2.axis('off')

        ax3.update({'xlabel': 'Redshift range',
                    'ylabel': r'$\frac{{F_{{sum, {band}}}}}{{max(F_{{sum, {band}}})}}$'.format(band='pixels' if mode == 'xor' else 'flux'),
                    'title': r'$z_{{prediction, {mode}}} = $'.format(mode=mode) + f'{z_infered:0.3f}'})
        ax3.set_yticks([0, 1])

        # By default, plot on screen unless an output address is provided
        output_address, maximize = None, False
        in_fig = save_close_fig_swicth(output_address, 'tight', in_fig, maximize, display_check)

    return

def redshift_permu_evaluation(spectrum, z_infered, obs_wave_arr, theo_wave_arr, in_fig=None, fig_cfg=None,
                            ax_cfg=None, label=None, rest_frame=False):

    # Display check for the user figures
    display_check = True if in_fig is None else False

    # Set figure format with the user 2_guides overwriting the default conf
    legend_check = True if label is not None else False

    print(f'Observed wavelengths: {obs_wave_arr}')
    print(f'Best matching wavelengths: {theo_wave_arr}')

    # Adjust the default theme
    PLT_CONF = theme.fig_defaults(fig_cfg)
    AXES_CONF = theme.ax_defaults(ax_cfg, spectrum.units_wave, spectrum.units_flux, spectrum.norm_flux)

    # Create and fill the figure
    with (rc_context(PLT_CONF)):

        in_fig, in_ax = plt.subplots()

        if AXES_CONF.get('title') is None:
         AXES_CONF['title'] = r'$z_{permutation} = $' + f'{z_infered:0.3f}'

        in_ax.set(**AXES_CONF)

        # Reference _frame for the plot
        wave_plot, flux_plot, z_corr, idcs_mask = frame_mask_switch(spectrum.wave, spectrum.flux, z_infered, rest_frame)

        # Plot spectrum
        in_ax.step(wave_plot / z_corr, flux_plot * z_corr, label=label, where='mid', color=theme.colors['fg'],
                   linewidth=theme.plt['spectrum_width'])

        for i, obs_wave in enumerate(obs_wave_arr):
            in_ax.axvline(obs_wave, linestyle='--')

        for i, theo_wave in enumerate(theo_wave_arr):
            in_ax.axvline(theo_wave, linestyle=':')

        # By default, plot on screen unless an output address is provided
        output_address, maximize = None, False
        in_fig = save_close_fig_swicth(output_address, 'tight', in_fig, maximize, display_check)

    return


def bands_filling_plot(axis, x, y, z_corr, idcs_mask, label, exclude_continua=True, color_dict=theme.colors, show_central=True):

    # Security check for low selection
    # TODO check this error crashing when continua bands are very small, need a better error message
    if y[idcs_mask[2]:idcs_mask[3]].size > 1:

        # Lower limit for the filled region
        if exclude_continua is False:
            low_lim = np.min(y[idcs_mask[0]:idcs_mask[5]])
            low_lim = 0 if np.isnan(low_lim) else low_lim
            x_interval = x[idcs_mask[2]:idcs_mask[3]]
            y_interval = y[idcs_mask[2]:idcs_mask[3]]
        else:
            x_interval = x[idcs_mask[2]:idcs_mask[3]]
            y_interval = y[idcs_mask[2]:idcs_mask[3]]
            m = (y_interval[-1] - y_interval[0])/(x_interval[-1] - x_interval[0])
            n = y_interval[0] - m * x_interval[0]
            low_lim = m * x_interval + n

        # Central bands
        if show_central:
            axis.fill_between(x_interval/z_corr, low_lim*z_corr, y_interval*z_corr,
                              facecolor=color_dict['line_band'], step='mid', alpha=0.25)

        # Continua bands exclusion
        if exclude_continua is False:
            axis.fill_between(x[idcs_mask[0]:idcs_mask[1]]/z_corr, low_lim*z_corr, y[idcs_mask[0]:idcs_mask[1]]*z_corr,
                              facecolor=color_dict['cont_band'], step='mid', alpha=0.25)
            axis.fill_between(x[idcs_mask[4]:idcs_mask[5]]/z_corr, low_lim*z_corr, y[idcs_mask[4]:idcs_mask[5]]*z_corr,
                              facecolor = color_dict['cont_band'], step = 'mid', alpha = 0.25)

    else:
        _logger.warning(f'The {label} band plot interval contains less than 1 pixel')

    return


def plot_peaks_troughs(spec, peak_idcs, detect_limit, continuum, match_bands, **kwargs):

    norm_flux = spec.norm_flux
    wave = spec.wave
    flux = spec.flux
    units_wave = spec.units_wave
    units_flux = spec.units_flux
    redshift = spec.redshift

    PLOT_CONF = theme.fig_defaults(None)
    AXES_CONF = theme.ax_defaults(None, units_wave, units_flux, norm_flux)

    wave_plot, flux_plot, z_corr, idcs_mask = frame_mask_switch(wave, flux, redshift, 'observed')

    continuum = continuum if continuum is not None else np.zeros(flux.size)

    if 'signal_peak' in match_bands.columns:
        idcs_detect = match_bands['signal_peak'].to_numpy(dtype=int)
    else:
        idcs_detect = np.zeros(match_bands.index.size, dtype=bool)

    with rc_context(PLOT_CONF):

        fig, ax = plt.subplots()
        ax.step(wave_plot, flux_plot, color=theme.colors['fg'], label='Object spectrum', where='mid')

        ax.scatter(wave_plot[peak_idcs], flux_plot[peak_idcs], marker='o', label='Peaks', color=theme.colors['fade_fg'], facecolors='none')
        ax.fill_between(wave_plot, continuum, detect_limit, facecolor=theme.colors['line_band'], label='Noise_region', alpha=0.5)
        ax.scatter(wave_plot[idcs_detect], flux_plot[idcs_detect], marker='o', label='Matched lines',
                   color=theme.colors['peak'], facecolors='none')

        if continuum is not None:
            ax.plot(wave_plot, continuum, label='Continuum')

        ax.scatter(wave_plot[idcs_mask], flux_plot[idcs_mask], label='Masked pixels', marker='x',
                   color=theme.colors['mask_marker'])

        ax.legend()
        ax.update(AXES_CONF)
        plt.tight_layout()
        plt.show()

    return


def line_profile_generator(line, x_array):

    # Get components
    line_list = [line] if line.group != 'b' else line.list_comps

    # Y array container
    curve_arr = np.zeros((len(line_list), x_array.size))

    # Compile the flux profile for the corresponding shape
    for i, trans_i in enumerate(line_list):

        match trans_i.profile:
            case 'g':
                curve_arr[i, :] = PROFILE_FUNCTIONS[trans_i.profile](x_array, line.measurements.amp[i],
                                                                              line.measurements.center[i],
                                                                              line.measurements.sigma[i])
            case 'l':
                curve_arr[i, :] = PROFILE_FUNCTIONS[trans_i.profile](x_array, line.measurements.amp[i],
                                                                              line.measurements.center[i],
                                                                              line.measurements.sigma[i])
            case 'v':
                curve_arr[i, :] = PROFILE_FUNCTIONS[trans_i.profile](x_array, line.measurements.amp[i],
                                                                              line.measurements.center[i],
                                                                              line.measurements.sigma[i],
                                                                              line.measurements.gamma[i])
            case 'pv':
                curve_arr[i, :] = PROFILE_FUNCTIONS[trans_i.profile](x_array, line.measurements.amp[i],
                                                                              line.measurements.center[i],
                                                                              line.measurements.sigma[i],
                                                                              line.measurements.frac[i])
            case 'pv':
                curve_arr[i, :] = PROFILE_FUNCTIONS[trans_i.profile](x_array, line.measurements.amp[i],
                                                                              line.measurements.center[i],
                                                                              line.measurements.sigma[i],
                                                                              line.measurements.frac[i])
            case 'e':
                curve_arr[i, :] = PROFILE_FUNCTIONS[trans_i.profile](x_array, line.measurements.amp[i],
                                                                              line.measurements.center[i],
                                                                              line.measurements.alpha[i])
            case 'pp':
                curve_arr[i, :] = PROFILE_FUNCTIONS[trans_i.profile](x_array, line.measurements.amp[i],
                                                                              line.measurements.center[i],
                                                                              line.measurements.sigma[i],
                                                                              line.measurements.frac[i],
                                                                              line.measurements.alpha[i])
            case 'p':
                curve_arr[i, :] = PROFILE_FUNCTIONS[trans_i.profile](x_array, line.measurements.a[i],
                                                                              line.measurements.b[i],
                                                                              line.measurements.center[i],
                                                                              line.measurements.alpha[i])
            case _:
                raise LiMe_Error(f'Line profile "{trans_i.profile}" for {trans_i.label} is not recognized. Please use '
                                 f'_p-g (gaussian)\n, _p-l (Lorentz)\n, _p-v (Voigt)\n, _p-pv (pseudo-Voigt)\n, '
                                 f'_p-e (exponential)\n, _p-pp (pseudo-power law)\n, _p-p (power law)')

    return curve_arr


class Plotter:

    def __init__(self):

        self._legends_dict = {}

        return


    def _plot_container(self, fig, ax, ax_cfg={}, gfit_type=False):

        #Plot for residual axis
        if gfit_type is True:

            # Two axes figure, upper one for the line lower for the residual
            gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
            spec_ax = plt.subplot(gs[0])
            resid_ax = plt.subplot(gs[1], sharex=spec_ax)
            fig, ax = None, (spec_ax, resid_ax)

        # Default plot
        else:

            if (fig is None) and (ax is None):
                fig, ax = plt.subplots()

            ax.set(**ax_cfg)

        return fig, ax


    def _line_matching_plot(self, axis, bands, x, y, z_corr, redshift):

        # Open the bands file the bands
        match_log = check_file_dataframe(bands)

        # Compute bands limits
        w3 = match_log.w3.values * (1 + redshift)
        w4 = match_log.w4.values * (1 + redshift)
        idcs_bands = np.searchsorted(x, np.array([w3, w4]))

        # Plot the detected line peaks
        if 'signal_peak' in match_log.columns:
            idcs_peaks = match_log['signal_peak'].values.astype(int)
            axis.scatter(x[idcs_peaks]/z_corr, y[idcs_peaks]*z_corr, label='Peaks',
                         facecolors='none', edgecolors=theme.colors['peak'])

        # Loop through the detections and plot the names
        for i, line_label in enumerate(match_log.index):
            line = Line.from_transition(line_label, data_frame=match_log)

            # Get the max flux on the region making the exception for 1 pixel bands
            idx_w3, idx_w4 = idcs_bands[:, i]
            max_region = np.max(y[idx_w3:idx_w4]) if idx_w3 != idx_w4 else y[idx_w3]
            x_region = x[idx_w3:idx_w4]

            x_text = line.wavelength * (1 + redshift)/z_corr
            y_text = max_region * 0.9 * z_corr
            text = line.label

            axis.text(x_text, y_text, text, rotation=270)
            axis.axvspan(x_region[0]/z_corr, x_region[-1]/z_corr, label='Matched line' if i == 0 else '_', alpha=0.30,
                         color=theme.colors['match_line'])

        return


    def _peak_plot(self, axis, log, list_comps, z_corr, norm_flux):

        peak_wave = log.loc[list_comps[0]].peak_wave/z_corr,
        peak_flux = log.loc[list_comps[0]].peak_flux*z_corr/norm_flux
        axis.scatter(peak_wave, peak_flux, facecolors=theme.colors['peak'])

        return


    def _cont_plot(self, axis, x, y, z_corr, norm_flux):

        # Plot the continuum,  Usine wavelength array and continuum form the first component
        # cont_wave = wave_array[:, 0]
        # cont_linear = cont_array[:, 0]
        axis.plot(x/z_corr, y*z_corr/norm_flux, color=theme.colors['cont'], linestyle='--', linewidth=0.5)

        return


    # def _plot_continuum_fit(self, continuum_fit, idcs_cont, low_lim, high_lim, threshold_factor, plot_title=''):
    #
    #     norm_flux = self._spec.norm_flux
    #     wave = self._spec.wave
    #     flux = self._spec.flux
    #     units_wave = self._spec.units_wave
    #     units_flux = self._spec.units_flux
    #     redshift = self._spec.redshift
    #
    #     # Assign the default axis labels
    #     PLOT_CONF = theme.fig_defaults(None)
    #     AXES_CONF = theme.ax_defaults(None, units_wave, units_flux, norm_flux)
    #
    #     wave_plot, flux_plot, z_corr, idcs_mask = frame_mask_switch(wave, flux, redshift, False)
    #
    #     with rc_context(PLOT_CONF):
    #
    #         fig, ax = plt.subplots()
    #
    #         # Object spectrum
    #         ax.step(wave_plot, flux_plot, label='Object spectrum', color=theme.colors['fg'], where='mid')
    #
    #         # Band limits
    #         label = r'$16^{{th}}/{} - 84^{{th}}\cdot{}$ flux percentiles band'.format(threshold_factor, threshold_factor)
    #         ax.axhspan(low_lim, high_lim, alpha=0.2, label=label, color=theme.colors['line_band'])
    #         ax.axhline(np.median(flux_plot[idcs_cont]), label='Median flux', linestyle=':', color='black')
    #
    #         # Masked and rectected pixels
    #         ax.scatter(wave_plot[~idcs_cont], flux_plot[~idcs_cont], label='Rejected pixels', color=theme.colors['peak'], facecolor='none')
    #         ax.scatter(wave_plot[idcs_mask], flux_plot[idcs_mask], marker='x', label='Masked pixels', color=theme.colors['mask_marker'])
    #
    #         # Output continuum
    #         ax.plot(wave_plot, continuum_fit, label='Continuum')
    #
    #         ax.update(AXES_CONF)
    #         ax.legend()
    #         plt.tight_layout()
    #         plt.show()
    #
    #     return


def mplcursor_parser(mpl_key_list, spec):

    if mplcursors_check and len(mpl_key_list) > 0:
        for i, (label, latex, curve_plt) in enumerate(mpl_key_list):
            label_complex = mplcursors_legend(label, spec.frame, latex, spec.norm_flux, spec.units_wave, spec.units_flux)
            mplcursors.cursor(curve_plt).connect("add", lambda sel, label=label_complex: sel.annotation.set_text(label))

    return


def spec_continuum_calculation(spec, wave, flux, cont_fit, idcs_cont, low_flux_limit, high_flux_limit,
                               smooth_scale, exclude_intvls, **kwargs):

    # Clear previous figure
    spec.plot.reset_figure()

    # Display check for input figures
    display_check = False if kwargs.get('in_fig') is not None else True

    # Adjust the default theme
    plt_cfg = theme.fig_defaults(kwargs.get('in_fig'))
    ax_labels_cfg = theme.ax_defaults(kwargs.get('ax_cfg'), spec)

    # Create and fill the figure
    with (rc_context(plt_cfg)):

        # Establish figure
        spec.fig = plt.figure() if kwargs.get('in_fig') is None else kwargs.get('in_fig')

        # Establish the axes
        spec.ax_list = spec.fig.add_subplot()
        spec.ax_list.set(**ax_labels_cfg)

        # Plot the spectrum
        label = kwargs.get('label') or ('Object spectrum' if smooth_scale is None else f'Smoothed spectrum ({smooth_scale} pixels)')
        spec.ax_list.step(wave, flux, label=label, where='mid', color=theme.colors['fg'], linewidth=theme.plt['spectrum_width'])

        # Flux uncertainty shaded area
        spec.ax_list.fill_between(wave, low_flux_limit, high_flux_limit, alpha=0.2, label='Flux threshold',
                                  color=theme.colors['inspection_uncertainty'])

        # Masked and rejected pixels
        spec.ax_list.scatter(wave[~idcs_cont], flux[~idcs_cont], label='Rejected peaks',
                             color=theme.colors['rejected_peak'], facecolor='none')


        # Output continuum
        spec.ax_list.plot(wave, cont_fit, label='Continuum', color=theme.colors['cont'], linestyle='--')

        # Excluded intervals
        if exclude_intvls is not None:
            i0 = np.searchsorted(wave, exclude_intvls[:, 0], side="right")
            i1 = np.searchsorted(wave, exclude_intvls[:, 1], side="left")
            for start, stop in zip(i0, i1):
                spec.ax_list.axvspan(wave[start], wave[stop], alpha=0.25, color=theme.colors['rejected_peak'])

        # Switch y_axis to logarithmic scale if requested
        if kwargs.get('log_scale'):
            spec.ax_list.set_yscale('log')

        # Show the legend:
        spec.ax_list.legend()

        # By default, plot on screen unless an output address is provided
        maximize = False if kwargs.get('output_address') is None else kwargs.get('output_address')
        save_close_fig_swicth(kwargs.get('output_address'), 'tight', spec.fig, maximize, display_check)

    return

def spec_peak_calculation(spec, match_bands, detect_limit, peak_idcs, continuum=None, **kwargs):

    # Clear previous figure
    spec.plot.reset_figure()

    # Display check for input figures
    display_check = False if kwargs.get('in_fig') is not None else True

    # Plotting settings
    rest_frame = kwargs.get('rest_frame') if  kwargs.get('rest_frame') is not None else False
    label = kwargs.get('label') if kwargs.get('label') is None else None

    # Idcs for the peaks
    if 'signal_peak' in match_bands.columns:
        idcs_detect = match_bands['signal_peak'].to_numpy(dtype=int)
    else:
        idcs_detect = np.zeros(match_bands.index.size, dtype=bool)

    # Adjust the default theme
    plt_cfg = theme.fig_defaults(kwargs.get('in_fig'))
    ax_labels_cfg = theme.ax_defaults(kwargs.get('ax_cfg'), spec)

    # Create and fill the figure
    with (rc_context(plt_cfg)):

        # Establish figure
        spec.plot.fig = plt.figure() if kwargs.get('in_fig') is None else kwargs.get('in_fig')

        # Establish the axes
        spec.plot.ax = spec.plot.fig.add_subplot()
        spec.plot.ax.set(**ax_labels_cfg)

        # Plot the data
        wave_plot, flux_plot, err_plot, z_corr, idcs_mask = frame_mask_switch(spec, rest_frame)

        spec.plot.ax.step(wave_plot, flux_plot, label=label, where='mid', color=theme.colors['fg'],
                                linewidth=theme.plt['spectrum_width'])

        spec.plot.ax.scatter(wave_plot[peak_idcs], flux_plot[peak_idcs], marker='o', label='Unmatched Peaks',
                             color=theme.colors['rejected_peak'], facecolors='none')

        spec.plot.ax.fill_between(wave_plot, continuum, detect_limit, facecolor=theme.colors['inspection_uncertainty'],
                                  label='Flux threshold', alpha=0.5)

        spec.plot.ax.scatter(wave_plot[idcs_detect], flux_plot[idcs_detect], marker='o', label='Matched lines',
                             color=theme.colors['peak'], facecolors=theme.colors['peak'])

        spec.plot.ax.plot(wave_plot / z_corr, spec.cont * z_corr, label='Continuum', color=theme.colors['cont'],
                          linestyle='--')

        # Show the legend:
        spec.plot.ax.legend()

        # Switch y_axis to logarithmic scale if requested
        if kwargs.get('log_scale'):
            spec.plot.ax.set_yscale('log')

        # By default, plot on screen unless an output address is provided
        maximize = False if kwargs.get('output_address') is None else kwargs.get('output_address')
        save_close_fig_swicth(kwargs.get('output_address'), 'tight', spec.plot.fig, maximize, display_check)

    return


def spec_profile_plotter(ax, spec, line_i, z_corr):

    # Compute the profile arrays
    wave_array = np.linspace(line_i.mask[2], line_i.mask[3], 100) * (1 + spec.redshift)
    cont_array = line_i.measurements.m_cont * wave_array + line_i.measurements.n_cont
    flux_array = line_profile_generator(line_i, wave_array)

    # Plot Continuum
    ax.plot(wave_array/z_corr, cont_array * z_corr / spec.norm_flux,
            color=theme.colors['cont'], linestyle='--', linewidth=0.5)

    # Combined component
    comb_array = (flux_array.sum(axis=0) + cont_array) * z_corr / spec.norm_flux
    line_format = color_selector(line_i.label, line_i.measurements.observations, 0, n_comps=1,
                                 scale_dict=theme.plt, colors_dict=theme.colors)
    curve_plot = ax.plot(wave_array / z_corr, comb_array, **line_format)

    # Merged components
    if line_i.group == 'b':

        mplcursor_list = []

        for j, y_arr in enumerate(flux_array):
            line_format = color_selector(line_i.list_comps[j].label, line_i.measurements.observations, j, n_comps=flux_array.shape[0],
                                         scale_dict=theme.plt, colors_dict=theme.colors)
            single_array = (cont_array + y_arr) * z_corr / spec.norm_flux
            curve_plot = ax.plot(wave_array/z_corr, single_array, **line_format)

            mplcursor_list.append((line_i.list_comps[j].label, line_i.list_comps[j].latex_label, curve_plot))

    else:
        mplcursor_list = [(line_i.label, line_i.latex_label, curve_plot)]

    return mplcursor_list


def spec_bands_plotter(ax, bands, x, y, z_corr, redshift, match_color=theme.colors['match_line']):

    # Compute bands limits
    w3 = bands.w3.values * (1 + redshift)
    w4 = bands.w4.values * (1 + redshift)
    idcs_bands = np.searchsorted(x, np.array([w3, w4]))

    # Loop through the detections and plot the names # TODO add reviewer here
    for i, line_label in enumerate(bands.index):
        line = Line.from_transition(line_label, data_frame=bands, verbose=False)
        try:

            # Get the max flux on the region making the exception for 1 pixel bands
            idx_w3, idx_w4 = idcs_bands[:, i]
            max_region = np.max(y[idx_w3:idx_w4]) if idx_w3 != idx_w4 else y[idx_w3]
            x_region = x[idx_w3:idx_w4]

            # Band shade area
            label = 'Matched line' if i == 0 else '_'
            span = ax.axvspan(x_region[0]/z_corr, x_region[-1]/z_corr, label=label, alpha=0.30, color=match_color, ec='none')

            # Label area
            # text = line_label
            x_text = line.wavelength * (1 + redshift) / z_corr
            y_text = max_region * 0.9 * z_corr
            # ax.text(x_text, y_text, text, rotation=270)
            # print('seguro', text)

            ax.annotate(line_label, xy=(line.wavelength * (1 + redshift) / z_corr, max_region * z_corr),
                        xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", rotation=270)
        except:
            print(LiMe_Error(f"Error plotting the band for line {line}"))

    # Plot the detected line peaks
    if 'signal_peak' in bands.columns:
        idcs_peaks = bands['signal_peak'].to_numpy()
        idcs_peaks = idcs_peaks[(~np.isnan(idcs_peaks) | np.isfinite(idcs_peaks))].astype(int)
        ax.scatter(x[idcs_peaks] / z_corr, y[idcs_peaks] * z_corr, label='Peaks',
                     facecolors='none', edgecolors=theme.colors['peak'])

    return


def spec_components_plotter(spec, ax, wave_plot, flux_plot, z_corr):

    if spec.infer.pred_arr is not None:

        # Use np.histogram to get the counts in each bin
        categories = np.sort(np.unique(spec.infer.pred_arr))
        legend_scatter = []

        for category in categories:
            if category != 0:

                # Get category properties
                feature_name = spec.infer.model_mgr.medium.number_feature_dict[category]
                feature_color = aspect.cfg['colors'][feature_name]
                idcs_feature = spec.infer.pred_arr == category
                legend_scatter.append(mlines.Line2D([], [], marker='o', color='w',
                                                    markerfacecolor=feature_color, markersize=8, label=feature_name))

                # Count the pixels for each category
                bins = [40, 60, 80, 100]
                counts, _ = np.histogram(spec.infer.conf_arr[idcs_feature], bins=bins)
                for idx_conf, count_conf in enumerate(counts):
                    if count_conf > 0:
                        # Get indeces matching the detections
                        idcs_count = np.where((bins[idx_conf] < spec.infer.conf_arr[idcs_feature]) &
                                              (spec.infer.conf_arr[idcs_feature] <= bins[idx_conf + 1]))[0]
                        idcs_nonnan = np.where(idcs_feature)[0][idcs_count]  # Returns indices where mask is True

                        # Generate nan arrays with the data to avoid filling non detections
                        wave_nan, flux_nan = np.full(wave_plot.size, np.nan), np.full(flux_plot.size, np.nan)
                        wave_nan[idcs_nonnan] = wave_plot[idcs_nonnan] / z_corr
                        flux_nan[idcs_nonnan] = flux_plot[idcs_nonnan] * z_corr

                        # Plot with the corresponding colors and linestyle
                        ax.step(wave_nan, flux_nan, label=feature_name, where='mid', color=feature_color,
                                     linestyle=category_conf_styles[idx_conf])

            # Legend category
            legend_category = ax.legend(handles=legend_scatter, edgecolor=theme.colors['fg'])
            legend_category.get_frame().set_linewidth(0.5)
            ax.add_artist(legend_category)

            # Create the extra legend with the confidence
            line1 = mlines.Line2D([], [], color=theme.colors['fg'], linestyle=':', label="> 40% conf.")
            line2 = mlines.Line2D([], [], color=theme.colors['fg'], linestyle='--', label="> 60% conf.")
            line3 = mlines.Line2D([], [], color=theme.colors['fg'], linestyle='-', label="> 80% conf.")
            legend_conf = ax.legend(handles=[line1, line2, line3], loc="lower center", ncol=3,
                                         edgecolor=theme.colors['fg'])
            legend_conf.get_frame().set_linewidth(0.5)

    else:
        _logger.warning(f'The observation does not have a components array. Please run "Spec.infer.components"')

    return


def spec_mask_plotter(axis, idcs_mask, x, y, z_corr, log=None, line_list=None, color_dict=theme.colors):

    # Spectrum mask
    if (idcs_mask is not None) and np.any(idcs_mask) and np.any(~np.isnan(y[idcs_mask])):
        # if not np.all(np.isnan(y_mask)): # Is this a new # TODO Do we need a special case all are masked?
        axis.scatter(x[idcs_mask]/z_corr, y[idcs_mask]*z_corr, marker='x', label='Masked pixels',
                     color=color_dict['mask_marker'])

    # Line masks
    if (log is not None) and ('pixel_mask' in log.columns):

        # Check lines which have been measured and have pixel mask
        line_arr = log.loc[log["pixel_mask"] != 'no'].drop_duplicates(subset="pixel_mask").index.to_numpy()
        line_arr = line_arr if line_list is None else np.intersect1d(line_arr, line_list)

        # Plot the data
        if line_arr.size > 0:
            for i, line in enumerate(line_arr):
                line_mask_limits = format_line_mask_option(log.loc[line, 'pixel_mask'], x)
                idcsMask = (x[:, None] >= line_mask_limits[:, 0]) & (x[:, None] <= line_mask_limits[:, 1])
                idcsMask = idcsMask.sum(axis=1).astype(bool)
                if idcsMask.sum() >= 1:
                    axis.scatter(x[idcsMask]/z_corr, y[idcsMask]*z_corr, marker="x", color=color_dict['mask_marker'])

    return


def line_band_plotter(axis, x, y, z_corr, idcs_mask, label, color_dict, show_central=True, show_adjacent=True):

    # Security check for low selection
    # TODO check this error crashing when continua bands are very small, need a better error message
    if y[idcs_mask[2]:idcs_mask[3]].size > 1:

        # Lower limit for the filled region
        if show_adjacent:
            low_lim = np.min(y[idcs_mask[0]:idcs_mask[5]])
            low_lim = 0 if np.isnan(low_lim) else low_lim
            x_interval = x[idcs_mask[2]:idcs_mask[3]]
            y_interval = y[idcs_mask[2]:idcs_mask[3]]
        else:
            x_interval = x[idcs_mask[2]:idcs_mask[3]]
            y_interval = y[idcs_mask[2]:idcs_mask[3]]
            m = (y_interval[-1] - y_interval[0]) / (x_interval[-1] - x_interval[0])
            n = y_interval[0] - m * x_interval[0]
            low_lim = m * x_interval + n

        # Central bands
        if show_central:
            axis.fill_between(x_interval / z_corr, low_lim * z_corr, y_interval * z_corr,
                              facecolor=color_dict['line_band'], step='mid', alpha=0.25)

        # Continua bands exclusion
        if show_adjacent:
            axis.fill_between(x[idcs_mask[0]:idcs_mask[1]] / z_corr, low_lim * z_corr,
                              y[idcs_mask[0]:idcs_mask[1]] * z_corr,
                              facecolor=color_dict['cont_band'], step='mid', alpha=0.25)
            axis.fill_between(x[idcs_mask[4]:idcs_mask[5]] / z_corr, low_lim * z_corr,
                              y[idcs_mask[4]:idcs_mask[5]] * z_corr,
                              facecolor=color_dict['cont_band'], step='mid', alpha=0.25)

    else:
        _logger.warning(f'The {label} band plot interval contains less than 1 pixel')

    return


def line_band_scaler(axis, y, y_scale, scale_dict=theme.plt):

    # If non-provided auto-decide
    if y_scale == 'auto':

        # Limits for the axes, ignore the divide by zero warning
        with np.errstate(divide='ignore', invalid='ignore'):
            neg_check = np.any(y < 0)
            y_max, y_min = np.nanmax(y), np.nanmin(y)
            ratio = np.abs(y_max/y_min)
            if (ratio > 25) or (ratio < 0.06):
                if neg_check:
                    if np.sum(y>0) > 1:
                        y_scale = {'value': 'symlog', 'linthresh': min(np.ceil(np.abs(y_min)), np.min(y[y>0]))}
                    else:
                        y_scale = {'value': 'symlog', 'linthresh': np.ceil(np.abs(y_min))}
                else:
                    y_scale = {'value': 'log'}
            else:
                y_scale = {'value': 'linear'}

        axis.set_yscale(**y_scale)

        if y_scale["value"] != 'linear':
            axis.text(0.12, 0.8, f'${y_scale["value"]}$',
                      fontsize=scale_dict['textsize_notes'], ha='center', va='center',
                      transform=axis.transAxes, alpha=0.5, color=theme.colors['fg'])
    else:
        axis.set_yscale(y_scale)

    return


def line_residual_plotter(axis, spec, line, x, y, err, z_corr, scale_dict=theme.plt):

    # Continuum level
    cont_level = line.measurements.cont / spec.norm_flux
    cont_std = line.measurements.cont_err / spec.norm_flux

    # Compute the profile arrays
    cont_array = line.measurements.m_cont * x + line.measurements.n_cont
    flux_array = line_profile_generator(line, x)
    comb_array = (flux_array.sum(axis=0) + cont_array)

    # Lower plot residual
    label_residual = r'$\frac{F_{obs} - F_{fit}}{F_{cont}}$'
    residual = ((y - comb_array / spec.norm_flux) / cont_level)
    axis.step(x / z_corr, residual * z_corr, where='mid', color=theme.colors['fg'],
              linewidth=scale_dict['spectrum_width'])

    # Shade Continuum flux standard deviation # TODO revisit this calculation
    label = r'$\sigma_{Continuum}/\overline{F_{cont}}$'
    y_limit = cont_std / cont_level
    axis.fill_between(x / z_corr, -y_limit, +y_limit, facecolor='yellow', alpha=0.5, label=label)

    # Shade the pixel error spectrum if available:
    if err is not None:
        label = r'$\sigma_{pixel}/\overline{F(cont)}$'
        err_norm = err / cont_level
        axis.fill_between(x / z_corr, -err_norm * z_corr, err_norm * z_corr, label=label, facecolor='salmon', alpha=0.3)

    # Residual y axis limit from std at line location
    idx_w3, idx_w4 = np.searchsorted(x, np.array(line.mask[2:4]) * (1 + spec.redshift))
    resd_limit = np.std(residual[idx_w3:idx_w4]) * 5

    try:
        axis.set_ylim(-resd_limit, resd_limit)
    except ValueError:
        _logger.warning(f'Nan or inf entries in axis limit for input bands')

    # Residual plot labeling
    axis.legend(loc='center right')
    axis.set_ylabel(label_residual)

    # Spectrum mask
    # _masks_plot(axis, [list_comps[0]], x, y, z_corr, log, spec_mask, color_dict=theme.colors)

    return



class SpectrumFigures:

    def __init__(self, spectrum):

        # Lime spectrum object with the scientific data
        self._spec = spectrum

        # Container for the matplotlib figures
        self.fig, self.ax = None, None

        return

    def reset_figure(self):

        if self.fig is not None and isinstance(self.fig, figure.Figure):
            plt.close(self.fig)

        if self.ax is not None:
            self.ax = None

        return

    def spectrum(self, fname=None, label=None, bands=None, rest_frame=False, log_scale=False,
                 show_profiles=True, show_cont=False, show_err=False, show_masks=True, show_components=False,
                 in_fig=_NO_FIG, fig_cfg=None, ax_cfg=None, maximize=False):

        """
        Plot the spectrum flux as a function of wavelength.

        This method generates a 1D spectrum visualization, optionally including
        fitted profiles, continua, error regions, and masks. The plot can be
        displayed interactively or saved directly to a file.

        Parameters
        ----------
        fname : str, optional
            Output file path for saving the plot (e.g., ``"spectrum.png"``).
            If not provided, the plot is displayed in a new window.
        label : str, optional
            Label for the spectrum in the plot legend. Default is ``"Observed spectrum"``.
        bands : pandas.DataFrame, str, or pathlib.Path, optional
            Bands dataframe or path to its file. If provided, line bands are
            overplotted (see :ref:`bands documentation <line-bands-doc>`).
        rest_frame : bool, optional
            If ``True``, plot the spectrum in rest-frame wavelengths. Default is
            ``False`` (observed frame).
        log_scale : bool, optional
            If ``True``, display the flux on a logarithmic y-scale. Default is
            ``False``.
        show_profiles : bool, optional
            If ``True`` (default), overlay the fitted line profiles stored in
            the spectrum’s frame.
        show_cont : bool, optional
            If ``True``, overlay the fitted continuum and its 1σ uncertainty
            region. Default is ``False``.
        show_err : bool, optional
            If ``True``, fill a shaded region representing flux uncertainties
            (``err_flux``). Default is ``False``.
        show_masks : bool, optional
            If ``True`` (default), display masked pixels as red markers.
        show_components : bool, optional
            If ``True``, plot spectral components as colored sections in the spectrum if available
        in_fig : matplotlib.figure.Figure, optional
            Existing Matplotlib figure to plot into. If not provided, a new
            figure is created automatically.
        fig_cfg : dict, optional
            Matplotlib figure configuration dictionary (e.g., size, DPI).
            See `matplotlib.RcParams
            <https://matplotlib.org/stable/api/matplotlib_configuration_api.html#matplotlib.RcParams>`_.
        ax_cfg : dict, optional
            Axes configuration dictionary with optional keys:
            ``"xlabel"``, ``"ylabel"``, and ``"title"``.
            Missing keys fall back to defaults.
        maximize : bool, optional
            If ``True``, maximize the plot window after rendering. Default is
            ``False``.

        Notes
        -----
        - The figure is constructed using the current theme defaults (see
          ``theme.fig_defaults`` and ``theme.ax_defaults``).
        - If :mod:`mplcursors` is installed, interactive tooltips are available:
          left-click on a fitted profile to display its parameters, and right-click
          to remove the annotation.
        - If both ``fname`` and ``in_fig`` are omitted, the figure is shown
          interactively on screen.
        - All wavelength and flux units follow the current spectrum settings
          (see :class:`lime.Spectrum` attributes ``units_wave`` and ``units_flux``).

        Examples
        --------
        Plot a basic observed-frame spectrum:

        >>> spec.plot.spectrum()

        Plot in rest-frame wavelengths with fitted profiles and continuum:

        >>> spec.plot.spectrum(rest_frame=True, show_profiles=True, show_cont=True)

        Save the spectrum with logarithmic scaling:

        >>> spec.plot.spectrum(fname="spectrum_log.png", log_scale=True)

        """

        # Clear previous figure
        self.reset_figure()

        # Display check for input figures
        display_check = True if in_fig is _NO_FIG else False

        # Set figure format with the user 2_guides overwriting the default conf
        legend_check = True if label is not None else False

        # Adjust the default theme
        plt_cfg = theme.fig_defaults(fig_cfg)
        ax_labels_cfg = theme.ax_defaults(ax_cfg, self._spec)

        # Create and fill the figure
        with ((rc_context(plt_cfg))):

            # Establish figure
            self.fig = plt.figure() if (in_fig is None) or (in_fig is _NO_FIG) else in_fig

            # Establish the axes
            self.ax = self.fig.add_subplot()
            self.ax.set(** ax_labels_cfg)

            # Reference _frame for the plot
            wave_plot, flux_plot, err_plot, z_corr, idcs_mask = frame_mask_switch(self._spec, rest_frame)

            # Plot the spectrum
            self.ax.step(wave_plot/z_corr, flux_plot * z_corr, label=label, where='mid', color=theme.colors['fg'],
                         linewidth=theme.plt['spectrum_width'], zorder=10)

            # Plot the uncertainty shaded area
            if show_err:
                if err_plot is not None:
                    self.ax.fill_between(x=wave_plot/z_corr,
                                         y1=(flux_plot-err_plot)* z_corr,
                                         y2=(flux_plot+err_plot) * z_corr,
                                         step='mid', alpha=0.2, color='lime', ec=None)
                else:
                    _logger.info(f'The input spectrum does not include an uncertainty array')

            # Plot bands if provided
            if bands is not None:
                spec_bands_plotter(self.ax, check_file_dataframe(bands), wave_plot, flux_plot, z_corr,
                                   self._spec.redshift)

            # Plot the fittings
            if show_profiles and self._spec.frame.size > 0:
                mplcursor_list = []
                for line_label in unique_line_arr(self._spec.frame):
                    line = Line.from_transition(line_label, data_frame=self._spec.frame)
                    mplcursor_list += spec_profile_plotter(self.ax, self._spec, line, z_corr)

                # Pop-ups
                mplcursor_parser(mplcursor_list, self._spec)

            # Plot the fit continuum
            if show_cont and self._spec.cont is not None:
                self.ax.plot(wave_plot/z_corr, self._spec.cont*z_corr, label='Continuum',
                             color=theme.colors['cont'], linestyle='--')

                low_limit, high_limit = self._spec.cont-self._spec.cont_std, self._spec.cont + self._spec.cont_std
                self.ax.fill_between(wave_plot/z_corr, low_limit*z_corr, high_limit*z_corr, alpha=0.2,
                                   color=theme.colors['inspection_uncertainty'])

            # Show components
            if show_components and hasattr(self._spec.infer, "pred_arr"):
                legend_check = False
                spec_components_plotter(self._spec, self.ax, wave_plot, flux_plot, z_corr)

            # Show the masks
            if show_masks:
                spec_mask_plotter(self.ax, idcs_mask, wave_plot, flux_plot, z_corr, self._spec.frame)

            # Switch y_axis to logarithmic scale if requested
            if log_scale:
                self.ax.set_yscale('log')

            # Add or remove legend according to the plot type:
            if legend_check:
                self.ax.legend()

            # By default, plot on screen unless an output address is provided
            save_close_fig_swicth(fname, 'tight', self.fig, maximize, display_check)

        return

    def grid(self, fname=None, rest_frame=True, y_scale='auto', n_cols=6, n_rows=None, col_row_scale=(2, 1.5),
             show_profiles=True, show_adjacent=False, in_fig=None, fig_cfg=None, ax_cfg=None, maximize=False):

        """
        Plot measured line cutouts and profile from the spectrum frame in a grid of subplots.

        This function arranges per-line panels into an ``n_rows × n_cols`` grid.
        The figure can be shown in a new window or saved to disk. If :mod:`mplcursors` is installed,
        left-clicking a fitted profile shows its parameters; right-click removes the annotation.

        Parameters
        ----------
        fname : str or pathlib.Path, optional
            File path to save the figure. If not provided, the figure is shown in a new window.
        rest_frame : bool, optional
            If ``True``, plot panels in rest-frame wavelength. Default is ``False``.
        y_scale : {"auto", "linear", "log"}, optional
            Y-axis scaling per panel. ``"auto"`` chooses a sensible scale; otherwise
            use Matplotlib’s ``"linear"``, ``"log"`` or ``symlog``. Default is ``"auto"``.
        n_cols : int, optional
            Number of columns in the grid. Default is ``6``.
        n_rows : int, optional
            Number of rows in the grid. If omitted, it may be inferred from the
            number of lines and ``n_cols``.
        col_row_scale : tuple of (float, float), optional
            Multiplicative factors for panel width and height (in inches).
            Default is ``(2.0, 1.5)``.
        show_profiles : bool, optional
            If ``True``, overlay fitted profiles where available. Default is ``False``.
        show_adjacent : bool, optional
            If ``True``, overlay the adjacent continua bands on the line plot cell``.
        in_fig : matplotlib.figure.Figure, optional
            Existing Matplotlib figure to plot into. If not provided, a new
            figure is created automatically.
        fig_cfg : dict, optional
            Figure configuration (e.g., size, DPI). See `matplotlib.RcParams
            <https://matplotlib.org/stable/api/matplotlib_configuration_api.html#matplotlib.RcParams>`_.
        ax_cfg : dict, optional
            Axes label/title overrides with any of the keys ``"xlabel"``, ``"ylabel"``,
            and ``"title"``. Missing keys fall back to defaults.
        maximize : bool, optional
            If ``True``, maximize the window after rendering. Default is ``False``.

        Notes
        -----
        - Lines are selected from the current spectrum lines log (``Spectrum.frame``).
        - If both ``output_address`` is omitted and no interactive backend is
          available, the figure will not display; provide a filename to save.

        Examples
        --------
        Save a grid to file:

        >>> fig = spec.plot.lines_grid(output_address="lines_grid.png", n_cols=6, n_rows=3)

        Show fitted profiles with logarithmic scaling:

        >>> fig = spec.plot.lines_grid(include_fits=True, y_scale="log")

        Enlarge panels and plot in rest frame:

        >>> fig = spec.plot.lines_grid(rest_frame=True, col_row_scale=(2.5, 2.0))
        """


        # Check lines have been measured
        if self._spec.frame.index.size > 0:

            # Clear previous figure
            self.reset_figure()

            # Display check for input user figure
            display_check = True if in_fig is None else False

            # Get the line list and compute the number of rows
            line_list = unique_line_arr(self._spec.frame)

            # Compute the rows and columns number
            if line_list.size > n_cols:
                n_rows = n_rows or int(np.ceil(line_list.size / n_cols))
            else:
                n_cols, n_rows = line_list.size, 1

            # User configuration overwrites default configuration
            size_conf = {'figure.figsize': (n_cols * col_row_scale[0], n_rows * col_row_scale[1])}
            size_conf = size_conf if fig_cfg is None else {**size_conf, **fig_cfg}
            plt_cfg = theme.fig_defaults(size_conf, fig_type='grid')

            # Launch the interative figure
            with rc_context(plt_cfg):

                # Figure structure
                self.fig = plt.figure() if in_fig is None else in_fig
                grid_spec = self.fig.add_gridspec(nrows=n_rows, ncols=n_cols)
                self.ax =[None] * line_list.size

                # Container for mpl_cursors
                mplcursor_list = []

                # Loop throught the lines and plot the line for each axis.
                for i, line_label in enumerate(line_list):
                    self.ax[i] = plt.subplot(grid_spec[i])

                    # Check components
                    line_i = Line.from_transition(line_label, data_frame=self._spec.frame)

                    # Reference _frame for the plot
                    wave_plot, flux_plot, err_plot, z_corr, idcs_mask = frame_mask_switch(self._spec, rest_frame)

                    # Establish the limits for the line spectrum plot
                    idcs_m = np.searchsorted(wave_plot, np.array(line_i.mask) * (1 + self._spec.redshift))
                    idx_blue = idcs_m[0] - 5 if idcs_m[0] > 5 else idcs_m[0]
                    idx_red = idcs_m[-1] + 5 if idcs_m[-1] < idcs_m[-1] + 5 else idcs_m[-1]

                    # Plot the spectrum
                    self.ax[i].step(wave_plot[idx_blue:idx_red] / z_corr, flux_plot[idx_blue:idx_red] * z_corr,
                               where='mid', color=theme.colors['fg'], linewidth=theme.plt['spectrum_width'])

                    # Continuum bands
                    line_band_plotter(self.ax[i], wave_plot, flux_plot, z_corr, idcs_m, line_i.label, theme.colors,
                                      show_central=True, show_adjacent=show_adjacent)

                    # Plot the profiles and link to the mplcursors pop-ups
                    if show_profiles and (line_i.measurements is not None):
                        mplcursor_list += spec_profile_plotter(self.ax[i], self._spec, line_i, z_corr)

                    # Set the scale
                    line_band_scaler(self.ax[i], flux_plot[idx_blue:idx_red] * z_corr, y_scale)

                    # # Plot the masked pixels
                    # _masks_plot(self.ax[0], [line_i], wave_plot[idx_blue:idx_red], flux_plot[idx_blue:idx_red],
                    #             z_corr, log, idcs_mask[idx_blue:idx_red], theme.colors)

                    # Formatting the figure
                    self.ax[i].yaxis.set_major_locator(plt.NullLocator())
                    self.ax[i].xaxis.set_major_locator(plt.NullLocator())

                    self.ax[i].update({'title': line_i.latex_label})
                    self.ax[i].yaxis.set_ticklabels([])
                    self.ax[i].axes.yaxis.set_visible(False)

            # Pop-ups
            mplcursor_parser(mplcursor_list, self._spec)

            # By default, plot on screen unless an output address is provided
            save_close_fig_swicth(fname, 'tight', self.fig, maximize, display_check)

        else:
            _logger.info('The bands log does not contain lines')

        return

    def bands(self, label=None, bands=None, fname=None, rest_frame=False, y_scale='auto', show_profile=True,
              show_err=False, show_bands=True, show_cont=False, fig_cfg=None, ax_cfg=None, in_fig=None, maximize=False):

        """
        Plot a single spectral line with optional profile fit and continua bands.

        If ``label`` is not provided, the last measured line in the spectrum log is
        selected. A bands table (DataFrame or file path) may be supplied to resolve
        the target line or to fetch its band limits (see
        :ref:`bands documentation <line-bands-doc>`).

        Parameters
        ----------
        label : str, optional
            Line label to display. If ``None``, use the last line found in the
            spectrum’s measurement log.
        bands : pandas.DataFrame, str, or pathlib.Path, optional
            Bands table or a path to it. Used to resolve the line (when needed) and
            to obtain the band edges (w1–w6).
        fname : str or pathlib.Path, optional
            File path to save the figure. If not provided, the plot is shown
            interactively.
        rest_frame : bool, optional
            If ``True``, plot using rest-frame wavelengths. Default is ``False``.
        y_scale : {"auto", "linear", "log"}, optional
            Y-axis scaling for the panel. ``"auto"`` chooses a sensible scale based
            on the local data; otherwise use Matplotlib’s ``"linear"`` or ``"log"``.
            Default is ``"auto"``.
        show_profile : bool, optional
            If ``True`` (default), overlay the fitted line profile and a residuals
            panel when fit results are available.
        show_err : bool, optional
            If ``True``, fill a shaded band showing the flux uncertainty (``err_flux``)
            within the central line region. Default is ``False``.
        show_bands : bool, optional
            If ``True`` (default), draw the central and continuum bands for reference.
        fig_cfg : dict, optional
            Matplotlib figure configuration (e.g., size, DPI). See
            `matplotlib.RcParams
            <https://matplotlib.org/stable/api/matplotlib_configuration_api.html#matplotlib.RcParams>`_.
        ax_cfg : dict, optional
            Axes configuration with any of the keys ``"xlabel"``, ``"ylabel"``,
            and ``"title"``. Missing keys fall back to defaults.
        in_fig : matplotlib.figure.Figure, optional
            Existing figure to plot into. If ``None``, a new figure is created.
        maximize : bool, optional
            If ``True``, maximize the window after rendering. Default is ``False``.

        Notes
        -----
        - If :mod:`mplcursors` is installed, profile components become interactive:
          left-click to show fit parameters, right-click to remove the annotation.
        - When ``show_profile`` is enabled and fit results exist, the layout includes
          a residuals axis aligned in wavelength with the main panel.
        - Masked pixels are indicated within the line window; wavelength and flux
          units follow the current spectrum settings.

        Examples
        --------
        Plot the last measured line in observed frame:

        >>> spec.plot.bands()

        Plot a specific line using bands from a file and show uncertainties:

        >>> spec.plot.bands("O3_5007A", bands="my_bands.xlsx", show_err=True)

        Save a rest-frame plot with a logarithmic scale:

        >>> spec.plot.bands("H1_4861A", rest_frame=True, y_scale="log", fname="hbeta_rest.png")

        """

        # Check which line should be plotted
        line = parse_bands_arguments(label, bands, self._spec.frame)#, self._spec.norm_flux)

        # Proceed to plot
        if line is not None:

            # Clear previous figure
            self.reset_figure()

            # Display check for input figures
            display_check = False if in_fig is not None else True

            # Check if the line has measuring data
            show_profile = show_profile and (line.measurements.intg_flux is not None)

            # Adjust the default theme
            plt_cfg = theme.fig_defaults(fig_cfg, fig_type='bands')
            ax_labels_cfg = theme.ax_defaults(ax_cfg, self._spec)

            # Create and fill the figure
            with rc_context(plt_cfg):

                # Establish figure
                self.fig = plt.figure() if in_fig is None else in_fig

                # Establish the axes
                if show_profile:
                    grid_ax = self.fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3, 1])
                    spec_ax = plt.subplot(grid_ax[0])
                    resid_ax = plt.subplot(grid_ax[1], sharex=spec_ax)
                    self.ax = (spec_ax, resid_ax)
                else:
                    self.ax = [self.fig.add_subplot()]
                self.ax[0].set(**ax_labels_cfg)

                # Reference _frame for the plot
                wave_plot, flux_plot, err_plot, z_corr, idcs_mask = frame_mask_switch(self._spec, rest_frame)

                # Establish the limits for the line spectrum plot
                idcs_bands = line.index_bands(self._spec.wave, self._spec.redshift, just_band_edges=True)

                # Plot the spectrum
                self.ax[0].step(wave_plot[idcs_bands[0]:idcs_bands[5]]/z_corr, flux_plot[idcs_bands[0]:idcs_bands[5]] * z_corr,
                              where='mid', color=theme.colors['fg'], linewidth=theme.plt['spectrum_width'])

                # Line bands
                if show_profile:
                    line_band_plotter(self.ax[0], wave_plot, flux_plot, z_corr, idcs_bands, line, color_dict=theme.colors,
                                      show_central=True, show_adjacent=show_bands)

                    # Central point
                    self.ax[0].scatter(line.measurements.peak_wave/z_corr, line.measurements.cont*z_corr/self._spec.norm_flux,
                                       marker='+', facecolor=theme.colors['fg'], linewidth=float(theme.plt['spectrum_width'])/2)

                # Plot the uncertainty
                if show_err and (self._spec.err_flux is not None):
                    err_plot = self._spec.err_flux.data
                    self.ax[0].fill_between(x=wave_plot[idcs_bands[2]:idcs_bands[3]] / z_corr,
                                          y1=(flux_plot[idcs_bands[2]:idcs_bands[3]] - err_plot[idcs_bands[2]:idcs_bands[3]]) * z_corr,
                                          y2=(flux_plot[idcs_bands[2]:idcs_bands[3]] + err_plot[idcs_bands[2]:idcs_bands[3]]) * z_corr,
                                          step='mid', alpha=1, color=theme.colors['line_band'], ec=None)

                # Add the fitting results
                if show_profile:

                    # Plot profile
                    mplcursor_list = spec_profile_plotter(self.ax[0], self._spec, line, z_corr)

                    # Residual flux component
                    line_residual_plotter(self.ax[1], self._spec, line,
                                            wave_plot[idcs_bands[0]:idcs_bands[5]],
                                            flux_plot[idcs_bands[0]:idcs_bands[5]],
                                            err_plot[idcs_bands[0]:idcs_bands[5]] if err_plot is not None else None,
                                            z_corr)

                    # Pop-ups
                    mplcursor_parser(mplcursor_list, self._spec)

                # Plot the masked pixels
                spec_mask_plotter(self.ax[0],
                                  idcs_mask[idcs_bands[0]:idcs_bands[5]],
                                  wave_plot[idcs_bands[0]:idcs_bands[5]],
                                  flux_plot[idcs_bands[0]:idcs_bands[5]],
                                  z_corr, self._spec.frame, line.param_arr('label'))

                # Display the legend
                if show_profile:
                    self.ax[0].legend()

                # Set the scale
                line_band_scaler(self.ax[0], y=flux_plot[idcs_bands[0]:idcs_bands[5]] * z_corr, y_scale=y_scale)

                # Mask plotter
                # spec_mask_plotter(self.)


                if show_cont:
                    if self._spec.cont is not None:
                        # Plot the spectrum default."cont_width" = '0.5' cont
                        self.ax[0].plot(wave_plot[idcs_bands[0]:idcs_bands[5]] / z_corr,
                                        self._spec.cont[idcs_bands[0]:idcs_bands[5]] * z_corr,
                                        color=theme.colors['cont'], linewidth=2,
                                        linestyle="dashed", label='fitted continuum')

                    else:
                        _logger.info('The input spectrum does not have have a fitted continuum')

                # By default, plot on screen unless an output address is provided
                save_close_fig_swicth(fname, 'tight', self.fig, maximize, display_check)

        else:
            _logger.info(f'The line "{line}" was not found in the spectrum log for plotting.')

        return

    def velocity_profile(self, line=None, fname=None, y_scale='linear', in_fig=None, fig_cfg=None, ax_cfg=None,
                         maximize=False):

        """
        Plot a line with the velocity profile percentiles.

        This visualization converts the wavelength frame to the velocity frame and displays the velocity
        percentils (``v_1``, ``v_5``, ``v_10``, ``v_50``, ``v_90``, ``v_95``, ``v_99``), median velocity (``v_med``)
        the full width at zero intensity (``FWZI``) and the ``w80 = v_90 - v_10`` width.

        Parameters
        ----------
        line : str, optional
            Line label to display. If ``None`` the last line fitted is displayed.
        band : array-like, optional
            Six-element band definition ``[w1, w2, w3, w4, w5, w6]`` **in rest-frame
            wavelength** used to window the spectrum around the target line. If
            ``None``, the band is looked up from the line database or the spectrum log.
        y_scale : {"linear", "log"}, optional
            Y-axis scale for the flux in the velocity panel. Default is ``"linear"``.
        in_fig : matplotlib.figure.Figure, optional
            Existing figure to plot into. If ``None``, a new figure is created.
        fig_cfg : dict, optional
            Matplotlib figure configuration (e.g., size, DPI). See `matplotlib.RcParams
            <https://matplotlib.org/stable/api/matplotlib_configuration_api.html#matplotlib.RcParams>`_.
        ax_cfg : dict, optional
            Axes configuration with optional keys such as ``"xlabel"``, ``"ylabel"``,
            and ``"title"``. Missing keys fall back to defaults for velocity plots.
        fname : str or pathlib.Path, optional
            Output file path to save the figure. If not provided, the figure is shown
            interactively.
        maximize : bool, optional
            If ``True``, maximize the window after rendering. Default is ``False``.

        Notes
        -----
        - The velocity axis is computed as
          :math:`v = c\\,\\frac{\\lambda - \\lambda_{\\rm peak}}{\\lambda_{\\rm peak}}`,
          with ``c`` in km/s and ``λ_peak`` from the line fit (``measurements.peak_wave``).
        - The plotting window is derived from the provided or resolved band and the
          current spectrum redshift; rest-frame band edges are shifted to the observed
          frame internally.
        - Axis labels and styling use the velocity-plot defaults from the active theme;
          pass ``ax_cfg`` and ``fig_cfg`` to override.

        Examples
        --------
        Plot the velocity profile for a measured line:

        >>> spec.plot.velocity_profile(line="O3_5007A")

        Plot the velocity profile and save the plot:

        >>> band = [4995, 5000, 5004, 5011, 5016, 5022]  # rest-frame Angstroms
        >>> spec.plot.velocity_profile(line="O3_5007A", band=band, fname="o3_velocity.png")

        Use logarithmic y-scale and a custom title:

        >>> spec.plot.velocity_profile("H1_4861A", y_scale="log", ax_cfg={"title": "Hβ velocity profile"})
        """

        # Check which line should be plotted
        line = parse_bands_arguments(line, None, self._spec.frame)

        # Display check for the user figures
        display_check = True if in_fig is None else False

        # Adjust the default theme
        PLT_CONF = theme.fig_defaults(fig_cfg)
        AXES_CONF = theme.ax_defaults(ax_cfg, self._spec, fig_type='velocity')

        # Line spectrum
        wave_plot, flux_plot, err_plot, z_corr, idcs_mask = frame_mask_switch(self._spec, rest_frame=False)

        # Establish the limits for the line spectrum plot
        idcs_bands = line.index_bands(self._spec.wave, self._spec.redshift, just_band_edges=True)

        # Velocity spectrum for the line region
        flux_plot = flux_plot[idcs_bands[0]:idcs_bands[5]]
        cont_plot = line.measurements.m_cont * wave_plot[idcs_bands[0]:idcs_bands[5]] + line.measurements.n_cont
        vel_plot = c_KMpS * (wave_plot[idcs_bands[0]:idcs_bands[5]] - line.measurements.peak_wave) / line.measurements.peak_wave

        # Line edges
        w_limits = np.array([line.measurements.w_i, line.measurements.w_f])
        v_i, v_f = c_KMpS * (w_limits - line.measurements.peak_wave) / line.measurements.peak_wave

        # Create and fill the figure
        with rc_context(PLT_CONF):

            # Figure and its axis
            if in_fig is None:
                self._fig, self._ax = plt.subplots()
            else:
                self._fig = in_fig
                self._ax = self._fig.add_subplot()

            self._ax.set(**AXES_CONF)
            trans = self._ax.get_xaxis_transform()

            # Plot the spectrum
            self._ax.step(vel_plot, flux_plot, label=line.latex_label, where='mid', color=theme.colors['fg'],
                          linewidth=theme.plt['spectrum_width'])

            # Velocity percentiles
            target_percen = ['v_1', 'v_5', 'v_10', 'v_50', 'v_90', 'v_95', 'v_99']
            for i_percentil, percentil in enumerate(target_percen):

                vel_per = line.measurements.__getattribute__(percentil)
                label_text = None if i_percentil > 0 else r'$v_{Pth}$'
                self._ax.axvline(x=vel_per, label=label_text, color=theme.colors['fg'], linestyle='dotted', alpha=0.5)

                label_plot = r'$v_{{{}}}$'.format(percentil[2:])
                self._ax.text(vel_per, 0.80, label_plot, ha='center', va='center', rotation='vertical',
                              backgroundcolor=theme.colors['bg'], transform=trans, alpha=0.5)

            # Velocity edges
            label_v_i, label_v_f = r'$v_{{0}}$', r'$v_{{100}}$'
            self._ax.axvline(x=v_i, alpha=0.5, color=theme.colors['fg'], linestyle='dotted')
            self._ax.text(v_i, 0.80, label_v_i, ha='center', va='center', rotation='vertical',
                          backgroundcolor=theme.colors['bg'], transform=trans, alpha=0.5)

            self._ax.axvline(x=v_f, alpha=0.5, color=theme.colors['fg'], linestyle='dotted')
            self._ax.text(v_f, 0.80, label_v_f, ha='center', va='center', rotation='vertical',
                          backgroundcolor=theme.colors['bg'], transform=trans, alpha=0.5)

            # Plot the line profile
            self._ax.plot(vel_plot, cont_plot, linestyle='--', color=theme.colors['fg'])

            # Plot velocity bands
            w80 = line.measurements.v_90-line.measurements.v_10
            label_arrow = r'$w_{{80}}={:0.1f}\,Km/s$'.format(w80)
            p1 = patches.FancyArrowPatch((line.measurements.v_10, 0.4),
                                         (line.measurements.v_90, 0.4),
                                         label=label_arrow,
                                         arrowstyle='<->',
                                         color='tab:blue',
                                         transform=trans,
                                         mutation_scale=20)
            self._ax.add_patch(p1)

            # Plot FWHM bands
            label_arrow = r'$FWZI={:0.1f}\,Km/s$'.format(line.measurements.FWZI)
            p2 = patches.FancyArrowPatch((v_i, 0),
                                         (v_f, 0),
                                         label=label_arrow,
                                         arrowstyle='<->',
                                         color='tab:red',
                                         transform=self._ax.transData,
                                         mutation_scale=20)
            self._ax.add_patch(p2)

            # Median velocity
            label_vmed = r'$v_{{med}}={:0.1f}\,Km/s$'.format(line.measurements.v_med)
            self._ax.axvline(x=line.measurements.v_med, color=theme.colors['fg'], label=label_vmed, linestyle='dashed', alpha=0.5)

            # Peak velocity
            label_vmed = r'$v_{{peak}}$'
            self._ax.axvline(x=0.0, color=theme.colors['fg'], label=label_vmed, alpha=0.5)

            # Set the scale
            _auto_flux_scale(self._ax, y=flux_plot, y_scale=y_scale)

            # Legend
            self._ax.legend()

            # By default, plot on screen unless an output address is provided
            save_close_fig_swicth(fname, 'tight', in_fig, maximize, display_check)

        return

    def show(self, **kwargs):

        """
        Display the current Matplotlib figure.

        This is a convenience wrapper around :func:`matplotlib.pyplot.show`, used
        to display the most recently generated plot from the current plotting object
        (e.g., spectrum, bands, or velocity profile figures).

        Parameters
        ----------
        **kwargs
            Additional keyword arguments are passed directly to
            :func:`matplotlib.pyplot.show`. Common options include:

            - ``block`` : bool, optional
              If ``True`` (default), block execution until the plot window is closed.
              If ``False``, display the plot without blocking.

        Notes
        -----
        - This method should be called after generating a plot with other ``spec.plot`` functions if you want to render it interactively.
        - If you want to add your own artist to a LiMe figure you can set ``in_fig=None`` and then use ``spec.plot.show()``.

        Examples
        --------
        Generate a spectrum plot and plot it at a certain instance:

        >>> spec.plot.spectrum(in_fig=None)
        >>> spec.plot.show()

        Generate a spectrum plot, add some data and then open figure window:

        >>> spec.plot.spectrum(in_fig=None)
        >>> spec.plot.ax.scatter(w1, f1, label='Data point')
        >>> spec.plot.show()
        """

        plt.show(**kwargs)

        return


class CubeFigures:

    def __init__(self, cube):

        # Lime spectrum object with the scientific data
        self._cube = cube

        # Container for the matplotlib figures
        self.fig, self.ax = None, None

        return

    def reset_figure(self):

        if self.fig is not None and isinstance(self.fig, figure.Figure):
            plt.close(self.fig)

        if self.ax is not None:
            self.ax = None

        return

    def cube(self, line_bg, bands=None, line_fg=None, fname=None, min_pctl_bg=60, cont_pctls_fg=(90, 95, 99),
             bg_cmap='gray', fg_cmap='viridis', bg_norm=None, fg_norm=None, masks_file=None, masks_cmap='viridis_r',
             masks_alpha=0.2, wcs=None, fig_cfg=None, ax_cfg=None, in_fig=None, maximise=False):

        """
        Plot a 2D flux map from an IFS cube with optional line contours and spatial masks.

        The background image is the **band-summed flux** of ``line_bg``; its band limits are
        taken from ``bands`` (or from the default line database if ``bands`` is not provided).
        Optionally, foreground **contours** from a second line (``line_fg``) are overplotted.
        Binary spatial masks can be drawn on top for context.

        Parameters
        ----------
        line_bg : str
            Line label for the background image (band-summed flux). Bands are resolved
            from ``bands`` or the default database. See :ref:`bands documentation <line-bands-doc>`.
        bands : pandas.DataFrame, str, or pathlib.Path, optional
            Bands table (or path) providing the six-band definitions for lines.
            If ``None``, the default LiMe bands database is used.
        line_fg : str, optional
            Line label for foreground **contours**. If provided, contours are computed
            from that line’s band-summed flux.
        fname : str or pathlib.Path, optional
            Path to save the figure. If not provided, the plot is shown in a window.
        min_pctl_bg : float, optional
            Minimum percentile of the background band-summed flux used to set the lower
            display limit when ``bg_norm`` is not supplied. Default is ``60``.
        cont_pctls_fg : tuple of float, optional
            Sorted percentiles of the foreground band-summed flux used as contour levels
            (e.g., ``(90, 95, 99)``). Default is ``(90, 95, 99)``.
        bg_cmap : str, optional
            Colormap name for the background image. Default is ``"gray"``.
        fg_cmap : str, optional
            Colormap name for the foreground contours. Default is ``"viridis"``.
        bg_norm : matplotlib.colors.Normalize, optional
            Normalization for the background image. If ``None``, a symmetric log
            normalization is used (see `Matplotlib color normalizations
            <https://matplotlib.org/stable/gallery/images_contours_and_fields/colormap_normalizations.html>`_).
        fg_norm : matplotlib.colors.Normalize, optional
            Normalization for the foreground contours. If ``None``, a logarithmic
            normalization is used (see `LogNorm
            <https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.LogNorm.html>`_).
        masks_file : str or pathlib.Path, optional
            Path to a FITS file with binary spatial masks to overlay. If provided,
            masks are rendered semi-transparently.
        masks_cmap : str, optional
            Colormap used to render the binary masks. Default is ``"viridis_r"``.
        masks_alpha : float, optional
            Alpha transparency for mask overlays (0–1). Default is ``0.2``.
        wcs : astropy.wcs.WCS, optional
            World Coordinate System to use for the axes. If ``None``, the parent cube’s
            WCS is used (if available). See `Astropy WCS
            <https://docs.astropy.org/en/stable/wcs/index.html>`_.
        fig_cfg : dict, optional
            Matplotlib figure configuration (e.g., size, DPI). See `matplotlib.RcParams
            <https://matplotlib.org/stable/api/matplotlib_configuration_api.html#matplotlib.RcParams>`_.
        ax_cfg : dict, optional
            Axes label/title overrides with any of the keys ``"xlabel"``, ``"ylabel"``,
            and ``"title"``. Missing keys fall back to defaults.
        in_fig : matplotlib.figure.Figure, optional
            Existing figure to plot into. If ``None``, a new figure is created.
        maximise : bool, optional
            If ``True``, maximize the window after rendering. Default is ``False``.


        Notes
        -----
        - **Defaults for normalizations:** when ``bg_norm``/``fg_norm`` are not supplied,
          the function uses symmetric-log for the background and log for the contours.
          The background’s lower limit is derived from the ``min_pctl_bg`` percentile.
        - Depending on the dinamic range of the flux the percentile selection can be highly unlinear.
          The user is encouraged to try different bands and percentiles to tailor the visualization criteria.
        - **Size consistency:** foreground, background, and mask images must share
          the same spatial shape; a mismatch raises an error.
        - **WCS handling:** if a valid WCS is provided (or available from the cube),
          axes are created with a WCS projection and labeled accordingly.

        Examples
        --------
        Plot [O III] as the background and Hα contours with default normalizations:

        >>> cube.plot.cube("O3_5007A", line_fg="H1_6563A")

        Save the map with custom percentiles and a different colormap:

        >>> cube.plot.cube("H1_6563A", cont_pctls_fg=(85, 92, 98),
        ...                 bg_cmap="bone", fname="halpha_map.png")

        Overlay spatial masks from a FITS file:

        >>> cube.plot.cube("O3_5007A", masks_file="O3_masks.fits", masks_alpha=0.3)
        """

        # Prepare the background image data
        line_bg, bg_image, bg_levels, bg_norm = determine_cube_images(self._cube, line_bg, bands,
                                                                      min_pctl_bg, bg_norm, contours_check=False)

        # Prepare the foreground image data
        line_fg, fg_image, fg_levels, fg_norm = determine_cube_images(self._cube, line_fg, bands,
                                                                      cont_pctls_fg, fg_norm, contours_check=True)

        # Mesh for the contours
        fg_mesh = None if line_fg is None else np.meshgrid(np.arange(0, fg_image.shape[1]),
                                                           np.arange(0, fg_image.shape[0]))

        # Load the masks
        masks_dict = load_spatial_mask(masks_file)

        # Check that the images have the same size
        check_image_size(bg_image, fg_image, masks_dict)

        # Clear previous image
        self.reset_figure()

        # Display check for input figures
        display_check = False if in_fig is not None else True

        # Use the input wcs or use the parent one
        wcs = self._cube.wcs if wcs is None else wcs
        slices = None if wcs is None else ('x', 'y', 1) if wcs.naxis == 3 else ('x', 'y')

        # Set the plot format where the user's overwrites the default
        size_conf = {'figure.figsize': (4 if masks_file is None else 5.5, 4)}
        size_conf = size_conf if fig_cfg is None else {**size_conf, **fig_cfg}
        plt_cfg = theme.fig_defaults(size_conf, fig_type='cube')
        ax_labels_cfg = theme.ax_defaults(ax_cfg, self._cube, fig_type='cube', line_bg=line_bg, line_fg=line_fg,
                                          masks_dict=masks_dict, wcs=wcs)

        # Create and fill the figure
        with rc_context(plt_cfg):

            # Make the figure and axes
            self.fig = plt.figure() if in_fig is None else in_fig
            self.ax = self.fig.add_subplot() if wcs is None else self.fig.add_subplot(projection=wcs, slices=slices)
            self.ax.update(ax_labels_cfg)

            # Plot the image
            _ = image_plot(self.ax, bg_image, fg_image, fg_levels, fg_mesh, bg_norm, fg_norm, bg_cmap, fg_cmap)

            # Plot the spatial masks
            if len(masks_dict) > 0:
                legend_hdl = spatial_mask_plot(self.ax, masks_dict, masks_cmap, masks_alpha, self._cube.units_flux)
                self.ax.legend(handles=legend_hdl, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

            # By default, plot on screen unless an output address is provided
            self.fig.canvas.draw()
            save_close_fig_swicth(fname, 'tight', self.fig, maximise, plot_check=display_check)

        return


class SampleFigures(Plotter):

    def __init__(self, sample):

        # Instantiate the dependencies
        Plotter.__init__(self)

        # Lime spectrum object with the scientific data
        self._sample = sample

        # Container for the matplotlib figures
        self._fig, self._ax = None, None
        self._legend_handle = None

        return

    def spectra(self, obj_idcs=None, log_scale=False, output_address=None, rest_frame=False, include_fits=False,
                include_err=False, legend_handle='levels', in_fig=None, in_axis=None, fig_cfg=None, ax_cfg=None, maximize=False):

        if self._sample.load_function is not None:

            legend_check = True

            norm_flux = self._sample.load_params.get('norm_flux')
            units_wave = self._sample.units_wave
            units_flux = self._sample.units_flux

            # Adjust the default theme
            PLT_CONF = theme.fig_defaults(fig_cfg)
            AXES_CONF = theme.ax_defaults(ax_cfg, units_wave, units_flux, norm_flux)

            # Get the spectra list to plot
            if obj_idcs is None:
                sub_sample = self._sample
            else:
                sub_sample = self._sample[obj_idcs]

            # Check for logs without lines # TODO we need common method for just providing the first entry
            if 'line' in sub_sample.index.names:
                obj_idcs = sub_sample.frame.droplevel('line').index.unique()
            else:
                obj_idcs = sub_sample.index.unique()

            if len(obj_idcs) > 0:

                # Create and fill the figure
                with rc_context(PLT_CONF):

                    # Generate the figure object and figures
                    self._fig, self._ax = self._plot_container(in_fig, in_axis, AXES_CONF)

                    # Loop through the SMACS_v2.0 in the sample
                    for sample_idx in obj_idcs:

                        legend_label = label_generator(sample_idx, self._sample.frame, legend_handle)
                        spec = self._sample.load_function(self._sample.frame, sample_idx, self._sample.file_address,
                                                          **self._sample.load_params)

                        # Reference _frame for the plot
                        wave_plot, flux_plot, z_corr, idcs_mask = frame_mask_switch(spec.wave, spec.flux, spec.redshift,
                                                                                    rest_frame)

                        # Plot the spectrum
                        step = self._ax.step(wave_plot / z_corr, flux_plot * z_corr, label=legend_label, where='mid',
                                      linewidth=theme.plt['spectrum_width'])

                        if include_err and spec.err_flux is not None:
                            err_plot = spec.err_flux.data
                            self._ax.fill_between(x=wave_plot / z_corr, y1=(flux_plot - err_plot) * z_corr,
                                                  y2=(flux_plot + err_plot) * z_corr,
                                                  step='mid', alpha=0.1, color=step[0].get_color())

                        # List of lines in the log
                        line_list = spec.frame.index.values

                        # Plot the fittings
                        if include_fits:

                            # Do not include the legend as the labels are necessary for mplcursors
                            legend_check = False

                            if line_list.size > 0:

                                wave_array, gaussian_array = profiles_computation(line_list, spec.frame, (1 + spec.redshift))
                                wave_array, cont_array = linear_continuum_computation(line_list, spec.frame, (1 + spec.redshift))

                                # Single component lines
                                line_g_list = self._gaussian_line_profiler(self._ax, line_list,
                                                                           wave_array, gaussian_array, cont_array,
                                                                           z_corr, spec.frame, spec.norm_flux, )

                                # Add the interactive pop-ups
                                self._mplcursor_parser(line_g_list, line_list, spec.frame, spec.norm_flux, spec.units_wave,
                                                       spec.units_flux)

                        # Plot the masked pixels
                        _masks_plot(self._ax, line_list, wave_plot, flux_plot, z_corr, spec.frame, idcs_mask)

                    # Switch y_axis to logarithmic scale if requested
                    if log_scale:
                        self._ax.set_yscale('log')

                    # Add or remove legend according to the plot type:
                    # TODO we should be able to separate labels from sample objects from line fits
                    if legend_check:
                        self._ax.legend()

                    # By default, plot on screen unless an output address is provided
                    save_close_fig_swicth(output_address, 'tight', self._fig, maximise=maximize)

            else:
                _logger.info(f'There are not observations with the input obj_idx "{obj_idcs}" to plot')

        else:
            _logger.info(f'The sample does not have a load function. The spectra cannot be plotted')

        return

    def properties(self, x_param, y_param, observation_list=None, x_param_err=None, y_param_err=None, labels_list=None,
                   groups_variable=None, output_address=None, in_fig=None, in_axis=None, plt_cfg={}, ax_cfg={},
                   log_scale=False):

        # Check selected variables are located in the panel columns
        for param in [x_param, y_param, x_param_err, y_param_err, groups_variable]:
            if param is not None:
                if param not in self._sample.frame:
                    raise LiMe_Error(f'Variable {param} is not found in the sample panel columns')

        # Panel slice
        slice_df = self._sample.frame.loc[observation_list]

        # Confirm selection has data
        if len(slice_df) > 0:

            # Get variables arrays
            data = {'x_param': slice_df[x_param].values, 'y_values': slice_df[y_param].values,
                    'x_err': slice_df[x_param_err].values if x_param_err in slice_df else None,
                    'y_err': slice_df[y_param_err].values if y_param_err in slice_df else None}

            # Split the data set if a group variable is provided
            idcs_dict = {}
            if groups_variable is None:
                idcs_dict['_'] = slice_df.index
            else:
                group_levels = np.sort(slice_df[groups_variable].unique())
                for i, level in enumerate(group_levels):
                    idcs_dict[level] = slice_df[groups_variable] == level

            # Ready the arrays:
            data_dict = {}
            for level, idcs in idcs_dict.items():
                data_dict[level] = {'x_param': slice_df.loc[idcs, x_param].values,
                                    'y_values': slice_df.loc[idcs, y_param].values,
                                    'x_err': slice_df.loc[idcs, x_param_err].values if x_param_err in slice_df else None,
                                    'y_err': slice_df.loc[idcs, y_param_err].values if y_param_err in slice_df else None}

            # Default figure format


            fig_format = {**{}, **{'figure.figsize': (10, 6)}}
            ax_format = {'xlabel': _LOG_COLUMNS_LATEX.get(x_param, x_param), 'ylabel': _LOG_COLUMNS_LATEX.get(y_param, y_param)}

            # User format overwrites default params
            plt_cfg.update(fig_format)
            ax_cfg.update(ax_format)

            with rc_context(plt_cfg):

                # Generate the figure object and figures
                self._fig, self._ax = self._plot_container(in_fig, in_axis, ax_cfg)

                # Plot data
                for group_label, group_dict in data_dict.items():
                    self._ax.errorbar(group_dict['x_param'], group_dict['y_values'], label=group_label,
                                      xerr=group_dict['x_err'], yerr=group_dict['y_err'], fmt='o')

                # Interactive display
                if mplcursors_check:

                    # If none provided use the object names
                    if labels_list is None:
                        labels_list = slice_df.index.values

                    mplcursors.cursor(self._ax).connect("add", lambda sel: sel.annotation.set_text(labels_list[sel.index]))

                if log_scale:
                    self._ax.set_yscale('log')

                # Display legend
                h, l = self._ax.get_legend_handles_labels()
                if len(h) > 0:
                    self._ax.legend()

                # By default, plot on screen unless an output address is provided
                save_close_fig_swicth(output_address, 'tight', self._fig)

        else:
            _logger.info(f'There is no data on the sample panel for the input observation selection')

        return


