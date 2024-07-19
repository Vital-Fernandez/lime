import logging
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt, gridspec, patches, rc_context, cm, colors

from .model import c_KMpS, profiles_computation, linear_continuum_computation
from .tools import latex_science_float, PARAMETER_LATEX_DICT
from .io import check_file_dataframe, _PARENT_BANDS, load_spatial_mask, LiMe_Error, _LOG_COLUMNS_LATEX
from .transitions import check_line_in_log, Line, label_decomposition, format_line_mask_option
from . import _setup_cfg

_logger = logging.getLogger('LiMe')


try:
    import mplcursors
    mplcursors_check = True
except ImportError:
    mplcursors_check = False

if mplcursors_check:
    from mplcursors._mplcursors import _default_annotation_kwargs as popupProps
    popupProps['bbox']['alpha'] = 0.9


def spectrum_figure_labels(units_wave, units_flux, norm_flux):

    # Wavelength axis units
    x_label = units_wave.to_string('latex')
    x_label = f'Wavelength ({x_label})'

    # Flux axis units
    norm_flux = units_flux.scale if norm_flux is None else norm_flux
    norm_label = r'\right)$' if norm_flux == 1 else r' \,\cdot\,{}\right)$'.format(latex_science_float(1 / norm_flux))

    y_label = f"Flux {units_flux.to_string('latex')}"
    y_label = y_label.replace('$\mathrm{', '$\left(')
    y_label = y_label.replace('}$', norm_label)

    return x_label, y_label


class Themer:

    def __init__(self, conf, style='default', library='matplotlib'):

        # TODO add tests to match the configuration

        # Attributes
        self.conf = None
        self.style = None
        self.base_conf = None
        self.colors = None
        self.library = None

        # Assign default
        self.conf = conf.copy()
        self.library = library
        self.set_style(style)

        return

    def fig_defaults(self, user_fig=None, fig_type=None):

        # Get plot configuration
        if fig_type is None:
            fig_conf = self.base_conf.copy()
        else:
            fig_conf = {** self.base_conf, **self.conf[self.library][fig_type]}

        # Get user configuration
        fig_conf = fig_conf if user_fig is None else {**fig_conf, **user_fig}

        return fig_conf

    def ax_defaults(self, user_ax, units_wave, units_flux, norm_flux, fig_type='default', **kwargs):

        # Default wavelength and flux
        if fig_type == 'default':

            # Spectrum labels x-wavelegth, y-flux # TODO without units
            x_label, y_label = spectrum_figure_labels(units_wave, units_flux, norm_flux)
            ax_cfg = {'xlabel': x_label, 'ylabel': y_label}

            # Update with the user configuration
            ax_cfg = ax_cfg if user_ax is None else {**ax_cfg, **user_ax}

        # Spatial cubes
        elif fig_type == 'cube':

            ax_cfg = {} if user_ax is None else user_ax.copy()

            # Define the title
            if ax_cfg.get('title') is None:

                title = r'{} band'.format(kwargs['line_bg'].latex_label[0])

                line_fg = kwargs.get('line_fg')
                if line_fg is not None:
                    title = f'{title} with {line_fg.latex_label[0]} contours'

                if len(kwargs['masks_dict']) > 0:
                    title += f'\n and spatial masks at foreground'

                ax_cfg['title'] = title

            # Define x axis
            if ax_cfg.get('xlabel') is None:
                ax_cfg['xlabel'] = 'x' if kwargs['wcs'] is None else 'RA'

            # Define y axis
            if ax_cfg.get('ylabel') is None:
                ax_cfg['ylabel'] = 'y' if kwargs['wcs'] is None else 'DEC'

            # Update with the user configuration
            ax_cfg = ax_cfg if user_ax is None else {**ax_cfg, **user_ax}

        elif fig_type == 'velocity':

            x_label = 'Velocity (Km/s)'

            # Flux axis units
            norm_flux = units_flux.scale if norm_flux is None else norm_flux
            norm_label = r'\right)$' if norm_flux == 1 else r' \,\cdot\,{}\right)$'.format(latex_science_float(1/norm_flux))

            y_label = f"Flux {units_flux.to_string('latex')}"
            y_label = y_label.replace('$\mathrm{', '$\left(')
            y_label = y_label.replace('}$', norm_label)

            ax_cfg = {'xlabel': x_label, 'ylabel': y_label}

            # Update with the user configuration
            ax_cfg = ax_cfg if user_ax is None else {**ax_cfg, **user_ax}

        # No labels
        else:
            ax_cfg = {}

            # Update with the user configuration
            ax_cfg = ax_cfg if user_ax is None else {**ax_cfg, **user_ax}

        return ax_cfg

    def set_style(self, style=None, fig_cfg=None, colors_conf=None):

        # Set the new style
        if style is not None:
            self.style = np.atleast_1d(style)
        else:
            self.style = np.atleast_1d('default')

        # Generate the default
        self.base_conf = self.conf[self.library]['default'].copy()
        for style in self.style:
            self.base_conf = {**self.base_conf, **self.conf[self.library][style]}

        # Add the new configuration
        if fig_cfg is not None:
            self.base_conf = {**self.base_conf, **fig_cfg}

        # Set the colors
        for i_style in self.style:
            if i_style in self.conf['colors']:
                self.colors = self.conf['colors'][style].copy()

        # Add the new colors
        if colors_conf is not None:
            self.colors = {**self.colors, **colors_conf}

        if self.colors is None:
            _logger.warning(f'The input style {self.style} does not have a LiMe color database')

        return


# LiMe figure labels and color formatter
theme = Themer(_setup_cfg)


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
        output_fig = None

        if file_path is None:

            # Tight layout
            if bbox_inches is not None:
                plt.tight_layout()

            # Window positioning and size
            maximize_center_fig(maximise)

            # Display
            plt.show()

        else:
            plt.savefig(file_path, bbox_inches=bbox_inches)

            # Close the figure in the case of printing
            if fig_obj is not None:
                plt.close(fig_obj)

    # Return the figure for output plotting
    else:
        output_fig = fig_obj

    return output_fig


def frame_mask_switch(wave_obs, flux_obs, redshift, rest_frame):

    # Doppler factor for rest _frame plots
    z_corr = (1 + redshift) if rest_frame else 1

    # Remove mask from plots and recover bad indexes
    if np.ma.isMaskedArray(wave_obs):
        idcs_mask = wave_obs.mask
        wave_plot, flux_plot = wave_obs.data, flux_obs.data

    else:
        idcs_mask = np.zeros(wave_obs.size).astype(bool)
        wave_plot, flux_plot = wave_obs, flux_obs

    return wave_plot, flux_plot, z_corr, idcs_mask


def _mplcursor_parser(line_g_list, list_comps, log, norm_flux, units_wave, units_flux):

    if mplcursors_check:
        for i, line_g in enumerate(line_g_list):
            line_label = list_comps[i]
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


def _auto_flux_scale(axis, y, y_scale):

    # If non-provided auto-decide
    if y_scale == 'auto':

        # Limits for the axes, ignore the divide by zero warning
        with np.errstate(divide='ignore', invalid='ignore'):
            neg_check = np.any(y < 0)
            y_max, y_min = np.nanmax(y), np.nanmin(y)
            # cont, std = np.nanmedian(y), np.nanstd(y)
            # high_limit = y_max + 2 * std
            # low_limit = y_min if (y_min - std < 0) and (y_min > 0) else y_min - 2 * std
            # axis.set_ylim(ymin=low_limit, ymax=high_limit)
            ratio = np.abs(y_max/y_min)
            if (ratio > 25) or (ratio < 0.06):
                if neg_check:
                    y_scale = 'symlog'
                else:
                    y_scale = 'log'
            else:
                y_scale = 'linear'

            # Add note on the scale:

    axis.set_yscale(y_scale)

    if y_scale != 'linear':
        axis.text(0.12, 0.8, f'${y_scale}$', fontsize=theme.colors['textsize_notes'], ha='center', va='center',
                  transform=axis.transAxes, alpha=0.5, color=theme.colors['fg'])

    return


def check_line_for_bandplot(in_label, user_band, spec, log_ref=_PARENT_BANDS):

    # If no line provided, use the one from the last fitting
    label = None
    if in_label is None:
        if spec.frame.index.size > 0:
            label = spec.frame.iloc[-1].name
            band = spec.frame.loc[label, 'w1':'w6'].values
        else:
            band = None
            _logger.info(f'The lines log is empty')

    # The user provides a line and a band
    elif (in_label is not None) and (user_band is not None):
        band = user_band

    # The user only provides a label
    else:

        # The line has not been measured before (just plot the region) # TODO I do not like this one
        if isinstance(in_label, str):
            if in_label.endswith('_b'):
                in_label = in_label[:-2]

        # Case of a float entry
        if in_label not in spec.frame.index:

            # First we try the user spec log
            if spec.frame.index.size > 0:
                test_label = check_line_in_log(in_label, spec.frame)
                if test_label != in_label:
                    label = test_label
                    band = spec.frame.loc[label, 'w1':'w6'].values

            # We use the reference log
            if label is None:
                label = check_line_in_log(in_label, log_ref)
                if label in log_ref.index:
                    band = log_ref.loc[label, 'w1':'w6'].values
                else:
                    band = None

        # The line has been plotted before
        else:
            label = in_label
            band = spec.frame.loc[in_label, 'w1':'w6'].values

    return label, band


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
        line = Line(line, band)

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

            mask_param = mask_header['PARAM']
            param_idx = mask_header['PARAMIDX']
            param_val = mask_header['PARAMVAL']
            n_spaxels = mask_header['NUMSPAXE']

            # Inverse the mask array for the plot
            inv_mask_array = np.ma.masked_array(mask_data, ~mask_data)

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

            cm_i = colors.ListedColormap([cmap_contours(idx_mask)])

            legend_list[idx_mask] = patches.Patch(color=cmap_contours(idx_mask), label=legend_i)

            ax.imshow(inv_mask_array, cmap=cm_i, vmin=0, vmax=1, alpha=mask_alpha)

    return legend_list


def spec_plot(ax, wave, flux, redshift, norm_flux, label='', rest_frame=False, log=None, include_fits=True,
              units_wave='A', units_flux='Flam', log_scale=False):

    # Reference frame for the plot
    wave_plot, flux_plot, z_corr, idcs_mask = frame_mask_switch(wave, flux, redshift, rest_frame)

    # Plot the spectrum
    ax.step(wave_plot / z_corr, flux_plot * z_corr, label=label, where='mid', color=theme.colors['fg'],
            linewidth=theme.colors['spectrum_width'])

    # List of lines in the log
    line_list = []
    if log is not None:
        if log.index.size > 0 and include_fits:
            line_list = log.index.values

            # Loop through the lines and plot them
            line_g_list = [None] * line_list.size
            for i, line_label in enumerate(line_list):

                line_i = Line.from_log(line_label, log)
                line_g_list[i] = _profile_plt(ax, line_i, z_corr, log, redshift, norm_flux)

            # Add the interactive pop-ups
            _mplcursor_parser(line_g_list, line_list, log, norm_flux, units_wave, units_flux)

    # Plot the masked pixels
    _masks_plot(ax, line_list, wave_plot, flux_plot, z_corr, log, idcs_mask)

    return


def _profile_plot(axis, x, y, label, idx_line=0, n_comps=1, observations_list='yes'):

    # Color and thickness
    if observations_list == 'no':

        # If only one component or combined
        if n_comps == 1:
            width_i, style, color = theme.colors['single_width'], '-', theme.colors['profile']

        # Component
        else:
            cmap = plt.get_cmap(theme.colors['comps_map'])
            width_i, style, color = theme.colors['comp_width'], ':', cmap(idx_line/n_comps)

    # Case where the line has an error
    else:
        width_i, style, color = theme.colors['err_width'], '-', theme.colors['error']

    # Plot the profile
    line_g = axis.plot(x, y, label=label, linewidth=width_i, linestyle=style, color=color)

    return line_g


def color_selector(label, observations, idx_line, n_comps):

    # Color and thickness
    if observations == 'no':

        # If only one component or combined
        if n_comps == 1:
            width_i, style, color = theme.colors['single_width'], '-', theme.colors['profile']

        # Component
        else:
            cmap = plt.get_cmap(theme.colors['comps_map'])
            width_i, style, color = theme.colors['comp_width'], ':', cmap(idx_line/n_comps)

    # Case where the line has an error
    else:
        width_i, style, color = theme.colors['err_width'], '-', 'red'

    # Make dictionary with the params
    cont_format = dict(label=label, color=color, linestyle=style, linewidth=width_i)

    return cont_format


def _profile_plt(axis, line, z_cor, log, redshift, norm_flux):

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
        cont_format = dict(label=None, color=theme.colors['cont'], linestyle='--', linewidth=0.5)
        axis.plot(wave_array/z_cor, cont_array[:, 0] * z_cor / norm_flux, **cont_format)

    # Plot combined gaussian profile if blended
    if (idx_line == 0) and (n_comps > 1):
        comb_array = (flux_array.sum(axis=1) + cont_i) * z_cor / norm_flux
        line_format = color_selector(None, line.observations, 0, 1)
        axis.plot(wave_i / z_cor, comb_array, **line_format)

    # Gaussian component plot
    single_array = (flux_i + cont_i) * z_cor / norm_flux
    line_format = color_selector(line.label, line.observations, idx_line, n_comps)
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
            else: # TODO remove if profile_comps not "no"
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

    def _line_matching_plot(self, axis, match_log, x, y, z_corr, redshift, units_wave):

        # Plot the detected line peaks
        if 'signal_peak' in match_log.columns:
            idcs_linePeaks = match_log['signal_peak'].values.astype(int)
            axis.scatter(x[idcs_linePeaks]/z_corr, y[idcs_linePeaks]*z_corr, label='Peaks',
                         facecolors='none', edgecolors=theme.colors['peak'])

        # Get the line labels and the bands labels for the lines
        wave_array, latex = label_decomposition(match_log.index.values, params_list=('wavelength', 'latex_label'))

        w3 = match_log.w3.values * (1 + redshift)
        w4 = match_log.w4.values * (1 + redshift)
        idcsLineBand = np.searchsorted(x, np.array([w3, w4]))

        # Loop through the detections and plot the names
        for i in np.arange(latex.size):
            label = 'Matched line' if i == 0 else '_'
            max_region = np.max(y[idcsLineBand[0, i]:idcsLineBand[1, i]])
            axis.axvspan(w3[i]/z_corr, w4[i]/z_corr, label=label, alpha=0.30, color=theme.colors['match_line'])
            axis.text(wave_array[i] * (1 + redshift) / z_corr, max_region * 0.9 * z_corr, latex[i], rotation=270)

        return

    def _bands_plot(self, axis, x, y, z_corr, idcs_mask, label):

        cont_dict = {'facecolor': theme.colors['cont_band'], 'step': 'mid', 'alpha': 0.25}
        line_dict = {'facecolor': theme.colors['line_band'], 'step': 'mid', 'alpha': 0.25}

        # Plot
        if len(y[idcs_mask[0]:idcs_mask[5]]) > 1:
            low_lim = np.min(y[idcs_mask[0]:idcs_mask[5]])
            low_lim = 0 if np.isnan(low_lim) else low_lim
            axis.fill_between(x[idcs_mask[0]:idcs_mask[1]]/z_corr, low_lim*z_corr, y[idcs_mask[0]:idcs_mask[1]]*z_corr, **cont_dict)
            axis.fill_between(x[idcs_mask[2]:idcs_mask[3]]/z_corr, low_lim*z_corr, y[idcs_mask[2]:idcs_mask[3]]*z_corr, **line_dict)
            axis.fill_between(x[idcs_mask[4]:idcs_mask[5]]/z_corr, low_lim*z_corr, y[idcs_mask[4]:idcs_mask[5]]*z_corr, **cont_dict)
        else:
            _logger.warning(f'The {label} band plot interval contains less than 1 pixel')

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

    def _plot_continuum_fit(self, continuum_fit, idcs_cont, low_lim, high_lim, threshold_factor, plot_title=''):

        norm_flux = self._spec.norm_flux
        wave = self._spec.wave
        flux = self._spec.flux
        units_wave = self._spec.units_wave
        units_flux = self._spec.units_flux
        redshift = self._spec.redshift

        # Assign the default axis labels
        PLOT_CONF = theme.fig_defaults(None)
        AXES_CONF = theme.ax_defaults(None, units_wave, units_flux, norm_flux)

        wave_plot, flux_plot, z_corr, idcs_mask = frame_mask_switch(wave, flux, redshift, False)

        with rc_context(PLOT_CONF):

            fig, ax = plt.subplots()

            # Object spectrum
            ax.step(wave_plot, flux_plot, label='Object spectrum', color=theme.colors['fg'], where='mid')

            # Band limits
            label = r'$16^{{th}}/{} - 84^{{th}}\cdot{}$ flux percentiles band'.format(threshold_factor, threshold_factor)
            ax.axhspan(low_lim, high_lim, alpha=0.2, label=label, color=theme.colors['line_band'])
            ax.axhline(np.median(flux_plot[idcs_cont]), label='Median flux', linestyle=':', color='black')

            # Masked and rectected pixels
            ax.scatter(wave_plot[~idcs_cont], flux_plot[~idcs_cont], label='Rejected pixels', color=theme.colors['peak'], facecolor='none')
            ax.scatter(wave_plot[idcs_mask], flux_plot[idcs_mask], marker='x', label='Masked pixels', color=theme.colors['mask_marker'])

            # Output continuum
            ax.plot(wave_plot, continuum_fit, label='Continuum')

            ax.update(AXES_CONF)
            ax.legend()
            plt.tight_layout()
            plt.show()

        return

    def _plot_peak_detection(self, peak_idcs, detect_limit, continuum, match_bands):


        norm_flux = self._spec.norm_flux
        wave = self._spec.wave
        flux = self._spec.flux
        units_wave = self._spec.units_wave
        units_flux = self._spec.units_flux
        redshift = self._spec.redshift

        PLOT_CONF = theme.fig_defaults(None)
        AXES_CONF = theme.ax_defaults(None, units_wave, units_flux, norm_flux)

        wave_plot, flux_plot, z_corr, idcs_mask = frame_mask_switch(wave, flux, redshift, 'observed')

        continuum = continuum if continuum is not None else np.zeros(flux.size)

        idcs_detect = match_bands['signal_peak'].to_numpy(dtype=int)

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


class SpectrumFigures(Plotter):

    def __init__(self, spectrum):

        # Instantiate the dependencies
        Plotter.__init__(self)

        # Lime spectrum object with the scientific data
        self._spec = spectrum

        # Container for the matplotlib figures
        self._fig, self._ax = None, None

        return

    def spectrum(self, output_address=None, label=None, line_bands=None, rest_frame=False, log_scale=False,
                 include_fits=True, include_cont=False, in_fig=None, fig_cfg={}, ax_cfg={}, maximize=False,
                 detection_band=None):

        """

        This function plots the spectrum flux versus wavelength.

        The user can include the line bands on the plot if added via the ``line_bands`` attribute.

        The user can provide a label for the spectrum legend via the ``label`` argument.

        If the user provides an ``output_address`` the plot will be stored into an image file instead of being displayed
        into a window.

        If the user has installed the library `mplcursors <https://mplcursors.readthedocs.io/en/stable/>`_, a left-click
        on a fitted profile will pop-up properties of the fitting, right-click to delete the annotation. This requires
        ``include_fits=True``.

        By default, this function creates a matplotlib figure and axes set to plot the data. However, the user can
        provide their own ``in_fig`` to plot the data. This will return the data-plotted figure object.

        The default axes and plot titles can be modified via the ``ax_cfg``. These dictionary keys are "xlabel", "ylabel"
        and "title". It is not necessary to include all the keys in this argument.

        :param output_address: File location to store the plot.
        :type output_address: str, optional

        :param label: Label for the spectrum plot legend. The default label is 'Observed spectrum'.
        :type label: str, optional

        :param line_bands: Bands Dataframe (or path to dataframe).
        :type line_bands: pd.Dataframe, str, path, optional

        :param rest_frame: Set to True for a display in rest frame. The default value is False
        :type rest_frame: bool, optional

        :param log_scale: Set to True for a display with a logarithmic scale flux. The default value is False
        :type log_scale: bool, optional

        :param include_fits: Set to True to display fitted profiles. The default value is False.
        :type include_fits:  bool, optional

        :param include_cont: Set to True to display fitted continuum. The default value is False.
        :type include_cont: bool, optional

        :param fig_cfg: `Matplotlib RcParams <https://matplotlib.org/stable/api/matplotlib_configuration_api.html#matplotlib.RcParams>`_
                        parameters for the figure format
        :type fig_cfg: dict, optional

        :param ax_cfg: Dictionary with the plot "xlabel", "ylabel" and "title" values.
        :type ax_cfg: dict, optional

        :param in_fig: Matplotlib figure object to plot the data.
        :type in_fig: matplotlib.figure

        :param maximize: Maximise plot window. The default value is False.
        :type maximize:  bool, optional

        """

        # Display check for the user figures
        display_check = True if in_fig is None else False

        # Set figure format with the user inputs overwriting the default conf
        legend_check = True if label is not None else False

        # Adjust the default theme
        PLT_CONF = theme.fig_defaults(fig_cfg)
        AXES_CONF = theme.ax_defaults(ax_cfg, self._spec.units_wave, self._spec.units_flux, self._spec.norm_flux)

        # Create and fill the figure
        with rc_context(PLT_CONF):

            # Generate the figure object and figures
            if in_fig is None:
                in_fig, in_ax = self._plot_container(in_fig, None, AXES_CONF)
            else:
                in_ax = in_fig.add_subplot()
                in_ax.set(**AXES_CONF)

            # Reference _frame for the plot
            wave_plot, flux_plot, z_corr, idcs_mask = frame_mask_switch(self._spec.wave, self._spec.flux,
                                                                        self._spec.redshift, rest_frame)

            # Plot the spectrum
            in_ax.step(wave_plot / z_corr, flux_plot * z_corr, label=label, where='mid', color=theme.colors['fg'],
                       linewidth=theme.colors['spectrum_width'])

            # Plot peaks and troughs if provided
            if line_bands is not None:
                line_bands = check_file_dataframe(line_bands, pd.DataFrame)
                self._line_matching_plot(in_ax, line_bands, wave_plot, flux_plot, z_corr, self._spec.redshift,
                                         self._spec.units_wave)

            # Plot the fittings
            if include_fits and self._spec.frame is not None:

                # List of lines in the log
                line_list = self._spec.frame.index.values

                # Do not include the legend as the labels are necessary for mplcursors # TODO improve mechanics
                legend_check = False

                if line_list.size > 0:

                    # Loop through the lines and plot them
                    profile_list = [None] * line_list.size
                    for i, line_label in enumerate(line_list):

                        line_i = Line.from_log(line_label, self._spec.frame)
                        profile_list[i] = _profile_plt(in_ax, line_i, z_corr, self._spec.frame, self._spec.redshift,
                                                       self._spec.norm_flux)

                    # Add the interactive pop-ups
                    _mplcursor_parser(profile_list, line_list, self._spec.frame, self._spec.norm_flux, self._spec.units_wave,
                                      self._spec.units_flux)

                # Plot the masked pixels
                _masks_plot(in_ax, line_list, wave_plot, flux_plot, z_corr, self._spec.frame, idcs_mask, theme.colors)

            # Plot the normalize continuum
            if include_cont and self._spec.cont is not None:
                in_ax.plot(wave_plot/z_corr, self._spec.cont*z_corr, label='Continuum', color=theme.colors['fade_fg'], linestyle='--')

                low_limit, high_limit = self._spec.cont-self._spec.cont_std, self._spec.cont + self._spec.cont_std
                in_ax.fill_between(wave_plot/z_corr, low_limit*z_corr, high_limit*z_corr, alpha=0.2,
                                   color=theme.colors['fade_fg'])

            # Include the detection bands
            if detection_band is not None:

                detec_obj = getattr(self._spec.infer, detection_band)

                if detec_obj.confidence is not None:

                    # Boundaries array for confidence intervals
                    bounds = np.array([0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

                    # Adjust color map to match lower detection limit to fg color
                    cmap = plt.get_cmap(theme.colors['mask_map'])
                    cmaplist = [cmap(i) for i in range(cmap.N)]
                    cmaplist[0] = theme.colors['fg']
                    cmap = colors.LinearSegmentedColormap.from_list('mcm', cmaplist, bounds.size-1)
                    norm = colors.BoundaryNorm(bounds * 100, cmap.N)

                    # Iterate through the confidence intervals and plot the step spectrum
                    for i in range(1, len(bounds)):
                        if i > 1:
                            idcs = detec_obj(bounds[i-1]*100, confidence_max=bounds[i]*100)
                            wave_nan, flux_nan = np.full(wave_plot.size, np.nan), np.full(flux_plot.size, np.nan)
                            wave_nan[idcs], flux_nan[idcs] = wave_plot[idcs] / z_corr, flux_plot[idcs] * z_corr

                            in_ax.step(wave_nan, flux_nan, label=label, where='mid', color=cmap(i-1))

                    # Color bar
                    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
                    sm.set_array([])
                    cbar = plt.colorbar(sm, ax=in_ax)
                    cbar.set_label('Detection confidence %', rotation=270, labelpad=35)


                else:
                    _logger.warning(f'The line detection bands confidence has not been calculated. They are not included'
                                    f' on plot.')


                # Switch y_axis to logarithmic scale if requested
            if log_scale:
                in_ax.set_yscale('log')

            # Add or remove legend according to the plot type:
            if legend_check:
                in_ax.legend()

            # By default, plot on screen unless an output address is provided
            in_fig = save_close_fig_swicth(output_address, 'tight', in_fig, maximize, display_check)

        return in_fig

    def grid(self, output_address=None, rest_frame=True, y_scale='auto', n_cols=6, n_rows=None, col_row_scale=(2, 1.5),
             include_fits=True, in_fig=None, fig_cfg=None, ax_cfg=None, maximize=False):

        """

        This function plots the lines from the object spectrum log as a grid.

        If the user has installed the library `mplcursors <https://mplcursors.readthedocs.io/en/stable/>`_, a left-click
        on a fitted profile will pop-up properties of the fitting, right-click to delete the annotation.

        If the user provides an ``output_address`` the plot will be stored into an image file instead of being displayed
        into a window.

        The default axes and plot titles can be modified via the ``ax_cfg``. These dictionary keys are "xlabel", "ylabel"
        and "title". It is not necessary to include all the keys in this argument.

        By default, this function creates a matplotlib figure and axes set to plot the data. However, the user can
        provide their own ``in_fig`` to plot the data. This will return the data-plotted figure object.

        :param output_address: Image file address for plot.
        :type output_address: str, pathlib.Path, optional

        :param rest_frame: Set to True to plot the spectrum to rest frame. Optional False.
        :type rest_frame: bool, optional

        :param y_scale: Matplotlib `scale keyword <https://lime-stable.readthedocs.io/en/latest/inputs/n_inputs3_line_bands.html>`_. The default value is "auto".
        :type y_scale: str, optional.

        :param n_cols: Number of columns in plot grid. The default value is 6.
        :type n_cols: int, optional.

        :param n_rows: Number of rows in plot grid.
        :type n_rows: int, optional.

        :param col_row_scale: Multiplicative factor for the grid plots width and height. The default value is (2, 1.5).
        :type col_row_scale: tuple, optional.

        :param include_fits: Set to True to display fitted profiles. The default value is False.
        :type include_fits:  bool, optional

        :param fig_cfg: Matplotlib `RcParams <https://matplotlib.org/stable/api/matplotlib_configuration_api.html#matplotlib.RcParams>`_
                        parameters for the figure format
        :type fig_cfg: dict, optional

        :param ax_cfg: Dictionary with the plot "xlabel", "ylabel" and "title" values.
        :type ax_cfg: dict, optional

        :param maximize: Maximise plot window. The default value is False.
        :type maximize:  bool, optional

        """

        # Display check for the user figures
        display_check = True if in_fig is None else False

        # Link to observation log
        log = self._spec.frame

        # Check that the log type and content
        if not isinstance(log, pd.DataFrame):
            _logger.critical('The bands log for the grid plot must be a pandas dataframe')

        # Proceed
        if len(log.index) > 0:

            lineList = log.index
            n_lines = lineList.size

            # Compute the number of rows configuration
            if n_lines > n_cols:
                if n_rows is None:
                    n_rows = int(np.ceil(n_lines / n_cols))
            else:
                n_cols, n_rows = n_lines, 1
            n_grid = n_cols * n_rows

            # Set the plot format where the user's overwrites the default
            size_conf = {'figure.figsize': (n_cols * col_row_scale[0], n_rows * col_row_scale[1])}
            size_conf = size_conf if fig_cfg is None else {**size_conf, **fig_cfg}

            PLT_CONF = theme.fig_defaults(size_conf, fig_type='grid')
            AXES_CONF = theme.ax_defaults(ax_cfg, self._spec.units_wave, self._spec.units_flux, self._spec.norm_flux,
                                          fig_type=None)

            # Launch the interative figure
            with rc_context(PLT_CONF):

                # Generate the figure if not provided
                if in_fig is None:
                    in_fig = plt.figure()

                # Generate the axis
                grid_spec = in_fig.add_gridspec(nrows=n_rows, ncols=n_cols)

                for i in np.arange(n_grid):

                    if i < n_lines:

                        in_ax = plt.subplot(grid_spec[i])

                        # Check components
                        line_i = Line.from_log(lineList[i], log, self._spec.norm_flux)

                        # Reference _frame for the plot
                        wave_plot, flux_plot, z_corr, idcs_mask = frame_mask_switch(self._spec.wave, self._spec.flux,
                                                                                    self._spec.redshift, rest_frame)

                        # Establish the limits for the line spectrum plot
                        idcs_m = np.searchsorted(wave_plot, line_i.mask * (1 + self._spec.redshift))
                        idx_blue = idcs_m[0] - 5 if idcs_m[0] > 5 else idcs_m[0]
                        idx_red = idcs_m[-1] + 5 if idcs_m[-1] < idcs_m[-1] + 5 else idcs_m[-1]

                        # Plot the spectrum
                        in_ax.step(wave_plot[idx_blue:idx_red] / z_corr, flux_plot[idx_blue:idx_red] * z_corr,
                                   where='mid', color=theme.colors['fg'], linewidth=theme.colors['spectrum_width'])

                        # Continuum bands
                        self._bands_plot(in_ax, wave_plot, flux_plot, z_corr, idcs_m, line_i)

                        # Plot the masked pixels
                        _masks_plot(in_ax, [line_i], wave_plot[idx_blue:idx_red], flux_plot[idx_blue:idx_red],
                                    z_corr, log, idcs_mask[idx_blue:idx_red], theme.colors)

                        # Plot the fitting results
                        if include_fits:

                            line_list, profiles_list = [line_i.label], line_i._p_shape
                            wave_array, gaussian_array = profiles_computation([line_i.label], log,
                                                                              (1 + self._spec.redshift), profiles_list)
                            wave_array, cont_array = linear_continuum_computation(line_list, self._spec.frame,
                                                                                  (1 + self._spec.redshift))

                            # Single component lines
                            line_g_list = _gaussian_line_profiler(in_ax, line_list, wave_array, gaussian_array,
                                                                  cont_array, z_corr, log, self._spec.norm_flux)

                            # Add the interactive pop-ups
                            _mplcursor_parser(line_g_list, line_list, log, self._spec.norm_flux,
                                              self._spec.units_wave, self._spec.units_flux)

                        # Formatting the figure
                        in_ax.yaxis.set_major_locator(plt.NullLocator())
                        in_ax.xaxis.set_major_locator(plt.NullLocator())

                        in_ax.update({'title': line_i.latex_label})
                        in_ax.yaxis.set_ticklabels([])
                        in_ax.axes.yaxis.set_visible(False)

                        # Scale each
                        _auto_flux_scale(in_ax, flux_plot[idx_blue:idx_red] * z_corr, y_scale)

                # Show the image
                in_fig = save_close_fig_swicth(output_address, 'tight', in_fig, maximize, display_check)

        else:
            _logger.info('The bands log does not contain lines')

        return in_fig

    def bands(self, line=None, output_address=None, include_fits=True, rest_frame=False, y_scale='auto', fig_cfg=None,
              ax_cfg=None, in_fig=None, maximize=False):

        """

        This function plots a spectrum ``line``. If a ``line`` is not provided the function will select the last line
        from the measurements log.

        The user can also introduce a ``bands`` dataframe (or its file path) to query the input ``line``.

        If the user provides an ``output_address`` the plot will be stored into an image file instead of being displayed
        in a window.

        The ``y_scale`` argument sets the flux scale for the lines grid. The default "auto" value automatically switches
        between the `matplotlib scale keywords <https://lime-stable.readthedocs.io/en/latest/inputs/n_inputs3_line_bands.html>`_,
        otherwise the user can set a uniform scale for all.

        The default axes and plot titles can be modified via the ``ax_cfg``. These dictionary keys are "xlabel", "ylabel"
        and "title". It is not necessary to include all the keys in this argument.

        :param line: Line label to display.
        :type line: str, optional

        :param output_address: File location to store the plot.
        :type output_address: str, optional

        :param include_fits: Set to True to display fitted profiles. The default value is False.
        :type include_fits:  bool, optional

        :param rest_frame: Set to True for a display in rest frame. The default value is False
        :type rest_frame: bool, optional

        :param y_scale: `Matplotlib scale keyword <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.yscale.html>`_. The default value is "auto".
        :type y_scale: str, optional.

        :param in_fig: Matplotlib figure object to plot the data.
        :type in_fig: matplotlib.figure

        :param fig_cfg: Dictionary with the matplotlib `rcParams parameters <https://matplotlib.org/stable/tutorials/introductory/customizing.html#customizing-with-dynamic-rc-settings>`_ .
        :type fig_cfg: dict, optional

        :param ax_cfg: Dictionary with the plot "xlabel", "ylabel" and "title" values.
        :type ax_cfg: dict, optional

        :param maximize: Maximise plot window. The default value is False.
        :type maximize:  bool, optional

        :return:
        """

        # TODO check plot without fit
        # Unpack variables
        log, norm_flux, redshift = self._spec.frame, self._spec.norm_flux, self._spec.redshift
        units_wave, units_flux = self._spec.units_wave, self._spec.units_flux

        # Display check for the user figures
        display_check = True if in_fig is None else False

        # If not line is provided use the last one
        if line is None:
            if log.index.size > 0:
                line = log.index[-1]

        # Reconstruct the line for the analysis
        if line is not None:
            line = Line.from_log(line, log, norm_flux)

        # Proceed to plot
        if line is not None:

            # Guess whether we need both lines
            include_fits = include_fits and (line.intg_flux is not None)

            # Adjust the default theme
            PLT_CONF = theme.fig_defaults(fig_cfg, fig_type='bands')
            AXES_CONF = theme.ax_defaults(ax_cfg, units_wave, units_flux, norm_flux)

            # Create and fill the figure
            with (rc_context(PLT_CONF)):

                # Generate the figure if not provided
                if in_fig is None:
                    in_fig = plt.figure()

                # Different figure desing if there isn't a fitting
                if include_fits:
                    grid_ax = in_fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3, 1])
                    spec_ax = plt.subplot(grid_ax[0])
                    resid_ax = plt.subplot(grid_ax[1], sharex=spec_ax)
                    in_ax = (spec_ax, resid_ax)
                else:
                    in_ax = [in_fig.add_subplot()]

                in_ax[0].set(**AXES_CONF)

                # Reference _frame for the plot
                wave_plot, flux_plot, z_corr, idcs_mask = frame_mask_switch(self._spec.wave, self._spec.flux,
                                                                            redshift, rest_frame)
                err_plot = self._spec.err_flux

                # Establish the limits for the line spectrum plot
                idcs_bands = line.index_bands(self._spec.wave, self._spec.redshift, just_band_edges=True)

                # Plot the spectrum
                label = '' if include_fits else line
                in_ax[0].step(wave_plot[idcs_bands[0]:idcs_bands[5]] / z_corr, flux_plot[idcs_bands[0]:idcs_bands[5]] * z_corr,
                              where='mid', color=theme.colors['fg'], label=label, linewidth=theme.colors['spectrum_width'])

                # Add the fitting results
                if include_fits:

                    # Check components
                    list_comps = line.group_label.split('+') if line.blended_check else [line.label]

                    wave_array, gaussian_array = profiles_computation(list_comps, log, (1 + redshift), line._p_shape)
                    wave_array, cont_array = linear_continuum_computation(list_comps, log, (1 + redshift))

                    # Continuum bands
                    self._bands_plot(in_ax[0], wave_plot, flux_plot, z_corr, idcs_bands, line)

                    # Gaussian profiles
                    idcs_lines = self._spec.frame.index.isin(list_comps)
                    line_g_list = _gaussian_line_profiler(in_ax[0], list_comps, wave_array, gaussian_array, cont_array,
                                                          z_corr, log.loc[idcs_lines], norm_flux)

                    # Add the interactive text
                    _mplcursor_parser(line_g_list, list_comps, log, norm_flux, units_wave, units_flux)

                    # Residual flux component
                    err_region = None if err_plot is None else err_plot[idcs_bands[0]:idcs_bands[5]]
                    self._residual_line_plotter(in_ax[1],
                                                wave_plot[idcs_bands[0]:idcs_bands[5]], flux_plot[idcs_bands[0]:idcs_bands[5]],
                                                err_region, list_comps, z_corr, idcs_mask[idcs_bands[0]:idcs_bands[5]],
                                                line._p_shape)

                    # Synchronizing the x-axis
                    in_ax[1].set_xlim(in_ax[0].get_xlim())
                    in_ax[1].set_xlabel(AXES_CONF['xlabel'])
                    in_ax[0].set_xlabel(None)

                # Plot the masked pixels
                _masks_plot(in_ax[0], [line], wave_plot[idcs_bands[0]:idcs_bands[5]], flux_plot[idcs_bands[0]:idcs_bands[5]], z_corr,
                            log, idcs_mask[idcs_bands[0]:idcs_bands[5]], theme.colors)

                # Display the legend
                in_ax[0].legend()

                # Set the scale
                _auto_flux_scale(in_ax[0], y=flux_plot[idcs_bands[0]:idcs_bands[5]] * z_corr, y_scale=y_scale)

                # By default, plot on screen unless an output address is provided
                in_fig = save_close_fig_swicth(output_address, 'tight', in_fig, maximize, display_check)

        else:
            in_fig = None
            _logger.info(f'The line "{line}" was not found in the spectrum log for plotting.')

        return in_fig

    def velocity_profile(self, line=None, band=None, y_scale='linear', fig_cfg=None, ax_cfg=None, in_fig=None,
                         output_address=None, maximize=False):

        # Establish the line and band to user for the analysis
        line, band = check_line_for_bandplot(line, band, self._spec, _PARENT_BANDS)

        # Display check for the user figures
        display_check = True if in_fig is None else False

        # Adjust the default theme
        PLT_CONF = theme.fig_defaults(fig_cfg)
        AXES_CONF = theme.ax_defaults(ax_cfg, self._spec.units_wave, self._spec.units_flux, self._spec.norm_flux,
                                      fig_type='velocity')

        # Recover the line data
        line = Line.from_log(line, self._spec.frame, self._spec.norm_flux)

        # Line spectrum
        wave_plot, flux_plot, z_corr, idcs_mask = frame_mask_switch(self._spec.wave, self._spec.flux,
                                                                    self._spec.redshift, False)

        # Establish the limits for the line spectrum plot
        mask = band * (1 + self._spec.redshift)
        idcsM = np.searchsorted(wave_plot, mask)

        # Velocity spectrum for the line region
        flux_plot = flux_plot[idcsM[0]:idcsM[5]]
        cont_plot = line.m_cont * wave_plot[idcsM[0]:idcsM[5]] + line.n_cont
        vel_plot = c_KMpS * (wave_plot[idcsM[0]:idcsM[5]] - line.peak_wave) / line.peak_wave

        # Line edges
        w_limits = np.array([line.w_i, line.w_f])
        v_i, v_f = c_KMpS * (w_limits - line.peak_wave) / line.peak_wave
        idx_i, idx_f = np.searchsorted(wave_plot[idcsM[0]:idcsM[5]], w_limits)

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
                          linewidth=theme.colors['spectrum_width'])

            # Velocity percentiles
            target_percen = ['v_1', 'v_5', 'v_10', 'v_50', 'v_90', 'v_95', 'v_99']
            for i_percentil, percentil in enumerate(target_percen):

                vel_per = line.__getattribute__(percentil)
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
            w80 = line.v_90-line.v_10
            label_arrow = r'$w_{{80}}={:0.1f}\,Km/s$'.format(w80)
            p1 = patches.FancyArrowPatch((line.v_10, 0.4),
                                         (line.v_90, 0.4),
                                         label=label_arrow,
                                         arrowstyle='<->',
                                         color='tab:blue',
                                         transform=trans,
                                         mutation_scale=20)
            self._ax.add_patch(p1)

            # Plot FWHM bands
            label_arrow = r'$FWZI={:0.1f}\,Km/s$'.format(line.FWZI)
            p2 = patches.FancyArrowPatch((v_i, 0),
                                         (v_f, 0),
                                         label=label_arrow,
                                         arrowstyle='<->',
                                         color='tab:red',
                                         transform=self._ax.transData,
                                         mutation_scale=20)
            self._ax.add_patch(p2)

            # Median velocity
            label_vmed = r'$v_{{med}}={:0.1f}\,Km/s$'.format(line.v_med)
            self._ax.axvline(x=line.v_med, color=theme.colors['fg'], label=label_vmed, linestyle='dashed', alpha=0.5)

            # Peak velocity
            label_vmed = r'$v_{{peak}}$'
            self._ax.axvline(x=0.0, color=theme.colors['fg'], label=label_vmed, alpha=0.5)

            # Set the scale
            _auto_flux_scale(self._ax, y=flux_plot, y_scale=y_scale)

            # Legend
            self._ax.legend()

            # By default, plot on screen unless an output address is provided
            save_close_fig_swicth(output_address, 'tight', in_fig, maximize, display_check)

        return

    def _residual_line_plotter(self, axis, x, y, err, list_comps, z_corr, spec_mask, profile_list):

        # Unpack properties
        log, norm_flux, redshift = self._spec.frame, self._spec.norm_flux, self._spec.redshift

        # Continuum level
        cont_level = log.loc[list_comps[0], 'cont']
        cont_std = log.loc[list_comps[0], 'cont_err']

        # Calculate the fluxes for the residual plot
        cont_i_resd = linear_continuum_computation(list_comps, log, z_corr=(1 + redshift), x_array=x)
        gaussian_i_resd = profiles_computation(list_comps, log, (1 + redshift), profile_list, x_array=x)
        total_resd = gaussian_i_resd.sum(axis=1) + cont_i_resd[:, 0]

        # Lower plot residual
        label_residual = r'$\frac{F_{obs} - F_{fit}}{F_{cont}}$'
        residual = ((y - total_resd / norm_flux) / (cont_level/norm_flux))
        axis.step(x/z_corr, residual*z_corr, where='mid', color=theme.colors['fg'], linewidth=theme.colors['spectrum_width'])

        # Shade Continuum flux standard deviation # TODO revisit this calculation
        label = r'$\sigma_{Continuum}/\overline{F_{cont}}$'
        y_limit = cont_std / cont_level
        axis.fill_between(x/z_corr, -y_limit, +y_limit, facecolor='yellow', alpha=0.5, label=label)

        # Shade the pixel error spectrum if available:
        if err is not None:
            label = r'$\sigma_{pixel}/\overline{F(cont)}$'
            err_norm = err/(cont_level/norm_flux)
            axis.fill_between(x/z_corr, -err_norm * z_corr, err_norm * z_corr, label=label, facecolor='salmon', alpha=0.3)

        # Residual y axis limit from std at line location
        idx_w3, idx_w4 = np.searchsorted(x, log.loc[list_comps[0], 'w3':'w4'] * (1 + redshift))
        resd_limit = np.std(residual[idx_w3:idx_w4]) * 5

        try:
            axis.set_ylim(-resd_limit, resd_limit)
        except ValueError:
            _logger.warning(f'Nan or inf entries in axis limit for {self.bands}')

        # Residual plot labeling
        axis.legend(loc='center right')
        axis.set_ylabel(label_residual)

        # Spectrum mask
        _masks_plot(axis, [list_comps[0]], x, y, z_corr, log, spec_mask, color_dict=theme.colors)

        return

    def _continuum_iteration(self, wave, flux, continuum_fit, smooth_flux, idcs_cont, low_lim, high_lim, threshold_factor,
                             user_ax):

        PLT_CONF = theme.fig_defaults(None)
        AXES_CONF = theme.ax_defaults(user_ax, self._spec.units_wave, self._spec.units_flux, self._spec.norm_flux)

        smooth_check = True if np.all(flux != smooth_flux) else False
        # wave_plot, flux_plot, z_corr, idcs_mask = frame_mask_switch_2(wave, flux, self._spec.redshift, False)

        with rc_context(PLT_CONF):

            fig, ax = plt.subplots()

            # Object spectrum
            label_spec = 'Smoothed spectrum' if smooth_check else 'Object spectrum'
            ax.step(wave, smooth_flux, label=label_spec, color=theme.colors['fg'], where='mid', linewidth=theme.colors['spectrum_width'])

            # Unsmooth
            if smooth_check:
                ax.step(wave, flux, label='Input spectrum', color=theme.colors['fade_fg'], where='mid', linestyle=':',
                        linewidth=theme.colors['spectrum_width'])

            # Band limits
            label = r'$16^{{th}}/{} - 84^{{th}}\cdot{}$ flux percentiles band'.format(threshold_factor, threshold_factor)
            # ax.axhspan(low_lim, high_lim, alpha=0.2, label=label, color=theme.colors['line_band'])
            ax.axhline(np.median(smooth_flux[idcs_cont]), label='Median flux', linestyle=':', color='black')

            ax.fill_between(wave, low_lim, high_lim, alpha=0.2, color=theme.colors['inspection_negative'])

            # Masked and rectected pixels
            ax.scatter(wave[~idcs_cont], smooth_flux[~idcs_cont], label='Rejected pixels', color=theme.colors['peak'],
                       facecolor='none')

            # Output continuum
            ax.plot(wave, continuum_fit, label='Continuum')

            ax.update(AXES_CONF)
            ax.legend()
            plt.tight_layout()
            plt.show()

        return


class CubeFigures(Plotter):

    def __init__(self, cube):

        # Instantiate the dependencies
        Plotter.__init__(self)

        # Lime spectrum object with the scientific data
        self._cube = cube

        # Container for the matplotlib figures
        self._fig, self._ax = None, None

        return

    def cube(self, line, bands=None, line_fg=None, output_address=None, min_pctl_bg=60, cont_pctls_fg=(90, 95, 99),
             bg_cmap='gray', fg_cmap='viridis', bg_norm=None, fg_norm=None, masks_file=None, masks_cmap='viridis_r',
             masks_alpha=0.2, wcs=None, fig_cfg=None, ax_cfg=None, in_fig=None, maximise=False):


        """

        This function plots the map of a flux band sum for a cube integral field unit observation.

        The ``line`` argument provides the label for the background image. Its bands are read from the ``bands`` argument
        dataframe. If none is provided, the default lines database will be used to query the bands. Similarly, if the user
        provides a foreground ``line_fg`` the plot will include intensity contours from its corresponding band.

        The user can provide a map baground and foreground contours `matplotlib color normalization
        <https://matplotlib.org/stable/gallery/images_contours_and_fields/colormap_normalizations.html>`_. Otherwise, a
        logarithmic normalization will be used.

        If the user does not provide a color normalizations at ``bg_norm`` and ``fg_norm``. A logarithmic normalization
        will be used. In this scenario ``min_pctl_bg`` establishes the minimum flux percentile flux for the
        background image. The number and separation of flux foreground contours is calculated from the sequence in the
        ``cont_pctls_fg``.

        If the user provides the address to a binary fits file to a mask file, this will be overploted on the map as
        shaded pixels.

        :param line: Line label for the spatial map background image.
        :type line: str

        :param bands: Bands dataframe (or file address to the dataframe).
        :type bands: pandas.Dataframe, str, path.Pathlib, optional

        :param line_fg: Line label for the spatial map background image contours
        :type line_fg: str, optional

        :param output_address: File location to store the plot.
        :type output_address: str, optional

        :param min_pctl_bg: Minimum band flux percentile for spatial map background image. The default value is 60.
        :type min_pctl_bg: float, optional

        :param cont_pctls_fg: Band flux percentiles for foreground ``line_fg`` contours. The default value is (90, 95, 99)
        :type cont_pctls_fg: tuple, optional

        :param bg_cmap: Background image flux `color map <https://matplotlib.org/stable/gallery/images_contours_and_fields/colormap_normalizations.html>`_.
                        The default value is "gray".
        :type bg_cmap: str, optional

        :param fg_cmap: Foreground image flux `color map <https://matplotlib.org/stable/gallery/images_contours_and_fields/colormap_normalizations.html>`_.
                        The default value is "viridis".
        :type fg_cmap: str, optional

        :param bg_norm: Background image `color normalization <https://matplotlib.org/stable/gallery/images_contours_and_fields/colormap_normalizations.html>`_.
                        The default value is `SymLogNorm <https://matplotlib.org/stable/gallery/images_contours_and_fields/colormap_normalizations.html>`_.
        :type bg_norm: Normalization from matplotlib.colors, optional

        :param fg_norm: Foreground contours `color normalization <https://matplotlib.org/stable/gallery/images_contours_and_fields/colormap_normalizations.html>`_.
                        The default value is `LogNorm <https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.LogNorm.html>`_.
        :type fg_norm: Normalization from matplotlib.colors, optional

        :param masks_file: File address for binary spatial masks
        :type masks_file: str, optional

        :param masks_cmap: Binary masks `color map <https://matplotlib.org/stable/gallery/images_contours_and_fields/colormap_normalizations.html>`_.
        :type masks_cmap: str, optional

        :param masks_alpha: Transparency alpha value. The default value is 0.2 (0 to 1 scale).
        :type masks_alpha: float, optional

        :param wcs: Observation `world coordinate system <https://docs.astropy.org/en/stable/wcs/index.html>`_.
        :type wcs: astropy WCS, optional

        :param fig_cfg: `Matplotlib RcParams <https://matplotlib.org/stable/api/matplotlib_configuration_api.html#matplotlib.RcParams>`_
                        parameters for the figure format
        :type fig_cfg: dict, optional

        :param ax_cfg: Dictionary with the plot "xlabel", "ylabel" and "title" values.
        :type ax_cfg: dict, optional

        :param in_fig: Matplotlib figure object to plot the data.
        :type in_fig: matplotlib.figure

        :param maximize: Maximise plot window. The default value is False.
        :type maximize:  bool, optional

        """

        # Prepare the background image data
        line_bg, bg_image, bg_levels, bg_norm = determine_cube_images(self._cube, line, bands,
                                                                      min_pctl_bg, bg_norm, contours_check=False)

        # Prepare the foreground image data
        line_fg, fg_image, fg_levels, fg_norm = determine_cube_images(self._cube, line_fg, bands,
                                                                      cont_pctls_fg, fg_norm, contours_check=True)

        # Mesh for the countours
        if line_fg is not None:
            y, x = np.arange(0, fg_image.shape[0]), np.arange(0, fg_image.shape[1])
            fg_mesh = np.meshgrid(x, y)
        else:
            fg_mesh = None

        # Load the masks
        masks_dict = load_spatial_mask(masks_file)

        # Check that the images have the same size
        check_image_size(bg_image, fg_image, masks_dict)

        # Use the input wcs or use the parent one
        wcs = self._cube.wcs if wcs is None else wcs

        # False for embed figures
        display_check = True if in_fig is None else False

        # Set the plot format where the user's overwrites the default
        size_conf = {'figure.figsize': (4 if masks_file is None else 5.5, 4)}
        size_conf = size_conf if fig_cfg is None else {**size_conf, **fig_cfg}

        PLT_CONF = theme.fig_defaults(size_conf, fig_type='cube')
        AXES_CONF = theme.ax_defaults(ax_cfg, self._cube.units_wave, self._cube.units_flux, self._cube.norm_flux,
                                      fig_type='cube', line_bg=line_bg, line_fg=line_fg, masks_dict=masks_dict, wcs=wcs)

        # Create and fill the figure
        with rc_context(PLT_CONF):

            if in_fig is None:
                in_fig = plt.figure()

            if wcs is None:
                in_ax = in_fig.add_subplot()
            else:
                slices = ('x', 'y', 1) if wcs.naxis == 3 else ('x', 'y')
                in_ax = in_fig.add_subplot(projection=wcs, slices=slices)

            # Assing the axis
            self._fig, self._ax = in_fig, in_ax
            self._ax.update(AXES_CONF)

            # Plot the image
            image_plot(self._ax, bg_image, fg_image, fg_levels, fg_mesh, bg_norm, fg_norm, bg_cmap, fg_cmap)

            # Plot the spatial masks
            if len(masks_dict) > 0:
                legend_hdl = spatial_mask_plot(self._ax, masks_dict, masks_cmap, masks_alpha, self._cube.units_flux)
                self._ax.legend(handles=legend_hdl, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

            # By default, plot on screen unless an output address is provided
            self._fig.canvas.draw()
            out_fig = save_close_fig_swicth(output_address, 'tight', self._fig, maximise,
                                            plot_check=display_check)

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
                legend_handle='levels', in_fig=None, in_axis=None, fig_cfg=None, ax_cfg=None, maximize=False):

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
                        self._ax.step(wave_plot / z_corr, flux_plot * z_corr, label=legend_label, where='mid',
                                      linewidth=theme.colors['spectrum_width'])

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


            fig_format = {**STANDARD_PLOT, **{'figure.figsize': (10, 6)}}
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


