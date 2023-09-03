import logging
import numpy as np

from matplotlib import pyplot as plt, gridspec, patches, rc_context, cm, colors
from astropy.wcs import WCS
from astropy.io import fits

import pandas as pd
from pathlib import Path

from .model import c_KMpS, gaussian_profiles_computation, linear_continuum_computation
from .tools import blended_label_from_log, ASTRO_UNITS_KEYS, UNITS_LATEX_DICT, latex_science_float, PARAMETER_LATEX_DICT
from .tools import define_masks, format_line_mask_option
from .io import check_file_dataframe, save_log, _PARENT_BANDS, load_spatial_mask, LiMe_Error, _LOG_COLUMNS_LATEX
from .transitions import check_line_in_log, Line, label_decomposition

_logger = logging.getLogger('LiMe')


try:
    import mplcursors
    mplcursors_check = True
except ImportError:
    mplcursors_check = False

if mplcursors_check:
    from mplcursors._mplcursors import _default_annotation_kwargs as popupProps
    popupProps['bbox']['alpha'] = 0.9

PLOT_SIZE_FONT = {'figure.figsize': (10, 5),
                  'axes.titlesize': 14,
                  'axes.labelsize': 14,
                  'legend.fontsize': 12,
                  'xtick.labelsize': 12,
                  'ytick.labelsize': 12}

# 'mathtext.fontset': 'cm'

colorDict = {'bg': 'white', 'fg': 'black',
             'cont_band': '#8c564b', 'line_band': '#b5bd61',
             'color_cycle': ['#279e68', '#d62728', '#aa40fc', '#8c564b',
                             '#e377c2', '#7f7f7f', '#b5bd61', '#17becf', '#1f77b4', '#ff7f0e'],
             'matched_line': '#b5bd61',
             'peak': '#aa40fc',
             'trough': '#7f7f7f',
             'profile': '#1f77b4',
             'cont': '#ff7f0e',
             'error': 'red',
             'mask_map': 'viridis',
             'comps_map': 'Dark2',
             'mask_marker': 'red'}

PLOT_COLORS = {'figure.facecolor': colorDict['bg'], 'axes.facecolor': colorDict['bg'],
               'axes.edgecolor': colorDict['fg'], 'axes.labelcolor': colorDict['fg'],
               'xtick.color': colorDict['fg'],  'ytick.color': colorDict['fg'],
               'text.color': colorDict['fg'], 'legend.edgecolor': 'inherit', 'legend.facecolor': 'inherit'}

colorDictDark = {'bg': np.array((43, 43, 43))/255.0, 'fg': np.array((179, 199, 216))/255.0,
                 'red': np.array((43, 43, 43))/255.0, 'yellow': np.array((191, 144, 0))/255.0}

PLOT_COLORS_DARK = {'figure.facecolor': colorDictDark['bg'], 'axes.facecolor': colorDictDark['bg'],
                    'axes.edgecolor': colorDictDark['fg'], 'axes.labelcolor': colorDictDark['fg'],
                    'xtick.color': colorDictDark['fg'],  'ytick.color': colorDictDark['fg'],
                    'text.color': colorDictDark['fg'], 'legend.edgecolor': 'inherit', 'legend.facecolor': 'inherit'}

PLOT_COLORS = {}

STANDARD_PLOT = {**PLOT_SIZE_FONT, **PLOT_COLORS}

STANDARD_AXES = {'xlabel': r'Wavelength $(\AA)$', 'ylabel': r'Flux $(erg\,cm^{-2} s^{-1} \AA^{-1})$'}


def mplcursors_legend(line, log, latex_label, norm_flux, units_wave, units_flux):

    legend_text = latex_label + '\n'

    units_line_flux = ASTRO_UNITS_KEYS[units_wave] * ASTRO_UNITS_KEYS[units_flux]
    line_flux_latex = f'{units_line_flux:latex}'
    normFlux_latex = f' $({latex_science_float(norm_flux)})$' if norm_flux != 1 else ''

    intg_flux = latex_science_float(log.loc[line, 'intg_flux']/norm_flux)
    intg_err = latex_science_float(log.loc[line, 'intg_flux_err']/norm_flux)
    legend_text += r'$F_{{intg}} = {}\pm{}\,$'.format(intg_flux, intg_err) + line_flux_latex + normFlux_latex + '\n'

    gauss_flux = latex_science_float(log.loc[line, 'gauss_flux']/norm_flux)
    gauss_err = latex_science_float(log.loc[line, 'gauss_flux_err']/norm_flux)
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


def frame_mask_switch_2(wave_obs, flux_obs, redshift, user_choice):

    # Doppler factor for rest _frame plots
    z_corr = (1 + redshift) if user_choice else 1

    # Remove mask from plots and recover bad indexes
    if np.ma.is_masked(wave_obs):
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

    axis.set_yscale(y_scale)

    return


def check_line_for_bandplot(in_label, user_band, spec, log_ref=_PARENT_BANDS):

    # If no line provided, use the one from the last fitting
    label = None
    if in_label is None:
        if spec.log.index.size > 0:
            label = spec.log.iloc[-1].name
            band = spec.log.loc[label, 'w1':'w6'].values
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
        if in_label not in spec.log.index:

            # First we try the user spec log
            if spec.log.index.size > 0:
                test_label = check_line_in_log(in_label, spec.log)
                if test_label != in_label:
                    label = test_label
                    band = spec.log.loc[label, 'w1':'w6'].values

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
            band = spec.log.loc[in_label, 'w1':'w6'].values

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
        idcsEmis, idcsCont = define_masks(cube.wave, line.mask * (1 + cube.redshift), line.pixel_mask)
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


def image_map_labels(input_labels, wcs, line_bg, line_fg, masks_dict):

    if input_labels is None:
        output_labels = {}
    else:
        output_labels = input_labels.copy()

    # Define the title
    if output_labels.get('title') is None:
        title = r'{} band'.format(line_bg.latex_label[0])
        if line_fg is not None:
            title = f'{title} with {line_fg.latex_label[0]} contours'
        if len(masks_dict) > 0:
            title += f'\n and spatial masks'
        output_labels['title'] = title

    # Define x axis
    if output_labels.get('xlabel') is None:
        output_labels['xlabel'] = 'x' if wcs is None else 'RA'

    # Define y axis
    if output_labels.get('ylabel') is None:
        output_labels['ylabel'] = 'y' if wcs is None else 'DEC'

    return output_labels


def image_plot(ax, image_bg, image_fg, fg_levels, fg_mesh, bg_scale, fg_scale, bg_color, fg_color, cursor_cords=None):

    # Background image plot
    im = ax.imshow(image_bg, cmap=bg_color, norm=bg_scale)

    # Foreground contours
    if image_fg is not None:
        contours = ax.contour(fg_mesh[0], fg_mesh[1], image_fg, cmap=fg_color, levels=fg_levels, norm=fg_scale)
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

    cmap_contours = cm.get_cmap(mask_color, len(masks_dict))

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
                units_text = r'{:latex}'.format(ASTRO_UNITS_KEYS[units_flux])
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
              units_wave='A', units_flux='Flam', log_scale=False, color_dict=colorDict):


    # Reference frame for the plot
    wave_plot, flux_plot, z_corr, idcs_mask = frame_mask_switch_2(wave, flux, redshift, rest_frame)

    # Plot the spectrum
    ax.step(wave_plot / z_corr, flux_plot * z_corr, label=label, where='mid', color=color_dict['fg'])

    # List of lines in the log
    line_list = []
    if log is not None:
        if log.index.size > 0 and include_fits:
            line_list = log.index.values

            # Fitted continua and profiles
            wave_array, gaussian_array = gaussian_profiles_computation(line_list, log, (1 + redshift))
            wave_array, cont_array = linear_continuum_computation(line_list, log, (1 + redshift))


            # Single component lines
            line_g_list = _gaussian_line_profiler(ax, line_list, wave_array, gaussian_array, cont_array,
                                                  z_corr, log, norm_flux)

            # Add the interactive pop-ups
            _mplcursor_parser(line_g_list, line_list, log, norm_flux, units_wave, units_flux)

    # Plot the masked pixels
    _masks_plot(ax, line_list, wave_plot, flux_plot, z_corr, log, idcs_mask)

    return


def _profile_plot(axis, x, y, label, idx_line=0, n_comps=1, observations_list='yes', color_dict=colorDict):

    # Color and thickness
    if observations_list == 'no':

        # If only one component or combined
        if n_comps == 1:
            width_i, style, color = 1.5, '-', color_dict['profile']

        # Component
        else:
            cmap = cm.get_cmap(color_dict['comps_map'])

            """  /home/usuario/PycharmProjects/lime/src/lime/plots.py:725: MatplotlibDeprecationWarning: 
            The get_cmap function was deprecated in Matplotlib 3.7 and will be removed two minor releases later. 
            Use ``matplotlib.colormaps[name]`` or ``matplotlib.colormaps.get_cmap(obj)`` instead.
                cmap = cm.get_cmap(color_dict['comps_map'])
            """

            width_i, style, color = 2, ':', cmap(idx_line/n_comps)

    # Case where the line has an error
    else:
        width_i, style, color = 3, '-', 'red'

    # Plot the profile
    line_g = axis.plot(x, y, label=label, linewidth=width_i, linestyle=style, color=color)

    return line_g


def _gaussian_line_profiler(axis, line_list, wave_array, gaussian_array, cont_array, z_corr, log, norm_flux,
                            color_dict=colorDict):

    # Data for the plot
    idcs_lines = log.index.isin(line_list)
    observations = log.loc[idcs_lines].observations.values

    # Plot them
    line_g_list = [None] * len(line_list)
    for i, line in enumerate(line_list):

        # Check if blended or single/merged
        idcs_comp = None
        if (not line.endswith('_m')) and (log.loc[line, 'profile_label'] is not np.nan) and (len(line_list) > 1):
            profile_comps = log.loc[line, 'profile_label']
            if profile_comps is not None:
                profile_comps = profile_comps.split('+')
                idx_line = profile_comps.index(line)
                n_comps = len(profile_comps)
                if profile_comps.index(line) == 0:
                    idcs_comp = (log['profile_label'] == log.loc[line, 'profile_label']).values
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
            axis.plot(wave_i/z_corr, cont_i*z_corr/norm_flux, color=colorDict['cont'], label=None,
                      linestyle='--', linewidth=0.5)

        # Plot combined gaussian profile if blended
        if idcs_comp is not None:
            gauss_comb = gaussian_array[:, idcs_comp].sum(axis=1) + cont_i
            _profile_plot(axis, wave_i/z_corr, gauss_comb*z_corr/norm_flux, None,
                               idx_line=idx_line, n_comps=1, observations_list=observations[i], color_dict=color_dict)

        # Gaussian component plot
        line_g_list[i] = _profile_plot(axis, wave_i/z_corr, (gauss_i+cont_i)*z_corr/norm_flux, latex_label,
                                       idx_line=idx_line, n_comps=n_comps, observations_list=observations[i],
                                       color_dict=color_dict)

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


def parse_figure_format(input_conf, local_conf=None, default_conf=STANDARD_PLOT, theme=None):

    # Check whether there is an input configuration
    if input_conf is None:
        output_conf = {}
    else:
        output_conf = input_conf.copy()

    # Default configuration
    if local_conf is not None:
        output_conf = {**local_conf, **output_conf}

    # Final configuration
    output_conf = {**default_conf, **output_conf}

    return output_conf


def parse_labels_format(input_labels, units_wave, units_flux, norm_flux):

    # Check whether there are input labels
    if input_labels is None:
        output_labels = {}
    else:
        output_labels = input_labels.copy()

    # X axis label
    if output_labels.get('xlabel') is None:
        output_labels['xlabel'] = f'Wavelength $({UNITS_LATEX_DICT[units_wave]})$'

    # Y axis label
    if output_labels.get('ylabel') is None:
        norm_label = r' $\,/\,{}$'.format(latex_science_float(norm_flux)) if norm_flux != 1.0 else ''
        output_labels['ylabel'] = f'Flux $({UNITS_LATEX_DICT[units_flux]})$' + norm_label

    return output_labels


class Plotter:

    def __init__(self):

        self._color_dict = colorDict
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

    def _figure_format(self, fig_cfg, ax_cfg, norm_flux, units_wave=None, units_flux=None):

        # Adjust default theme
        PLOT_CONF, AXES_CONF = STANDARD_PLOT.copy(), STANDARD_AXES.copy()

        norm_label = r' $\,/\,{}$'.format(latex_science_float(norm_flux)) if norm_flux != 1.0 else ''

        if (units_wave is not None) and ('xlabel' not in ax_cfg):
            AXES_CONF['xlabel'] = f'Wavelength $({UNITS_LATEX_DICT[units_wave]})$'

        if (units_flux is not None) and ('ylabel' not in ax_cfg):
            AXES_CONF['ylabel'] = f'Flux $({UNITS_LATEX_DICT[units_flux]})$' + norm_label

        # User configuration overrites user
        PLT_CONF = {**PLOT_CONF, **fig_cfg}
        AXES_CONF = {**AXES_CONF, **ax_cfg}

        return PLT_CONF, AXES_CONF

    def _line_matching_plot(self, axis, match_log, x, y, z_corr, redshift, units_wave):

        # Plot peaks and troughs if provided
        color_peaks = (self._color_dict['peak'], self._color_dict['trough'])
        labels = ('Peaks', 'Troughs')
        line_types = ('emission', 'absorption')
        labels = ('Peaks', 'Troughs')

        # Plot the detected line peaks
        if 'signal_peak' in match_log.columns:
            idcs_linePeaks = match_log['signal_peak'].values.astype(int)
            axis.scatter(x[idcs_linePeaks]/z_corr, y[idcs_linePeaks]*z_corr, label='Peaks',
                         facecolors='none', edgecolors=self._color_dict['peak'])

        # Get the line labels and the bands labels for the lines
        # ion_array, wave_array, latex = label_decomposition(match_log.index.values, units_wave=units_wave)
        wave_array, latex = label_decomposition(match_log.index.values, params_list=('wavelength', 'latex_label'))

        w3 = match_log.w3.values * (1 + redshift)
        w4 = match_log.w4.values * (1 + redshift)
        idcsLineBand = np.searchsorted(x, np.array([w3, w4]))

        # Loop through the detections and plot the names
        for i in np.arange(latex.size):
            label = 'Matched line' if i == 0 else '_'
            max_region = np.max(y[idcsLineBand[0, i]:idcsLineBand[1, i]])
            axis.axvspan(w3[i]/z_corr, w4[i]/z_corr, label=label, alpha=0.30, color=self._color_dict['matched_line'])
            axis.text(wave_array[i] * (1 + redshift) / z_corr, max_region * 0.9 * z_corr, latex[i], rotation=270)

        return

    def _bands_plot(self, axis, x, y, z_corr, idcs_mask, label):

        cont_dict = {'facecolor': self._color_dict['cont_band'], 'step': 'mid', 'alpha': 0.25}
        line_dict = {'facecolor': self._color_dict['line_band'], 'step': 'mid', 'alpha': 0.25}

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
        axis.scatter(peak_wave, peak_flux, facecolors=self._color_dict['peak'])

        return

    def _cont_plot(self, axis, x, y, z_corr, norm_flux):

        # Plot the continuum,  Usine wavelength array and continuum form the first component
        # cont_wave = wave_array[:, 0]
        # cont_linear = cont_array[:, 0]
        axis.plot(x/z_corr, y*z_corr/norm_flux, color=self._color_dict['cont'], linestyle='--', linewidth=0.5)

        return

    def _plot_continuum_fit(self, continuum_fit, idcs_cont, low_lim, high_lim, threshold_factor, plot_title=''):

        PLOT_CONF = STANDARD_PLOT.copy()
        AXES_CONF = STANDARD_AXES.copy()

        norm_flux = self._spec.norm_flux
        wave = self._spec.wave
        flux = self._spec.flux
        units_wave = self._spec.units_wave
        units_flux = self._spec.units_flux
        redshift = self._spec.redshift

        norm_label = r' $\,/\,{}$'.format(latex_science_float(norm_flux)) if norm_flux != 1.0 else ''
        AXES_CONF['ylabel'] = f'Flux $({UNITS_LATEX_DICT[units_flux]})$' + norm_label
        AXES_CONF['xlabel'] = f'Wavelength $({UNITS_LATEX_DICT[units_wave]})$'
        AXES_CONF['title'] = plot_title

        wave_plot, flux_plot, z_corr, idcs_mask = frame_mask_switch_2(wave, flux, redshift, False)

        with rc_context(PLOT_CONF):

            fig, ax = plt.subplots()

            # Object spectrum
            ax.step(wave_plot, flux_plot, label='Object spectrum', color=self._color_dict['fg'], where='mid')

            # Band limits
            label = r'$16^{{th}}/{} - 84^{{th}}\cdot{}$ flux percentiles band'.format(threshold_factor, threshold_factor)
            ax.axhspan(low_lim, high_lim, alpha=0.2, label=label, color=self._color_dict['line_band'])
            ax.axhline(np.median(flux_plot[idcs_cont]), label='Median flux', linestyle=':', color='black')

            # Masked and rectected pixels
            ax.scatter(wave_plot[~idcs_cont], flux_plot[~idcs_cont], label='Rejected pixels', color=self._color_dict['peak'], facecolor='none')
            ax.scatter(wave_plot[idcs_mask], flux_plot[idcs_mask], marker='x', label='Masked pixels', color=self._color_dict['mask_marker'])

            # Output continuum
            ax.plot(wave_plot, continuum_fit, label='Continuum')

            ax.update(AXES_CONF)
            ax.legend()
            plt.tight_layout()
            plt.show()

        return

    def _plot_peak_detection(self, peak_idcs, detect_limit, continuum=None, plot_title='', ml_mask=None):

        PLOT_CONF = STANDARD_PLOT.copy()
        AXES_CONF = STANDARD_AXES.copy()

        norm_flux = self._spec.norm_flux
        wave = self._spec.wave
        flux = self._spec.flux
        units_wave = self._spec.units_wave
        units_flux = self._spec.units_flux
        redshift = self._spec.redshift

        norm_label = r' $\,/\,{}$'.format(latex_science_float(norm_flux)) if norm_flux != 1.0 else ''
        AXES_CONF['ylabel'] = f'Flux $({UNITS_LATEX_DICT[units_flux]})$' + norm_label
        AXES_CONF['xlabel'] = f'Wavelength $({UNITS_LATEX_DICT[units_wave]})$'
        AXES_CONF['title'] = plot_title

        wave_plot, flux_plot, z_corr, idcs_mask = frame_mask_switch_2(wave, flux, redshift, 'observed')

        continuum = continuum if continuum is not None else np.zeros(flux.size)

        with rc_context(PLOT_CONF):

            fig, ax = plt.subplots()
            ax.step(wave_plot, flux_plot, color=self._color_dict['fg'], label='Object spectrum', where='mid')

            if ml_mask is not None:
                if np.any(ml_mask):
                    ax.scatter(wave_plot[ml_mask], flux_plot[ml_mask], label='ML detection', color='palegreen')

            ax.scatter(wave_plot[peak_idcs], flux_plot[peak_idcs], marker='o', label='Peaks', color=self._color_dict['peak'], facecolors='none')
            ax.fill_between(wave_plot, continuum, detect_limit, facecolor=self._color_dict['line_band'], label='Noise_region', alpha=0.5)

            if continuum is not None:
                ax.plot(wave_plot, continuum, label='Continuum')

            ax.scatter(wave_plot[idcs_mask], flux_plot[idcs_mask], label='Masked pixels', marker='x',
                       color=self._color_dict['mask_marker'])

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
                 include_fits=False, include_cont=False, in_fig=None, fig_cfg={}, ax_cfg={}, maximize=False):

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
        fig_cfg.setdefault('figure.figsize', (8, 5))
        PLT_CONF, AXES_CONF = self._figure_format(fig_cfg, ax_cfg, norm_flux=self._spec.norm_flux,
                                                  units_wave=self._spec.units_wave, units_flux=self._spec.units_flux)

        # Create and fill the figure
        with rc_context(PLT_CONF):

            # Generate the figure object and figures
            if in_fig is None:
                in_fig, in_ax = self._plot_container(in_fig, None, AXES_CONF)
            else:
                in_ax = in_fig.add_subplot()
                in_ax.set(**AXES_CONF)

            # Reference _frame for the plot
            wave_plot, flux_plot, z_corr, idcs_mask = frame_mask_switch_2(self._spec.wave, self._spec.flux,
                                                                          self._spec.redshift, rest_frame)


            # Plot the spectrum
            in_ax.step(wave_plot / z_corr, flux_plot * z_corr, label=label, where='mid', color=self._color_dict['fg'])

            # Plot peaks and troughs if provided
            if line_bands is not None:
                line_bands = check_file_dataframe(line_bands, pd.DataFrame)
                self._line_matching_plot(in_ax, line_bands, wave_plot, flux_plot, z_corr, self._spec.redshift,
                                         self._spec.units_wave)

            # List of lines in the log
            line_list = self._spec.log.index.values

            # Plot the fittings
            if include_fits and self._spec.log is not None:

                # Do not include the legend as the labels are necessary for mplcursors
                legend_check = False

                if line_list.size > 0:

                    wave_array, gaussian_array = gaussian_profiles_computation(line_list, self._spec.log, (1 + self._spec.redshift))
                    wave_array, cont_array = linear_continuum_computation(line_list, self._spec.log, (1 + self._spec.redshift))

                    # Single component lines
                    line_g_list = _gaussian_line_profiler(in_ax, line_list,
                                                          wave_array, gaussian_array, cont_array,
                                                          z_corr, self._spec.log, self._spec.norm_flux,
                                                          color_dict=self._color_dict)

                    # Add the interactive pop-ups
                    _mplcursor_parser(line_g_list, line_list, self._spec.log, self._spec.norm_flux,
                                           self._spec.units_wave, self._spec.units_flux)

            # Plot the normalize continuum
            if include_cont and self._spec.cont is not None:
                in_ax.plot(wave_plot/z_corr, self._spec.cont, label='Fitted continuum', linestyle='--')

            # Plot the masked pixels
            _masks_plot(in_ax, line_list, wave_plot, flux_plot, z_corr, self._spec.log, idcs_mask, self._color_dict)

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
             include_fits=True, in_fig=None, fig_cfg={}, ax_cfg={}, maximize=False):

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
        log = self._spec.log

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
            default_fig_cfg = {'figure.figsize': (n_cols * col_row_scale[0], n_rows * col_row_scale[1]),
                               'axes.titlesize': 12}
            default_fig_cfg.update(fig_cfg)
            PLT_CONF, AXES_CONF = self._figure_format(default_fig_cfg, ax_cfg, norm_flux=self._spec.norm_flux,
                                                      units_wave=self._spec.units_wave, units_flux=self._spec.units_flux)
            AXES_CONF.pop('xlabel')

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
                        line_i = lineList[i]
                        blended_check, profile_label = blended_label_from_log(line_i, log)
                        list_comps = profile_label.split('+') if blended_check else [line_i]

                        # Reference _frame for the plot
                        wave_plot, flux_plot, z_corr, idcs_mask = frame_mask_switch_2(self._spec.wave, self._spec.flux,
                                                                                      self._spec.redshift, rest_frame)

                        # Establish the limits for the line spectrum plot
                        mask = log.loc[list_comps[0], 'w1':'w6'] * (1 + self._spec.redshift)
                        idcsM = np.searchsorted(wave_plot, mask)
                        idxL = idcsM[0] - 5 if idcsM[0] > 5 else idcsM[0]
                        idxH = idcsM[-1] + 5 if idcsM[-1] < idcsM[-1] + 5 else idcsM[-1]

                        # Plot the spectrum
                        in_ax.step(wave_plot[idxL:idxH] / z_corr, flux_plot[idxL:idxH] * z_corr, where='mid',
                                        color=self._color_dict['fg'])

                        # Continuum bands
                        self._bands_plot(in_ax, wave_plot, flux_plot, z_corr, idcsM, line_i)

                        # Plot the masked pixels
                        _masks_plot(in_ax, [line_i], wave_plot[idxL:idxH], flux_plot[idxL:idxH], z_corr, log,
                                         idcs_mask[idxL:idxH], self._color_dict)

                        # Plot the fitting results
                        if include_fits:

                            wave_array, gaussian_array = gaussian_profiles_computation([line_i], self._spec.log,
                                                                                       (1 + self._spec.redshift))
                            wave_array, cont_array = linear_continuum_computation([line_i], self._spec.log,
                                                                                  (1 + self._spec.redshift))

                            # Single component lines
                            line_g_list = _gaussian_line_profiler(in_ax, [line_i],
                                                                       wave_array, gaussian_array, cont_array,
                                                                       z_corr, self._spec.log, self._spec.norm_flux,
                                                                       color_dict=self._color_dict)

                            # Add the interactive pop-ups
                            _mplcursor_parser(line_g_list, [line_i], self._spec.log, self._spec.norm_flux,
                                              self._spec.units_wave, self._spec.units_flux)

                        # Formatting the figure
                        in_ax.yaxis.set_major_locator(plt.NullLocator())
                        in_ax.xaxis.set_major_locator(plt.NullLocator())

                        in_ax.update({'title': log.loc[line_i, 'latex_label']})
                        in_ax.yaxis.set_ticklabels([])
                        in_ax.axes.yaxis.set_visible(False)

                        # Scale each
                        _auto_flux_scale(in_ax, flux_plot[idxL:idxH] * z_corr, y_scale)

                # Show the image
                in_fig = save_close_fig_swicth(output_address, 'tight', in_fig, maximize, display_check)

        else:
            _logger.info('The bands log does not contain lines')

        return in_fig

    def bands(self, line=None, bands=None, output_address=None, include_fits=True, rest_frame=False, y_scale='auto',
              in_fig=None, fig_cfg={}, ax_cfg={}, maximize=False):

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

        :param bands: Bands array or dataframe (or its file) to display.
        :type bands: np.array, pandas.Dataframe, str, path.pathlib, optional

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
        log, norm_flux, redshift = self._spec.log, self._spec.norm_flux, self._spec.redshift
        units_wave, units_flux = self._spec.units_wave, self._spec.units_flux

        # Display check for the user figures
        display_check = True if in_fig is None else False

        # Establish the line and band to user for the analysis
        line, bands = check_line_for_bandplot(line, bands, self._spec, _PARENT_BANDS)

        # Proceed to plot
        if (line is not None) and (bands is not None):

            # Guess whether we need both lines
            include_fits = include_fits and (line in self._spec.log.index)

            # Adjust the default theme
            fig_cfg.setdefault('axes.labelsize', 14)
            PLT_CONF, AXES_CONF = self._figure_format(fig_cfg, ax_cfg, norm_flux, units_wave, units_flux)

            # Create and fill the figure
            with rc_context(PLT_CONF):

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
                wave_plot, flux_plot, z_corr, idcs_mask = frame_mask_switch_2(self._spec.wave, self._spec.flux,
                                                                              redshift, rest_frame)
                err_plot = self._spec.err_flux

                # Establish the limits for the line spectrum plot
                mask = bands * (1 + self._spec.redshift)
                idcsM = np.searchsorted(wave_plot, mask) # TODO remove this one
                idcsEmis, idcsCont = define_masks(self._spec.wave, bands * (1 + self._spec.redshift),
                                                  line_mask_entry=log.loc[line, 'pixel_mask'])
                idcs_line = idcsEmis + idcsCont

                # Plot the spectrum
                label = '' if include_fits else line
                in_ax[0].step(wave_plot[idcsM[0]:idcsM[5]] / z_corr, flux_plot[idcsM[0]:idcsM[5]] * z_corr,
                              where='mid', color=self._color_dict['fg'], label=label)

                # Add the fitting results
                if include_fits:

                    # Check components
                    blended_check, profile_label = blended_label_from_log(line, log)
                    list_comps = profile_label.split('+') if blended_check else [line]

                    wave_array, gaussian_array = gaussian_profiles_computation(list_comps, log, (1 + redshift))
                    wave_array, cont_array = linear_continuum_computation(list_comps, log, (1 + redshift))

                    # Continuum bands
                    self._bands_plot(in_ax[0], wave_plot, flux_plot, z_corr, idcsM, line)

                    # Gaussian profiles
                    idcs_lines = self._spec.log.index.isin(list_comps)
                    line_g_list = _gaussian_line_profiler(in_ax[0], list_comps, wave_array, gaussian_array, cont_array,
                                                               z_corr, log.loc[idcs_lines], norm_flux,
                                                               color_dict=self._color_dict)

                    # Add the interactive text
                    _mplcursor_parser(line_g_list, list_comps, log, norm_flux, units_wave, units_flux)


                    cont_linear_flux = (log.loc[list_comps[0], 'm_cont'] * wave_plot[idcsCont] + log.loc[list_comps[0], 'n_cont'])
                    linear_cont_std = np.std(cont_linear_flux - flux_plot[idcsCont]*norm_flux)

                    # Residual flux component
                    err_region = None if err_plot is None else err_plot[idcsM[0]:idcsM[5]]
                    self._residual_line_plotter(in_ax[1], wave_plot[idcsM[0]:idcsM[5]], flux_plot[idcsM[0]:idcsM[5]],
                                                err_region, list_comps, z_corr, idcs_mask[idcsM[0]:idcsM[5]], linear_cont_std)

                    # Synchronizing the x-axis
                    in_ax[1].set_xlim(in_ax[0].get_xlim())
                    in_ax[1].set_xlabel(AXES_CONF['xlabel'])
                    in_ax[0].set_xlabel(None)

                # Plot the masked pixels
                _masks_plot(in_ax[0], [line], wave_plot[idcsM[0]:idcsM[5]], flux_plot[idcsM[0]:idcsM[5]], z_corr,
                            log, idcs_mask[idcsM[0]:idcsM[5]], self._color_dict)

                # # Line location
                # if trans_line:
                #     in_ax[0].axvline(line.wavelength, linestyle='--', alpha=0.5, color=self._color_dict['fg'])

                # Display the legend
                in_ax[0].legend()

                # Set the scale
                _auto_flux_scale(in_ax[0], y=flux_plot[idcsM[0]:idcsM[5]] * z_corr, y_scale=y_scale)

                # By default, plot on screen unless an output address is provided
                in_fig = save_close_fig_swicth(output_address, 'tight', in_fig, maximize, display_check)

        else:
            in_fig = None
            _logger.info(f'The line "{line}" was not found in the spectrum log for plotting.')

        return in_fig

    def velocity_profile(self,  line=None, band=None, y_scale='linear', plt_cfg={}, ax_cfg={}, in_fig=None,
                         output_address=None, maximize=False):

        # Establish the line and band to user for the analysis
        line, band = check_line_for_bandplot(line, band, self._spec, _PARENT_BANDS)

        # Display check for the user figures
        display_check = True if in_fig is None else False

        # Adjust the default theme
        ax_cfg.setdefault('xlabel',  'Velocity (Km/s)')
        PLT_CONF, AXES_CONF = self._figure_format(plt_cfg, ax_cfg, self._spec.norm_flux, self._spec.units_wave,
                                                  self._spec.units_flux)

        # Recover the line data
        line = Line.from_log(line, self._spec.log, self._spec.norm_flux)

        # Line spectrum
        wave_plot, flux_plot, z_corr, idcs_mask = frame_mask_switch_2(self._spec.wave, self._spec.flux,
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
            self._ax.step(vel_plot, flux_plot, label=line.latex_label, where='mid', color=self._color_dict['fg'])

            # Velocity percentiles
            target_percen = ['v_1', 'v_5', 'v_10', 'v_50', 'v_90', 'v_95', 'v_99']
            for i_percentil, percentil in enumerate(target_percen):

                vel_per = line.__getattribute__(percentil)
                label_text = None if i_percentil > 0 else r'$v_{Pth}$'
                self._ax.axvline(x=vel_per, label=label_text, color=self._color_dict['fg'], linestyle='dotted', alpha=0.5)

                label_plot = r'$v_{{{}}}$'.format(percentil[2:])
                self._ax.text(vel_per, 0.80, label_plot, ha='center', va='center', rotation='vertical',
                              backgroundcolor=self._color_dict['bg'], transform=trans, alpha=0.5)

            # Velocity edges
            label_v_i, label_v_f = r'$v_{{0}}$', r'$v_{{100}}$'
            self._ax.axvline(x=v_i, alpha=0.5, color=self._color_dict['fg'], linestyle='dotted')
            self._ax.text(v_i, 0.80, label_v_i, ha='center', va='center', rotation='vertical',
                          backgroundcolor=self._color_dict['bg'], transform=trans, alpha=0.5)

            self._ax.axvline(x=v_f, alpha=0.5, color=self._color_dict['fg'], linestyle='dotted')
            self._ax.text(v_f, 0.80, label_v_f, ha='center', va='center', rotation='vertical',
                          backgroundcolor=self._color_dict['bg'], transform=trans, alpha=0.5)

            # Plot the line profile
            self._ax.plot(vel_plot, cont_plot, linestyle='--', color=self._color_dict['fg'])

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
            self._ax.axvline(x=line.v_med, color=self._color_dict['fg'], label=label_vmed, linestyle='dashed', alpha=0.5)

            # Peak velocity
            label_vmed = r'$v_{{peak}}$'
            self._ax.axvline(x=0.0, color=self._color_dict['fg'], label=label_vmed, alpha=0.5)

            # Set the scale
            _auto_flux_scale(self._ax, y=flux_plot, y_scale=y_scale)

            # Legend
            self._ax.legend()

            # By default, plot on screen unless an output address is provided
            save_close_fig_swicth(output_address, 'tight', in_fig, maximize, display_check)

        return

    def _residual_line_plotter(self, axis, x, y, err, list_comps, z_corr, spec_mask, cont_std):

        # Unpack properties
        log, norm_flux, redshift = self._spec.log, self._spec.norm_flux, self._spec.redshift

        # Continuum level
        cont_level = log.loc[list_comps[0], 'cont']
        # cont_std = log.loc[list_comps[0], 'std_cont']

        # Continuum std (we don't use line std_cont to compare both calculations)


        # Calculate the fluxes for the residual plot
        cont_i_resd = linear_continuum_computation(list_comps, log, z_corr=(1 + redshift), x_array=x)
        gaussian_i_resd = gaussian_profiles_computation(list_comps, log, z_corr=(1 + redshift), x_array=x)
        total_resd = gaussian_i_resd.sum(axis=1) + cont_i_resd[:, 0]

        # Lower plot residual
        label_residual = r'$\frac{F_{obs} - F_{fit}}{F_{cont}}$'
        residual = ((y - total_resd / norm_flux) / (cont_level/norm_flux))
        axis.step(x/z_corr, residual*z_corr, where='mid', color=self._color_dict['fg'])

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
        axis.legend(loc='upper left')
        axis.set_ylabel(label_residual, fontsize=22)

        # Spectrum mask
        _masks_plot(axis, [list_comps[0]], x, y, z_corr, log, spec_mask, color_dict=self._color_dict)

        return

    def _continuum_iteration(self, continuum_fit, idcs_cont, low_lim, high_lim, threshold_factor, plot_title=''):

        PLOT_CONF = STANDARD_PLOT.copy()
        AXES_CONF = STANDARD_AXES.copy()

        norm_label = r' $\,/\,{}$'.format(latex_science_float(self._spec.norm_flux)) if self._spec.norm_flux != 1.0 else ''
        AXES_CONF['ylabel'] = f'Flux $({UNITS_LATEX_DICT[self._spec.units_flux]})$' + norm_label
        AXES_CONF['xlabel'] = f'Wavelength $({UNITS_LATEX_DICT[self._spec.units_wave]})$'
        AXES_CONF['title'] = plot_title

        wave_plot, flux_plot, z_corr, idcs_mask = frame_mask_switch_2(self._spec.wave, self._spec.flux,
                                                                    self._spec.redshift, False)

        with rc_context(PLOT_CONF):

            fig, ax = plt.subplots()

            # Object spectrum
            ax.step(wave_plot, flux_plot, label='Object spectrum', color=self._color_dict['fg'], where='mid')

            # Band limits
            label = r'$16^{{th}}/{} - 84^{{th}}\cdot{}$ flux percentiles band'.format(threshold_factor, threshold_factor)
            ax.axhspan(low_lim, high_lim, alpha=0.2, label=label, color=self._color_dict['line_band'])
            ax.axhline(np.median(flux_plot[idcs_cont]), label='Median flux', linestyle=':', color='black')

            # Masked and rectected pixels
            ax.scatter(wave_plot[~idcs_cont], flux_plot[~idcs_cont], label='Rejected pixels', color=self._color_dict['peak'], facecolor='none')
            ax.scatter(wave_plot[idcs_mask], flux_plot[idcs_mask], marker='x', label='Masked pixels', color=self._color_dict['mask_marker'])

            # Output continuum
            ax.plot(wave_plot, continuum_fit, label='Continuum')

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

        self.param_conv = {'SN_line': r'$\frac{S}{N}_{line}$',
                           'SN_cont': r'$\frac{S}{N}_{cont}$',
                           self._cube.units_flux: None}

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

        # State the plot labelling
        AXES_CONF = image_map_labels(ax_cfg, wcs, line_bg, line_fg, masks_dict)

        # User figure format overwrite default format
        display_check = True if in_fig is None else False
        local_cfg = {'figure.figsize': (5 if masks_file is None else 10, 5), 'axes.titlesize': 12, 'legend.fontsize': 10}
        PLT_CONF = parse_figure_format(fig_cfg, local_cfg)

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

        return

    def spectra(self, obj_idcs=None, log_scale=False, output_address=None, rest_frame=False, include_fits=False,
                in_fig=None, in_axis=None, plt_cfg={}, ax_cfg={}):

        if self._sample.load_function is not None:

            legend_check = True
            plt_cfg.setdefault('figure.figsize', (10, 6))

            norm_flux = self._sample.load_params.get('norm_flux')
            units_wave = self._sample.load_params.get('units_wave')
            units_flux = self._sample.load_params.get('units_flux')
            PLT_CONF, AXES_CONF = self._figure_format(plt_cfg, ax_cfg, norm_flux=norm_flux,
                                                      units_wave=units_wave, units_flux=units_flux)

            # Get the spectra list to plot
            if obj_idcs is None:
                sub_sample = self._sample
            else:
                sub_sample = self._sample[obj_idcs]

            # Check for logs without lines
            if 'line' in sub_sample.index.names:
                obj_idcs = sub_sample.log.droplevel('line').index.unique()
            else:
                obj_idcs = sub_sample.index.unique()

            if len(obj_idcs) > 0:

                # Create and fill the figure
                with rc_context(PLT_CONF):

                    # Generate the figure object and figures
                    self._fig, self._ax = self._plot_container(in_fig, in_axis, AXES_CONF)

                    # Loop through the SMACS_v2.0 in the sample
                    for sample_idx in obj_idcs:

                        spec_label, spec_file = sample_idx[0], sample_idx[1]
                        legend_label = ", ".join(map(str, sample_idx))
                        # spec = self._sample.get_observation(spec_label, spec_file)
                        spec = self._sample.load_function(self._sample.log, sample_idx, **self._sample.load_params)

                        # Reference _frame for the plot
                        wave_plot, flux_plot, z_corr, idcs_mask = frame_mask_switch_2(spec.wave, spec.flux, spec.redshift,
                                                                                      rest_frame)

                        # Plot the spectrum
                        self._ax.step(wave_plot / z_corr, flux_plot * z_corr, label=legend_label, where='mid')

                        # List of lines in the log
                        line_list = spec.log.index.values

                        # Plot the fittings
                        if include_fits:

                            # Do not include the legend as the labels are necessary for mplcursors
                            legend_check = False

                            if line_list.size > 0:

                                wave_array, gaussian_array = gaussian_profiles_computation(line_list, spec.log, (1 + spec.redshift))
                                wave_array, cont_array = linear_continuum_computation(line_list, spec.log, (1 + spec.redshift))

                                # Single component lines
                                line_g_list = self._gaussian_line_profiler(self._ax, line_list,
                                                                           wave_array, gaussian_array, cont_array,
                                                                           z_corr, spec.log, spec.norm_flux,)

                                # Add the interactive pop-ups
                                self._mplcursor_parser(line_g_list, line_list, spec.log, spec.norm_flux, spec.units_wave,
                                                       spec.units_flux)

                        # Plot the masked pixels
                        _masks_plot(self._ax, line_list, wave_plot, flux_plot, z_corr, spec.log, idcs_mask)

                    # Switch y_axis to logarithmic scale if requested
                    if log_scale:
                        self._ax.set_yscale('log')

                    # Add or remove legend according to the plot type:
                    # TODO we should be able to separate labels from sample objects from line fits
                    if legend_check:
                        self._ax.legend()

                    # By default, plot on screen unless an output address is provided
                    save_close_fig_swicth(output_address, 'tight', self._fig)

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
                if param not in self._sample.log:
                    raise LiMe_Error(f'Variable {param} is not found in the sample panel columns')

        # Panel slice
        slice_df = self._sample.log.loc[observation_list]

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