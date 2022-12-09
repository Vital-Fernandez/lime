import logging
import numpy as np

from matplotlib import pyplot as plt, gridspec, patches, rc_context, cm, colors
from astropy.wcs import WCS
from astropy.io import fits

import pandas as pd
from pathlib import Path

from .model import c_KMpS, gaussian_profiles_computation, linear_continuum_computation
from .tools import label_decomposition, blended_label_from_log, ASTRO_UNITS_KEYS, UNITS_LATEX_DICT, latex_science_float, PARAMETER_LATEX_DICT
from .tools import define_masks, format_line_mask_option
from .io import load_log, save_log, _PARENT_BANDS, load_spatial_masks
from .transitions import check_line_in_log, Line

_logger = logging.getLogger('LiMe')


try:
    import mplcursors
    mplcursors_check = True
except ImportError:
    mplcursors_check = False

if mplcursors_check:
    from mplcursors._mplcursors import _default_annotation_kwargs as popupProps
    popupProps['bbox']['alpha'] = 0.9

PLOT_SIZE_FONT = {'figure.figsize': (10, 5), 'axes.titlesize': 18, 'axes.labelsize': 16, 'legend.fontsize': 12,
                 'xtick.labelsize': 16, 'ytick.labelsize': 16, 'font.family': 'Times New Roman', 'mathtext.fontset':'cm'}

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
    intg_err = latex_science_float(log.loc[line, 'intg_err']/norm_flux)
    legend_text += r'$F_{{intg}} = {}\pm{}\,$'.format(intg_flux, intg_err) + line_flux_latex + normFlux_latex + '\n'

    gauss_flux = latex_science_float(log.loc[line, 'gauss_flux']/norm_flux)
    gauss_err = latex_science_float(log.loc[line, 'gauss_err']/norm_flux)
    legend_text += r'$F_{{gauss}} = {}\pm{}\,$'.format(gauss_flux, gauss_err) + line_flux_latex + normFlux_latex + '\n'

    v_r = r'{:.1f}'.format(log.loc[line, 'v_r'])
    v_r_err = r'{:.1f}'.format(log.loc[line, 'v_r_err'])
    legend_text += r'$v_{{r}} = {}\pm{}\,\frac{{km}}{{s}}$'.format(v_r, v_r_err) + '\n'

    sigma_vel = r'{:.1f}'.format(log.loc[line, 'sigma_vel'])
    sigma_vel_err = r'{:.1f}'.format(log.loc[line, 'sigma_vel_err'])
    legend_text += r'$\sigma_{{g}} = {}\pm{}\,\frac{{km}}{{s}}$'.format(sigma_vel, sigma_vel_err)

    return legend_text


def spatial_mask_generator(mask_param, wavelength_array, flux_cube, contour_levels, signal_band, cont_band=None,
                           mask_ref="", output_address=None, min_level=None, show_plot=False, fits_header=None,
                           plt_cfg={}, ax_cfg={'xlabel': 'RA', 'ylabel': 'DEC'}):

    """
    This function computes a spatial mask for an input flux image given an array of limits for a certain intensity parameter.
    Currently, the only one implemented is the percentile intensity. If an output address is provided, the mask is saved as a fits file
    where each intensity level mask is stored in its corresponding page. The parameter calculation method, its intensity and mask
    index are saved in the corresponding HDU header as PARAM, PARAMIDX and PARAMVAL.

    :param image_flux: Matrix with the image flux to be spatially masked.
    :type image_flux: np.array()

    :param mask_param: Flux intensity model from which the masks are calculated. The options available are 'flux',
           'SN_line' and 'SN_cont'.
    :type mask_param: str, optional

    :param contour_levels: Vector in decreasing order with the parameter values for the mask_param chosen.
    :type contour_levels: np.array()

    :param mask_ref: String label for the mask. If none provided the masks will be named in cardinal order.
    :type mask_ref: str, optional

    :param output_address: Output address for the mask fits file.
    :type output_address: str, optional

    :param min_level: Minimum level for the masks calculation. If none is provided the minimum value from the contour_levels
                      vector will be used.
    :type min_level: float, optional

    :param show_plot: If true a plot will be displayed with the mask calculation. Additionally, if an output_address is
                      provided the plot will be saved in the parent folder as image taking into consideration the
                      mask_ref value.
    :type show_plot: bool, optional

    :param fits_header: Dictionary with key-values to be included in the output .fits file header.
    :type fits_header: dict, optional

    :return:
    """

    # TODO overwrite spatial mask file not update

    # Compute the image flux from the band signal_band, cont_band
    idcs_signal = np.searchsorted(wavelength_array, signal_band)

    # Check the contour vector is in decreasing order
    assert np.all(np.diff(contour_levels) > 0), '- ERROR contour_levels are not in increasing order for spatial mask'
    contour_levels_r = np.flip(contour_levels)

    # Check the logic for the mask calculation
    assert mask_param in ['flux', 'SN_line', 'SN_cont'], f'\n- ERROR {mask_param} is not recognise for the spatial mask calculation'

    # Compute the band slice
    signal_slice = flux_cube[idcs_signal[0]:idcs_signal[1], :, :]

    # Compute the continuum band
    if cont_band is not None:
        idcs_cont = np.searchsorted(wavelength_array, cont_band)
        cont_slice = flux_cube[idcs_cont[0]:idcs_cont[1], :, :]

    # Compute the mask diagnostic
    if mask_param == 'flux':
        default_title = 'Spaxel flux percentile masks'
        param_image = signal_slice.sum(axis=0)

    # S/N cont
    elif mask_param == 'SN_cont':
        default_title = 'Spaxel continuum S/N percentile masks'
        param_image = np.nanmean(signal_slice, axis=0) / np.nanstd(signal_slice, axis=0)

    # S/N line
    else:
        default_title = 'Spaxel emission line S/N percentile masks'
        N_elem = idcs_cont[1] - idcs_cont[0]

        Amp_image = np.nanmax(signal_slice, axis=0) - np.nanmean(cont_slice, axis=0)
        std_image = np.nanstd(cont_slice, axis=0)

        param_image = (np.sqrt(2*N_elem*np.pi)/6) * (Amp_image/std_image)

    # Percentiles vector for the target parameter
    param_array = np.nanpercentile(param_image, contour_levels_r)

    # If minimum level not provided by user use lowest contour_level
    min_level = param_array[-1] if min_level is None else min_level

    # Containers for the mask parameters
    mask_dict = {}
    param_level = {}
    boundary_dict = {}

    # Loop throught the counter levels and compute the
    for i, n_levels in enumerate(param_array):

        # # Operation every element
        if i == 0:
            maParamImage = np.ma.masked_where((param_image >= param_array[i]) &
                                              (param_image >= min_level),
                                               param_image)

        else:
            maParamImage = np.ma.masked_where((param_image >= param_array[i]) &
                                              (param_image < param_array[i - 1]) &
                                              (param_image >= min_level),
                                               param_image)

        if np.sum(maParamImage.mask) > 0:
            mask_dict[f'mask_{i}'] = maParamImage.mask
            boundary_dict[f'mask_{i}'] = contour_levels_r[i]
            param_level[f'mask_{i}'] = param_array[i]

    # Output folder computed from the output address
    fits_folder = Path(output_address).parent if output_address is not None else None

    # Plot the combined masks
    if (fits_folder is not None) or show_plot:

        # Adjust default theme
        PLT_CONF = STANDARD_PLOT.copy()
        AXES_CONF = STANDARD_AXES.copy()
        AXES_CONF['title'] = default_title

        # User configuration overrites user
        PLT_CONF = {**PLT_CONF, **plt_cfg}
        AXES_CONF = {**AXES_CONF, **ax_cfg}

        with rc_context(PLT_CONF):

            if fits_header is None:
                fig, ax = plt.subplots(figsize=(12, 12))
            else:
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(projection=WCS(fits_header), slices=('x', 'y'))

            cmap = cm.get_cmap(colorDict['mask_map'], len(mask_dict))
            legend_list = [None] * len(mask_dict)

            for idx_region, region_items in enumerate(mask_dict.items()):

                region_label, region_mask = region_items

                # Inverse the mask array for the plot
                inv_mask_array = np.ma.masked_array(region_mask.data, ~region_mask)

                # Prepare the labels for each mask to add to imshow
                ext_name = f'{mask_ref}_{region_label}'
                percentile_ref = f'{mask_param}' + r'$_{{{}th}}$'.format(boundary_dict[region_label])
                param_percentile = f'${latex_science_float(param_array[idx_region], dec=3)}$'
                mask_voxels = np.sum(region_mask)

                legend_text = f'{ext_name}: {percentile_ref} = {param_percentile} ({mask_voxels} voxels)'
                legend_list[idx_region] = patches.Patch(color=cmap(idx_region), label=legend_text)

                cm_i = colors.ListedColormap(['black', cmap(idx_region)])
                ax.imshow(inv_mask_array, cmap=cm_i, vmin=0, vmax=1)

            ax.legend(handles=legend_list, loc=2)
            ax.set(**AXES_CONF)

            # Save the mask image
            if fits_folder is not None:

                if mask_ref is None:
                    output_image = fits_folder/f'mask_contours.png'
                else:
                    output_image = fits_folder/f'{mask_ref}_mask_contours.png'

                plt.savefig(output_image)

            # Display the mask if requested
            if show_plot:
                save_close_fig_swicth(None, 'tight', fig)


    # Save the mask to a fits file:
    if output_address is not None:

        fits_address = Path(output_address)

        for idx_region, region_items in enumerate(mask_dict.items()):
            region_label, region_mask = region_items

            # Metadata for the fits page
            header_dict = {'PARAM': mask_param,
                           'PARAMIDX': boundary_dict[region_label],
                           'PARAMVAL': param_level[region_label],
                           'NUMSPAXE': np.sum(region_mask)}
            fits_hdr = fits.Header(header_dict)

            if fits_header is not None:
                fits_hdr.update(fits_header)

            # Extension for the mask
            mask_ext = region_label if mask_ref is None else f'{mask_ref}_{region_label}'

            # Mask HDU
            mask_hdu = fits.ImageHDU(name=mask_ext, data=region_mask.astype(int), ver=1, header=fits_hdr)

            if fits_address.is_file():
                try:
                    fits.update(fits_address, data=mask_hdu.data, header=mask_hdu.header, extname=mask_ext, verify=True)
                except KeyError:
                    fits.append(fits_address, data=mask_hdu.data, header=mask_hdu.header, extname=mask_ext)
            else:
                hdul = fits.HDUList([fits.PrimaryHDU(), mask_hdu])
                hdul.writeto(fits_address, overwrite=True, output_verify='fix')

    return


def save_close_fig_swicth(file_path=None, bbox_inches=None, fig_obj=None, maximize=False, plot_check=True):

    # By default, plot on screen unless an output address is provided
    if plot_check:

        if file_path is None:

            # Tight layout
            if bbox_inches is not None:
                plt.tight_layout()

            # Window positioning and size
            maximize_center_fig(maximize)

            # Display
            plt.show()

        else:
            plt.savefig(file_path, bbox_inches=bbox_inches)

            # Close the figure in the case of printing
            if fig_obj is not None:
                plt.close(fig_obj)

    return


def save_redshift_table(object, redshift, file_address):

    if redshift != 0:
        filePath = Path(file_address)

        if filePath.parent.is_dir():

            # Create a new dataframe and save it
            if not filePath.is_file():
                df = pd.DataFrame(data=redshift, index=[object], columns=['redshift'])

            # Replace or append to dataframe
            else:
                df = pd.read_csv(filePath, delim_whitespace=True, header=0, index_col=0)
                df.loc[object, 'redshift'] = redshift

            # Save back
            with open(filePath, 'wb') as output_file:
                string_DF = df.to_string()
                output_file.write(string_DF.encode('UTF-8'))

        else:
            _logger.warning(f'Output redshift table folder does not exist at {file_address}')

    return


def save_or_clear_log(log, log_address, activeLines, log_parameters=['w1', 'w2', 'w3', 'w4', 'w5', 'w6']):

    if np.sum(activeLines) == 0:
        if log_address.is_file():
            log_address.unlink()
    else:
        if log_address is not None:
            save_log(log.loc[activeLines], log_address, parameters=log_parameters)
        else:
            _logger.warning(r"Not output redshift lob provided, the selection won't be stored")

    return


def frame_mask_switch(wave_obs, flux_obs, redshift, user_choice):

    assert user_choice in ['observed', 'rest'], f'- ERROR: _frame of reference {user_choice} not recognized. ' \
                                                f'Please use "observed" or "rest".'

    # Doppler factor for rest _frame plots
    z_corr = (1 + redshift) if user_choice == 'rest' else 1

    # Remove mask from plots and recover bad indexes
    if np.ma.is_masked(wave_obs):
        idcs_mask = wave_obs.mask
        wave_plot, flux_plot = wave_obs.data, flux_obs.data

    else:
        idcs_mask = np.zeros(flux_obs.size).astype(bool)
        wave_plot, flux_plot = wave_obs, flux_obs

    return wave_plot, flux_plot, z_corr, idcs_mask


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

    # mng = plt.get_current_fig_manager()
    # mng.resize(*mng.window.maxsize())

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
            _logger.info(f'The log is empty')

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
                band = log_ref.loc[label, 'w1':'w6'].values

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


def determine_cube_images(cube, line, band, bands_frame, percentiles, color_scale, contours_check=False):

    # Generate line object
    if line is not None:

        # Determine the line of reference
        line = Line(line, band, ref_log=bands_frame)

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


def image_map_labels(title, wcs, line_bg, line_fg, masks_dict):

    # Define the title
    if title is None:
        title = r'{} band'.format(line_bg.latex[0])
        if line_fg is not None:
            # title = f'{} with {} contours'.format(title, line_fg.latex[0])
            title = f'{title} with {line_fg.latex[0]} contours'
        if len(masks_dict) > 0:
            title += f'\n and spatial masks'

    # Generate the figure
    if wcs is not None:
        x_label, y_label = 'RA', 'DEC'
    else:
        x_label, y_label = 'x', 'y'

    return title, x_label, y_label


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
            legend_i += f'${latex_science_float(param_val, dec=3)}$ ({n_spaxels} voxels)'

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
        if (not line.endswith('_m')) and (log.loc[line, 'profile_label'] != 'no') and (len(line_list) > 1):
            profile_comps = log.loc[line, 'profile_label'].split('-')
            idx_line = profile_comps.index(line)
            n_comps = len(profile_comps)
            if profile_comps.index(line) == 0:
                idcs_comp = (log['profile_label'] == log.loc[line, 'profile_label']).values
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
            axis.scatter(x_mask, y_mask, marker='x', label='Masked pixels', color=color_dict['mask_marker'])

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


class LiMePlots:

    def __init__(self, ):

        self._color_dict = colorDict
        self._legends_dict = {}

        return

    def plot_spectrum(self, comp_array=None, peaks_table=None, match_log=None, noise_region=None,
                      log_scale=False, plt_cfg={}, ax_cfg={}, spec_label='Object spectrum', output_address=None,
                      include_fits=False, log=None, frame='observed'):

        """

        This function plots the spectrum defined by the `Spectrum class <https://lime-stable.readthedocs.io/en/latest/documentation/api.html#lime.treatment.Spectrum>`_

        The user can include an additional flux array (for example the uncertainty spectrum) to be plotted.

        Additionally, the user can include the outputs from the `.match_line_mask <https://lime-stable.readthedocs.io/en/latest/documentation/api.html#lime.treatment.Spectrum.match_line_mask>`_
        function to plot the emission peaks and the matched lines. Moreover, if the parameter ``include_fits=True`` the plot
        will include the gaussian profiles stored in the lines ``.log``.

        The user can specify the plot _frame of reference via the ``_frame='obs'`` or ``_frame='rest'`` parameter. Moreover,
        the user can provide dictionaries for the matplotlib `figure <https://matplotlib.org/stable/api/matplotlib_configuration_api.html#matplotlib.rcParams>`_
        and `axis <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set.html#matplotlib.axes.Axes.set>`_ styles.

        Finally, if the user provides an ``output_address``, the spectrum will be saved as an image instead of being displayed.

        :param comp_array: Additional flux array to be plotted alongside the spectrum flux.
        :type comp_array: numpy.array, optional

        :param peaks_table: Table with the emission and absorptions detected by the `.match_line_mask function <https://lime-stable.readthedocs.io/en/latest/documentation/api.html#lime.treatment.Spectrum.match_line_mask>`_
        :type peaks_table: astropy.Table, optional

        :param match_log: Lines log with the emission/absorptions which have matched the peaks/trough by the .match_line_mask.
        :type match_log: pandas.Dataframe, optional

        :param noise_region: 2 value array with the wavelength limits. This region will be shaded in the output plot.
        :type noise_region: np.array, optional

        :param log_scale: Set to True for a vertical (flux) axis logarithmic scale. The default value is False
        :type log_scale: bool, optional

        :param plt_cfg: Dictionary with the configuration for the matplotlib `rcParams routine <https://matplotlib.org/stable/tutorials/introductory/customizing.html#customizing-with-dynamic-rc-settings>`_ .
        :type plt_cfg: bool, optional

        :param ax_cfg: Dictionary with the configuration for the matplotlib axes style.
        :type ax_cfg: bool, optional

        :param spec_label: Label for the spectrum plot legend, The default value is 'Observed spectrum'
        :type spec_label: str, optional

        :param output_address: File location to store the plot as an image. If provided, the plot won't be displayed on
                               the screen.
        :type output_address: str, optional

        :param include_fits: Check to include the gaussian profile fittings in the plot. The default value is False.
        :type include_fits: Check to include the gaussian profile fittings in the plot.

        :param frame: Frame of reference for the spectrum plot: "observed" or "rest". The default value is observed.
        :param _frame: str, optional

        """

        # Adjust default theme
        PLOT_CONF = STANDARD_PLOT.copy()
        AXES_CONF = STANDARD_AXES.copy()

        PLOT_CONF['figure.figsize'] = (10, 6)
        norm_label = r' $\,/\,{}$'.format(latex_science_float(self.norm_flux)) if self.norm_flux != 1.0 else ''
        AXES_CONF['ylabel'] = f'Flux $({UNITS_LATEX_DICT[self.units_flux]})$' + norm_label
        AXES_CONF['xlabel'] = f'Wavelength $({UNITS_LATEX_DICT[self.units_wave]})$'

        # User configuration overrites user
        PLT_CONF = {**PLOT_CONF, **plt_cfg}
        AXES_CONF = {**AXES_CONF, **ax_cfg}

        # Use the memory log if none is provided
        log = self.log if log is None else log

        legend_check = True
        with rc_context(PLT_CONF):

            fig, ax = plt.subplots()
            ax.set(**AXES_CONF)

            # Reference _frame for the plot
            wave_plot, flux_plot, z_corr, idcs_mask = frame_mask_switch(self.wave, self.flux, self.redshift, frame)

            # Plot the spectrum
            ax.step(wave_plot/z_corr, flux_plot*z_corr, label=spec_label, where='mid', color=self._color_dict['fg'])

            # Plot the continuum if available
            if comp_array is not None:
                assert len(comp_array) == len(wave_plot), '- ERROR: comp_array and wavelength array have mismatching length'
                ax.step(wave_plot/z_corr, comp_array, label='Sigma Continuum', linestyle=':', where='mid')

            # Plot peaks and troughs if provided
            if (peaks_table is not None) or (match_log is not None):

                color_peaks = (self._color_dict['peak'], self._color_dict['trough'])
                labels = ('Peaks', 'Troughs')
                line_types = ('emission', 'absorption')
                labels = ('Peaks', 'Troughs')

                if peaks_table is not None:
                    line_types = ('emission', 'absorption')
                    for i in range(2):
                        idcs_emission = peaks_table['line_type'] == line_types[i]
                        idcs_linePeaks = np.array(peaks_table[idcs_emission]['line_center_index'])
                        ax.scatter(wave_plot[idcs_linePeaks]/z_corr, flux_plot[idcs_linePeaks]*z_corr, label=labels[i],
                                   facecolors='none', edgecolors=color_peaks[i])

                else:
                    if 'signal_peak' in match_log:
                        idcs_linePeaks = match_log['signal_peak'].values.astype(int)

                        ax.scatter(wave_plot[idcs_linePeaks] / z_corr, flux_plot[idcs_linePeaks] * z_corr, label='Peaks',
                                   facecolors='none', edgecolors=self._color_dict['peak'])

            # Shade regions of matched lines if provided
            if match_log is not None:
                ion_array, wave_array, latex_array = label_decomposition(match_log.index.values, units_wave=self.units_wave)
                w3, w4 = match_log.w3.values * (1+self.redshift), match_log.w4.values * (1+self.redshift)
                idcsLineBand = np.searchsorted(wave_plot, np.array([w3, w4]))

                first_check = True
                for i in np.arange(latex_array.size):
                    label = 'Matched line' if first_check else '_'
                    max_region = np.max(flux_plot[idcsLineBand[0, i]:idcsLineBand[1, i]])
                    ax.axvspan(w3[i]/z_corr, w4[i]/z_corr, label=label, alpha=0.30, color=self._color_dict['matched_line'])
                    ax.text(wave_array[i] * (1+self.redshift)/z_corr, max_region * 0.9 * z_corr, latex_array[i], rotation=270)
                    first_check = False

            # Shade noise region if provided
            if noise_region is not None:
                ax.axvspan(noise_region[0], noise_region[1], alpha=0.15, color='tab:cyan', label='Noise region')

            # Plot the line fittings
            if include_fits:

                legend_check = False
                w3_array, w4_array = self.log.w3.values, self.log.w4.values

                # Compute the individual profiles
                wave_array, gaussian_array = gaussian_profiles_computation(log.index.values, log, (1 + self.redshift))
                wave_array, cont_array = linear_continuum_computation(log.index.values, log, (1 + self.redshift))

                # Separating blended from unblended lines
                idcs_nonBlended = (self.log.index.str.endswith('_m')) | (self.log.profile_label == 'no').values

                # Plot single lines
                line_list = self.log.loc[idcs_nonBlended].index
                for line in line_list:

                    i = self.log.index.get_loc(line)

                    # Determine the line region
                    idcs_plot = ((w3_array[i] - 5) * z_corr <= wave_plot) & (wave_plot <= (w4_array[i] + 5) * z_corr)

                    # Plot the gaussian profiles
                    wave_i = wave_array[:, i][..., None]
                    cont_i = cont_array[:, i][..., None]
                    gauss_i = gaussian_array[:, i][..., None]

                    self._gaussian_profiles_plotting([line], self.log,
                                                     wave_plot[idcs_plot], flux_plot[idcs_plot], z_corr,
                                                     axis=ax, cont_bands=None,
                                                     wave_array=wave_i, cont_array=cont_i,
                                                     gaussian_array=gauss_i)

                    # Plot masked pixels if possible
                    self._mask_pixels_plotting(line, wave_plot[idcs_plot], flux_plot[idcs_plot], z_corr, ax, self.log)

                # Plot combined lines
                profile_list = self.log.loc[~idcs_nonBlended, 'profile_label'].unique()
                for profile_group in profile_list:

                    idcs_group = (self.log.profile_label == profile_group)
                    i_group = np.where(idcs_group)[0]

                    # Determine the line region
                    idcs_plot = ((w3_array[i_group[0]] - 1) * z_corr <= wave_plot) & (wave_plot <= (w4_array[i_group[0]] + 1) * z_corr)

                    # Plot the gauss curve elements
                    wave_i = wave_array[:, i_group[0]:i_group[-1]+1]
                    cont_i = cont_array[:, i_group[0]:i_group[-1]+1]
                    gauss_i = gaussian_array[:, i_group[0]:i_group[-1]+1]

                    line_comps = profile_group.split('-')
                    self._gaussian_profiles_plotting(line_comps, self.log,
                                                     wave_plot[idcs_plot], flux_plot[idcs_plot], z_corr,
                                                     axis=ax, cont_bands=None,
                                                     wave_array=wave_i, cont_array=cont_i,
                                                     gaussian_array=gauss_i)

            # Add the mplcursors legend
            if mplcursors_check and include_fits:
                for label, lineProfile in self._legends_dict.items():
                    mplcursors.cursor(lineProfile).connect("add", lambda sel, label=label: sel.annotation.set_text(label))

            # Plot the masked pixels
            if self._masked_inputs:
                idcs_mask = self.flux.mask
                ax.scatter(wave_plot[idcs_mask]/z_corr, flux_plot[idcs_mask]*z_corr, marker='x', label='Masked pixels',
                           color=self._color_dict['mask_marker'])

            # Switch y_axis to logarithmic scale if requested
            if log_scale:
                ax.set_yscale('log')

            # # Add the figure labels
            # _ax.set(**AXES_CONF)

            # Add or remove legend according to the plot type:
            if legend_check:
                ax.legend()

            # By default plot on screen unless an output address is provided
            save_close_fig_swicth(output_address, 'tight', fig)

        return

    def plot_fit_components(self, line=None, plt_cfg={}, ax_cfg={}, output_address=None, log_scale=False, frame='observed'):

        # Get lattest fit if line not provided (confirmed the lattest line is not blended and hence the _b not in log)
        if (line is None) and (self.line is not None):
            if self.line.endswith('_b'):
                line = self.line[:-2]

        line = self.line if line is None else line

        # Confirm if it is a blended line
        blended_check, profile_label = blended_label_from_log(line, self.log)

        # Adjust default theme
        PLT_CONF = STANDARD_PLOT.copy()
        AXES_CONF = STANDARD_AXES.copy()
        PLT_CONF['axes.labelsize'] = 14
        norm_label = r' $\,/\,{}$'.format(latex_science_float(self.norm_flux)) if self.norm_flux != 1.0 else ''
        AXES_CONF['ylabel'] = f'Flux $({UNITS_LATEX_DICT[self.units_flux]})$' + norm_label
        AXES_CONF['xlabel'] = f'Wavelength $({UNITS_LATEX_DICT[self.units_wave]})$'

        # User configuration overrites user
        PLT_CONF = {**PLT_CONF, **plt_cfg}
        AXES_CONF = {**AXES_CONF, **ax_cfg}

        with rc_context(PLT_CONF):

            # List the profile components
            list_comps = profile_label.split('-') if blended_check else [line]

            # In case the input list of lines is different from the order stored in the logs, the latter takes precedence
            if blended_check:
                log_list = list(self.log.loc[self.log.profile_label == profile_label].index.values)
                list_comps = log_list if not np.array_equal(list_comps, log_list) else list_comps

            # Reference _frame for the plot
            wave_plot, flux_plot, z_corr, idcs_mask = frame_mask_switch(self.wave, self.flux, self.redshift, frame)

            # Determine the line region # WARNING it needs to be a bit larger than original mask
            w1 = self.log.loc[line, 'w1'] * (1 + self.redshift)
            w6 = self.log.loc[line, 'w6'] * (1 + self.redshift)
            idx1, idx6 = np.searchsorted(wave_plot, (w1, w6))
            idcs_plot = np.full(wave_plot.size, False)

            idx_low = idx1 - 1 if idx1-1 >= 0 else idx1
            idx_high = idx6 + 1 if idx6+1 < wave_plot.size else idx6
            idcs_plot[idx_low:idx_high] = True

            # Continuum level
            cont_level = self.log.loc[line, 'cont']
            cont_std = self.log.loc[list_comps[0], 'std_cont']

            # Calculate the line components for upper plot
            wave_array, cont_array = linear_continuum_computation(list_comps, self.log, z_corr=(1+self.redshift))
            wave_array, gaussian_array = gaussian_profiles_computation(list_comps, self.log, z_corr=(1+self.redshift))

            # Calculate the fluxes for the residual plot
            cont_i_resd = linear_continuum_computation(list_comps, self.log, z_corr=(1+self.redshift), x_array=wave_plot[idcs_plot])
            gaussian_i_resd = gaussian_profiles_computation(list_comps, self.log, z_corr=(1+self.redshift), x_array=wave_plot[idcs_plot])
            total_resd = gaussian_i_resd.sum(axis=1) + cont_i_resd[:, 0]

            # Two axes figure, upper one for the line lower for the residual
            gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
            spec_ax = plt.subplot(gs[0])
            resid_ax = plt.subplot(gs[1], sharex=spec_ax)

            # Plot the Line spectrum
            color = self._color_dict['fg']
            spec_ax.step(wave_plot[idcs_plot]/z_corr, flux_plot[idcs_plot]*z_corr, where='mid', color=color)

            # Plot the gauss curve elements
            self._gaussian_profiles_plotting(list_comps, self.log, wave_plot[idcs_plot], flux_plot[idcs_plot], z_corr,
                                             axis=spec_ax, peak_check=False, cont_bands=True,
                                             wave_array=wave_array, cont_array=cont_array, gaussian_array=gaussian_array)

            # Lower plot residual
            label_residual = r'$\frac{F_{obs} - F_{fit}}{F_{cont}}$'
            residual = ((flux_plot[idcs_plot] - total_resd/self.norm_flux)/(cont_level/self.norm_flux))
            resid_ax.step(wave_plot[idcs_plot]/z_corr, residual*z_corr, where='mid', color=self._color_dict['fg'])

            # Shade Continuum flux standard deviation # TODO revisit this calculation
            label = r'$\sigma_{Continuum}/\overline{F_{cont}}$'
            y_limit = cont_std/cont_level
            resid_ax.fill_between(wave_plot[idcs_plot]/z_corr, -y_limit, +y_limit, facecolor='yellow', alpha=0.5, label=label)

            # Marked masked pixels if they are there
            if np.any(idcs_mask): # TODO this check should be for the line region Combine with next
                x_mask = wave_plot[idcs_plot][idcs_mask[idcs_plot]]
                y_mask = flux_plot[idcs_plot][idcs_mask[idcs_plot]]
                spec_ax.scatter(x_mask/z_corr, y_mask*z_corr, marker="x", color=self._color_dict['mask_marker'],
                                label='Masked pixels')

            # Plot masked pixels if possible
            self._mask_pixels_plotting(list_comps[0], wave_plot, flux_plot, z_corr, spec_ax, self.log)

            # Shade the pixel error spectrum if available:
            if self.err_flux is not None:
                label = r'$\sigma_{pixel}/\overline{F(cont)}$'
                err_norm = self.err_flux[idcs_plot] / (cont_level/self.norm_flux)
                resid_ax.fill_between(wave_plot[idcs_plot]/z_corr, -err_norm*z_corr, err_norm*z_corr, label=label,
                                      facecolor='salmon', alpha=0.3)

            # Switch y_axis to logarithmic scale if requested
            if log_scale:
                spec_ax.set_yscale('log')

            if mplcursors_check:
                for label, lineProfile in self._legends_dict.items():
                    mplcursors.cursor(lineProfile).connect("add", lambda sel, label=label: sel.annotation.set_text(label))

            # Add the figure labels
            spec_ax.set(**AXES_CONF)
            spec_ax.set_xlabel(None)
            spec_ax.legend()

            # Spec upper and lower limit based on absorption or emission
            if self._emission_check:
                spec_ax.set_ylim(None, self.log.loc[line, 'peak_flux']*z_corr/self.norm_flux*2)
            else:
                spec_ax.set_ylim(self.log.loc[line, 'peak_flux']*z_corr/self.norm_flux/2, None)

            # Residual x axis limit from spec axis
            resid_ax.set_xlim(spec_ax.get_xlim())

            # Residual y axis limit from std at line location
            idx_w3, idx_w4 = np.searchsorted(wave_plot[idcs_plot], self.log.loc[line, 'w3':'w4'] * (1+self.redshift))
            resd_limit = np.std(residual[idx_w3:idx_w4]) * 5

            try:
                resid_ax.set_ylim(-resd_limit, resd_limit)
            except ValueError:
                _logger.warning(f'Nan or inf entries in axis limit for {self.line}')

            # Residual plot labeling
            resid_ax.legend(loc='upper left')
            resid_ax.set_ylabel(label_residual, fontsize=22)
            resid_ax.set_xlabel(AXES_CONF['xlabel'])

            # By default plot on screen unless an output address is provided
            save_close_fig_swicth(output_address, 'tight', fig_obj=None)

        return

    def plot_line_velocity(self, line=None, output_address=None, log_scale=False, plt_cfg={}, ax_cfg={}):

        # Get lattest fit if line not provided (confirmed the lattest line is not blended and hence the _b not in log)
        if (line is None) and (self.line is not None):
            if self.line.endswith('_b'):
                line = self.line[:-2]

        line = self.line if line is None else line

        # Adjust default theme
        PLOT_CONF = STANDARD_PLOT.copy()
        AXES_CONF = STANDARD_AXES.copy()
        norm_label = r' $\,/\,{}$'.format(latex_science_float(self.norm_flux)) if self.norm_flux != 1.0 else ''
        AXES_CONF['ylabel'] = f'Flux $({UNITS_LATEX_DICT[self.units_flux]})$' + norm_label
        AXES_CONF['xlabel'] = 'Velocity (Km/s)'

        # User configuration overrites user
        PLT_CONF = {**PLOT_CONF, **plt_cfg}
        AXES_CONF = {**AXES_CONF, **ax_cfg}

        # Establish spectrum line and continua regions
        idcsEmis, idcsCont = define_masks(self.wave_rest, self.mask)

        # Load parameters from log
        peak_wave = self.log.loc[line, 'peak_wave']
        m_cont, n_cont = self.log.loc[line, 'm_cont'], self.log.loc[line, 'n_cont']
        latex_label = self.log.loc[line, 'latex_label']

        # Reference _frame for the plot
        wave_plot, flux_plot, z_corr, idcs_mask = frame_mask_switch(self.wave, self.flux, self.redshift, user_choice='observed')

        # Velocity spectrum for the line region
        flux_plot = flux_plot[idcsEmis]
        cont_plot = (m_cont * wave_plot[idcsEmis] + n_cont)/self.norm_flux
        vel_plot = c_KMpS * (wave_plot[idcsEmis] - peak_wave) / peak_wave

        vel_med = self.log.loc[line, 'v_med']
        target_percentiles = ['v_5', 'v_10', 'v_50', 'v_90', 'v_95']
        vel_percentiles = self.log.loc[line, target_percentiles].values
        FWZI = self.log.loc[line, 'FWZI']

        # Line edges
        w_i, w_f = self.log.loc[line, 'w_i'], self.log.loc[line, 'w_f']
        v_i, v_f = c_KMpS * (np.array([w_i, w_f]) - peak_wave) / peak_wave
        idx_i, idx_f = np.searchsorted(wave_plot[idcsEmis], (w_i, w_f))

        # Generate the figure
        with rc_context(PLT_CONF):

            # Plot the data
            fig, ax = plt.subplots()
            trans = ax.get_xaxis_transform()

            # Plot line spectrum
            ax.step(vel_plot, flux_plot, label=latex_label, where='mid', color=self._color_dict['fg'])

            # Velocity percentiles
            for i_percentil, percentil in enumerate(target_percentiles):

                label_text = None if i_percentil > 0 else r'$v_{Pth}$'
                ax.axvline(x=vel_percentiles[i_percentil], label=label_text, color=self._color_dict['fg'],
                              linestyle='dotted', alpha=0.5)

                label_plot = r'$v_{{{}}}$'.format(percentil[2:])
                ax.text(vel_percentiles[i_percentil], 0.80, label_plot, ha='center', va='center',
                           rotation='vertical', backgroundcolor=self._color_dict['bg'], transform=trans, alpha=0.5)

            # Velocity edges
            label_v_i, label_v_f = r'$v_{{0}}$', r'$v_{{100}}$'
            ax.axvline(x=v_i, alpha=0.5, color=self._color_dict['fg'], linestyle='dotted')
            ax.text(v_i, 0.50, label_v_i, ha='center', va='center', rotation='vertical', backgroundcolor=self._color_dict['bg'],
                    transform=trans, alpha=0.5)
            ax.axvline(x=v_f, alpha=0.5, color=self._color_dict['fg'], linestyle='dotted')
            ax.text(v_f, 0.50, label_v_f, ha='center', va='center', rotation='vertical', backgroundcolor=self._color_dict['bg'],
                    transform=trans, alpha=0.5)

            # Plot the line profile
            ax.plot(vel_plot, cont_plot, linestyle='--', color=self._color_dict['fg'])

            # Plot velocity bands
            w80 = vel_percentiles[1]-vel_percentiles[3]
            label_arrow = r'$w_{{80}}={:0.1f}\,Km/s$'.format(w80)
            p1 = patches.FancyArrowPatch((vel_percentiles[1], 0.4),
                                         (vel_percentiles[3], 0.4),
                                         label=label_arrow,
                                         arrowstyle='<->',
                                         color='tab:blue',
                                         transform=trans,
                                         mutation_scale=20)
            ax.add_patch(p1)

            # Plot FWHM bands
            label_arrow = r'$FWZI={:0.1f}\,Km/s$'.format(FWZI)
            p2 = patches.FancyArrowPatch((vel_plot[idx_i], cont_plot[idx_i]),
                                         (vel_plot[idx_f], cont_plot[idx_f]),
                                         label=label_arrow,
                                         arrowstyle='<->',
                                         color='tab:red',
                                         transform=ax.transData,
                                         mutation_scale=20)
            ax.add_patch(p2)

            # Median velocity
            label_vmed = r'$v_{{med}}={:0.1f}\,Km/s$'.format(vel_med)
            ax.axvline(x=vel_med, color=self._color_dict['fg'], label=label_vmed, linestyle='dashed', alpha=0.5)

            # Peak velocity
            label_vmed = r'$v_{{peak}}$'
            ax.axvline(x=0.0, color=self._color_dict['fg'], label=label_vmed, alpha=0.5)

            # Switch y_axis to logarithmic scale if requested
            if log_scale:
                ax.set_yscale('log')

            # Add the figure labels
            ax.set(**AXES_CONF)
            ax.legend()

            # By default plot on screen unless an output address is provided
            save_close_fig_swicth(output_address, 'tight', fig)

        return

    def plot_line_grid(self, log=None, plt_cfg={}, n_cols=5, n_rows=None, output_address=None, log_scale=True, frame='observed',
                       print_flux=False, y_scale='auto'):


        # TODO case only one line flatten fails
        # Line labels to plot
        line_list = log.index.values
        ion_array, wave_array, latex_array = label_decomposition(line_list, units_wave=self.units_wave)

        # Establish the grid shape
        if line_list.size > n_cols:
            if n_rows is None:
                n_rows = int(np.ceil(line_list.size / n_cols))
        else:
            n_cols = line_list.size
            n_rows = 1

        # Increasing the size according to the row number
        STANDARD_PLOT_grid = STANDARD_PLOT.copy()
        STANDARD_PLOT_grid['figure.figsize'] = (n_cols*5, n_rows*5)
        STANDARD_PLOT_grid['axes.titlesize'] = 12

        # New configuration overrites the old
        plt_cfg = {**STANDARD_PLOT_grid, **plt_cfg}

        with rc_context(plt_cfg):

            n_axes, n_lines = n_cols * n_rows, line_list.size

            # Reference _frame for the plot
            wave_plot, flux_plot, z_corr, idcs_mask = frame_mask_switch(self.wave, self.flux, self.redshift, frame)

            # w1 = self.log.w1.values * (1 + self.redshift)
            # w6 = self.log.w6.values * (1 + self.redshift)
            # idcsLines = ((w1 - 5) <= wave_plot[:, None]) & (wave_plot[:, None] <= (w6 + 5))

            # Determine the line region # WARNING it needs to be a bit larger than original mask
            w1 = self.log.w1.values * (1 + self.redshift)
            w6 = self.log.w6.values * (1 + self.redshift)
            idx1, idx6 = np.searchsorted(wave_plot, (w1, w6))

            idx_low = idx1 - 1
            idx_high = idx6 + 1

            fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols)
            if (n_rows + n_cols) > 2:
                axesList = ax.flatten()
            else:
                axesList = [ax]

            # Compute the gaussian profiles
            wave_array, cont_array = linear_continuum_computation(line_list, self.log, (1+self.redshift))
            wave_array, gaussian_array = gaussian_profiles_computation(line_list, self.log, (1+self.redshift))

            # Loop through the lines
            for i, ax_i in enumerate(axesList):

                if i < n_lines:

                    # Plot the spectrum
                    color = self._color_dict['fg']
                    ax_i.step(wave_plot[idx_low[i]:idx_high[i]]/z_corr, flux_plot[idx_low[i]:idx_high[i]]*z_corr,
                              where='mid', color=color)

                    # Plot the gauss curve elements
                    wave_i = wave_array[:, i][..., None]
                    cont_i = cont_array[:, i][..., None]
                    gauss_i = gaussian_array[:, i][..., None]

                    self._gaussian_profiles_plotting([line_list[i]], self.log,
                                                    wave_plot[idx_low[i]:idx_high[i]], flux_plot[idx_low[i]:idx_high[i]],
                                                     z_corr, axis=ax_i, cont_bands=True,
                                                     wave_array=wave_i, cont_array=cont_i,
                                                     gaussian_array=gauss_i)

                    # Plot masked pixels if possible
                    self._mask_pixels_plotting(line_list[i], wave_plot[idx_low[i]:idx_high[i]], flux_plot[idx_low[i]:idx_high[i]],
                                               z_corr, ax_i, self.log)

                    # Display flux if neccesary
                    if print_flux:

                        units_line_flux = ASTRO_UNITS_KEYS[self.units_wave] * ASTRO_UNITS_KEYS[self.units_flux]
                        units_flux_latex = f'{units_line_flux:latex}'
                        normFlux_latex = f' ${latex_science_float(self.norm_flux)}$' if self.norm_flux != 1 else ''

                        intg_flux = latex_science_float(log.loc[line_list[i], 'intg_flux'] / self.norm_flux, dec=3)
                        intg_err = latex_science_float(log.loc[line_list[i], 'intg_err'] / self.norm_flux, dec=3)

                        gauss_flux = latex_science_float(log.loc[line_list[i], 'gauss_flux'] / self.norm_flux, dec=3)
                        gauss_err = latex_science_float(log.loc[line_list[i], 'gauss_err'] / self.norm_flux, dec=3)

                        box_text = r'$F_{{intg}} = {}\pm{}\,$'.format(intg_flux, intg_err)
                        box_text += f'({normFlux_latex}) {units_flux_latex}\n'
                        box_text += r'$F_{{gauss}} = {}\pm{}\,$'.format(gauss_flux, gauss_err)
                        box_text += f'({normFlux_latex}) {units_flux_latex} '

                        ax_i.text(0.01, 0.9, box_text, transform=ax_i.transAxes)

                    # Axis format
                    ax_i.yaxis.set_major_locator(plt.NullLocator())
                    ax_i.yaxis.set_ticklabels([])
                    ax_i.xaxis.set_major_locator(plt.NullLocator())
                    ax_i.axes.yaxis.set_visible(False)
                    ax_i.set_title(latex_array[i])

                    # Switch y_axis to logarithmic scale if requested
                    _auto_flux_scale(ax_i, flux_plot[idx_low[i]:idx_high[i]], y_scale)
                    # if log_scale:
                    #     ax_i.set_yscale('log')

                # Clear not filled axes
                else:
                    fig.delaxes(ax_i)

            # Add the mplcursors legend
            if mplcursors_check:
                for label, lineProfile in self._legends_dict.items():
                    mplcursors.cursor(lineProfile).connect("add", lambda sel, label=label: sel.annotation.set_text(label))

            # By default plot on screen unless an output address is provided
            save_close_fig_swicth(output_address, 'tight', fig)

        return

    def _gaussian_profiles_plotting(self, list_comps, log, x, y, z_corr, axis, peak_check=False,
                                    cont_bands=True, wave_array=None, cont_array=None, gaussian_array=None):

        cmap = cm.get_cmap(self._color_dict['comps_map'])

        # Shade band regions if provided
        if cont_bands:
            mask = log.loc[list_comps[0], 'w1':'w6'].values
            idcsLine, idcsBlue, idcsRed = define_masks(x/(1 + self.redshift), mask, merge_continua=False)
            shade_line, shade_cont = self._color_dict['line_band'], self._color_dict['cont_band']
            low_lim = np.nanmin(y*z_corr)
            axis.fill_between(x[idcsBlue]/z_corr, low_lim, y[idcsBlue]*z_corr, facecolor=shade_cont, step='mid', alpha=0.25)
            axis.fill_between(x[idcsLine]/z_corr, low_lim, y[idcsLine]*z_corr, facecolor=shade_line, step='mid', alpha=0.25)
            axis.fill_between(x[idcsRed]/z_corr, low_lim, y[idcsRed]*z_corr, facecolor=shade_cont, step='mid', alpha=0.25)

        # Plot the peak flux if requested
        if peak_check and (log is not None):
            peak_wave = log.loc[list_comps[0]].peak_wave/z_corr,
            peak_flux = log.loc[list_comps[0]].peak_flux*z_corr/self.norm_flux
            axis.scatter(peak_wave, peak_flux, facecolors='red')

        # Plot the Gaussian profile
        if (gaussian_array is not None) and (cont_array is not None):

            idcs_lines = log.index.isin(list_comps)
            observations_list = log.loc[idcs_lines, 'observations'].values
            ion_array, wavelength_array, latex_array = label_decomposition(list_comps, units_wave=self.units_wave)

            # Plot the continuum,  Usine wavelength array and continuum form the first component
            cont_wave = wave_array[:, 0]
            cont_linear = cont_array[:, 0]
            axis.plot(cont_wave/z_corr, cont_linear*z_corr/self.norm_flux, color=self._color_dict['cont'],
                      label=None, linestyle='--', linewidth=0.5)

            # Individual components
            for i, line in enumerate(list_comps):

                # Color and thickness
                if len(list_comps) == 1:
                    width_i = 2 if observations_list[i] == 'no' else 3
                    style_i = '-'
                    color_i = self._color_dict['profile'] if observations_list[i] == 'no' else 'red'
                else:
                    idx_line = list_comps.index(line)
                    width_i = 2 if observations_list[i] == 'no' else 3
                    style_i = ':'
                    color_i = cmap(idx_line / len(list_comps)) if observations_list[i] == 'no' else 'red'

                # Plot the profile
                label = latex_array[i]
                x = wave_array[:, i]
                y = gaussian_array[:, i] + cont_array[:, i]
                line_g = axis.plot(x/z_corr, y*z_corr/self.norm_flux, label=label, linewidth=width_i,
                                                                      linestyle=style_i, color=color_i)

                # Compute mplcursors box text
                if mplcursors_check:
                    latex_label = log.loc[line, 'latex_label']
                    label_complex = mplcursors_legend(line, log, latex_label, self.norm_flux, self.units_wave, self.units_flux)
                    self._legends_dict[label_complex] = line_g

            # Combined profile if applicable
            if len(list_comps) > 1:

                # Combined flux compuation
                total_flux = gaussian_array.sum(axis=1)
                line_profile = (total_flux + cont_linear)

                width_i, style_i, color_i = 1, '-', self._color_dict['profile']
                axis.plot(cont_wave/z_corr, line_profile*z_corr/self.norm_flux, color=color_i, linestyle=style_i,
                                                                                linewidth=width_i)

        return

    def _mask_pixels_plotting(self, line, x, y, z_corr, axis, log):

        if 'pixel_mask' in log.columns:  # TODO remove this one at release
            pixel_mask = log.loc[line, 'pixel_mask']
            if pixel_mask != 'no':
                line_mask_limits = format_line_mask_option(pixel_mask, x)
                idcsMask = (x[:, None] >= line_mask_limits[:, 0]) & (x[:, None] <= line_mask_limits[:, 1])
                idcsMask = idcsMask.sum(axis=1).astype(bool)
                if np.sum(idcsMask) >= 1:
                    axis.scatter(x[idcsMask] / z_corr, y[idcsMask] * z_corr, marker="x",
                                 color=self._color_dict['mask_marker'])

        return

    def _plot_continuum_fit(self, continuum_fit, idcs_cont, low_lim, high_lim, threshold_factor, plot_title=''):

        PLOT_CONF = STANDARD_PLOT.copy()
        AXES_CONF = STANDARD_AXES.copy()

        norm_label = r' $\,/\,{}$'.format(latex_science_float(self.norm_flux)) if self.norm_flux != 1.0 else ''
        AXES_CONF['ylabel'] = f'Flux $({UNITS_LATEX_DICT[self.units_flux]})$' + norm_label
        AXES_CONF['xlabel'] = f'Wavelength $({UNITS_LATEX_DICT[self.units_wave]})$'
        AXES_CONF['title'] = plot_title

        wave_plot, flux_plot, z_corr, idcs_mask = frame_mask_switch(self.wave, self.flux, self.redshift, 'observed')

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

        norm_label = r' $\,/\,{}$'.format(latex_science_float(self.norm_flux)) if self.norm_flux != 1.0 else ''
        AXES_CONF['ylabel'] = f'Flux $({UNITS_LATEX_DICT[self.units_flux]})$' + norm_label
        AXES_CONF['xlabel'] = f'Wavelength $({UNITS_LATEX_DICT[self.units_wave]})$'
        AXES_CONF['title'] = plot_title

        wave_plot, flux_plot, z_corr, idcs_mask = frame_mask_switch(self.wave, self.flux, self.redshift, 'observed')

        continuum = continuum if continuum is not None else np.zeros(self.flux.size)

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
        ion_array, wave_array, latex = label_decomposition(match_log.index.values, units_wave=units_wave)
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


class SpectrumFigures(Plotter):

    def __init__(self, spectrum):

        # Instantiate the dependencies
        Plotter.__init__(self)

        # Lime spectrum object with the scientific data
        self._spec = spectrum

        # Container for the matplotlib figures
        self._fig, self._ax = None, None

        return

    def spectrum(self, extra_comp=None, line_bands=None, label=None, noise_region=None, log_scale=False,
                 output_address=None, rest_frame=False, include_fits=False, in_fig=None, in_ax=None, plt_cfg={}, ax_cfg={},
                 maximize=False, return_fig=False):

        """

        This function plots the spectrum defined by the `Spectrum class <https://lime-stable.readthedocs.io/en/latest/documentation/api.html#lime.treatment.Spectrum>`_

        The user can include an additional flux array (for example the uncertainty spectrum) to be plotted.

        Additionally, the user can include the outputs from the `.match_line_mask <https://lime-stable.readthedocs.io/en/latest/documentation/api.html#lime.treatment.Spectrum.match_line_mask>`_
        function to plot the emission peaks and the matched lines. Moreover, if the parameter ``include_fits=True`` the plot
        will include the gaussian profiles stored in the lines ``.log``.

        The user can specify the plot _frame of reference via the ``_frame='obs'`` or ``_frame='rest'`` parameter. Moreover,
        the user can provide dictionaries for the matplotlib `figure <https://matplotlib.org/stable/api/matplotlib_configuration_api.html#matplotlib.rcParams>`_
        and `axis <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set.html#matplotlib.axes.Axes.set>`_ styles.

        Finally, if the user provides an ``output_address``, the spectrum will be saved as an image instead of being displayed.

        :param comp_array: Additional flux array to be plotted alongside the spectrum flux.
        :type comp_array: numpy.array, optional

        :param peaks_table: Table with the emission and absorptions detected by the `.match_line_mask function <https://lime-stable.readthedocs.io/en/latest/documentation/api.html#lime.treatment.Spectrum.match_line_mask>`_
        :type peaks_table: astropy.Table, optional

        :param match_log: Lines log with the emission/absorptions which have matched the peaks/trough by the .match_line_mask.
        :type match_log: pandas.Dataframe, optional

        :param noise_region: 2 value array with the wavelength limits. This region will be shaded in the output plot.
        :type noise_region: np.array, optional

        :param log_scale: Set to True for a vertical (flux) axis logarithmic scale. The default value is False
        :type log_scale: bool, optional

        :param plt_cfg: Dictionary with the configuration for the matplotlib `rcParams routine <https://matplotlib.org/stable/tutorials/introductory/customizing.html#customizing-with-dynamic-rc-settings>`_ .
        :type plt_cfg: bool, optional

        :param ax_cfg: Dictionary with the configuration for the matplotlib axes style.
        :type ax_cfg: bool, optional

        :param label: Label for the spectrum plot legend, The default value is 'Observed spectrum'
        :type label: str, optional

        :param output_address: File location to store the plot as an image. If provided, the plot won't be displayed on
                               the screen.
        :type output_address: str, optional

        :param include_fits: Check to include the gaussian profile fittings in the plot. The default value is False.
        :type include_fits: Check to include the gaussian profile fittings in the plot.

        :param frame: Frame of reference for the spectrum plot: "observed" or "rest". The default value is observed.
        :param _frame: str, optional

        """

        # Display check for the user figures
        display_check = True if in_fig is None else False

        # Set figure format with the user inputs overwriting the default conf
        legend_check = True if label is not None else False
        plt_cfg.setdefault('figure.figsize', (10, 6))
        PLT_CONF, AXES_CONF = self._figure_format(plt_cfg, ax_cfg, norm_flux=self._spec.norm_flux,
                                                  units_wave=self._spec.units_wave, units_flux=self._spec.units_flux)

        # Create and fill the figure
        with rc_context(PLT_CONF):

            # Generate the figure object and figures
            if in_fig is None:
                in_fig, in_ax = self._plot_container(in_fig, in_ax, AXES_CONF)
            else:
                in_ax = in_fig.add_subplot()
                in_ax.set(**AXES_CONF)

            # Reference _frame for the plot
            wave_plot, flux_plot, z_corr, idcs_mask = frame_mask_switch_2(self._spec.wave, self._spec.flux,
                                                                          self._spec.redshift, rest_frame, )

            # Plot the spectrum
            in_ax.step(wave_plot / z_corr, flux_plot * z_corr, label=label, where='mid', color=self._color_dict['fg'])

            # Ass extra spectra if requested # TODO a more complex mechanic would be usefull
            if extra_comp is not None:
                if len(extra_comp) == len(wave_plot):
                    in_ax.step(wave_plot / z_corr, extra_comp, label='Sigma Continuum', linestyle=':', where='mid')
                else:
                    _logger.warning('The extra component array has different length than the spectrum wavelength array. '
                                    'It could not be plotted')

            # Plot peaks and troughs if provided
            if line_bands is not None:
                self._line_matching_plot(in_ax, line_bands, wave_plot, flux_plot, z_corr, self._spec.redshift,
                                         self._spec.units_wave)

            # Shade noise region if provided # TODO state colors for this one
            if noise_region is not None:
                in_ax.axvspan(noise_region[0], noise_region[1], alpha=0.15, color='tab:cyan', label='Noise region')

            # List of lines in the log
            line_list = self._spec.log.index.values

            # Plot the fittings
            if include_fits:

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

            # Plot the masked pixels
            _masks_plot(in_ax, line_list, wave_plot, flux_plot, z_corr, self._spec.log, idcs_mask, self._color_dict)

            # Switch y_axis to logarithmic scale if requested
            if log_scale:
                in_ax.set_yscale('log')

            # Add or remove legend according to the plot type:
            if legend_check:
                in_ax.legend()

            # By default, plot on screen unless an output address is provided
            save_close_fig_swicth(output_address, 'tight', in_fig, maximize, display_check)

        return in_fig

    def grid(self, log=None, rest_frame=True, y_scale='auto', include_fits=True, output_address=None, n_cols=6,
             n_rows=None, col_row_scale=(2, 1.5), maximize=False, plt_cfg={}, ax_cfg={}, in_fig=None, in_ax=None):

        # Display check for the user figures
        display_check = True if in_fig is None else False

        # If not mask provided use the log
        if log is None:
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
            default_fig_cfg.update(plt_cfg)
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
                        list_comps = profile_label.split('-') if blended_check else [line_i]

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

                    # else:
                    #     in_fig.delaxes(in_ax)

                # Show the image
                save_close_fig_swicth(output_address, 'tight', in_fig, maximize, display_check)

        else:
            _logger.info('The bands log does not contain lines')

        return in_fig

    def band(self, line=None, band_edges=None, include_fits=True, rest_frame=False, y_scale='auto', trans_line=False,
             in_fig=None, in_axis=None, plt_cfg={}, ax_cfg={}, output_address=None, maximise=False, return_fig=False):

        # Display check for the user figures
        display_check = True if in_fig is None else False

        # Establish the line and band to user for the analysis
        line, band_edges = check_line_for_bandplot(line, band_edges, self._spec, _PARENT_BANDS)

        # Guess whether we need both lines
        include_fits = include_fits and (line in self._spec.log.index)

        # Adjust the default theme
        plt_cfg.setdefault('axes.labelsize', 14)
        PLT_CONF, AXES_CONF = self._figure_format(plt_cfg, ax_cfg, self._spec.norm_flux, self._spec.units_wave,
                                                  self._spec.units_flux)

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
                                                                          self._spec.redshift, rest_frame)
            err_plot = self._spec.err_flux

            # Establish the limits for the line spectrum plot
            mask = band_edges * (1 + self._spec.redshift)
            idcsM = np.searchsorted(wave_plot, mask)

            # Plot the spectrum
            label = '' if include_fits else line
            in_ax[0].step(wave_plot[idcsM[0]:idcsM[5]] / z_corr, flux_plot[idcsM[0]:idcsM[5]] * z_corr,
                             where='mid', color=self._color_dict['fg'], label=label)

            # Add the fitting results
            if include_fits:

                # Check components
                blended_check, profile_label = blended_label_from_log(line, self._spec.log)
                list_comps = profile_label.split('-') if blended_check else [line]

                wave_array, gaussian_array = gaussian_profiles_computation(list_comps, self._spec.log, (1 + self._spec.redshift))
                wave_array, cont_array = linear_continuum_computation(list_comps, self._spec.log, (1 + self._spec.redshift))

                # Continuum bands
                self._bands_plot(in_ax[0], wave_plot, flux_plot, z_corr, idcsM, line)

                # Gaussian profiles
                idcs_lines = self._spec.log.index.isin(list_comps)
                line_g_list = _gaussian_line_profiler(in_ax[0], list_comps, wave_array, gaussian_array, cont_array,
                                                           z_corr, self._spec.log.loc[idcs_lines], self._spec.norm_flux,
                                                           color_dict=self._color_dict)

                # Add the interactive text
                _mplcursor_parser(line_g_list, list_comps, self._spec.log, self._spec.norm_flux, self._spec.units_wave,
                                       self._spec.units_flux)

                # Residual flux component
                err_region = None if err_plot is None else err_plot[idcsM[0]:idcsM[5]]
                self._residual_line_plotter(in_ax[1], wave_plot[idcsM[0]:idcsM[5]], flux_plot[idcsM[0]:idcsM[5]],
                                            err_region, list_comps, z_corr, self._spec.log, self._spec.norm_flux,
                                            self._spec.redshift, idcs_mask[idcsM[0]:idcsM[5]])

                # Synchronizing the x-axis
                in_ax[1].set_xlim(in_ax[0].get_xlim())
                in_ax[1].set_xlabel(AXES_CONF['xlabel'])
                in_ax[0].set_xlabel(None)

            # Plot the masked pixels
            _masks_plot(in_ax[0], [line], wave_plot[idcsM[0]:idcsM[5]], flux_plot[idcsM[0]:idcsM[5]], z_corr,
                        self._spec.log, idcs_mask[idcsM[0]:idcsM[5]], self._color_dict)

            # Line location
            if trans_line:
                in_ax[0].axvline(line.wavelength, linestyle='--', alpha=0.5, color=self._color_dict['fg'])

            # Display the legend
            in_ax[0].legend()

            # Set the scale
            _auto_flux_scale(in_ax[0], y=flux_plot[idcsM[0]:idcsM[5]] * z_corr, y_scale=y_scale)

            # By default, plot on screen unless an output address is provided
            save_close_fig_swicth(output_address, 'tight', in_fig, maximise, display_check)

            return

    def velocity_profile(self,  line=None, band=None, y_scale='auto', plt_cfg={}, ax_cfg={}, in_fig=None,
                         output_address=None):

        # Establish the line and band to user for the analysis
        line, band = check_line_for_bandplot(line, band, self._spec, _PARENT_BANDS)

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
                self._ax = self.fig.add_subplot()

            self._ax.set(**AXES_CONF)
            trans = self._ax.get_xaxis_transform()

            # Plot the spectrum
            self._ax.step(vel_plot, flux_plot, label=line.latex_label, where='mid', color=self._color_dict['fg'])

            # Velocity percentiles
            target_percen = ['v_5', 'v_10', 'v_50', 'v_90', 'v_95']
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
            self._ax.text(v_i, 0.50, label_v_i, ha='center', va='center', rotation='vertical',
                          backgroundcolor=self._color_dict['bg'], transform=trans, alpha=0.5)

            self._ax.axvline(x=v_f, alpha=0.5, color=self._color_dict['fg'], linestyle='dotted')
            self._ax.text(v_f, 0.50, label_v_f, ha='center', va='center', rotation='vertical',
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
            p2 = patches.FancyArrowPatch((vel_plot[idx_i], cont_plot[idx_i]),
                                         (vel_plot[idx_f], cont_plot[idx_f]),
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
            save_close_fig_swicth(output_address, 'tight', fig_obj=None)

        return

    def _residual_line_plotter(self, axis, x, y, err, list_comps, z_corr, log, norm_flux, redshift, spec_mask):

        # Continuum level
        cont_level = self._spec.log.loc[list_comps[0], 'cont']
        cont_std = self._spec.log.loc[list_comps[0], 'std_cont']

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
            _logger.warning(f'Nan or inf entries in axis limit for {self.band}')

        # Residual plot labeling
        axis.legend(loc='upper left')
        axis.set_ylabel(label_residual, fontsize=22)

        # Spectrum mask
        _masks_plot(axis, [list_comps[0]], x, y, z_corr, log, spec_mask, color_dict=self._color_dict)

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

    def cube(self, line, band=None, percentil_bg=60, line_fg=None, band_fg=None, percentils_fg=[90, 95, 99], bands_frame=None,
             bg_scale=None, fg_scale=None, bg_color='gray', fg_color='viridis', mask_color='viridis_r', mask_alpha=0.2,
             wcs=None, plt_cfg={}, ax_cfg={}, in_fig=None, in_ax=None, title=None, masks_file=None, output_address=None,
             maximise=False):


        # Prepare the background image data
        line_bg, bg_image, bg_levels, bg_scale = determine_cube_images(self._cube, line, band, bands_frame,
                                                                       percentil_bg, bg_scale, contours_check=False)

        # Prepare the foreground image data
        line_fg, fg_image, fg_levels, fg_scale = determine_cube_images(self._cube, line_fg, band_fg, bands_frame,
                                                                       percentils_fg, fg_scale, contours_check=True)

        # Mesh for the countours
        if line_fg is not None:
            y, x = np.arange(0, fg_image.shape[0]), np.arange(0, fg_image.shape[1])
            fg_mesh = np.meshgrid(x, y)
        else:
            fg_mesh = None

        # Load the masks
        masks_dict = load_spatial_masks(masks_file)

        # Check that the images have the same size
        check_image_size(bg_image, fg_image, masks_dict)

        # State the plot labelling
        title, x_label, y_label = image_map_labels(title, wcs, line_bg, line_fg, masks_dict)

        # User configuration overrites user
        ax_cfg.setdefault('xlabel', x_label)
        ax_cfg.setdefault('ylabel', y_label)
        ax_cfg.setdefault('title', title)
        plt_cfg.setdefault('figure.figsize', (5 if masks_file is None else 10, 5))
        plt_cfg.setdefault('axes.titlesize', 12)
        plt_cfg.setdefault('legend.fontsize', 10)

        PLT_CONF = {**STANDARD_PLOT, **plt_cfg}
        AXES_CONF = {**STANDARD_AXES, **ax_cfg}

        # Create and fill the figure
        with rc_context(PLT_CONF):

            # Generate the figure object
            if (in_fig is None) and (in_ax is None):
                if wcs is None:
                    in_fig, in_ax = plt.subplots()
                else:
                    in_fig = plt.figure()
                    in_ax = in_fig.add_subplot(projection=wcs, slices=('x', 'y', 1)) # TODO this extra dimension...

            # Assing the axis
            self._fig, self._ax = in_fig, in_ax
            self._ax.update(AXES_CONF)

            # Plot the image
            image_plot(self._ax, bg_image, fg_image, fg_levels, fg_mesh, bg_scale, fg_scale, bg_color, fg_color)

            # Plot the spatial masks
            if len(masks_dict) > 0:
                legend_hdl = spatial_mask_plot(self._ax, masks_dict, mask_color, mask_alpha, self._cube.units_flux)
                self._ax.legend(handles=legend_hdl, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

            # By default, plot on screen unless an output address is provided
            self._fig.canvas.draw()
            save_close_fig_swicth(output_address, 'tight', self._fig, maximise)

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

    def spectra(self, log_scale=False, output_address=None, rest_frame=False, include_fits=False,
                in_fig=None, in_axis=None, plt_cfg={}, ax_cfg={}):

        # Set figure format with the user inputs overwriting the default conf
        legend_check = True
        plt_cfg.setdefault('figure.figsize', (10, 6))
        PLT_CONF, AXES_CONF = self._figure_format(plt_cfg, ax_cfg, norm_flux=self._sample.norm_flux,
                                                  units_wave=self._sample.units_wave, units_flux=self._sample.units_flux)

        # Create and fill the figure
        with rc_context(PLT_CONF):

            # Generate the figure object and figures
            self._fig, self._ax = self._plot_container(in_fig, in_axis, AXES_CONF)

            # Loop through the spectra in the sample
            for obj_label, spec in self._sample.items():

                # Reference _frame for the plot
                wave_plot, flux_plot, z_corr, idcs_mask = frame_mask_switch_2(spec.wave, spec.flux, spec.redshift,
                                                                              rest_frame)

                # Plot the spectrum
                self._ax.step(wave_plot / z_corr, flux_plot * z_corr, label=obj_label, where='mid')

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

        return


