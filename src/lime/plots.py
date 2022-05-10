import numpy as np
import pandas as pd

from matplotlib import pyplot as plt, gridspec, patches, rc_context, cm, colors
from astropy.wcs import WCS
from astropy.io import fits

from functools import partial
from collections import Sequence
from pathlib import Path

from .model import c_KMpS, gaussian_profiles_computation, linear_continuum_computation, format_line_mask_option
from .tools import label_decomposition, kinematic_component_labelling, blended_label_from_log

try:
    import pylatex
    pylatex_check = True
except ImportError:
    pylatex_check = False

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


def latex_science_float(f, dec=2):
    float_str = f'{f:.{dec}g}'
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str


def format_for_table(entry, rounddig=4, rounddig_er=2, scientific_notation=False, nan_format='none'):

    if rounddig_er == None: #TODO declare a universal tool
        rounddig_er = rounddig

    # Check None entry
    if entry != None:

        # Check string entry
        if isinstance(entry, (str, bytes)):
            formatted_entry = entry

        elif isinstance(entry, (pylatex.MultiColumn, pylatex.MultiRow, pylatex.utils.NoEscape)):
            formatted_entry = entry

        # Case of Numerical entry
        else:

            # Case of an array
            scalarVariable = True
            if isinstance(entry, (Sequence, np.ndarray)):

                # Confirm is not a single value array
                if len(entry) == 1:
                    entry = entry[0]
                # Case of an array
                else:
                    scalarVariable = False
                    formatted_entry = '_'.join(entry)  # we just put all together in a "_' joined string

            # Case single scalar
            if scalarVariable:

                # Case with error quantified # TODO add uncertainty protocol for table
                # if isinstance(entry, UFloat):
                #     formatted_entry = round_sig(nominal_values(entry), rounddig,
                #                                 scien_notation=scientific_notation) + r'$\pm$' + round_sig(
                #         std_devs(entry), rounddig_er, scien_notation=scientific_notation)

                # Case single float
                if np.isnan(entry):
                    formatted_entry = nan_format

                # Case single float
                else:
                    formatted_entry = numberStringFormat(entry, rounddig)
    else:
        # None entry is converted to None
        formatted_entry = 'None'

    return formatted_entry


def table_fluxes(lines_df, table_address, header_format_latex, table_type='pdf', fit_conf={}):

    # Check pylatex is install else leave
    if pylatex_check:
        pass
    else:
        print(f'\n- WARNING: pylatex is not installed. The table at {table_address} could not be generated')
        return

    # Establish the headers for the table
    n_columns = lines_df.columns.size
    columns_format_list = ['Line'] + ['None'] * n_columns
    for i, column in enumerate(lines_df.columns.values):
        columns_format_list[i + 1] = header_format_latex[column]

    if 'Components' in columns_format_list:
        idx_blended_label = columns_format_list.index('Components')
    else:
        idx_blended_label = None

    # Get the line latex label for the table
    ion_array, wavelength_array, latexLabel_array = label_decomposition(lines_df.index.values, comp_dict=fit_conf)

    # Create pdf
    pdf = PdfMaker()
    pdf.create_pdfDoc(pdf_type='table')
    pdf.pdf_insert_table(columns_format_list)

    # Loop through the lines
    obsLines = lines_df.index.values
    for i, lineLabel in enumerate(obsLines):
        row_raw = [latexLabel_array[i]] + list(lines_df.loc[lineLabel].values)

        # Exclude the _ from the blended label
        if idx_blended_label is not None:
            row_raw[idx_blended_label] = pylatex.utils.escape_latex(row_raw[idx_blended_label])

        # Add row to the table
        lastRow_check = True if lineLabel == obsLines[-1] else False
        pdf.addTableRow(row_raw, last_row=lastRow_check)

    # Save the pdf table
    if table_type == 'pdf':
        try:
            pdf.generate_pdf(table_address, clean_tex=True)
        except:
            print('-- PDF compilation failure')

    return


def numberStringFormat(value, cifras=4):

    if value > 0.001:
        newFormat = f'{value:.{cifras}f}'
    else:
        newFormat = f'{value:.{cifras}e}'

    return newFormat


def mplcursors_legend(line, log, latex_label, norm_flux):

    if len(latex_label) == 1:
        legend_text = latex_label[0] + '\n'

    elif line.endswith('_m'):
        legend_text = '+'.join(latex_label) + '\n'

    else:
        ion, wave, latex = label_decomposition(line, scalar_output=True)
        legend_text = latex + '\n'

    intg_flux = latex_science_float(log.loc[line, 'intg_flux']/norm_flux)
    intg_err = latex_science_float(log.loc[line, 'intg_err']/norm_flux)
    normFlux = latex_science_float(norm_flux)
    legend_text += r'$F_{{intg}} = {}\pm{}\,({})$'.format(intg_flux, intg_err, normFlux) + '\n'

    gauss_flux = latex_science_float(log.loc[line, 'gauss_flux']/norm_flux)
    gauss_err = latex_science_float(log.loc[line, 'gauss_err']/norm_flux)
    legend_text += r'$F_{{gauss}} = {}\pm{}\,({})$'.format(gauss_flux, gauss_err, normFlux) + '\n'

    v_r = r'{:.1f}'.format(log.loc[line, 'v_r'])
    v_r_err = r'{:.1f}'.format(log.loc[line, 'v_r_err'])
    legend_text += r'$v_{{r}} = {}\pm{}\,(km/s)$'.format(v_r, v_r_err) + '\n'

    sigma_vel = r'{:.1f}'.format(log.loc[line, 'sigma_vel'])
    sigma_vel_err = r'{:.1f}'.format(log.loc[line, 'sigma_vel_err'])
    legend_text += r'$\sigma_{{g}} = {}\pm{}\,(km/s)$'.format(sigma_vel, sigma_vel_err)

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

            if fits_folder is not None:

                if mask_ref is None:
                    output_image = fits_folder/f'mask_contours.png'
                else:
                    output_image = fits_folder/f'{mask_ref}_mask_contours.png'

                plt.savefig(output_image)

            if show_plot:
                plt.show()

            plt.close(fig)

    # Save to a fits file:
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


class LiMePlots:

    def __init__(self):

        self._color_dict = colorDict
        self._legends_dict = {}

        return

    def plot_spectrum(self, comp_array=None, peaks_table=None, match_log=None, noise_region=None,
                      log_scale=False, plt_cfg={}, ax_cfg={}, spec_label='spectrum', output_address=None,
                      include_fits=False, log=None, frame='observed'):

        """

        This function plots the spectrum defined by the `Spectrum class <https://lime-stable.readthedocs.io/en/latest/documentation/api.html#lime.treatment.Spectrum>`_

        The user can include an additional flux array (for example the uncertainty spectrum) to be plotted.

        Additionally, the user can include the outputs from the `.match_line_mask <https://lime-stable.readthedocs.io/en/latest/documentation/api.html#lime.treatment.Spectrum.match_line_mask>`_
        function to plot the emission peaks and the matched lines. Moreover, if the parameter ``include_fits=True`` the plot
        will include the gaussian profiles stored in the lines ``.log``.

        The user can specify the plot frame of reference via the ``frame='obs'`` or ``frame='rest'`` parameter. Moreover,
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
        :param frame: str, optional

        """

        # Adjust default theme
        PLOT_CONF = STANDARD_PLOT.copy()
        AXES_CONF = STANDARD_AXES.copy()
        PLOT_CONF['figure.figsize'] = (10, 6)

        # User configuration overrites user
        PLT_CONF = {**PLOT_CONF, **plt_cfg}
        AXES_CONF = {**AXES_CONF, **ax_cfg}

        # Use the memory log if none is provided
        log = self.log if log is None else log

        legend_check = True
        with rc_context(PLT_CONF):

            fig, ax = plt.subplots()

            # Reference frame for the plot
            wave_plot, flux_plot, z_corr, idcs_mask = self.frame_mask_switch(self.wave, self.flux, self.redshift, frame)

            # Plot the spectrum
            ax.step(wave_plot/z_corr, flux_plot*z_corr, label=spec_label, where='mid', color=self._color_dict['fg'])

            # Plot the continuum if available
            if comp_array is not None:
                assert len(comp_array) == len(wave_plot), '- ERROR: comp_array and wavelength array have mismatching length'
                ax.step(wave_plot/z_corr, comp_array, label='Sigma Continuum', linestyle=':', where='mid')

            # Plot peaks and troughs if provided
            if peaks_table is not None:
                color_peaks = (self._color_dict['peak'], self._color_dict['trough'])
                line_types = ('emission', 'absorption')
                labels = ('Peaks', 'Troughs')
                for i in range(2):
                    idcs_emission = peaks_table['line_type'] == line_types[i]
                    idcs_linePeaks = np.array(peaks_table[idcs_emission]['line_center_index'])
                    ax.scatter(wave_plot[idcs_linePeaks]/z_corr, flux_plot[idcs_linePeaks]*z_corr, label=labels[i],
                               facecolors='none', edgecolors=color_peaks[i])

            # Shade regions of matched lines if provided
            if match_log is not None:
                ion_array, wave_array, latex_array = label_decomposition(match_log.index.values)
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

                    self.gaussian_profiles_plotting([line], self.log,
                                                    wave_plot[idcs_plot], flux_plot[idcs_plot], z_corr,
                                                    axis=ax, frame=frame, cont_bands=None,
                                                    wave_array=wave_i, cont_array=cont_i,
                                                    gaussian_array=gauss_i)

                    # Plot masked pixels if possible
                    self.mask_pixels_plotting(line, wave_plot[idcs_plot], flux_plot[idcs_plot], z_corr, ax, self.log)

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
                    self.gaussian_profiles_plotting(line_comps, self.log,
                                                    wave_plot[idcs_plot], flux_plot[idcs_plot], z_corr,
                                                    axis=ax, frame=frame, cont_bands=None,
                                                    wave_array=wave_i, cont_array=cont_i,
                                                    gaussian_array=gauss_i)

            # Add the mplcursors legend
            if mplcursors_check and include_fits:
                for label, lineProfile in self._legends_dict.items():
                    mplcursors.cursor(lineProfile).connect("add", lambda sel, label=label: sel.annotation.set_text(label))

            # Switch y_axis to logarithmic scale if requested
            if log_scale:
                ax.set_yscale('log')

            # Add the axis normalization to the flux units if non provided
            if self.norm_flux != 1.0:
                AXES_CONF['ylabel'] = AXES_CONF['ylabel'] + r' $\,/\,{}$'.format(latex_science_float(self.norm_flux))

            # Add the figure labels
            ax.set(**AXES_CONF)

            # Add or remove legend according to the plot type:
            if legend_check:
                ax.legend()

            # By default plot on screen unless an output address is provided
            if output_address is None:
                plt.tight_layout()
                plt.show()
            else:
                plt.savefig(output_address, bbox_inches='tight')

            # Close the figure before leaving
            plt.close(fig)

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
        AXES_CONF.pop('xlabel')

        # User configuration overrites user
        PLT_CONF = {**PLT_CONF, **plt_cfg}
        AXES_CONF = {**AXES_CONF, **ax_cfg}

        with rc_context(PLT_CONF):

            # List the profile components
            list_comps = profile_label.split('-') if blended_check else [line]

            # Reference frame for the plot
            wave_plot, flux_plot, z_corr, idcs_mask = self.frame_mask_switch(self.wave, self.flux, self.redshift, frame)

            # Determine the line region
            w1 = self.log.loc[line, 'w1'] * (1 + self.redshift)
            w6 = self.log.loc[line, 'w6'] * (1 + self.redshift)
            idcs_plot = ((w1 - 5) <= wave_plot) & (wave_plot <= (w6 + 5))

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
            self.gaussian_profiles_plotting(list_comps, self.log, wave_plot[idcs_plot], flux_plot[idcs_plot], z_corr,
                                            axis=spec_ax, frame=frame, peak_check=True, cont_bands=True,
                                            wave_array=wave_array, cont_array=cont_array, gaussian_array=gaussian_array,
                                            mplcursors_active=True)

            # Lower plot residual
            label_residual = r'$\frac{F_{obs} - F_{fit}}{F_{cont}}$'
            residual = ((flux_plot[idcs_plot] - total_resd/self.norm_flux)/(cont_level/self.norm_flux))
            resid_ax.step(wave_plot[idcs_plot]/z_corr, residual*z_corr, where='mid', color=self._color_dict['fg'])

            # Shade Continuum flux standard deviation # TODO revisit this calculation
            label = r'$\sigma_{Continuum}/\overline{F_{cont}}$'
            y_limit = cont_std/cont_level
            resid_ax.fill_between(wave_plot[idcs_plot]/z_corr, -y_limit, +y_limit, facecolor='yellow', alpha=0.5, label=label)

            # Marked masked pixels if they are there
            if idcs_mask is not None:
                x_mask = wave_plot[idcs_plot][idcs_mask[idcs_plot]]
                y_mask = flux_plot[idcs_plot][idcs_mask[idcs_plot]]
                spec_ax.scatter(x_mask/z_corr, y_mask*z_corr, marker="x", color=self._color_dict['mask_marker'])

            # Plot masked pixels if possible
            self.mask_pixels_plotting(list_comps[0], wave_plot, flux_plot, z_corr, spec_ax, self.log)

            # Shade the pixel error spectrum if available:
            if self.err_flux is not None:
                label = r'$\sigma_{pixel}/\overline{F(cont)}$'
                err_norm = self.err_flux[idcs_plot] / (cont_level/self.norm_flux)
                resid_ax.fill_between(wave_plot[idcs_plot]/z_corr, -err_norm*z_corr, err_norm*z_corr, label=label,
                                      facecolor='salmon', alpha=0.3)

            # Add the flux normalization to units if non provided
            if self.norm_flux != 1.0:
                norm_label = AXES_CONF['ylabel'] + r' $\,/\,{}$'.format(latex_science_float(self.norm_flux))
                AXES_CONF['ylabel'] = norm_label

            # Switch y_axis to logarithmic scale if requested
            if log_scale:
                spec_ax.set_yscale('log')

            if mplcursors_check:
                for label, lineProfile in self._legends_dict.items():
                    mplcursors.cursor(lineProfile).connect("add", lambda sel, label=label: sel.annotation.set_text(label))

            # Add the figure labels
            spec_ax.set(**AXES_CONF)
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
            resid_ax.set_ylim(-resd_limit, resd_limit)

            # Residual plot labeling
            resid_ax.legend(loc='upper left')
            resid_ax.set_ylabel(label_residual, fontsize=22)
            resid_ax.set_xlabel(r'Wavelength $(\AA)$')

            # By default plot on screen unless an output address is provided
            if output_address is None:
                plt.tight_layout()
                plt.show()
            else:
                plt.savefig(output_address, bbox_inches='tight')

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
        AXES_CONF['xlabel'] = 'Velocity (Km/s)'

        # User configuration overrites user
        PLT_CONF = {**PLOT_CONF, **plt_cfg}
        AXES_CONF = {**AXES_CONF, **ax_cfg}

        # Establish spectrum line and continua regions
        idcsEmis, idcsCont = self.define_masks(self.wave_rest, self.mask)

        # Load parameters from log
        peak_wave = self.log.loc[line, 'peak_wave']
        m_cont, n_cont = self.log.loc[line, 'm_cont'], self.log.loc[line, 'n_cont']
        latex_label = self.log.loc[line, 'latex_label']

        # Reference frame for the plot
        wave_plot, flux_plot, z_corr, idcs_mask = self.frame_mask_switch(self.wave, self.flux, self.redshift, user_choice='observed')

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

            # Add the axis normalization to the flux units if non provided
            if self.norm_flux != 1.0:
                AXES_CONF['ylabel'] = AXES_CONF['ylabel'] + r' $\,/\,{}$'.format(latex_science_float(self.norm_flux))

            # Add the figure labels
            ax.set(**AXES_CONF)
            ax.legend()

            # By default plot on screen unless an output address is provided
            if output_address is None:
                plt.tight_layout()
                plt.show()
            else:
                plt.savefig(output_address, bbox_inches='tight')

            # Close the figure before leaving
            plt.close(fig)

        return

    def plot_line_grid(self, log, plt_cfg={}, ncols=10, nrows=None, output_address=None, log_scale=True, frame='observed'):

        # Line labels to plot
        line_list = log.index.values
        ion_array, wave_array, latex_array = label_decomposition(line_list)

        # Define plot axes grid size
        if nrows is None:
            nrows = int(np.ceil(line_list.size / ncols))

        # Increasing the size according to the row number
        STANDARD_PLOT_grid = STANDARD_PLOT.copy()
        STANDARD_PLOT_grid['figure.figsize'] = (ncols * 3, nrows * 3)
        STANDARD_PLOT_grid['axes.titlesize'] = 12

        # New configuration overrites the old
        plt_cfg = {**STANDARD_PLOT_grid, **plt_cfg}

        with rc_context(plt_cfg):

            n_axes, n_lines = ncols * nrows, line_list.size

            # Reference frame for the plot
            wave_plot, flux_plot, z_corr, idcs_mask = self.frame_mask_switch(self.wave, self.flux, self.redshift, frame)

            w1 = self.log.w1.values * (1 + self.redshift)
            w6 = self.log.w6.values * (1 + self.redshift)
            idcsLines = ((w1 - 5) <= wave_plot[:, None]) & (wave_plot[:, None] <= (w6 + 5))

            fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
            axesList = ax.flatten()

            # Compute the gaussian profiles
            wave_array, cont_array = linear_continuum_computation(line_list, self.log, (1+self.redshift))
            wave_array, gaussian_array = gaussian_profiles_computation(line_list, self.log, (1+self.redshift))

            # Loop through the lines
            for i, ax_i in enumerate(axesList):

                if i < n_lines:

                    # Plot the spectrum
                    color = self._color_dict['fg']
                    ax_i.step(wave_plot[idcsLines[:, i]]/z_corr, flux_plot[idcsLines[:, i]]*z_corr, where='mid', color=color)

                    # Plot the gauss curve elements
                    wave_i = wave_array[:, i][..., None]
                    cont_i = cont_array[:, i][..., None]
                    gauss_i = gaussian_array[:, i][..., None]

                    self.gaussian_profiles_plotting([line_list[i]], self.log,
                                                    wave_plot[idcsLines[:, i]], flux_plot[idcsLines[:, i]], z_corr,
                                                    axis=ax_i, frame=frame, cont_bands=True,
                                                    wave_array=wave_i, cont_array=cont_i,
                                                    gaussian_array=gauss_i)

                    # Plot masked pixels if possible
                    self.mask_pixels_plotting(line_list[i], wave_plot[idcsLines[:, i]], flux_plot[idcsLines[:, i]],
                                              z_corr, ax_i, self.log)

                    # Axis format
                    ax_i.yaxis.set_major_locator(plt.NullLocator())
                    ax_i.yaxis.set_ticklabels([])
                    ax_i.xaxis.set_major_locator(plt.NullLocator())
                    ax_i.axes.yaxis.set_visible(False)
                    ax_i.set_title(latex_array[i])

                    # Switch y_axis to logarithmic scale if requested
                    if log_scale:
                        ax_i.set_yscale('log')

                # Clear not filled axes
                else:
                    fig.delaxes(ax_i)

            # Add the mplcursors legend
            if mplcursors_check:
                for label, lineProfile in self._legends_dict.items():
                    mplcursors.cursor(lineProfile).connect("add", lambda sel, label=label: sel.annotation.set_text(label))

            # By default plot on screen unless an output address is provided
            if output_address is None:
                plt.tight_layout()
                plt.show()
            else:
                plt.savefig(output_address, bbox_inches='tight')

            plt.close(fig)

        return

    def frame_mask_switch(self, wave_obs, flux_obs, redshift, user_choice, flux_coeff=1):

        assert user_choice in ['observed', 'rest'], f'- ERROR: frame of reference {user_choice} not recognized. ' \
                                                    f'Please use "observed" or "rest".'

        # Doppler factor for rest frame plots
        z_corr = (1 + redshift) if user_choice == 'rest' else 1

        # Remove mask from plots and recover bad indeces
        if np.ma.is_masked(wave_obs):
            idcs_mask = wave_obs.mask
            wave_plot, flux_plot = wave_obs.data, flux_obs.data
            flux_plot[idcs_mask] = flux_plot[idcs_mask]/self.norm_flux

        else:
            idcs_mask = None
            wave_plot, flux_plot = wave_obs, flux_obs

        return wave_plot, flux_plot, z_corr, idcs_mask

    def gaussian_profiles_plotting(self, list_comps, log, x, y, z_corr, axis, frame='observed', peak_check=False,
                                   cont_bands=None, wave_array=None, cont_array=None, gaussian_array=None, mplcursors_active=True):

        cmap = cm.get_cmap(self._color_dict['comps_map'])

        # Shade band regions if provided
        if cont_bands is not None:
            mask = self.log.loc[list_comps[0], 'w1':'w6'].values
            idcsLine, idcsBlue, idcsRed = self.define_masks(x/(1 + self.redshift), mask, merge_continua=False)
            shade_line, shade_cont = self._color_dict['line_band'], self._color_dict['cont_band']
            axis.fill_between(x[idcsBlue]/z_corr, 0, y[idcsBlue]*z_corr, facecolor=shade_cont, step='mid', alpha=0.25)
            axis.fill_between(x[idcsLine]/z_corr, 0, y[idcsLine]*z_corr, facecolor=shade_line, step='mid', alpha=0.25)
            axis.fill_between(x[idcsRed]/z_corr, 0, y[idcsRed]*z_corr, facecolor=shade_cont, step='mid', alpha=0.25)

        # Plot the peak flux if requested
        if peak_check and (log is not None):
            peak_wave = log.loc[list_comps[0]].peak_wave/z_corr,
            peak_flux = log.loc[list_comps[0]].peak_flux*z_corr/self.norm_flux
            axis.scatter(peak_wave, peak_flux, facecolors='red')

        # Plot the Gaussian profile
        if (gaussian_array is not None) and (cont_array is not None):

            idcs_lines = log.index.isin(list_comps)
            observations_list = log.loc[idcs_lines, 'observations'].values
            ion_array, wavelength_array, latex_array = label_decomposition(list_comps)

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
                    label_complex = mplcursors_legend(line, log, latex_array, self.norm_flux)
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

    def mask_pixels_plotting(self, line, x, y, z_corr, axis, log):

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


class PdfMaker:

    def __init__(self):
        """

        """
        self.pdf_type = None
        self.pdf_geometry_options = {'right': '1cm',
                                     'left': '1cm',
                                     'top': '1cm',
                                     'bottom': '2cm'}
        self.table = None
        self.theme_table = None

        # TODO add dictionary with numeric formats for tables depending on the variable

    def create_pdfDoc(self, pdf_type=None, geometry_options=None, document_class=u'article', theme='white'):
        """

        :param pdf_type:
        :param geometry_options:
        :param document_class:
        :param theme:
        :return:
        """
        # TODO integrate this into the init
        # Case for a complete .pdf or .tex
        self.theme_table = theme

        if pdf_type is not None:

            self.pdf_type = pdf_type

            # Update the geometry if necessary (we coud define a dictionary distinction)
            if pdf_type == 'graphs':
                pdf_format = {'landscape': 'true'}
                self.pdf_geometry_options.update(pdf_format)

            elif pdf_type == 'table':
                pdf_format = {'landscape': 'true',
                              'paperwidth': '30in',
                              'paperheight': '30in'}
                self.pdf_geometry_options.update(pdf_format)

            if geometry_options is not None:
                self.pdf_geometry_options.update(geometry_options)

            # Generate the doc
            self.pdfDoc = pylatex.Document(documentclass=document_class, geometry_options=self.pdf_geometry_options)

            if theme == 'dark':
                self.pdfDoc.append(pylatex.NoEscape('\definecolor{background}{rgb}{0.169, 0.169, 0.169}'))
                self.pdfDoc.append(pylatex.NoEscape('\definecolor{foreground}{rgb}{0.702, 0.780, 0.847}'))
                self.pdfDoc.append(pylatex.NoEscape(r'\arrayrulecolor{foreground}'))

            if pdf_type == 'table':
                self.pdfDoc.packages.append(pylatex.Package('preview', options=['active', 'tightpage', ]))
                self.pdfDoc.packages.append(pylatex.Package('hyperref', options=['unicode=true', ]))
                self.pdfDoc.append(pylatex.NoEscape(r'\pagenumbering{gobble}'))
                self.pdfDoc.packages.append(pylatex.Package('nicefrac'))
                self.pdfDoc.packages.append(pylatex.Package('siunitx'))
                self.pdfDoc.packages.append(pylatex.Package('makecell'))
                # self.pdfDoc.packages.append(pylatex.Package('color', options=['usenames', 'dvipsnames', ]))  # Package to crop pdf to a figure
                self.pdfDoc.packages.append(pylatex.Package('colortbl', options=['usenames', 'dvipsnames', ]))  # Package to crop pdf to a figure
                self.pdfDoc.packages.append(pylatex.Package('xcolor', options=['table']))

            elif pdf_type == 'longtable':
                self.pdfDoc.append(pylatex.NoEscape(r'\pagenumbering{gobble}'))

        return

    def pdf_create_section(self, caption, add_page=False):

        with self.pdfDoc.create(pylatex.Section(caption)):
            if add_page:
                self.pdfDoc.append(pylatex.NewPage())

    def add_page(self):

        self.pdfDoc.append(pylatex.NewPage())

        return

    def pdf_insert_image(self, image_address, fig_loc='htbp', width=r'1\textwidth'):

        with self.pdfDoc.create(pylatex.Figure(position='h!')) as fig_pdf:
            fig_pdf.add_image(image_address, pylatex.NoEscape(width))

        return

    def pdf_insert_table(self, column_headers=None, table_format=None, addfinalLine=True, color_font=None,
                         color_background=None):

        # Set the table format
        if table_format is None:
            table_format = 'l' + 'c' * (len(column_headers) - 1)

        # Case we want to insert the table in a pdf
        if self.pdf_type != None:

            if self.pdf_type == 'table':
                self.pdfDoc.append(pylatex.NoEscape(r'\begin{preview}'))

                # Initiate the table
                with self.pdfDoc.create(pylatex.Tabular(table_format)) as self.table:
                    if column_headers != None:
                        self.table.add_hline()
                        # self.table.add_row(list(map(str, column_headers)), escape=False, strict=False)
                        output_row = list(map(partial(format_for_table), column_headers))

                        # if color_font is not None:
                        #     for i, item in enumerate(output_row):
                        #         output_row[i] = NoEscape(r'\color{{{}}}{}'.format(color_font, item))
                        #
                        # if color_background is not None:
                        #     for i, item in enumerate(output_row):
                        #         output_row[i] = NoEscape(r'\cellcolor{{{}}}{}'.format(color_background, item))

                        if (color_font is not None) or (self.theme_table != 'white'):
                            if self.theme_table == 'dark' and color_font is None:
                                color_font = 'foreground'

                            for i, item in enumerate(output_row):
                                output_row[i] = pylatex.NoEscape(r'\color{{{}}}{}'.format(color_font, item))

                        if (color_background is not None) or (self.theme_table != 'white'):
                            if self.theme_table == 'dark' and color_background is None:
                                color_background = 'background'

                            for i, item in enumerate(output_row):
                                output_row[i] = pylatex.NoEscape(r'\cellcolor{{{}}}{}'.format(color_background, item))

                        self.table.add_row(output_row, escape=False, strict=False)
                        if addfinalLine:
                            self.table.add_hline()

            elif self.pdf_type == 'longtable':

                # Initiate the table
                with self.pdfDoc.create(pylatex.LongTable(table_format)) as self.table:
                    if column_headers != None:
                        self.table.add_hline()
                        self.table.add_row(list(map(str, column_headers)), escape=False)
                        if addfinalLine:
                            self.table.add_hline()

        # Table .tex without preamble
        else:
            self.table = pylatex.Tabu(table_format)
            if column_headers != None:
                self.table.add_hline()
                # self.table.add_row(list(map(str, column_headers)), escape=False, strict=False)
                output_row = list(map(partial(format_for_table), column_headers))
                self.table.add_row(output_row, escape=False, strict=False)
                if addfinalLine:
                    self.table.add_hline()

        return

    def pdf_insert_longtable(self, column_headers=None, table_format=None):

        # Set the table format
        if table_format is None:
            table_format = 'l' + 'c' * (len(column_headers) - 1)

        # Case we want to insert the table in a pdf
        if self.pdf_type != None:

            if self.pdf_type == 'table':
                self.pdfDoc.append(pylatex.NoEscape(r'\begin{preview}'))

                # Initiate the table
            with self.pdfDoc.create(pylatex.Tabu(table_format)) as self.table:
                if column_headers != None:
                    self.table.add_hline()
                    self.table.add_row(map(str, column_headers), escape=False)
                    self.table.add_hline()

                    # Table .tex without preamble
        else:
            self.table = pylatex.LongTable(table_format)
            if column_headers != None:
                self.table.add_hline()
                self.table.add_row(list(map(str, column_headers)), escape=False)
                self.table.add_hline()

    def addTableRow(self, input_row, row_format='auto', rounddig=4, rounddig_er=None, last_row=False, color_font=None,
                    color_background=None):

        # Default formatting
        if row_format == 'auto':
            output_row = list(map(partial(format_for_table, rounddig=rounddig), input_row))

        # TODO clean this theming to default values
        if (color_font is not None) or (self.theme_table != 'white'):
            if self.theme_table == 'dark' and color_font is None:
                color_font = 'foreground'

            for i, item in enumerate(output_row):
                output_row[i] = pylatex.NoEscape(r'\color{{{}}}{}'.format(color_font, item))

        if (color_background is not None) or (self.theme_table != 'white'):
            if self.theme_table == 'dark' and color_background is None:
                color_background = 'background'

            for i, item in enumerate(output_row):
                output_row[i] = pylatex.NoEscape(r'\cellcolor{{{}}}{}'.format(color_background, item))

        # Append the row
        self.table.add_row(output_row, escape=False, strict=False)

        # Case of the final row just add one line
        if last_row:
            self.table.add_hline()

    def fig_to_pdf(self, label=None, fig_loc='htbp', width=r'1\textwidth', add_page=False, *args, **kwargs):

        with self.pdfDoc.create(pylatex.Figure(position=fig_loc)) as plot:
            plot.add_plot(width=pylatex.NoEscape(width), placement='h', *args, **kwargs)

            if label is not None:
                plot.add_caption(label)

        if add_page:
            self.pdfDoc.append(pylatex.NewPage())

    def generate_pdf(self, output_address, clean_tex=True):

        if self.pdf_type is None:
            self.table.generate_tex(str(output_address))

        else:
            if self.pdf_type == 'table':
                self.pdfDoc.append(pylatex.NoEscape(r'\end{preview}'))
            self.pdfDoc.generate_pdf(filepath=str(output_address), clean_tex=clean_tex, compiler='pdflatex')

        return
