import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, rcParams, rcParamsDefault, gridspec, patches, rc_context, cm, legend
from scipy.interpolate import interp1d
from functools import partial
from collections import Sequence

from .model import c_KMpS, gaussian_profiles_computation, linear_continuum_computation
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
             'mask_map': 'viridis'}

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

FLUX_TEX_TABLE_HEADERS = [r'$Transition$', '$EW(\AA)$', '$F(\lambda)$', '$I(\lambda)$']
FLUX_TXT_TABLE_HEADERS = [r'$Transition$', 'EW', 'EW_error', 'F(lambda)', 'F(lambda)_error', 'I(lambda)', 'I(lambda)_error']

KIN_TEX_TABLE_HEADERS = [r'$Transition$', r'$Comp$', r'$v_{r}\left(\nicefrac{km}{s}\right)$', r'$\sigma_{int}\left(\nicefrac{km}{s}\right)$', r'Flux $(\nicefrac{erg}{cm^{-2} s^{-1} \AA^{-1})}$']
KIN_TXT_TABLE_HEADERS = [r'$Transition$', r'$Comp$', 'v_r', 'v_r_error', 'sigma_int', 'sigma_int_error', 'flux', 'flux_error']


def latex_science_float(f, dec=2):
    float_str = f'{f:.{dec}g}'
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str


def format_for_table(entry, rounddig=4, rounddig_er=2, scientific_notation=False, nan_format='-'):

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


def table_fluxes(lines_df, table_address, pyneb_rc=None, scaleTable=1000, table_type='pdf'):

    # Check pylatex is install else leave
    if pylatex_check:
        pass
    else:
        print('\n- WARNING: pylatex is not installed. Flux table could not be generated')
        return

    if table_type == 'pdf':
        output_address = f'{table_address}'
    if table_type == 'txt-ascii':
        output_address = f'{table_address}.txt'

    # Measure line fluxes
    pdf = PdfMaker()
    pdf.create_pdfDoc(pdf_type='table')
    pdf.pdf_insert_table(FLUX_TEX_TABLE_HEADERS)

    # Dataframe as container as a txt file
    tableDF = pd.DataFrame(columns=FLUX_TXT_TABLE_HEADERS[1:])

    # Normalization line
    if 'H1_4861A' in lines_df.index:
        flux_Hbeta = lines_df.loc['H1_4861A', 'intg_flux']
    else:
        flux_Hbeta = scaleTable

    obsLines = lines_df.index.values
    for lineLabel in obsLines:

        label_entry = lines_df.loc[lineLabel, 'latex_label']
        wavelength = lines_df.loc[lineLabel, 'wavelength']
        eqw, eqwErr = lines_df.loc[lineLabel, 'eqw'], lines_df.loc[lineLabel, 'eqw_err']

        flux_intg = lines_df.loc[lineLabel, 'intg_flux'] / flux_Hbeta * scaleTable
        flux_intgErr = lines_df.loc[lineLabel, 'intg_err'] / flux_Hbeta * scaleTable
        flux_gauss = lines_df.loc[lineLabel, 'gauss_flux'] / flux_Hbeta * scaleTable
        flux_gaussErr = lines_df.loc[lineLabel, 'gauss_err'] / flux_Hbeta * scaleTable

        if (lines_df.loc[lineLabel, 'profile_label'] != 'no') and ('_m' not in lineLabel):
            flux, fluxErr = flux_gauss, flux_gaussErr
            label_entry = label_entry + '$_{gauss}$'
        else:
            flux, fluxErr = flux_intg, flux_intgErr

        # Correct the flux
        if pyneb_rc is not None:
            corr = pyneb_rc.getCorrHb(wavelength)
            intensity, intensityErr = flux * corr, fluxErr * corr
            intensity_entry = r'${:0.2f}\,\pm\,{:0.2f}$'.format(intensity, intensityErr)
        else:
            intensity, intensityErr = '-', '-'
            intensity_entry = '-'

        eqw_entry = r'${:0.2f}\,\pm\,{:0.2f}$'.format(eqw, eqwErr)
        flux_entry = r'${:0.2f}\,\pm\,{:0.2f}$'.format(flux, fluxErr)

        # Add row of data
        tex_row_i = [label_entry, eqw_entry, flux_entry, intensity_entry]
        txt_row_i = [label_entry, eqw, eqwErr, flux, fluxErr, intensity, intensityErr]

        lastRow_check = True if lineLabel == obsLines[-1] else False
        pdf.addTableRow(tex_row_i, last_row=lastRow_check)
        tableDF.loc[lineLabel] = txt_row_i[1:]

    if pyneb_rc is not None:

        # Data last rows
        row_Hbetaflux = [r'$H\beta$ $(erg\,cm^{-2} s^{-1} \AA^{-1})$',
                         '',
                         flux_Hbeta,
                         flux_Hbeta * pyneb_rc.getCorr(4861)]

        row_cHbeta = [r'$c(H\beta)$',
                      '',
                      float(pyneb_rc.cHbeta),
                      '']
    else:
        # Data last rows
        row_Hbetaflux = [r'$H\beta$ $(erg\,cm^{-2} s^{-1} \AA^{-1})$',
                         '',
                         flux_Hbeta,
                         '-']

        row_cHbeta = [r'$c(H\beta)$',
                      '',
                      '-',
                      '']

    pdf.addTableRow(row_Hbetaflux, last_row=False)
    pdf.addTableRow(row_cHbeta, last_row=False)
    tableDF.loc[row_Hbetaflux[0]] = row_Hbetaflux[1:] + [''] * 3
    tableDF.loc[row_cHbeta[0]] = row_cHbeta[1:] + [''] * 3

    # Format last rows
    pdf.table.add_hline()
    pdf.table.add_hline()

    # Save the pdf table
    if table_type == 'pdf':
        try:
            pdf.generate_pdf(table_address, clean_tex=True)
        except:
            print('-- PDF compilation failure')

    # Save the txt table
    if table_type == 'txt-ascii':
        with open(output_address, 'wb') as output_file:
            string_DF = tableDF.to_string()
            string_DF = string_DF.replace('$', '')
            output_file.write(string_DF.encode('UTF-8'))

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
            wave_plot, flux_plot, z_corr, mask_corr = self.plot_frame_switch(self.wave, self.flux, self.redshift, frame)

            # Plot the spectrum
            ax.step(wave_plot, flux_plot, label=spec_label, where='mid', color=self._color_dict['fg'])

            # Plot the continuum if available
            if comp_array is not None:
                assert len(comp_array) == len(wave_plot), '- ERROR: comp_array and wavelength array have mismatching length'
                ax.step(wave_plot, comp_array, label='Sigma Continuum', linestyle=':', where='mid')

            # Plot peaks and troughs if provided
            if peaks_table is not None:
                color_peaks = (self._color_dict['peak'], self._color_dict['trough'])
                line_types = ('emission', 'absorption')
                labels = ('Peaks', 'Troughs')
                for i in range(2):
                    idcs_emission = peaks_table['line_type'] == line_types[i]
                    idcs_linePeaks = np.array(peaks_table[idcs_emission]['line_center_index'])
                    ax.scatter(wave_plot[idcs_linePeaks], flux_plot[idcs_linePeaks], label=labels[i], facecolors='none',
                               edgecolors=color_peaks[i])

            # Shade regions of matched lines if provided
            if match_log is not None:
                ion_array, wave_array, latex_array = label_decomposition(match_log.index.values)
                w3 = match_log.w3.values
                w4 = match_log.w4.values
                mean_flux = np.nanmean(flux_plot)
                idcsLineBand = np.searchsorted(wave_plot, np.array([w3, w4]) * mask_corr)

                first_check = True
                for i in np.arange(latex_array.size):
                    label = 'Matched line' if first_check else '_'
                    max_region = np.max(flux_plot[idcsLineBand[0, i]:idcsLineBand[1, i]])
                    ax.axvspan(w3[i] * mask_corr, w4[i] * mask_corr, label=label, alpha=0.30, color=self._color_dict['matched_line'])
                    ax.text(wave_array[i] * mask_corr, max_region * 0.9, latex_array[i], rotation=270)
                    first_check = False

            # Shade noise region if provided
            if noise_region is not None:
                ax.axvspan(noise_region[0], noise_region[1], alpha=0.15, color='tab:cyan', label='Noise region')

            # Plot the line fittings
            if include_fits:

                legend_check = False
                w3_array, w4_array = self.log.w3.values, self.log.w4.values

                # Compute the individual profiles
                wave_array, gaussian_array = gaussian_profiles_computation(log.index.values, log, z_corr, mask_corr)
                wave_array, cont_array = linear_continuum_computation(log.index.values, log, z_corr, mask_corr)

                # Separating blended from unblended lines
                idcs_nonBlended = (self.log.index.str.endswith('_m')) | (self.log.profile_label == 'no').values

                # Plot single lines
                line_list = self.log.loc[idcs_nonBlended].index
                for line in line_list:

                    i = self.log.index.get_loc(line)

                    # Determine the line region
                    idcs_plot = ((w3_array[i] - 5) * mask_corr <= wave_plot) & (wave_plot <= (w4_array[i] + 5) * mask_corr)

                    # Plot the gauss curve elements
                    wave_i = wave_array[:, i][..., None]
                    cont_i = cont_array[:, i][..., None]
                    gauss_i = gaussian_array[:, i][..., None]

                    line_comps = [line]
                    self.gaussian_profiles_plotting(line_comps, self.log,
                                                    wave_plot[idcs_plot], flux_plot[idcs_plot], z_corr,
                                                    axis=ax, frame=frame, cont_bands=None,
                                                    wave_array=wave_i, cont_array=cont_i,
                                                    gaussian_array=gauss_i)

                # Plot combined lines
                profile_list = self.log.loc[~idcs_nonBlended, 'profile_label'].unique()
                for profile_group in profile_list:

                    idcs_group = (self.log.profile_label == profile_group)
                    i_group = np.where(idcs_group)[0]

                    # Determine the line region
                    idcs_plot = ((w3_array[i_group[0]] - 1) * mask_corr <= wave_plot) & (wave_plot <= (w4_array[i_group[0]] + 1) * mask_corr)

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
            wave_plot, flux_plot, z_corr, mask_corr = self.plot_frame_switch(self.wave, self.flux, self.redshift, frame)

            # Determine the line region
            w1, w6 = self.log.loc[line, 'w1'], self.log.loc[line, 'w6']
            idcs_plot = ((w1 - 5) * mask_corr <= wave_plot) & (wave_plot <= (w6 + 5) * mask_corr)

            # Continuum level
            cont_level = self.log.loc[line, 'cont'] * z_corr/self.norm_flux
            cont_std = self.log.loc[list_comps[0], 'std_cont'] * z_corr/self.norm_flux

            # Calculate the line components for upper plot
            wave_array, cont_array = linear_continuum_computation(list_comps, self.log, z_corr, mask_corr)
            wave_array, gaussian_array = gaussian_profiles_computation(list_comps, self.log, z_corr, mask_corr)

            # Calculate the fluxes for the residual plot
            cont_i_resd = linear_continuum_computation(list_comps, self.log, z_corr, mask_corr, x_array=wave_plot[idcs_plot])
            gaussian_i_resd = gaussian_profiles_computation(list_comps, self.log, z_corr, mask_corr, x_array=wave_plot[idcs_plot])
            total_resd = (gaussian_i_resd.sum(axis=1) + cont_i_resd[:, 0])/self.norm_flux

            # Two axes figure, upper one for the line lower for the residual
            gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
            spec_ax = plt.subplot(gs[0])
            resid_ax = plt.subplot(gs[1], sharex=spec_ax)

            # Plot the Line spectrum
            color = self._color_dict['fg']
            spec_ax.step(wave_plot[idcs_plot], flux_plot[idcs_plot], where='mid', color=color)

            # Plot the gauss curve elements
            self.gaussian_profiles_plotting(list_comps, self.log, wave_plot[idcs_plot], flux_plot[idcs_plot], z_corr,
                                            axis=spec_ax, frame=frame, peak_check=False, cont_bands=True,
                                            wave_array=wave_array, cont_array=cont_array, gaussian_array=gaussian_array,
                                            mplcursors_active=True)

            # Lower plot residual
            label_residual = r'$\frac{F_{obs} - F_{fit}}{F_{cont}}$'
            residual = ((flux_plot[idcs_plot] - total_resd)/cont_level)
            resid_ax.step(wave_plot[idcs_plot], residual, where='mid', color=self._color_dict['fg'])

            # Shade Continuum flux standard deviation # TODO revisit this calculation
            label = r'$\sigma_{Continuum}/\overline{F_{cont}}$'
            y_limit = cont_std / cont_level
            resid_ax.fill_between(wave_plot[idcs_plot], -y_limit, +y_limit, facecolor='yellow', alpha=0.5, label=label)

            # Shade the pixel error spectrum if available:
            if self.err_flux is not None:
                label = r'$\sigma_{pixel}/\overline{F(cont)}$'
                err_norm = self.err_flux[idcs_plot] * z_corr/cont_level
                resid_ax.fill_between(wave_plot[idcs_plot], -err_norm, err_norm, label=label, facecolor='salmon', alpha=0.3)

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
                spec_ax.set_ylim(None, self.log.loc[line, 'peak_flux']/self.norm_flux*2)

            else:
                spec_ax.set_ylim(self.log.loc[line, 'peak_flux']/self.norm_flux/2, None)

            # Residual x axis limit from spec axis
            resid_ax.set_xlim(spec_ax.get_xlim())

            # Residual y axis limit from std at line location
            idx_w3, idx_w4 = np.searchsorted(wave_plot[idcs_plot], self.log.loc[line, 'w3':'w4'] * mask_corr)
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

        # ion, wave, latexLabel = label_decomposition(line_label, scalar_output=True)

        # Adjust default theme
        PLOT_CONF = STANDARD_PLOT.copy()
        AXES_CONF = STANDARD_AXES.copy()
        AXES_CONF['xlabel'] = 'Velocity (Km/s)'

        # User configuration overrites user
        PLT_CONF = {**PLOT_CONF, **plt_cfg}
        AXES_CONF = {**AXES_CONF, **ax_cfg}

        # Establish spectrum line and continua regions
        idcsEmis, idcsCont = self.define_masks(self.wave_rest, self.flux, self.mask)

        # Load parameters from log
        peak_wave = self.log.loc[line, 'peak_wave']
        pixel_width = np.diff(self.wave[idcsEmis]).mean()
        m_cont, n_cont = self.log.loc[line, 'm_cont'], self.log.loc[line, 'n_cont']
        latex_label = self.log.loc[line, 'latex_label']
        intg_flux = self.log.loc[line, 'intg_flux']

        print(peak_wave)

        # Velocity spectrum for the line region
        flux_plot = self.flux[idcsEmis]
        cont_plot = (m_cont * self.wave[idcsEmis] + n_cont)/self.norm_flux
        vel_plot = c_KMpS * (self.wave[idcsEmis] - peak_wave) / peak_wave

        # Velocity values
        vel_med = np.median(vel_plot)

        target_percentiles = np.array([2, 5, 10, 50, 90, 95, 98])
        percentile_array = np.cumsum(flux_plot-cont_plot) * pixel_width/(intg_flux/self.norm_flux) * 100
        percentInterp = interp1d(percentile_array, vel_plot, kind='slinear')
        vel_percentiles = percentInterp(target_percentiles)

        # Generate the figure
        with rc_context(PLT_CONF):

            # Plot the data
            fig, ax = plt.subplots()
            trans = ax.get_xaxis_transform()

            # Plot line spectrum
            ax.step(vel_plot, flux_plot, label=latex_label, where='mid', color=self._color_dict['fg'])

            for i_percentil, percentil in enumerate(target_percentiles):

                label_text = None if i_percentil > 0 else r'$v_{Pth}$'
                ax.axvline(x=vel_percentiles[i_percentil], label=label_text, color=self._color_dict['fg'],
                              linestyle='dotted', alpha=0.5)

                label_plot = r'$v_{{{}}}$'.format(percentil)
                ax.text(vel_percentiles[i_percentil], 0.80, label_plot, ha='center', va='center',
                           rotation='vertical', backgroundcolor=self._color_dict['bg'], transform=trans, alpha=0.5)

            # Plot the line profile
            ax.plot(vel_plot, cont_plot, linestyle='--')

            # Plot velocity bands
            w80 = vel_percentiles[4]-vel_percentiles[2]
            label_arrow = r'$w_{{80}}={:0.1f}\,Km/s$'.format(w80)
            p1 = patches.FancyArrowPatch((vel_percentiles[2], 0.5),
                                         (vel_percentiles[4], 0.5),
                                         label=label_arrow,
                                         arrowstyle='<->',
                                         color='tab:blue',
                                         transform=trans,
                                         mutation_scale=20)
            ax.add_patch(p1)

            # Velocity percentiles
            label_vmed = r'$v_{{med}}={:0.1f}\,Km/s$'.format(vel_med)
            ax.axvline(x=vel_med, color=self._color_dict['fg'], label=label_vmed, linestyle='dashed', alpha=0.5)

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
            wave_plot, flux_plot, z_corr, mask_corr = self.plot_frame_switch(self.wave, self.flux, self.redshift, frame)

            w1_array, w6_array = self.log.w1.values, self.log.w6.values

            fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
            axesList = ax.flatten()

            # Compute the gaussian profiles
            wave_array, cont_array = linear_continuum_computation(line_list, self.log, z_corr, mask_corr)
            wave_array, gaussian_array = gaussian_profiles_computation(line_list, self.log, z_corr, mask_corr)

            # Loop through the lines
            for i, ax_i in enumerate(axesList):

                if i < n_lines:

                    # Determine the line region # TODO without these extra pixels the plot bands break
                    idcs_plot = ((w1_array[i] - 5) * mask_corr <= wave_plot) & (wave_plot <= (w6_array[i] + 5) * mask_corr)

                    # Plot the spectrum
                    color = self._color_dict['fg']
                    ax_i.step(wave_plot[idcs_plot], flux_plot[idcs_plot], where='mid', color=color)

                    # Plot the gauss curve elements
                    wave_i = wave_array[:, i][..., None]
                    cont_i = cont_array[:, i][..., None]
                    gauss_i = gaussian_array[:, i][..., None]

                    self.gaussian_profiles_plotting([line_list[i]], self.log,
                                                    wave_plot[idcs_plot], flux_plot[idcs_plot], z_corr,
                                                    axis=ax_i, frame=frame, cont_bands=True,
                                                    wave_array=wave_i, cont_array=cont_i,
                                                    gaussian_array=gauss_i)

                    # if mplcursors:
                    #     mplcursors.cursor().connect("add", lambda sel: sel.annotation.set_text(sel.artist.get_label()))

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

    def table_kinematics(self, lines_df, table_address, flux_normalization=1.0):

        # TODO this could be included in sr.print
        tex_address = f'{table_address}'
        txt_address = f'{table_address}.txt'

        # Measure line fluxes
        pdf = PdfMaker()
        pdf.create_pdfDoc(pdf_type='table')
        pdf.pdf_insert_table(KIN_TEX_TABLE_HEADERS)

        # Dataframe as container as a txt file
        tableDF = pd.DataFrame(columns=KIN_TXT_TABLE_HEADERS[1:])

        obsLines = lines_df.index.values
        for lineLabel in obsLines:

            if not lineLabel.endswith('_b'):
                label_entry = lines_df.loc[lineLabel, 'latexLabel']

                # Establish component:
                blended_check = (lines_df.loc[lineLabel, 'profile_label'] != 'no') and ('_m' not in lineLabel)
                if blended_check:
                    blended_group = lines_df.loc[lineLabel, 'profile_label']
                    comp = 'n1' if lineLabel.count('_') == 1 else lineLabel[lineLabel.rfind('_')+1:]
                else:
                    comp = 'n1'
                comp_label, lineEmisLabel = kinematic_component_labelling(label_entry, comp)

                wavelength = lines_df.loc[lineLabel, 'wavelength']
                v_r, v_r_err =  lines_df.loc[lineLabel, 'v_r':'v_r_err']
                sigma_vel, sigma_vel_err = lines_df.loc[lineLabel, 'sigma_vel':'sigma_vel_err']

                flux_intg = lines_df.loc[lineLabel, 'intg_flux']
                flux_intgErr = lines_df.loc[lineLabel, 'intg_err']
                flux_gauss = lines_df.loc[lineLabel, 'gauss_flux']
                flux_gaussErr = lines_df.loc[lineLabel, 'gauss_err']

                # Format the entries
                vr_entry = r'${:0.1f}\,\pm\,{:0.1f}$'.format(v_r, v_r_err)
                sigma_entry = r'${:0.1f}\,\pm\,{:0.1f}$'.format(sigma_vel, sigma_vel_err)

                if blended_check:
                    flux, fluxErr = flux_gauss, flux_gaussErr
                    label_entry = lineEmisLabel
                else:
                    flux, fluxErr = flux_intg, flux_intgErr

                # Correct the flux
                flux_entry = r'${:0.2f}\,\pm\,{:0.2f}$'.format(flux, fluxErr)

                # Add row of data
                tex_row_i = [label_entry, comp_label, vr_entry, sigma_entry, flux_entry]
                txt_row_i = [lineLabel, comp_label.replace(' ', '_'), v_r, v_r_err, sigma_vel, sigma_vel_err, flux, fluxErr]

                lastRow_check = True if lineLabel == obsLines[-1] else False
                pdf.addTableRow(tex_row_i, last_row=lastRow_check)
                tableDF.loc[lineLabel] = txt_row_i[1:]

        pdf.table.add_hline()

        # Save the pdf table
        try:
            pdf.generate_pdf(tex_address)
        except:
            print('-- PDF compilation failure')

        # Save the txt table
        with open(txt_address, 'wb') as output_file:
            string_DF = tableDF.to_string()
            string_DF = string_DF.replace('$', '')
            output_file.write(string_DF.encode('UTF-8'))

        return

    def plot_frame_switch(self, wave_obs, flux_obs, redshift, user_choice):

        assert user_choice in ['observed', 'rest'], f'- ERROR: frame of reference {user_choice} not recognized. ' \
                                                    f'Please use "observed" or "rest".'
        if user_choice == 'rest':
            z_corr = (1 + redshift)
            masc_corr = 1
            flux_plot = flux_obs * z_corr
            wave_plot = wave_obs / z_corr
        else:
            z_corr = 1
            masc_corr = (1 + redshift)
            flux_plot = flux_obs
            wave_plot = wave_obs

        return wave_plot, flux_plot, z_corr, masc_corr

    def gaussian_profiles_plotting(self, list_comps, log, x, y, z_corr, axis, frame='observed', peak_check=False,
                                   cont_bands=None, wave_array=None, cont_array=None, gaussian_array=None, mplcursors_active=True):

        # Shade band regions if provided
        cmap = cm.get_cmap('Dark2')

        if cont_bands is not None:
            # Establish line and continua bands
            mask = self.log.loc[list_comps[0], 'w1':'w6'].values
            mask_corr = 1 if frame == 'rest' else (1 + self.redshift)

            idcsLine, idcsBlue, idcsRed = self.define_masks(x/mask_corr, y, mask, merge_continua=False)
            axis.fill_between(x[idcsBlue], 0, y[idcsBlue], facecolor=self._color_dict['cont_band'], step='mid', alpha=0.25)
            axis.fill_between(x[idcsLine], 0, y[idcsLine], facecolor=self._color_dict['line_band'], step='mid', alpha=0.25)
            axis.fill_between(x[idcsRed], 0, y[idcsRed], facecolor=self._color_dict['cont_band'], step='mid', alpha=0.25)

        # Plot the peak flux if requested
        if peak_check and (log is not None):
            peak_wave = log.loc[list_comps[0]].peak_wave / z_corr,
            peak_flux = log.loc[list_comps[0]].peak_flux * z_corr/self.norm_flux
            axis.scatter(peak_wave, peak_flux, facecolors='red')

        # Plot the Gaussian profile
        if (gaussian_array is not None) and (cont_array is not None):

            idcs_lines = log.index.isin(list_comps)
            observations_list = log.loc[idcs_lines, 'observations'].values
            ion_array, wavelength_array, latex_array = label_decomposition(list_comps)

            # Plot the continuum,  Usine wavelength array and continuum form the first component
            cont_wave = wave_array[:, 0]
            cont_linear = cont_array[:, 0] / self.norm_flux
            axis.plot(cont_wave, cont_linear, color=self._color_dict['cont'], label=None, linestyle='--', linewidth=0.5)

            # Individual components
            first_check = True
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
                y = (gaussian_array[:, i] + cont_array[:, i]) / self.norm_flux
                line_g = axis.plot(wave_array[:, i], y, label=label, linewidth=width_i, linestyle=style_i, color=color_i)

                # Compute mplcursors box text
                if mplcursors_check:
                    label_complex = mplcursors_legend(line, log, latex_array, self.norm_flux)
                    self._legends_dict[label_complex] = line_g

                first_check = False

            # Combined profile if applicable
            if len(list_comps) > 1:

                # Combined flux compuation
                total_flux = gaussian_array.sum(axis=1) / self.norm_flux
                line_profile = (total_flux + cont_linear)

                width_i, style_i, color_i = 1, '-', self._color_dict['profile']
                axis.plot(cont_wave, line_profile, color=color_i, linestyle=style_i, linewidth=width_i)

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
