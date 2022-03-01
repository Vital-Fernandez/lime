import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, rcParams, rcParamsDefault, spines, gridspec, patches
from scipy.interpolate import interp1d
from sys import exit
from functools import partial
from collections import Sequence

from .tools import label_decomposition, kinematic_component_labelling
from .model import c_KMpS, gaussian_model, linear_model

try:
    import pylatex
    pylatex_check = True
except ImportError:
    pylatex_check = False


STANDARD_PLOT = {'figure.figsize': (12, 5),
                 'axes.titlesize': 12,
                 'axes.labelsize': 12,
                 'legend.fontsize': 10,
                 'xtick.labelsize': 10,
                 'ytick.labelsize': 10}

background_color = np.array((43, 43, 43))/255.0
foreground_color = np.array((179, 199, 216))/255.0
red_color = np.array((43, 43, 43))/255.0
yellow_color = np.array((191, 144, 0))/255.0

DARK_PLOT = {'figure.figsize': (14, 7),
             'axes.titlesize': 14,
             'axes.labelsize': 14,
             'legend.fontsize': 12,
             'xtick.labelsize': 12,
             'ytick.labelsize': 12,
             'text.color': foreground_color,
             'figure.facecolor': background_color,
             'axes.facecolor': background_color,
             'axes.edgecolor': foreground_color,
             'axes.labelcolor': foreground_color,
             'xtick.color': foreground_color,
             'ytick.color': foreground_color,
             'legend.edgecolor': 'inherit',
             'legend.facecolor': 'inherit'}

STANDARD_AXES = {'xlabel': r'Wavelength $(\AA)$', 'ylabel': r'Flux $(erg\,cm^{-2} s^{-1} \AA^{-1})$'}

FLUX_TEX_TABLE_HEADERS = [r'$Transition$', '$EW(\AA)$', '$F(\lambda)$', '$I(\lambda)$']
FLUX_TXT_TABLE_HEADERS = [r'$Transition$', 'EW', 'EW_error', 'F(lambda)', 'F(lambda)_error', 'I(lambda)', 'I(lambda)_error']

KIN_TEX_TABLE_HEADERS = [r'$Transition$', r'$Comp$', r'$v_{r}\left(\nicefrac{km}{s}\right)$', r'$\sigma_{int}\left(\nicefrac{km}{s}\right)$', r'Flux $(\nicefrac{erg}{cm^{-2} s^{-1} \AA^{-1})}$']
KIN_TXT_TABLE_HEADERS = [r'$Transition$', r'$Comp$', 'v_r', 'v_r_error', 'sigma_int', 'sigma_int_error', 'flux', 'flux_error']


def latex_science_float(f):
    float_str = "{0:.2g}".format(f)
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

        if (lines_df.loc[lineLabel, 'profile_label'] != 'None') and ('_m' not in lineLabel):
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


class LiMePlots:

    def __init__(self):

        return

    def plot_spectrum(self, comp_array=None, peaks_table=None, match_log=None, noise_region=None,
                      log_scale=False, plt_cfg={}, ax_cfg={}, spec_label='Observed spectrum', output_address=None,
                      dark_mode=False, include_fits=False, frame='observed'):

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

        :param plt_cfg: Dictionary with the configuration for the matplotlib rcParams style.
        :type plt_cfg: bool, optional

        :param ax_cfg: Dictionary with the configuration for the matplotlib axes style.
        :type ax_cfg: bool, optional

        :param spec_label: Label for the spectrum plot legend
        :type spec_label: str, optional

        :param output_address: File location to store the plot as an image. If provided, the plot won't be displayed on the screen.
        :type output_address: str, optional

        :param dark_mode: Check for the dark plot theme. The default value is False.
        :type dark_mode: bool, optional

        :param include_fits: Check to include the gaussian profile fittings in the plot. The default value is False.
        :type include_fits: Check to include the gaussian profile fittings in the plot.

        :param frame: Frame of reference for the spectrum plot: "observed" or "rest". The default value is observed.
        :param frame: str, optional

        """

        # Plot Configuration # TODO implement better switch between white and black themes
        if dark_mode:
            defaultConf = DARK_PLOT.copy()
            foreground = defaultConf['text.color']
        else:
            defaultConf = STANDARD_PLOT.copy()
            foreground = 'tab:blue'

        defaultConf.update(plt_cfg)
        rcParams.update(defaultConf)
        fig, ax = plt.subplots()

        # Redshift correction for the flux
        if frame == 'rest':
            z_corr = (1 + self.redshift)
            flux_plot = self.flux * z_corr
            wave_plot = self.wave_rest
        else:
            z_corr = 1
            flux_plot = self.flux
            wave_plot = self.wave

        # Plot the spectrum
        ax.step(wave_plot, flux_plot, label=spec_label, color=foreground, where='mid')

        # Plot the continuum if available
        if comp_array is not None:
            assert len(comp_array) == len(wave_plot), '- ERROR: The input comp_array length does not match the spectrum wavelength array length'
            ax.step(wave_plot, comp_array, label='Sigma Continuum', linestyle=':', where='mid', color='black')

        # Plot astropy detected lines if available
        if peaks_table is not None:
            color_peaks = ('tab:purple', 'tab:green')
            line_types = ('emission', 'absorption')
            labels = ('Emission peaks', 'Absorption peaks')
            for i in range(2):
                idcs_emission = peaks_table['line_type'] == line_types[i]
                idcs_linePeaks = np.array(peaks_table[idcs_emission]['line_center_index'])
                ax.scatter(wave_plot[idcs_linePeaks], flux_plot[idcs_linePeaks], label=labels[i], facecolors='none',
                           edgecolors=color_peaks[i])

        if match_log is not None:
            ion_array, wave_array, latex_array = label_decomposition(match_log.index.values)
            w_cor = 1 if frame == 'rest' else (1+self.redshift)
            w3 = match_log.w3.values * w_cor
            w4 = match_log.w4.values * w_cor

            first_instance = True
            for i in np.arange(latex_array.size):
                color_area = 'tab:green'
                ax.axvspan(w3[i], w4[i], alpha=0.25, color=color_area, label='Matched line' if first_instance else '_')
                ax.text(wave_array[i] * w_cor, 0, latex_array[i], rotation=270)
                first_instance = False

        if noise_region is not None:
            ax.axvspan(noise_region[0], noise_region[1], alpha=0.15, color='tab:cyan', label='Noise region')

        if include_fits:
            first_instance = True
            for lineLabel in self.log.index:

                w3, w4 = self.log.loc[lineLabel, 'w3'], self.log.loc[lineLabel, 'w4']
                m_cont, n_cont = self.log.loc[lineLabel, 'm_cont'], self.log.loc[lineLabel, 'n_cont']
                amp, center, sigma = self.log.loc[lineLabel, 'amp'], self.log.loc[lineLabel, 'center'], self.log.loc[lineLabel, 'sigma']
                observations = self.log.loc[lineLabel, 'observations']


                # Rest frame
                if frame == 'rest':
                    w3, w4 = w3 * (1+self.redshift), w4 * (1+self.redshift)
                    wave_range = np.linspace(w3, w4, int((w4-w3)*3))
                    cont = (m_cont * wave_range + n_cont) * z_corr
                    wave_range = wave_range / (1+self.redshift)
                    center = center/(1+self.redshift)

                # Observed frame
                else:
                    w3, w4 = w3 * (1+self.redshift), w4 * (1+self.redshift)
                    wave_range = np.linspace(w3, w4, int((w4-w3)*3))
                    cont = (m_cont * wave_range + n_cont) * z_corr

                width_curve = 0.5
                color_fit = 'tab:red'

                # Check if the measurement had an error:
                if observations != '':
                    color_fit = 'black'
                    width_curve = 2.5

                line_profile = gaussian_model(wave_range, amp, center, sigma) * z_corr
                ax.plot(wave_range, cont/self.norm_flux, ':', color='tab:purple', linewidth=0.5)
                ax.plot(wave_range, (line_profile+cont)/self.norm_flux, color=color_fit, linewidth=width_curve, label='Gaussian component' if first_instance else '_')
                # ax.scatter(wave_peak, flux_peak, color='tab:blue')
                first_instance = False

        if log_scale:
            ax.set_yscale('log')

        if self.norm_flux != 1:
            if 'ylabel' not in ax_cfg:
                y_label = STANDARD_AXES['ylabel']
                if self.norm_flux != 1.0:
                    norm_label = y_label + r' $\,/\,{}$'.format(latex_science_float(self.norm_flux))
                    ax_cfg['ylabel'] = norm_label

        ax_dict = {**STANDARD_AXES, **ax_cfg}
        ax.set(**ax_dict)
        # ax.update({**STANDARD_AXES, **ax_cfg})
        ax.legend()

        if output_address is None:
            plt.tight_layout()
            plt.show()
        else:
            plt.savefig(output_address, bbox_inches='tight')

        plt.close(fig)

        rcParams.update(rcParamsDefault)

        return

    def plot_fit_components(self, lmfit_output=None, line_label=None, fig_conf={}, ax_conf={}, output_address=None,
                                  log_scale=False, frame='rest', dark_mode=False):

        # Determine line Label:
        # TODO this function should read from lines log
        # TODO this causes issues if vary is false... need a better way to get label
        line_label = line_label if line_label is not None else self.line
        ion, wave, latexLabel = label_decomposition(line_label, scalar_output=True)

        # Plot Configuration
        if dark_mode:
            defaultConf = DARK_PLOT.copy()
            foreground = defaultConf['text.color']
            color_fit = 'yellow'
            err_shade = 'tab:yellow'
        else:
            defaultConf = STANDARD_PLOT.copy()
            foreground = 'black'
            color_fit = 'tab:blue'
            err_shade = 'tab:blue'
        width_curve = 0.7

        # Plot Configuration
        defaultConf.update(fig_conf)
        rcParams.update(defaultConf)

        defaultConf = STANDARD_AXES.copy()
        defaultConf.update(ax_conf)

        # Case in which no emission line is introduced
        if lmfit_output is None:
            fig, ax = plt.subplots()
            ax = [ax]
        else:
            # fig, ax = plt.subplots(n_rows=2)
            gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
            spec_ax = plt.subplot(gs[0])
            grid_ax = plt.subplot(gs[1], sharex=spec_ax)
            ax = [spec_ax, grid_ax]

        if frame == 'obs':
            z_cor = 1
            wave_plot = self.wave
            flux_plot = self.flux
        elif frame == 'rest':
            z_cor = 1 + self.redshift
            wave_plot = self.wave / z_cor
            flux_plot = self.flux * z_cor
        else:
            exit(f'-- Plot with frame name {frame} not recognize. Code will stop.')


        # Establish spectrum line and continua regions
        idcsEmis, idcsContBlue, idcsContRed = self.define_masks(self.wave_rest,
                                                                self.flux,
                                                                self.mask,
                                                                merge_continua=False)
        idcs_plot = (wave_plot[idcsContBlue][0] - 5 <= wave_plot) & (wave_plot <= wave_plot[idcsContRed][-1] + 5)

        # Plot line spectrum
        ax[0].step(wave_plot[idcs_plot], flux_plot[idcs_plot], label=r'Observed spectrum: {}'.format(latexLabel),
                   where='mid', color=foreground)
        ax[0].scatter(self.peak_wave/z_cor, self.peak_flux*z_cor, color='tab:blue', alpha=0.7)

        # Plot selection regions
        ax[0].fill_between(wave_plot[idcsContBlue], 0, flux_plot[idcsContBlue], facecolor='tab:orange', step='mid', alpha=0.07)
        ax[0].fill_between(wave_plot[idcsEmis], 0, flux_plot[idcsEmis], facecolor='tab:green', step='mid', alpha=0.07)
        ax[0].fill_between(wave_plot[idcsContRed], 0, flux_plot[idcsContRed], facecolor='tab:orange', step='mid', alpha=0.07)

        # Axes formatting
        if self.norm_flux != 1.0:
            defaultConf['ylabel'] = defaultConf['ylabel'] + " $/{{{0:.2g}}}$".format(self.norm_flux)

        if log_scale:
            ax[0].set_yscale('log')

        # Plot the Gaussian fit if available
        if lmfit_output is not None:

            # Recover values from fit
            x_in, y_in = lmfit_output.userkws['x'], lmfit_output.data

            # Resample gaussians
            wave_resample = np.linspace(x_in[0], x_in[-1], 200)
            flux_resample = lmfit_output.eval_components(x=wave_resample)

            # Check if the measurement had an error:
            if self.observations != '':
                color_fit = 'black'
                width_curve = 3

            # Plot input data
            # ax[0].scatter(x_in/z_cor, y_in*z_cor, color='tab:red', label='Input data', alpha=0.4)
            ax[0].plot(x_in/z_cor, lmfit_output.best_fit*z_cor, label='Gaussian fit', color=color_fit, linewidth=width_curve)

            # Plot individual components
            if not self.blended_check:
                contLabel = f'{line_label}_cont_'
            else:
                contLabel = f'{self.profile_label.split("-")[0]}_cont_'

            cont_flux = flux_resample.get(contLabel, 0.0)
            for comp_label, comp_flux in flux_resample.items():
                comp_flux = comp_flux + cont_flux if comp_label != contLabel else comp_flux
                ax[0].plot(wave_resample/z_cor, comp_flux*z_cor, label=f'{comp_label}', linestyle='--')

            # Continuum residual plot:
            residual = (y_in - lmfit_output.best_fit)/self.cont
            ax[1].step(x_in/z_cor, residual*z_cor, where='mid', color=foreground)

            # Err residual plot if available:
            if self.err_flux is not None:
                label = r'$\sigma_{Error}/\overline{F(cont)}$'
                err_norm = np.sqrt(self.err_flux[idcs_plot])/self.cont
                ax[1].fill_between(wave_plot[idcs_plot], -err_norm*z_cor, err_norm*z_cor, facecolor='tab:red', alpha=0.5, label=label)

            label = r'$\sigma_{Continuum}/\overline{F(cont)}$'
            y_low, y_high = -self.std_cont / self.cont, self.std_cont / self.cont
            ax[1].fill_between(x_in/z_cor, y_low*z_cor, y_high*z_cor, facecolor=err_shade, alpha=0.3, label=label)

            # Residual plot labeling
            ax[1].set_xlim(ax[0].get_xlim())
            ax[1].set_ylim(2*residual.min(), 2*residual.max())
            ax[1].legend(loc='upper left')
            ax[1].set_ylabel(r'$\frac{F_{obs}}{F_{fit}} - 1$')
            ax[1].set_xlabel(r'Wavelength $(\AA)$')

        ax[0].legend()
        ax[0].update(defaultConf)

        if output_address is None:
            plt.tight_layout()
            plt.show()

        else:
            plt.savefig(output_address, bbox_inches='tight')

        rcParams.update(rcParamsDefault)

        return

    def plot_line_velocity(self, line_label=None, fig_conf={}, ax_conf={}, output_address=None,
                                  log_scale=False, plot_title='', dark_mode=False):

        # Determine line Label:
        # TODO this function should read from lines log
        # TODO this causes issues if vary is false... need a better way to get label
        line_label = line_label if line_label is not None else self.line
        ion, wave, latexLabel = label_decomposition(line_label, scalar_output=True)

        # Plot Configuration
        if dark_mode:
            defaultConf = DARK_PLOT.copy()
            foreground = defaultConf['text.color']
            background = defaultConf['figure.facecolor']
            color_fit = 'yellow'
        else:
            defaultConf = STANDARD_PLOT.copy()
            foreground = 'black'
            background = 'white'
            color_fit = 'tab:blue'

        # Plot Configuration
        defaultConf.update(fig_conf)
        rcParams.update(defaultConf)

        defaultConf = STANDARD_AXES.copy()
        defaultConf.update(ax_conf)

        # Establish spectrum line and continua regions
        idcsEmis, idcsContBlue, idcsContRed = self.define_masks(self.wave_rest,
                                                                self.flux,
                                                                self.mask,
                                                                merge_continua=False)

        z_cor = 1
        vel_plot = c_KMpS * (self.wave[idcsEmis]-self.peak_wave) / self.peak_wave
        flux_plot = self.flux[idcsEmis]
        cont_plot = self.m_cont * self.wave[idcsEmis] + self.n_cont

        # Velocity values
        vel_med = np.median(vel_plot)

        target_percentiles = np.array([2, 5, 10, 50, 90, 95, 98])
        percentile_array = np.cumsum(flux_plot-cont_plot) * self.pixelWidth / self.intg_flux * 100
        percentInterp = interp1d(percentile_array, vel_plot, kind='slinear')
        vel_percentiles = percentInterp(target_percentiles)

        # Plot the data
        fig, ax = plt.subplots()
        ax = [ax]
        trans = ax[0].get_xaxis_transform()

        # Plot line spectrum
        ax[0].step(vel_plot, flux_plot, label=latexLabel, where='mid', color=foreground)

        for i_percentil, percentil in enumerate(target_percentiles):
            label_plot = r'$v_{{{}}}$'.format(percentil)
            label_text = None if i_percentil > 0 else r'$v_{Pth}$'
            ax[0].axvline(x=vel_percentiles[i_percentil], label=label_text, color=foreground, linestyle='dotted', alpha=0.5)
            ax[0].text(vel_percentiles[i_percentil], 0.80, label_plot, ha='center', va='center',
                    rotation='vertical', backgroundcolor=background, transform=trans, alpha=0.5)

        ax[0].plot(vel_plot, cont_plot, linestyle='--')

        w80 = vel_percentiles[4]-vel_percentiles[2]
        label_arrow = r'$w_{{80}}={:0.1f}\,Km/s$'.format(w80)
        p1 = patches.FancyArrowPatch((vel_percentiles[2], 0.5),
                                     (vel_percentiles[4], 0.5),
                                     label=label_arrow,
                                     arrowstyle='<->',
                                     color='tab:blue',
                                     transform=trans,
                                     mutation_scale=20)
        ax[0].add_patch(p1)

        label_vmed = r'$v_{{med}}={:0.1f}\,Km/s$'.format(vel_med)
        ax[0].axvline(x=vel_med, color=foreground, label=label_vmed, linestyle='dashed', alpha=0.5)

        label_vmed = r'$v_{{peak}}$'
        ax[0].axvline(x=0.0, color=foreground, label=label_vmed, alpha=0.5)


        # Axes formatting
        defaultConf['xlabel'] = 'Velocity (Km/s)'
        if self.norm_flux != 1.0:
            defaultConf['ylabel'] = defaultConf['ylabel'] + " $/{{{0:.2g}}}$".format(self.norm_flux)

        defaultConf['title'] = plot_title
        if log_scale:
            ax[0].set_yscale('log')

        ax[0].legend()
        ax[0].update(defaultConf)

        if output_address is None:
            plt.tight_layout()
            plt.show()
        else:
            plt.savefig(output_address, bbox_inches='tight')

        rcParams.update(rcParamsDefault)

        return

    def plot_line_grid(self, log, plotConf={}, ncols=10, nrows=None, output_address=None, log_scale=True, frame='obs'):

        # TODO put black lines for error in plotting

        # Line labels to plot
        lineLabels = log.index.values

        # Define plot axes grid size
        if nrows is None:
            nrows = int(np.ceil(lineLabels.size / ncols))
        if 'figure.figsize' not in plotConf:
            nrows = int(np.ceil(lineLabels.size / ncols))
            plotConf['figure.figsize'] = (ncols * 3, nrows * 3)
        n_axes, n_lines = ncols * nrows, lineLabels.size

        if frame == 'obs':
            z_cor = 1
            wave_plot = self.wave
            flux_plot = self.flux
        elif frame == 'rest':
            z_cor = 1 + self.redshift
            wave_plot = self.wave / z_cor
            flux_plot = self.flux * z_cor
        else:
            exit(f'-- Plot with frame name {frame} not recognize. Code will stop.')

        # Figure configuration
        defaultConf = STANDARD_PLOT.copy()
        defaultConf['axes.titlesize'] = 8
        defaultConf.update(plotConf)
        rcParams.update(defaultConf)

        fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
        axesList = ax.flatten()

        # Loop through the lines
        for i in np.arange(n_axes):
            if i < n_lines:

                # Line data
                lineLabel = lineLabels[i]
                lineWaves = log.loc[lineLabel, 'w1':'w6'].values
                latexLabel = log.loc[lineLabel, 'latex_label']

                # Establish spectrum line and continua regions
                idcsEmis, idcsContBlue, idcsContRed = self.define_masks(self.wave_rest,
                                                                        self.flux,
                                                                        lineWaves,
                                                                        merge_continua=False)
                idcs_plot = (wave_plot[idcsContBlue][0] - 5 <= wave_plot) & (
                            wave_plot <= wave_plot[idcsContRed][-1] + 5)

                # Plot observation
                ax_i = axesList[i]
                ax_i.step(wave_plot[idcs_plot], flux_plot[idcs_plot], where='mid')
                ax_i.fill_between(wave_plot[idcsContBlue], 0, flux_plot[idcsContBlue], facecolor='tab:orange', step="mid", alpha=0.2)
                ax_i.fill_between(wave_plot[idcsEmis], 0, flux_plot[idcsEmis], facecolor='tab:blue', step="mid", alpha=0.2)
                ax_i.fill_between(wave_plot[idcsContRed], 0, flux_plot[idcsContRed], facecolor='tab:orange', step="mid", alpha=0.2)

                if set(['m_cont', 'n_cont', 'amp', 'center', 'sigma']).issubset(log.columns):

                    line_params = log.loc[lineLabel, ['m_cont', 'n_cont']].values
                    gaus_params = log.loc[lineLabel, ['amp', 'center', 'sigma']].values

                    # Plot curve fitting
                    if (not pd.isnull(line_params).any()) and (not pd.isnull(gaus_params).any()):

                        wave_resample = np.linspace(self.wave[idcs_plot][0], self.wave[idcs_plot][-1], 500)

                        m_cont, n_cont = line_params /self.norm_flux
                        line_resample = linear_model(wave_resample, m_cont, n_cont)

                        amp, mu, sigma = gaus_params
                        amp = amp/self.norm_flux
                        gauss_resample = gaussian_model(wave_resample, amp, mu, sigma) + line_resample
                        ax_i.plot(wave_resample/z_cor, gauss_resample*z_cor, '--', color='tab:purple', linewidth=1.50)

                    else:
                        for child in ax_i.get_children():
                            if isinstance(child, spines.Spine):
                                child.set_color('tab:red')

                # Axis format
                ax_i.yaxis.set_major_locator(plt.NullLocator())
                ax_i.yaxis.set_ticklabels([])
                ax_i.xaxis.set_major_locator(plt.NullLocator())
                ax_i.axes.yaxis.set_visible(False)
                ax_i.set_title(latexLabel)

                if log_scale:
                    ax_i.set_yscale('log')

            # Clear not filled axes
            else:
                fig.delaxes(axesList[i])

        if output_address is None:
            plt.tight_layout()
            plt.show()
        else:
            plt.savefig(output_address, bbox_inches='tight')

        plt.close(fig)

        rcParams.update(rcParamsDefault)

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
                blended_check = (lines_df.loc[lineLabel, 'profile_label'] != 'None') and ('_m' not in lineLabel)
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
