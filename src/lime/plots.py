import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, rcParams, spines, gridspec, patches
from scipy.interpolate import interp1d
from sys import exit

from .tools import label_decomposition, kinematic_component_labelling
from .model import c_KMpS, gaussian_model, linear_model
from .io import PdfMaker

STANDARD_PLOT = {'figure.figsize': (14, 7),
                 'axes.titlesize': 14,
                 'axes.labelsize': 14,
                 'legend.fontsize': 12,
                 'xtick.labelsize': 12,
                 'ytick.labelsize': 12}

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


class LiMePlots:

    def __init__(self):

        return

    def plot_spectrum(self, continuumFlux=None, obsLinesTable=None, matchedLinesDF=None, noise_region=None,
                      log_scale=False, plotConf={}, axConf={}, specLabel='Observed spectrum', output_address=None,
                      dark_mode=True):

        # Plot Configuration
        if dark_mode:
            defaultConf = DARK_PLOT.copy()
            foreground = defaultConf['text.color']
        else:
            defaultConf = STANDARD_PLOT.copy()
            foreground = 'tab:blue'

        defaultConf.update(plotConf)
        rcParams.update(defaultConf)
        fig, ax = plt.subplots()

        # Plot the spectrum # TODO implement better switch between white and black themes
        ax.step(self.wave_rest, self.flux, label=specLabel, color=foreground)

        # Plot the continuum if available
        if continuumFlux is not None:
            ax.step(self.wave_rest, continuumFlux, label='Sigma Continuum', linestyle=':')

        # Plot astropy detected lines if available
        if obsLinesTable is not None:
            idcs_emission = obsLinesTable['line_type'] == 'emission'
            idcs_linePeaks = np.array(obsLinesTable[idcs_emission]['line_center_index'])
            ax.scatter(self.wave_rest[idcs_linePeaks], self.flux[idcs_linePeaks], label='Detected lines', facecolors='none',
                       edgecolors='tab:purple')

        if matchedLinesDF is not None:
            idcs_foundLines = (matchedLinesDF.observation.isin(('detected', 'not identified'))) & \
                              (matchedLinesDF.wavelength >= self.wave_rest[0]) & \
                              (matchedLinesDF.wavelength <= self.wave_rest[-1])
            if 'latexLabel' in matchedLinesDF:
                lineLatexLabel = matchedLinesDF.loc[idcs_foundLines].latexLabel.values
            else:
                lineLatexLabel = matchedLinesDF.loc[idcs_foundLines].index.values
            lineWave = matchedLinesDF.loc[idcs_foundLines].wavelength.values
            w3, w4 = matchedLinesDF.loc[idcs_foundLines].w3.values, matchedLinesDF.loc[idcs_foundLines].w4.values
            observation = matchedLinesDF.loc[idcs_foundLines].observation.values

            for i in np.arange(lineLatexLabel.size):
                if observation[i] == 'detected':
                    color_area = 'tab:red' if observation[i] == 'not identified' else 'tab:green'
                    ax.axvspan(w3[i], w4[i], alpha=0.25, color=color_area)
                    ax.text(lineWave[i], 0, lineLatexLabel[i], rotation=270)

        if noise_region is not None:
            ax.axvspan(noise_region[0], noise_region[1], alpha=0.15, color='tab:cyan', label='Noise region')

        if log_scale:
            ax.set_yscale('log')

        if self.normFlux != 1:
            if 'ylabel' not in axConf:
                y_label = STANDARD_AXES['ylabel']
                if self.normFlux != 1.0:
                    norm_label = y_label + r' $\,/\,{}$'.format(latex_science_float(self.normFlux))
                    axConf['ylabel'] = norm_label

        ax.update({**STANDARD_AXES, **axConf})
        ax.legend()

        if output_address is None:
            plt.tight_layout()
            plt.show()
        else:
            plt.savefig(output_address, bbox_inches='tight')

        plt.close(fig)

        return

    def plot_fit_components(self, lmfit_output=None, line_label=None, fig_conf={}, ax_conf={}, output_address=None,
                                  log_scale=False, frame='rest', dark_mode=True):

        # Determine line Label:
        # TODO this function should read from lines log
        # TODO this causes issues if vary is false... need a better way to get label
        line_label = line_label if line_label is not None else self.lineLabel
        ion, wave, latexLabel = label_decomposition(line_label, scalar_output=True)

        # Plot Configuration
        if dark_mode:
            defaultConf = DARK_PLOT.copy()
            foreground = defaultConf['text.color']
            color_fit = 'yellow'
        else:
            defaultConf = STANDARD_PLOT.copy()
            foreground = defaultConf['text.color']
            color_fit = 'tab:blue'

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
            # fig, ax = plt.subplots(nrows=2)
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
                                                                self.lineWaves,
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
        if self.normFlux != 1.0:
            defaultConf['ylabel'] = defaultConf['ylabel'] + " $\\times{{{0:.2g}}}$".format(self.normFlux)

        if log_scale:
            ax[0].set_yscale('log')


        # Plot the Gaussian fit if available
        if lmfit_output is not None:

            # Recover values from fit
            x_in, y_in = lmfit_output.userkws['x'], lmfit_output.data

            # Resample gaussians
            wave_resample = np.linspace(x_in[0], x_in[-1], 200)
            flux_resample = lmfit_output.eval_components(x=wave_resample)

            # Plot input data
            # ax[0].scatter(x_in/z_cor, y_in*z_cor, color='tab:red', label='Input data', alpha=0.4)
            ax[0].plot(x_in/z_cor, lmfit_output.best_fit*z_cor, label='Gaussian fit', color=color_fit, linewidth=0.7)

            # Plot individual components
            if not self.blended_check:
                contLabel = f'{line_label}_cont_'
            else:
                contLabel = f'{self.blended_label.split("-")[0]}_cont_'

            cont_flux = flux_resample.get(contLabel, 0.0)
            for comp_label, comp_flux in flux_resample.items():
                comp_flux = comp_flux + cont_flux if comp_label != contLabel else comp_flux
                ax[0].plot(wave_resample/z_cor, comp_flux*z_cor, label=f'{comp_label}', linestyle='--')

            # Continuum residual plot:
            residual = (y_in - lmfit_output.best_fit)/self.cont
            ax[1].step(x_in/z_cor, residual*z_cor, where='mid', color=foreground)

            # Err residual plot if available:
            if self.errFlux is not None:
                label = r'$\sigma_{Error}/\overline{F(cont)}$'
                err_norm = np.sqrt(self.errFlux[idcs_plot])/self.cont
                ax[1].fill_between(wave_plot[idcs_plot], -err_norm*z_cor, err_norm*z_cor, facecolor='tab:red', alpha=0.5, label=label)

            label = r'$\sigma_{Continuum}/\overline{F(cont)}$'
            y_low, y_high = -self.std_cont / self.cont, self.std_cont / self.cont
            ax[1].fill_between(x_in/z_cor, y_low*z_cor, y_high*z_cor, facecolor='yellow', alpha=0.2, label=label)

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

        return

    def plot_line_velocity(self, line_label=None, fig_conf={}, ax_conf={}, output_address=None,
                                  log_scale=False, plot_title='', dark_mode=True):

        # Determine line Label:
        # TODO this function should read from lines log
        # TODO this causes issues if vary is false... need a better way to get label
        line_label = line_label if line_label is not None else self.lineLabel
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
                                                                self.lineWaves,
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
        if self.normFlux != 1.0:
            defaultConf['ylabel'] = defaultConf['ylabel'] + " $/{{{0:.2g}}}$".format(self.normFlux)

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

        return

    def plot_line_grid(self, linesDF, plotConf={}, ncols=10, nrows=None, output_address=None, log_scale=True, frame='rest'):

        # Line labels to plot
        lineLabels = linesDF.index.values

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
        defaultConf.update(plotConf)
        rcParams.update(defaultConf)

        fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
        axesList = ax.flatten()

        # Loop through the lines
        for i in np.arange(n_axes):
            if i < n_lines:

                # Line data
                lineLabel = lineLabels[i]
                lineWaves = linesDF.loc[lineLabel, 'w1':'w6'].values
                latexLabel = linesDF.loc[lineLabel, 'latexLabel']

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

                if set(['m_cont', 'n_cont', 'amp', 'center', 'sigma']).issubset(linesDF.columns):

                    line_params = linesDF.loc[lineLabel, ['m_cont', 'n_cont']].values
                    gaus_params = linesDF.loc[lineLabel, ['amp', 'center', 'sigma']].values

                    # Plot curve fitting
                    if (not pd.isnull(line_params).any()) and (not pd.isnull(gaus_params).any()):

                        wave_resample = np.linspace(self.wave[idcs_plot][0], self.wave[idcs_plot][-1], 500)

                        m_cont, n_cont = line_params /self.normFlux
                        line_resample = linear_model(wave_resample, m_cont, n_cont)

                        amp, mu, sigma = gaus_params
                        amp = amp/self.normFlux
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

        return

    def table_fluxes(self, lines_df, table_address, pyneb_rc=None, scaleTable=1000):

        # TODO this could be included in sr.print
        tex_address = f'{table_address}'
        txt_address = f'{table_address}.txt'

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

            label_entry = lines_df.loc[lineLabel, 'latexLabel']
            wavelength = lines_df.loc[lineLabel, 'wavelength']
            eqw, eqwErr = lines_df.loc[lineLabel, 'eqw'], lines_df.loc[lineLabel, 'eqw_err']

            flux_intg = lines_df.loc[lineLabel, 'intg_flux'] / flux_Hbeta * scaleTable
            flux_intgErr = lines_df.loc[lineLabel, 'intg_err'] / flux_Hbeta * scaleTable
            flux_gauss = lines_df.loc[lineLabel, 'gauss_flux'] / flux_Hbeta * scaleTable
            flux_gaussErr = lines_df.loc[lineLabel, 'gauss_err'] / flux_Hbeta * scaleTable

            if (lines_df.loc[lineLabel, 'blended_label'] != 'None') and ('_m' not in lineLabel):
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
        try:
            pdf.generate_pdf(table_address, clean_tex=True)
        except:
            print('-- PDF compilation failure')

        # Save the txt table
        with open(txt_address, 'wb') as output_file:
            string_DF = tableDF.to_string()
            string_DF = string_DF.replace('$', '')
            output_file.write(string_DF.encode('UTF-8'))

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
                blended_check = (lines_df.loc[lineLabel, 'blended_label'] != 'None') and ('_m' not in lineLabel)
                if blended_check:
                    blended_group = lines_df.loc[lineLabel, 'blended_label']
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

