import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, rcParams, rcParamsDefault, spines, gridspec, patches, colors, cm
from scipy.interpolate import interp1d
from sys import exit
from astropy.io import fits
from astropy.table import Table
from .tools import label_decomposition, kinematic_component_labelling
from .model import c_KMpS, gaussian_model, linear_model
from .io import PdfMaker, load_lines_log
from astropy.wcs import WCS
import time
import copy

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


class LiMePlots:

    def __init__(self):

        return

    def plot_spectrum(self, cont_flux=None, peaks_table=None, matched_DF=None, noise_region=None,
                      log_scale=False, plt_cfg={}, ax_cfg={}, spec_label='Observed spectrum', output_address=None,
                      dark_mode=False, profile_fittings=False, frame='obs'):

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
        if cont_flux is not None:
            ax.step(wave_plot, cont_flux, label='Sigma Continuum', linestyle=':')

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

        if matched_DF is not None:
            idcs_foundLines = (matched_DF.observation.isin(('detected', 'not identified'))) & \
                              (matched_DF.wavelength >= self.wave_rest[0]) & \
                              (matched_DF.wavelength <= self.wave_rest[-1])
            if 'latexLabel' in matched_DF:
                lineLatexLabel = matched_DF.loc[idcs_foundLines].latexLabel.values
            else:
                lineLatexLabel = matched_DF.loc[idcs_foundLines].index.values
            lineWave = matched_DF.loc[idcs_foundLines].wavelength.values
            w_cor = 1 if frame == 'rest' else (1+self.redshift)
            w3 = matched_DF.loc[idcs_foundLines].w3.values * w_cor
            w4 = matched_DF.loc[idcs_foundLines].w4.values * w_cor
            observation = matched_DF.loc[idcs_foundLines].observation.values

            first_instance = True
            for i in np.arange(lineLatexLabel.size):
                if observation[i] == 'detected':
                    color_area = 'tab:red' if observation[i] == 'not identified' else 'tab:green'
                    ax.axvspan(w3[i], w4[i], alpha=0.25, color=color_area, label='Matched line' if first_instance else '_')
                    ax.text(lineWave[i] * w_cor, 0, lineLatexLabel[i], rotation=270)
                    first_instance = False

        if noise_region is not None:
            ax.axvspan(noise_region[0], noise_region[1], alpha=0.15, color='tab:cyan', label='Noise region')

        if profile_fittings:
            first_instance = True
            for lineLabel in self.log.index:

                w3, w4 = self.log.loc[lineLabel, 'w3'], self.log.loc[lineLabel, 'w4']
                m_cont, n_cont = self.log.loc[lineLabel, 'm_cont'], self.log.loc[lineLabel, 'n_cont']
                amp, center, sigma = self.log.loc[lineLabel, 'amp'], self.log.loc[lineLabel, 'center'], self.log.loc[lineLabel, 'sigma']
                wave_peak, flux_peak = self.log.loc[lineLabel, 'peak_wave'], self.log.loc[lineLabel, 'peak_flux'],

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

                line_profile = gaussian_model(wave_range, amp, center, sigma) * z_corr
                ax.plot(wave_range, cont/self.norm_flux, ':', color='tab:purple', linewidth=0.5)
                ax.plot(wave_range, (line_profile+cont)/self.norm_flux, color='tab:red', linewidth=0.5, label='Gaussian component' if first_instance else '_')
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

        ax.update({**STANDARD_AXES, **ax_cfg})
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
                latexLabel = log.loc[lineLabel, 'latexLabel']

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


# class CubeInspector(Spectrum):
#
#     """
#     This class produces an interative matplotlib window for the muse data cubes. On the left axis with the cube slice
#     image you can right click a voxel for its corresponding spectrum to be plotted on the right axis.
#     """
#
#     def __init__(self, wavelength_array, cube_flux, image_bg, image_fg=None, contour_levels_fg=None,
#                  init_coord=None, header=None, fig_conf=None, axes_conf={}, min_bg_percentil=60, lines_log_address=None):
#
#
#
#         self.fig = None
#         self.ax0, self.ax1, self.in_ax = None, None, None
#         self.grid_mesh = None
#         self.cube_flux = cube_flux
#         self.wave = wavelength_array
#         self.header = header
#         self.image_bg = image_bg
#         self.image_fg = image_fg
#         self.contour_levels_fg = contour_levels_fg
#         self.fig_conf = STANDARD_PLOT.copy()
#         self.axes_conf = {}
#         self.axlim_dict = {}
#         self.min_bg_percentil = min_bg_percentil
#         self.hdul_linelog = None
#
#         # Read the figure configuration
#         self.fig_conf = STANDARD_PLOT if fig_conf is None else fig_conf
#         rcParams.update(self.fig_conf)
#
#         # Read the axes format
#         if 'image' in axes_conf:
#             default_conf = {'xlabel': r'RA', 'ylabel': r'DEC', 'title': f'Cube flux slice'}
#             default_conf.update(axes_conf['image'])
#             self.axes_conf['image'] = default_conf
#         else:
#             self.axes_conf['image'] = {'xlabel': r'RA', 'ylabel': r'DEC', 'title': f'Cube flux slice'}
#
#         if 'spectrum' in axes_conf:
#             self.axes_conf['spectrum'] = STANDARD_AXES.update(axes_conf['spectrum'])
#         else:
#             self.axes_conf['spectrum'] = STANDARD_AXES
#
#         # Figure structure
#         self.fig = plt.figure(figsize=(18, 5))
#         gs = gridspec.GridSpec(nrows=1, ncols=2, figure=self.fig, width_ratios=[1, 2], height_ratios=[1])
#         self.fig.canvas.mpl_connect('button_press_event', self.on_click)
#         self.fig.canvas.mpl_connect('axes_enter_event', self.on_enter_axes)
#
#         # Axes configuration
#         if self.header is None:
#             self.ax0 = self.fig.add_subplot(gs[0])
#         else:
#             sky_wcs = WCS(self.header)
#             self.ax0 = self.fig.add_subplot(gs[0], projection=sky_wcs, slices=('x', 'y', 1))
#         self.ax1 = self.fig.add_subplot(gs[1])
#
#         # Image mesh grid
#         frame_size = self.cube_flux.shape
#         y, x = np.arange(0, frame_size[1]), np.arange(0, frame_size[2])
#         self.grid_mesh = np.meshgrid(x, y)
#
#         # If not central coord is provided use the middle point
#         if init_coord is None:
#             init_coord = int(self.cube_flux.shape[1]/2), int(self.cube_flux.shape[2]/2)
#
#         # Load the complete fits lines log if input
#         if lines_log_address is not None:
#             # start = time.time()
#             # linesLog_dict = {}
#             # with fits.open(lines_log_address) as hdul:
#             #     for i in np.arange(len(hdul)):
#             #         linesLog_dict[hdul[i].name] = hdul[i].data
#             # end = time.time()
#             # print(end-start)
#
#             start = time.time()
#             # with fits.open(lines_log_address, lazy_load_hdus=False) as hdul:
#             #     self.hdu_list = fits.open(lines_log_address, lazy_load_hdus=False)
#             self.hdul_linelog = fits.open(lines_log_address, lazy_load_hdus=False)
#             end = time.time()
#             print(end-start)
#
#         # Generate the plot
#         self.plot_map_voxel(self.image_bg, init_coord, self.image_fg, self.contour_levels_fg)
#         plt.show()
#
#         # Close the lins log if it has been opened
#         if isinstance(self.hdul_linelog, fits.hdu.HDUList):
#             self.hdul_linelog.close()
#
#         return
#
#     def plot_map_voxel(self, image_bg, voxel_coord=None, image_fg=None, flux_levels=None):
#
#         frame = 'obs'
#         self.norm_flux = 1e-20
#
#         min_flux = np.nanpercentile(image_bg, self.min_bg_percentil)
#         norm_color_bg = colors.SymLogNorm(linthresh=min_flux,
#                                           vmin=min_flux,
#                                           base=10)
#         self.ax0.imshow(image_bg, cmap=cm.gray, norm=norm_color_bg)
#
#         # Emphasize input coordinate
#         idx_j, idx_i = voxel_coord
#         if voxel_coord is not None:
#             self.ax0.plot(idx_i, idx_j, '+', color='red')
#
#         # Plot contours image
#         if image_fg is not None:
#             self.ax0.contour(self.grid_mesh[0], self.grid_mesh[1], image_fg, cmap='viridis', levels=flux_levels,
#                              norm=colors.LogNorm())
#
#         # Voxel spectrum
#         if voxel_coord is not None:
#             flux_voxel = self.cube_flux[:, idx_j, idx_i]
#             self.ax1.step(self.wave, flux_voxel, where='mid')
#
#         # Plot the emission line fittings:
#         if self.hdul_linelog is not None:
#             ext_name = f'{idx_j}-{idx_i}_LINELOG'
#
#             if ext_name in self.hdul_linelog:
#                 start = time.time()
#                 lineslogDF = Table.read(self.hdul_linelog[ext_name]).to_pandas()
#                 lineslogDF.set_index('index', inplace=True)
#                 self.log = lineslogDF
#                 end = time.time()
#                 print(end-start)
#             else:
#                 self.log = None
#             print(f'Cargar el pendejo {end-start}')
#             # try:
#             #     self.log = load_lines_log(self.lines_log_address, ext=ext_name)
#             # except:
#             #     self.log = None
#
#             if self.log is not None:
#
#                 flux_corr = 1
#                 self.redshift = 0.004691
#
#                 for line in self.log.index:
#
#                     w3, w4 = self.log.loc[line, 'w3'], self.log.loc[line, 'w4']
#                     m_cont, n_cont = self.log.loc[line, 'm_cont'], self.log.loc[line, 'n_cont']
#                     amp, center, sigma = self.log.loc[line, 'amp'], self.log.loc[line, 'center'], \
#                                          self.log.loc[line, 'sigma']
#                     wave_peak, flux_peak = self.log.loc[line, 'peak_wave'], self.log.loc[
#                         line, 'peak_flux'],
#
#                     # Rest frame
#                     if frame == 'rest':
#                         w3, w4 = w3 * (1 + self.redshift), w4 * (1 + self.redshift)
#                         wave_range = np.linspace(w3, w4, int((w4 - w3) * 3))
#                         cont = (m_cont * wave_range + n_cont) * flux_corr
#                         wave_range = wave_range / (1 + self.redshift)
#                         center = center / (1 + self.redshift)
#                         wave_peak = wave_peak / (1 + self.redshift)
#                         flux_peak = flux_peak * flux_corr / self.norm_flux
#
#                     # Observed frame
#                     else:
#                         w3, w4 = w3 * (1 + self.redshift), w4 * (1 + self.redshift)
#                         wave_range = np.linspace(w3, w4, int((w4 - w3) * 3))
#                         cont = (m_cont * wave_range + n_cont) * flux_corr
#
#                     line_profile = gaussian_model(wave_range, amp, center, sigma) * flux_corr
#                     self.ax1.plot(wave_range, cont / self.norm_flux, ':', color='tab:purple', linewidth=0.5)
#                     self.ax1.plot(wave_range, (line_profile + cont) / self.norm_flux, color='tab:red', linewidth=0.5)
#
#         self.axes_conf['spectrum']['title'] = f'Voxel {idx_j} - {idx_i}'
#
#         # Update the axis
#         self.ax0.update(self.axes_conf['image'])
#         self.ax1.update(self.axes_conf['spectrum'])
#
#         return
#
#     def on_click(self, event, mouse_trigger_buttton=3):
#
#         """
#         This method defines launches the new plot selection once the user clicks on an image voxel. By default this is a
#         a right click on a minimum three button mouse
#         :param event: This variable represents the user action on the plot
#         :param mouse_trigger_buttton: Number-coded mouse button which defines the button launching the voxel selection
#         :return:
#         """
#
#         if self.in_ax == self.ax0:
#
#             if event.button == mouse_trigger_buttton:
#
#                 # Save axes zoom
#                 self.save_zoom()
#
#                 # Save clicked coordinates for next plot
#                 idx_j, idx_i = np.rint(event.ydata).astype(int), np.rint(event.xdata).astype(int)
#                 print(f'Current voxel: {idx_j}-{idx_i} (mouse button {event.button})')
#
#                 # Remake the drawing
#                 self.ax0.clear()
#                 self.ax1.clear()
#                 self.plot_map_voxel(self.image_bg, (idx_j, idx_i), self.image_fg, self.contour_levels_fg)
#
#                 # Reset the image
#                 self.reset_zoom()
#                 self.fig.canvas.draw()
#
#     def on_enter_axes(self, event):
#         self.in_ax = event.inaxes
#
#     def save_zoom(self):
#         self.axlim_dict['image_xlim'] = self.ax0.get_xlim()
#         self.axlim_dict['image_ylim'] = self.ax0.get_ylim()
#         self.axlim_dict['spec_xlim'] = self.ax1.get_xlim()
#         self.axlim_dict['spec_ylim'] = self.ax1.get_ylim()
#
#     def reset_zoom(self):
#         self.ax0.set_xlim(self.axlim_dict['image_xlim'])
#         self.ax0.set_ylim(self.axlim_dict['image_ylim'])
#         self.ax1.set_xlim(self.axlim_dict['spec_xlim'])
#         self.ax1.set_ylim(self.axlim_dict['spec_ylim'])
#
