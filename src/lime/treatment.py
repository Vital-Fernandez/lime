import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt, rcParams, spines, gridspec, patches

from lmfit import fit_report
from scipy.interpolate import interp1d


from .model import EmissionFitting, linear_model, gaussian_model, c_KMpS
from .tools import label_decomposition, kinematic_component_labelling
from .plots import LiMePlots
from .io import _LOG_EXPORT, LOG_COLUMNS, lineslogFile_to_DF


class Spectrum(EmissionFitting, LiMePlots):

    """
    This class provides a set of tools to measure emission lines from ionized gas to study its chemistry and kinematics


    """

    def __init__(self, input_wave=None, input_flux=None, input_err=None, linesDF_address=None, redshift=0,
                 normFlux=1, crop_waves=None):

        # Load parent classes
        EmissionFitting.__init__(self)
        LiMePlots.__init__(self)

        # Class attributes
        self.wave = None
        self.flux = None
        self.errFlux = None
        self.normFlux = normFlux
        self.redshift = redshift
        self.linesLogAddress = linesDF_address
        self.linesDF = None

        # Start cropping the input spectrum if necessary
        if crop_waves is not None:
            idcs_cropping = (input_wave >= crop_waves[0]) & (input_wave <= crop_waves[1])
            input_wave = input_wave[idcs_cropping]
            input_flux = input_flux[idcs_cropping]
            if input_err is not None:
                input_err = input_err[idcs_cropping]

        # Apply the redshift correction
        if input_wave is not None:
            self.wave_rest = input_wave / (1 + self.redshift)
            if (input_wave is not None) and (input_flux is not None):
                self.wave = input_wave
                self.flux = input_flux # * (1 + self.redshift)
                if input_err is not None:
                    self.errFlux = input_err # * (1 + self.redshift)

        # Normalize the spectrum
        if input_flux is not None:
            self.flux = self.flux / self.normFlux
            if input_err is not None:
                self.errFlux = self.errFlux / self.normFlux

        # Generate empty dataframe to store measurement use cwd as default storing folder
        self.linesLogAddress = linesDF_address
        if self.linesLogAddress is None:
            self.linesDF = pd.DataFrame(columns=LOG_COLUMNS.keys())

        # Otherwise use the one from the user
        else:
            if Path(self.linesLogAddress).is_file():
                self.linesDF = lineslogFile_to_DF(linesDF_address)
            else:
                print(f'-- WARNING: linesLog not found at {self.linesLogAddress}')

        return

    def fit_from_wavelengths(self, label, line_wavelengths, user_conf={}, algorithm='lmfit'):
        """
        This function fits an emission line by providing its label, location and an optional fit configuration. The
        algorithm accounts for the object redshift if it was provided by the user and corrects the input
        line_wavelengths

        :param str label: Line reference incluiding the ion and wavelength. Example: O3_5007A
        :param np.ndarray line_wavelengths: Array with 6 wavelength values defining an emision line left continuum,  emission region and right continuum
        :param dict user_conf: Dictionary with the user configuration for the fitting
        :param algorithm: Algorithm for the line profile fitting (Not implemented)
        """
        # For security previous measurement is cleared and a copy of the user configuration is used
        self.clear_fit()
        fit_conf = user_conf.copy()

        # Label the current measurement
        self.lineLabel = label
        self.lineWaves = line_wavelengths

        # Establish spectrum line and continua regions
        idcsEmis, idcsCont = self.define_masks(self.wave_rest, self.flux, self.lineWaves)

        # Integrated line properties
        emisWave, emisFlux = self.wave[idcsEmis], self.flux[idcsEmis]
        contWave, contFlux = self.wave[idcsCont], self.flux[idcsCont]
        err_array = self.errFlux[idcsEmis] if self.errFlux is not None else None
        self.line_properties(emisWave, emisFlux, contWave, contFlux, err_array, bootstrap_size=1000)

        # Check if blended line
        if self.lineLabel in fit_conf:
            self.blended_label = fit_conf[self.lineLabel]
            if '_b' in self.lineLabel:
                self.blended_check = True

        # Check the kinematics import
        self.import_kinematics_from_line(fit_conf, z_cor=1 + self.redshift)

        # Gaussian fitting # TODO Add logic for very small lines
        idcsLine = idcsEmis + idcsCont
        x_array = self.wave[idcsLine]
        y_array = self.flux[idcsLine]
        w_array = 1.0/self.errFlux[idcsLine] if self.errFlux is not None else np.full(x_array.size, 1.0 / self.std_cont)
        self.gauss_lmfit(self.lineLabel, x_array, y_array, w_array, fit_conf, self.linesDF, z_obj=self.redshift)

        # Safe the results to log DF
        self.results_to_database(self.lineLabel, self.linesDF, fit_conf)

        return

    def import_kinematics_from_line(self, user_conf, z_cor):

        # Check if line kinematics are contained in blended line
        if self.blended_label != 'None':
            childs_list = self.blended_label.split('-')
        else:
            childs_list = np.array(self.lineLabel, ndmin=1)

        for child_label in childs_list:
            parent_label = user_conf.get(f'{child_label}_kinem')

            if parent_label is not None:

                # Case we want to copy from previous line and the data is not available
                if (parent_label not in self.linesDF.index) and (not self.blended_check):
                    print(
                        f'-- WARNING: {parent_label} has not been measured. Its kinematics were not copied to {child_label}')

                else:
                    ion_parent, wtheo_parent, latex_parent = label_decomposition(parent_label, scalar_output=True)
                    ion_child, wtheo_child, latex_child = label_decomposition(child_label, scalar_output=True)

                    # Copy v_r and sigma_vel in wavelength units
                    for param_ext in ('center', 'sigma'):
                        param_label_child = f'{child_label}_{param_ext}'

                        # Warning overwritten existing configuration
                        if param_label_child in user_conf:
                            print(f'-- WARNING: {param_label_child} overwritten by {parent_label} kinematics in configuration input')

                        # Case where parent and child are in blended group
                        if parent_label in childs_list:
                            param_label_parent = f'{parent_label}_{param_ext}'
                            param_expr_parent = f'{wtheo_child / wtheo_parent:0.8f}*{param_label_parent}'

                            user_conf[param_label_child] = {'expr': param_expr_parent}

                        # Case we want to copy from previously measured line
                        else:
                            mu_parent = self.linesDF.loc[parent_label, ['center', 'center_err']].values
                            sigma_parent = self.linesDF.loc[parent_label, ['sigma', 'sigma_err']].values

                            if param_ext == 'center':
                                param_value = wtheo_child / wtheo_parent * (mu_parent / z_cor)
                            else:
                                param_value = wtheo_child / wtheo_parent * sigma_parent

                            user_conf[param_label_child] = {'value': param_value[0], 'vary': False}
                            user_conf[f'{param_label_child}_err'] = param_value[1]

        return

    def results_to_database(self, lineLabel, linesDF, fit_conf, export_params=_LOG_EXPORT):

        # Recover label data
        if self.blended_check:
            line_components = self.blended_label.split('-')
        else:
            line_components = np.array([lineLabel], ndmin=1)

        ion, waveRef, latexLabel = label_decomposition(line_components, combined_dict=fit_conf)

        # Loop through the line components
        for i, line in enumerate(line_components):

            # Convert current measurement to a pandas series container
            line_log = pd.Series(index=LOG_COLUMNS.keys())
            line_log['ion', 'wavelength', 'latexLabel'] = ion[i], waveRef[i], latexLabel[i]
            line_log['w1': 'w6'] = self.lineWaves

            # Treat every line
            for param in export_params:

                # Get component parameter
                if LOG_COLUMNS[param][2]:
                    param_value = self.__getattribute__(param)[i]
                else:
                    param_value = self.__getattribute__(param)

                # De normalize
                if LOG_COLUMNS[param][0]:
                    param_value = param_value * self.normFlux

                line_log[param] = param_value

            # Assign line series to dataframe
            linesDF.loc[line] = line_log

        return

    def display_results(self, label=None, show_fit_report=True, show_plot=False, log_scale=True, frame='obs'):

        # Case no line as input: Show the current measurement
        if label is None:
            if self.lineLabel is not None:
                label = self.lineLabel
                output_ref = (f'Input line: {label}\n'
                              f'- Line regions: {self.lineWaves}\n'
                              f'- Spectrum: normalization flux: {self.normFlux}; redshift {self.redshift}\n'
                              f'- Peak: wavelength {self.peak_wave:.2f}; peak intensity {self.peak_flux:.2f}\n'
                              f'- Continuum: slope {self.m_cont:.2e}; intercept {self.n_cont:.2e}\n')

                if self.blended_check:
                    mixtureComponents = np.array(self.blended_label.split('-'))
                else:
                    mixtureComponents = np.array([label], ndmin=1)

                if mixtureComponents.size == 1:
                    output_ref += f'- Intg Eqw: {self.eqw[0]:.2f} +/- {self.eqw_err[0]:.2f}\n'

                output_ref += f'- Intg flux: {self.intg_flux:.3f} +/- {self.intg_err:.3f}\n'

                for i, lineRef in enumerate(mixtureComponents):
                    output_ref += (f'- {lineRef} gaussian fitting:\n'
                                   f'-- Gauss flux: {self.gauss_flux[i]:.3f} +/- {self.gauss_err[i]:.3f}\n'
                                   f'-- Amplitude: {self.amp[i]:.3f} +/- {self.amp_err[i]:.3f}\n'
                                   f'-- Center: {self.center[i]:.2f} +/- {self.center_err[i]:.2f}\n'
                                   f'-- Sigma: {self.sigma[i]:.2f} +/- {self.sigma_err[i]:.2f}\n\n')
            else:
                output_ref = f'- No measurement performed\n'

        # Case with line input: search and show that measurement
        elif self.linesDF is not None:
            if label in self.linesDF.index:
                output_ref = self.linesDF.loc[label].to_string
            else:
                output_ref = f'- WARNING: {label} not found in  lines table\n'
        else:
            output_ref = '- WARNING: Measurement lines log not defined\n'

        # Display the print lmfit report if available
        if show_fit_report:
            if self.fit_output is not None:
                output_ref += f'- LmFit output:\n{fit_report(self.fit_output)}\n'
            else:
                output_ref += f'- LmFit output not available\n'

        # Show the result
        print(output_ref)

        # Display plot
        if show_plot:
            self.plot_fit_components(self.fit_output, log_scale=log_scale, frame=frame)

        return

    @staticmethod
    def save_lineslog(linesDF, file_address):

        with open(file_address, 'wb') as output_file:
            string_DF = linesDF.to_string()
            output_file.write(string_DF.encode('UTF-8'))

        return

    # def plot_spectrum(self, continuumFlux=None, obsLinesTable=None, matchedLinesDF=None, noise_region=None,
    #                   log_scale=False, plotConf={}, axConf={}, specLabel='Observed spectrum', output_address=None,
    #                   dark_mode=True):
    #
    #     # Plot Configuration
    #     if dark_mode:
    #         defaultConf = DARK_PLOT.copy()
    #         foreground = defaultConf['text.color']
    #     else:
    #         defaultConf = STANDARD_PLOT.copy()
    #         foreground = 'tab:blue'
    #     defaultConf.update(plotConf)
    #     rcParams.update(defaultConf)
    #     fig, ax = plt.subplots()
    #
    #     # Plot the spectrum # TODO implement better switch between white and black themes
    #     ax.step(self.wave_rest, self.flux, label=specLabel, color=foreground)
    #
    #     # Plot the continuum if available
    #     if continuumFlux is not None:
    #         ax.step(self.wave_rest, continuumFlux, label='Sigma Continuum', linestyle=':')
    #
    #     # Plot astropy detected lines if available
    #     if obsLinesTable is not None:
    #         idcs_emission = obsLinesTable['line_type'] == 'emission'
    #         idcs_linePeaks = np.array(obsLinesTable[idcs_emission]['line_center_index'])
    #         ax.scatter(self.wave_rest[idcs_linePeaks], self.flux[idcs_linePeaks], label='Detected lines', facecolors='none',
    #                    edgecolors='tab:purple')
    #
    #     if matchedLinesDF is not None:
    #         idcs_foundLines = (matchedLinesDF.observation.isin(('detected', 'not identified'))) & \
    #                           (matchedLinesDF.wavelength >= self.wave_rest[0]) & \
    #                           (matchedLinesDF.wavelength <= self.wave_rest[-1])
    #         if 'latexLabel' in matchedLinesDF:
    #             lineLatexLabel = matchedLinesDF.loc[idcs_foundLines].latexLabel.values
    #         else:
    #             lineLatexLabel = matchedLinesDF.loc[idcs_foundLines].index.values
    #         lineWave = matchedLinesDF.loc[idcs_foundLines].wavelength.values
    #         w3, w4 = matchedLinesDF.loc[idcs_foundLines].w3.values, matchedLinesDF.loc[idcs_foundLines].w4.values
    #         observation = matchedLinesDF.loc[idcs_foundLines].observation.values
    #
    #         for i in np.arange(lineLatexLabel.size):
    #             if observation[i] == 'detected':
    #                 color_area = 'tab:red' if observation[i] == 'not identified' else 'tab:green'
    #                 ax.axvspan(w3[i], w4[i], alpha=0.25, color=color_area)
    #                 ax.text(lineWave[i], 0, lineLatexLabel[i], rotation=270)
    #
    #     if noise_region is not None:
    #         ax.axvspan(noise_region[0], noise_region[1], alpha=0.15, color='tab:cyan', label='Noise region')
    #
    #     if log_scale:
    #         ax.set_yscale('log')
    #
    #     if self.normFlux != 1:
    #         if 'ylabel' not in axConf:
    #             y_label = STANDARD_AXES['ylabel']
    #             if self.normFlux != 1.0:
    #                 norm_label = y_label + r' $\,/\,{}$'.format(latex_science_float(self.normFlux))
    #                 axConf['ylabel'] = norm_label
    #
    #     ax.update({**STANDARD_AXES, **axConf})
    #     ax.legend()
    #
    #     if output_address is None:
    #         plt.tight_layout()
    #         plt.show()
    #     else:
    #         plt.savefig(output_address, bbox_inches='tight')
    #
    #     plt.close(fig)
    #
    #
    #     return
    #
    # def plot_fit_components(self, lmfit_output=None, line_label=None, fig_conf={}, ax_conf={}, output_address=None,
    #                               log_scale=False, frame='rest', dark_mode=True):
    #
    #     # Determine line Label:
    #     # TODO this function should read from lines log
    #     # TODO this causes issues if vary is false... need a better way to get label
    #     line_label = line_label if line_label is not None else self.lineLabel
    #     ion, wave, latexLabel = label_decomposition(line_label, scalar_output=True)
    #
    #     # Plot Configuration
    #     if dark_mode:
    #         defaultConf = DARK_PLOT.copy()
    #         foreground = defaultConf['text.color']
    #         color_fit = 'yellow'
    #     else:
    #         defaultConf = STANDARD_PLOT.copy()
    #         foreground = defaultConf['text.color']
    #         color_fit = 'tab:blue'
    #
    #     # Plot Configuration
    #     defaultConf.update(fig_conf)
    #     rcParams.update(defaultConf)
    #
    #     defaultConf = STANDARD_AXES.copy()
    #     defaultConf.update(ax_conf)
    #
    #     # Case in which no emission line is introduced
    #     if lmfit_output is None:
    #         fig, ax = plt.subplots()
    #         ax = [ax]
    #     else:
    #         # fig, ax = plt.subplots(nrows=2)
    #         gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    #         spec_ax = plt.subplot(gs[0])
    #         grid_ax = plt.subplot(gs[1], sharex=spec_ax)
    #         ax = [spec_ax, grid_ax]
    #
    #     if frame == 'obs':
    #         z_cor = 1
    #         wave_plot = self.wave
    #         flux_plot = self.flux
    #     elif frame == 'rest':
    #         z_cor = 1 + self.redshift
    #         wave_plot = self.wave / z_cor
    #         flux_plot = self.flux * z_cor
    #     else:
    #         exit(f'-- Plot with frame name {frame} not recognize. Code will stop.')
    #
    #
    #     # Establish spectrum line and continua regions
    #     idcsEmis, idcsContBlue, idcsContRed = self.define_masks(self.wave_rest,
    #                                                             self.flux,
    #                                                             self.lineWaves,
    #                                                             merge_continua=False)
    #     idcs_plot = (wave_plot[idcsContBlue][0] - 5 <= wave_plot) & (wave_plot <= wave_plot[idcsContRed][-1] + 5)
    #
    #     # Plot line spectrum
    #     ax[0].step(wave_plot[idcs_plot], flux_plot[idcs_plot], label=r'Observed spectrum: {}'.format(latexLabel),
    #                where='mid', color=foreground)
    #     ax[0].scatter(self.peak_wave/z_cor, self.peak_flux*z_cor, color='tab:blue', alpha=0.7)
    #
    #     # Plot selection regions
    #     ax[0].fill_between(wave_plot[idcsContBlue], 0, flux_plot[idcsContBlue], facecolor='tab:orange', step='mid', alpha=0.07)
    #     ax[0].fill_between(wave_plot[idcsEmis], 0, flux_plot[idcsEmis], facecolor='tab:green', step='mid', alpha=0.07)
    #     ax[0].fill_between(wave_plot[idcsContRed], 0, flux_plot[idcsContRed], facecolor='tab:orange', step='mid', alpha=0.07)
    #
    #     # Axes formatting
    #     if self.normFlux != 1.0:
    #         defaultConf['ylabel'] = defaultConf['ylabel'] + " $\\times{{{0:.2g}}}$".format(self.normFlux)
    #
    #     if log_scale:
    #         ax[0].set_yscale('log')
    #
    #
    #     # Plot the Gaussian fit if available
    #     if lmfit_output is not None:
    #
    #         # Recover values from fit
    #         x_in, y_in = lmfit_output.userkws['x'], lmfit_output.data
    #
    #         # Resample gaussians
    #         wave_resample = np.linspace(x_in[0], x_in[-1], 200)
    #         flux_resample = lmfit_output.eval_components(x=wave_resample)
    #
    #         # Plot input data
    #         # ax[0].scatter(x_in/z_cor, y_in*z_cor, color='tab:red', label='Input data', alpha=0.4)
    #         ax[0].plot(x_in/z_cor, lmfit_output.best_fit*z_cor, label='Gaussian fit', color=color_fit, linewidth=0.7)
    #
    #         # Plot individual components
    #         if not self.blended_check:
    #             contLabel = f'{line_label}_cont_'
    #         else:
    #             contLabel = f'{self.blended_label.split("-")[0]}_cont_'
    #
    #         cont_flux = flux_resample.get(contLabel, 0.0)
    #         for comp_label, comp_flux in flux_resample.items():
    #             comp_flux = comp_flux + cont_flux if comp_label != contLabel else comp_flux
    #             ax[0].plot(wave_resample/z_cor, comp_flux*z_cor, label=f'{comp_label}', linestyle='--')
    #
    #         # Continuum residual plot:
    #         residual = (y_in - lmfit_output.best_fit)/self.cont
    #         ax[1].step(x_in/z_cor, residual*z_cor, where='mid', color=foreground)
    #
    #         # Err residual plot if available:
    #         if self.errFlux is not None:
    #             label = r'$\sigma_{Error}/\overline{F(cont)}$'
    #             err_norm = np.sqrt(self.errFlux[idcs_plot])/self.cont
    #             ax[1].fill_between(wave_plot[idcs_plot], -err_norm*z_cor, err_norm*z_cor, facecolor='tab:red', alpha=0.5, label=label)
    #
    #         label = r'$\sigma_{Continuum}/\overline{F(cont)}$'
    #         y_low, y_high = -self.std_cont / self.cont, self.std_cont / self.cont
    #         ax[1].fill_between(x_in/z_cor, y_low*z_cor, y_high*z_cor, facecolor='yellow', alpha=0.2, label=label)
    #
    #         # Residual plot labeling
    #         ax[1].set_xlim(ax[0].get_xlim())
    #         ax[1].set_ylim(2*residual.min(), 2*residual.max())
    #         ax[1].legend(loc='upper left')
    #         ax[1].set_ylabel(r'$\frac{F_{obs}}{F_{fit}} - 1$')
    #         ax[1].set_xlabel(r'Wavelength $(\AA)$')
    #
    #     ax[0].legend()
    #     ax[0].update(defaultConf)
    #
    #     if output_address is None:
    #         plt.tight_layout()
    #         plt.show()
    #     else:
    #         plt.savefig(output_address, bbox_inches='tight')
    #
    #     return
    #
    # def plot_line_velocity(self, lmfit_output=None, line_label=None, fig_conf={}, ax_conf={}, output_address=None,
    #                               log_scale=False, frame='rest', plot_title='', dark_mode=True):
    #
    #     # Determine line Label:
    #     # TODO this function should read from lines log
    #     # TODO this causes issues if vary is false... need a better way to get label
    #     line_label = line_label if line_label is not None else self.lineLabel
    #     ion, wave, latexLabel = label_decomposition(line_label, scalar_output=True)
    #
    #     # Plot Configuration
    #     if dark_mode:
    #         defaultConf = DARK_PLOT.copy()
    #         foreground = defaultConf['text.color']
    #         background = defaultConf['figure.facecolor']
    #         color_fit = 'yellow'
    #     else:
    #         defaultConf = STANDARD_PLOT.copy()
    #         foreground = 'black'
    #         background = 'white'
    #         color_fit = 'tab:blue'
    #
    #     # Plot Configuration
    #     defaultConf.update(fig_conf)
    #     rcParams.update(defaultConf)
    #
    #     defaultConf = STANDARD_AXES.copy()
    #     defaultConf.update(ax_conf)
    #
    #     # Establish spectrum line and continua regions
    #     idcsEmis, idcsContBlue, idcsContRed = self.define_masks(self.wave_rest,
    #                                                             self.flux,
    #                                                             self.lineWaves,
    #                                                             merge_continua=False)
    #
    #     z_cor = 1
    #     vel_plot = c_KMpS * (self.wave[idcsEmis]-self.peak_wave) / self.peak_wave
    #     flux_plot = self.flux[idcsEmis]
    #     cont_plot = self.m_cont * self.wave[idcsEmis] + self.n_cont
    #
    #     # Velocity values
    #     vel_med = np.median(vel_plot)
    #
    #     target_percentiles = np.array([2, 5, 10, 50, 90, 95, 98])
    #     percentile_array = np.cumsum(flux_plot-cont_plot) * self.pixelWidth / self.intg_flux * 100
    #     percentInterp = interp1d(percentile_array, vel_plot, kind='slinear')
    #     vel_percentiles = percentInterp(target_percentiles)
    #
    #
    #     # percentile_array = np.zeros(vel_plot.size)
    #     # target_percentiles = np.array([2, 5, 10, 50, 90, 95, 98])
    #     # for i_pix in np.arange(vel_plot.size):
    #     #     i_flux = (flux_plot[:i_pix].sum() - cont_plot[:i_pix].sum()) * self.pixelWidth
    #     #     percentile_array[i_pix] = i_flux / self.intg_flux * 100
    #     # Interpolation = interp1d(percentile_array, vel_plot, kind='slinear')
    #     # vel_percentiles = Interpolation(target_percentiles)
    #
    #     # Plot the data
    #     fig, ax = plt.subplots()
    #     ax = [ax]
    #     trans = ax[0].get_xaxis_transform()
    #
    #     # Plot line spectrum
    #     ax[0].step(vel_plot, flux_plot, label=latexLabel, where='mid', color=foreground)
    #
    #     for i_percentil, percentil in enumerate(target_percentiles):
    #         label_plot = r'$v_{{{}}}$'.format(percentil)
    #         label_text = None if i_percentil > 0 else r'$v_{Pth}$'
    #         ax[0].axvline(x=vel_percentiles[i_percentil], label=label_text, color=foreground, linestyle='dotted', alpha=0.5)
    #         ax[0].text(vel_percentiles[i_percentil], 0.80, label_plot, ha='center', va='center',
    #                 rotation='vertical', backgroundcolor=background, transform=trans, alpha=0.5)
    #
    #     ax[0].plot(vel_plot, cont_plot, linestyle='--')
    #
    #     w80 = vel_percentiles[4]-vel_percentiles[2]
    #     label_arrow = r'$w_{{80}}={:0.1f}\,Km/s$'.format(w80)
    #     p1 = patches.FancyArrowPatch((vel_percentiles[2], 0.5),
    #                                  (vel_percentiles[4], 0.5),
    #                                  label=label_arrow,
    #                                  arrowstyle='<->',
    #                                  color='tab:blue',
    #                                  transform=trans,
    #                                  mutation_scale=20)
    #     ax[0].add_patch(p1)
    #
    #     label_vmed = r'$v_{{med}}={:0.1f}\,Km/s$'.format(vel_med)
    #     ax[0].axvline(x=vel_med, color=foreground, label=label_vmed, linestyle='dashed', alpha=0.5)
    #
    #     label_vmed = r'$v_{{peak}}$'
    #     ax[0].axvline(x=0.0, color=foreground, label=label_vmed, alpha=0.5)
    #
    #
    #     # Axes formatting
    #     defaultConf['xlabel'] = 'Velocity (Km/s)'
    #     if self.normFlux != 1.0:
    #         defaultConf['ylabel'] = defaultConf['ylabel'] + " $/{{{0:.2g}}}$".format(self.normFlux)
    #
    #     defaultConf['title'] = plot_title
    #     if log_scale:
    #         ax[0].set_yscale('log')
    #
    #     ax[0].legend()
    #     ax[0].update(defaultConf)
    #
    #     if output_address is None:
    #         plt.tight_layout()
    #         plt.show()
    #     else:
    #         plt.savefig(output_address, bbox_inches='tight')
    #
    #     return w80
    #
    # def plot_line_grid(self, linesDF, plotConf={}, ncols=10, nrows=None, output_address=None, log_scale=True, frame='rest'):
    #
    #     # Line labels to plot
    #     lineLabels = linesDF.index.values
    #
    #     # Define plot axes grid size
    #     if nrows is None:
    #         nrows = int(np.ceil(lineLabels.size / ncols))
    #     if 'figure.figsize' not in plotConf:
    #         nrows = int(np.ceil(lineLabels.size / ncols))
    #         plotConf['figure.figsize'] = (ncols * 3, nrows * 3)
    #     n_axes, n_lines = ncols * nrows, lineLabels.size
    #
    #     if frame == 'obs':
    #         z_cor = 1
    #         wave_plot = self.wave
    #         flux_plot = self.flux
    #     elif frame == 'rest':
    #         z_cor = 1 + self.redshift
    #         wave_plot = self.wave / z_cor
    #         flux_plot = self.flux * z_cor
    #     else:
    #         exit(f'-- Plot with frame name {frame} not recognize. Code will stop.')
    #
    #     # Figure configuration
    #     defaultConf = STANDARD_PLOT.copy()
    #     defaultConf.update(plotConf)
    #     rcParams.update(defaultConf)
    #
    #     fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
    #     axesList = ax.flatten()
    #
    #     # Loop through the lines
    #     for i in np.arange(n_axes):
    #         if i < n_lines:
    #
    #             # Line data
    #             lineLabel = lineLabels[i]
    #             lineWaves = linesDF.loc[lineLabel, 'w1':'w6'].values
    #             latexLabel = linesDF.loc[lineLabel, 'latexLabel']
    #
    #
    #             # Establish spectrum line and continua regions
    #             idcsEmis, idcsContBlue, idcsContRed = self.define_masks(self.wave_rest,
    #                                                                     self.flux,
    #                                                                     lineWaves,
    #                                                                     merge_continua=False)
    #             idcs_plot = (wave_plot[idcsContBlue][0] - 5 <= wave_plot) & (
    #                         wave_plot <= wave_plot[idcsContRed][-1] + 5)
    #
    #             # Plot observation
    #             ax_i = axesList[i]
    #             ax_i.step(wave_plot[idcs_plot], flux_plot[idcs_plot], where='mid')
    #             ax_i.fill_between(wave_plot[idcsContBlue], 0, flux_plot[idcsContBlue], facecolor='tab:orange', step="mid", alpha=0.2)
    #             ax_i.fill_between(wave_plot[idcsEmis], 0, flux_plot[idcsEmis], facecolor='tab:blue', step="mid", alpha=0.2)
    #             ax_i.fill_between(wave_plot[idcsContRed], 0, flux_plot[idcsContRed], facecolor='tab:orange', step="mid", alpha=0.2)
    #
    #             if set(['m_cont', 'n_cont', 'amp', 'center', 'sigma']).issubset(linesDF.columns):
    #
    #                 line_params = linesDF.loc[lineLabel, ['m_cont', 'n_cont']].values
    #                 gaus_params = linesDF.loc[lineLabel, ['amp', 'center', 'sigma']].values
    #
    #                 # Plot curve fitting
    #                 if (not pd.isnull(line_params).any()) and (not pd.isnull(gaus_params).any()):
    #
    #                     wave_resample = np.linspace(self.wave[idcs_plot][0], self.wave[idcs_plot][-1], 500)
    #
    #                     m_cont, n_cont = line_params /self.normFlux
    #                     line_resample = linear_model(wave_resample, m_cont, n_cont)
    #
    #                     amp, mu, sigma = gaus_params
    #                     amp = amp/self.normFlux
    #                     gauss_resample = gaussian_model(wave_resample, amp, mu, sigma) + line_resample
    #                     ax_i.plot(wave_resample/z_cor, gauss_resample*z_cor, '--', color='tab:purple', linewidth=1.50)
    #
    #                 else:
    #                     for child in ax_i.get_children():
    #                         if isinstance(child, spines.Spine):
    #                             child.set_color('tab:red')
    #
    #             # Axis format
    #             ax_i.yaxis.set_major_locator(plt.NullLocator())
    #             ax_i.yaxis.set_ticklabels([])
    #             ax_i.xaxis.set_major_locator(plt.NullLocator())
    #             ax_i.axes.yaxis.set_visible(False)
    #             ax_i.set_title(latexLabel)
    #
    #             if log_scale:
    #                 ax_i.set_yscale('log')
    #
    #         # Clear not filled axes
    #         else:
    #             fig.delaxes(axesList[i])
    #
    #     if output_address is None:
    #         plt.tight_layout()
    #         plt.show()
    #     else:
    #         plt.savefig(output_address, bbox_inches='tight')
    #
    #     plt.close(fig)
    #
    #     return
    #
    # def table_fluxes(self, lines_df, table_address, pyneb_rc=None, scaleTable=1000):
    #
    #     # TODO this could be included in sr.print
    #     tex_address = f'{table_address}'
    #     txt_address = f'{table_address}.txt'
    #
    #     # Measure line fluxes
    #     pdf = PdfMaker()
    #     pdf.create_pdfDoc(pdf_type='table')
    #     pdf.pdf_insert_table(FLUX_TEX_TABLE_HEADERS)
    #
    #     # Dataframe as container as a txt file
    #     tableDF = pd.DataFrame(columns=FLUX_TXT_TABLE_HEADERS[1:])
    #
    #     # Normalization line
    #     if 'H1_4861A' in lines_df.index:
    #         flux_Hbeta = lines_df.loc['H1_4861A', 'intg_flux']
    #     else:
    #         flux_Hbeta = scaleTable
    #
    #     obsLines = lines_df.index.values
    #     for lineLabel in obsLines:
    #
    #         label_entry = lines_df.loc[lineLabel, 'latexLabel']
    #         wavelength = lines_df.loc[lineLabel, 'wavelength']
    #         eqw, eqwErr = lines_df.loc[lineLabel, 'eqw'], lines_df.loc[lineLabel, 'eqw_err']
    #
    #         flux_intg = lines_df.loc[lineLabel, 'intg_flux'] / flux_Hbeta * scaleTable
    #         flux_intgErr = lines_df.loc[lineLabel, 'intg_err'] / flux_Hbeta * scaleTable
    #         flux_gauss = lines_df.loc[lineLabel, 'gauss_flux'] / flux_Hbeta * scaleTable
    #         flux_gaussErr = lines_df.loc[lineLabel, 'gauss_err'] / flux_Hbeta * scaleTable
    #
    #         if (lines_df.loc[lineLabel, 'blended_label'] != 'None') and ('_m' not in lineLabel):
    #             flux, fluxErr = flux_gauss, flux_gaussErr
    #             label_entry = label_entry + '$_{gauss}$'
    #         else:
    #             flux, fluxErr = flux_intg, flux_intgErr
    #
    #         # Correct the flux
    #         if pyneb_rc is not None:
    #             corr = pyneb_rc.getCorrHb(wavelength)
    #             intensity, intensityErr = flux * corr, fluxErr * corr
    #             intensity_entry = r'${:0.2f}\,\pm\,{:0.2f}$'.format(intensity, intensityErr)
    #         else:
    #             intensity, intensityErr = '-', '-'
    #             intensity_entry = '-'
    #
    #         eqw_entry = r'${:0.2f}\,\pm\,{:0.2f}$'.format(eqw, eqwErr)
    #         flux_entry = r'${:0.2f}\,\pm\,{:0.2f}$'.format(flux, fluxErr)
    #
    #         # Add row of data
    #         tex_row_i = [label_entry, eqw_entry, flux_entry, intensity_entry]
    #         txt_row_i = [label_entry, eqw, eqwErr, flux, fluxErr, intensity, intensityErr]
    #
    #         lastRow_check = True if lineLabel == obsLines[-1] else False
    #         pdf.addTableRow(tex_row_i, last_row=lastRow_check)
    #         tableDF.loc[lineLabel] = txt_row_i[1:]
    #
    #     if pyneb_rc is not None:
    #
    #         # Data last rows
    #         row_Hbetaflux = [r'$H\beta$ $(erg\,cm^{-2} s^{-1} \AA^{-1})$',
    #                          '',
    #                          flux_Hbeta,
    #                          flux_Hbeta * pyneb_rc.getCorr(4861)]
    #
    #         row_cHbeta = [r'$c(H\beta)$',
    #                       '',
    #                       float(pyneb_rc.cHbeta),
    #                       '']
    #     else:
    #         # Data last rows
    #         row_Hbetaflux = [r'$H\beta$ $(erg\,cm^{-2} s^{-1} \AA^{-1})$',
    #                          '',
    #                          flux_Hbeta,
    #                          '-']
    #
    #         row_cHbeta = [r'$c(H\beta)$',
    #                       '',
    #                       '-',
    #                       '']
    #
    #     pdf.addTableRow(row_Hbetaflux, last_row=False)
    #     pdf.addTableRow(row_cHbeta, last_row=False)
    #     tableDF.loc[row_Hbetaflux[0]] = row_Hbetaflux[1:] + [''] * 3
    #     tableDF.loc[row_cHbeta[0]] = row_cHbeta[1:] + [''] * 3
    #
    #     # Format last rows
    #     pdf.table.add_hline()
    #     pdf.table.add_hline()
    #
    #     # Save the pdf table
    #     try:
    #         pdf.generate_pdf(table_address, clean_tex=True)
    #     except:
    #         print('-- PDF compilation failure')
    #
    #     # Save the txt table
    #     with open(txt_address, 'wb') as output_file:
    #         string_DF = tableDF.to_string()
    #         string_DF = string_DF.replace('$', '')
    #         output_file.write(string_DF.encode('UTF-8'))
    #
    #     return
    #
    # def table_kinematics(self, lines_df, table_address, flux_normalization=1.0):
    #
    #     # TODO this could be included in sr.print
    #     tex_address = f'{table_address}'
    #     txt_address = f'{table_address}.txt'
    #
    #     # Measure line fluxes
    #     pdf = PdfMaker()
    #     pdf.create_pdfDoc(pdf_type='table')
    #     pdf.pdf_insert_table(KIN_TEX_TABLE_HEADERS)
    #
    #     # Dataframe as container as a txt file
    #     tableDF = pd.DataFrame(columns=KIN_TXT_TABLE_HEADERS[1:])
    #
    #     obsLines = lines_df.index.values
    #     for lineLabel in obsLines:
    #
    #         if not lineLabel.endswith('_b'):
    #             label_entry = lines_df.loc[lineLabel, 'latexLabel']
    #
    #             # Establish component:
    #             blended_check = (lines_df.loc[lineLabel, 'blended_label'] != 'None') and ('_m' not in lineLabel)
    #             if blended_check:
    #                 blended_group = lines_df.loc[lineLabel, 'blended_label']
    #                 comp = 'n1' if lineLabel.count('_') == 1 else lineLabel[lineLabel.rfind('_')+1:]
    #             else:
    #                 comp = 'n1'
    #             comp_label, lineEmisLabel = kinematic_component_labelling(label_entry, comp)
    #
    #             wavelength = lines_df.loc[lineLabel, 'wavelength']
    #             v_r, v_r_err =  lines_df.loc[lineLabel, 'v_r':'v_r_err']
    #             sigma_vel, sigma_vel_err = lines_df.loc[lineLabel, 'sigma_vel':'sigma_vel_err']
    #
    #             flux_intg = lines_df.loc[lineLabel, 'intg_flux']
    #             flux_intgErr = lines_df.loc[lineLabel, 'intg_err']
    #             flux_gauss = lines_df.loc[lineLabel, 'gauss_flux']
    #             flux_gaussErr = lines_df.loc[lineLabel, 'gauss_err']
    #
    #             # Format the entries
    #             vr_entry = r'${:0.1f}\,\pm\,{:0.1f}$'.format(v_r, v_r_err)
    #             sigma_entry = r'${:0.1f}\,\pm\,{:0.1f}$'.format(sigma_vel, sigma_vel_err)
    #
    #             if blended_check:
    #                 flux, fluxErr = flux_gauss, flux_gaussErr
    #                 label_entry = lineEmisLabel
    #             else:
    #                 flux, fluxErr = flux_intg, flux_intgErr
    #
    #             # Correct the flux
    #             flux_entry = r'${:0.2f}\,\pm\,{:0.2f}$'.format(flux, fluxErr)
    #
    #             # Add row of data
    #             tex_row_i = [label_entry, comp_label, vr_entry, sigma_entry, flux_entry]
    #             txt_row_i = [lineLabel, comp_label.replace(' ', '_'), v_r, v_r_err, sigma_vel, sigma_vel_err, flux, fluxErr]
    #
    #             lastRow_check = True if lineLabel == obsLines[-1] else False
    #             pdf.addTableRow(tex_row_i, last_row=lastRow_check)
    #             tableDF.loc[lineLabel] = txt_row_i[1:]
    #
    #     pdf.table.add_hline()
    #
    #     # Save the pdf table
    #     try:
    #         pdf.generate_pdf(tex_address)
    #     except:
    #         print('-- PDF compilation failure')
    #
    #     # Save the txt table
    #     with open(txt_address, 'wb') as output_file:
    #         string_DF = tableDF.to_string()
    #         string_DF = string_DF.replace('$', '')
    #         output_file.write(string_DF.encode('UTF-8'))
    #
    #     return

    def clear_fit(self):
        super().__init__()

