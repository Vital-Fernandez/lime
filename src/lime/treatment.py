import numpy as np
import pandas as pd
from pathlib import Path
from lmfit import fit_report as lmfit_fit_report
from sys import exit
from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS

from .model import EmissionFitting
from .tools import label_decomposition, LineFinder
from .plots import LiMePlots, STANDARD_PLOT, STANDARD_AXES
from .io import _LOG_EXPORT, LOG_COLUMNS, load_lines_log, save_line_log
from .model import gaussian_model

from matplotlib import pyplot as plt, rcParams, colors, cm, gridspec
from matplotlib.widgets import SpanSelector


class Spectrum(EmissionFitting, LiMePlots, LineFinder):

    """
    This class provides a set of tools to measure lines from the spectra of ionized gas. The user provides a spectrum
    with input arrays for the observation wavelength and flux.

    Optionally, the user can provide the sigma spectrum with the pixel uncertainty. This array must be in the same
    units as the ``input_flux``.

    It is recommended to provide the object redshift and a flux normalization. This guarantees the functionality of
    the class functions.

    Finally, the user can also provide a two value array with the same wavelength limits. This array must be in the
    same units and frame of reference as the ``.input_wave``.

    :param input_wave: wavelength array
    :type input_wave: numpy.array

    :param input_flux: flux array
    :type input_flux: numpy.array

    :param input_err: sigma array of the ``input_flux``
    :type input_err: numpy.array, optional

    :param redshift: spectrum redshift
    :type redshift: float, optional

    :param norm_flux: spectrum flux normalization
    :type norm_flux: float, optional

    :param crop_waves: wavelength array crop values
    :type norm_flux: float, optional
    """

    def __init__(self, input_wave=None, input_flux=None, input_err=None, redshift=0, norm_flux=1.0, crop_waves=None):

        # Load parent classes
        LineFinder.__init__(self)
        EmissionFitting.__init__(self)
        LiMePlots.__init__(self)

        # Class attributes
        self.wave = None
        self.wave_rest = None
        self.flux = None
        self.err_flux = None
        self.norm_flux = norm_flux
        self.redshift = redshift
        self.log = None

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
                self.flux = input_flux  # * (1 + self.redshift)
                if input_err is not None:
                    self.err_flux = input_err  # * (1 + self.redshift)

        # Normalize the spectrum
        if input_flux is not None:
            self.flux = self.flux / self.norm_flux
            if input_err is not None:
                self.err_flux = self.err_flux / self.norm_flux

        # Generate empty dataframe to store measurement use cwd as default storing folder
        self.log = pd.DataFrame(columns=LOG_COLUMNS.keys())

        return

    def fit_from_wavelengths(self, line, mask, user_cfg={}, fit_method='leastsq', emission=True, adjacent_cont=True):

        """

        This function fits a line given its line and spectral mask. The line notation consists in the transition
        ion and wavelength (with units) separated by an underscore, i.e. O3_5007A.

        The location mask consists in a 6 values array with the wavelength boundaries for the line location and two
        adjacent continua. These wavelengths must be sorted by increasing order and in the rest frame.

        The user can specify the properties of the fitting: Number of components and parameter boundaries. Please check
        the documentation for the complete details.

        The user can specify the minimization algorithm for the `LmFit library <https://lmfit.github.io/lmfit-py/fitting.html#lmfit.minimizer.Minimizer.minimize>`_.

        By default, the algorithm assumes an emission line. The user can set the parameter ``emission=False`` for an
        absorption.

        If the sigma spectrum was not provided, the fitting estimates the pixel uncertainty from the adjacent continua flux
        standard deviation assuming a linear profile. If the parameter ``adjacent_cont=True`` the adjacent continua is also
        use to calculate the continuum at the line location. Otherwise, only the line continuum is calculated only with the
        first and last pixel in the line band (the 3rd and 4th values in the ``line_wavelengths`` array)

        :param line: line line in the ``LiMe`` notation, i.e. H1_6563A_b
        :type line: string

        :param mask: 6 wavelengths spectral mask with the blue continuum, line and red continuum bands.
        :type mask: numpy.ndarray

        :param user_cfg: Dictionary with the fitting configuration.
        :type user_cfg: dict, optional

        :param algorithm: Minimizing algorithm for the LmFit library. The default method is ``leastsq``.
        :type algorithm: str, optional

        :param emission: Boolean check for the line type. The default is ``True`` for an emission line
        :type emission: bool, optional

        :param adjacent_cont: Boolean check for the line continuum calculation. The default value ``True`` includes the
                              adjacent continua array
        :type adjacent_cont: bool, optional

        """

        # For security previous measurement is cleared and a copy of the user configuration is used
        self.clear_fit()
        fit_conf = user_cfg.copy()

        # Estate the minimizing method for the fitting (this parameter is not restored to default by the self.clear_fit function)
        self._minimize_method = fit_method


        # Label the current measurement
        self.line = line
        self.lineWaves = mask

        # Global fit parameters
        self._emission_check = emission
        self._cont_from_adjacent = adjacent_cont

        # Check if the masks are within the range
        if not np.any((self.lineWaves < self.wave_rest[0]) | (self.lineWaves > self.wave_rest[-1])):

            # Establish spectrum line and continua regions
            idcsEmis, idcsCont = self.define_masks(self.wave_rest, self.flux, self.lineWaves)

            # Integrated line properties
            emisWave, emisFlux = self.wave[idcsEmis], self.flux[idcsEmis]
            contWave, contFlux = self.wave[idcsCont], self.flux[idcsCont]
            err_array = self.err_flux[idcsEmis] if self.err_flux is not None else None
            self.line_properties(emisWave, emisFlux, contWave, contFlux, err_array, bootstrap_size=1000)

            # Check if blended line
            if self.line in fit_conf:
                self.blended_label = fit_conf[self.line]
                if '_b' in self.line:
                    self.blended_check = True

            # Import kinematics if requested
            self.import_line_kinematics(fit_conf, z_cor=1 + self.redshift)

            # Gaussian fitting # TODO Add logic for very small lines
            idcsLine = idcsEmis + idcsCont
            x_array = self.wave[idcsLine]
            y_array = self.flux[idcsLine]
            w_array = 1.0 / self.err_flux[idcsLine] if self.err_flux is not None else np.full(x_array.size,
                                                                                              1.0 / self.std_cont)
            self.gauss_lmfit(self.line, x_array, y_array, w_array, fit_conf, self.log, z_obj=self.redshift)

            # Safe the results to log DF
            self.results_to_database(self.line, self.log, fit_conf)

        else:
            print(
                f'- {self.line} mask beyond spectrum limits (w_min = {self.wave_rest[0]:0.1f}, w_max = {self.wave_rest[-1]:0.1f}):')
            print(f' -- {self.lineWaves}')

        return

    def display_results(self, line=None, fit_report=False, plot=True, log_scale=True, frame='obs',
                        output_address=None):

        """

        This function plots or prints the fitting of a line. If no line specified, the values used are those from the
        last fitting.

        The ``fit_report=True`` includes the `the LmFit log <https://lmfit.github.io/lmfit-py/fitting.html#getting-and-printing-fit-reports>`_.

        If an ``output_address`` is provided, the outputs will be saved to a file instead of displayed on the terminal/window.
        The text file will have the address output file extension changed to .txt, while the plot will use the one provided
        by the user.

        :param line: line fitting to query. If none is provided, the one from the last fitting is considered
        :type line: str, optional

        :param fit_report: Summary of the fitting inputs and outputs. The default value is ``False``
        :type fit_report: bool, optional

        :param plot: Plot of the fitting inputs and outputs. The default value is ``True``
        :type plot: bool, optional

        :param log_scale: Check for the scale of the flux (vertical) axis in the plot. The default value is ``True``
        :type log_scale: bool, optional

        :param frame: Frame of reference for the plot. The default value is the observation frame ``frame='obs'``
        :type frame: str, optional

        :param output_address: Output address for the measurement report and/or plot. If provided the results will be stored
                               instead of displayed on screen.
        :type output_address: str, optional

        """

        # Check if the user provided an output file address
        if output_address is not None:
            output_path = Path(output_address)

        # Fitting report
        if fit_report:

            if line is None:
                if self.line is not None:
                    line = self.line
                    output_ref = (f'\nLine line: {line}\n'
                                  f'- Line mask: {self.lineWaves}\n'
                                  f'- Normalization flux: {self.norm_flux}\n'
                                  f'- Redshift: {self.redshift}\n'
                                  f'- Peak wavelength: {self.peak_wave:.2f}; peak intensity: {self.peak_flux:.2f}\n'
                                  f'- Cont. slope: {self.m_cont:.2e}; Cont. intercept: {self.n_cont:.2e}\n')

                    if self.blended_check:
                        mixtureComponents = np.array(self.blended_label.split('-'))
                    else:
                        mixtureComponents = np.array([line], ndmin=1)

                    output_ref += f'\n- {line} Intg flux: {self.intg_flux:.3f} +/- {self.intg_err:.3f}\n'

                    if mixtureComponents.size == 1:
                        output_ref += f'- {line} Eqw (intg): {self.eqw[0]:.2f} +/- {self.eqw_err[0]:.2f}\n'

                    for i, lineRef in enumerate(mixtureComponents):
                        output_ref += (f'\n- {lineRef} gaussian fitting:\n'
                                       f'-- Gauss flux: {self.gauss_flux[i]:.3f} +/- {self.gauss_err[i]:.3f}\n'
                                       # f'-- Amplitude: {self.amp[i]:.3f} +/- {self.amp_err[i]:.3f}\n'
                                       f'-- Center: {self.center[i]:.2f} +/- {self.center_err[i]:.2f}\n'
                                       f'-- Sigma (km/s): {self.sigma_vel[i]:.2f} +/- {self.sigma_vel_err[i]:.2f}\n')
                else:
                    output_ref = f'- No measurement performed\n'

            # Case with line input: search and show that measurement
            elif self.log is not None:
                if line in self.log.index:
                    output_ref = self.log.loc[line].to_string
                else:
                    output_ref = f'- WARNING: {line} not found in  lines table\n'
            else:
                output_ref = '- WARNING: Measurement lines log not defined\n'

            # Display the print lmfit report if available
            if fit_report:
                if self.fit_output is not None:
                    output_ref += f'\n- LmFit output:\n{lmfit_fit_report(self.fit_output)}\n'
                else:
                    output_ref += f'\n- LmFit output not available\n'

            # Display the report
            if output_address is None:
                print(output_ref)
            else:
                output_txt = f'{output_path.parent/output_path.stem}.txt'
                with open(output_txt, 'w+') as fh:
                    fh.write(output_ref)

        # Fitting plot
        if plot:

            if output_address is None:
                self.plot_fit_components(self.fit_output, log_scale=log_scale, frame=frame)
            else:
                self.plot_fit_components(self.fit_output, log_scale=log_scale, frame=frame, output_address=output_address)

        return

    def import_line_kinematics(self, user_conf, z_cor):

        # Check if imported kinematics come from blended component
        if self.blended_label != 'None':
            childs_list = self.blended_label.split('-')
        else:
            childs_list = np.array(self.line, ndmin=1)

        for child_label in childs_list:
            parent_label = user_conf.get(f'{child_label}_kinem')

            if parent_label is not None:

                # Case we want to copy from previous line and the data is not available
                if (parent_label not in self.log.index) and (not self.blended_check):
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
                            print(
                                f'-- WARNING: {param_label_child} overwritten by {parent_label} kinematics in configuration input')

                        # Case where parent and child are in blended group
                        if parent_label in childs_list:
                            param_label_parent = f'{parent_label}_{param_ext}'
                            param_expr_parent = f'{wtheo_child / wtheo_parent:0.8f}*{param_label_parent}'

                            user_conf[param_label_child] = {'expr': param_expr_parent}

                        # Case we want to copy from previously measured line
                        else:
                            mu_parent = self.log.loc[parent_label, ['center', 'center_err']].values
                            sigma_parent = self.log.loc[parent_label, ['sigma', 'sigma_err']].values

                            if param_ext == 'center':
                                param_value = wtheo_child / wtheo_parent * (mu_parent / z_cor)
                            else:
                                param_value = wtheo_child / wtheo_parent * sigma_parent

                            user_conf[param_label_child] = {'value': param_value[0], 'vary': False}
                            user_conf[f'{param_label_child}_err'] = param_value[1]

        return

    def results_to_database(self, lineLabel, linesDF, fit_conf, export_params=_LOG_EXPORT):

        # Recover line data
        if self.blended_check:
            line_components = self.blended_label.split('-')
        else:
            line_components = np.array([lineLabel], ndmin=1)

        ion, waveRef, latexLabel = label_decomposition(line_components, comp_dict=fit_conf)

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
                    param_value = param_value * self.norm_flux

                line_log[param] = param_value

            # Assign line series to dataframe
            linesDF.loc[line] = line_log

        return

    def clear_fit(self):
        super().__init__()
        return


class MaskInspector(Spectrum):

    def __init__(self, lines_log_address, log, input_wave=None, input_flux=None, input_err=None, redshift=0,
                 norm_flux=1.0, crop_waves=None, ncols=10, nrows=None):

        # TODO modified lines are changed in the output file. Need to update.

        # Output file address
        self.linesLogAddress = Path(lines_log_address)

        # Assign attributes to the parent class
        super().__init__(input_wave, input_flux, input_err, redshift, norm_flux, crop_waves)

        # DF is provided
        if log is not None:
            self.log = pd.DataFrame.copy(log)

        # DF is not provided
        else:

            # Lines log address is provided and we read the DF from it
            if Path(self.linesLogAddress).is_file():
                self.log = load_lines_log(self.linesLogAddress)

            # Lines log not provide code ends
            else:
                print(f'- ERROR: No lines log provided by the user nor can be found the lines log file at address:'
                      f' {lines_log_address}')
                exit()

        # Figure grid
        n_lines = len(self.log.index)
        if n_lines > ncols:
            if nrows is None:
                nrows = int(np.ceil(n_lines / ncols))
        else:
            ncols = n_lines
            nrows = 1

        defaultConf = STANDARD_PLOT.copy()
        plotConf = {'figure.figsize': (nrows * 2, 8)}
        defaultConf.update(plotConf)
        rcParams.update(defaultConf)

        self.fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
        self.ax = ax.flatten()
        self.in_ax = None
        self.dict_spanSelec = {}
        self.axConf = {}

        # Plot function
        self.plot_line_mask_selection(logscale='auto', grid_size=nrows * ncols)
        plt.gca().axes.yaxis.set_ticklabels([])

        try:
            manager = plt.get_current_fig_manager()
            manager.window.showMaximized()
        except:
            print('-- Window could not be maximized')

        plt.tight_layout()
        plt.show()
        plt.close(self.fig)

        return

    def plot_line_mask_selection(self, logscale='auto', grid_size=None):

        # Plot data
        lineLabels = self.log.index.values
        n_lines = lineLabels.size

        # Generate plot
        for i in np.arange(grid_size):
            if i < n_lines:
                self.lineWaves = self.log.loc[lineLabels[i], 'w1':'w6'].values
                self.plot_line_region_i(self.ax[i], lineLabels[i], logscale=logscale)
                self.dict_spanSelec[f'spanner_{i}'] = SpanSelector(self.ax[i],
                                                                   self.on_select,
                                                                   'horizontal',
                                                                   useblit=True,
                                                                   rectprops=dict(alpha=0.5, facecolor='tab:blue'))

            # Clear not filled axes
            else:
                self.fig.delaxes(self.ax[i])

        bpe = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        aee = self.fig.canvas.mpl_connect('axes_enter_event', self.on_enter_axes)

        return

    def plot_line_region_i(self, ax, lineLabel, limitPeak=5, logscale='auto'):

        # Plot line region:
        ion, lineWave, latexLabel = label_decomposition(lineLabel, scalar_output=True)

        # Decide type of plot
        non_nan = (~pd.isnull(self.lineWaves)).sum()

        # Incomplete selections
        if non_nan < 6:  # selections

            # Peak region
            idcsLinePeak = (lineWave - limitPeak <= self.wave_rest) & (self.wave_rest <= lineWave + limitPeak)
            wavePeak, fluxPeak = self.wave_rest[idcsLinePeak], self.flux[idcsLinePeak]

            # Plot region
            idcsLineArea = (lineWave - limitPeak * 2 <= self.wave_rest) & (
                        lineWave - limitPeak * 2 <= self.lineWaves[3])
            waveLine, fluxLine = self.wave_rest[idcsLineArea], self.flux[idcsLineArea]

            # Plot the line region
            ax.step(waveLine, fluxLine)

            # Fill the user selections
            if non_nan == 2:
                idx1, idx2 = np.searchsorted(self.wave_rest, self.lineWaves[0:2])
                ax.fill_between(self.wave_rest[idx1:idx2], 0.0, self.flux[idx1:idx2], facecolor='tab:green',
                                step='mid', alpha=0.5)

            if non_nan == 4:
                idx1, idx2, idx3, idx4 = np.searchsorted(self.wave_rest, self.lineWaves[0:4])
                ax.fill_between(self.wave_rest[idx1:idx2], 0.0, self.flux[idx1:idx2], facecolor='tab:green',
                                step='mid', alpha=0.5)
                ax.fill_between(self.wave_rest[idx3:idx4], 0.0, self.flux[idx3:idx4], facecolor='tab:green',
                                step='mid', alpha=0.5)

        # Complete selections
        else:

            # Get line regions
            idcsContLeft = (self.lineWaves[0] <= self.wave_rest) & (self.wave_rest <= self.lineWaves[1])
            idcsContRight = (self.lineWaves[4] <= self.wave_rest) & (self.wave_rest <= self.lineWaves[5])

            idcsLinePeak = (lineWave - limitPeak <= self.wave_rest) & (self.wave_rest <= lineWave + limitPeak)
            idcsLineArea = (self.lineWaves[2] <= self.wave_rest) & (self.wave_rest <= self.lineWaves[3])

            waveCentral, fluxCentral = self.wave_rest[idcsLineArea], self.flux[idcsLineArea]
            wavePeak, fluxPeak = self.wave_rest[idcsLinePeak], self.flux[idcsLinePeak]

            idcsLinePlot = (self.lineWaves[0] - 5 <= self.wave_rest) & (self.wave_rest <= self.lineWaves[5] + 5)
            waveLine, fluxLine = self.wave_rest[idcsLinePlot], self.flux[idcsLinePlot]

            # Plot the line
            ax.step(waveLine, fluxLine)

            # Fill the user selections
            ax.fill_between(waveCentral, 0, fluxCentral, step="pre", alpha=0.4)
            ax.fill_between(self.wave_rest[idcsContLeft], 0, self.flux[idcsContLeft], facecolor='tab:orange',
                            step="pre", alpha=0.2)
            ax.fill_between(self.wave_rest[idcsContRight], 0, self.flux[idcsContRight], facecolor='tab:orange',
                            step="pre", alpha=0.2)

        # Plot format
        ax.yaxis.set_major_locator(plt.NullLocator())
        ax.xaxis.set_major_locator(plt.NullLocator())

        ax.update({'title': lineLabel})
        ax.yaxis.set_ticklabels([])
        ax.axes.yaxis.set_visible(False)

        idxPeakFlux = np.argmax(fluxPeak)
        ax.set_ylim(ymin=np.min(fluxLine) / 5, ymax=fluxPeak[idxPeakFlux] * 1.25)

        if logscale == 'auto':
            if fluxPeak[idxPeakFlux] > 5 * np.median(fluxLine):
                ax.set_yscale('log')

        return

    def on_select(self, w_low, w_high):

        # Check we are not just clicking on the plot
        if w_low != w_high:

            # Count number of empty entries to determine next step
            non_nans = (~pd.isnull(self.lineWaves)).sum()

            # Case selecting 1/3 region
            if non_nans == 0:
                self.lineWaves[0] = w_low
                self.lineWaves[1] = w_high

            # Case selecting 2/3 region
            elif non_nans == 2:
                self.lineWaves[2] = w_low
                self.lineWaves[3] = w_high
                self.lineWaves = np.sort(self.lineWaves)

            # Case selecting 3/3 region
            elif non_nans == 4:
                self.lineWaves[4] = w_low
                self.lineWaves[5] = w_high
                self.lineWaves = np.sort(self.lineWaves)

            elif non_nans == 6:
                self.lineWaves = np.sort(self.lineWaves)

                # Caso que se corrija la region de la linea
                if w_low > self.lineWaves[1] and w_high < self.lineWaves[4]:
                    self.lineWaves[2] = w_low
                    self.lineWaves[3] = w_high

                # Caso que se corrija el continuum izquierdo
                elif w_low < self.lineWaves[2] and w_high < self.lineWaves[2]:
                    self.lineWaves[0] = w_low
                    self.lineWaves[1] = w_high

                # Caso que se corrija el continuum derecho
                elif w_low > self.lineWaves[3] and w_high > self.lineWaves[3]:
                    self.lineWaves[4] = w_low
                    self.lineWaves[5] = w_high

                # Case we want to select the complete region
                elif w_low < self.lineWaves[0] and w_high > self.lineWaves[5]:

                    # # Remove line from dataframe and save it
                    # self.remove_lines_df(self.current_df, self.Current_Label)
                    #
                    # # Save lines log df
                    # self.save_lineslog_dataframe(self.current_df, self.lineslog_df_address)

                    # Clear the selections
                    # self.lineWaves = np.array([np.nan] * 6)
                    print(f'\n-- The line {self.line} mask has been removed')

                else:
                    print('- WARNING: Unsucessful line selection:')
                    print(f'-- {self.line}: w_low: {w_low}, w_high: {w_high}')

            # Check number of measurements after selection
            non_nans = (~pd.isnull(self.lineWaves)).sum()

            # Proceed to re-measurement if possible:
            if non_nans == 6:
                # TODO add option to perform the measurement a new
                # self.clear_fit()
                # self.fit_from_wavelengths(self.line, self.lineWaves, user_cfg={})

                # Parse the line regions to the dataframe
                self.results_to_database(self.lineLabel, self.log, fit_conf={}, export_params=[])

                # Save the corrected mask to a file
                self.store_measurement()

            # Redraw the line measurement
            self.in_ax.clear()
            self.plot_line_region_i(self.in_ax, self.lineLabel, logscale='auto')
            self.in_fig.canvas.draw()

        return

    def on_enter_axes(self, event):

        # Assign new axis
        self.in_fig = event.canvas.figure
        self.in_ax = event.inaxes

        # TODO we need a better way to index than the latex line
        # Recognise line line
        idx_line = self.log.index == self.in_ax.get_title()
        self.lineLabel = self.log.loc[idx_line].index.values[0]
        self.lineWaves = self.log.loc[idx_line, 'w1':'w6'].values[0]

        # Restore measurements from log
        # self.database_to_attr()

        # event.inaxes.patch.set_edgecolor('red')
        # event.canvas.draw()

    def on_click(self, event):

        if event.dblclick:
            print(self.lineLabel)
            print(f'{event.button}, {event.x}, {event.y}, {event.xdata}, {event.ydata}')

    def store_measurement(self):

        # Read file in the stored address
        if self.linesLogAddress.is_file():
            file_DF = load_lines_log(self.linesLogAddress)

            # Add new line to the DF and sort it if it was new
            if self.lineLabel in file_DF.index:
                file_DF.loc[self.lineLabel, 'w1':'w6'] = self.lineWaves
            else:
                file_DF.loc[self.lineLabel, 'w1':'w6'] = self.lineWaves

                # Sort the lines by theoretical wavelength
                lineLabels = file_DF.index.values
                ion_array, wavelength_array, latexLabel_array = label_decomposition(lineLabels)
                file_DF = file_DF.iloc[wavelength_array.argsort()]

        # If the file does not exist (or it is the first time)
        else:
            file_DF = self.log

        # Save to a file
        save_line_log(file_DF, self.linesLogAddress)

        return


class CubeFitsInspector(Spectrum):

    def __init__(self, input_wave, input_cube_flux, image_bg, image_fg=None, contour_levels_fg=None,
                 min_bg_percentil=60,
                 redshift=0, norm_flux=1, lines_log_address=None, fits_header=None, fig_conf=None, axes_conf={}):

        # Assign attributes to the parent class
        super().__init__(input_wave, input_flux=None, redshift=redshift, norm_flux=norm_flux)

        self.fig = None
        self.ax0, self.ax1, self.in_ax = None, None, None
        self.grid_mesh = None
        self.cube_flux = input_cube_flux
        self.wave = input_wave
        self.header = fits_header
        self.image_bg = image_bg
        self.image_fg = image_fg
        self.contour_levels_fg = contour_levels_fg
        self.fig_conf = STANDARD_PLOT.copy()
        self.axes_conf = {}
        self.axlim_dict = {}
        self.min_bg_percentil = min_bg_percentil
        self.hdul_linelog = None

        # Read the figure configuration
        self.fig_conf = STANDARD_PLOT if fig_conf is None else fig_conf
        rcParams.update(self.fig_conf)

        # Read the axes format
        if 'image' in axes_conf:
            default_conf = {'xlabel': r'RA', 'ylabel': r'DEC', 'title': f'Cube flux slice'}
            default_conf.update(axes_conf['image'])
            self.axes_conf['image'] = default_conf
        else:
            self.axes_conf['image'] = {'xlabel': r'RA', 'ylabel': r'DEC', 'title': f'Cube flux slice'}

        if 'spectrum' in axes_conf:
            self.axes_conf['spectrum'] = STANDARD_AXES.update(axes_conf['spectrum'])
        else:
            self.axes_conf['spectrum'] = STANDARD_AXES

        # Figure structure
        self.fig = plt.figure(figsize=(18, 5))
        gs = gridspec.GridSpec(nrows=1, ncols=2, figure=self.fig, width_ratios=[1, 2], height_ratios=[1])
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('axes_enter_event', self.on_enter_axes)

        # Axes configuration
        if self.header is None:
            self.ax0 = self.fig.add_subplot(gs[0])
        else:
            sky_wcs = WCS(self.header)
            self.ax0 = self.fig.add_subplot(gs[0], projection=sky_wcs, slices=('x', 'y', 1))
        self.ax1 = self.fig.add_subplot(gs[1])

        # Image mesh grid
        frame_size = self.cube_flux.shape
        y, x = np.arange(0, frame_size[1]), np.arange(0, frame_size[2])
        self.grid_mesh = np.meshgrid(x, y)

        # Use central voxels as initial coordinate
        init_coord = int(self.cube_flux.shape[1] / 2), int(self.cube_flux.shape[2] / 2)

        # Load the complete fits lines log if input
        if lines_log_address is not None:
            self.hdul_linelog = fits.open(lines_log_address, lazy_load_hdus=False)

        # Generate the plot
        self.plot_map_voxel(self.image_bg, init_coord, self.image_fg, self.contour_levels_fg)
        plt.show()

        # Close the lins log if it has been opened
        if isinstance(self.hdul_linelog, fits.hdu.HDUList):
            self.hdul_linelog.close()

        return

    def plot_map_voxel(self, image_bg, voxel_coord=None, image_fg=None, flux_levels=None, frame='obs'):

        min_flux = np.nanpercentile(image_bg, self.min_bg_percentil)
        norm_color_bg = colors.SymLogNorm(linthresh=min_flux,
                                          vmin=min_flux,
                                          base=10)
        self.ax0.imshow(image_bg, cmap=cm.gray, norm=norm_color_bg)

        # Emphasize input coordinate
        idx_j, idx_i = voxel_coord
        if voxel_coord is not None:
            self.ax0.plot(idx_i, idx_j, '+', color='red')

        # Plot contours image
        if image_fg is not None:
            self.ax0.contour(self.grid_mesh[0], self.grid_mesh[1], image_fg, cmap='viridis', levels=flux_levels,
                             norm=colors.LogNorm())

        # Voxel spectrum
        if voxel_coord is not None:
            flux_voxel = self.cube_flux[:, idx_j, idx_i]
            self.ax1.step(self.wave, flux_voxel, where='mid')

        # Plot the emission line fittings:
        if self.hdul_linelog is not None:
            ext_name = f'{idx_j}-{idx_i}_LINELOG'

            if ext_name in self.hdul_linelog:
                lineslogDF = Table.read(self.hdul_linelog[ext_name]).to_pandas()
                lineslogDF.set_index('index', inplace=True)
                self.log = lineslogDF
            else:
                self.log = None

            if self.log is not None:

                if frame == 'rest':
                    z_corr = (1 + self.redshift)
                    flux_plot = self.flux * z_corr
                    wave_plot = self.wave_rest
                else:
                    z_corr = 1
                    flux_plot = self.flux
                    wave_plot = self.wave

                for lineLabel in self.log.index:

                    w3, w4 = self.log.loc[lineLabel, 'w3'], self.log.loc[lineLabel, 'w4']
                    m_cont, n_cont = self.log.loc[lineLabel, 'm_cont'], self.log.loc[lineLabel, 'n_cont']
                    amp, center, sigma = self.log.loc[lineLabel, 'amp'], self.log.loc[lineLabel, 'center'], \
                                         self.log.loc[lineLabel, 'sigma']
                    wave_peak, flux_peak = self.log.loc[lineLabel, 'peak_wave'], self.log.loc[
                        lineLabel, 'peak_flux']
                    blended_label = self.log.loc[lineLabel, 'blended_label']

                    # Rest frame
                    if frame == 'rest':
                        w3, w4 = w3 * (1 + self.redshift), w4 * (1 + self.redshift)
                        wave_range = np.linspace(w3, w4, int((w4 - w3) * 3))
                        cont = (m_cont * wave_range + n_cont)
                        wave_range = wave_range / (1 + self.redshift)
                        center = center / (1 + self.redshift)

                    # Observed frame
                    else:
                        w3, w4 = w3 * (1 + self.redshift), w4 * (1 + self.redshift)
                        wave_range = np.linspace(w3, w4, int((w4 - w3) * 4))
                        cont = (m_cont * wave_range + n_cont) * z_corr

                    line_profile = gaussian_model(wave_range, amp, center, sigma) * z_corr

                    if blended_label == 'None':
                        color_curve = 'tab:red'
                        style_curve = '-'
                        width_curve = 0.5
                        self.ax1.plot(wave_range, cont / self.norm_flux, ':', color='tab:purple', linewidth=0.5)

                    else:
                        style_curve = ':'
                        width_curve = 2
                        cmap = cm.get_cmap()
                        list_comps = blended_label.split('-')
                        idx_line = list_comps.index(lineLabel)
                        color_curve = cmap(idx_line / len(list_comps))

                    self.ax1.plot(wave_range, (line_profile + cont) / self.norm_flux, color=color_curve,
                                  linestyle=style_curve, linewidth=width_curve)

                # TODO we need a method just for this
                idcs_blended = self.log['blended_label'] != 'None'
                blended_groups = self.log.loc[idcs_blended, 'blended_label'].unique()
                for lineGroup in blended_groups:
                    list_comps = lineGroup.split('-')
                    for i, lineLabel in enumerate(list_comps):

                        w3, w4 = self.log.loc[lineLabel, 'w3'], self.log.loc[lineLabel, 'w4']
                        m_cont, n_cont = self.log.loc[lineLabel, 'm_cont'], self.log.loc[lineLabel, 'n_cont']
                        amp, center, sigma = self.log.loc[lineLabel, 'amp'], self.log.loc[lineLabel, 'center'], \
                                             self.log.loc[lineLabel, 'sigma']
                        wave_peak, flux_peak = self.log.loc[lineLabel, 'peak_wave'], self.log.loc[
                            lineLabel, 'peak_flux']
                        blended_label = self.log.loc[lineLabel, 'blended_label']

                        # Rest frame
                        if frame == 'rest':
                            w3, w4 = w3 * (1 + self.redshift), w4 * (1 + self.redshift)
                            if i == 0:
                                wave_range = np.linspace(w3, w4, int((w4 - w3) * 3))
                            cont = (m_cont * wave_range + n_cont)
                            wave_range = wave_range / (1 + self.redshift)
                            center = center / (1 + self.redshift)

                        # Observed frame
                        else:
                            w3, w4 = w3 * (1 + self.redshift), w4 * (1 + self.redshift)
                            if i == 0:
                                wave_range = np.linspace(w3, w4, int((w4 - w3) * 3))
                            cont = (m_cont * wave_range + n_cont) * z_corr

                        if i == 0:
                            line_profile = gaussian_model(wave_range, amp, center, sigma) * z_corr
                        else:
                            line_profile += gaussian_model(wave_range, amp, center, sigma) * z_corr

                    self.ax1.plot(wave_range, (line_profile + cont) / self.norm_flux, color='tab:red',
                                  linestyle='-', linewidth=0.5)

        self.axes_conf['spectrum']['title'] = f'Voxel {idx_j} - {idx_i}'

        # Update the axis
        self.ax0.update(self.axes_conf['image'])
        self.ax1.update(self.axes_conf['spectrum'])

        return

    def on_click(self, event, mouse_trigger_buttton=3):

        """
        This method defines launches the new plot selection once the user clicks on an image voxel. By default this is a
        a right click on a minimum three button mouse
        :param event: This variable represents the user action on the plot
        :param mouse_trigger_buttton: Number-coded mouse button which defines the button launching the voxel selection
        :return:
        """

        if self.in_ax == self.ax0:

            if event.button == mouse_trigger_buttton:
                # Save axes zoom
                self.save_zoom()

                # Save clicked coordinates for next plot
                idx_j, idx_i = np.rint(event.ydata).astype(int), np.rint(event.xdata).astype(int)
                print(f'Current voxel: {idx_j}-{idx_i} (mouse button {event.button})')

                # Remake the drawing
                self.ax0.clear()
                self.ax1.clear()
                self.plot_map_voxel(self.image_bg, (idx_j, idx_i), self.image_fg, self.contour_levels_fg)

                # Reset the image
                self.reset_zoom()
                self.fig.canvas.draw()

    def on_enter_axes(self, event):
        self.in_ax = event.inaxes

    def save_zoom(self):
        self.axlim_dict['image_xlim'] = self.ax0.get_xlim()
        self.axlim_dict['image_ylim'] = self.ax0.get_ylim()
        self.axlim_dict['spec_xlim'] = self.ax1.get_xlim()
        self.axlim_dict['spec_ylim'] = self.ax1.get_ylim()

    def reset_zoom(self):

        self.ax0.set_xlim(self.axlim_dict['image_xlim'])
        self.ax0.set_ylim(self.axlim_dict['image_ylim'])
        self.ax1.set_xlim(self.axlim_dict['spec_xlim'])
        self.ax1.set_ylim(self.axlim_dict['spec_ylim'])
