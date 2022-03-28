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
from .plots import LiMePlots, STANDARD_PLOT, STANDARD_AXES, colorDict
from .io import _LOG_DTYPES_REC, _LOG_EXPORT, _LOG_COLUMNS, load_lines_log, save_line_log
from .model import gaussian_profiles_computation, linear_continuum_computation

from matplotlib import pyplot as plt, rcParams, colors, cm, gridspec, rc_context
from matplotlib.widgets import SpanSelector
from matplotlib.widgets import RadioButtons

try:
    import mplcursors
    mplcursors_check = True
except ImportError:
    mplcursors_check = False

if mplcursors_check:
    from mplcursors._mplcursors import _default_annotation_kwargs as popupProps
    popupProps['bbox']['alpha'] = 0.9


class Spectrum(EmissionFitting, LiMePlots, LineFinder):

    """
    This class provides a set of tools to measure lines from the spectra of ionized gas. The user provides a spectrum
    with input arrays for the observation wavelength and flux.

    Optionally, the user can provide the sigma spectrum with the pixel uncertainty. This array must be in the same
    units as the ``input_flux``.

    It is recommended to provide the object redshift and a flux normalization. This guarantees the functionality of
    the class functions.

    Finally, the user can also provide a two value array with the same wavelength limits. This array must be in the
    same units and frame of reference as the ``.wave``.

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
    :type norm_flux: np.array, optional

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
        self.log = pd.DataFrame(np.empty(0, dtype=_LOG_DTYPES_REC))

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

        # Security checks for the mask wavelengths
        assert np.all(np.diff(mask) >= 0), f'\n- Error: the {line} mask is not sorted'
        assert self.wave_rest[0] < mask[0], f'\n- Error: the {line} mask low mask limit (w1 = {mask[0]:.2f}) is below the spectrum rest frame limit (w_min = {self.wave_rest[0]:.2f})'
        assert self.wave_rest[-1] > mask[-1], f'\n- Error: the {line} mask up mask limit (w6 = {mask[-1]:.2f}) is above the spectrum rest frame limit (w_min = {self.wave_rest[-1]:.2f})'

        # For security previous measurement is cleared and a copy of the user configuration is used
        self.clear_fit()
        fit_conf = user_cfg.copy()

        # Estate the minimizing method for the fitting (this parameter is not restored to default by the self.clear_fit function)
        self._minimize_method = fit_method

        # Label the current measurement
        self.line = line
        self.mask = mask

        # Global fit parameters
        self._emission_check = emission
        self._cont_from_adjacent = adjacent_cont

         # Establish spectrum line and continua regions
        idcsEmis, idcsCont = self.define_masks(self.wave_rest, self.flux, self.mask)

        # Integrated line properties
        emisWave, emisFlux = self.wave[idcsEmis], self.flux[idcsEmis]
        contWave, contFlux = self.wave[idcsCont], self.flux[idcsCont]
        err_array = self.err_flux[idcsEmis] if self.err_flux is not None else None

        # Store error very small mask
        if emisWave.size <= 1:
            if self.observations == 'no':
                self.observations = 'Small_line_band'
            else:
                self.observations += 'Small_line_band'
            print(f'-- WARNING: Line band mask is too small for line {line}')

        self.line_properties(emisWave, emisFlux, contWave, contFlux, err_array, bootstrap_size=1000)

        # Check if blended line
        if self.line in fit_conf:
            self.profile_label = fit_conf[self.line]
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

        return

    def display_results(self, line=None, fit_report=False, plot=True, log_scale=True, frame='observed',
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
                                  f'- Line mask: {self.mask}\n'
                                  f'- Normalization flux: {self.norm_flux}\n'
                                  f'- Redshift: {self.redshift}\n'
                                  f'- Peak wavelength: {self.peak_wave:.2f}; peak intensity: {self.peak_flux:.2f}\n'
                                  f'- Cont. slope: {self.m_cont:.2e}; Cont. intercept: {self.n_cont:.2e}\n')

                    if self.blended_check:
                        mixtureComponents = np.array(self.profile_label.split('-'))
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
                self.plot_fit_components(log_scale=log_scale, frame=frame)
            else:
                self.plot_fit_components(log_scale=log_scale, frame=frame, output_address=output_address)

        return

    def import_line_kinematics(self, user_conf, z_cor):

        # Check if imported kinematics come from blended component
        if self.profile_label != 'no':
            childs_list = self.profile_label.split('-')
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
            line_components = self.profile_label.split('-')
        else:
            line_components = np.array([lineLabel], ndmin=1)

        ion, waveRef, latexLabel = label_decomposition(line_components, comp_dict=fit_conf)

        # Loop through the line components
        for i, line in enumerate(line_components):

            # Convert current measurement to a pandas series container
            linesDF.loc[line, ['ion', 'wavelength', 'latex_label']] = ion[i], waveRef[i], latexLabel[i]
            linesDF.loc[line, 'w1':'w6'] = self.mask

            # line_log = pd.Series(index=LOG_COLUMNS.keys())
            # line_log['ion', 'wavelength', 'latex_label'] = ion[i], waveRef[i], latexLabel[i]
            # line_log['w1': 'w6'] = self.mask

            # Treat every line
            for param in export_params:

                # Get component parameter
                if _LOG_COLUMNS[param][2]:
                    param_value = self.__getattribute__(param)[i]
                else:
                    param_value = self.__getattribute__(param)

                # De normalize
                if _LOG_COLUMNS[param][0]:
                    param_value = param_value * self.norm_flux

                linesDF.loc[line, param] = param_value

        return

    def clear_fit(self):
        super().__init__()
        return


class MaskInspector(Spectrum):

    def __init__(self, log_address, input_wave=None, input_flux=None, input_err=None, redshift=0,
                 norm_flux=1.0, crop_waves=None, n_cols=10, n_rows=None, lines_interval=None):

        """
        This class plots the masks from the ``log_address`` as a grid for the input spectrum as a grid. Clicking and
        dragging the mouse within a line cell will update the line band region, both in the plot and the ``log_address``
        file provided.

        Assuming that the band wavelengths `w1` and `w2` specify the adjacent blue (left continuum), the `w3` and `w4`
        wavelengths specify the line band and the `w5` and `w6` wavelengths specify the adjacent red (right continuum)
        the interactive selection has the following rules:

        * The plot wavelength range is always 5 pixels beyond the mask bands. Therefore dragging the mouse beyond the
          mask limits (below `w1` or above `w6`) will change the displayed range. This can be used to move beyond the
          original mask limits.

        * Selections between the `w2` and `w5` wavelength bands are always assigned to the line region mask as the new
          `w3` and `w4` values.

        * Due to the previous point, to increase the `w2` value or to decrease `w5` value the user must select a region
          between `w1` and `w3` or `w4` and `w6` respectively.

        The user can limit the number of lines displayed on the screen using the ``lines_interval`` parameter. This
        parameter can be an array of strings with the labels of the target lines or a two value integer array with the
        interval of lines to plot.

        :param log_address: Address for the lines log mask file.
        :type log_address: str

        :param input_wave: Wavelength array of the input spectrum.
        :type input_wave: numpy.array

        :param input_flux: Flux array for the input spectrum.
        :type input_flux: numpy.array

        :param input_err: Sigma array of the `input_flux`
        :type input_err: numpy.array, optional

        :param redshift: Spectrum redshift
        :type redshift: float, optional

        :param norm_flux: Spectrum flux normalization
        :type norm_flux: float, optional

        :param crop_waves: Wavelength limits in a two value array
        :type crop_waves: np.array, optional

        :param n_cols: Number of columns of the grid plot
        :type n_cols: integer

        :param n_rows: Number of columns of the grid plot
        :type n_rows: integer

        :param lines_interval: List of lines or mask file line interval to display on the grid plot. In the later case
                               this interval must be a two value array.
        :type lines_interval: list
        """

        # Output file address
        self.linesLogAddress = Path(log_address)

        # Assign attributes to the parent class
        super().__init__(input_wave, input_flux, input_err, redshift, norm_flux, crop_waves)

        # Lines log address is provided and we read the DF from it
        if Path(self.linesLogAddress).is_file():
            self.log = load_lines_log(self.linesLogAddress)

        # Lines log not provide code ends
        else:
            print(f'- ERROR: No lines log provided by the user nor can be found the lines log file at address:'
                  f' {log_address}')
            exit()

        # Only plotting the lines in the lines interval
        self.line_inter = lines_interval
        if lines_interval is None:
            n_lines = len(self.log.index)
            self.target_lines = self.log.index.values

        else:
            # Array of strings
            if isinstance(lines_interval[0], str):
                n_lines = len(lines_interval)
                self.target_lines = np.array(lines_interval, ndmin=1)

            # Array of integers
            else:
                n_lines = lines_interval[1] - lines_interval[0]
                self.target_lines = self.log[lines_interval[0]:lines_interval[1]].index.values

        # Establish the grid shape
        if n_lines > n_cols:
            if n_rows is None:
                n_rows = int(np.ceil(n_lines / n_cols))
        else:
            n_cols = n_lines
            n_rows = 1

        # Adjust the plot theme
        PLOT_CONF = STANDARD_PLOT.copy()
        AXES_CONF = STANDARD_AXES.copy()
        PLOT_CONF['figure.figsize'] = (n_rows * 2, 8)
        AXES_CONF.pop('xlabel')

        with rc_context(PLOT_CONF):

            self.fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols)
            self.ax = ax.flatten()
            self.in_ax = None
            self.dict_spanSelec = {}
            self.axConf = {}

            # Plot function
            self.plot_line_mask_selection(logscale='auto', grid_size=n_rows * n_cols)
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
        # for i, line in enumerate(self.target_lines):
        for i in range(grid_size):
            if i < n_lines:
                line = self.target_lines[i]
                if line in self.log.index:
                    self.mask = self.log.loc[line, 'w1':'w6'].values
                    self.plot_line_region_i(self.ax[i], line, logscale=logscale)
                    self.dict_spanSelec[f'spanner_{i}'] = SpanSelector(self.ax[i],
                                                                       self.on_select,
                                                                       'horizontal',
                                                                       useblit=True,
                                                                       rectprops=dict(alpha=0.5, facecolor='tab:blue'))
                else:
                    print(f'- WARNING: line {line} not found in the input mask')

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
        non_nan = (~pd.isnull(self.mask)).sum()

        # Incomplete selections
        if non_nan < 6:  # selections

            # Peak region
            idcsLinePeak = (lineWave - limitPeak <= self.wave_rest) & (self.wave_rest <= lineWave + limitPeak)
            wavePeak, fluxPeak = self.wave_rest[idcsLinePeak], self.flux[idcsLinePeak]

            # Plot region
            idcsLineArea = (lineWave - limitPeak * 2 <= self.wave_rest) & (
                        lineWave - limitPeak * 2 <= self.mask[3])
            waveLine, fluxLine = self.wave_rest[idcsLineArea], self.flux[idcsLineArea]

            # Plot the line region
            ax.step(waveLine, fluxLine)

            # Fill the user selections
            if non_nan == 2:
                idx1, idx2 = np.searchsorted(self.wave_rest, self.mask[0:2])
                ax.fill_between(self.wave_rest[idx1:idx2], 0.0, self.flux[idx1:idx2], facecolor=self._color_dict['cont_band'],
                                step='mid', alpha=0.5)

            if non_nan == 4:
                idx1, idx2, idx3, idx4 = np.searchsorted(self.wave_rest, self.mask[0:4])
                ax.fill_between(self.wave_rest[idx1:idx2], 0.0, self.flux[idx1:idx2], facecolor=self._color_dict['cont_band'],
                                step='mid', alpha=0.5)
                ax.fill_between(self.wave_rest[idx3:idx4], 0.0, self.flux[idx3:idx4], facecolor=self._color_dict['cont_band'],
                                step='mid', alpha=0.5)

        # Complete selections
        else:

            # Get line regions
            idcsContLeft = (self.mask[0] <= self.wave_rest) & (self.wave_rest <= self.mask[1])
            idcsContRight = (self.mask[4] <= self.wave_rest) & (self.wave_rest <= self.mask[5])

            idcsLinePeak = (lineWave - limitPeak <= self.wave_rest) & (self.wave_rest <= lineWave + limitPeak)
            idcsLineArea = (self.mask[2] <= self.wave_rest) & (self.wave_rest <= self.mask[3])

            waveCentral, fluxCentral = self.wave_rest[idcsLineArea], self.flux[idcsLineArea]
            wavePeak, fluxPeak = self.wave_rest[idcsLinePeak], self.flux[idcsLinePeak]

            idcsLinePlot = (self.mask[0] - 5 <= self.wave_rest) & (self.wave_rest <= self.mask[5] + 5)
            waveLine, fluxLine = self.wave_rest[idcsLinePlot], self.flux[idcsLinePlot]

            # Plot the line
            ax.step(waveLine, fluxLine, color=self._color_dict['fg'], where='mid')

            # Fill the user selections
            ax.fill_between(waveCentral, 0, fluxCentral, step="mid", alpha=0.4, facecolor=self._color_dict['line_band'])
            ax.fill_between(self.wave_rest[idcsContLeft], 0, self.flux[idcsContLeft], facecolor=self._color_dict['cont_band'],
                            step="mid", alpha=0.2)
            ax.fill_between(self.wave_rest[idcsContRight], 0, self.flux[idcsContRight], facecolor=self._color_dict['cont_band'],
                            step="mid", alpha=0.2)

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
            non_nans = (~pd.isnull(self.mask)).sum()

            # Case selecting 1/3 region
            if non_nans == 0:
                self.mask[0] = w_low
                self.mask[1] = w_high

            # Case selecting 2/3 region
            elif non_nans == 2:
                self.mask[2] = w_low
                self.mask[3] = w_high
                self.mask = np.sort(self.mask)

            # Case selecting 3/3 region
            elif non_nans == 4:
                self.mask[4] = w_low
                self.mask[5] = w_high
                self.mask = np.sort(self.mask)

            elif non_nans == 6:
                self.mask = np.sort(self.mask)

                # Caso que se corrija la region de la linea
                if w_low > self.mask[1] and w_high < self.mask[4]:
                    self.mask[2] = w_low
                    self.mask[3] = w_high

                # Caso que se corrija el continuum izquierdo
                elif w_low < self.mask[2] and w_high < self.mask[2]:
                    self.mask[0] = w_low
                    self.mask[1] = w_high

                # Caso que se corrija el continuum derecho
                elif w_low > self.mask[3] and w_high > self.mask[3]:
                    self.mask[4] = w_low
                    self.mask[5] = w_high

                # Case we want to select the complete region
                elif w_low < self.mask[0] and w_high > self.mask[5]:

                    # # Remove line from dataframe and save it
                    # self.remove_lines_df(self.current_df, self.Current_Label)
                    #
                    # # Save lines log df
                    # self.save_lineslog_dataframe(self.current_df, self.lineslog_df_address)

                    # Clear the selections
                    # self.mask = np.array([np.nan] * 6)

                    print(f'\n-- The line {self.line} mask has been removed')

                else:
                    print('- WARNING: Unsucessful line selection:')
                    print(f'-- {self.line}: w_low: {w_low}, w_high: {w_high}')

            # Check number of measurements after selection
            non_nans = (~pd.isnull(self.mask)).sum()

            # Proceed to re-measurement if possible:
            if non_nans == 6:

                # TODO add option to perform the measurement a new
                # self.clear_fit()
                # self.fit_from_wavelengths(self.line, self.mask, user_cfg={})

                # Parse the line regions to the dataframe
                self.results_to_database(self.line, self.log, fit_conf={}, export_params=[])

                # Save the corrected mask to a file
                self.store_measurement()

            # Redraw the line measurement
            self.in_ax.clear()
            self.plot_line_region_i(self.in_ax, self.line, logscale='auto')
            self.in_fig.canvas.draw()

        return

    def on_enter_axes(self, event):

        # Assign new axis
        self.in_fig = event.canvas.figure
        self.in_ax = event.inaxes

        # TODO we need a better way to index than the latex label
        # Recognise line line
        idx_line = self.log.index == self.in_ax.get_title()
        self.line = self.log.loc[idx_line].index.values[0]
        self.mask = self.log.loc[idx_line, 'w1':'w6'].values[0]

        # Restore measurements from log
        # self.database_to_attr()

        # event.inaxes.patch.set_edgecolor('red')
        # event.canvas.draw()

    def on_click(self, event):

        if event.dblclick:
            print(self.line)
            print(f'{event.button}, {event.x}, {event.y}, {event.xdata}, {event.ydata}')

    def store_measurement(self):

        # Read file in the stored address
        if self.linesLogAddress.is_file():
            file_DF = load_lines_log(self.linesLogAddress)

            # Add new line to the DF and sort it if it was new
            if self.line in file_DF.index:
                file_DF.loc[self.line, 'w1':'w6'] = self.mask
            else:
                file_DF.loc[self.line, 'w1':'w6'] = self.mask

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


class CubeInspector(Spectrum):

    def __init__(self, wave, cube_flux, image_bg, image_fg=None, contour_levels=None, color_norm=None,
                 redshift=0, lines_log_address=None, fits_header=None, plt_cfg={}, ax_cfg={},
                 ext_suffix='_LINESLOG', mask_file=None):

        """
        This class provides an interactive plot for IFU (Integra Field Units) data cubes consisting in two figures:
        On the left-hand side, a 2D image of the cube read from the ``image_bg`` parameter. This plot can include
        intensity contours from the ``contour_levels`` parameter. By default, the intensity contours are calculated from
        the ``image_bg`` matrix array unless an optional foreground ``image_fg`` array is provided. The spaxel selection
        is accomplished with a mouse right click.

        On the right-hand side, the selected spaxel spectrum is plotted. The user can select either window region using
        the matplotlib window *zoom* or *span* tools. As a new spaxel is selected the plotting limits in either figure
        should not change. To restore the plot axes limits you can click the *reset* (house icon).

        The user can provide a ``lines_log_address`` with the line measurements of the plotted object. In this case,
        the plot will include the fitted line profiles. The *.fits* file HDUs will be queried by the spaxel coordinates.
        The default format is ``{idx_j}-{idx_i}_LINESLOG)`` where idx_j and idx_i are the spaxel Y and X coordinates
        respectively

        :param wave: One dimensional array with the spectra wavelength range.
        :type wave: numpy.array

        :param cube_flux: Three dimensional array with the IFU cube flux
        :type cube_flux: numpy.array

        :param image_bg Two dimensional array with the flux band image for the plot background
        :type image_bg: numpy.array

        :param image_fg: Two dimensional array with the flux band image to plot foreground contours
        :type image_fg: numpy.array, optional

        :param contour_levels: One dimensional array with the flux contour levels in increasing order.
        :type contour_levels: numpy.array, optional

        :param color_norm: `Color normalization <https://matplotlib.org/stable/tutorials/colors/colormapnorms.html#sphx-glr-tutorials-colors-colormapnorms-py>`_
                            form the galaxy image plot
        :type color_norm: matplotlib.colors.Normalize, optional

        :param redshift: Object astronomical redshift
        :type redshift: float, optional

        :param lines_log_address: Address of the *.fits* file with the object line measurements.
        :type lines_log_address: str, optional

        :param fits_header: *.fits* header with the entries for the astronomical coordinates plot conversion.
        :type fits_header: dict, optional

        :param plt_cfg: Dictionary with the configuration for the matplotlib rcParams style.
        :type plt_cfg: dict, optional

        :param ax_cfg: Dictionary with the configuration for the matplotlib axes style.
        :type ax_cfg: dict, optional

        :param ext_suffix: Suffix of the line logs extensions. The default value is “_LINESLOG”.
        :type ext_suffix: str, optional

        """

        # Assign attributes to the parent class
        super().__init__(wave, input_flux=None, redshift=redshift, norm_flux=1)

        # Data attributes
        self.grid_mesh = None
        self.cube_flux = cube_flux
        self.wave = wave
        self.header = fits_header
        self.image_bg = image_bg
        self.image_fg = image_fg
        self.contour_levels_fg = contour_levels
        self.hdul_linelog = None
        self.ext_log = ext_suffix

        # Mask correction attributes
        self.mask_file = None
        self.mask_ext = None
        self.mask_dict = {}
        self.mask_color = None
        self.mask_array = None

        # Plot attributes
        self.fig = None
        self.ax0, self.ax1, self.in_ax = None, None, None
        self.fig_conf = None
        self.axes_conf = {}
        self.axlim_dict = {}
        self.color_norm = color_norm
        self.mask_color_i = None
        self.key_coords = None
        self.marker = None

        # Scenario we use the background image also for the contours
        if (image_fg is None) and (contour_levels is not None):
            self.image_fg = image_bg

        # State the figure and axis format
        self.fig_conf = STANDARD_PLOT.copy()
        self.axes_conf = {'image': {'xlabel': r'RA', 'ylabel': r'DEC', 'title': f'Cube flux slice'},
                          'spectrum': STANDARD_AXES}

        # Adjust the default theme
        self.fig_conf['figure.figsize'] = (18, 6)
        # self.fig_conf['ytick.labelsize'] = 10
        # self.fig_conf['xtick.labelsize'] = 10

        # Update to the user configuration
        self.fig_conf = {**self.fig_conf, **plt_cfg}

        for plot_type in ('image', 'spectrum'):
            if plot_type in ax_cfg:
                self.axes_conf[plot_type] = {**self.axes_conf[plot_type], **ax_cfg[plot_type]}

        # Prepare the mask correction attributes
        if mask_file is not None:

            assert Path(mask_file).is_file(), f'- ERROR: mask file at {mask_file} not found'

            self.mask_file = mask_file
            with fits.open(self.mask_file) as maskHDUs:

                # Save the fits data to restore later
                for HDU in maskHDUs:
                    if HDU.name != 'PRIMARY':
                        self.mask_dict[HDU.name] = (HDU.data.astype('bool'), HDU.header)
                self.mask_array = np.array(list(self.mask_dict.keys()))

                # Get the target mask
                self.mask_ext = self.mask_array[0]

        # Generate the figure
        with rc_context(self.fig_conf):

            # Figure structure
            self.fig = plt.figure()
            gs = gridspec.GridSpec(nrows=1, ncols=2, figure=self.fig, width_ratios=[1, 2], height_ratios=[1])

            # Spectrum plot
            self.ax1 = self.fig.add_subplot(gs[1])

            # Create subgrid for buttons if mask file provided
            if self.mask_ext is not None:
                gs_image = gridspec.GridSpecFromSubplotSpec(nrows=2, ncols=1, subplot_spec=gs[0], height_ratios=[0.8, 0.2])
            else:
                gs_image = gs

            # Image axes Astronomical coordinates if provided
            if self.header is None:
                self.ax0 = self.fig.add_subplot(gs_image[0])
            else:
                self.ax0 = self.fig.add_subplot(gs_image[0], projection=WCS(self.header), slices=('x', 'y', 1))

            # Buttons axis if provided
            if self.mask_ext is not None:
                self.ax2 = self.fig.add_subplot(gs_image[1])
                radio = RadioButtons(self.ax2, list(self.mask_array))

            # Image mesh grid
            frame_size = self.cube_flux.shape
            y, x = np.arange(0, frame_size[1]), np.arange(0, frame_size[2])
            self.grid_mesh = np.meshgrid(x, y)

            # Use central voxel as initial coordinate
            self.key_coords = int(self.cube_flux.shape[1] / 2), int(self.cube_flux.shape[2] / 2)

            # Load the complete fits lines log if input
            if lines_log_address is not None:
                self.hdul_linelog = fits.open(lines_log_address, lazy_load_hdus=False)

            # Generate the plot
            self.plot_map_voxel(self.image_bg, self.key_coords, self.image_fg, self.contour_levels_fg)

            # Connect the widgets
            self.fig.canvas.mpl_connect('button_press_event', self.on_click)
            self.fig.canvas.mpl_connect('axes_enter_event', self.on_enter_axes)
            if self.mask_file is not None:
                radio.on_clicked(self.mask_selection)

            # Display the figures
            plt.show()

            # Close the lines log if it has been opened
            if isinstance(self.hdul_linelog, fits.hdu.HDUList):
                self.hdul_linelog.close()

        return

    def plot_map_voxel(self, image_bg, voxel_coord=None, image_fg=None, flux_levels=None, frame='observed'):

        # Background image
        self.im = self.ax0.imshow(image_bg, cmap=cm.gray, norm=self.color_norm)

        # Emphasize input coordinate
        idx_j, idx_i = voxel_coord
        if voxel_coord is not None:

            # Delete previous if it is there
            if self.marker is not None:
                self.marker.remove()
                self.marker = None

            self.marker, = self.ax0.plot(idx_i, idx_j, '+', color='red')

        # Plot contours image
        if image_fg is not None:
            self.ax0.contour(self.grid_mesh[0], self.grid_mesh[1], image_fg, cmap='viridis', levels=flux_levels,
                             norm=colors.LogNorm())

        # Voxel spectrum
        if voxel_coord is not None:
            flux_voxel = self.cube_flux[:, idx_j, idx_i]
            self.ax1.step(self.wave, flux_voxel, label='', where='mid', color=self._color_dict['fg'])

        # Plot the emission line fittings:
        if self.hdul_linelog is not None:

            ext_name = f'{idx_j}-{idx_i}{self.ext_log}'

            # Better sorry than permission. Faster?
            try:
                lineslogDF = Table.read(self.hdul_linelog[ext_name]).to_pandas()
                lineslogDF.set_index('index', inplace=True)
                self.log = lineslogDF

            except KeyError:
                self.log = None

            if self.log is not None:

                # Plot all lines encountered in the voxel log
                line_list = self.log.index.values

                # Reference frame for the plot
                wave_plot, flux_plot, z_corr, mask_corr = self.plot_frame_switch(self.wave, flux_voxel, self.redshift, frame)

                # Compute the individual profiles
                wave_array, gaussian_array = gaussian_profiles_computation(line_list, self.log, z_corr, mask_corr)
                wave_array, cont_array = linear_continuum_computation(line_list, self.log, z_corr, mask_corr)

                # Mask with
                w3_array, w4_array = self.log.w3.values, self.log.w4.values

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
                                                    axis=self.ax1, frame=frame, cont_bands=None,
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
                                                    axis=self.ax1, frame=frame, cont_bands=None,
                                                    wave_array=wave_i, cont_array=cont_i,
                                                    gaussian_array=gauss_i)

        # Overplot the mask region
        if self.mask_file is not None:

            inv_masked_array = np.ma.masked_where(~self.mask_dict[self.mask_ext][0], self.image_bg)

            cmap_contours = cm.get_cmap(colorDict['mask_map'], self.mask_array.size)
            idx_color = np.argwhere(self.mask_array == self.mask_ext)[0][0]
            cm_i = colors.ListedColormap([cmap_contours(idx_color)])

            self.ax0.imshow(inv_masked_array, cmap=cm_i, vmin=0, vmax=1, alpha=0.5)

        # Add the mplcursors legend
        if mplcursors_check and (self.hdul_linelog is not None):
            for label, lineProfile in self._legends_dict.items():
                mplcursors.cursor(lineProfile).connect("add", lambda sel, label=label: sel.annotation.set_text(label))

        # Update the axis
        self.axes_conf['spectrum']['title'] = f'Voxel {idx_j} - {idx_i}'
        self.ax0.update(self.axes_conf['image'])
        self.ax1.update(self.axes_conf['spectrum'])

        return

    def on_click(self, event, new_voxel_button=3, mask_button='m'):

        if self.in_ax == self.ax0:

            # Save axes zoom
            self.save_zoom()

            if event.button == new_voxel_button:

                # Save clicked coordinates for next plot
                self.key_coords = np.rint(event.ydata).astype(int), np.rint(event.xdata).astype(int)

                # Remake the drawing
                self.im.remove()# self.ax0.clear()
                self.ax1.clear()
                self.plot_map_voxel(self.image_bg, self.key_coords, self.image_fg, self.contour_levels_fg)

                # Reset the image
                self.reset_zoom()
                self.fig.canvas.draw()

            if event.dblclick:

                if self.mask_file is not None:

                    # Save clicked coordinates for next plot
                    self.key_coords = np.rint(event.ydata).astype(int), np.rint(event.xdata).astype(int)

                    # Add or remove voxel from mask:
                    self.spaxel_selection()

                    # # Save the new mask:
                    # hdul = fits.HDUList([fits.PrimaryHDU()])
                    # for mask_name, mask_attr in self.mask_dict.items():
                    #     hdul.append(fits.ImageHDU(name=mask_name, data=mask_attr[0].astype(int), ver=1, header=mask_attr[1]))
                    # hdul.writeto(self.mask_file, overwrite=True, output_verify='fix')

                    # Remake the drawing
                    self.im.remove()# self.ax0.clear()
                    self.ax1.clear()
                    self.plot_map_voxel(self.image_bg, self.key_coords, self.image_fg, self.contour_levels_fg)

                    # Reset the image
                    self.reset_zoom()
                    self.fig.canvas.draw()

            return

    def mask_selection(self, mask_label):

        # Assign the mask
        self.mask_ext = mask_label

        # Replot the figure
        self.save_zoom()
        self.im.remove()# self.ax0.clear()
        self.ax1.clear()
        self.plot_map_voxel(self.image_bg, self.key_coords, self.image_fg, self.contour_levels_fg)

        # Reset the image
        self.reset_zoom()
        self.fig.canvas.draw()

        return

    def spaxel_selection(self):

        for mask, mask_data in self.mask_dict.items():

            mask_matrix = mask_data[0]
            if mask == self.mask_ext:
                mask_matrix[self.key_coords[0], self.key_coords[1]] = not mask_matrix[self.key_coords[0], self.key_coords[1]]
            else:
                mask_matrix[self.key_coords[0], self.key_coords[1]] = False

        return

    def on_enter_axes(self, event):
        self.in_ax = event.inaxes

    def save_zoom(self):

        self.axlim_dict['image_xlim'] = self.ax0.get_xlim()
        self.axlim_dict['image_ylim'] = self.ax0.get_ylim()
        self.axlim_dict['spec_xlim'] = self.ax1.get_xlim()
        self.axlim_dict['spec_ylim'] = self.ax1.get_ylim()

        return

    def reset_zoom(self):

        self.ax0.set_xlim(self.axlim_dict['image_xlim'])
        self.ax0.set_ylim(self.axlim_dict['image_ylim'])
        self.ax1.set_xlim(self.axlim_dict['spec_xlim'])
        self.ax1.set_ylim(self.axlim_dict['spec_ylim'])

        return