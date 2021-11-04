import numpy as np
import pandas as pd
from pathlib import Path
from astropy.io import fits

from lmfit import fit_report

from .model import EmissionFitting
from .tools import label_decomposition
from .plots import LiMePlots
from .io import _LOG_EXPORT, LOG_COLUMNS, load_lines_log


class Spectrum(EmissionFitting, LiMePlots):

    """
    This class provides a set of tools to measure emission lines from ionized gas to study its chemistry and kinematics

    :ivar wave: Wavelength array
    :ivar flux: Flux array
    """

    def __init__(self, input_wave=None, input_flux=None, input_err=None, linesDF_address=None, redshift=0,
                 normFlux=1.0, crop_waves=None):

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
                self.linesDF = load_lines_log(linesDF_address)
            else:
                print(f'-- WARNING: linesLog not found at {self.linesLogAddress}')

        return

    def fit_from_wavelengths(self, label, line_wavelengths, user_cfg={}, algorithm='lmfit'):

        """
        This function fits an emission line by providing its label, location and an optional fit configuration. The
        algorithm accounts for the object redshift if it was provided by the user and corrects the input
        line_wavelengths

        :param str label: Line reference incluiding the ion and wavelength. Example: O3_5007A
        :param np.ndarray line_wavelengths: Array with 6 wavelength values defining an emision line left continuum,  emission region and right continuum
        :param dict user_cfg: Dictionary with the user configuration for the fitting
        :param algorithm: Algorithm for the line profile fitting (Not implemented)
        """

        # For security previous measurement is cleared and a copy of the user configuration is used
        self.clear_fit()
        fit_conf = user_cfg.copy()

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

        # Import kinematics if requested
        self.import_line_kinematics(fit_conf, z_cor=1 + self.redshift)

        # Gaussian fitting # TODO Add logic for very small lines
        idcsLine = idcsEmis + idcsCont
        x_array = self.wave[idcsLine]
        y_array = self.flux[idcsLine]
        w_array = 1.0/self.errFlux[idcsLine] if self.errFlux is not None else np.full(x_array.size, 1.0 / self.std_cont)
        self.gauss_lmfit(self.lineLabel, x_array, y_array, w_array, fit_conf, self.linesDF, z_obj=self.redshift)

        # Safe the results to log DF
        self.results_to_database(self.lineLabel, self.linesDF, fit_conf)

        return

    def import_line_kinematics(self, user_conf, z_cor):

        # Check if imported kinematics come from blended component
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

    def display_results(self, label=None, show_fit_report=False, show_plot=False, log_scale=True, frame='obs'):

        # Case no line as input: Show the current measurement
        if label is None:
            if self.lineLabel is not None:
                label = self.lineLabel
                output_ref = (f'\nLine label: {label}\n'
                              f'- Line regions: {self.lineWaves}\n'
                              f'- Normalization flux: {self.normFlux}\n'
                              f'- Redshift: {self.redshift}\n'
                              f'- Peak wavelength: {self.peak_wave:.2f}; peak intensity: {self.peak_flux:.2f}\n'
                              f'- Cont. slope: {self.m_cont:.2e}; Cont. intercept: {self.n_cont:.2e}\n')

                if self.blended_check:
                    mixtureComponents = np.array(self.blended_label.split('-'))
                else:
                    mixtureComponents = np.array([label], ndmin=1)

                output_ref += f'\n- {label} Intg flux: {self.intg_flux:.3f} +/- {self.intg_err:.3f}\n'

                if mixtureComponents.size == 1:
                    output_ref += f'- {label} Eqw (intg): {self.eqw[0]:.2f} +/- {self.eqw_err[0]:.2f}\n'

                for i, lineRef in enumerate(mixtureComponents):
                    output_ref += (f'\n- {lineRef} gaussian fitting:\n'
                                   f'-- Gauss flux: {self.gauss_flux[i]:.3f} +/- {self.gauss_err[i]:.3f}\n'
                                   # f'-- Amplitude: {self.amp[i]:.3f} +/- {self.amp_err[i]:.3f}\n'
                                   f'-- Center: {self.center[i]:.2f} +/- {self.center_err[i]:.2f}\n'
                                   f'-- Sigma (km/s): {self.sigma_vel[i]:.2f} +/- {self.sigma_vel_err[i]:.2f}\n')
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
                output_ref += f'\n- LmFit output:\n{fit_report(self.fit_output)}\n'
            else:
                output_ref += f'\n- LmFit output not available\n'

        # Show the result
        print(output_ref)

        # Display plot
        if show_plot:
            self.plot_fit_components(self.fit_output, log_scale=log_scale, frame=frame)

        return

    def clear_fit(self):
        super().__init__()
        return

