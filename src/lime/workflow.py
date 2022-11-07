import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sys import exit

from .model import LineFitting
from .tools import define_masks, label_decomposition
from .transitions import Line
from .io import _LOG_EXPORT, _LOG_COLUMNS, load_lines_log, progress_bar

_logger = logging.getLogger('LiMe')


def check_file(input, variable_type):

    if isinstance(input, variable_type):
        output = input

    elif Path(input).is_file():
        output = load_lines_log(input)

    else:
        _logger.critical(f'Not file found at {input}.\nPlease introduce a {variable_type} as input or check the'
                         f'file address.')
        exit()

    return output


def review_bands(line, emis_wave, cont_wave, limit_narrow=6):

    # Review the transition bands before
    emis_band_lengh = emis_wave.size if not np.ma.is_masked(emis_wave) else np.sum(~emis_wave.mask)
    cont_band_length = cont_wave.size if not np.ma.is_masked(cont_wave) else np.sum(~cont_wave.mask)

    if emis_band_lengh / emis_wave.size < 0.5:
        _logger.warning(f'The line band for {line.label} has very few valid pixels')

    if cont_band_length / cont_wave.size < 0.5:
        _logger.warning(f'The continuum band for {line.label} has very few valid pixels')

    # Store error very small mask
    if emis_band_lengh <= 1:
        if line.observations == 'no':
            line.observations = 'Small_line_band'
        else:
            line.observations += '-Small_line_band'
        _logger.warning(f'- Line {line.label} mask band is too small ({emis_wave.size} value array): {emis_wave}')

    # Check if the line is very narrow for fit initial conditions
    line._narrow_check = True if emis_band_lengh <= limit_narrow else False

    return


def import_line_kinematics(line, z_cor, log, units_wave):

    # Check if imported kinematics come from blended component
    if line.profile_label != 'no':
        childs_list = line.profile_label.split('-')
    else:
        childs_list = np.array(line.label, ndmin=1)

    for child_label in childs_list:

        parent_label = line._fit_conf.get(f'{child_label}_kinem')

        if parent_label is not None:

            # Case we want to copy from previous line and the data is not available
            if (parent_label not in log.index) and (not line.blended_check):
                _logger.warning(f'{parent_label} has not been measured. Its kinematics were not copied to {child_label}')

            else:
                ion_parent, wtheo_parent, latex_parent = label_decomposition(parent_label, scalar_output=True, units_wave=units_wave)
                ion_child, wtheo_child, latex_child = label_decomposition(child_label, scalar_output=True, units_wave=units_wave)

                # Copy v_r and sigma_vel in wavelength units
                for param_ext in ('center', 'sigma'):
                    param_label_child = f'{child_label}_{param_ext}'

                    # Warning overwritten existing configuration
                    if param_label_child in line._fit_conf:
                        _logger.warning(f'{param_label_child} overwritten by {parent_label} kinematics in configuration input')

                    # Case where parent and child are in blended group
                    if parent_label in childs_list:
                        param_label_parent = f'{parent_label}_{param_ext}'
                        param_expr_parent = f'{wtheo_child / wtheo_parent:0.8f}*{param_label_parent}'

                        line._fit_conf[param_label_child] = {'expr': param_expr_parent}

                    # Case we want to copy from previously measured line
                    else:
                        mu_parent = log.loc[parent_label, ['center', 'center_err']].values
                        sigma_parent = log.loc[parent_label, ['sigma', 'sigma_err']].values

                        if param_ext == 'center':
                            param_value = wtheo_child / wtheo_parent * (mu_parent / z_cor)
                        else:
                            param_value = wtheo_child / wtheo_parent * sigma_parent

                        line._fit_conf[param_label_child] = {'value': param_value[0], 'vary': False}
                        line._fit_conf[f'{param_label_child}_err'] = param_value[1]

    return


def results_to_log(line, log, norm_flux, units_wave, export_params=_LOG_EXPORT):

    # # Recover line data
    # if line.blended_check:
    #     line_components = line.profile_label.split('-')
    # else:
    #     line_components = np.array([line.label], ndmin=1)
    #
    # ion, waveRef, latexLabel = label_decomposition(line_components, comp_dict=line._fit_conf, units_wave=units_wave)

    # Loop through the line components
    for i, comp in enumerate(line.list_comps):

        # Convert current measurement to a pandas series container
        log.loc[comp, ['ion', 'wavelength', 'latex_label']] = line.ion[i], line.wave[i], line.latex[i]
        log.loc[comp, 'w1':'w6'] = line.mask

        # Treat every line
        for param in export_params:

            # Get component parameter
            if _LOG_COLUMNS[param][2]:
                param_value = line.__getattribute__(param)[i]
            else:
                param_value = line.__getattribute__(param)

            # De normalize
            if _LOG_COLUMNS[param][0]:
                param_value = param_value * norm_flux

            log.loc[comp, param] = param_value

    return


class LineTreatment(LineFitting):

    def __init__(self, spectrum):

        # Instantiate the dependencies
        LineFitting.__init__(self)

        # Lime spectrum object with the scientific data
        self._spec = spectrum
        self.line = None

    def band(self, label, mask=None, fit_conf=None, fit_method='leastsq', emission_check=True, cont_from_bands=True,
             temp=10000.0):

        """

         This function fits a line given its line and spectral mask. The line notation consists in the transition
         ion and wavelength (with units) separated by an underscore, i.e. O3_5007A.

         The location mask consists in a 6 values array with the wavelength boundaries for the line location and two
         adjacent continua. These wavelengths must be sorted by increasing order and in the rest _frame.

         The user can specify the properties of the fitting: Number of components and parameter boundaries. Please check
         the documentation for the complete description.

         The user can specify the minimization algorithm for the `LmFit library <https://lmfit.github.io/lmfit-py/fitting.html#lmfit.minimizer.Minimizer.minimize>`_.

         By default, the algorithm assumes an emission line. The user can set the parameter ``emission=False`` for an
         absorption.

         If the sigma spectrum was not provided, the fitting estimates the pixel uncertainty from the adjacent continua flux
         standard deviation assuming a linear profile. If the parameter ``adjacent_cont=True`` the adjacent continua is also
         use to calculate the continuum at the line location. Otherwise, only the line continuum is calculated only with the
         first and last pixel in the line band (the 3rd and 4th values in the ``line_wavelengths`` array)

         For the calculation of the thermal broadening on the emission lines the user can include the line electron
         temperature in Kelvin. The default value is 10000 K

         :param line: line line in the ``LiMe`` notation, i.e. H1_6563A_b
         :type line: string

         :param mask: 6 wavelengths spectral mask with the blue continuum, line and red continuum bands.
         :type mask: numpy.ndarray

         :param user_cfg: Dictionary with the fitting configuration.
         :type user_cfg: dict, optional

         :param fit_method: Minimizing algorithm for the LmFit library. The default method is ``leastsq``.
         :type fit_method: str, optional

         :param emission: Boolean check for the line type. The default is ``True`` for an emission line
         :type emission: bool, optional

         :param adjacent_cont: Boolean check for the line continuum calculation. The default value ``True`` includes the
                               adjacent continua array
         :type adjacent_cont: bool, optional

         :param temp_line: Line electron temperature for the thermal broadening calculation.
         :type temp_line: bool, optional

         """

        # Interpret the input line
        self.line = Line(label, mask, fit_conf, emission_check, cont_from_bands)

        # Get the bands regions
        idcsEmis, idcsCont = define_masks(self._spec.wave, self.line.mask * (1 + self._spec.redshift), self.line.pixel_mask)
        emisWave, emisFlux = self._spec.wave[idcsEmis], self._spec.flux[idcsEmis]
        contWave, contFlux = self._spec.wave[idcsCont], self._spec.flux[idcsCont]
        err_array = self._spec.err_flux[idcsEmis] if self._spec.err_flux is not None else None

        # Check the bands size
        review_bands(self.line, emisWave, contWave)

        # Non-parametric measurements
        self.integrated_properties(self.line, emisWave, emisFlux, contWave, contFlux, err_array)

        # Import kinematics if requested
        import_line_kinematics(self.line, 1 + self._spec.redshift, self._spec.log, self._spec.units_wave)

        # Combine bands
        idcsLine = idcsEmis + idcsCont
        x_array, y_array = self._spec.wave[idcsLine], self._spec.flux[idcsLine]

        # Fit weights according to input err
        if self._spec.err_flux is None:
            w_array = np.full(x_array.size, 1.0 / self.line.std_cont)
        else:
            w_array = 1.0 / self._spec.err_flux[idcsLine]

        # Gaussian fitting
        self.profile_fitting(self.line, x_array, y_array, w_array, self._spec.redshift, fit_method, temp, self._spec.units_wave,
                             self._spec.inst_FWHM)

        # Save the line parameters to the dataframe
        results_to_log(self.line, self._spec.log, self._spec.norm_flux, self._spec.units_wave)

        return

    def frame(self, bands_df, fit_conf=None,  label=None, fit_method='leastsq', emission_check=True, cont_from_bands=True,
              temp=10000.0, progress_output=None, plot_fits=False):

        # Check if the lines table is a dataframe or a file
        bands_df = check_file(bands_df, pd.DataFrame)

        # Crop the analysis to the target lines
        if label is not None:
            idcs = bands_df.index.isin(label)
            line_list = bands_df.loc[idcs].index.values
        else:
            line_list = bands_df.index.values

        # Loop text decision
        bar_output = True if progress_output == 'bar' else False

        # Loop through the lines
        n_lines = len(line_list)
        for i in np.arange(n_lines):

            line = line_list[i]

            # Progress message
            if progress_output is not None:
                if bar_output:
                    progress_bar(i+1, n_lines, post_text=f'of lines')
                else:
                    print(f'{i+1}/{n_lines}) {line}')

            # Fit the lines
            self.band(line, bands_df.loc[line, 'w1':'w6'].values, fit_conf, fit_method, emission_check, cont_from_bands,
                      temp)

            if plot_fits:
                self._spec.plot.line()

        # self.band(line, bands_df.loc[line, 'w1':'w6'].values, fit_conf, fit_method, emission_check,
        #                   cont_from_bands, temp)


        #
        # # Loop through the lines
        # if progress_output is None:
        #
        #     # No message:
        #     for line in line_list:
        #         self.band(line, bands_df.loc[line, 'w1':'w6'].values, fit_conf, fit_method, emission_check,
        #                   cont_from_bands, temp)
        #
        # # Progress bar
        # elif progress_output == 'bar':
        #     n_lines = line_list.size
        #
        #     for i, line in enumerate(line_list):
        #         progress_bar(i, n_lines, post_text=f'of lines')
        #         self.band(line, bands_df.loc[line, 'w1':'w6'].values, fit_conf, fit_method, emission_check,
        #                   cont_from_bands, temp)
        #
        # # Line labels and numbers
        # elif progress_output == 'labels':
        #
        #     n_lines = line_list.size
        #     for i, line in enumerate(line_list):
        #         print(f'{i}/{n_lines}) {line}')
        #         self.band(line, bands_df.loc[line, 'w1':'w6'].values, fit_conf, fit_method, emission_check,
        #                   cont_from_bands, temp)
        #
        # # Line name and number:
        # else:
        #
        #     for line in line_list:
        #         self.band(line, bands_df.loc[line, 'w1':'w6'].values, fit_conf, fit_method, emission_check,
        #                   cont_from_bands, temp)


        return

