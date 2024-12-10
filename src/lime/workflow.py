import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from astropy.io import fits
from time import time

from .model import LineFitting, signal_to_noise_rola, gaussian_model, sigma_corrections, c_KMpS
from .plots import redshift_fit_evaluation
from .tools import ProgressBar, join_fits_files, extract_wcs_header, pd_get, unit_conversion
from .transitions import Line, air_to_vacuum_function
from .io import check_file_dataframe, check_file_array_mask, log_to_HDU, results_to_log, load_frame, LiMe_Error, check_fit_conf, _PARENT_BANDS
from lmfit.models import PolynomialModel

try:
    import aspect
    aspect_check = True
except ImportError:
    aspect_check = False


_logger = logging.getLogger('LiMe')


def review_bands(line, emis_wave, cont_wave, emis_flux, cont_flux, limit_narrow=7):

    # Default value
    proceed = True

    # Mask check
    mask_check = np.ma.isMaskedArray(emis_flux)

    # Confirm is not all the pixels are masked
    if mask_check:
        if np.all(emis_flux.mask) or np.all(cont_flux.mask):
            proceed = False

    # Review the number and type of pixel values
    if proceed:

        # Review the transition bands before
        emis_band_lengh = emis_wave.size if not mask_check else np.sum(~emis_wave.mask)
        cont_band_length = cont_wave.size if not mask_check else np.sum(~cont_wave.mask)

        if emis_band_lengh / emis_wave.size < 0.5:
            _logger.warning(f'The line band for {line.label} has very few valid pixels')

        if cont_band_length / cont_wave.size < 0.5:
            if cont_band_length > 0:
                _logger.warning(f'The continuum band for {line.label} has very few valid pixels)')
            else:
                _logger.warning(f"The continuum bands for {line.label} have 0 pixels. It won't be measured")
                proceed = False

        # Store error very small mask
        if emis_band_lengh <= 1:
            if line.observations == 'no':
                line.observations = 'Small_line_band'
            else:
                line.observations += '-Small_line_band'

            if np.ma.isMaskedArray(emis_wave):
                length = np.sum(~emis_wave.mask)
            else:
                length = emis_band_lengh
            _logger.warning(f'The  {line.label} band is too small ({length} length array): {emis_wave}')

        # Security check not all the pixels are zero
        if emis_flux.sum() == 0:
            _logger.warning(f'The {line.label} line pixels sum is zero, it has been excluded from the analysis')
            proceed = False

        if cont_flux.sum() == 0:
            _logger.warning(f'The {line.label} continuum pixels sum is zero, it has been excluded from the analysi')
            proceed = False

    return proceed


def import_line_kinematics(line, z_cor, log, units_wave, fit_conf):

    # Check if imported kinematics come from blended component
    if line.group_label is not None:
        childs_list = line.group_label.split('+')
    else:
        childs_list = np.array(line.label, ndmin=1)

    for child_label in childs_list:

        parent_label = fit_conf.get(f'{child_label}_kinem')

        if parent_label is not None:

            # Case we want to copy from previous line and the data is not available
            if (parent_label not in log.index) and (not line.blended_check):
                _logger.info(f'{parent_label} has not been measured. Its kinematics were not copied to {child_label}')

            else:

                line_parent = Line(parent_label)
                line_child = Line(child_label)
                wtheo_parent, wtheo_child = line_parent.wavelength[0], line_child.wavelength[0]

                # Copy v_r and sigma_vel in wavelength units
                for param_ext in ('center', 'sigma'):
                    param_label_child = f'{child_label}_{param_ext}'

                    # Warning overwritten existing configuration
                    if param_label_child in fit_conf:
                        _logger.warning(f'{param_label_child} overwritten by {parent_label} kinematics in configuration input')

                    # Case where parent and child are in blended group
                    if parent_label in childs_list:
                        param_label_parent = f'{parent_label}_{param_ext}'
                        param_expr_parent = f'{wtheo_child/wtheo_parent:0.8f}*{param_label_parent}'

                        fit_conf[param_label_child] = {'expr': param_expr_parent}

                    # Case we want to copy from previously measured line
                    else:
                        mu_parent = log.loc[parent_label, ['center', 'center_err']].values
                        sigma_parent = log.loc[parent_label, ['sigma', 'sigma_err']].values

                        if param_ext == 'center':
                            param_value = wtheo_child / wtheo_parent * (mu_parent / z_cor)
                        else:
                            param_value = wtheo_child / wtheo_parent * sigma_parent

                        fit_conf[param_label_child] = {'value': param_value[0], 'vary': False}
                        fit_conf[f'{param_label_child}_err'] = param_value[1]

    return


def check_spectrum_bands(line, wave_rest_array):

    valid_check = True
    wave_rest_array = wave_rest_array.data if np.ma.isMaskedArray(wave_rest_array) else wave_rest_array

    if line.mask is not None:
        if (wave_rest_array[0] <= line.mask[0]) and (line.mask[-1] <= wave_rest_array[-1]):
            pass
        else:
            _logger.warning(f'Line {line} limits (w1={line.mask[0]}, w6={line.mask[-1]}) outside spectrum wavelength '
                            f'range (wmin={wave_rest_array[0]}, wmax={wave_rest_array[-1]}) (rest frame values)')
            valid_check = False
    else:
        _logger.warning(f'Line {line} was not found on the input bands database.')
        valid_check = False

    return valid_check


def check_cube_bands(input_bands, mask_list, fit_cfg):

    if input_bands is None:

        # Recover the mask_configuration as a list
        for mask_name in mask_list:

            mask_fit_cfg = fit_cfg.get(f'{mask_name}_line_fitting')

            missing_mask = False
            if mask_fit_cfg is not None:
                if mask_fit_cfg.get('bands') is None:
                    missing_mask = True
            else:
                missing_mask = True

            if missing_mask:
                error_message = 'No input "bands" provided. In this case you need to include the \n' \
                                f'you need to specify an "bands=log_file_address" entry the ' \
                                f'"[{mask_name}_file]" of your fitting configuration file'
                raise LiMe_Error(error_message)

    return


def recover_level_conf(fit_cfg, mask_key, default_key):

    default_cfg = fit_cfg.get(f'{default_key}_line_fitting') if default_key is not None else None
    mask_cfg = fit_cfg.get(f'{mask_key}_line_fitting') if mask_key is not None else None

    # Case there are not leveled entries
    if (default_cfg is None) and (mask_cfg is None):
        output_conf = fit_cfg

    # Proceed to update the levels
    else:

        # Default configuration
        default_conf = {} if default_cfg is None else default_cfg
        default_detect = default_conf.get('line_detection')

        # Mask conf
        mask_conf = {} if mask_cfg is None else mask_cfg
        mask_detect = mask_conf.get('line_detection')

        # Update the levels
        output_conf = {**default_conf, **mask_conf}

        # If no line detection don't add it
        if mask_detect is not None:
            output_conf['line_detection'] = mask_detect
        elif default_detect is not None:
            output_conf['line_detection'] = default_detect
        else:
            pass

    return output_conf


def check_compound_line_exclusion(line, lines_df):

    # Confirm the dataframe includes the group of lines
    group_label = pd_get(lines_df, line, 'group_label', transform='none')

    # Confirm if the line is in the group of lines
    if group_label is not None:
        comp_list = group_label.split('+')
        measure_check = False if line in comp_list else True
    else:
        measure_check = True

    return measure_check


def continuum_model_fit(x_array, y_array, idcs, degree):

    poly3Mod = PolynomialModel(prefix=f'poly_{degree}', degree=degree)
    poly3Params = poly3Mod.guess(y_array[idcs], x=x_array[idcs])

    try:
        poly3Out = poly3Mod.fit(y_array[idcs], poly3Params, x=x_array[idcs])
        cont_fit = poly3Out.eval(x=x_array)

    except TypeError:
        _logger.warning(f'- The continuum fitting polynomial has more degrees ({degree}) than data points')
        cont_fit = np.full(x_array.size, np.nan)

    return cont_fit


def compute_z_key(redshift, lines_lambda, wave_matrix, amp_arr, sigma_arr):

    # Compute the observed line wavelengths
    obs_lambda = lines_lambda * (1 + redshift)
    obs_lambda = obs_lambda[(obs_lambda > wave_matrix[0, 0]) & (obs_lambda < wave_matrix[0, -1])]

    if obs_lambda.size > 0:

        # Compute indexes ion array
        idcs_obs = np.searchsorted(wave_matrix[0, :], obs_lambda)

        # Compute lambda arrays:
        sigma_lines = sigma_arr[idcs_obs]
        mu_lines = wave_matrix[0, :][idcs_obs]

        # Compute the Gaussian bands
        x_matrix = wave_matrix[:idcs_obs.size, :]
        gauss_matrix = gaussian_model(x_matrix, amp_arr, mu_lines[:, None], sigma_lines[:, None])
        gauss_arr = gauss_matrix.sum(axis=0)

        # Set maximum to 1:
        idcs_one = gauss_arr > 1
        gauss_arr[idcs_one] = 1

    else:
        gauss_arr = None

    return gauss_arr


def line_bands(wave_intvl=None, lines_list=None, particle_list=None, redshift=None, units_wave='Angstrom', sig_digits=None,
               vacuum_waves=False, ref_bands=None, update_labels=True, update_latex=True):
    """

    This function returns `LiMe bands database <https://lime-stable.readthedocs.io/en/latest/inputs/n_inputs3_line_bands.html>`_
    as a pandas dataframe.

    If the user provides a wavelength array (``wave_inter``) the output dataframe will be limited to the lines within
    this wavelength interval.

    Similarly, the user provides a ``lines_list`` or a ``particle_list`` the output bands will be limited to the these
    lists. These inputs must follow `LiMe notation style <https://lime-stable.readthedocs.io/en/latest/inputs/n_inputs2_line_labels.html>`_

    If the user provides a redshift value alongside the wavelength interval (``wave_intvl``) the output bands will be
    limited to the transitions at that observed range.

    The user can specify the desired wavelength units using the `astropy string format <https://docs.astropy.org/en/stable/units/ref_api.html>`_
    or introducing the `astropy unit variable  <https://docs.astropy.org/en/stable/units/index.html>`_. The default value
    unit is angstroms.

    The argument ``sig_digits`` determines the number of decimal figures for the line labels.

    The user can request the output line labels and bands wavelengths in vacuum setting ``vacuum=True``. This conversion
    is done using the relation from `Greisen et al. (2006) <https://www.aanda.org/articles/aa/abs/2006/05/aa3818-05/aa3818-05.html>`_.

    Instead of the default LiMe database, the user can provide a ``ref_bands`` dataframe (or the dataframe file address)
    to use as the reference database.

    :param wave_intvl: Wavelength interval for output line transitions.
    :type wave_intvl: list, numpy.array, lime.Spectrum, lime.Cube, optional

    :param lines_list: Line list for output line bands.
    :type lines_list: list, numpy.array, optional

    :param particle_list: Particle list for output line bands.
    :type particle_list: list, numpy.array, optional

    :param redshift: Redshift interval for output line bands.
    :type redshift: list, numpy.array, optional

    :param units_wave: Labels and bands wavelength units. The default value is "A".
    :type units_wave: str, optional

    :param sig_digits: Number of decimal figures for the line labels.
    :type sig_digits: int, optional

    :param vacuum_waves: Set to True for vacuum wavelength values. The default value is False.
    :type vacuum_waves: bool, optional

    :param ref_bands: Reference bands dataframe. The default value is None.
    :type ref_bands: pandas.Dataframe, str, pathlib.Path, optional

    :return:
    """

    # Use the default lime mask if none provided
    if ref_bands is None:
        ref_bands = _PARENT_BANDS

    # Load the reference bands
    bands_df = check_file_dataframe(ref_bands, pd.DataFrame)

    # Check for modifications on labels units or wavelengths
    new_format = False

    # Convert to vacuum wavelengths if requested
    if vacuum_waves:
        bands_df['wavelength'] = bands_df['wave_vac']
        bands_lim_columns = ['w1', 'w2', 'w3', 'w4', 'w5', 'w6']
        bands_df[bands_lim_columns] = air_to_vacuum_function(bands_df[bands_lim_columns].to_numpy())
        new_format = True

    # Convert to requested units
    if units_wave != 'Angstrom':
        wave_columns = ['wave_vac', 'wavelength', 'w1', 'w2', 'w3', 'w4', 'w5', 'w6']
        conversion_factor = unit_conversion(in_units='Angstrom', out_units=units_wave, wave_array=1)
        bands_df.loc[:, wave_columns] = bands_df.loc[:, wave_columns] * conversion_factor
        bands_df['units_wave'] = units_wave
        new_format = True

    # First slice by wavelength and redshift
    idcs_rows = np.ones(bands_df.index.size).astype(bool)
    if wave_intvl is not None:

        # Establish the lower and upper wavelength limits
        w_min, w_max = wave_intvl[0], wave_intvl[-1]

        # Account for redshift
        redshift = redshift if redshift is not None else 0
        wavelength_array = bands_df['wavelength'].to_numpy() * (1 + redshift)

        # Compare with wavelength values
        idcs_rows = idcs_rows & (wavelength_array >= w_min) & (wavelength_array <= w_max)

    # Second slice by particle
    if particle_list is not None:
        idcs_rows = idcs_rows & bands_df.particle.isin(particle_list)

    # Finally slice by the name of the lines
    if lines_list is not None:
        idcs_rows = idcs_rows & bands_df.index.isin(lines_list)

    # Final table
    bands_df = bands_df.loc[idcs_rows]

    # Update the labels to reflect new wavelengths and units if necessary and user requests it
    if new_format and update_labels:
        for label in bands_df.index:
            line = Line(label, band=bands_df)
            line.update_label(sig_digits=sig_digits, update_latex=update_latex)
            bands_df.rename(index={label: line.label}, inplace=True)

    return bands_df


class SpecRetriever:

    def __init__(self, spectrum):

        self._spec = spectrum

        return

    def line_bands(self, lines_list=None, particle_list=None, sig_digits=None, vacuum_waves=False, ref_bands=None, update_labels=True,
                   update_latex=False, components_detection=False, bands_kinematic_width=None):

        # Remove the mask from the wavelength array if necessary
        wave_intvl = self._spec.wave.data if np.ma.isMaskedArray(self._spec.wave) else self._spec.wave

        # Compute the bands to match the observation
        bands = line_bands(wave_intvl, lines_list, particle_list, redshift=self._spec.redshift, units_wave=self._spec.units_wave,
                           sig_digits=sig_digits, vacuum_waves=vacuum_waves, ref_bands=ref_bands, update_labels=update_labels,
                           update_latex=update_latex)

        # Adjust the middle bands to the core
        if bands_kinematic_width is not None:

            w_central = bands.wavelength.to_numpy() * (1 + self._spec.redshift)
            idcs_central = (w_central > self._spec.wave.data[0]) & (w_central < self._spec.wave.data[-1])

            HalfBox_pixels = np.round(3 * (bands_kinematic_width / c_KMpS) * 800)
            HalfBox_pixels = np.int64(np.maximum(HalfBox_pixels, 3))

            idcs = np.searchsorted(wave_intvl, w_central)
            idcsW3 = idcs + HalfBox_pixels
            idcsW4 = idcs - HalfBox_pixels

            bands.loc[idcs_central, 'w3'] = self._spec.wave.data[idcsW3]
            bands.loc[idcs_central, 'w4'] = self._spec.wave.data[idcsW4]

        # Filter the table to match the line detections
        if components_detection:
            if self._spec.features.pred_arr is not None:

                # Create masks for all intervals
                starts = bands.w3.to_numpy()[:, None] * (1 + self._spec.redshift)
                ends = bands.w4.to_numpy()[:, None] * (1 + self._spec.redshift)

                # Check if x values fall within each interval
                in_intervals = (self._spec.wave.data >= starts) & (self._spec.wave.data < ends)

                # Check where y equals the target category
                is_target_category = np.isin(self._spec.features.pred_arr, (3, 7, 9))

                # Combine the masks to count target_category occurrences in each interval
                counts = np.sum(in_intervals & is_target_category, axis=1)

                # Check which intervals satisfy the minimum count condition
                idcs = counts >= 3
                bands = bands.loc[idcs]

            else:
                raise(LiMe_Error(f'The aspect line detection algorithm needs to be run before matching the bands'))

        return bands


class SpecTreatment(LineFitting):

    def __init__(self, spectrum):

        # Instantiate the dependencies
        LineFitting.__init__(self)

        # Lime spectrum object with the scientific data
        self._spec = spectrum
        self.line = None
        self._i_line = 0
        self._n_lines = 0

    def bands(self, label, bands=None, fit_conf=None, min_method='least_squares', profile='g-emi', cont_from_bands=True,
              temp=10000.0, id_conf_prefix=None, default_conf_prefix='default'):

        """

        This function fits a line on the spectrum object from a given band.

        The first input is the line ``label``. The user can provide a string with the default `LiMe notation
        <https://lime-stable.readthedocs.io/en/latest/inputs/n_inputs2_line_labels.html>`_. Otherwise, the user can
        provide the transition wavelength in the same units as the spectrum and the transition will be queried from the
        ``bands`` argument.

        The second input is the line ``bands`` this argument can be a six value array with the same units as the
        spectrum wavelength specifying the line position and continua location. Otherwise, the ``bands`` can be a pandas
        dataframe (or the frame address) and the wavelength array will be automatically query from it.

        If the ``bands`` are not provided by the user, the default bands database will be used. You can learn more on
        `the bands documentation <https://lime-stable.readthedocs.io/en/latest/inputs/n_inputs3_line_bands.html>`_.

        The third input is a dictionary the fitting configuration ``fit_conf`` attribute. You can learn more on the
        `profile fitting documentation <https://lime-stable.readthedocs.io/en/latest/inputs/n_inputs4_fit_configuration.html>`_.

        The ``min_method`` argument provides the minimization algorithm for the `LmFit functions
        <https://lmfit.github.io/lmfit-py/fitting.html#lmfit.minimizer.Minimizer.minimize>`_.

        By default, the profile fitting assumes an emission Gaussian shape, with ``profile="g-emi"``. The profile keywords
        are described on the `label documentation <https://lime-stable.readthedocs.io/en/latest/inputs/n_inputs2_line_labels.html>`_

        The ``cont_from_bands=True`` argument forces the continuum to be measured from the adjacent line bands. If
        ``cont_from_bands=False`` the continuum gradient is calculated from the first and last pixel from the line band
        (w3-w4)

        For the calculation of the thermal broadening on the emission lines the user can include the line electron
        temperature in Kelvin. The default value ``temp`` is 10000 K.

        :param label: Line label or wavelength transition to be queried on the ``bands`` dataframe.
        :type label: str, float, optional

        :param bands: Bands six-value array, bands dataframe (or file address to the dataframe).
        :type bands: np.array, pandas.dataframe, str, Path, optional

        :param fit_conf: Fitting configuration.
        :type fit_conf: dict, optional

        :param min_method: `Minimization algorithm <https://lmfit.github.io/lmfit-py/fitting.html#lmfit.minimizer.Minimizer.minimize>`_.
                            The default value is 'least_squares'
        :type min_method: str, optional

        :param profile: Profile type for the fitting. The default value ``g-emi`` (Gaussian-emission).
        :type profile: str, optional

        :param cont_from_bands: Check for continuum calculation from adjacent bands. The default value is True.
        :type cont_from_bands: bool, optional

        :param temp: Transition electron temperature for thermal broadening calculation. The default value is 10000K.
        :type temp: bool, optional

        :param default_conf_prefix: Label for the default configuration section in the ```fit_conf`` variable.
        :type default_conf_prefix: str, optional

        :param id_conf_prefix: Label for the object configuration section in the ```fit_conf`` variable.
        :type id_conf_prefix: str, optional

        """

        # Make a copy of the fitting configuartion
        # fit_conf = {} if fit_conf is None else fit_conf.copy()
        input_conf = check_fit_conf(fit_conf, default_conf_prefix, id_conf_prefix)

        # Interpret the input line
        self.line = Line(label, bands, input_conf, profile, cont_from_bands)

        # Check if the line location is provided
        bands_integrity = check_spectrum_bands(self.line, self._spec.wave_rest)

        if bands_integrity:

            # Get the bands regions
            idcsEmis, idcsCont = self.line.index_bands(self._spec.wave, self._spec.redshift)

            emisWave, emisFlux = self._spec.wave[idcsEmis], self._spec.flux[idcsEmis]
            emisErr = None if self._spec.err_flux is None else self._spec.err_flux[idcsEmis]

            contWave, contFlux = self._spec.wave[idcsCont], self._spec.flux[idcsCont]
            contErr = None if self._spec.err_flux is None else self._spec.err_flux[idcsCont]

            # Check the bands size
            proceed = review_bands(self.line, emisWave, contWave, emisFlux, contFlux)

            if proceed:

                # Default line type is in emission unless all are absorption
                emission_check = False if np.all(~self.line._p_type) else True

                # Non-parametric measurements
                self.integrated_properties(self.line, emisWave, emisFlux, emisErr, contWave, contFlux, contErr, emission_check)

                # Import kinematics if requested
                import_line_kinematics(self.line, 1 + self._spec.redshift, self._spec.frame, self._spec.units_wave, input_conf)

                # Combine bands
                idcsLine = idcsEmis + idcsCont
                x_array, y_array = self._spec.wave[idcsLine], self._spec.flux[idcsLine]
                emisErr = None if self._spec.err_flux is None else self._spec.err_flux[idcsLine]

                # Gaussian fitting
                self.profile_fitting(self.line, x_array, y_array, emisErr, self._spec.redshift, input_conf, min_method)

                # Instrumental and thermal corrections for the lines
                sigma_corrections(self.line, idcsEmis, emisWave, self._spec.res_power, temp)

                # Recalculate the SNR with the gaussian parameters
                err_cont = self.line.cont_err if self._spec.err_flux is None else np.mean(self._spec.err_flux[idcsEmis])
                self.line.snr_line = signal_to_noise_rola(self.line.amp, err_cont, self.line.n_pixels)

                # Save the line parameters to the dataframe
                results_to_log(self.line, self._spec.frame, self._spec.norm_flux)

        return

    def frame(self, bands, fit_conf=None, min_method='least_squares', profile='g-emi', cont_from_bands=True,
              temp=10000.0, line_list=None, default_conf_prefix='default', id_conf_prefix=None, line_detection=False,
              plot_fit=False, progress_output='bar'):

        """

        This function measures multiple lines on the spectrum object from a bands dataframe.

        The input ``bands_df`` can be a pandas.Dataframe or a link to its file.

        The argument ``fit_conf`` provides the `profile-fitting configuration <https://lime-stable.readthedocs.io/en/latest/inputs/n_inputs4_fit_configuration.html>`_.

        The ``min_method`` argument provides the minimization algorithm for the `LmFit functions
        <https://lmfit.github.io/lmfit-py/fitting.html#lmfit.minimizer.Minimizer.minimize>`_.

        By default, the profile fitting assumes an emission Gaussian shape, with ``profile="g-emi"``. The profile keywords
        are described on the `label documentation <https://lime-stable.readthedocs.io/en/latest/inputs/n_inputs2_line_labels.html>`_

        The ``cont_from_bands=True`` argument forces the continuum to be measured from the adjacent line bands. If
        ``cont_from_bands=False`` the continuum gradient is calculated from the first and last pixel from the line band
        (w3-w4).

        For the calculation of the thermal broadening on the emission lines the user can include the line electron
        temperature in Kelvin. The default value ``temp`` is 10000 K.

        The user can limit the fitting to certain bands with the ``lines_list`` argument.

        If the input ``fit_conf`` has multiple sections, this function will read the parameters from the ``default_conf_key``
        argument, whose default value is "default". If the input dictionary also has a section title with the
        ``id_conf_label`` _line_fitting the ``default_conf_key`` _line_fitting parameters will be **updated** by the
        object configuration.

        If ``line_detection=True`` the input ``bands_df`` measurements will be limited to those bands with a line detection.
        The local configuration for the line detection algorithm can be provided from the fit_conf entries.

        If ``plot_fit=True`` this function will plot profile after each fitting.

        The ``progress_output`` argument determines the progress console message. A "bar" value will show a progress bar,
        while a "counter" value will print a message with the current line being measured. Finally, a None value will not
        show any message.

        :param bands: Bands dataframe (or file address to the dataframe).
        :type bands: pandas.Dataframe, str, path.Pathlib

        :param fit_conf: Fitting configuration.
        :type fit_conf: dict, optional

        :param min_method: `Minimization algorithm <https://lmfit.github.io/lmfit-py/fitting.html#lmfit.minimizer.Minimizer.minimize>`_.
                            The default value is 'least_squares'
        :type min_method: str, optional

        :param profile: Profile type for the fitting. The default value ``g-emi`` (Gaussian-emission).
        :type profile: str, optional

        :param cont_from_bands: Check for continuum calculation from adjacent bands. The default value is True.
        :type cont_from_bands: bool, optional

        :param temp: Transition electron temperature for thermal broadening calculation. The default value is 10000K.
        :type temp: bool, optional

        :param line_list: Line list to measure from the bands dataframe.
        :type line_list: list, optional

        :param default_conf_prefix: Label for the default configuration section in the ```fit_conf`` variable.
        :type default_conf_prefix: str, optional

        :param id_conf_prefix: Label for the object configuration section in the ```fit_conf`` variable.
        :type id_conf_prefix: str, optional

        :param line_detection: Set to True to run the dectection line algorithm prior to line measurements.
        :type line_detection: bool, optional

        :param plot_fit: Set to True to plot the profile fitting at each iteration.
        :type plot_fit: bool, optional

        :param progress_output: Progress message output. The options are "bar" (default), "counter" and "None".
        :type progress_output: str, optional

        """

        # Check if the lines log is a dataframe or a file address
        bands = check_file_dataframe(bands, pd.DataFrame)

        if bands is not None:

            # Crop the analysis to the target lines
            if line_list is not None:
                idcs = bands.index.isin(line_list)
                bands = bands.loc[idcs]

            # Load configuration
            input_conf = check_fit_conf(fit_conf, default_conf_prefix, id_conf_prefix, line_detection=line_detection)

            # Line detection if requested
            if line_detection:

                # Review the configuration entries
                cont_fit_conf = input_conf.get('continuum', {})
                detect_conf = input_conf.get('line_detection', {})

                # Perform the line detection
                self._spec.fit.continuum(**cont_fit_conf)
                bands = self._spec.line_detection(bands, **detect_conf)

            # Define lines to treat through the lines
            label_list = bands.index.to_numpy()
            self._n_lines = label_list.size

            # Loop through the lines
            if self._n_lines > 0:

                # On screen progress bar
                pbar = ProgressBar(progress_output, f'{self._n_lines} lines')
                if progress_output is not None:
                    print(f'\nLine fitting progress:')

                for self._i_line in np.arange(self._n_lines):

                    # Ignore line if part of a blended/merge group
                    line = label_list[self._i_line]
                    measure_check = check_compound_line_exclusion(line, bands)

                    if measure_check:

                        # Progress message
                        pbar.output_message(self._i_line, self._n_lines, pre_text="", post_text=f'({line})')

                        # Fit the lines
                        self.bands(line, bands, input_conf, min_method, profile, cont_from_bands, temp,
                                   id_conf_prefix=None, default_conf_prefix=None)

                        if plot_fit:
                            self._spec.plot.bands()

            else:
                msg = f'No lines were measured from the input dataframe:\n - line_list: {line_list}\n - line_detection: {line_detection}'
                _logger.debug(msg)

        else:
            _logger.info(f'Not input dataframe. Lines were not measured')

        return

    def continuum(self, degree_list, emis_threshold, abs_threshold=None, smooth_length=None, plot_steps=False):

        """

        This function fits the spectrum continuum in an iterative process. The user specifies two parameters: the ``degree_list``
        for the fitted polynomial and the ``threshold_list``` for the multiplicative standard deviation factor. At each
        interation points beyond this flux threshold are excluded from the continuum current fittings. Consequently,
        the user should aim towards more constrictive parameter values at each iteration.

        The user can specify a window length over which the spectrum will be smoothed before fitting the continuum using
        the ``smooth_length`` parameter.

        The user can visually inspect the fitting output graphically setting the parameter ``plot_steps=True``.

        :param degree_list: Integer list with the degree of the continuum polynomial
        :type degree_list: list

        :param emis_threshold: Float list for the multiplicative continuum standard deviation flux factor
        :type emis_threshold: list

        :param smooth_length: Size of the smoothing window to convolve the spectrum. The default value is None.
        :type smooth_length: integer, optional

        :param plot_steps: Set to "True" to plot the fitted continuum at each iteration.
        :type plot_steps: bool, optional

        :return:

        """

        # Create a pre-Mask based on the original mask if available # TODO np.ma.is_masked es malo quitalo
        if np.ma.isMaskedArray(self._spec.flux):
            mask_cont = ~self._spec.flux.mask
            input_wave, input_flux = self._spec.wave.data, self._spec.flux.data
        else:
            mask_cont = np.ones(self._spec.flux.size).astype(bool)
            input_wave, input_flux = self._spec.wave, self._spec.flux

        # Smooth the spectrum
        if smooth_length is not None:
            smoothing_window = np.ones(smooth_length) / smooth_length
            input_flux_s = np.convolve(input_flux, smoothing_window, mode='same')
        else:
            input_flux_s = input_flux

        # Loop through the fitting degree
        abs_threshold = emis_threshold if abs_threshold is None else abs_threshold
        for i, degree in enumerate(degree_list):

            # First iteration use percentile limits for an initial fit
            if i == 0:
                low_lim, high_lim = np.nanpercentile(input_flux_s[mask_cont], (16, 84))
                mask_cont_0 = mask_cont & (input_flux_s >= low_lim) & (input_flux_s <= high_lim)
                cont_fit = continuum_model_fit(input_wave, input_flux_s, mask_cont_0, degree)

            # Establishing the flux limits
            std_flux = np.nanstd((input_flux_s - cont_fit)[mask_cont])
            low_lim, high_lim = cont_fit - abs_threshold[i] * std_flux, cont_fit + emis_threshold[i] * std_flux

            # Add new entries to the mask
            mask_cont = mask_cont & (input_flux_s >= low_lim) & (input_flux_s <= high_lim)

            # Fit continuum
            cont_fit = continuum_model_fit(input_wave, input_flux_s, mask_cont, degree)

            # Compute the continuum and assign replace the value outside the bands the new continuum
            if plot_steps:
                ax_cfg = {'title':f'Continuum fitting, iteration ({i+1}/{len(degree_list)})'}
                self._spec.plot._continuum_iteration(input_wave, input_flux, cont_fit, input_flux_s, mask_cont, low_lim,
                                                     high_lim, emis_threshold[i], ax_cfg)

        # Include the standard deviation of the spectrum for the unmasked pixels
        self._spec.cont = cont_fit if not np.ma.isMaskedArray(self._spec.flux) else np.ma.masked_array(cont_fit,
                                                                                                       self._spec.flux.mask)
        self._spec.cont_std = np.std((input_flux_s - cont_fit)[mask_cont])

        return

    def redshift(self, bands, z_nsteps=10000, z_min=0, z_max=10, only_detection=True, sig_digits=2, plot_results=False):

        if not aspect_check:
            _logger.info("ASPECT has not been installed the redshift measurements won't be constrained to line features")

        # Get the features array
        if aspect_check:
            if self._spec.features.pred_arr is None:
                _logger.warning("The observations does not have a components dectection array please run ASPECT")
                pred_arr, conf_arr = None, None
            else:
                pred_arr, conf_arr = self._spec.features.pred_arr, self._spec.features.conf_arr

        # Use the dectection bands if provided
        if pred_arr is not None:
            idcs_lines = (pred_arr == 3) | (pred_arr == 7) #TODO add "emission" and "doublet" number checks
        else:
            idcs_lines = None

        # Decide if proceed
        if only_detection is False:
            measure_check = True
            idcs_lines = np.ones(idcs_lines.shape).astype(bool)
        else:
            measure_check = True if np.any(idcs_lines) else False

        # Continue with measurement
        if measure_check:

            # Extract the data
            pixel_mask = None if not np.ma.isMaskedArray(self._spec.flux) else self._spec.flux.mask
            wave_arr = self._spec.wave.data if pixel_mask is not None else self._spec.wave
            flux_arr = self._spec.flux.data if pixel_mask is not None else self._spec.flux
            data_mask = ~pixel_mask

            # Compute the resolution params
            deltalamb_arr = np.diff(wave_arr)
            R_arr = wave_arr[1:] / deltalamb_arr
            FWHM_arr = wave_arr[1:] / R_arr
            sigma_arr = np.zeros(wave_arr.size)
            sigma_arr[:-1] = FWHM_arr / (2 * np.sqrt(2 * np.log(2))) * 1.5
            sigma_arr[-1] = sigma_arr[-2]

            # Lines selection
            theo_lambda = bands.wavelength.to_numpy()

            # Parameters for the brute analysis
            z_arr = np.linspace(z_min, z_max, z_nsteps)
            wave_matrix = np.tile(wave_arr, (theo_lambda.size, 1))
            flux_sum = np.zeros(z_arr.size)

            # Combine line and pixel_mask
            mask = data_mask & idcs_lines

            # Loop throught the redshift steps
            if not np.all(~mask):
                for i, z_i in enumerate(z_arr):

                    # Generate the redshift key
                    gauss_arr = compute_z_key(z_i, theo_lambda, wave_matrix, 1, sigma_arr)

                    # Compute flux cumulative sum
                    flux_sum[i] = 0 if gauss_arr is None else np.sum(flux_arr[mask] * gauss_arr[mask])

                z_infer = np.round(z_arr[np.argmax(flux_sum)], decimals=sig_digits)

            # No lines or all masked
            else:
                z_infer = None

            if plot_results and (z_infer is not None):
                gauss_arr_max = compute_z_key(z_infer, theo_lambda, wave_matrix, 1, sigma_arr)
                redshift_fit_evaluation(self._spec, z_infer, mask, gauss_arr_max, z_arr, flux_sum)

        # Do not attempt measurement
        else:
            z_infer = None

        return z_infer


class CubeTreatment(LineFitting):

    def __init__(self, cube):

        # Instantiate the dependencies
        LineFitting.__init__(self)

        # Lime spectrum object with the scientific data
        self._cube = cube
        self._spec = None

    def spatial_mask(self, mask_file, output_address, bands=None, fit_conf=None, mask_list=None, line_list=None,
                     log_ext_suffix='_LINELOG', min_method='least_squares', profile='g-emi', cont_from_bands=True,
                     temp=10000.0, default_conf_prefix='default', line_detection=False, progress_output='bar',
                     plot_fit=False, header=None, join_output_files=True):

        """

        This function measures lines on an IFS cube from an input binary spatial ``mask_file``.

        The results are stored in a multipage ".fits" file, each page contains a measurements and it is named after the
        spatial array coordinates and the ``log_ext_suffix`` (i.e. "idx_j-idx_i_LINELOG")

        The input ``bands`` can be a pandas.Dataframe or an address to the file. The user can specify one bands file
        per mask page on the ``mask_file``. To do this, the ``fit_conf`` argument must include a section for every mask
        on the ``mask_list`` (i.e. "Mask1_line_fitting"). This function will check for a key "bands" and load the
        corresponding bands.

        The fitting configuration in the ``fit_conf`` argument accepts a three-level configuration. At the lowest level,
        The ``default_conf_key`` points towards the default configuration for all the spaxels analyzed on the cube
        (i.e. "default_line_fitting"). At an intermediate level, the parameters from the section with a name from the
        ``mask_list`` (i.e. "Mask1_line_fitting") will be applied to the spaxels in the corresponding mask. Finally, at
        the highest level, the user can provide a spaxel fitting configuration with the spatial array coordiantes
        "50-28_LINELOG". In all these cases the higher level configurate **updates** the lower levels (only common entries
        are replaced)

        .. attention::
            In this multi-level configuration design, the higher level entries **update** the lower level entries:
            only shared entries are overwritten, the final configuration will include all the entries from the
            default mask and spaxel sections.

        If the ``line_detection`` is set to True the function proceeds to run the line detection algorithm prior to the
        fitting of the lines. The user provide the configuration parameters for the line_detection function in the
        ``fit_conf`` argument. At the default, mask or spaxel configuration the user needs to specify these entries with
        the "function name" + "." + "function argument" (i.e. "line_detection.emission_type='emission'"). The multi-level
        configuration described above will be applied to this function parameters as well.

        .. note::
            The parameters for the ``line.detection`` can be found on the documentation. The user doesn't need to specify
            a "lime_detection.bands" parameter. The input bands from the corresponding mask will be used.

        :param mask_file: Address of binary spatial mask file
        :type mask_file: str, pathlib.Path

        :param output_address: File address for the output measurements log.
        :type output_address: str, pathlib.Path

        :param bands: Bands dataframe (or file address to the dataframe).
        :type bands: pandas.Dataframe, str, path.Pathlib

        :param fit_conf: Fitting configuration.
        :type fit_conf: dict, optional

        :param mask_list: Masks name list to explore on the ``masks_file``.
        :type mask_list: list, optional

        :param line_list: Line list to measure from the bands dataframe.
        :type line_list: list, optional

        :param log_ext_suffix: Suffix for the measurements log pages. The default value is "_LINELOG".
        :type log_ext_suffix: str, optional.

        :param min_method: `Minimization algorithm <https://lmfit.github.io/lmfit-py/fitting.html#lmfit.minimizer.Minimizer.minimize>`_.
                            The default value is 'least_squares'
        :type min_method: str, optional

        :param profile: Profile type for the fitting. The default value ``g-emi`` (Gaussian-emission).
        :type profile: str, optional

        :param cont_from_bands: Check for continuum calculation from adjacent bands. The default value is True.
        :type cont_from_bands: bool, optional

        :param temp: Transition electron temperature for thermal broadening calculation. The default value is 10000K.
        :type temp: bool, optional

        :param default_conf_prefix: Label for the default configuration section in the ```fit_conf`` variable.
        :type default_conf_prefix: str, optional

        :param line_detection: Set to True to run the dectection line algorithm prior to line measurements.
        :type line_detection: bool, optional

        :param plot_fit: Set to True to plot the spectrum lines fitting at each iteration.
        :type plot_fit: bool, optional

        :param progress_output: Progress message output. The options are "bar" (default), "counter" and "None".
        :type progress_output: str, optional

        :param header: Dictionary for parameter ".fits" file headers.
        :type header: dict, optional

        :param join_output_files: In the case of multiple masks, join the individual output ".fits" files into a single
                                  one. If set to False there will be one output file named per mask named after it. The
                                  default value is True.
        :type join_output_files: bool, optional

        """
        if bands is not None:
            bands = check_file_dataframe(bands, pd.DataFrame)

        # Check if the mask variable is a file or an array
        mask_maps = check_file_array_mask(mask_file, mask_list)
        mask_list = np.array(list(mask_maps.keys()))
        mask_data_list = list(mask_maps.values())

        # Check the mask configuration is included if there are no masks
        input_masks = mask_list if bands is None else None
        input_conf = check_fit_conf(fit_conf, default_key=None, group_key=None, group_list=input_masks)

        # Check we are not missing bands
        # check_cube_bands(bands, mask_list, fit_conf)

        # Check if the output log folder exists
        output_address = Path(output_address)
        address_dir = output_address.parent
        if not address_dir.is_dir():
            raise LiMe_Error(f'The folder of the output log file does not exist at {output_address}')
        address_stem = output_address.stem

        # Determine the spaxels to treat at each mask
        spax_counter, total_spaxels, spaxels_dict = 0, 0, {}
        for idx_mask, mask_data in enumerate(mask_data_list):
            spa_mask, hdr_mask = mask_data
            idcs_spaxels = np.argwhere(spa_mask)

            total_spaxels += len(idcs_spaxels)
            spaxels_dict[idx_mask] = idcs_spaxels

        # Header data
        hdr_coords = extract_wcs_header(self._cube.wcs, drop_axis='spectral') if self._cube.wcs is not None else None

        # Loop through the masks
        n_masks = len(mask_list)
        mask_log_files_list = [address_dir/f'{address_stem}_MASK-{mask_name}.fits' for mask_name in mask_list]

        for i in np.arange(n_masks):

            # HDU_container
            hdul_log = fits.HDUList([fits.PrimaryHDU()])

            # Mask progress indexing
            mask_name = mask_list[i]
            mask_hdr = mask_data_list[i][1]
            idcs_spaxels = spaxels_dict[i]

            # Recover the fitting configuration
            mask_conf = check_fit_conf(input_conf, default_conf_prefix, mask_name)

            # Load the mask log if provided
            if bands is None:
                bands_file = mask_conf['bands']
                bands_path = Path(bands_file).absolute() if bands_file[0] == '.' else Path(bands_file)
                bands_in = load_frame(bands_path)
            else:
                bands_in = bands

            # Loop through the spaxels
            n_spaxels = idcs_spaxels.shape[0]
            n_lines, start_time = 0, time()

            print(f'\nSpatial mask {i + 1}/{n_masks}) {mask_name} ({n_spaxels} spaxels)')
            pbar = ProgressBar(progress_output, f'mask')
            for j in np.arange(n_spaxels):

                idx_j, idx_i = idcs_spaxels[j]
                spaxel_label = f'{idx_j}-{idx_i}'

                # Get the spaxel fitting configuration
                spaxel_conf = input_conf.get(f'{spaxel_label}_line_fitting')
                spaxel_conf = mask_conf if spaxel_conf is None else {**mask_conf, **spaxel_conf}

                # Spaxel progress message
                pbar.output_message(j, n_spaxels, pre_text="", post_text=f'(spaxel coordinate. {idx_j}-{idx_i})')

                # Get spaxel data
                spaxel = self._cube.get_spectrum(idx_j, idx_i, spaxel_label)

                # Fit the lines
                spaxel.fit.frame(bands_in, spaxel_conf, line_list=line_list, min_method=min_method,
                                 line_detection=line_detection, profile=profile, cont_from_bands=cont_from_bands,
                                 temp=temp, progress_output=None, plot_fit=None, id_conf_prefix=None,
                                 default_conf_prefix=None)

                # Count the number of measurements
                n_lines += spaxel.frame.index.size

                # Create page header with the default data
                hdr_i = fits.Header()

                # Add WCS information
                if hdr_coords is not None:
                    hdr_i.update(hdr_coords)

                # Add user information
                if header is not None:
                    page_hdr = header.get(f'{spaxel_label}{log_ext_suffix}', None)
                    page_hdr = header if page_hdr is None else page_hdr
                    hdr_i.update(page_hdr)

                # Save to a fits file
                linesHDU = log_to_HDU(spaxel.frame, ext_name=f'{spaxel_label}{log_ext_suffix}', header_dict=hdr_i)

                if linesHDU is not None:
                    hdul_log.append(linesHDU)

                # Plot the fittings if requested:
                if plot_fit:
                    spaxel.plot.spectrum(include_fits=True, rest_frame=True)

            # Save the log at each new mask
            hdul_log.writeto(mask_log_files_list[i], overwrite=True, output_verify='ignore')
            hdul_log.close()

            # Computation time and message
            end_time = time()
            elapsed_time = end_time - start_time
            print(f'\n{n_lines} lines measured in {elapsed_time/60:0.2f} minutes.')

        if join_output_files:
            output_comb_file = f'{address_dir/address_stem}.fits'

            # In case of only one file just rename it
            if len(mask_list) == 1:
                mask_0_path = Path(mask_log_files_list[0])
                mask_0_path.rename(Path(output_comb_file))
            else:
                print(f'\nJoining spatial log files ({",".join(mask_list)}) -> {output_comb_file}')
                join_fits_files(mask_log_files_list, output_comb_file, delete_after_join=join_output_files)

        # else:
        #     # Just one mask and Join is False
        #     if len(mask_list) == 1:
        #         output_comb_file = f'{address_dir / address_stem}.fits'
        #         mask_0_path = Path(mask_log_files_list[0])
        #         mask_0_path.rename(Path(output_comb_file))


        return


