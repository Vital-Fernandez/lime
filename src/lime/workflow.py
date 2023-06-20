import logging
import numpy as np
import pandas as pd
from pathlib import Path
from astropy.io import fits

from . import Error
from .model import LineFitting, signal_to_noise_rola
from .tools import define_masks, ProgressBar
from .recognition import LINE_DETECT_PARAMS
from .transitions import Line
from .io import check_file_dataframe, load_spatial_mask, log_to_HDU, results_to_log, load_log, extract_wcs_header, LiMe_Error
from lmfit.models import PolynomialModel

_logger = logging.getLogger('LiMe')


def check_file_array_mask(var, mask_list=None):

    # Check if file
    if isinstance(var, (str, Path)):

        input = Path(var)
        if input.is_file():
            mask_dict = load_spatial_mask(var, mask_list)
        else:
            raise Error(f'No spatial mask file at {Path(var).as_posix()}')

    # Array
    elif isinstance(var, (np.ndarray, list)):

        # Re-adjust the variable
        var = np.ndarray(var, ndmin=3)
        masks_array = np.squeeze(np.array_split(var, var.shape[0], axis=0))

        # Confirm boolean array
        if masks_array.dtype != bool:
            _logger.warning(f'The input mask array should have a boolean variables (True/False)')

        # Incase user gives a str
        mask_list = [mask_list] if isinstance(mask_list, str) else mask_list

        # Check if there is a mask list
        if len(mask_list) == 0:
            mask_list = [f'SPMASK{i}' for i in range(masks_array.shape[0])]

        # Case the number of masks names and arrays is different
        elif masks_array.shape[0] != len(mask_list):
            _logger.warning(f'The number of input spatial mask arrays is different than the number of mask names')

        # Everything is fine
        else:
            mask_list = mask_list

        # Create mask dict with empty headers
        mask_dict = dict(zip(mask_list, (masks_array, {})))

    else:

        raise Error(f'Input mask format {type(input)} is not recognized for a mask file. Please declare a fits file, a'
                    f' numpy array or a list/array of numpy arrays')

    return mask_dict


def review_bands(line, emis_wave, cont_wave, limit_narrow=7):

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

        if np.ma.is_masked(emis_wave):
            length = np.sum(~emis_wave.mask)
        else:
            length = emis_band_lengh
        _logger.warning(f'The  {line.label} band is too small ({length} length array): {emis_wave}')

    return


def import_line_kinematics(line, z_cor, log, units_wave, fit_conf):

    # Check if imported kinematics come from blended component
    if line.profile_label != 'no':
        childs_list = line.profile_label.split('+')
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


class SpecTreatment(LineFitting):

    def __init__(self, spectrum):

        # Instantiate the dependencies
        LineFitting.__init__(self)

        # Lime spectrum object with the scientific data
        self._spec = spectrum
        self.line = None

    def bands(self, label, bands=None, fit_conf=None, min_method='least_squares', profile='g-emi', cont_from_bands=True,
              temp=10000.0):

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

        """

        # Make a copy of the fitting configuartion
        fit_conf = {} if fit_conf is None else fit_conf.copy()

        # Interpret the input line
        self.line = Line(label, bands, fit_conf, profile, cont_from_bands)

        # Check if the line location is provided
        if self.line.mask is not None:

            # Get the bands regions
            idcsEmis, idcsCont = define_masks(self._spec.wave, self.line.mask * (1 + self._spec.redshift),
                                              line_mask_entry=self.line.pixel_mask)

            emisWave, emisFlux = self._spec.wave[idcsEmis], self._spec.flux[idcsEmis]
            emisErr = None if self._spec.err_flux is None else self._spec.err_flux[idcsEmis]

            contWave, contFlux = self._spec.wave[idcsCont], self._spec.flux[idcsCont]
            contErr = None if self._spec.err_flux is None else self._spec.err_flux[idcsCont]

            # Check the bands size
            review_bands(self.line, emisWave, contWave)

            # Non-parametric measurements
            self.integrated_properties(self.line, emisWave, emisFlux, emisErr, contWave, contFlux, contErr)

            # Import kinematics if requested
            import_line_kinematics(self.line, 1 + self._spec.redshift, self._spec.log, self._spec.units_wave, fit_conf)

            # Combine bands
            idcsLine = idcsEmis + idcsCont
            x_array, y_array = self._spec.wave[idcsLine], self._spec.flux[idcsLine]
            emisErr = None if self._spec.err_flux is None else self._spec.err_flux[idcsLine]

            # Gaussian fitting
            self.profile_fitting(self.line, x_array, y_array, emisErr, self._spec.redshift, fit_conf, min_method, temp,
                                 self._spec.inst_FWHM)

            # Recalculate the SNR with the gaussian parameters
            err_cont = self.line.std_cont if self._spec.err_flux is None else np.mean(self._spec.err_flux[idcsEmis])
            self.line.snr_line = signal_to_noise_rola(self.line.amp, err_cont, self.line.n_pixels)

            # Save the line parameters to the dataframe
            results_to_log(self.line, self._spec.log, self._spec.norm_flux, self._spec.units_wave)

        else:
            _logger.warning(f'The line band for transition {self.line} was not found on the input database')

        return

    def frame(self, bands, fit_conf=None, min_method='least_squares', profile='g-emi', cont_from_bands=True,
              temp=10000.0, line_list=None, default_conf_key='default', id_conf_label=None, line_detection=False,
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

        :param default_conf_key: Label for the default configuration section in the ```fit_conf`` variable.
        :type default_conf_key: str, optional

        :param id_conf_label: Label for the object configuration section in the ```fit_conf`` variable.
        :type id_conf_label: str, optional

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

            # Load configuration for fits
            if fit_conf is not None:

                # Just pass the input conf
                if default_conf_key is None:
                    input_conf = {**fit_conf}

                # Update the configuration
                else:
                    input_conf = fit_conf.get(f'{default_conf_key}_line_fitting', {**fit_conf})

                    # User configuration overwrite default
                    if id_conf_label is not None:  # User configuration overwrite default
                        input_conf = {**input_conf, **fit_conf.get(f'{id_conf_label}_line_fitting', {})}
            else:
                input_conf = {}

            # Line detection if requested
            if line_detection:

                detect_conf = fit_conf.get('line_detection', {})
                bands = self._spec.line_detection(bands, **detect_conf)

            # Loop through the lines
            label_list = bands.index.to_numpy()
            bands_matrix = bands.loc[:, 'w1':'w6'].to_numpy()
            n_lines = label_list.size

            # # Check the mask values
            # if self.mask is not None:
            #
            #     if np.any(np.isnan(self.mask)): # TODO Add this to the fitting section
            #         _logger.warning(f'The line {label} band contains nan entries: {self.mask}')
            #
            #     if not all(diff(self.mask) >= 0):
            #         _logger.info(f'The line {label} band wavelengths are not sorted: {self.mask}')
            #
            #     if not all(self.mask[2] < self.wavelength) and not all(self.wavelength < self.mask[3]):
            #         _logger.warning(f'The line {label} transition at {self.wavelength} is outside the line band wavelengths:'
            #                         f' w3 = {self.mask[2]};  w4 = {self.mask[3]}')

            # _logger.info(f'The line {self.label} has the "{modularity_label}" suffix but the transition components '
            #              f'have not been specified') # TODO move these warnings just for the fittings (not the creation)


            pbar = ProgressBar(progress_output, f'{n_lines} lines')
            if n_lines > 0:
                for i in np.arange(n_lines):

                    # Current line
                    line = label_list[i]

                    # Progress message
                    pbar.output_message(i, n_lines, pre_text="", post_text=line)

                    # Fit the lines
                    self.bands(line, bands_matrix[i], input_conf, min_method, profile, cont_from_bands, temp)

                    if plot_fit:
                        self._spec.plot.bands()

            else:
                msg = f'No lines were measured from the input dataframe:\n - line_list: {line_list}\n - line_detection: {line_detection}'
                _logger.info(msg)

        else:
            _logger.info(f'Not input dataframe. Lines were not measured')

        return

    def continuum(self, degree_list=[3, 7, 7, 7], threshold_list=[5, 3, 2, 2], plot_steps=True):

        # Check for a masked array
        if np.ma.is_masked(self._spec.flux):
            mask_cont = ~self._spec.flux.mask
            input_wave, input_flux = self._spec.wave.data, self._spec.flux.data
        else:
            mask_cont = np.ones(self._spec.flux.size).astype(bool)
            input_wave, input_flux = self._spec.wave, self._spec.flux

        # Loop through the fitting degree
        for i, degree in enumerate(degree_list):

            # Establishing the flux limits
            low_lim, high_lim = np.percentile(input_flux[mask_cont], (16, 84))
            low_lim, high_lim = low_lim / threshold_list[i], high_lim * threshold_list[i]

            # Add new entries to the mask
            mask_cont = mask_cont & (input_flux >= low_lim) & (input_flux <= high_lim)

            poly3Mod = PolynomialModel(prefix=f'poly_{degree}', degree=degree)
            poly3Params = poly3Mod.guess(input_flux[mask_cont], x=input_wave[mask_cont])

            try:
                poly3Out = poly3Mod.fit(input_flux[mask_cont], poly3Params, x=input_wave[mask_cont])
                self._spec.cont = poly3Out.eval(x=input_wave)

            except TypeError:
                _logger.warning(f'- The continuum fitting polynomial has more degrees ({degree}) than data points')
                self._spec.cont = np.full(input_wave.size, np.nan)

            # Compute the continuum and assign replace the value outside the bands the new continuum
            if plot_steps:
                title = f'Continuum fitting, iteration ({i+1}/{len(degree_list)})'
                continuum_full = poly3Out.eval(x=self._spec.wave.data)
                self._spec.plot._continuum_iteration(continuum_full, mask_cont, low_lim, high_lim, threshold_list[i], title)

        # Include the standard deviation of the spectrum for the unmasked pixels
        self._spec.cont_std = np.std((self._spec.flux - self._spec.cont)[mask_cont])

        return


class CubeTreatment(LineFitting):

    def __init__(self, cube):

        # Instantiate the dependencies
        LineFitting.__init__(self)

        # Lime spectrum object with the scientific data
        self._cube = cube
        self._spec = None

    def spatial_mask(self, mask_file, output_log_file, bands=None, fit_conf=None, mask_list=None, line_list=None,
                     log_ext_suffix='_LINESLOG', min_method='least_squares', profile='g-emi',
                     cont_from_bands=True, temp=10000.0, default_conf_key='default', line_detection=False,
                     progress_output='bar', plot_fit=False, header=None, n_save=100):

        """

        This function measures lines on an IFS cube from an input binary spatial ``mask_file``.

        The results are stored in a multipage ".fits" file, each page contains a measurements and it is named after the
        spatial array coordinates and the ``log_ext_suffix`` (i.e. "idx_j-idx_i_LINESLOG")

        The input ``bands`` can be a pandas.Dataframe or an address to the file. The user can specify one bands file
        per mask page on the ``mask_file``. To do this, the ``fit_conf`` argument must include a section for every mask
        on the ``mask_list`` (i.e. "Mask1_line_fitting"). This function will check for a key "bands" and load the
        corresponding bands.

        The fitting configuration in the ``fit_conf`` argument accepts a three-level configuration. At the lowest level,
        The ``default_conf_key`` points towards the default configuration for all the spaxels analyzed on the cube
        (i.e. "default_line_fitting"). At an intermediate level, the parameters from the section with a name from the
        ``mask_list`` (i.e. "Mask1_line_fitting") will be applied to the spaxels in the corresponding mask. Finally, at
        the highest level, the user can provide a spaxel fitting configuration with the spatial array coordiantes
        "50-28_LINESLOG". In all these cases the higher level configurate **updates** the lower levels (only common entries
        are replaced)

        .. attention::
            In this multi-level configuration design, the higher level entries **update** the lower level entries:
            only shared entries are overwritten, the final configuration will include all the entries from the
            default mask and spaxel sections.

        If the ``line_detection`` is set to True the function proceeds to run the line detection algorithm prior to the
        fitting of the lines. The user provide the configuration parameters for the line_detection function in the
        ``fit_conf`` argument. At the default, mask or spaxel configuration the user needs to specify these entries with
        the "function name" + "." + "function argument" (i.e. "line_detection.line_type='emission'"). The multi-level
        configuration described above will be applied to this function parameters as well.

        .. note::
            The parameters for the ``line.detection`` can be found on the documentation. The user doesn't need to specify
            a "lime_detection.bands" parameter. The input bands from the corresponding mask will be used.

        :param mask_file: Address of binary spatial mask file
        :type mask_file: str, pathlib.Path

        :param output_log_file: File address for the output measurements log.
        :type output_log_file: str, pathlib.Path

        :param bands: Bands dataframe (or file address to the dataframe).
        :type bands: pandas.Dataframe, str, path.Pathlib

        :param fit_conf: Fitting configuration.
        :type fit_conf: dict, optional

        :param mask_list: Masks name list to explore on the ``masks_file``.
        :type mask_list: list, optional

        :param line_list: Line list to measure from the bands dataframe.
        :type line_list: list, optional

        :param log_ext_suffix: Suffix for the measurements log pages. The default value is "_LINESLOG".
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

        :param default_conf_key: Label for the default configuration section in the ```fit_conf`` variable.
        :type default_conf_key: str, optional

        :param line_detection: Set to True to run the dectection line algorithm prior to line measurements.
        :type line_detection: bool, optional

        :param plot_fit: Set to True to plot the spectrum lines fitting at each iteration.
        :type plot_fit: bool, optional

        :param progress_output: Progress message output. The options are "bar" (default), "counter" and "None".
        :type progress_output: str, optional

        :param header: Dictionary for parameter ".fits" file headers.
        :type header: dict, optional

        :param n_save: Spectra number after which saving the measurements log. The default value is 100. 
        :type n_save: int, optional

        """

        # Check if the mask variable is a file or an array
        mask_dict = check_file_array_mask(mask_file, mask_list)

        # Unpack mask dictionary
        mask_list = np.array(list(mask_dict.keys()))
        mask_data_list = list(mask_dict.values())

        # Default fitting configuration
        fit_conf = {} if fit_conf is None else fit_conf.copy()
        default_conf = fit_conf.get(f'{default_conf_key}_line_fitting', {})

        # Check if the lines table is a dataframe or a file
        if bands is not None:
            bands = check_file_dataframe(bands, pd.DataFrame)

        # Check all the masks line fitting for a lines dataframe
        else:
            for mask_name in mask_list:

                existing_mask_check = True
                mask_conf = fit_conf.get(f'{mask_name}_line_fitting', None)

                if mask_conf is not None:
                    log_address_value = mask_conf.get('bands', None)
                    if log_address_value is None:
                        existing_mask_check = False
                else:
                    existing_mask_check = False

                if existing_mask_check is not True:
                    error_message = 'Since an input "bands" log for the analysis was not introduce in the function,\n' \
                                   f'you need to specify an "bands=log_file_address" entry the ' \
                                   f'"[{mask_name}_file]" of your configuration file'
                    raise Error(error_message)

        # Check if the output log folder exists
        output_log_file = Path(output_log_file)
        if not output_log_file.parent.is_dir():
            raise LiMe_Error(f'The folder of the output log file does not exist at {output_log_file}')

        # Determine the spaxels to treat at each mask
        total_spaxels, spaxels_dict = 0, {}
        for idx_mask, mask_data in enumerate(mask_data_list):
            spa_mask, hdr_mask = mask_data
            idcs_spaxels = np.argwhere(spa_mask)

            total_spaxels += len(idcs_spaxels)
            spaxels_dict[idx_mask] = idcs_spaxels

        # Spaxel counter to save the data everytime n_save is reached
        spax_counter = 0

        # HDU_container
        hdul_log = fits.HDUList([fits.PrimaryHDU()])

        # Header data
        if self._cube.wcs is not None:
            hdr_coords = extract_wcs_header(self._cube.wcs, drop_dispersion_axis=True)
        else:
            hdr_coords = None

        # Loop through the masks
        n_masks = len(mask_list)
        for i in np.arange(n_masks):

            # Mask progress indexing
            mask_name = mask_list[i]
            mask_hdr = mask_data_list[i][1]
            idcs_spaxels = spaxels_dict[i]

            # Get mask line fitting configuration
            mask_conf = {**default_conf, **fit_conf.get(f'{mask_name}_line_fitting', {})}
            mask_detect = mask_conf.get('line_detection', {}).copy()

            # Load the mask log if provided
            if bands is None:
                bands_file = fit_conf[f'{mask_name}_line_fitting']['bands']
                bands_path = Path(bands_file).absolute() if bands_file[0] == '.' else Path(bands_file)
                bands_in = load_log(bands_path)
            else:
                bands_in = bands

            # Loop through the spaxels
            n_spaxels = idcs_spaxels.shape[0]
            pbar = ProgressBar(progress_output, f'{n_spaxels} spaxels')
            print(f'\n\nSpatial mask {i + 1}/{n_masks}) {mask_name} ({n_spaxels} spaxels)')
            for j in np.arange(n_spaxels):

                idx_j, idx_i = idcs_spaxels[j]
                spaxel_label = f'{idx_j}-{idx_i}'

                # Get the spaxel fitting configuration
                spaxel_conf = fit_conf.get(f'{spaxel_label}_line_fitting')

                # Update the dicts functions of the spaxel conf
                if line_detection and (spaxel_conf is not None):
                    spaxel_conf['line_detection'] = {**mask_detect, **spaxel_conf.get('line_detection', {})}

                # Update mask conf with spaxel conf
                spaxel_conf = mask_conf if spaxel_conf is None else {**mask_conf, **spaxel_conf}

                # Spaxel progress message
                pbar.output_message(j, n_spaxels, pre_text="", post_text=f'Coord. {idx_j}-{idx_i}')

                # Get spaxel data
                spaxel = self._cube.get_spectrum(idx_j, idx_i, spaxel_label)

                # Fit the lines
                spaxel.fit.frame(bands_in, spaxel_conf, line_list=line_list, min_method=min_method,
                                 line_detection=line_detection, profile=profile,
                                 cont_from_bands=cont_from_bands, temp=temp, progress_output=None, plot_fit=None,
                                 id_conf_label=None, default_conf_key=None)

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
                linesHDU = log_to_HDU(spaxel.log, ext_name=f'{spaxel_label}{log_ext_suffix}', header_dict=hdr_i)
                hdul_log.append(linesHDU)

                # Save the data every 100 spaxels
                if spax_counter < n_save:
                    spax_counter += 1
                else:
                    spax_counter = 0
                    hdul_log.writeto(output_log_file, overwrite=True, output_verify='fix')

                # Plot the fittings if requested:
                if plot_fit:
                    spaxel.plot.spectrum(include_fits=True, rest_frame=True)

            # Save the log at each new mask
            hdul_log.writeto(output_log_file, overwrite=True, output_verify='fix')

        return


