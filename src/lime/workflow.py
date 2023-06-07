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
from .io import check_file_dataframe, load_spatial_mask, log_to_HDU, results_to_log, load_log, extract_wcs_header
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

    def band(self, label, band_edges=None, fit_conf=None, fit_method='least_squares', profile='g-emi',
             cont_from_bands=True, temp=10000.0):

        """

         This function fits a line given its line and spectral mask. The line notation consists in the transition
         particle and wavelength (with units) separated by an underscore, i.e. O3_5007A.

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

         :param band_edges: 6 wavelengths spectral mask with the blue continuum, line and red continuum bands.
         :type band_edges: numpy.ndarray

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

        # Make a copy of the fitting configuartion
        fit_conf = {} if fit_conf is None else fit_conf.copy()

        # Interpret the input line
        self.line = Line(label, band_edges, fit_conf, profile, cont_from_bands)

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
        self.profile_fitting(self.line, x_array, y_array, emisErr, self._spec.redshift, fit_conf, fit_method, temp,
                             self._spec.inst_FWHM)

        # Recalculate the SNR with the gaussian parameters
        err_cont = self.line.std_cont if self._spec.err_flux is None else np.mean(self._spec.err_flux[idcsEmis])
        self.line.snr_line = signal_to_noise_rola(self.line.amp, err_cont, self.line.n_pixels)

        # Save the line parameters to the dataframe
        results_to_log(self.line, self._spec.log, self._spec.norm_flux, self._spec.units_wave)

        return

    def frame(self, bands_df, fit_conf=None, lines_list=None, fit_method='least_squares', line_detection=False,
              profile='g-emi', cont_from_bands=True, temp=10000.0, plot_fit=False,
              obj_ref=None, default_conf_key='default_line_fitting', user_detect_conf=None, progress_output='bar'):

        # Check if the lines log is a dataframe or a file address
        bands_df = check_file_dataframe(bands_df, pd.DataFrame)

        if bands_df is not None:

            # Crop the analysis to the target lines
            if lines_list is not None:
                idcs = bands_df.index.isin(lines_list)
                bands_df = bands_df.loc[idcs]

            # Load configuration for fits
            if fit_conf is not None:

                # Just pass the input conf
                if default_conf_key is None:
                    input_conf = {**fit_conf}

                # Update the configuration
                else:
                    input_conf = fit_conf.get(default_conf_key, {**fit_conf})

                    # User configuration overwrite default
                    if obj_ref is not None:  # User configuration overwrite default
                        input_conf = {**input_conf, **fit_conf.get(f'{obj_ref}_line_fitting', {})}
            else:
                input_conf = {}

            # Line detection if requested
            if line_detection:

                # User configuration overwrites default
                if user_detect_conf is not None:
                    detect_conf = {**LINE_DETECT_PARAMS, **user_detect_conf}
                else:
                    detect_conf = LINE_DETECT_PARAMS.copy()
                detect_conf['input_log'] = bands_df

                # Perform de line detection
                bands_df = self._spec.line_detection(**detect_conf)

            # Loop through the lines
            label_list = bands_df.index.to_numpy()
            bands_matrix = bands_df.loc[:, 'w1':'w6'].to_numpy()
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
                    self.band(line, bands_matrix[i], input_conf, fit_method, profile, cont_from_bands, temp)

                    if plot_fit:
                        self._spec.plot.band()

            else:
                msg = f'No lines were measured from the input dataframe:\n - line_list: {lines_list}\n - line_detection: {line_detection}'
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

    def spatial_mask(self, spatial_mask, bands_frame=None, fit_conf=None, masks_list=None, lines_list=None,
                     fit_method='least_squares', line_detection=False, profile='g-emi', cont_from_bands=True,
                     temp=10000.0, output_log=None, default_conf_key='default_line_fitting', log_ext_suffix='_LINESLOG',
                     progress_output='bar', plot_fit=False, n_save=100, user_hdr=None):

        # Check if the mask variable is a file or an array
        mask_dict = check_file_array_mask(spatial_mask, masks_list)

        # Unpack mask dictionary
        mask_list = np.array(list(mask_dict.keys()))
        mask_data_list = list(mask_dict.values())

        # Default fitting configuration
        fit_conf = {} if fit_conf is None else fit_conf
        default_conf = fit_conf.get(default_conf_key, {})

        # Check if the lines table is a dataframe or a file
        if bands_frame is not None:
            bands_df = check_file_dataframe(bands_frame, pd.DataFrame)

        # Check all the masks line fitting for a lines dataframe
        else:
            for mask_name in mask_list:

                error_message = 'You did not provide an input bands log for the analysis.\n' \
                                f'In this case you need to specify an "input_log=log_file_address" in the ' \
                                f'"[{mask_name}_line_detection]" of your configuration file'

                mask_conf = fit_conf.get(f'{mask_name}_line_detection', None)
                if mask_conf is not None:
                    log_address_value = mask_conf.get('input_log', None)
                    if log_address_value is None:
                        raise Error(error_message)
                else:
                    raise Error(error_message)

        # Check if output log
        if output_log is None:
            raise(Error(f'No output log file address to save the line measurements log'))
        else:
            output_log = Path(output_log)
            if not output_log.parent.is_dir():
                raise(Error(f'The folder of the output log file does not exist at {output_log}'))

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

            # Get mask line detection configuration
            if line_detection:
                detect_conf = {**LINE_DETECT_PARAMS, **fit_conf.get(f'{mask_name}_line_detection', {})}
            else:
                detect_conf = None

            # Get mask line fitting configuration
            mask_conf = {**default_conf, **fit_conf.get(f'{mask_name}_line_fitting', {})}

            # Load the mask log if provided
            if bands_frame is None:
                bands_mask_path = detect_conf['input_log']
                bands_mask_path = Path().absolute()/bands_mask_path[1:] if bands_mask_path[0] == '.' else Path(bands_mask_path)
                bands_df = load_log(bands_mask_path)

            # Loop through the spaxels
            n_spaxels = idcs_spaxels.shape[0]
            pbar = ProgressBar(progress_output, f'{n_spaxels} spaxels')
            print(f'\n\nSpatial mask {i + 1}/{n_masks}) {mask_name} ({n_spaxels} spaxels)')
            for j in np.arange(n_spaxels):

                idx_j, idx_i = idcs_spaxels[j]
                spaxel_label = f'{idx_j}-{idx_i}'

                # Get the spaxel fitting configuration
                spaxel_conf = fit_conf.get(f'{spaxel_label}_line_fitting')
                spaxel_conf = mask_conf if spaxel_conf is None else {**mask_conf, **spaxel_conf}

                # Spaxel progress message
                pbar.output_message(j, n_spaxels, pre_text="", post_text=f'Coord. {idx_j}-{idx_i}')

                # Get spaxel data
                spaxel = self._cube.get_spaxel(idx_j, idx_i, spaxel_label)

                # Fit the lines
                spaxel.fit.frame(bands_df, spaxel_conf, lines_list=lines_list, fit_method=fit_method,
                                 line_detection=line_detection, profile=profile,
                                 cont_from_bands=cont_from_bands, temp=temp, progress_output=None, plot_fit=None,
                                 obj_ref=None, default_conf_key=None, user_detect_conf=detect_conf)

                # Create page header with the default data
                hdr_i = fits.Header()

                # Add WCS information
                if hdr_coords is not None:
                    hdr_i.update(hdr_coords)

                # Add user information
                if user_hdr is not None:
                    page_hdr = user_hdr.get(f'{spaxel_label}{log_ext_suffix}', None)
                    page_hdr = user_hdr if page_hdr is None else page_hdr
                    hdr_i.update(page_hdr)

                # Save to a fits file
                linesHDU = log_to_HDU(spaxel.log, ext_name=f'{spaxel_label}{log_ext_suffix}', header_dict=hdr_i)
                hdul_log.append(linesHDU)

                # Save the data every 100 spaxels
                if spax_counter < n_save:
                    spax_counter += 1
                else:
                    spax_counter = 0
                    hdul_log.writeto(output_log, overwrite=True, output_verify='fix')

                # Plot the fittings if requested:
                if plot_fit:
                    spaxel.plot.spectrum(include_fits=True, rest_frame=True)

            # Save the log at each new mask
            hdul_log.writeto(output_log, overwrite=True, output_verify='fix')

        return


