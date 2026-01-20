import logging

import numpy as np
from pathlib import Path
from astropy.io import fits
from time import time
from lmfit.models import PolynomialModel

import lime
from lime.tools import ProgressBar, join_fits_files, extract_wcs_header, pd_get, au
from lime.rsrc_manager import lineDB
from lime.fitting.lines import LineFitting, signal_to_noise_rola, sigma_corrections, k_gFWHM, velocity_to_wavelength_band, profiles_computation, linear_continuum_computation
from lime.transitions import Line, lines_frame
from lime.retrieve.line_bands import determine_line_groups, groupify_lines_df
from lime.io import check_file_dataframe, check_file_array_mask, log_to_HDU, results_to_log, load_frame, LiMe_Error, check_fit_conf, lime_cfg
from lime.fitting.redshift import RedshiftFitting
from lime.plotting.plots import spec_continuum_calculation
from scipy import stats

try:
    import aspect
    aspect_check = True
except ImportError:
    aspect_check = False


_logger = logging.getLogger('LiMe')


def review_bands(spec, line, min_line_pixels=3, min_cont_pixels=2, user_cont_source='central', user_err_from_bands=False):

    # Check if the line bands are provided
    if line.mask is None:
        _logger.warning(f"Line {line} was not found on the input bands database. It won't be measured")
        return None

    # Check if the line is within the w3, w4 limits
    limit_blue, limit_red = spec.wave.compressed()[0], spec.wave.compressed()[-1]
    if ((line.mask[2] * (1 + spec.redshift)) < limit_blue) or ((line.mask[3] * (1 + spec.redshift)) > limit_red):
        _logger.warning(f"Line {line} bands are outside spectrum wavelengh range: w3 < w_min_rest ({line.mask[2]} < {limit_blue}) or"
                                                                               f" w4 > w_max_rest ({line.mask[3]} > {limit_red})"
                                                                               f" it won't be measured")
        return None

    # Check if the spectrum does not have the error arr but the user has requested it
    if spec.err_flux is None and user_err_from_bands is False:
        _logger.warning(f'The observation does not have an error spectrum but the fit command has requested not to use '
                        f'the adjacent bands to compute the uncertainty. Please set the "user_err_from_bands=True" to perform'
                        f' a measurement.')
        return None

    # Compute the line and adjacent continua indeces:
    idcsEmis, idcsCont = line.index_bands(spec.wave, spec.redshift)

    if user_cont_source == 'fit':
        if spec.cont is None:
            raise LiMe_Error(f"The continuum has not been fit. Please run 'Spectrum.fit.continuum' before measuring lines"
                             f"or change the 'user_cont_source'.")
            idcsCont = idcsEmis

    # Logic for the bands source
    cont_from_bands = True if user_cont_source == 'adjacent' else False

    # Check if all the flux entries are masked
    emis_flux, cont_flux = spec.flux[idcsEmis], spec.flux[idcsCont]
    if np.all(emis_flux.mask):
        _logger.warning(f"Line {line} flux is fully masked. It won't be measured")
        return None
    if np.all(cont_flux.mask) and cont_from_bands:
        _logger.warning(f"Line {line} adjacent continua flux is fully masked. It won't be measured")
        return None

    # Check if all the flux entries are zero
    if not np.any(emis_flux):
        _logger.warning(f"Line {line} flux entries are all 0. It won't be measured")
        return None
    if not np.any(cont_flux) and cont_from_bands:
        _logger.warning(f"Line {line} continuum flux entries are all 0. It won't be measured")
        return None

    # Check if the line selection is too narrow
    if np.sum(~emis_flux.mask) < min_line_pixels:
        _logger.warning(f"Line {line} has only {np.sum(~emis_flux.mask)} pixels. It won't be measured")
        return None

    # Check if the continua selection is too narrow
    if (np.sum(~cont_flux.mask) < min_cont_pixels) and cont_from_bands:
        _logger.warning(f"Line {line} continuum bands have only {np.sum(~cont_flux.mask)} pixels. It won't be measured")
        return None

    return idcsEmis, idcsCont


def import_line_kinematics(line, z_cor, log, fit_conf):

    # Check if imported kinematics come from blended component
    for idx_child, child_label in enumerate(line.list_comps):

        # Check for kinem order
        parent_label = fit_conf.get(f'{child_label}_kinem')
        if (parent_label is not None) and line.group == 'b':

            # Tied kinematics in blended profile
            if parent_label in line.list_comps:
                idx_parent = line.list_comps.index(parent_label)
                factor = f'{line.list_comps[idx_child].wavelength / line.list_comps[idx_parent].wavelength:0.8f}'
                fit_conf[f'{child_label}_center'] = {'expr': f'{factor}*{parent_label}_center'}
                fit_conf[f'{child_label}_sigma'] = {'expr': f'{factor}*{parent_label}_sigma'}

            # Import kinematics from previously measured
            elif parent_label in log.index:
                mu_parent = log.loc[parent_label, ['center', 'center_err']].to_numpy()
                sigma_parent = log.loc[parent_label, ['sigma', 'sigma_err']].to_numpy()
                wave_ratio = line.list_comps[idx_child].wavelength/log.loc[parent_label, 'wavelength']

                center_child_arr = wave_ratio * (mu_parent / z_cor)
                sigma_child_arr = wave_ratio * sigma_parent

                # Store the value on the dictionary
                fit_conf[f'{child_label}_center'] = {'value': center_child_arr[0], 'vary': False}
                fit_conf[f'{child_label}_sigma'] = {'value': sigma_child_arr[0], 'vary': False}

                # Error for the propagation
                fit_conf[f'{child_label}_center_err'] = center_child_arr[1]
                fit_conf[f'{child_label}_sigma_err'] = sigma_child_arr[1]

            # Line has not been measured before found
            else:
                _logger.info(f'\n{parent_label} has not been found on the input lines frame for {child_label} kinematics export.'
                             f'\n - Please check you are using the right line label and that the line has been measured prior '
                             f'prior to the current fitting.')

    return


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
    group_label = pd_get(lines_df, line, 'group_label', transform='none', nan_to_none=True)

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


def res_power_approx(wavelength_arr):

    delta_lambda = np.ediff1d(wavelength_arr, to_end=0)
    delta_lambda[-1] = delta_lambda[-2]

    return wavelength_arr/delta_lambda



class SpecRetriever:

    def __init__(self, spectrum):

        self._spec = spectrum

        return

    def lines_frame(self, band_vsigma=70, n_sigma=4, adjust_central_band=True, instrumental_correction=True,
                    exclude_bands_masked=True, map_band_vsigma=None, grouped_lines=None, automatic_grouping=False,
                    Rayleigh_threshold=2, fit_cfg=None, default_cfg_prefix='default', obj_cfg_prefix=None,
                    update_default=True, line_list=None, particle_list=None, sig_digits=4, ref_bands=None,
                    vacuum_waves=False, update_labels=False, update_latex=False, rejected_lines=None, components=None,
                    save_group_label=False):

        # Remove the mask from the wavelength array if necessary
        wave_intvl = self._spec.wave.compressed()

        # Check configuration format
        in_cfg = check_fit_conf(fit_cfg, default_cfg_prefix, obj_cfg_prefix, update_default) if fit_cfg else None

        # Get the parameters from configuration file if provided
        map_band_vsigma = in_cfg['map_band_vsigma'] if in_cfg and ('map_band_vsigma' in in_cfg) else map_band_vsigma
        rejected_lines = in_cfg['rejected_lines'] if in_cfg and ('rejected_lines' in in_cfg) else rejected_lines
        grouped_lines = in_cfg['grouped_lines'] if in_cfg and ('grouped_lines' in in_cfg) else grouped_lines

        # Crop the bands to match the observation
        bands = lines_frame(wave_intvl, line_list, particle_list, redshift=self._spec.redshift,
                            units_wave=self._spec.units_wave, sig_digits=sig_digits, ref_bands=ref_bands,
                            vacuum_waves=vacuum_waves, update_labels=update_labels, update_latex=update_latex,
                            rejected_lines=rejected_lines)

        # Compute the resolving power if necessary
        if self._spec.res_power is not None:
            res_power = self._spec.res_power
        else:
            res_power = res_power_approx(wave_intvl) if (instrumental_correction or fit_cfg is not None) else None

        # Adjust the middle bands to match the line width
        if adjust_central_band:

            # Expected transitions in the observed frame
            lambda_obs = bands.wavelength.to_numpy() * (1 + self._spec.redshift)

            # Add correction for the instrumental broadening
            if instrumental_correction:

                # Indexes for the lines emission peak
                idcs = np.searchsorted(wave_intvl, lambda_obs)

                # Use the instrumental resolution if available
                delta_lambda_inst = lambda_obs / (res_power[idcs] * k_gFWHM)

            # Constant velocity width
            else:
                delta_lambda_inst = 0

            # Use unique or specific velocity sigma for the bands
            if map_band_vsigma is not None:
                band_vsigma = np.full(lambda_obs.size, band_vsigma)
                for idx in bands.index.get_indexer(map_band_vsigma.keys()):
                    if idx > -1:
                        band_vsigma[idx] = map_band_vsigma[bands.index[idx]]

            # Convert to spectral width
            delta_lambda = velocity_to_wavelength_band(n_sigma, band_vsigma, lambda_obs, delta_lambda_inst)

            # Add new values to database in the rest frame
            bands['w3'] = (lambda_obs - delta_lambda) / (1 + self._spec.redshift)
            bands['w4'] = (lambda_obs + delta_lambda) / (1 + self._spec.redshift)

        # Remove from the output bands those which have all their pixels masked
        if exclude_bands_masked:
            idcs_w3_w4 = np.searchsorted(self._spec.wave.data/(1+self._spec.redshift), bands.loc[:, 'w3':'w4'])
            idcs_valid = [idx for idx, start, end in zip(bands.index, idcs_w3_w4[:, 0], idcs_w3_w4[:, 1])
                          if not np.all(self._spec.flux[start:end].mask)]
            bands = bands.loc[idcs_valid]

        # Combine the blended/merged lines in the bands table
        if in_cfg:

            # Determine the line groups
            groups_dict = determine_line_groups(self._spec, bands, in_cfg, grouped_lines, automatic_grouping, n_sigma,
                                                Rayleigh_threshold)

            # Apply the changes
            groupify_lines_df(bands, in_cfg, groups_dict, self._spec, save_group_label)

        # Filter the table to match the line detections
        if components:
            if aspect_check:
                if self._spec.infer.pred_arr is not None:

                    # Create masks for all intervals
                    starts = bands.w3.to_numpy()[:, None] * (1 + self._spec.redshift)
                    ends = bands.w4.to_numpy()[:, None] * (1 + self._spec.redshift)

                    # Check if x values fall within each interval
                    in_intervals = (self._spec.wave.data >= starts) & (self._spec.wave.data < ends)

                    # Check where y equals the target category
                    shape_indexes = [aspect.cfg['shape_number'][comp] for comp in components]
                    is_target_category = np.isin(self._spec.infer.pred_arr, shape_indexes)

                    # Combine the masks to count target_category occurrences in each interval
                    counts = np.sum(in_intervals & is_target_category, axis=1)

                    # Check which intervals satisfy the minimum count condition
                    idcs = counts >= 4
                    bands = bands.loc[idcs]

                else:
                    raise LiMe_Error(f'No components data. Please run the components detection algorithm')
            else:
                raise LiMe_Error(f'Aspect is not installed')

        return bands

    def spectrum(self, fname=None, line_label=None, ref_frame=None, split_components=False, **kwargs):

        # Headers for the default list
        headers = np.array(["wave", "flux", "err_flux", "pixel_mask"])

        # Use the observation frame if none is provided
        frame = self._spec.frame if ref_frame is None else ref_frame

        # By default report complete spectrum
        idcs = (0, None)

        # If a line is provided get indexes for the bands limits
        line_measured = False
        if line_label is not None:
            if frame is not None:
                if line_label in frame.index:
                    bands_limits = frame.loc[line_label, 'w1':'w6']
                    idcs_bands = np.searchsorted(self._spec.wave.data, bands_limits * (1 + self._spec.redshift))
                    idcs = (idcs_bands[0], idcs_bands[5])
                    line_measured = True
                else:
                    _logger.warning(f'Line {line_label} not found on observation frame')
            else:
                _logger.warning(f'No lines measured on object')

        # Compute the bands
        if line_measured:

            # Declare line object and the components and its components from the frame
            line = Line.from_transition(line_label, data_frame=frame)
            line_list = line.list_comps

            # Compute the linear components
            gaussian_arr = profiles_computation(line_list, frame, 1 + self._spec.redshift, line.profile,
                                                x_array=self._spec.wave.data[idcs[0]: idcs[1]])
            linear_arr = linear_continuum_computation(line_list, frame, 1 + self._spec.redshift, x_array=self._spec.wave.data[idcs[0]: idcs[1]])

            # Determine which component you want to extract:
            if split_components is False:
                gaussian_arr = gaussian_arr.sum(axis=1) + linear_arr[:, 0]
                gaussian_arr = gaussian_arr.reshape(-1, 1)
                line_hdrs = [line_label]
            else:
                gaussian_arr = gaussian_arr + linear_arr[:, 0][:, np.newaxis]
                line_hdrs = line_list

            # Add the line list to the headers
            headers = np.append(headers, line_hdrs)

        # Container for the data
        out_arr = np.full((self._spec.wave.data[idcs[0]: idcs[1]].size, len(headers)), np.nan)

        # Fill the array:
        out_arr[:, 0] = self._spec.wave.data[idcs[0]: idcs[1]]
        out_arr[:, 1] = self._spec.flux.data[idcs[0]: idcs[1]] * self._spec.norm_flux

        # Err array if it exists
        if self._spec.err_flux is not None:
            out_arr[:, 2] = self._spec.err_flux[idcs[0]: idcs[1]].data * self._spec.norm_flux

        # Pixel mask if any is invalid
        if np.any(self._spec.wave.mask):
            out_arr[:, 3] = self._spec.wave[idcs[0]: idcs[1]].mask

        # Add the components
        if line_measured:
            for i, line_comp in enumerate(line_hdrs):
                out_arr[:, 4 + i] = gaussian_arr[:, i]

        # Crop array if some columns are missing
        nan_columns = np.zeros(out_arr.shape[1]).astype(bool)
        nan_columns[:4] = np.all(np.isnan(out_arr[:, :4]), axis=0)
        out_arr = out_arr[:, ~nan_columns]

        # Headers
        headers = headers[~nan_columns]

        # Formatting for the data
        spec_hdrs_list = ['%.18e', '%.18e', '%.18e', '%d']
        spec_hdrs_list = spec_hdrs_list + ['%.18e'] * len(line_hdrs) if line_measured else spec_hdrs_list
        array_fmt = np.array(spec_hdrs_list)
        array_fmt = list(array_fmt[~nan_columns])

        # Update defaults with user-provided values
        default_kwargs = {"fmt": array_fmt, "delimiter": ' '}
        default_kwargs.update(kwargs)

        # Create header
        if default_kwargs.get('header') is None:
            default_kwargs['header'] = default_kwargs['delimiter'].join(headers)

        # Dictionary with parameters
        if 'footer' not in default_kwargs:
            footer_dict = {'LiMe': f"v{lime_cfg['metadata']['version']}",
                            'units_wave': self._spec.units_wave, 'units_flux':  self._spec.units_flux,
                            'redshift': self._spec.redshift, 'norm_flux': self._spec.norm_flux, 'id_label': self._spec.label}
            footer_str = "\n".join(f"{key}:{value}" for key, value in footer_dict.items())
            default_kwargs['footer'] = footer_str

        # Return a recarray with the spectrum data
        if fname is None:
            output = np.core.records.fromarrays([out_arr[:, i] for i in range(out_arr.shape[1])], names=list(headers))

        # Save to a file
        else:
            np.savetxt(fname, out_arr, **default_kwargs)
            output = None

        return output

    def rebinned(self, disp_intvl=None, pixel_width=None, pixel_number=None, constant_pixel_width=True,
                 return_spectrum=False):

        """
        Rebin the spectrum onto a new dispersion grid and return binned flux (and errors).

        Exactly **one** of ``disp_intvl``, ``pixel_width``, or ``pixel_number`` must be
        provided to define the target bins:

        - ``disp_intvl``: explicit bin **edges** (1D array, strictly increasing).
        - ``pixel_width``: constant bin width (same units as wavelength); bins span the
          full observed range ``[wave[0], wave[-1])``.
        - ``pixel_number``: group pixels by this integer factor. If ``constant_pixel_width``
          is ``True`` (default), uses an average native dispersion to build uniform-width
          bins; otherwise, takes every ``pixel_number``-th native wavelength as an edge
          (non-uniform bin widths).

        The binned flux is the mean flux per bin. If an uncertainty array is available,
        per-bin errors are combined as the square-root of the sum of variances divided
        by the number of contributing samples.

        Parameters
        ----------
        disp_intvl : array-like, optional
            Monotonic array of bin **edges** for the target dispersion grid. If provided,
            it overrides the other arguments. Values outside the native spectral range
            will trigger a warning.
        pixel_width : float, optional
            Desired constant bin width (in wavelength units). Used to generate uniform
            bin edges covering the full native wavelength range.
        pixel_number : int, optional
            Number of native pixels to aggregate per bin. Must be ``>= 2``. If a
            non-integer is given, it is rounded and a message is logged.
        constant_pixel_width : bool, optional
            When rebinned by ``pixel_number``:
            - If ``True`` (default), compute a uniform bin width as the average native
              spacing times ``pixel_number``.
            - If ``False``, construct edges by taking every ``pixel_number``-th native
              wavelength (non-uniform bins).
        return_spectrum : bool, optional
            If ``False`` (default), return the binned wavelength, flux and flux uncertainty arrays.
            If ``True``, return a new :class:`lime.Spectrum` instance containing the binned data.

        Returns
        -------
        disp_centers : numpy.ndarray
            Binned wavelength array (same units as disp_intvl if provided).
        flux_binned : numpy.ndarray
            Binned flux array.
        err_binned : numpy.ndarray or None
            Binned flux uncertainty array.
        spectrum : lime.Spectrum, optional
            Returned only if ``return_spectrum=True``. A new spectrum object with the binned data.

        Raises
        ------
        ValueError
            If none or more than one of ``disp_intvl``, ``pixel_width``, ``pixel_number``
            are provided, or if ``pixel_number < 2``.

        Notes
        -----
        - ``disp_intvl`` is interpreted as **edges** (length = ``nbins + 1``); the
          returned ``disp_centers`` are computed as edge midpoints.
        - Binning uses ``scipy.stats.binned_statistic(..., statistic='mean')`` for flux.
        - If the spectrum contains masked or NaN values, ensure they are appropriately
          handled upstream; NaNs within a bin will propagate to the mean.
        """

        # Check only one approach is necessary
        provided = {"dispersion inteval": disp_intvl, "pixel width": pixel_width, "pixel number": pixel_number}
        active = [name for name, val in provided.items() if val is not None]

        # Enforce only one rebinning array
        if len(active) == 0:
            raise ValueError("No arguments were provided to compute the new spectral resolution")

        if len(active) > 1:
            raise ValueError(f"Arguments {active} are mutually exclusive. Please only one.")

        # Extract the spectrum data
        mask_arr = self._spec.wave.mask
        wave_arr = self._spec.wave.data
        flux_arr = self._spec.flux.data

        # Check the input disespersion interval
        if disp_intvl is not None:

            # Limit warnings
            if disp_intvl[0] < wave_arr[0]:
                _logger.warning(f'The input lower dispersion value is below the spectral range: disp_intvl {disp_intvl[0]} < {wave_arr[0]}')
            if disp_intvl[-1] > wave_arr[-1]:
                _logger.warning(f'The input higher dispersion value is above the spectral range: disp_intvl {disp_intvl[-1]} > {wave_arr[-1]}')
            bin_width = np.nanmean(np.diff(disp_intvl))

        else:

            # Compute the wavelength range based on the pixel width
            if pixel_width is not None:
                disp_intvl = np.arange(wave_arr[0], wave_arr[-1], pixel_width)
                bin_width = pixel_width

            # Compute the wavelength range based on a number of pixels
            else:

                # Make sure the inputs make sense good values
                if pixel_number is not None and pixel_number >= 2:
                    if not float(pixel_number).is_integer():
                        _logger.info(f'The input pixel number has been rounded from {pixel_number} to {round(pixel_number)}')
                    pixel_number = round(pixel_number)
                else:
                    raise ValueError(f" In the number of pixels rebinning you need the input value must be above 1.")

                # Calculate the difference:
                if constant_pixel_width:
                    bin_width = np.nanmean(np.diff(wave_arr)) * pixel_number
                    disp_intvl = np.arange(wave_arr[0], wave_arr[-1], bin_width)
                else:
                    disp_intvl = wave_arr[pixel_number::pixel_number]
                    bin_width = disp_intvl[-1] - disp_intvl[-2]

        # Make the binning calculation
        flux_binned, edges, binnumber = stats.binned_statistic(wave_arr, flux_arr, statistic='mean', bins=disp_intvl)
        disp_intvl = disp_intvl[:-1] + bin_width/2

        if self._spec.err_flux is not None:
            err_arr = self._spec.err_flux.data
            sum_sq_errors = np.bincount(binnumber, weights=err_arr ** 2)
            bin_counts = np.bincount(binnumber)
            N_bins = flux_binned.size
            sum_sq_errors_filtered = sum_sq_errors[1: N_bins + 1]
            bin_counts_filtered = bin_counts[1: N_bins + 1]
            err_binned = np.sqrt(sum_sq_errors_filtered) / bin_counts_filtered
        else:
            err_binned = None

        if not return_spectrum:
            return disp_intvl, flux_binned, err_binned

        else:
            return lime.Spectrum(input_wave=disp_intvl,
                                 input_flux=flux_binned * self._spec.norm_flux if self._spec.norm_flux else flux_binned,
                                 input_err=err_binned * self._spec.norm_flux if self._spec.norm_flux else err_binned,
                                 redshift=self._spec.redshift, res_power=self._spec.res_power,
                                 units_wave=self._spec.units_wave, units_flux=self._spec.units_flux,
                                 norm_flux=self._spec.norm_flux)

    def normalization(self, return_spectrum=False, **kwargs):

        """
        Normalize the spectrum by its continuum.

        If the continuum has not been previously computed (``self._spec.cont is None``),
        this method automatically fits it by calling
        ``self._spec.fit.continuum(**kwargs)`` before performing the normalization.

        The normalized flux is defined as ``flux / continuum``. If a flux uncertainty
        array is available, the normalized uncertainty is propagated assuming
        independent errors in the flux and continuum.

        Parameters
        ----------
        return_spectrum : bool, optional
            If ``False`` (default), return the normalized flux and uncertainty arrays.
            If ``True``, return a new :class:`lime.Spectrum` instance containing the
            normalized data.
        **kwargs
            Additional keyword arguments passed directly to
            ``lime.Spectrum.fit.continuum`` when the continuum needs to be computed.

        Returns
        -------
        flux_norm : astropy.units.Quantity
            Continuum-normalized flux array.
        err_norm : astropy.units.Quantity or bool
            Uncertainty of the normalized flux. If the original spectrum does not
            contain a flux uncertainty array, this returns ``False``.
        spectrum : lime.Spectrum, optional
            Returned only if ``return_spectrum=True``. A new spectrum object with
            dimensionless flux units and ``norm_flux=1``.

        Notes
        -----
        - The uncertainty propagation follows:

          ``σ(F/C) = |F/C| * sqrt[(σ_F / F)^2 + (σ_C / C)^2]``

          where ``F`` is the flux and ``C`` is the continuum.
        - The returned spectrum preserves the original wavelength grid, redshift,
          spectral resolution, and pixel mask.
        - The continuum is computed only once and cached in the parent spectrum.
        """

        # Compute the object continuum if not provided
        if self._spec.cont is None:
            self._spec.fit.continuum(**kwargs)

        # Normalize the flux
        flux_norm = self._spec.flux / self._spec.cont

        # Normalize the flux uncertainty
        if self._spec.err_flux is not None:
            err_norm = np.abs(flux_norm) * np.sqrt((self._spec.err_flux / self._spec.flux) ** 2 +
                                                   (self._spec.cont_std / self._spec.cont) ** 2)
        else:
            err_norm = False

        # Return the normalized flux and uncertainty array
        if not return_spectrum:
            return flux_norm, err_norm

        # Return a LiMe spectrum
        else:
            return lime.Spectrum(self._spec.wave.data, flux_norm.data, err_norm.data, redshift=self._spec.redshift,
                                 units_wave=self._spec.units_wave, units_flux=au.dimensionless_unscaled, norm_flux=1,
                                 res_power=self._spec.res_power, pixel_mask=flux_norm.mask)

    def upper_line_limit(self, line, bands=None, signal_to_noise=8, err_from_bands=False, continua_sigma=True):

        # Use frame if available
        bands = self._spec.frame if bands is None else check_file_dataframe(bands)

        # Check if line is avaible to calculations
        if (bands is None) or (line not in bands.index):
            raise LiMe_Error('Upper flux limit requires the line band. Please input the lines frame or include in the '
                             'spec.lines_frame measurements' if bands is None else f'Line "{line}" not found in the input bands')

        line = Line.from_transition(line, data_frame=bands)
        idcs_central, idcs_continua = line.index_bands(self._spec.wave, self._spec.redshift)
        lambda_central = line.wavelength * (1 + self._spec.redshift)

        if err_from_bands:
            sigma_cont = np.mean(self._spec.err_flux[idcs_central]) * self._spec.norm_flux
        else:
            sigma_cont = line.measurements.cont_err

        print(line.measurements.intg_flux)
        print(line.measurements.profile_flux)

        delta_lambda = np.mean(np.diff(self._spec.wave[idcs_central]))

        R_line = np.mean(self._spec.res_power[idcs_central])

        g_const = np.sqrt(np.pi/(2*np.log(2)))

        upper_flux = signal_to_noise * sigma_cont * np.sqrt(g_const * (lambda_central / R_line) * delta_lambda)

        print('lambda_central', lambda_central)
        print('delta_lambda',delta_lambda)
        print('sigma_cont',sigma_cont)
        print('R_line',R_line)
        print('g_const',g_const)
        print('upper_flux',upper_flux)

        return upper_flux


class SpecTreatment(LineFitting, RedshiftFitting):

    def __init__(self, spectrum):

        # Instantiate the dependencies
        LineFitting.__init__(self)

        # Lime spectrum object with the scientific data
        self._spec = spectrum
        self.line = None
        self._i_line = 0
        self._n_lines = 0

    def bands(self, label, bands=None, fit_cfg=None, min_method='least_squares', profile=None, shape=None,
              cont_source='central', err_from_bands=None, temp=10000.0, default_cfg_prefix='default', obj_cfg_prefix=None,
              update_default=True):

        """
        Fit a spectral line from defined bands (see :ref:`bands documentation <line-bands-doc>`.).

        This method performs a full line measurement and profile fitting. The line is query from the default
        line database if no ``bands`` dataframe is provided.

        The function will also query the input ``fit_cfg`` dictionary if provided for the configuration settings.

        Parameters
        ----------
        label : str or float
            Line label in `LiMe notation
            <https://lime-stable.readthedocs.io/en/latest/inputs/n_inputs2_line_labels.html>`_,
            or a transition wavelength (in the same units as the spectrum).
            If a numeric wavelength is given, the corresponding transition is
            queried from the ``bands`` table (or falling back to the default database if not provided).
        bands : pandas.DataFrame, str, or pathlib.Path, optional
            Either:
              * A bands DataFrame or file path to one.
              * If ``None``, the default LiMe bands database is used.
        fit_cfg : dict, str, or pathlib.Path, optional
            A dictionary or a path to a .toml file.
        min_method : str, optional
            Minimization algorithm used by :mod:`lmfit`.
            Supported methods are listed in the
            `lmfit.minimizer.Minimizer.minimize
            <https://lmfit.github.io/lmfit-py/fitting.html#lmfit.minimizer.Minimizer.minimize>`_
            documentation. Default is ``"least_squares"``.
        profile : str, optional
            Profile type for fitting (e.g., ``"g"`` for Gaussian, ``"l"`` Lorentz, ...).
            If none is provided, the algorithm will use the default profile from the lines' database.
        shape : str, optional
            Line shape for the fitted profiles: "emi" for emission or "abs" for absroption.
            If none is provided, the algorithm will use the default shape from the lines' database.
        cont_source : {'central', 'adjacent', 'fitted'}, optional
            Method used to estimate the continuum level for line fitting.
            - ``'central'`` — use the edges of the central line band (w₃–w₄).
            - ``'adjacent'`` — use the adjacent continuum bands (w₁–w₂ and w₅–w₆).
            - ``'fitted'`` — use a previously fitted continuum model from the spectrum.
            The default is ``'central'``.
        err_from_bands : bool or None, optional
            If ``True``, estimate the pixel uncertainty from the continuum bands.
            If ``None``, use the spectrum’s ``err_flux`` data or fall back to the
            continuum regions if not available (False). The default value is None.
        temp : float, optional
            Electron temperature in Kelvin used to compute the thermal broadening
            correction for the fitted lines. Default is 10,000 K.
        default_cfg_prefix : str, optional
            Section key prefix for the default configuration in ``fit_cfg``.
            Default is ``"default"``.
        obj_cfg_prefix : str, optional
            Section key prefix for the object-specific configuration in ``fit_cfg``.
        update_default : bool, optional
            If ``True`` (default), merge parameters from ``obj_cfg_prefix`` into
            ``default_cfg_prefix``. If ``False``, the object configuration is used falling back to the default if not available.

        Returns
        -------
        None
            The results are stored in the spectrum’s internal ``frame`` attribute
            and in ``self.line.measurements``.

        Notes
        -----
        - The method performs the following steps:
          1. Parse or retrieve the line definition using :meth:`~lime.Line.from_transition`.
          2. Select line and continuum regions based on the provided or default ``bands``.
          3. Estimate the continuum level and its uncertainty.
          4. Compute non-parametric line properties (e.g., integrated flux).
          5. Perform optional profile fitting via :mod:`lmfit` using ``min_method``.
          6. Apply instrumental and thermal corrections to measured line widths.
          7. Recalculate the signal-to-noise ratio and store all results in the spectrum’s log frame.
        - Continuum and error estimations are controlled via the
          ``cont_from_bands`` and ``err_from_bands`` flags.
        - Thermal broadening corrections use the provided ``temp`` parameter.
        - All results are written to the current spectrum’s measurement log
          and accessible through ``Spectrum.frame``.

        Examples
        --------
        Fit a Gaussian emission line using the default configuration:

        >>> spec.bands("O3_5007A")

        Use a custom bands table and configuration dictionary:

        >>> spec.bands("H1_4861A", bands="my_bands.xlsx", fit_cfg=my_fit_cfg)

        Change the minimization algorithm and temperature:

        >>> spec.bands("O2_3726A", min_method="nelder", temp=12000)

        Fit a line providing the central wavelength directly:

        >>> spec.bands(5007.0, bands=my_bands_df)
        """

        # Make a copy of the fitting configuration
        input_conf = check_fit_conf(fit_cfg, default_cfg_prefix, obj_cfg_prefix, update_default)

        # User 2_guides override default behaviour for the pixel error and the continuum calculation
        err_from_bands = True if (err_from_bands is None) and (self._spec.err_flux is None) else err_from_bands

        # Interpret the input line
        if isinstance(label, str):
            self.line = Line.from_transition(label, input_conf,
                                             data_frame=lineDB.frame if bands is None else check_file_dataframe(bands, copy_input=False),
                                             def_shape=shape,
                                             def_profile=profile)
        else:
            self.line = label

        # Check the line selection is valid
        idcs_selection = review_bands(self._spec, self.line, user_cont_source=cont_source, user_err_from_bands=err_from_bands)
        if idcs_selection is not None:

            # Unpack the line selections
            idcs_line, idcs_continua = idcs_selection

            # Compute line continuum
            cont_arr = self.continuum_calculation(idcs_line, idcs_continua, cont_source, err_from_bands)

            # Compute line flux error
            pixel_err_arr = self.pixel_error_calculation(idcs_continua, err_from_bands)

            # Non-parametric measurements
            self.integrated_properties(self.line, self._spec.wave[idcs_line], self._spec.flux[idcs_line],
                                       pixel_err_arr[idcs_line], cont_arr[idcs_line])

            # Import kinematics if requested
            import_line_kinematics(self.line, 1 + self._spec.redshift, self._spec.frame, input_conf)

            # Profile fitting measurements
            idcs_fitting = idcs_selection[0] if cont_source == 'central' else idcs_selection[0] + idcs_selection[1]
            self.profile_fitting(self.line,
                                 x_arr=self._spec.wave[idcs_fitting],
                                 y_arr=self._spec.flux[idcs_fitting],
                                 err_arr=pixel_err_arr[idcs_fitting],
                                 cont_arr=cont_arr[idcs_fitting] if cont_source == 'fit' else None,
                                 user_conf=input_conf, fit_method=min_method)

            # Instrumental and thermal corrections for the lines
            sigma_corrections(self.line, idcs_line, self._spec.wave[idcs_line], self._spec.res_power, temp)

            # Recalculate the SNR with the profile parameters
            self.line.measurements.snr_line = signal_to_noise_rola(self.line.measurements.amp,
                                                                   self.line.measurements.cont_err,
                                                                   self.line.measurements.n_pixels)

            # Save the line parameters
            results_to_log(self.line, self._spec.frame, self._spec.norm_flux)

        return


    def frame(self, bands, fit_cfg=None, min_method='least_squares', profile=None, shape=None, cont_source='central',
              err_from_bands=None, temp=10000.0, line_list=None, default_cfg_prefix='default', obj_cfg_prefix=None,
              update_default=True, line_detection=False, plot_fit=False, progress_output='bar'):

        """
        Measure multiple spectral lines from a bands dataframe
        (see :ref:`bands documentation <line-bands-doc>`).

        This method automates the fitting of the lines on the input lines frame.
        It iterates through all listed transitions,
        performing continuum estimation, line detection (optional), and profile
        fitting using the configuration provided in ``fit_cfg``.

        Parameters
        ----------
        bands : pandas.DataFrame, str, or pathlib.Path
            Bands table defining the line labels and bands limits (w1 to w6) for each line.
        fit_cfg : dict, str, or pathlib.Path, optional
            Fitting configuration dictionary or a path to a TOML configuration file.
            See the `profile fitting documentation
            <https://lime-stable.readthedocs.io/en/latest/inputs/n_inputs4_fit_configuration.html>`_.
        min_method : str, optional
            Minimization algorithm used by :mod:`lmfit`.
            See the
            `lmfit.minimizer.Minimizer.minimize
            <https://lmfit.github.io/lmfit-py/fitting.html#lmfit.minimizer.Minimizer.minimize>`_
            documentation for available options. Default is ``"least_squares"``.
        profile : str, optional
            Profile type for fitting (e.g., ``"g"`` for Gaussian, ``"l"`` for Lorentzian).
            Defaults to the line database entry if omitted.
        shape : str, optional
            Line shape keyword (``"emi"`` for emission or ``"abs"`` for absorption).
            Defaults to the line database entry if omitted.
        cont_source : {'central', 'adjacent', 'fitted'}, optional
            Method used to estimate the continuum level for line fitting.
            - ``'central'`` — use the edges of the central line band (w₃–w₄).
            - ``'adjacent'`` — use the adjacent continuum bands (w₁–w₂ and w₅–w₆).
            - ``'fitted'`` — use a previously fitted continuum model from the spectrum.
            The default is ``'central'``.
        err_from_bands : bool or None, optional
            If ``True``, estimate the pixel uncertainty from the continuum bands.
            If ``None``, use the spectrum’s ``err_flux`` data or fall back to the
            continuum regions if not available (False). The default value is None.
        temp : float, optional
            Electron temperature (K) used to compute the thermal broadening correction.
            Default is 10,000 K.
        line_list : list of str, optional
            Subset of line labels from the bands table to process.
            If ``None``, all entries in ``bands`` are measured.
        default_cfg_prefix : str, optional
            Section key prefix for the default configuration in ``fit_cfg``.
            Default is ``"default"``.
        obj_cfg_prefix : str, optional
            Section key prefix for the object-specific configuration in ``fit_cfg``.
        update_default : bool, optional
            If ``True`` (default), merge parameters from ``obj_cfg_prefix`` into
            ``default_cfg_prefix``. If ``False``, the object configuration is used falling back to the default if not available.
        line_detection : bool, optional
            If ``True``, run the continuum fitting and line threshodling to confirme the presence of lines before
            measurements. The functions parameters must be specified in the ``fit_cfg`` (e.g., entries under ``"peaks_troughs"``, ``"continuum"``).
            Default is ``False``.
        plot_fit : bool, optional
            If ``True``, display the profile fit after each iteration.
        progress_output : {"bar", "counter", None}, optional
            Controls progress display in the console.
            - ``"bar"`` (default): show a dynamic progress bar.
            - ``"counter"``: print current line number and label.
            - ``None``: suppress console output.

        Returns
        -------
        None
            The resulting measurements are stored in the spectrum’s internal
            ``frame`` attribute and in ``self.line.measurements``.

        Notes
        -----
        - This method performs the following sequence for each line:
          1. Optionally apply continuum and line detection preprocessing steps if enabled via ``line_detection=True`` and the appropicate keys are found at ``fit_cfg``.
          2. Retrieve the line list from the ``bands`` table or the default database.
          3. Estimate the continuum level and its uncertainty.
          4. Perform non-parametric measurements (e.g., flux, EW, FWHM).
          5. Run profile fitting using :mod:`lmfit` according to ``min_method``.
          6. Apply instrumental and thermal width corrections (via ``Spectrum.res_power`` and ``temp``).
          7. Recalculate SNR and store results in the spectrum’s log frame.
        - Progress reporting is configurable through ``progress_output``.
        - Use ``line_detection=True`` to automatically threshold and select
          only detected lines before fitting.

        Examples
        --------
        Measure all lines from a bands file:

        >>> spec.frame("my_bands.xlsx", fit_cfg="my_fit_config.toml")

        Run the fit with a progress bar:

        >>> spec.frame(bands_df, progress_output="bar")

        Limit to a subset of lines:

        >>> spec.frame(bands_df, line_list=["O3_5007A", "H1_4861A"])

        Enable automatic line detection:

        >>> spec.frame(bands_df, line_detection=True)

        """

        # Check if the lines log is a dataframe or a file address
        bands = check_file_dataframe(bands)

        if bands is not None:

            # Crop the analysis to the target lines
            if line_list is not None:
                idcs = bands.index.isin(line_list)
                bands = bands.loc[idcs]

            # Load configuration
            input_conf = check_fit_conf(fit_cfg, default_cfg_prefix, obj_cfg_prefix, update_default=update_default,
                                        line_detection=line_detection)

            # Line detection if requested
            if line_detection:

                # Review the configuration entries
                cont_fit_conf = input_conf.get('continuum', None)
                if cont_fit_conf:
                    self._spec.fit.continuum(**cont_fit_conf)
                else:
                    _logger.warning(f'No "continuum" entry in input configuration file. No continuum fitting will be applied')


                # Perform the line detection
                detect_conf = input_conf.get('peaks_troughs', {})
                if detect_conf:
                    bands = self._spec.infer.peaks_troughs(bands, **detect_conf)
                else:
                    _logger.warning(f'No "peaks_troughs" entry in input configuration file. No line thresholding will be applied')

            else:
                cont_fit_conf, detect_conf = None, None

            # Define lines to treat through the lines
            label_list = bands.index.to_numpy()
            self._n_lines = label_list.size

            # Loop through the lines
            if self._n_lines > 0:

                # On screen progress bar
                pbar = ProgressBar(progress_output, f'{self._n_lines} lines')
                if progress_output is not None:
                    print(f'\nLine fitting progress{" (continuum fitting)" if cont_fit_conf is not None else ""}'
                                                  f'{" (line detection)" if detect_conf is not None else ""}:')

                for self._i_line in np.arange(self._n_lines):

                    # Ignore line if part of a blended/merge group
                    line = label_list[self._i_line]
                    measure_check = check_compound_line_exclusion(line, bands)

                    if measure_check:

                        # Progress message
                        pbar.output_message(self._i_line, self._n_lines, pre_text="", post_text=f'({line})')

                        # Fit the lines
                        self.bands(line, bands, input_conf, min_method, profile, shape, cont_source=cont_source,
                                   err_from_bands=err_from_bands, temp=temp, obj_cfg_prefix=None, default_cfg_prefix=None)

                        if plot_fit:
                            self._spec.plot.bands()

            else:
                msg = f'No lines were measured from the input dataframe:\n - line_list: {line_list}\n - line_detection: {line_detection}'
                _logger.debug(msg)

        else:
            _logger.info(f'Not input dataframe. Lines were not measured')

        return

    def continuum(self, degree_list, emis_threshold, abs_threshold=None, smooth_scale=None, exclude_intvls=None,
                  rest_intvls=False,  plot_steps=False, **kwargs):

        """
        Fit the spectrum continuum via polynomial clipping.

        This routine estimates the continuum by iteratively fitting polynomials and
        sigma-clipping outliers above (emission) and below (absorption) a flux threshold.
        At each iteration, points outside configurable residual thresholds are excluded
        and the polynomial is refitted on the remaining pixels.
        The user may optionally provide a list of wavelength intervals to be excluded from
        the continuum fitting. By default, these limits are assumed to be defined in the
        observed frame.

        Parameters
        ----------
        degree_list : list of int
            Polynomial degree to use at each iteration.
            The number of iterations equals ``len(degree_list)``.
        emis_threshold : list of float
            Upper (emission-side) clipping factors, in units of number of residual standard
            deviation for each iteration. Must have the same length as ``degree_list``.
        abs_threshold : list of float, optional
            Lower (absorption-side) clipping factors, also in units of number of residual
            standard deviation. If ``None``, the values in ``emis_threshold`` are reused
            for the lower limit. Must match the length of ``degree_list`` when provided.
        smooth_scale : int, optional
            Window size (in pixels) for a moving-average smoothing applied to the input
            flux before fitting. If ``None``, no smoothing is applied.
        exclude_intvls : list of tuple(float, float), optional
            List of wavelength intervals (low, high) to exclude from the continuum fitting.
            By default, intervals are interpreted in the observed frame.
        rest_intvls : bool, optional
            If ``True``, the wavelength intervals in ``exclude_intvls`` are assumed to be
            defined in the rest frame and are converted to the observed frame using
            ``λ_obs = λ_rest × (1 + z)`` prior to mask computation.
        plot_steps : bool, optional
            If ``True``, display a diagnostic plot after each iteration showing the
            current fit, clipping limits, and kept/rejected pixels.
        **kwargs
            Additional keyword arguments forwarded to the plotting helper if ``plot_steps=True``
            (e.g., figure size, axis, title customization).

        Returns
        -------
        None
            The method updates the spectrum in place, setting:
            - ``self._spec.cont`` : masked array of the final continuum model
            - ``self._spec.cont_std`` : float, standard deviation of residuals on kept pixels

        Notes
        -----
        - **Initialization:** The first iteration seeds the mask using the 16th–84th
          percentile flux range of unmasked pixels, then fits the initial polynomial.
        - **Clipping:** After each fit, residuals are computed and the standard
          deviation is measured over currently kept pixels. New keep/reject limits are:
          ``low = model - abs_threshold[i] * std`` and
          ``high = model + emis_threshold[i] * std``.
        - **Masking:** Existing pixel masks are honored; clipping only modifies the
          continuum-selection mask on top of the original flux mask.
        - **Smoothing:** When ``smooth_scale`` is provided, a boxcar (length
          ``smooth_scale``) is convolved with the flux prior to fitting; the continuum
          itself is always evaluated on the original wavelength grid.

        Examples
        --------
        Fit a three-iteration continuum with increasingly restrictive clipping:

        >>> degrees = [1, 2, 2]
        >>> thr_hi  = [5.0, 3.0, 2.0]     # emission-side thresholds (σ)
        >>> thr_lo  = [5.0, 3.0, 2.0]     # absorption-side thresholds (σ)
        >>> spec.fit.continuum(degrees, thr_hi, abs_threshold=thr_lo, smooth_scale=11)

        Show diagnostic plots at each iteration:

        >>> spec.fit.continuum([2, 2], [3.0, 2.0], plot_steps=True, title="Continuum fit")

        Exclude known emission-line regions (defined in the rest frame) from the continuum fit:

        >>> spec.continuum(degree_list=[3, 2], emis_threshold=[3.0, 2.0], exclude_intvls=[(4861, 4875), (6548, 6584)], rest_intvls=True)

        """

        # Create a pre-Mask based on the original mask if available
        input_wave = self._spec.wave.data
        input_flux = self._spec.flux.data
        input_mask = ~self._spec.flux.mask

        # Correction if intervals are in the rest frame
        z_corr = 1 + self._spec.redshift if rest_intvls else 1

        # Adjust the mask to exclude wavelengt intervals
        if exclude_intvls is not None:
            exclude_intvls = np.asarray(exclude_intvls, dtype=float)

            # Check the input has the right format
            if exclude_intvls.ndim != 2 or exclude_intvls.shape[1] != 2:
                raise ValueError('The argument "exclude_intvls" must be a list of (low, high) pairs')

            # Loop through the wavelength intervals and add them to the mask
            exclude_intvls = exclude_intvls * z_corr
            i0 = np.searchsorted(input_wave, exclude_intvls[:, 0], side="right")
            i1 = np.searchsorted(input_wave, exclude_intvls[:, 1], side="left")
            for start, stop in zip(i0, i1):
                input_mask[start:stop] = False

        # Smooth the spectrum
        if smooth_scale is not None:
            smoothing_window = np.ones(smooth_scale) / smooth_scale
            input_flux = np.convolve(input_flux, smoothing_window, mode='same')

        # Loop through the fitting degree
        abs_threshold = emis_threshold if abs_threshold is None else abs_threshold
        for i, degree in enumerate(degree_list):

            # First iteration use percentile limits for an initial fit
            if i == 0:
                low_lim, high_lim = np.nanpercentile(input_flux[input_mask], (16, 84))
                mask_cont_0 = input_mask & (input_flux >= low_lim) & (input_flux <= high_lim)
                cont_fit = continuum_model_fit(input_wave, input_flux, mask_cont_0, degree)

            # Establishing the flux limits
            std_flux = np.nanstd((input_flux - cont_fit)[input_mask])
            low_lim, high_lim = cont_fit - abs_threshold[i] * std_flux, cont_fit + emis_threshold[i] * std_flux

            # Add new entries to the mask
            input_mask = input_mask & (input_flux >= low_lim) & (input_flux <= high_lim)

            # Fit continuum
            cont_fit = continuum_model_fit(input_wave, input_flux, input_mask, degree)

            # Compute the continuum and assign replace the value outside the bands the new continuum
            if plot_steps:
                input_kwargs = {'title':f'Continuum fitting, iteration ({i+1}/{len(degree_list)})'}
                input_kwargs.update(kwargs)
                spec_continuum_calculation(self._spec, input_wave, input_flux, cont_fit, input_mask, low_lim,
                                           high_lim, smooth_scale, exclude_intvls,  **input_kwargs)

        # Include the standard deviation of the spectrum for the unmasked pixels
        self._spec.cont = np.ma.masked_array(cont_fit, self._spec.flux.mask)
        self._spec.cont_std = np.std((input_flux - cont_fit)[input_mask])

        return


class CubeTreatment(LineFitting):

    def __init__(self, cube):

        # Instantiate the dependencies
        LineFitting.__init__(self)

        # Lime spectrum object with the scientific data
        self._cube = cube
        self._spec = None

    def spatial_mask(self, mask_file, fname, bands=None, fit_cfg=None, mask_list=None, line_list=None,
                     log_ext_suffix='_LINELOG', min_method='least_squares', profile=None, shape=None, cont_source='central',
                     err_from_bands=False, temp=10000.0, default_cfg_prefix='default', update_default=True,
                     line_detection=False, progress_output='bar', plot_fit=False, header=None, join_output_files=True):

        """
        Measure lines across an IFS cube using one or more spatial masks.

        This routine iterates over spaxels selected by binary masks (from ``mask_file``),
        measures the requested lines for each spaxel, and writes the results to one or
        more multi-extension FITS logs. Each spaxel’s measurements are saved in a
        dedicated extension named ``"{j}-{i}{log_ext_suffix}"`` (e.g., ``"25-30_LINELOG"``).

        The bands to use for measurements can be supplied globally via ``bands`` or,
        per-mask, via entries in ``fit_cfg``. Line-fitting behavior is configured via
        ``fit_cfg`` with a multi-level override scheme (default → mask → spaxel).

        Parameters
        ----------
        mask_file : str or pathlib.Path or dict or numpy.ndarray
            Source of spatial masks. Can be a FITS file produced by
            :meth:`~lime.Cube.spatial_masking`, a dictionary mapping mask names to
            boolean arrays, or a boolean array.
        fname : str or pathlib.Path
            Output path for the combined measurements log (or the base name, if generating one .fits file per mask).
        bands : pandas.DataFrame, str, or pathlib.Path, optional
            Bands table (or path) to use for all masks/spaxels. If ``None``, the
            method will look for a per-mask bands path in ``fit_cfg``.
        fit_cfg : dict or str or pathlib.Path, optional
            Fitting configuration (dict or path to a TOML file). Supports a
            three-level override hierarchy:
              1) ``default_cfg_prefix`` section (global defaults),
              2) mask-level section named after the mask (e.g., ``"MASK_A"``),
              3) spaxel-level section named ``"{j}-{i}_line_fitting"``.
        mask_list : list of str, optional
            Subset of masks in ``mask_file`` to process. If ``None``, all masks are used.
        line_list : list of str, optional
            Subset of line labels to measure from the bands table. If ``None``,
            all lines present in the bands are measured.
        log_ext_suffix : str, optional
            Suffix appended to each FITS extension name with results.
            Default is ``"_LINELOG"``.
        min_method : str, optional
            Minimization algorithm used by :mod:`lmfit`. See
            `lmfit.minimizer.Minimizer.minimize
            <https://lmfit.github.io/lmfit-py/fitting.html#lmfit.minimizer.Minimizer.minimize>`_.
            Default is ``"least_squares"``.
        profile : str, optional
            Profile identifier for fitting (e.g., ``"g"`` for Gaussian).
            If ``None``, falls back to the line database defaults.
        shape : str, optional
            Line shape: ``"emi"`` for emission or ``"abs"`` for absorption.
            If ``None``, falls back to database defaults.
        cont_source : {'central', 'adjacent', 'fitted'}, optional
            Method used to estimate the continuum level for line fitting.
            - ``'central'`` — use the edges of the central line band (w₃–w₄).
            - ``'adjacent'`` — use the adjacent continuum bands (w₁–w₂ and w₅–w₆).
            - ``'fitted'`` — use a previously fitted continuum model from the spectrum.
            The default is ``'central'``.
        err_from_bands : bool or None, optional
            If ``True``, estimate the pixel uncertainty from the continuum bands.
            If ``None``, use the spectrum’s ``err_flux`` data or fall back to the
            continuum regions if not available (False). The default value is None.
        temp : float, optional
            Electron temperature (K) for thermal broadening corrections.
            Default is ``10000.0``.
        default_cfg_prefix : str, optional
            Section name in ``fit_cfg`` containing global defaults.
            Default is ``"default"``.
        update_default : bool, optional
            If ``True`` (default), apply higher-level overrides by updating lower-level
            dictionaries (shared keys only).
        line_detection : bool, optional
            If ``True``, run the continuum fitting and line threshodling to confirme the presence of lines before
            measurements. The functions parameters must be specified in the ``fit_cfg`` (e.g., entries under ``"peaks_troughs"``, ``"continuum"``).
            Default is ``False``.
        progress_output : {"bar", "counter", None}, optional
            Console progress reporting mode. Default is ``"bar"``.
        plot_fit : bool, optional
            If ``True``, render a plot of each spaxel’s fitted lines during processing.
            Default is ``False``.
        header : dict, optional
            Extra FITS header keywords to add per spaxel page in the output logs.
            If a key matching the extension name (e.g., ``"25-30_LINELOG"``) exists,
            that dict is used; otherwise ``header`` is treated as a global header.
        join_output_files : bool, optional
            If multiple masks are processed, merge the per-mask logs into a single
            FITS file named after ``fname``. When ``False``, keep one output file
            per mask (named ``"{stem}_MASK-{mask}.fits"``). Default is ``True``.

        Returns
        -------
        None
            Results are written to disk as one or more FITS files. Each spaxel’s
            measurements are stored in a binary table extension.

        Notes
        -----
        - **Configuration hierarchy:** Values are resolved in the order
          *default → mask → spaxel*. Higher levels **update** shared keys only if
          ``update_default=True``. Otherwise, the method applies a fallback protocol
          where only the parameters explicitly defined in each section are used.
        - **WCS headers:** Spatial WCS keywords are added to each extension header when
          available, using the parent cube’s WCS metadata.

        Examples
        --------
        Use a single bands table for all masks and join outputs:

        >>> cube.obs.spatial_mask(
        ...     mask_file="O3_masks.fits",
        ...     fname="logs/o3_all_masks.fits",
        ...     bands="my_bands.xlsx",
        ...     fit_cfg="fit_config.toml",
        ...     progress_output="bar",
        ... )

        Provide bands per mask via the configuration and keep files separate:

        >>> cube.obs.spatial_mask(
        ...     mask_file="O3_masks.fits",
        ...     fname="logs/o3_base.fits",
        ...     fit_cfg="fit_config.toml",
        ...     join_output_files=False,
        ... )

        Limit measured lines and enable line detection:

        >>> cube.obs.spatial_mask(
        ...     mask_file="masks.fits",
        ...     fname="logs/selected_lines.fits",
        ...     line_list=["O3_5007A", "H1_4861A"],
        ...     line_detection=True,
        ... )
        """

        if bands is not None:
            bands = check_file_dataframe(bands)

        # Check if the mask variable is a file or an array
        mask_maps = check_file_array_mask(mask_file, mask_list)
        mask_list = np.array(list(mask_maps.keys()))
        mask_data_list = list(mask_maps.values())

        # Check the mask configuration is included if there are no masks
        input_masks = mask_list if bands is None else None
        input_conf = check_fit_conf(fit_cfg, default_key=None, obj_key=None, update_default=update_default,
                                    group_list=input_masks)

        # Check if the output log folder exists
        fname = Path(fname)
        address_dir = fname.parent
        if not address_dir.is_dir():
            raise LiMe_Error(f'The folder of the output log file does not exist at {fname}')
        address_stem = fname.stem

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
            mask_conf = check_fit_conf(input_conf, default_cfg_prefix, mask_name)

            # Load the mask log if provided
            if bands is None:
                bands_file = Path(mask_conf['bands']).resolve()
                if bands_file.exists():
                    bands_in = load_frame(bands_file)
                else:
                    err_msg = (f'Bands file not found at: {bands_file}.'
                               f'\n- Resolving from log section - entry: '
                               f'\n [{mask_name}_line_fitting]'
                               f'\n bands = {mask_conf["bands"]}')
                    raise LiMe_Error(err_msg)
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
                                 line_detection=line_detection, profile=profile, shape=shape,
                                 cont_source=cont_source, err_from_bands=err_from_bands,
                                 temp=temp, progress_output=None, plot_fit=None,
                                 obj_cfg_prefix=None, default_cfg_prefix=None, update_default=update_default)

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
                    spaxel.plot.spectrum(rest_frame=True)

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

        return


