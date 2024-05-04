import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging

from pathlib import Path
from scipy import signal
from lmfit.models import PolynomialModel
from inspect import signature
from .io import LiMe_Error, check_file_dataframe, _PARENT_BANDS
from .transitions import label_decomposition
from . import _setup_cfg
from .model import gaussian_model

try:
    import joblib
    joblib_check = True
except ImportError:
    joblib_check = False


_logger = logging.getLogger('LiMe')

MACHINE_PATH = None
FLUX_PIXEL_CONV = np.linspace(0,1,33)


def compute_line_width(idx_peak, spec_flux, delta_i, min_delta=2, emission_check=True):

    """
    Algororithm to measure emision line width given its peak location
    :param idx_peak:
    :param spec_flux:
    :param delta_i:
    :param min_delta:
    :return:
    """

    i = idx_peak

    if emission_check:
        while (spec_flux[i] > spec_flux[i + delta_i]) or (np.abs(idx_peak - (i + delta_i)) <= min_delta):
            i += delta_i
    else:
        while (spec_flux[i] < spec_flux[i + delta_i]) or (np.abs(idx_peak - (i + delta_i)) <= min_delta):
            i += delta_i

    return i


def detection_function(x_ratio):

    # Original
    # 2.5 + 1/np.square(x_ratio - 0.1) + 0.5 * np.square(x_ratio)

    return 0.5 * np.power(x_ratio, 2) - 0.5 * x_ratio + 5


def feature_scaling(data, transformation='min-max', log_base=None, axis=1):

    if transformation == 'min-max':
        data_min_array = data.min(axis=axis, keepdims=True)
        data_max_array = data.max(axis=axis, keepdims=True)
        data_norm = (data - data_min_array) / (data_max_array - data_min_array)

    elif transformation == 'log':
        y_cont = data - data.min(axis=1, keepdims=True) + 1
        data_norm = np.emath.logn(log_base, y_cont)

    elif transformation == 'log-min-max':
        data_cont = data - data.min(axis=1, keepdims=True) + 1
        log_data = np.emath.logn(log_base, data_cont)
        log_min_array, log_max_array = log_data.min(axis=axis, keepdims=True), log_data.max(axis=axis, keepdims=True)
        data_norm = (log_data - log_min_array) / (log_max_array - log_min_array)

    else:
        raise KeyError(f'Input scaling "{transformation}" is not recognized.')

    return data_norm


def enbox_spectrum(input_flux, box_size, range_box):

    # Use only the true entries from the mask
    flux_array = input_flux if not np.ma.isMaskedArray(input_flux) else input_flux.data[~input_flux.mask]

    # Reshape to the detection interval
    n_intervals = flux_array.size - box_size + 1
    flux_array = flux_array[np.arange(n_intervals)[:, None] + range_box]

    # # Remove nan entries
    # idcs_nan_rows = np.isnan(input_flux).any(axis=1)
    # flux_array = input_flux[~idcs_nan_rows, :]

    return flux_array


def flux_to_image(flux_array, approximation, model_2D):

    if model_2D is not None:

        img_flux_array = np.tile(flux_array[:, None, :, :], (1, approximation.size, 1, 1))
        img_flux_array = img_flux_array > approximation[::-1, None, None]
        img_flux_array = img_flux_array.astype(int)
        img_flux_array = img_flux_array.reshape((flux_array.shape[0], 1, -1, flux_array.shape[-1]))
        img_flux_array = img_flux_array.squeeze()

        # Original
        # flux_array_0 = flux_array[:, :, 0]
        # Old_img = np.tile(flux_array_0[:, None, :], (1, approximation.size, 1))
        # Old_img = Old_img > approximation[::-1, None]
        # Old_img = Old_img.astype(int)
        # Old_img = Old_img.reshape((flux_array_0.shape[0], 1, -1))
        # Old_img = Old_img.squeeze()

    else:
      img_flux_array = None

    return img_flux_array


def prop_func(data_arr, box_width):

    prop_arr = np.zeros(data_arr.shape, dtype=bool)
    idcs_true = np.argwhere(data_arr) + np.arange(box_width)
    idcs_true = idcs_true.flatten()
    idcs_true = idcs_true[idcs_true < data_arr.size] # in case the error propagation gives a great
    prop_arr[idcs_true] = True

    # Original
    # self.mask = np.zeros(n_pixels, dtype=bool)
    # idcs_detect = np.argwhere(pred_array[:, 0]) + range_box
    # idcs_detect = idcs_detect.flatten()
    # idcs_detect = idcs_detect[idcs_detect < pred_array[:, 0].size]
    # self.mask[idcs_detect] = True

    return prop_arr


def check_lisa(model1D, model2D):

    if model1D is None:
        coeffs1D = np.array(_setup_cfg['linear']['model1D_coeffs']), np.array(_setup_cfg['linear']['model1D_intercept'])
    else:
        model1D_job = joblib.load(model1D)
        coeffs1D = np.squeeze(model1D_job.coef_), np.squeeze(model1D_job.intercept_)

    if model2D is None:
        coeffs2D = np.array(_setup_cfg['linear']['model2D_coeffs']), np.array(_setup_cfg['linear']['model2D_intercept'])
    else:
        model2D_job = joblib.load(model2D)
        coeffs2D = np.squeeze(model2D_job.coef_), np.squeeze(model2D_job.intercept_)

    return coeffs1D, coeffs2D


class LineFinder:

    def __init__(self, machine_model_path=MACHINE_PATH):

        # self.ml_model = joblib.load(machine_model_path) # THIS CAN be warning at opening the file

        return

    def continuum_fitting(self, degree_list=[3, 7, 7, 7], threshold_list=[5, 3, 2, 2], plot_results=False,
                          return_std=False):

        # Check for a masked array
        if np.ma.isMaskedArray(self.flux):
            mask_cont = ~self.flux.mask
            input_wave, input_flux = self.wave.data, self.flux.data
        else:
            mask_cont = np.ones(self.flux.size).astype(bool)
            input_wave, input_flux = self.wave, self.flux


        # Loop through the fitting degree
        for i, degree in enumerate(degree_list):

            # Establishing the flux limits
            input_flux_selection = input_flux[mask_cont]
            if np.ma.isMaskedArray(input_flux_selection): # Bugged numpy
                low_lim, high_lim = np.nanpercentile(input_flux_selection.filled(np.nan), (16, 84))
            else:
                low_lim, high_lim = np.percentile(input_flux_selection, (16, 84))

            low_lim, high_lim = low_lim / threshold_list[i], high_lim * threshold_list[i]

            # Add new entries to the mask
            mask_cont = mask_cont & (input_flux >= low_lim) & (input_flux <= high_lim)

            poly3Mod = PolynomialModel(prefix=f'poly_{degree}', degree=degree)
            poly3Params = poly3Mod.guess(input_flux[mask_cont], x=input_wave[mask_cont])

            try:
                poly3Out = poly3Mod.fit(input_flux[mask_cont], poly3Params, x=input_wave[mask_cont])
                continuum_fit = poly3Out.eval(x=input_wave)

            except TypeError:
                _logger.warning(f'- The continuum fitting polynomial as degree ({degree}) is larger than data points'
                                f' number')
                continuum_fit = np.full(input_wave.size, np.nan)

            # Compute the continuum and assign replace the value outside the bands the new continuum
            if plot_results:
                title = f'Continuum fitting, iteration ({i+1}/{len(degree_list)})'
                continuum_full = poly3Out.eval(x=self.wave.data)
                self.plot._plot_continuum_fit(continuum_full, mask_cont, low_lim, high_lim, threshold_list[i], title)

        # Include the standard deviation of the spectrum for the unmasked pixels
        if return_std:
            std_spec = np.std((self.flux-continuum_fit)[mask_cont])
            output_params = (continuum_fit, std_spec)
        else:
            output_params = continuum_fit

        return output_params

    def ml_line_detection(self, continuum, box_width=11, model= None):

        if model is None:
            model = joblib.load(MACHINE_PATH)

        # Normalize the flux
        input_flux = self.flux if not np.ma.isMaskedArray(self.flux) else self.flux.data
        input_flux = np.log10((input_flux/continuum - 1) + 10)

        # Reshape to the training dimensions
        input_flux = np.array(input_flux, ndmin=2)

        # Container for the true pixels
        detection_mask = np.zeros(self.flux.size).astype(bool)

        # Case of 1D
        spectrum_pixels = input_flux.shape[1]
        for i in np.arange(spectrum_pixels):
            if i + box_width <= spectrum_pixels:
                y = input_flux[:, i:i + box_width]
                if not np.any(np.isnan(y)):
                    detection_mask[i:i + box_width] = detection_mask[i:i + box_width] | model.predict(y)[0]
                    # print(f'y {i} ({np.sum(y)}): {self.ml_model.predict(y)[0]}')

        return detection_mask

    def line_detection(self, bands, sigma_threshold=3, emission_type=True, width_tol=5, band_modification=None,
                       continuum_array=None, continuum_std=None, plot_steps=False):

        """

        This function compares the input lines bands in the observation spectrum to confirm the presence of lines.

        The input bands can be specified as a pandas dataframe or the path to its file via the ``bands_df`` argument.

        The continuum needs to be fit a priori with the Spectrum.fit.continuum function or assigning a ``continuum_array``
        and a ``continuum_std``.

        The ``sigma_threshold`` establishes the standard deviation factor beyond which a positive line detection is assumed.

        By default the algorithm seeks for emission lines, set ``emission_type`` equal to False for absorption lines.

        The additional arguments provide additional tools to adjust the line detection and show the steps/results.

        :param bands: Input bands dataframe or the address to its file.
        :type bands: pandas.Dataframe, str, pathlib.Path

        :param sigma_threshold: Continuum standard deviation factor for line detection. The default value is 3.
        :type sigma_threshold: float, optional

        :param emission_type: Line type. The default value is "True" for emission lines.
        :type emission_type: str, optional

        :param width_tol: Minimum number of pixels between peaks/troughs. The default value is 5.
        :type width_tol: float, optional

        :param band_modification: Method to adjust the line band with. The default value is None.
        :type band_modification: str, optional

        :param ml_detection: Machine learning algorithm to detect lines. The default value is None.
        :type ml_detection: str, optional

        :param plot_steps: Plot the detected peaks/troughs. The default value is False
        :type plot_steps: bool, optional

        """

        # Check for the peaks of the emission lines
        continuum_array = self.cont if continuum_array is None else continuum_array
        continuum_std = self.cont_std if continuum_std is None else continuum_std

        # Get indeces of peaks
        limit_threshold = sigma_threshold * continuum_std
        limit_threshold = continuum_array + limit_threshold if emission_type else continuum_array + limit_threshold
        idcs_peaks, _ = signal.find_peaks(self.flux, height=limit_threshold, distance=width_tol)

        # Match peaks with theoretical lines
        bands = check_file_dataframe(bands, pd.DataFrame)
        matched_DF = self.label_peaks(idcs_peaks, bands, width_tol=width_tol, band_modification=band_modification,
                                      line_type=emission_type)

        # Plot the results
        if plot_steps:
            self.plot._plot_peak_detection(idcs_peaks, limit_threshold, continuum_array, matched_DF)

        return matched_DF

    #
    def label_peaks(self, peak_table, mask_df, line_type='emission', width_tol=5, band_modification=None, detect_check=False):

        # TODO auto param should be changed to boolean
        # Establish the type of input values for the peak indexes, first numpy array
        if isinstance(peak_table, np.ndarray):
            idcsLinePeak = peak_table

        # Specutils table
        else:
            # Query the lines from the astropy finder tables #
            if len(peak_table) != 0:
                idcsLineType = peak_table['emission_type'] == line_type
                idcsLinePeak = np.array(peak_table[idcsLineType]['line_center_index'])
            else:
                idcsLinePeak = np.array([])

        # Security check in case no lines detected
        if len(idcsLinePeak) == 0:
            return pd.DataFrame(columns=mask_df.columns)

        # Prepare dataframe to stored the matched lines
        matched_DF = pd.DataFrame.copy(mask_df)
        matched_DF['signal_peak'] = np.nan

        # Theoretical wave values
        waveTheory = label_decomposition(matched_DF.index.values, params_list=['wavelength'])[0]
        matched_DF['wavelength'] = waveTheory

        # Match the lines with the theoretical emission
        tolerance = np.diff(self.wave_rest).mean() * width_tol
        matched_DF['observation'] = 'not_detected'
        unidentifiedLine = dict.fromkeys(matched_DF.columns.values, np.nan)

        # Get the wavelength peaks
        wave_peaks = self.wave_rest[idcsLinePeak]

        for i in np.arange(wave_peaks.size):

            idx_array = np.where(np.isclose(a=waveTheory.astype(np.float64), b=wave_peaks[i], atol=tolerance))

            if len(idx_array[0]) == 0:
                unknownLineLabel = 'xy_{:.0f}A'.format(np.round(wave_peaks[i]))

                # Scheme to avoid repeated lines
                if (unknownLineLabel not in matched_DF.index) and detect_check:
                    newRow = unidentifiedLine.copy()
                    newRow.update({'wavelength': wave_peaks[i], 'w3': wave_peaks[i] - 5, 'w4': wave_peaks[i] + 5,
                                   'observation': 'not_identified'})
                    matched_DF.loc[unknownLineLabel] = newRow

            else:

                row_index = matched_DF.index[matched_DF.wavelength == waveTheory[idx_array[0][0]]]
                matched_DF.loc[row_index, 'observation'] = 'detected'
                matched_DF.loc[row_index, 'signal_peak'] = idcsLinePeak[i]
                theoLineLabel = row_index[0]

                blended_check = True if '_b' in theoLineLabel else False
                minSeparation = 4 if blended_check else 2

                # Width is only computed for blended lines
                if band_modification is not None:
                    if band_modification == 'auto':
                        if blended_check is False:
                            emission_check = True if line_type == 'emission' else False
                            idx_min = compute_line_width(idcsLinePeak[i], self.flux, delta_i=-1, min_delta=minSeparation, emission_check=emission_check)
                            idx_max = compute_line_width(idcsLinePeak[i], self.flux, delta_i=1, min_delta=minSeparation, emission_check=emission_check)
                            matched_DF.loc[row_index, 'w3'] = self.wave_rest[idx_min]
                            matched_DF.loc[row_index, 'w4'] = self.wave_rest[idx_max]

        # Include_only_detected
        idcs_unknown = matched_DF['observation'] == 'not_detected'
        matched_DF.drop(index=matched_DF.loc[idcs_unknown].index.values, inplace=True)

        # Sort by wavelength
        matched_DF.sort_values('wavelength', inplace=True)
        matched_DF.drop(columns=['wavelength', 'observation'], inplace=True)

        return matched_DF


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

class FeatureClassifier:

    def __init__(self, data_array, box_width, model, n_pixels):

        # Attributes
        self.pred_matrix = None
        self.confidence = None
        self.n_pixels = n_pixels

        # Recover the models linear coefficients:
        w, b = model

        # Compute the prediction
        self.pred_matrix = np.tensordot(data_array, w, axes=([1], [0])) + b
        self.pred_matrix = self.pred_matrix > 0

        # # Propagate the true detections for the box window # Convolution
        # kernel = np.r_[np.zeros(box_width - 1), np.ones(box_width)][None]
        # column_kernel = kernel.reshape(-1, 1)
        # self.pred_matrix = signal.convolve2d(self.pred_matrix, column_kernel, mode='same').astype(bool)

        # Propagate the true detections for the box window
        self.pred_matrix = np.apply_along_axis(prop_func, 0, self.pred_matrix, box_width)

        # Confidence array from Montecarlo
        self.confidence = self.pred_matrix.sum(axis=1)

        return

    def __call__(self, confidence=None, entry=None, confidence_max=100):

        mask = np.zeros(self.n_pixels).astype(bool)

        if confidence is not None:

            if (confidence < 0) or (confidence > 100):
                raise LiMe_Error(f'Please define the detection confidence in an [0, 100] interval. '
                                 f'The input value was "{confidence}".')
            else:

                mask[:self.confidence.shape[0]] = (self.confidence >= confidence) & (confidence_max >= self.confidence)

        elif entry is not None:

            if (entry < 0) or (entry > 100):
                raise LiMe_Error(f'Please define the Monte Carlo entry in the [0, 100] interval. '
                                 f'The input value was "{entry}".')

            mask[:self.pred_matrix.shape[0]] = self.pred_matrix[:, entry]

        else:
            raise LiMe_Error(f'Please define a confidence interval or an Monte Carlo for the bands detection mask')

        return mask


class DetectionInference:

    def __init__(self, spectrum):

        self._spec = spectrum
        self.narrow_detect = None
        self.box_width = None
        self.range_box = None
        self.n_mc = 100

        self.line_1d_pred = None
        self.line_2d_pred = None
        self.line_pred = None

        return

    def bands(self, box_width=None, approximation=None, scale_type='min-max', log_base=10000, model_1d_path=None,
              model_2d_path=None):

        # Assign default values
        self.box_width = box_width if box_width is not None else _setup_cfg['linear']['box_width']
        self.range_box = np.arange(self.box_width)
        approximation = FLUX_PIXEL_CONV if approximation is None else approximation

        # Load the models
        # model1D = joblib.load(model_1d_path)
        # model2D = joblib.load(model_2d_path)
        model1D, model2D = check_lisa(model_1d_path, model_2d_path)

        # Reshape to the detection interval (n x box_size) matrix
        input_flux = enbox_spectrum(self._spec.flux, self.box_width, self.range_box)

        # Add random noise for Monte Carlo
        input_flux = self.monte_carlo_expansion(input_flux)

        # Normalize the flux
        input_flux = feature_scaling(input_flux, transformation=scale_type, log_base=log_base)

        # Reshape for detection types
        input_flux_2D = flux_to_image(input_flux, approximation, model2D)

        # Perform and store the detections
        self.feature_detection(input_flux, input_flux_2D, model1D, model2D, self._spec.flux.size)

        return

    def monte_carlo_expansion(self, flux_array, noise_scale=None):

        # Get the noise scale for the selections
        if noise_scale is None:
            noise_scale = self._spec.err_flux if self._spec.err_flux is not None else self._spec.cont_std

            if noise_scale is None:
                _logger.warning(f"No flux uncertainty provided for the line detection. There won't be a confidence value"
                                f" for the predictions.")
                self.n_mc = 1

        # Single flux uncertainty
        noise_matrix_shape = (flux_array.shape[0], flux_array.shape[1], self.n_mc)
        if isinstance(noise_scale, float):
            noise_array = np.random.normal(0, noise_scale, size=noise_matrix_shape)

        # Array of flux uncertainty
        else:
            # Rescale to the box format
            input_err = enbox_spectrum(noise_scale, self.box_width, self.range_box)
            noise_array = np.random.normal(0, input_err[..., None], size=noise_matrix_shape)

        # Add the noise
        flux_array = flux_array[:, :, np.newaxis] + noise_array

        # flux_0 = flux_array * 1
        # fluxMC_0 = flux_mc[:, :, 0]
        # noise_0 = noise_array[:, :, 0]
        # np.all(np.isclose(fluxMC_0 - noise_0, flux_0, atol=0.0000000000001))

        return flux_array

    def feature_detection(self, flux_array_1d, flux_array_2d, model_1d, model_2d, n_pixels):

        # Perform the line 1D detection
        self.line_1d_pred = FeatureClassifier(flux_array_1d, self.box_width, model_1d, n_pixels)

        # Perform the line 2d detection
        self.line_2d_pred = FeatureClassifier(flux_array_2d, self.box_width, model_2d, n_pixels)

        return

    def redshift(self, detection_bands=None, z_step_resolution=10000, z_max=10):

        # Get spectra and its mask
        wave_obs = self._spec.wave.data
        flux_obs = self._spec.flux.data
        mask = ~self._spec.flux.mask if np.ma.isMaskedArray(self._spec.flux) else np.ones(flux_obs.size)
        flux_scaled = (flux_obs - np.nanmin(flux_obs)) / (np.nanmax(flux_obs) - np.nanmin(flux_obs))

        # Compute the resolution params
        deltalamb_arr = np.diff(wave_obs)
        R_arr = wave_obs[1:] / deltalamb_arr
        FWHM_arr = wave_obs[1:] / R_arr
        sigma_arr = np.zeros(wave_obs.size)
        sigma_arr[:-1] = FWHM_arr / (2 * np.sqrt(2 * np.log(2)))
        sigma_arr[-1] = sigma_arr[-2]
        sigma_arr = sigma_arr * 2

        # Lines selection
        ref_lines = ['H1_1216A', 'He2_1640A', 'O2_3726A', 'H1_4340A', 'H1_4861A', 'O3_4959A', 'O3_5007A',
                     'H1_6563A', 'S3_9530A', 'He1_10830A']
        theo_lambda = _PARENT_BANDS.loc[ref_lines].wavelength.to_numpy()

        # Parameters for the brute analysis
        z_arr = np.linspace(0, z_max, z_step_resolution)
        wave_matrix = np.tile(wave_obs, (theo_lambda.size, 1))
        F_sum = np.zeros(z_arr.size)

        # Use the dectection bands if provided
        detection_weight = np.zeros(wave_obs.size)
        if detection_bands is not None:
            detec_obj = getattr(self._spec.infer, detection_bands)
            idcs = detec_obj(80)
            detection_weight[idcs] = 1
            mask = mask & idcs
        else:
            detection_weight = np.ones(wave_obs.size)
            idcs = detection_weight.astype(bool)
            mask = mask & idcs

        for i, z_i in enumerate(z_arr):

            # Generate the redshift key
            gauss_arr = compute_z_key(z_i, theo_lambda, wave_matrix, 1, sigma_arr)

            # Compute flux cumulative sum
            F_sum[i] = 0 if gauss_arr is None else np.sum(flux_obs[mask] * gauss_arr[mask])

        z_max = z_arr[np.argmax(F_sum)]

        # fig, ax = plt.subplots()
        # ax.scatter(wave_obs[mask], flux_obs[mask])
        # # ax.step(wave_obs[idcs], flux_obs[idcs], where='mid')
        # plt.show()

        # # Plot the addition:
        # fig, ax = plt.subplots()
        # ax.step(z_arr, F_sum, where='mid', color='tab:blue')
        # ax.axvline(0.0475, color='red', linestyle='--', alpha=0.5)
        # ax.axvline(z_max, color='blue', linestyle='--', alpha=0.5)
        # plt.show()

        # Plot keys at max:
        # gauss_arr_max = compute_z_key(z_max, theo_lambda, wave_matrix, 1, sigma_arr)
        # gauss_arr_true = compute_z_key(0.0475, theo_lambda, wave_matrix, 1, sigma_arr)
        # F_sum_max = np.sum(flux_obs[mask] * gauss_arr_max[mask])
        # F_sum_true = np.sum(flux_obs[mask] * gauss_arr_true[mask])
        # fig, ax = plt.subplots()
        # ax.step(wave_obs[mask], flux_scaled[mask], where='mid', color='tab:blue')
        # ax.step(wave_obs[mask], gauss_arr_max[mask], where='mid', color='tab:orange')
        # plt.show()

        # fig, ax = plt.subplots()
        # ax.plot(self._spec.wave, sigma_arr)
        # plt.show()

        return z_max


# Line finder default parameters
LINE_DETECT_PARAMS = signature(LineFinder.line_detection).parameters
LINE_DETECT_PARAMS = {key: value.default for key, value in LINE_DETECT_PARAMS.items()}
LINE_DETECT_PARAMS.pop('self')
