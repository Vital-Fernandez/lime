import numpy as np
import pandas as pd
import logging

from pathlib import Path
from scipy import signal
from lmfit.models import PolynomialModel
from inspect import signature
from .io import LiMe_Error, check_file_dataframe
from .transitions import label_decomposition
import astropy.units as au

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


class LineFinder:

    def __init__(self, machine_model_path=MACHINE_PATH):

        # self.ml_model = joblib.load(machine_model_path) # THIS CAN be warning at opening the file

        return

    def continuum_fitting(self, degree_list=[3, 7, 7, 7], threshold_list=[5, 3, 2, 2], plot_results=False,
                          return_std=False):

        # Check for a masked array
        if np.ma.is_masked(self.flux):
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

    def peak_detection(self, limit_threshold=None, continuum=None, distance=4, ml_mask=None, plot_results=False):

        # No user imput provided compute the intensity threshold from the 84th percentil
        limit_threshold = np.percentile(self.flux, 84) if limit_threshold is None else limit_threshold
        limit_threshold = limit_threshold + continuum if continuum is not None else limit_threshold

        peak_fp, _ = signal.find_peaks(self.flux, height=limit_threshold, distance=distance)

        # Plot the results
        if plot_results:
            self.plot._plot_peak_detection(peak_fp, limit_threshold, continuum, ml_mask=ml_mask,
                                      plot_title='Peak detection results ')

        return peak_fp

    def ml_line_detection(self, continuum, box_width=11, model= None):

        if model is None:
            model = joblib.load(MACHINE_PATH)

        # Normalize the flux
        input_flux = self.flux if not np.ma.is_masked(self.flux) else self.flux.data
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

    def line_detection(self, bands=None, cont_fit_degree=(3, 7, 7, 7), cont_int_thres=(5, 3, 2, 2), noise_sigma_factor=3,
                       line_type='emission', width_tol=5, band_modification=None, ml_detection=None, plot_cont_calc=False,
                       plot_peak_calc=False):

        """

        This function compares the input lines bands in the observation spectrum to confirm the presence of lines.

        The input bands can be specified as a pandas dataframe or the path to its file via the ``bands_df`` argument.

        Prior to the line detection, the spectrum continuum is fit in an iterative process. The ``cont_fit_degree`` array
        provides the order of the fitting polynomial, while the ``cont_int_thres`` array provides the threshold intensity
        factor for the threshold flux above and below the continuum to exclude at each iteration.

        After the continuum has been normalized, the ``noise_sigma_factor`` establishes the standard deviation factor
        beyond which a positive line detection is assumed.

        The additional arguments provide additional tools to adjust the line detection and show the steps/results.

        :param bands: Input bands dataframe or the address to its file.
        :type bands: pandas.Dataframe, str, pathlib.Path

        :param cont_fit_degree: Continuum polynomial fitting degree.
        :type cont_fit_degree: tuple, optional

        :param cont_int_thres: Continuum maximum intensity threshold to include pixels.
        :type cont_int_thres: tuple, optional

        :param noise_sigma_factor: Continuum standard deviation factor for line detection. The default value is 3.
        :type noise_sigma_factor: float, optional

        :param line_type: Line type. The default value is "emission".
        :type line_type: str, optional

        :param width_tol: Minimum number of pixels between peaks/troughs. The default value is 5.
        :type width_tol: float, optional

        :param band_modification: Method to adjust the line band with. The default value is None.
        :type band_modification: str, optional

        :param ml_detection: Machine learning algorithm to detect lines. The default value is None.
        :type ml_detection: str, optional

        :param plot_cont_calc: Plot the continuum fit at each iteration. The default value is False
        :type plot_cont_calc: bool, optional

        :param plot_peak_calc: Plot the detected peaks/troughs. The default value is False
        :type plot_peak_calc: bool, optional

        """

        # TODO input log should not be None... Maybe read database

        # Fit the continuum
        cont_flux, cond_Std = self.continuum_fitting(degree_list=cont_fit_degree, threshold_list=cont_int_thres,
                                                     return_std=True, plot_results=plot_cont_calc)

        # Check via machine learning algorithm
        if ml_detection is not None:
            if joblib_check:
                self.ml_model = joblib.load(MACHINE_PATH)
                ml_mask = self.ml_line_detection(cont_flux) if ml_detection else None
            else:
                raise ImportError(f'Need to install joblib library to use machine learning detection')
        else:
            ml_mask = None

        # Check for the peaks of the emission lines
        detec_min = noise_sigma_factor * cond_Std
        idcs_peaks = self.peak_detection(detec_min, cont_flux, plot_results=plot_peak_calc, ml_mask=ml_mask)

        # Compare against the theoretical values
        bands = check_file_dataframe(bands, pd.DataFrame)
        if bands is not None:

            # Match peaks with theoretical lines
            matched_DF = self.label_peaks(idcs_peaks, bands, width_tol=width_tol, band_modification=band_modification,
                                          line_type=line_type)

            return matched_DF

        else:

            return

    def label_peaks(self, peak_table, mask_df, line_type='emission', width_tol=5, band_modification=None, detect_check=False):

        # TODO auto param should be changed to boolean
        # Establish the type of input values for the peak indexes, first numpy array
        if isinstance(peak_table, np.ndarray):
            idcsLinePeak = peak_table

        # Specutils table
        else:
            # Query the lines from the astropy finder tables #
            if len(peak_table) != 0:
                idcsLineType = peak_table['line_type'] == line_type
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

        # from matplotlib import pyplot as plt
        # fig, ax = plt.subplots()
        # ax.step(self.wave_rest, self.flux, where='mid')
        # ax.scatter(self.wave_rest[idcsLinePeak], self.flux[idcsLinePeak])
        # plt.show()
        # np.ma.isMaskedArray(wave_peaks)

        # # Only treat pixels outisde the masks
        # if np.ma.is_masked(wave_peaks):
        #     idcsLinePeak = idcsLinePeak[~wave_peaks.mask]
        #     wave_peaks = wave_peaks[~wave_peaks.mask].data

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


class DetectionInference:

    def __init__(self, spectrum):

        self._spec = spectrum
        self.narrow_detect = None

        return

    def bands(self, box_width, approximation=None, scale_type='min-max', machine_path=None, log_base=10000):

        # Assign default values
        approximation = FLUX_PIXEL_CONV if approximation is None else approximation
        machine_path = MACHINE_PATH if machine_path is None else machine_path

        # Load the model
        model = joblib.load(machine_path)
        model_dim = model.n_features_in_

        # Recover the flux (without mask?)
        input_flux = self._spec.flux if not np.ma.is_masked(self._spec.flux) else self._spec.flux.data

        # Reshape to the detection interval
        range_box = np.arange(box_width)
        n_intervals = input_flux.size - box_width + 1
        input_flux = input_flux[np.arange(n_intervals)[:, None] + range_box]

        # Remove nan entries
        idcs_nan_rows = np.isnan(input_flux).any(axis=1)
        input_flux = input_flux[~idcs_nan_rows, :]

        # Add Monte Carlo columns (7858, 13) To (7858, 13, 100)


        # Normalize the flux
        input_flux = feature_scaling(input_flux, transformation=scale_type, log_base=log_base)

        # Perform the 1D detection
        if box_width == model_dim:
            detection_array = model.predict(input_flux)
        else:
            array_2D = np.tile(input_flux[:, None, :], (1, approximation.size, 1))
            array_2D = array_2D > approximation[::-1, None]
            array_2D = array_2D.astype(int)
            array_2D = array_2D.reshape((input_flux.shape[0], 1, -1))
            array_2D = array_2D.squeeze()
            detection_array = model.predict(array_2D)

        # Reshape array original shape and add with of positive entries # TODO this shape could cause issues with IFUs
        mask = np.zeros(self._spec.flux.shape, dtype=bool)
        idcs_detect = np.argwhere(detection_array) + range_box
        idcs_detect = idcs_detect.flatten()
        idcs_detect = idcs_detect[idcs_detect < detection_array.size]
        mask[idcs_detect] = True

        return mask


# Line finder default parameters
LINE_DETECT_PARAMS = signature(LineFinder.line_detection).parameters
LINE_DETECT_PARAMS = {key: value.default for key, value in LINE_DETECT_PARAMS.items()}
LINE_DETECT_PARAMS.pop('self')
