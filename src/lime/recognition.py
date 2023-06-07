import numpy as np
import pandas as pd
import joblib
import logging

from pathlib import Path
from scipy import signal
from sys import exit
from lmfit.models import PolynomialModel
from inspect import signature
from .io import LiMe_Error
from .transitions import label_decomposition
import astropy.units as au

_logger = logging.getLogger('LiMe')

MACHINE_PATH = Path(__file__).parent/'resources'/'LogitistRegression_v2_cost1_logNorm.joblib'

# TODO replace this mechanic
WAVE_UNITS_DEFAULT, FLUX_UNITS_DEFAULT = au.AA, au.erg / au.s / au.cm ** 2 / au.AA


try:
    from specutils import Spectrum1D, SpectralRegion
    from specutils.manipulation import noise_region_uncertainty
    from specutils.fitting import find_lines_derivative
    specutils_check = True

except ImportError:
    specutils_check = False


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
            low_lim, high_lim = np.percentile(input_flux[mask_cont], (16, 84))
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

        # Index the intensity peaks
        # input_flux = self.flux.data if np.ma.is_masked()
        # mask_valid = ~self.flux.mask if np.ma.is_masked(self.flux) else np.ones(self.flux.data.size).astype(bool)
        # peak_fp, _ = signal.find_peaks(self.flux.data[mask_valid], height=limit_threshold[mask_valid], distance=distance)

        peak_fp, _ = signal.find_peaks(self.flux, height=limit_threshold, distance=distance)

        # Plot the results
        if plot_results:
            self._plot_peak_detection(peak_fp, limit_threshold, continuum, ml_mask=ml_mask,
                                      plot_title='Peak detection results ')

        return peak_fp

    def ml_line_detection(self, continuum, box_width=11):

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
                    detection_mask[i:i + box_width] = detection_mask[i:i + box_width] | self.ml_model.predict(y)[0]
                    # print(f'y {i} ({np.sum(y)}): {self.ml_model.predict(y)[0]}')

        return detection_mask

    def line_detection(self, input_log=None, poly_degree=[3, 7, 7, 7], emis_threshold=[5, 3, 2, 2], noise_sigma_factor=3,
                       line_type='emission', width_tol=5, width_mode='fixed', ml_detection=False, plot_cont_calc=False,
                       plot_peak_calc=False):

        # TODO input log should not be None... Maybe read database

        # Fit the continuum
        cont_flux, cond_Std = self.continuum_fitting(degree_list=poly_degree, threshold_list=emis_threshold,
                                                     return_std=True, plot_results=plot_cont_calc)

        # Check via machine learning algorithm
        if ml_detection:
            self.ml_model = joblib.load(MACHINE_PATH)
            ml_mask = self.ml_line_detection(cont_flux) if ml_detection else None
        else:
            ml_mask = None

        # Check for the peaks of the emission lines
        detec_min = noise_sigma_factor * cond_Std
        idcs_peaks = self.peak_detection(detec_min, cont_flux, plot_results=plot_peak_calc, ml_mask=ml_mask)

        # Compare against the theoretical values
        if input_log is not None:

            # Match peaks with theoretical lines
            matched_DF = self.label_peaks(idcs_peaks, input_log, width_tol=width_tol, width_mode=width_mode,
                                          line_type=line_type)

            return matched_DF

        else:

            return

    def match_line_mask(self, log, noise_region, detect_threshold=3, emis_threshold=(4, 4), abs_threshold=(1.5, 1.5),
                        poly_degree=(3, 7), width_tol=5, line_type='emission', width_mode='fixed'):

        """
        This function compares a spectrum flux peaks and troughs with the input lines mask log to confirm the presence of
        emission or absorption lines. The user can specify the line type with the ``line_type='emission'`` or ``line_type='absorption'``
        parameter.

        The user must specify a wavelength range (in the rest _frame) to establish the region from which the spectrum
        noise standard deviation is calculated. This region must not have absorptions or emissions.

        The treatment requires a normalized spectrum (continuum at zero). This is done by fitting the spectrum
        as a polynomial function in an iterative process. The user can specify in the ``poly_degree`` parameter as an
        an array with values in increasing magnitude for the polynomial order. The user should specify ``emis_threshold``
        and ``abs_threshold`` for the emission and absorptions intensity threshold. This masking is necessary to avoid
        intense emission/absorptions affecting the continuum normalization.

        Afterwards, the task runs the `find_lines_derivative <https://specutils.readthedocs.io/en/stable/api/specutils.fitting.find_lines_derivative.html>`_
        function to find peaks and troughs in the normalized spectrum. The intensity threshold in this detection is read
        from the ``detect_threshold`` parameter. The output table from this function is the first return of this task.

        In the next step, the task compares the input lines ``log`` bands with the peaks/troughs location from the `find_lines_derivative <https://specutils.readthedocs.io/en/stable/api/specutils.fitting.find_lines_derivative.html>`_
        output. Those matching the line band limits (w3, w4 in the ``log``) plus the tolerance in the ``width_tol``
        parameter (in the spectrum wavelength units) count as a positive detection. These lines are the second return
        as a new log.

        Finally, the task can attempt to adjust the line band width to the width of the emission/absorption feature.
        In the ``width_mode='auto'`` the non-blended lines w3, w4 values in the output log band will be changed to the
        first pixel wavelength, starting from the maximum/minimum at which there is an increase/decrease in flux intensity
        for emission and absorption lines respectively. In blended lines, the w3, w4 values are not modified.
        In the ``width_mode='fix'`` setting the line masks wavelengths are not modified.

        :param log: Lines log with the masks. The required columns are: the line label (DF index), w1, w2, w3, w4, w5 and w6.
                    These wavelengths must be in the rest _frame.
        :type log: pandas.DataFrame

        :param noise_region: 2 value array with the wavelength limits for the noise region (in rest _frame).
        :type noise_region: numpy.array

        :param detect_threshold: Intensity factor for the continuum signal for an emission/absorption detection.
        :type detect_threshold: float, optional

        :param emis_threshold: Array with the intensity factor for the emission features masking during the continuum normalization.
        :type emis_threshold: numpy.array, optional

        :param abs_threshold: Array with the intensity factor for the absorption features masking during the continuum normalization.
        :type abs_threshold: numpy.array, optional

        :param poly_degree: Array with the polynomial order for the iterative fitting for the continuum normalization in increasing order.
        :type poly_degree: numpy.array, optional

        :param width_tol: Tolerance for the peak/trough detection with respect to the input line masks w3 and w4 values.
        :type width_tol: float, optional

        :param line_type: Type of lines matched in the output lines log. Accepted values are 'emission' and 'absorption'
        :type line_type: str, optional

        :param width_mode: Scheme for the line band mask detection. If set to "fixed" the input w3 and w4 values won't be modified.
        :type width_tol: str, optional

        :return: Table with the peaks/trough detected and log with the matched lines.
        :rtype: astropy.Table and pandas.DataFrame

        """

        # Remove the continuum
        if specutils_check:

            # Convert the noise region to the the observed _frame
            noise_region_obs = noise_region * (1 + self.redshift)

            # Normalize the continuum to zero
            flux_no_continuum = self.remove_continuum(noise_region_obs, emis_threshold, abs_threshold, poly_degree)

            # Find the emission, absorption peaks
            peaks_table = self.peak_indexing(flux_no_continuum, noise_region_obs, detect_threshold)

            # Match peaks with theoretical lines
            matched_DF = self.label_peaks(peaks_table, log, width_tol=width_tol, width_mode=width_mode,
                                          line_type=line_type)

            return peaks_table, matched_DF

        else:
            exit(f'\n- WARNING: specutils is not installed')
            return None, None

    def remove_continuum(self, noise_region, emis_threshold, abs_threshold, cont_degree_list):

        assert self.wave[0] < noise_region[0] and noise_region[1] < self.wave[-1], \
            f'Error noise region {self.wave[0]/(1+self.redshift)} < {noise_region[0]/(1+self.redshift)} ' \
            f'and {noise_region[1]/(1+self.redshift)} < {self.wave[-1]/(1+self.redshift)}'

        # Wavelength range and flux to use
        input_wave, input_flux = self.wave, self.flux

        # Identify high flux regions
        idcs_noiseRegion = (noise_region[0] <= input_wave) & (input_wave <= noise_region[1])
        noise_mean, noise_std = input_flux[idcs_noiseRegion].mean(), input_flux[idcs_noiseRegion].std()

        # Perform several continuum fits to improve the line detection
        for i in range(len(emis_threshold)):

            # Mask line regions
            emisLimit = emis_threshold[i] * (noise_mean + noise_std)
            absoLimit = (noise_mean + noise_std) / abs_threshold[i]
            # emisLimit = emis_threshold[i][0] * (noise_mean + noise_std)
            # absoLimit = (noise_mean + noise_std) / emis_abs_threshold[i][1]
            idcsLineMask = np.where((input_flux >= absoLimit) & (input_flux <= emisLimit))
            wave_masked, flux_masked = input_wave[idcsLineMask], input_flux[idcsLineMask]

            # Perform continuum fits iteratively
            poly3Mod = PolynomialModel(prefix=f'poly_{cont_degree_list[i]}', degree=cont_degree_list[i])
            poly3Params = poly3Mod.guess(flux_masked, x=wave_masked)
            poly3Out = poly3Mod.fit(flux_masked, poly3Params, x=wave_masked)

            input_flux = input_flux - poly3Out.eval(x=input_wave) + noise_mean

        return input_flux - noise_mean

    def peak_indexing(self, flux, noise_region, line_threshold=3):

        assert self.wave[0] < noise_region[0] and noise_region[1] < self.wave[-1], \
            f'Error noise region {self.wave[0]/(1+self.redshift)} < {noise_region[0]/(1+self.redshift)} ' \
            f'and {noise_region[1]/(1+self.redshift)} < {self.wave[-1]/(1+self.redshift)}'

        # Establish noise values
        idcs_noiseRegion = (noise_region[0] <= self.wave) & (self.wave <= noise_region[1])
        noise_region = SpectralRegion(noise_region[0] * WAVE_UNITS_DEFAULT, noise_region[1] * WAVE_UNITS_DEFAULT)
        flux_threshold = line_threshold * flux[idcs_noiseRegion].std()

        input_spectrum = Spectrum1D(flux * FLUX_UNITS_DEFAULT, self.wave * WAVE_UNITS_DEFAULT)
        input_spectrum = noise_region_uncertainty(input_spectrum, noise_region)
        linesTable = find_lines_derivative(input_spectrum, flux_threshold)

        return linesTable

    def label_peaks(self, peak_table, mask_df, line_type='emission', width_tol=5, width_mode='auto', detect_check=False):

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
        waveTheory = label_decomposition(matched_DF.index.values, output_params=['wavelength'])[0]
        matched_DF['wavelength'] = waveTheory

        # Match the lines with the theoretical emission
        tolerance = np.diff(self.wave_rest).mean() * width_tol
        matched_DF['observation'] = 'not_detected'
        unidentifiedLine = dict.fromkeys(matched_DF.columns.values, np.nan)

        # Get the wavelength peaks
        wave_peaks = self.wave_rest[idcsLinePeak]

        # Only treat pixels outisde the masks
        if np.ma.is_masked(wave_peaks):
            idcsLinePeak = idcsLinePeak[~wave_peaks.mask]
            wave_peaks = wave_peaks[~wave_peaks.mask].data

        for i in np.arange(wave_peaks.size):

            idx_array = np.where(np.isclose(a=waveTheory.astype(np.float), b=wave_peaks[i], atol=tolerance))

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
                if width_mode == 'auto':
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


# Line finder default parameters
LINE_DETECT_PARAMS = signature(LineFinder.line_detection).parameters
LINE_DETECT_PARAMS = {key: value.default for key, value in LINE_DETECT_PARAMS.items()}
LINE_DETECT_PARAMS.pop('self')
