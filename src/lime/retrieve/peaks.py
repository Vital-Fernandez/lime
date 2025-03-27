# import logging
#
# import numpy as np
# import pandas as pd
#
# from scipy import signal
# from ..io import check_file_dataframe
# from ..transitions import label_decomposition
#
#
# try:
#     import joblib
#     joblib_check = True
# except ImportError:
#     joblib_check = False
#
#
# _logger = logging.getLogger('LiMe')
#
# MACHINE_PATH = None
# FLUX_PIXEL_CONV = np.linspace(0,1,33)
#
#
# def compute_line_width(idx_peak, spec_flux, delta_i, min_delta=2, emission_check=True):
#
#     """
#     Algororithm to measure emision line width given its peak location
#     :param idx_peak:
#     :param spec_flux:
#     :param delta_i:
#     :param min_delta:
#     :return:
#     """
#
#     i = idx_peak
#
#     if emission_check:
#         while (spec_flux[i] > spec_flux[i + delta_i]) or (np.abs(idx_peak - (i + delta_i)) <= min_delta):
#             i += delta_i
#     else:
#         while (spec_flux[i] < spec_flux[i + delta_i]) or (np.abs(idx_peak - (i + delta_i)) <= min_delta):
#             i += delta_i
#
#     return i
#
#
#
#
# class LineFinder:
#
#     def __init__(self):
#
#         return
#
#     def line_detection(self, bands, sigma_threshold=3, emission_type=True, width_tol=5, band_modification=None,
#                        continuum_array=None, continuum_std=None, plot_steps=False):
#
#         """
#
#         This function compares the input lines bands in the observation spectrum to confirm the presence of lines.
#
#         The input bands can be specified as a pandas dataframe or the path to its file via the ``bands_df`` argument.
#
#         The continuum needs to be fit a priori with the Spectrum.fit.continuum function or assigning a ``continuum_array``
#         and a ``continuum_std``.
#
#         The ``sigma_threshold`` establishes the standard deviation factor beyond which a positive line detection is assumed.
#
#         By default the algorithm seeks for emission lines, set ``emission_type`` equal to False for absorption lines.
#
#         The additional arguments provide additional utils to adjust the line detection and show the steps/results.
#
#         :param bands: Input bands dataframe or the address to its file.
#         :type bands: pandas.Dataframe, str, pathlib.Path
#
#         :param sigma_threshold: Continuum standard deviation factor for line detection. The default value is 3.
#         :type sigma_threshold: float, optional
#
#         :param emission_type: Line type. The default value is "True" for emission lines.
#         :type emission_type: str, optional
#
#         :param width_tol: Minimum number of pixels between peaks/troughs. The default value is 5.
#         :type width_tol: float, optional
#
#         :param band_modification: Method to adjust the line band with. The default value is None.
#         :type band_modification: str, optional
#
#         :param ml_detection: Machine learning algorithm to detect lines. The default value is None.
#         :type ml_detection: str, optional
#
#         :param plot_steps: Plot the detected peaks/troughs. The default value is False
#         :type plot_steps: bool, optional
#
#         """
#
#         # TODO Lime2.0 replace by warning with new retrieve line bands
#         # Check for the peaks of the emission lines
#         continuum_array = self.cont if continuum_array is None else continuum_array
#         continuum_std = self.cont_std if continuum_std is None else continuum_std
#
#         # Get indeces of peaks
#         limit_threshold = sigma_threshold * continuum_std
#         limit_threshold = continuum_array + limit_threshold if emission_type else continuum_array + limit_threshold
#         idcs_peaks, _ = signal.find_peaks(self.flux, height=limit_threshold, distance=width_tol)
#
#         # Match peaks with theoretical lines
#         bands = check_file_dataframe(bands)
#         matched_DF = self.label_peaks(idcs_peaks, bands, width_tol=width_tol, band_modification=band_modification,
#                                       line_type=emission_type)
#
#         # Plot the results
#         if plot_steps:
#             self.plot._plot_peak_detection(idcs_peaks, limit_threshold, continuum_array, matched_DF)
#
#         return matched_DF
#
#
#     def label_peaks(self, peak_table, mask_df, line_type='emission', width_tol=5, band_modification=None, detect_check=False):
#
#         # TODO auto param should be changed to boolean
#         # Establish the type of input values for the peak indexes, first numpy array
#         if isinstance(peak_table, np.ndarray):
#             idcsLinePeak = peak_table
#
#         # Specutils table
#         else:
#             # Query the lines from the astropy finder archives #
#             if len(peak_table) != 0:
#                 idcsLineType = peak_table['emission_type'] == line_type
#                 idcsLinePeak = np.array(peak_table[idcsLineType]['line_center_index'])
#             else:
#                 idcsLinePeak = np.array([])
#
#         # Security check in case no lines detected
#         if len(idcsLinePeak) == 0:
#             return pd.DataFrame(columns=mask_df.columns)
#
#         # Exclude bands not withing the regime:
#         w0, wf = self.wave_rest.data[~self.wave_rest.mask][0],  self.wave_rest.data[~self.wave_rest.mask][-1]
#         idcs_selection = (mask_df.w3 > w0) & (mask_df.w4 < wf)
#
#         # Prepare dataframe to stored the matched lines
#         matched_DF = mask_df.loc[idcs_selection].copy()
#         matched_DF['signal_peak'] = np.nan
#
#         # Theoretical wave values
#         waveTheory = label_decomposition(matched_DF.index.values, params_list=['wavelength'])[0]
#         matched_DF['wavelength'] = waveTheory
#
#         # Match the lines with the theoretical emission
#         tolerance = np.diff(self.wave_rest).mean() * width_tol
#         matched_DF['observation'] = 'not_detected'
#         unidentifiedLine = dict.fromkeys(matched_DF.columns.values, np.nan)
#
#         # Get the wavelength peaks
#         wave_peaks = self.wave_rest[idcsLinePeak]
#
#         for i in np.arange(wave_peaks.size):
#
#             idx_array = np.where(np.isclose(a=waveTheory.astype(np.float64), b=wave_peaks[i], atol=tolerance))
#
#             if len(idx_array[0]) == 0:
#                 unknownLineLabel = 'xy_{:.0f}A'.format(np.round(wave_peaks[i]))
#
#                 # Scheme to avoid repeated lines
#                 if (unknownLineLabel not in matched_DF.index) and detect_check:
#                     newRow = unidentifiedLine.copy()
#                     newRow.update({'wavelength': wave_peaks[i], 'w3': wave_peaks[i] - 5, 'w4': wave_peaks[i] + 5,
#                                    'observation': 'not_identified'})
#                     matched_DF.loc[unknownLineLabel] = newRow
#
#             else:
#
#                 row_index = matched_DF.index[matched_DF.wavelength == waveTheory[idx_array[0][0]]]
#                 matched_DF.loc[row_index, 'observation'] = 'detected'
#                 matched_DF.loc[row_index, 'signal_peak'] = idcsLinePeak[i]
#                 theoLineLabel = row_index[0]
#
#                 blended_check = True if '_b' in theoLineLabel else False
#                 minSeparation = 4 if blended_check else 2
#
#                 # Width is only computed for blended lines
#                 if band_modification is not None:
#                     if band_modification == 'auto':
#                         if blended_check is False:
#                             emission_check = True if line_type == 'emission' else False
#                             idx_min = compute_line_width(idcsLinePeak[i], self.flux, delta_i=-1, min_delta=minSeparation, emission_check=emission_check)
#                             idx_max = compute_line_width(idcsLinePeak[i], self.flux, delta_i=1, min_delta=minSeparation, emission_check=emission_check)
#                             matched_DF.loc[row_index, 'w3'] = self.wave_rest[idx_min]
#                             matched_DF.loc[row_index, 'w4'] = self.wave_rest[idx_max]
#
#         # Include_only_detected
#         idcs_unknown = matched_DF['observation'] == 'not_detected'
#         matched_DF.drop(index=matched_DF.loc[idcs_unknown].index.values, inplace=True)
#
#         # Sort by wavelength
#         matched_DF.sort_values('wavelength', inplace=True)
#         matched_DF.drop(columns=['wavelength', 'observation'], inplace=True)
#
#         # Security check all bands are within selection
#         wave_peak_matched = self.wave_rest[matched_DF.signal_peak.to_numpy().astype(int)]
#         idcs_valid = (matched_DF.w3 < wave_peak_matched) & (matched_DF.w4 > wave_peak_matched)
#         matched_DF = matched_DF.loc[idcs_valid]
#
#         return matched_DF