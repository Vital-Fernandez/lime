import logging

import numpy as np
import pandas as pd

from scipy import signal
from lime.io import check_file_dataframe, LiMe_Error
from lime.transitions import label_decomposition
from lime.plotting.plots import spec_peak_calculation

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




class LineFinder:

    def __init__(self):

        return

    def peaks_troughs(self, bands, sigma_threshold=3, emission_shape=True, width_tol=5,
                      continuum_array=None, continuum_std=None, plot_steps=False, **kwargs):

        """

        This function compares the input lines bands in the observation spectrum to confirm the presence of lines.

        The input bands can be specified as a pandas dataframe or the path to its file via the ``bands_df`` argument.

        The continuum needs to be fit a priori with the Spectrum.fit.continuum function or assigning a ``continuum_array``
        and a ``continuum_std``.

        The ``sigma_threshold`` establishes the standard deviation factor beyond which a positive line detection is assumed.

        By default the algorithm seeks for emission lines, set ``emission_type`` equal to False for absorption lines.

        The additional arguments provide additional utils to adjust the line detection and show the steps/results.

        :param bands: Input bands dataframe or the address to its file.
        :type bands: pandas.Dataframe, str, pathlib.Path

        :param sigma_threshold: Continuum standard deviation factor for line detection. The default value is 3.
        :type sigma_threshold: float, optional

        :param emission_shape: Line type. The default value is "True" for emission lines.
        :type emission_shape: str, optional

        :param width_tol: Minimum number of pixels between peaks/troughs. The default value is 5.
        :type width_tol: float, optional

        :param ml_detection: Machine learning algorithm to detect lines. The default value is None.
        :type ml_detection: str, optional

        :param plot_steps: Plot the detected peaks/troughs. The default value is False
        :type plot_steps: bool, optional

        """

        # TODO Lime2.0 replace by warning with new retrieve line bands
        # Check for the peaks of the emission lines
        continuum_array = self._spec.cont if continuum_array is None else continuum_array
        continuum_std = self._spec.cont_std if continuum_std is None else continuum_std

        # Check the continuum has been calculated in advance
        if (continuum_array is None) or (continuum_std is None):
            raise LiMe_Error(f'Please provide a continuum and std array to the function or run "Spectrum.fit.continuum"'
                             f' prior to the peak detection')

        # Get indeces of peaks
        limit_threshold = sigma_threshold * continuum_std
        limit_threshold = continuum_array + limit_threshold if emission_shape else continuum_array + limit_threshold
        idcs_peaks, _ = signal.find_peaks(self._spec.flux, height=limit_threshold, distance=width_tol)

        # Match peaks with theoretical lines
        bands = check_file_dataframe(bands)
        matched_DF = self.label_peaks(idcs_peaks, bands, width_tol=width_tol, line_type=emission_shape)

        # Plot the results
        if plot_steps:
            spec_peak_calculation(self._spec, matched_DF, limit_threshold, idcs_peaks, continuum_array, **kwargs)

        return matched_DF


    def label_peaks(self, idcs_peaks, bands_df, line_type='emission', width_tol=5):

        # Security check in case no lines detected
        if len(idcs_peaks) == 0 or bands_df.index.size == 0:
            return pd.DataFrame(columns=bands_df.columns)

        # Add theoretical wavelength values if necessary
        if 'wavelength' not in bands_df.columns:
            bands_df['wavelength'] = label_decomposition(bands_df.index.values, params_list=['wavelength'], verbose=False)[0]

        # Get bands limits indexes
        idcs_w3 = np.searchsorted(self._spec.wave_rest, bands_df.w3)
        idcs_w4 = np.searchsorted(self._spec.wave_rest, bands_df.w4)

        # Get the bands matching
        band_contains_peak = (idcs_peaks[None, :] > idcs_w3[:, None]) & (idcs_peaks[None, :] < idcs_w4[:, None])
        idcs_matched_bands = band_contains_peak.any(axis=1)
        idcs_matched_peaks = idcs_peaks[band_contains_peak.argmax(axis=1)[idcs_matched_bands]]

        # Crop the bands to the detection
        bands_df.loc[idcs_matched_bands, 'observation'] = 'detected'
        bands_df.loc[idcs_matched_bands, 'signal_peak'] = idcs_matched_peaks

        return bands_df.loc[idcs_matched_bands]