import numpy as np
import pandas as pd
from lmfit.models import PolynomialModel

import astropy.units as au
from specutils import Spectrum1D, SpectralRegion
from specutils.manipulation import noise_region_uncertainty
from specutils.fitting import find_lines_derivative

VAL_LIST = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
SYB_LIST = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]

WAVE_UNITS_DEFAULT, FLUX_UNITS_DEFAULT = au.AA, au.erg / au.s / au.cm ** 2 / au.AA


def int_to_roman(num):
    i, roman_num = 0, ''
    while num > 0:
        for _ in range(num // VAL_LIST[i]):
            roman_num += SYB_LIST[i]
            num -= VAL_LIST[i]
        i += 1
    return roman_num


def label_decomposition(input_lines, recomb_atom=('H1', 'He1', 'He2'), blended_dict={}, scalar_output=False,
                        user_format={}):

    """
    This function returns the wavelength, ion and the standard transition latex from the default LiMe line notation.

    :param input_lines: A string or array of strings with the lime transition notation, e.g. O3_5007A
    :type input_lines: str, list

    :param recomb_atom: An array with the ions producing photons from a recombination process. By default the function
                        assumes that these are H1, He1 and He2 while the metal ions produce photons from a collisional
                        excited state
    :type recomb_atom: str, list

    :param blended_dict: Dictionary with the user latex format for the latex labels, overwritting the default notation.
    :type blended_dict: dict, optional

    :param scalar_output: Boolean for a scalar output in case of a single input line input
    :type scalar_output: bool, optional

    :param user_format: Dictionary with the user notation for the latex labels. This overwrites the default notation.
    :type user_format: dict, optional

    :return: 3 arrays (or scalars) with the input line(s) transition ion, wavelength and scientific notation in latex format.
    :rtype: numpy.ndarray

    :Example:
        >>> import lime
        >>> lime.label_decomposition('O3_5007A', scalar_output=True)
        O3, 5007.0, '$5007\\AA\\,[OIII]$'
        >>> lime.label_decomposition('H1_6563A_b', blended_dict={"H1_6563A_b":"H1_6563A-N2_6584A-N2_6548A"})
        ['H1'], [6563.], ['$6563\\AA\\,HI+6584\\AA\\,[NII]+6548\\AA\\,[NII]$']

    """



    # Confirm input array has one dimension
    # TODO for blended lines it may be better to return all the blended components individually
    input_lines = np.array(input_lines, ndmin=1)

    # TODO current workflow breaks if repeated labels
    uniq, count = np.unique(input_lines, return_counts=True)
    assert not np.any(count > 1), '- ERROR: The input line labels in lime.label_decomposition includes repeated entries, please remove them'

    # Containers for input data
    ion_dict, wave_dict, latexLabel_dict = {}, {}, {}

    for lineLabel in input_lines:
        if lineLabel not in user_format:
            # Check if line reference corresponds to blended component
            mixture_line = False
            if '_b' in lineLabel or '_m' in lineLabel:
                mixture_line = True
                if lineLabel in blended_dict:
                    lineRef = blended_dict[lineLabel]
                else:
                    lineRef = lineLabel[:-2]
            else:
                lineRef = lineLabel

            # Split the components if they exists
            lineComponents = lineRef.split('-')

            # Decomponse each component
            latexLabel = ''
            for line_i in lineComponents:

                # Get ion:
                if 'r_' in line_i: # Case recombination lines
                    ion = line_i[0:line_i.find('_')-1]
                else:
                    ion = line_i[0:line_i.find('_')]

                # Get wavelength and their units # TODO add more units and more facilities for extensions
                ext_n = line_i.count('_')
                if (line_i.endswith('A')) or (ext_n > 1):
                    wavelength = line_i[line_i.find('_') + 1:line_i.rfind('A')]
                    units = '\AA'
                    ext = f'-{line_i[line_i.rfind("_")+1:]}' if ext_n > 1 else ''
                else:
                    wavelength = line_i[line_i.find('_') + 1:]
                    units = ''
                    ext = ''

                # Get classical ion notation
                atom, ionization = ion[:-1], int(ion[-1])
                ionizationRoman = int_to_roman(ionization)

                # Define the label
                if ion in recomb_atom:
                    comp_Label = wavelength + units + '\,' + atom + ionizationRoman + ext
                else:
                    comp_Label = wavelength + units + '\,' + '[' + atom + ionizationRoman + ']' + ext

                # In the case of a mixture line we take the first entry as the reference
                if mixture_line:
                    if len(latexLabel) == 0:
                        ion_dict[lineRef] = ion
                        wave_dict[lineRef] = float(wavelength)
                        latexLabel += comp_Label
                    else:
                        latexLabel += '+' + comp_Label

                # This logic will expand the blended lines, but the output list will be larger than the input one
                else:
                    ion_dict[line_i] = ion
                    wave_dict[line_i] = float(wavelength)
                    latexLabel_dict[line_i] = '$'+comp_Label+'$'

            if mixture_line:
                latexLabel_dict[lineRef] = '$'+latexLabel +'$'

        else:
            ion_dict[lineLabel], wave_dict[lineLabel], latexLabel_dict[lineLabel] = user_format[lineLabel]

    # Convert to arrays
    label_array = np.array([*ion_dict.keys()], ndmin=1)
    ion_array = np.array([*ion_dict.values()], ndmin=1)
    wavelength_array = np.array([*wave_dict.values()], ndmin=1)
    latexLabel_array = np.array([*latexLabel_dict.values()], ndmin=1)

    assert label_array.size == wavelength_array.size, 'Output ions do not match wavelengths size'
    assert label_array.size == latexLabel_array.size, 'Output ions do not match labels size'

    if ion_array.size == 1 and scalar_output:
        return ion_array[0], wavelength_array[0], latexLabel_array[0]
    else:
        return ion_array, wavelength_array, latexLabel_array


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


def kinematic_component_labelling(line_latex_label, comp_ref):
    """

    :param line_latex_label:
    :param comp_ref:
    :return:
    """
    if len(comp_ref) != 2:
        print(f'-- Warning: Components label for {line_latex_label} is {comp_ref}. Code only prepare for a 2 character description (ex. n1, w2...)')

    number = comp_ref[-1]
    letter = comp_ref[0]

    if letter in ('n', 'w'):
        if letter == 'n':
            comp_label = f'Narrow {number}'
        if letter == 'w':
            comp_label = f'Wide {number}'
    else:
        comp_label = f'{letter}{number}'

    if '-' in line_latex_label:
        lineEmisLabel = line_latex_label.replace(f'-{comp_ref}', '')
    else:
        lineEmisLabel = line_latex_label

    return comp_label, lineEmisLabel


class LineFinder:

    def __init__(self):

        return

    def match_line_mask(self, mask_df, noise_region, line_threshold=3, emis_abs_thresholds=((4, 4), (1.5, 1.5)),
                        cont_degree_list=(3, 7), width_tol=5, line_type='emission', width_mode='auto'):
        """

        :param mask_df: pandas DataFrame with the line masks.
        :param noise_region: This region describes the
        :param line_threshold:
        :param emis_abs_thresholds:
        :param cont_degree_list:
        :param width_tol:
        :param line_type:
        :param width_mode:
        :return:
        """
        # Convert the observed region to the observed frame
        noise_region_obs = noise_region * (1 + self.redshift)

        # Remove the continuum
        flux_no_continuum = self.remove_continuum(noise_region_obs, emis_abs_thresholds, cont_degree_list)

        # Find the emission, absorption peaks
        peaks_table = self.peak_indexing(flux_no_continuum, noise_region_obs, line_threshold)

        # Match peaks with theoretical lines
        matched_DF = self.label_peaks(peaks_table, mask_df, width_tol=width_tol, width_mode=width_mode,
                                      line_type=line_type)

        return peaks_table, matched_DF

    def remove_continuum(self, noise_region, emis_abs_threshold, cont_degree_list):

        assert self.wave[0] < noise_region[0] and noise_region[1] < self.wave[-1], \
            f'Error noise region {self.wave[0]/(1+self.redshift)} < {noise_region[0]/(1+self.redshift)} ' \
            f'and {noise_region[1]/(1+self.redshift)} < {self.wave[-1]/(1+self.redshift)}'

        # Wavelength range and flux to use
        input_wave, input_flux = self.wave, self.flux # TODO does this modify my flux

        # Identify high flux regions
        idcs_noiseRegion = (noise_region[0] <= input_wave) & (input_wave <= noise_region[1])
        noise_mean, noise_std = input_flux[idcs_noiseRegion].mean(), input_flux[idcs_noiseRegion].std()

        # Perform several continuum fits to improve the line detection
        for i in range(len(emis_abs_threshold)):
            # Mask line regions
            emisLimit = emis_abs_threshold[i][0] * (noise_mean + noise_std)
            absoLimit = (noise_mean + noise_std) / emis_abs_threshold[i][1]
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

        matched_DF = pd.DataFrame.copy(mask_df)

        # TODO auto param should be changed to boolean

        # Query the lines from the astropy finder tables #
        idcsLineType = peak_table['line_type'] == line_type
        idcsLinePeak = np.array(peak_table[idcsLineType]['line_center_index'])
        wave_peaks = self.wave_rest[idcsLinePeak]

        # Theoretical wave values
        ion_array, waveTheory, latexLabel_array = label_decomposition(matched_DF.index.values)
        matched_DF['wavelength'] = waveTheory

        # Match the lines with the theoretical emission
        tolerance = np.diff(self.wave_rest).mean() * width_tol
        matched_DF['observation'] = 'not detected'
        unidentifiedLine = dict.fromkeys(matched_DF.columns.values, np.nan)

        for i in np.arange(wave_peaks.size):

            idx_array = np.where(np.isclose(a=waveTheory, b=wave_peaks[i], atol=tolerance))

            if len(idx_array[0]) == 0:
                unknownLineLabel = 'xy_{:.0f}A'.format(np.round(wave_peaks[i]))

                # Scheme to avoid repeated lines
                if (unknownLineLabel not in matched_DF.index) and detect_check:
                    newRow = unidentifiedLine.copy()
                    newRow.update({'wavelength': wave_peaks[i], 'w3': wave_peaks[i] - 5, 'w4': wave_peaks[i] + 5,
                                   'observation': 'not identified'})
                    matched_DF.loc[unknownLineLabel] = newRow

            else:

                row_index = matched_DF.index[matched_DF.wavelength == waveTheory[idx_array[0][0]]]
                matched_DF.loc[row_index, 'observation'] = 'detected'
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

                # Else leave the values already there
                # else:
                #     idx_min = compute_line_width(idcsLinePeak[i], self.flux, delta_i=-1, min_delta=minSeparation)
                #     idx_max = compute_line_width(idcsLinePeak[i], self.flux, delta_i=1, min_delta=minSeparation)
                #     matched_DF.loc[row_index, 'w3'] = self.wave_rest[idx_min]
                #     matched_DF.loc[row_index, 'w4'] = self.wave_rest[idx_max]

        # if include_unknown is False:
        #     idcs_unknown = theoLineDF['observation'] == 'not detected'
        #     theoLineDF.drop(index=theoLineDF.loc[idcs_unknown].index.values, inplace=True)

        # Sort by wavelength
        matched_DF.sort_values('wavelength', inplace=True)

        # Latex labels
        ion_array, wavelength_array, latexLabel_array = label_decomposition(matched_DF.index.values)
        matched_DF['latexLabel'] = latexLabel_array

        return matched_DF