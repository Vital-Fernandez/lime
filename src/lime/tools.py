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


def label_decomposition(input_lines, recomb_atom=('H1', 'He1', 'He2'), combined_dict={}, scalar_output=False,
                        user_format={}):

    # Confirm input array has one dimension
    input_lines = np.array(input_lines, ndmin=1)

    # Containers for input data
    ion_dict, wave_dict, latexLabel_dict = {}, {}, {}

    for lineLabel in input_lines:
        if lineLabel not in user_format:
            # Check if line reference corresponds to blended component
            mixture_line = False
            if '_b' in lineLabel or '_m' in lineLabel:
                mixture_line = True
                if lineLabel in combined_dict:
                    lineRef = combined_dict[lineLabel]
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


def compute_line_width(idx_peak, spec_flux, delta_i, min_delta=2):
    """
    Algororithm to measure emision line width given its peak location
    :param idx_peak:
    :param spec_flux:
    :param delta_i:
    :param min_delta:
    :return:
    """

    i = idx_peak
    while (spec_flux[i] > spec_flux[i + delta_i]) or (np.abs(idx_peak - (i + delta_i)) <= min_delta):
        i += delta_i

    return i


def kinematic_component_labelling(line_latex_label, comp_ref):

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


# def match_lines(wave_rest, flux,  obsLineTable, maskDF, lineType='emission', tol=5, blendedLineList=[],
#                 detect_check=False, find_line_borders='Auto', include_unknown=False):
#
#     #TODO maybe we should remove not detected from output
#     theoLineDF = pd.DataFrame.copy(maskDF)
#
#     # Query the lines from the astropy finder tables # TODO Expand technique for absorption lines
#     idcsLineType = obsLineTable['line_type'] == lineType
#     idcsLinePeak = np.array(obsLineTable[idcsLineType]['line_center_index'])
#     waveObs = wave_rest[idcsLinePeak]
#
#     # Theoretical wave values
#     ion_array, waveTheory, latexLabel_array = label_decomposition(theoLineDF.index.values)
#     theoLineDF['wavelength'] = waveTheory
#     # waveTheory = theoLineDF.wavelength.values
#
#     # Match the lines with the theoretical emission
#     tolerance = np.diff(wave_rest).mean() * tol
#     theoLineDF['observation'] = 'not detected'
#     unidentifiedLine = dict.fromkeys(theoLineDF.columns.values, np.nan)
#
#     for i in np.arange(waveObs.size):
#
#         idx_array = np.where(np.isclose(a=waveTheory, b=waveObs[i], atol=tolerance))
#
#         if len(idx_array[0]) == 0:
#             unknownLineLabel = 'xy_{:.0f}A'.format(np.round(waveObs[i]))
#
#             # Scheme to avoid repeated lines
#             if (unknownLineLabel not in theoLineDF.index) and detect_check:
#                 newRow = unidentifiedLine.copy()
#                 newRow.update({'wavelength': waveObs[i], 'w3': waveObs[i] - 5, 'w4': waveObs[i] + 5,
#                                'observation': 'not identified'})
#                 theoLineDF.loc[unknownLineLabel] = newRow
#
#         else:
#
#             row_index = theoLineDF.index[theoLineDF.wavelength == waveTheory[idx_array[0][0]]]
#             theoLineDF.loc[row_index, 'observation'] = 'detected'
#             theoLineLabel = row_index[0]
#
#             # TODO lines like Halpha+[NII] this does not work, we should add exclusion
#             if find_line_borders == True:
#                 minSeparation = 4 if theoLineLabel in blendedLineList else 2
#                 idx_min = compute_line_width(idcsLinePeak[i], flux, delta_i=-1, min_delta=minSeparation)
#                 idx_max = compute_line_width(idcsLinePeak[i], flux, delta_i=1, min_delta=minSeparation)
#                 theoLineDF.loc[row_index, 'w3'] = wave_rest[idx_min]
#                 theoLineDF.loc[row_index, 'w4'] = wave_rest[idx_max]
#             else:
#                 if find_line_borders == 'Auto':
#                     if '_b' not in theoLineLabel:
#                         minSeparation = 4 if theoLineLabel in blendedLineList else 2
#                         idx_min = compute_line_width(idcsLinePeak[i], flux, delta_i=-1, min_delta=minSeparation)
#                         idx_max = compute_line_width(idcsLinePeak[i], flux, delta_i=1, min_delta=minSeparation)
#                         theoLineDF.loc[row_index, 'w3'] = wave_rest[idx_min]
#                         theoLineDF.loc[row_index, 'w4'] = wave_rest[idx_max]
#
#     if include_unknown is False:
#         idcs_unknown = theoLineDF['observation'] == 'not detected'
#         theoLineDF.drop(index=theoLineDF.loc[idcs_unknown].index.values, inplace=True)
#
#     # Sort by wavelength
#     theoLineDF.sort_values('wavelength', inplace=True)
#
#     # Latex labels
#     ion_array, wavelength_array, latexLabel_array = label_decomposition(theoLineDF.index.values)
#     theoLineDF['latexLabel'] = latexLabel_array
#     theoLineDF['blended_label'] = 'None'
#
#     return theoLineDF
#
#
# def line_finder(wave_rest, flux, noiseWaveLim, intLineThreshold=3, verbose=False):
#
#     assert noiseWaveLim[0] > wave_rest[0] or noiseWaveLim[1] < wave_rest[-1]
#
#     # Establish noise values
#     idcs_noiseRegion = (noiseWaveLim[0] <= wave_rest) & (wave_rest <= noiseWaveLim[1])
#     noise_region = SpectralRegion(noiseWaveLim[0] * WAVE_UNITS_DEFAULT, noiseWaveLim[1] * WAVE_UNITS_DEFAULT)
#     flux_threshold = intLineThreshold * flux[idcs_noiseRegion].std()
#
#     input_spectrum = Spectrum1D(flux * FLUX_UNITS_DEFAULT, wave_rest * WAVE_UNITS_DEFAULT)
#     input_spectrum = noise_region_uncertainty(input_spectrum, noise_region)
#     linesTable = find_lines_derivative(input_spectrum, flux_threshold)
#
#     if verbose:
#         print(linesTable)
#
#     return linesTable
#
#
# def continuum_remover(wave_rest, flux, noiseRegionLims, intLineThreshold=((4, 4), (1.5, 1.5)), degree=(3, 7)):
#
#     assert wave_rest[0] < noiseRegionLims[0] and noiseRegionLims[1] < wave_rest[-1], \
#         f'Error noise region {wave_rest[0]} < {noiseRegionLims[0]} and {noiseRegionLims[1]} < {wave_rest[-1]}'
#
#     # Identify high flux regions
#     idcs_noiseRegion = (noiseRegionLims[0] <= wave_rest) & (wave_rest <= noiseRegionLims[1])
#     noise_mean, noise_std = flux[idcs_noiseRegion].mean(), flux[idcs_noiseRegion].std()
#
#     # Perform several continuum fits to improve the line detection
#     input_wave, input_flux = wave_rest, flux
#     for i in range(len(intLineThreshold)):
#
#         # Mask line regions
#         emisLimit = intLineThreshold[i][0] * (noise_mean + noise_std)
#         absoLimit = (noise_mean + noise_std) / intLineThreshold[i][1]
#         idcsLineMask = np.where((input_flux >= absoLimit) & (input_flux <= emisLimit))
#         wave_masked, flux_masked = input_wave[idcsLineMask], input_flux[idcsLineMask]
#
#         # Perform continuum fits iteratively
#         poly3Mod = PolynomialModel(prefix=f'poly_{degree[i]}', degree=degree[i])
#         poly3Params = poly3Mod.guess(flux_masked, x=wave_masked)
#         poly3Out = poly3Mod.fit(flux_masked, poly3Params, x=wave_masked)
#
#         input_flux = input_flux - poly3Out.eval(x=wave_rest) + noise_mean
#
#     return input_flux - noise_mean


class LineFinder:

    def __init__(self):

        return

    def match_line_mask(self, mask_df, noise_region, line_threshold=3, emis_abs_thresholds=((4, 4), (1.5, 1.5)),
                        cont_degree_list=(3, 7), width_tol=5, width_mode='auto'):

        # Convert the observed region to the observed frame
        noise_region_obs = noise_region * (1 + self.redshift)

        # Remove the continuum
        flux_no_continuum = self.remove_continuum(noise_region_obs, emis_abs_thresholds, cont_degree_list)

        # Find the emission, absorption peaks
        peaks_table = self.peak_indexing(flux_no_continuum, noise_region_obs, line_threshold)

        # Match peaks with theoretical lines
        matched_DF = self.label_peaks(peaks_table, mask_df, width_tol=width_tol, width_mode=width_mode)

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

        # Query the lines from the astropy finder tables # TODO Expand technique for absorption lines
        idcsLineType = peak_table['line_type'] == line_type
        idcsLinePeak = np.array(peak_table[idcsLineType]['line_center_index'])
        wave_peaks = self.wave_rest[idcsLinePeak]

        # Theoretical wave values
        ion_array, waveTheory, latexLabel_array = label_decomposition(matched_DF.index.values)
        matched_DF['wavelength'] = waveTheory
        # waveTheory = theoLineDF.wavelength.values

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
                        idx_min = compute_line_width(idcsLinePeak[i], self.flux, delta_i=-1, min_delta=minSeparation)
                        idx_max = compute_line_width(idcsLinePeak[i], self.flux, delta_i=1, min_delta=minSeparation)
                        matched_DF.loc[row_index, 'w3'] = self.wave_rest[idx_min]
                        matched_DF.loc[row_index, 'w4'] = self.wave_rest[idx_max]

                # Width is computed for every line
                else:
                    idx_min = compute_line_width(idcsLinePeak[i], self.flux, delta_i=-1, min_delta=minSeparation)
                    idx_max = compute_line_width(idcsLinePeak[i], self.flux, delta_i=1, min_delta=minSeparation)
                    matched_DF.loc[row_index, 'w3'] = self.wave_rest[idx_min]
                    matched_DF.loc[row_index, 'w4'] = self.wave_rest[idx_max]


        # if include_unknown is False:
        #     idcs_unknown = theoLineDF['observation'] == 'not detected'
        #     theoLineDF.drop(index=theoLineDF.loc[idcs_unknown].index.values, inplace=True)

        # Sort by wavelength
        matched_DF.sort_values('wavelength', inplace=True)

        # Latex labels
        ion_array, wavelength_array, latexLabel_array = label_decomposition(matched_DF.index.values)
        matched_DF['latexLabel'] = latexLabel_array

        return matched_DF