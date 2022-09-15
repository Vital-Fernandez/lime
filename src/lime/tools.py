import logging
import numpy as np
import pandas as pd
import joblib
import astropy.units as au
from lmfit.models import PolynomialModel
from sys import exit
from pathlib import Path
from scipy import signal

_logger = logging.getLogger('LiMe')

try:
    from specutils import Spectrum1D, SpectralRegion
    from specutils.manipulation import noise_region_uncertainty
    from specutils.fitting import find_lines_derivative
    specutils_check = True

except ImportError:
    specutils_check = False

VAL_LIST = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
SYB_LIST = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]

ASTRO_UNITS_KEYS = {'A': au.AA,
                    'um': au.um,
                    'nm': au.nm,
                    'Hz': au.Hz,
                    'cm': au.cm,
                    'mm': au.mm,
                    'Flam': au.erg/au.s/au.cm**2/au.AA,
                    'Fnu': au.erg/au.s/au.cm**2/au.Hz,
                    'Jy': au.Jy,
                    'mJy': au.mJy,
                    'nJy': au.nJy}

UNITS_LATEX_DICT = {'A': '\AA',
                    'um': '\mu\!m',
                    'nm': 'nm',
                    'Hz': 'Hz',
                    'cm': 'cm',
                    'mm': 'mm',
                    'Flam': r'erg\,cm^{-2}s^{-1}\AA^{-1}',
                    'Fnu': r'erg\,cm^{-2}s^{-1}\Hz^{-1}',
                    'Jy': 'Jy',
                    'mJy': 'mJy',
                    'nJy': 'nJy'}

DISPERSION_UNITS = ('A', 'um', 'nm', 'Hz', 'cm', 'mm')

FLUX_DENSITY_UNITS = ('Flam', 'Fnu', 'Jy', 'mJy', 'nJy')

WAVE_UNITS_DEFAULT, FLUX_UNITS_DEFAULT = au.AA, au.erg / au.s / au.cm ** 2 / au.AA

MACHINE_PATH = Path(__file__).parent/'resources'/'LogitistRegression_v2_cost1_logNorm.joblib'

_REFERENCE_LINES = np.array(['H1_1216A', 'C4_1549A', 'He2_1640A', 'O3]_1666A', 'C3]_1980A', 'Mg2_2800A', '[O3]_3727A',
                             '[Ne3]_3869A', 'H1_3889A', 'H1_3970A', 'H1_4341A',
                             'H1_4861A', '[O3]_5007A', 'H1_6563A', 'S3_9069A', 'S3_9531A', 'H1_10050A', 'He1_10830A', 'H1_10940A',  'H1_12820A',
                              'Fe2_16440A', 'H1_18756A', 'H1_19450A', 'Si6_19620A', 'He1_20590A', 'H2_21210A', 'H1_21661A', 'H2_24070A',
                              'Si6_24830A', 'H1_26259A',  'H1_32970A', 'PAH1_33000A', 'H1_37406A', 'H1_40522A', 'H1_46539A'])


#[Y erg/cm^2/s/A] = 2.99792458E+21 * [X1 W/m^2/Hz] / [X2 A]^2
# 2.99792458E+17 units
# 1 Jy = 10−26 W⋅m−2⋅Hz−1
# 1 Jy = 10−23 erg⋅s−1⋅cm−2⋅Hz−1
# 1 nm = 2.99792458E+17


def int_to_roman(num):
    i, roman_num = 0, ''
    while num > 0:
        for _ in range(num // VAL_LIST[i]):
            roman_num += SYB_LIST[i]
            num -= VAL_LIST[i]
        i += 1
    return roman_num


def label_decomposition(lines, recomb_atom=('H1', 'He1', 'He2'), comp_dict={}, scalar_output=False,
                        user_format={}, units_wave='A'):

    """

    This function returns the ion, wavelength and transition label latex from the input line list with the LiMe line
    notation.

    :param lines: A string or array of strings with the LiMe transition notation, e.g. O3_5007A
    :type lines: str, list

    :param recomb_atom: An array with the ions producing photons from a recombination process. By default the function
                        assumes that these are H1, He1 and He2 while the metal ions produce photons from a collisional
                        excited state.
    :type recomb_atom: str, list

    :param comp_dict: Dictionary with the user latex format for the latex labels, overwritting the default notation.
    :type comp_dict: dict, optional

    :param scalar_output: Boolean for a scalar output in case of a single input line input.
    :type scalar_output: bool, optional

    :param user_format: Dictionary with the user notation for the latex labels. This overwrites the default notation.
    :type user_format: dict, optional

    :param units_wave: Label wavelength units. The default value "A" is angstrom.
    :type units_wave: str, optional

    :return: 3 arrays (or scalars) with the input transition line(s) ion, wavelength and scientific notation in latex format.
    :rtype: numpy.ndarray

    :Example:
        >>> import lime
        >>> lime.label_decomposition('O3_5007A', scalar_output=True)
        O3, 5007.0, '$5007\\AA\\,[OIII]$'
        >>> lime.label_decomposition('H1_6563A_b', comp_dict={"H1_6563A_b":"H1_6563A-N2_6584A-N2_6548A"})
        ['H1'], [6563.], ['$6563\\AA\\,HI+6584\\AA\\,[NII]+6548\\AA\\,[NII]$']

    """

    # Confirm input array has one dimension
    # TODO for blended lines it may be better to return all the blended components individually
    lines = np.array(lines, ndmin=1)

    # TODO current workflow breaks if repeated labels
    try:
        uniq, count = np.unique(lines, return_counts=True)
        assert not np.any(count > 1)
    except AssertionError as err:
        _logger.critical('The input line list has repeated line names')
        raise err

    # Containers for input data
    ion_dict, wave_dict, latexLabel_dict = {}, {}, {}

    for lineLabel in lines:

        # Case the user provides his own format
        if lineLabel in user_format:
            ion_dict[lineLabel], wave_dict[lineLabel], latexLabel_dict[lineLabel] = user_format[lineLabel]

        # Default format
        else:

            # Check if line reference corresponds to blended component
            mixture_line = False
            if (lineLabel[-2:] == '_b') or (lineLabel[-2:] == '_m'):
                mixture_line = True
                if lineLabel in comp_dict:
                    lineRef = comp_dict[lineLabel]
                else:
                    lineRef = lineLabel[:-2]
            else:
                lineRef = lineLabel

            # Split the components if they exists
            lineComponents = lineRef.split('-')

            # Decomponse each component
            latexLabel = ''
            for line_i in lineComponents:

                # Check that the line has the right units
                kinem_comp_check = False if line_i.count('_') == 1 else True

                # Get ion:
                ion = line_i[0:line_i.find('_')]
                square_brackets = [bracket for bracket in ion if bracket in ['[', ']']]
                n_brackets = len(square_brackets)

                # if 'r_' in line_i: # Case recombination lines
                #     ion = line_i[0:line_i.find('_')-1]
                # else:
                #     ion = line_i[0:line_i.find('_')]

                # Get wavelength and their units # TODO add more units and more facilities for extensions
                # TODO warning if label does not have those units
                if (line_i.endswith(units_wave)) or kinem_comp_check:
                    wavelength = line_i[line_i.find('_') + 1:line_i.rfind(units_wave)]
                    units = UNITS_LATEX_DICT[units_wave]
                    ext = f'-{line_i[line_i.rfind("_")+1:]}' if kinem_comp_check else ''
                else:
                    wavelength = line_i[line_i.find('_') + 1:]
                    units = ''
                    ext = ''

                # Define the label # TODO Remove first check and make the anotation compulsary # Add option to exclude brackets or provide status, ionization element appart
                if n_brackets == 0:

                    atom, ionization = ion[:-1], int(ion[-1])
                    ionizationRoman = int_to_roman(ionization)

                    if ion in recomb_atom:
                        comp_line = f'{atom}{ionizationRoman}'
                    else:
                        comp_line = f'[{atom}{ionizationRoman}]'

                # Forbidden line
                elif n_brackets == 2:
                    atom, ionization = ion[1:-2], int(ion[-2])
                    ionizationRoman = int_to_roman(ionization)
                    comp_line = f'[{atom}{ionizationRoman}]'

                # Semi-forbidden line
                else:
                    atom, ionization = ion[0:-2], int(ion[-2])
                    ionizationRoman = int_to_roman(ionization)
                    comp_line = f'{atom}{ionizationRoman}]'

                # Adding the wavelength and units
                comp_line += f'\,{wavelength}{units}{ext}'

                # In the case of a mixture line we take component with the _b as the parent
                if mixture_line:
                    if lineLabel[:-2] == line_i:
                        ion_dict[lineRef] = ion
                        wave_dict[lineRef] = float(wavelength)
                        latexLabel = comp_line if len(latexLabel) == 0 else f'{latexLabel}+{comp_line}'
                    else:
                        latexLabel = comp_line if len(latexLabel) == 0 else f'{latexLabel}+{comp_line}'

                # This logic will expand the blended lines, but the output list will be larger than the input one
                else:
                    ion_dict[line_i] = ion
                    wave_dict[line_i] = float(wavelength)
                    latexLabel_dict[line_i] = '$'+comp_line+'$'

            if mixture_line:
                latexLabel_dict[lineRef] = '$'+latexLabel +'$'

    # Convert to arrays
    label_array = np.array([*ion_dict.keys()], ndmin=1)
    ion_array = np.array([*ion_dict.values()], ndmin=1)
    wavelength_array = np.array([*wave_dict.values()], ndmin=1)
    latexLabel_array = np.array([*latexLabel_dict.values()], ndmin=1)

    # Check if the number of output lines is the same
    if not label_array.size == wavelength_array.size:
        _logger.critical(f'Number of input lines is different from number of output lines')

    # If requested and single line, return the input as a scalar
    if ion_array.size == 1 and scalar_output:
        output = (ion_array[0], wavelength_array[0], latexLabel_array[0])
    else:
        output = (ion_array, wavelength_array, latexLabel_array)

    return output


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


def compute_FWHM0(idx_peak, spec_flux, delta_wave, cont_flux, emission_check=True):

    """

    :param idx_peak:
    :param spec_flux:
    :param delta_wave:
    :param cont_flux:
    :param emission_check:
    :return:
    """

    i = idx_peak
    i_final = 0 if delta_wave < 0 else spec_flux.size - 1

    if emission_check:
        while (spec_flux[i] >= cont_flux[i]) and (i != i_final):
            i += delta_wave
    else:
        while (spec_flux[i] <= cont_flux[i]) and (i != i_final):
            i += delta_wave

    return i


def blended_label_from_log(line, log):

    # Default values: single line
    blended_check = False

    if line in log.index:

        if log.loc[line, 'profile_label'] == 'no':
            profile_label = 'no'
        elif line.endswith('_m'):
            profile_label = log.loc[line, 'profile_label']
        else:
            blended_check = True
            profile_label = log.loc[line, 'profile_label']
    else:
        # TODO this causes and error if we forget the '_b' componentes in the configuration file need to check input cfg
        _logger.warning(f'The line {line} was not found on the input log. If you are specifying the components of a '
                        f'blended line in the fitting configuration, make sure you are not missing the "_b" subscript')

    return blended_check, profile_label


def latex_science_float(f, dec=2):
    float_str = f'{f:.{dec}g}'
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str


def unit_convertor(in_units, out_units, wave_array=None, flux_array=None, dispersion_units=None, sig_fig=None,
                   mask_check=False):

    # Converting the wavelength array
    if (in_units in DISPERSION_UNITS) and (out_units in DISPERSION_UNITS):
        input_mask = wave_array.mask if mask_check else None
        input_array = wave_array * ASTRO_UNITS_KEYS[in_units]
        output_array = input_array.to(ASTRO_UNITS_KEYS[out_units])

    # Converting the flux array
    elif (in_units in FLUX_DENSITY_UNITS) and (out_units in FLUX_DENSITY_UNITS):
        input_mask = flux_array.mask if mask_check else None
        input_array = flux_array * ASTRO_UNITS_KEYS[in_units]
        wave_unit_array = wave_array * ASTRO_UNITS_KEYS[dispersion_units]
        output_array = input_array.to(ASTRO_UNITS_KEYS[out_units], au.spectral_density(wave_unit_array))

    # Not recognized units
    else:
        _logger.warning(f'Input units {in_units} could not be converted to {out_units}')

    # Reapply the mask if necessary
    if mask_check:
        output_array = np.ma.masked_array(output_array.value, input_mask)
    else:
        output_array = output_array.value

    if sig_fig is None:
        return output_array
    else:
        return np.round(output_array, sig_fig)


def refraction_index_air_vacuum(wavelength_array, units='A'):

    refraction_index = (1 + 1e-6 * (287.6155 + 1.62887 / np.power(wavelength_array * 0.0001, 2) + 0.01360 / np.power(wavelength_array * 0.0001, 4)))

    return refraction_index


def air_to_vacuum_function(input_array, sig_fig=0):

    input_array = np.array(input_array, ndmin=1)

    if 'U' in str(input_array.dtype): #TODO finde better way
        ion_array, wave_array, latex_array = label_decomposition(input_array)
        air_wave = wave_array
    else:
        air_wave = input_array

    refraction_index = (1 + 1e-6 * (287.6155 + 1.62887/np.power(air_wave*0.0001, 2) + 0.01360/np.power(air_wave*0.0001, 4)))
    output_array = (air_wave * 0.0001 * refraction_index) * 10000

    if sig_fig == 0:
        output_array = np.round(output_array, sig_fig) if sig_fig != 0 else np.round(output_array, sig_fig).astype(int)

    if 'U' in str(input_array.dtype):
        vacuum_wave = output_array.astype(str)
        output_array = np.core.defchararray.add(ion_array, '_')
        output_array = np.core.defchararray.add(output_array, vacuum_wave)
        output_array = np.core.defchararray.add(output_array, 'A')

    return output_array


def spectral_mask_generator(wave_interval=None, lines_list=None, ion_list=None, z_range=None,
                            parent_mask_address=None, units_wave='A', sig_fig=4, vacuum_conversion=False):

    # Use the default lime mask if none provided
    if parent_mask_address is None:
        parent_mask_address = Path(__file__).parent/'resources'/'parent_mask.txt'

    # Check that the file exists: #TODO remove blended, merged extensions from parent mask (or recover them here)
    if not Path(parent_mask_address).is_file():
        _logger.warning(f'Parent mask file not found. The input user address was {parent_mask_address}')

    # Load the parent mask
    mask_df = pd.read_csv(parent_mask_address, delim_whitespace=True, header=0, index_col=0)

    # Recover line label components
    ion_array, wave_array, latex_array = label_decomposition(mask_df.index.values)
    idcs_rows = np.ones(mask_df.index.size).astype(bool)

    # Convert to vacuum wavelengths if requested
    if vacuum_conversion:

        # First the table data
        mask_df = mask_df.apply(air_to_vacuum_function)

        # The line headers
        wave_array = air_to_vacuum_function(wave_array)
        if sig_fig is not None:
            wave_array = np.round(wave_array, sig_fig) if sig_fig != 0 else np.round(wave_array, sig_fig).astype(int)

        # Reconstruct the line labels for the new units
        labels_array = np.core.defchararray.add(ion_array, '_')
        labels_array = np.core.defchararray.add(labels_array, wave_array.astype(str))
        labels_array = np.core.defchararray.add(labels_array, units_wave)

        # Rename the indeces
        mask_df.rename(index=dict(zip(mask_df.index.values, labels_array)), inplace=True)

    # Momentarily add columns with the transition wavelength, ion and latex notation
    mask_df['wave'], mask_df['ion'], mask_df['latex'] = wave_array, ion_array, latex_array

    # Convert the output mask to the units requested by the user
    if units_wave != 'A':

        conversion_factor = unit_convertor('A', units_wave,  wave_array=1, dispersion_units='dispersion axis', sig_fig=sig_fig)
        mask_df.loc[:, 'w1':'w6'] = mask_df.loc[:, 'w1':'w6'] * conversion_factor
        wave_array = wave_array * conversion_factor

        if sig_fig is not None:
            wave_array = np.round(wave_array, sig_fig) if sig_fig != 0 else np.round(wave_array, sig_fig).astype(int)

        # Reconstruct the line labels for the new units
        labels_array = np.core.defchararray.add(ion_array, '_')
        labels_array = np.core.defchararray.add(labels_array, wave_array.astype(str))
        labels_array = np.core.defchararray.add(labels_array, units_wave)

        # Rename the indeces
        mask_df.rename(index=dict(zip(mask_df.index.values, labels_array)), inplace=True)

    # First slice by wavelength and redshift
    if wave_interval is not None:

        # Establish the lower and upper wavelenght limits
        if np.ma.isMaskedArray(wave_interval):
            w_min, w_max = wave_interval.data[0], wave_interval.data[-1]
        else:
            w_min, w_max = wave_interval[0], wave_interval[-1]

        if z_range is not None:
            z_range = np.array(z_range, ndmin=1)
            w_min, w_max = w_min * (1 + z_range[0]), w_max * (1 + z_range[-1])

        idcs_rows = idcs_rows & (wave_array >= w_min) & (wave_array <= w_max)

    # Second slice by ion
    if ion_list is not None:
        idcs_rows = idcs_rows & mask_df.ion.isin(ion_list)

    # Finally slice by the name of the lines
    if lines_list is not None:
        idcs_rows = idcs_rows & mask_df.index.isin(lines_list)

    # Return the sliced the parent mask
    output_mask = mask_df.loc[idcs_rows, 'w1':'w6']

    return output_mask


def format_line_mask_option(entry_value, wave_array):

    # Check if several entries
    formatted_value = entry_value.split(',') if ',' in entry_value else [f'{entry_value}']

    # Check if interval or single pixel mask
    for i, element in enumerate(formatted_value):
        if '-' in element:
            formatted_value[i] = element.split('-')
        else:
            element = float(element)
            pix_width = (np.diff(wave_array).mean())/2
            formatted_value[i] = [element-pix_width, element+pix_width]

    formatted_value = np.array(formatted_value).astype(float)

    return formatted_value


def define_masks(wavelength_array, masks_array, merge_continua=True, line_mask_entry='no'):

    # Make sure it is a matrix
    masks_array = np.array(masks_array, ndmin=2)

    # Check if it is a masked array
    if np.ma.is_masked(wavelength_array):
        wave_arr = wavelength_array.data
    else:
        wave_arr = wavelength_array

    # Remove masked pixels from this function wavelength array
    if line_mask_entry != 'no':

        # Convert cfg mask string to limits
        line_mask_limits = format_line_mask_option(line_mask_entry, wave_arr)

        # Get masked indeces
        idcsMask = (wave_arr[:, None] >= line_mask_limits[:, 0]) & (wave_arr[:, None] <= line_mask_limits[:, 1])
        idcsValid = ~idcsMask.sum(axis=1).astype(bool)[:, None]

    else:
        idcsValid = np.ones(wave_arr.size).astype(bool)[:, None]

    # Find indeces for six points in spectrum
    idcsW = np.searchsorted(wave_arr, masks_array)

    # Emission region
    idcsLineRegion = ((wave_arr[idcsW[:, 2]] <= wave_arr[:, None]) & (wave_arr[:, None] <= wave_arr[idcsW[:, 3]]) & idcsValid).squeeze()
    
    # Return left and right continua merged in one array
    if merge_continua:

        idcsContRegion = (((wave_arr[idcsW[:, 0]] <= wave_arr[:, None]) &
                          (wave_arr[:, None] <= wave_arr[idcsW[:, 1]])) |
                          ((wave_arr[idcsW[:, 4]] <= wave_arr[:, None]) & (
                           wave_arr[:, None] <= wave_arr[idcsW[:, 5]])) & idcsValid).squeeze()

        return idcsLineRegion, idcsContRegion

    # Return left and right continua in separated arrays
    else:

        idcsContLeft = ((wave_arr[idcsW[:, 0]] <= wave_arr[:, None]) & (wave_arr[:, None] <= wave_arr[idcsW[:, 1]]) & idcsValid).squeeze()
        idcsContRight = ((wave_arr[idcsW[:, 4]] <= wave_arr[:, None]) & (wave_arr[:, None] <= wave_arr[idcsW[:, 5]]) & idcsValid).squeeze()

        return idcsLineRegion, idcsContLeft, idcsContRight


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

        # Loop throught the fitting degree
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
                self._plot_continuum_fit(continuum_full, mask_cont, low_lim, high_lim, threshold_list[i], title)

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

    def line_detection(self, poly_degree=[3, 7, 7, 7], emis_threshold=[5, 3, 2, 2], noise_sigma_factor=3, lines_log=None,
                       line_type='emission', width_tol=5, width_mode='fixed', ml_detection=False, plot_cont_calc=False,
                       plot_peak_calc=False):

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
        if lines_log is not None:

            # Match peaks with theoretical lines
            matched_DF = self.label_peaks(idcs_peaks, lines_log, width_tol=width_tol, width_mode=width_mode,
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
        ion_array, waveTheory, latexLabel_array = label_decomposition(matched_DF.index.values, units_wave=self.units_wave)
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

            idx_array = np.where(np.isclose(a=waveTheory, b=wave_peaks[i], atol=tolerance))

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
