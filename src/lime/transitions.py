import logging

import numpy as np
from numpy import array, abs, nan, all, diff, char, searchsorted, unique, empty, arange, zeros
from .tools import DISPERSION_UNITS, UNITS_LATEX_DICT
from .io import _PARENT_BANDS, _LOG_EXPORT, _LOG_COLUMNS, _LOG_DTYPES_REC, results_to_log
from pandas import DataFrame

_logger = logging.getLogger('LiMe')


VAL_LIST = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
SYB_LIST = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]


def int_to_roman(num):
    i, roman_num = 0, ''
    while num > 0:
        for _ in range(num // VAL_LIST[i]):
            roman_num += SYB_LIST[i]
            num -= VAL_LIST[i]
        i += 1
    return roman_num


def check_units_from_wave(str_wave):

    # One character unit
    units = str_wave[-1] if str_wave[-1] in DISPERSION_UNITS else None

    # Two characters unit
    if units is None:
        units = str_wave[-2:] if str_wave[-2:] in DISPERSION_UNITS else None

        # Warning for an unsuccessful notation
        if units is None:
            _logger.warning(f'The units from the transition "{str_wave}" could not be interpreted')
            wave = None
        else:
            wave = float(str_wave[:-2])

    else:
        wave = float(str_wave[:-1])

    return wave, units


def check_line_in_log(input_label, log=None, tol=1):

    # Guess the transition if only a wavelength provided
    if not isinstance(input_label, str):

        if log is not None:

            # Check the wavelength from the wave_obs column
            if ('wave_obs' in log.columns) and ('units_wave' in log.columns):
                ref_waves = log.wave_obs.values
                # units_wave = log.units_wave.values[0] # TODO add units to dataframe

            # Get it from the indexes
            else:
                ion_array, ref_waves = zip(*log.index.str.split('_').to_numpy())
                _wave0, units_wave = check_units_from_wave(ref_waves[0])
                ref_waves = char.strip(ref_waves, units_wave).astype(float)

            # Check if table rows are not sorted
            if not all(diff(ref_waves) >= 0): # TODO we might need to use something else than searchsorted
                _logger.warning(f'The lines log rows are not sorted from lower to higher wavelengths. This can cause '
                                f'issues to identify the lines using the transition wavelength. Try to use the string '
                                f'line label')

            # Locate the best candidate
            idx_closest = searchsorted(ref_waves, input_label)
            line_ion, line_wave = ion_array[idx_closest], ref_waves[idx_closest]

            # Check if input wavelength is close to the guessed line
            disc = abs(line_wave - input_label)
            if tol < disc < 1.5 * tol:
                _logger.info(f'The input line {input_label} has been identified with {log.iloc[idx_closest].name}')

            if disc > 1.5 * tol:
                _logger.warning(f'The closest line to the input line {input_label} is {log.iloc[idx_closest].name}. '
                                f'Please confirm that the database contains the target line and the units matched')

            input_label = log.iloc[idx_closest].name

        else:
            _logger.critical(f'The line "{input_label}" could not be identified: A lines log was not provided')

    return input_label


def label_components(input_label, scalar_output=False):

    # Confirm input array has one dimension
    input_label = array(input_label, ndmin=1)

    # Check there are unique elements
    uniq, count = unique(input_label, return_counts=True)
    if any(count > 1):
        _logger.critical(f'There are repeated entries in the input label: {input_label}')

    ion, wave = empty(input_label.size).astype(str), empty(input_label.size)
    units_wave, kinem = empty(input_label.size).astype(str), zeros(input_label.size, dtype=int)

    for i, label in enumerate(input_label):

        # Transition label items assuming '_' separation
        trans_items = label.split('_')

        # Element producing the transition
        ion[i] = trans_items[0]

        # Wavelength and units
        wave[i], units_wave[i] = check_units_from_wave(trans_items[1])

        # Kinematic element
        kinem[i] = 0
        if len(trans_items) > 2:
            if (trans_items[2][0] == 'w') or (
                    trans_items[2][0] == 'k'):  # TODO make this one always k, maybe a dictionary
                kinem[i] = trans_items[2][1]

    if scalar_output and input_label.size == 1:
        ion, wave, units_wave, kinem = ion[0], wave[0], units_wave[0], kinem[0]

    return ion, wave, units_wave, kinem


def latex_from_label(label, ion=None, wave=None, units_wave=None, kinem=None, recomb_atom=('H1', 'He1', 'He2'),
                     scalar_output=False, comps_dict={}):

    # Use input values if provided else compute from label
    if (ion is None) or (wave is None) or (wave is None):
        ion, wave, units_wave, kinem = label_components(label)

    # Brand to 1d array
    else:
        ion = array(ion, ndmin=1)
        wave = array(wave, ndmin=1)
        units_wave = array(units_wave, ndmin=1)
        kinem = array(kinem, ndmin=1)

    n_items = ion.size
    latex_array = empty(n_items).astype('object')

    for i in arange(n_items):

        atom, ionization = ion[i][:-1], int(ion[i][-1])

        units_latex = UNITS_LATEX_DICT[units_wave[i]]

        ionization_roman = int_to_roman(ionization)

        wave_round = int(wave[i])

        kinetic_comp = '' if kinem[i] == 0 else f'-k{kinem[i]}'

        if ion[i] in recomb_atom:
            latex_array[i] = f'${atom}{ionization_roman}{wave_round}{units_latex}{kinetic_comp}$'
        else:
            latex_array[i] = f'$[{atom}{ionization_roman}]{wave_round}{units_latex}{kinetic_comp}$'

    # Scalar output if requested and 1 length array
    if scalar_output:
        if latex_array.size == 1:
            latex_array = latex_array[0]

    return latex_array


def log_from_line_list(lines, comps_dict, headers=None):

    # TODO this is the new label decomposition
    if headers is None:
        headers = _LOG_DTYPES_REC

    # Minimum one size:
    lines = np.array(lines, ndmin=1)

    # Generate container
    log = DataFrame(np.empty(0, dtype=headers))

    # Loop through the lines
    for i, line in enumerate(lines):
        line_obj = Line(line, fit_conf=comps_dict)
        results_to_log(line_obj, log, norm_flux=1, units_wave=line_obj.units_wave)

    return log


class Line:

    def __init__(self, label, band=None, fit_conf=None, emission_check=True, cont_from_bands=True, ref_log=None,
                 z_line=None, interpret=True):

        self.label, self.mask = label, array([nan] * 6)
        self.blended_check, self.merged_check = False, False
        self.profile_label, self.list_comps = 'no', None
        self.ion, self.wavelength, self.units_wave, self.kinem, self.latex_label = None, None, None, None, None

        self.intg_flux, self.intg_err = None, None
        self.peak_wave, self.peak_flux = None, None
        self.eqw, self.eqw_err = None, None
        self.gauss_flux, self.gauss_err = None, None
        self.cont, self.std_cont =None, None
        self.m_cont, self.n_cont = None, None
        self.amp, self.center, self.sigma = None, None, None
        self.amp_err, self.center_err, self.sigma_err = None, None, None
        self.z_line = z_line
        self.v_r, self.v_r_err = None, None
        self.pixel_vel = None
        self.sigma_vel, self.sigma_vel_err = None, None
        self.sigma_thermal, self.sigma_instr = None, None
        self.snr_line, self.snr_cont = None, None
        self.observations, self.comments = 'no', 'no'
        self.pixel_mask = 'no'
        self.FWHM_intg, self.FWHM_g, self.FWZI = None, None, None
        self.w_i, self.w_f = None, None
        self.v_med, self.v_50 = None, None
        self.v_5, self.v_10 = None, None
        self.v_90, self.v_95 = None, None
        self.chisqr, self.redchi = None, None
        self.aic, self.bic = None, None
        self.pixelWidth = None

        self._fit_conf = None
        self._emission_check = emission_check
        self._cont_from_adjacent = cont_from_bands
        self._decimal_wave = False
        self._narrow_check = False

        # Interpret the line from the user reference
        if interpret:
            self._line_derivation(label, band, fit_conf, ref_log=ref_log)

        return

    @classmethod
    def from_log(cls, label, log=None, norm_flux=1):

        # Recover the label just in case
        label = check_line_in_log(label, log)

        # Create the line object
        inline = cls(label, interpret=False)

        if log is not None:

            if label in log.index:

                # Recover "simple" attributes
                for param in _LOG_EXPORT:

                    param_value = log.loc[label, param]

                    # Normalize
                    if _LOG_COLUMNS[param][0]:
                        param_value = param_value / norm_flux

                    inline.__setattr__(param, param_value)

                # Recover "complex" attributes
                for param in ['ion', 'wavelength', 'latex_label']:
                    inline.__setattr__(param, log.loc[label, param])

                # Band
                inline.mask = log.loc[label, 'w1':'w6'].values

                # Checks:
                if inline.profile_label != 'no':
                    inline.blended_check, inline.merged_check = False, False

                    # Merged line
                    if inline.label.endswith('_m'):
                        inline.merged_check = True
                        _, inline.units_wave = check_units_from_wave(label[:-1].split('_')[1])

                    # Blended line
                    else:
                        inline.blended_check = True
                        _, inline.units_wave = check_units_from_wave(label.split('_')[1])

                # Units transition
                else:
                    _, inline.units_wave = check_units_from_wave(label.split('_')[1])

                # List comps
                if inline.blended_check:
                    inline.list_comps = array(inline.profile_label.split('-'))

                else:
                    inline.list_comps = array([inline.label])

            else:
                _logger.warning(f'Input line {inline.label} not found in log')

        else:
            _logger.warning(f'No lines log introduced for the line {inline.label}')

        return inline

    def _line_derivation(self, label, band, fit_conf=None, ref_log=None):

        # Discriminate between string and float transitions
        ref_log = ref_log if ref_log is not None else _PARENT_BANDS
        self.label = check_line_in_log(label, ref_log)

        # Copy the input configuration dictionary # TODO do we need to do a copy here
        self._fit_conf = {} if fit_conf is None else fit_conf.copy()

        # List of components and label for the multi-profile
        self.profile_label = self._fit_conf.get(self.label, 'no')  # TODO change this one to None

        # Distinguish between merged and blended
        suffix = self.label[-2:]
        if (suffix == '_b') or (suffix == '_m'):
            if self.profile_label != 'no':
                if suffix == '_b':
                    self.blended_check = True
                else:
                    self.merged_check = True
            else:
                _logger.warning(f'The line {self.label} has the "{suffix}" suffix but no components have been specified '
                                f'for the fitting')

        # Blended lines have various elements in the list, single and merged only one
        self.list_comps = self.profile_label.split('-') if self.blended_check else [self.label]

        # Get transition data from label
        self.ion, self.wavelength, self.units_wave, self.kinem = label_components(self.list_comps)

        # Provide a band from the log if available the band
        if band is None:
            query_label = self.label if self.profile_label == 'no' else self.label[:-2]
            if query_label in ref_log.index:
                try:
                    self.mask = ref_log.loc[query_label, 'w1':'w6'].values
                except KeyError:
                    _logger.info(f'Failure to get bands for line {self.label} from input log')
        else:
            self.mask = band

        # Warn if the band wavelengths are not sorted and the theoretical value is not within the line region
        if not np.any(np.isnan(self.mask)):
            if not all(diff(self.mask) >= 0):
                _logger.warning(f'The line {label} band wavelengths are not sorted: {band}')
            if not all(self.mask[2] < self.wavelength) and not all(self.wavelength < self.mask[3]):
                _logger.warning(f'The line {label} transition at {self.wavelength} is outside the line band wavelengths: '
                                f'w3 = {self.mask[2]};  w4 = {self.mask[3]}')
        else:
            _logger.info(f'Transition {self.label} contains NaN entries')

        # Check if there are masked pixels in the line
        self.pixel_mask = self._fit_conf.get(f'{self.label}_mask', 'no')

        # Check if the wavelength has decimal transition
        self._decimal_wave = True if '.' in self.label else False

        # Generate the latex label
        if self.merged_check:       # Merged
            latex_list = list(latex_from_label(self.profile_label.split('-')))
            self.latex_label = array('+'.join(latex_list), ndmin=1)

        elif self.blended_check:    # Blended
            self.latex_label = latex_from_label(self.profile_label.split('-'))

        else:                       # Single
            self.latex_label = latex_from_label(None, self.ion, self.wavelength, self.units_wave, self.kinem)

        return

    def __str__(self):

        return self.label
