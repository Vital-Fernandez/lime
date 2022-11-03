import logging

from numpy import array, abs, nan, all, diff, char, searchsorted
from .tools import DISPERSION_UNITS, UNITS_LATEX_DICT
from .io import _PARENT_BANDS

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


def check_line_in_log(input_label, log=None, z_obj=0, tol=1):

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
            if not all(diff(ref_waves) >= 0):
                _logger.warning(f'The lines log rows are not sorted from lower to higher wavelengths. This can cause '
                                f'issues to identify the lines using the transition wavelength')

            # Locate the best candidate
            idx_closest = searchsorted(ref_waves, input_label * (1 + z_obj))

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


def latex_from_label(label, ion=None, wave=None, units_wave=None, kinetic_comp=None, recomb_atom=('H1', 'He1', 'He2')):

    # Use input values if provided else compute from label
    if (ion is None) or (wave is None) or (wave is None):
        ion, wave, units_wave, kinetic_comp = label_components(label)

    atom, ionization = ion[:-1], int(ion[-1])

    units_latex = UNITS_LATEX_DICT[units_wave]

    ionization_roman = int_to_roman(ionization)

    wave_round = int(wave)

    kinetic_comp = '' if kinetic_comp is None else f'-k{kinetic_comp}'

    if ion in recomb_atom:
        latex = f'${atom}{ionization_roman}{wave_round}{units_latex}{kinetic_comp}$'
    else:
        latex = f'$[{atom}{ionization_roman}]{wave_round}{units_latex}{kinetic_comp}$'

    return latex


def label_components(label):

    # Transition label items assuming '_' separation
    trans_items = label.split('_')

    # Element producing the transition
    ion = trans_items[0]

    # Wavelength and units
    wave, units_wave = check_units_from_wave(trans_items[1])

    # Kinematic element
    kinematic = None
    if len(trans_items) > 2:
        if trans_items[2][0] == 'w':
            kinematic = trans_items[2][1]
            # _logger.info(f'Kinematic component "{trans_items[2][0]}" in label {label} is not recognized')

    return ion, wave, units_wave, kinematic


class Line:

    def __init__(self, label, band=None, fit_conf=None, emission_check=True, cont_from_bands=True, ref_log=None,
                 z_line=0):

        self.line, self.mask = None, array([nan] * 6)
        self.blended_check, self.profile_label = False, 'no'
        self.list_comps = None
        self.ion, self.wave, self.units_wave, self.kinem, self.latex = None, None, None, None, None

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
        self._line_derivation(label, band, fit_conf, ref_log=ref_log)

        return

    def _line_derivation(self, label, band, fit_conf=None, ref_log=None):

        # Discriminate between string and float transitions
        ref_log = ref_log if ref_log is not None else _PARENT_BANDS
        self.label = check_line_in_log(label, ref_log, z_obj=self.z_line)

        # Copy the input configuration dictionary
        self._fit_conf = {} if fit_conf is None else fit_conf.copy()

        # Get transition data from label
        self.ion, self.wave, self.units_wave, self.kinem = label_components(self.label)

        # List of components in a blended or merged transition
        suffix = self.label[-2:]
        self.profile_label = self._fit_conf.get(self.label, 'no')  # TODO change this one to None
        if (suffix == '_b') or (suffix == '_m'):
            if self.profile_label != 'no':
                self.list_comps = self.profile_label.split('-')
                self.blended_check = True if suffix == '_b' else False
            else:
                _logger.warning(f'The line {self.line} has the "{suffix}" suffix but no components have been specified '
                                f'for the fitting')

        # Provide a band from the log if available the band
        if band is None:
            query_label = self.label if self.profile_label == 'no' else self.label[:-2]
            if query_label in ref_log.index:
                try:
                    self.mask = ref_log.loc[query_label, 'w1':'w6'].values
                except:
                    _logger.info(f'Failure to get bands for line {self.label} from input log')
        else:
            self.mask = band

        # Warn if the band wavelengths are not sorted and the theoretical value is not within the line region
        if self.mask is not None:
            if not all(diff(self.mask) >= 0):
                _logger.warning(f'The line {label} band wavelengths are not sorted: {band}')
            if not (self.mask[2] < self.wave < self.mask[3]):
                _logger.warning(f'The line {label} transition at {self.wave} is outside the line band wavelengths: '
                                f'w3 = {self.mask[2]};  w4 = {self.mask[3]}')

        # # Check if blended line
        # if self.label in self._fit_conf:
        #     self.profile_label = self._fit_conf[self.label]
        #     if '_b' in self.label:
        #         self.blended_check = True

        # Check if there are masked pixels in the line
        self.pixel_mask = self._fit_conf.get(f'{self.label}_mask', 'no')

        # Check if the wavelength has decimal transition
        self._decimal_wave = True if '.' in self.label else False

        # Compute the latex label
        if self.profile_label == 'no':
            self.latex = latex_from_label(None, self.ion, self.wave, self.units_wave, self.kinem)
        else:
            self.latex = latex_from_label(self.list_comps[0])
            for comp in self.list_comps[1:]:
                self.latex = f'{self.latex[:-1]}+{latex_from_label(comp)[1:]}'

        return
