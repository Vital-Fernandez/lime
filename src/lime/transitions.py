import logging

import numpy as np
import pandas as pd
from .io import _PARENT_BANDS, _LOG_EXPORT, _LOG_COLUMNS, check_file_dataframe, LiMe_Error
from pandas import DataFrame
from .tools import pd_get, au

_DEFAULT_PROFILE = 'g-emi'

_logger = logging.getLogger('LiMe')

_COMPs_KEYS = {'k': 'kinem',
               'p': 'profile_comp',
               't': 'transition_comp'}

ELEMENTS_DICT = dict(H='Hydrogen', He='Helium', Li='Lithium', Be='Beryllium', B='Boron', C='Carbon', N='Nitrogen', O='Oxygen', F='Fluorine',
                     Ne='Neon', Na='Sodium', Mg='Magnesium', Al='Aluminum', Si='Silicon', P='Phosphorus', S='Sulfur',
                     Cl='Chlorine', Ar='Argon', K='Potassium', Ca='Calcium', Sc='Scandium', Ti='Titanium', V='Vanadium',
                     Cr='Chromium', Mn='Manganese', Fe='Iron', Ni='Nickel', Co='Cobalt', Cu='Copper', Zn='Zinc',
                     Ga='Gallium', Ge='Germanium', As='Arsenic', Se='Selenium', Br='Bromine', Kr='Krypton', Rb='Rubidium',
                     Sr='Strontium', Y='Yttrium', Zr='Zirconium', Nb='Niobium', Mo='Molybdenum', Tc='Technetium',
                     Ru='Ruthenium', Rh='Rhodium', Pd='Palladium', Ag='Silver', Cd='Cadmium', In='Indium', Sn='Tin',
                     Sb='Antimony', Te='Tellurium', I='Iodine', Xe='Xenon', Cs='Cesium', Ba='Barium', La='Lanthanum',
                     Ce='Cerium', Pr='Praseodymium', Nd='Neodymium', Pm='Promethium', Sm='Samarium', Eu='Europium',
                     Gd='Gadolinium', Tb='Terbium', Dy='Dysprosium', Ho='Holmium', Er='Erbium', Tm='Thulium',
                     Yb='Ytterbium', Lu='Lutetium', Hf='Hafnium', Ta='Tantalum', W='Tungsten', Re='Rhenium', Os='Osmium',
                     Ir='Iridium', Pt='Platinum', Au='Gold', Hg='Mercury', Tl='Thallium', Pb='Lead', Bi='Bismuth',
                     Th='Thorium', Pa='Protactinium', U='Uranium', Np='Neptunium', Pu='Plutonium', Am='Americium',
                     Cm='Curium', Bk='Berkelium', Cf='Californium', Es='Einsteinium', Fm='Fermium', Md='Mendelevium',
                     No='Nobelium', Lr='Lawrencium', Rf='Rutherfordium', Db='Dubnium', Sg='Seaborgium', Bh='Bohrium',
                     Hs='Hassium', Mt='Meitnerium', Ds='Darmstadtium', Rg='Roentgenium', Cn='Copernicium', Nh='Nihonium',
                     Fl='Flerovium', Mc='Moscovium', Lv='Livermorium', Ts='Tennessine', Og='Oganesson')


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


def recover_ionization(string):

    part_len = len(string)
    index = part_len - 1  # Start at the last character
    while index >= 0 and string[index].isdigit():
        index -= 1

    # All characters
    if index == part_len - 1:
        particle, ionization = string, None

    # All numbers
    elif index < 0:
        particle, ionization = string, None

    # Particle and ionization
    else:
        particle, ionization = string[0:index+1], int(string[index+1:])

    return particle, ionization


def recover_transition(particle):

    # Check for PyNeb particle notation
    element = ELEMENTS_DICT.get(particle.symbol)
    if (element is not None) and (particle.ionization is not None):

        if particle.label in ('H1', 'He1', 'He2'):
            transition = 'rec'
        else:
            transition = 'col'
    else:
        transition = None

    return transition


def particle_notation(particle, transition):

    # Pre-assigned a transition time if particle and ionization
    if transition is None:
        transition = recover_transition(particle)

    if transition is not None:

        try:

            # Get ionization numeral
            ionization_roman = int_to_roman(particle.ionization)
            part_label = f'{particle.symbol}{ionization_roman}'

            # Collisional excited
            if transition == 'col':
                part_label = f'$[{part_label}]'

            # Semi-forbidden
            elif transition == 'sem':
                part_label = f'${part_label}]'

            # Recombination
            else:
                part_label = f'${part_label}'

        except:
            part_label = f'{particle}-$'

    else:
        part_label = f'{particle}-$'

    return part_label


def air_to_vacuum_function(input_array, sig_fig=None):

    input_array = np.array(input_array, ndmin=1)

    if 'U' in str(input_array.dtype): #TODO finde better way

        ion_array, wave_array, latex_array = label_decomposition(input_array)
        air_wave = wave_array
    else:
        air_wave = input_array

    refraction_index = (1 + 1e-6 * (287.6155 + 1.62887/np.power(air_wave*0.0001, 2) + 0.01360/np.power(air_wave*0.0001, 4)))
    output_array = (air_wave * 0.0001 * refraction_index) * 10000

    if sig_fig is not None:
        output_array = np.round(output_array, sig_fig) if sig_fig != 0 else np.round(output_array, sig_fig).astype(int)

    if 'U' in str(input_array.dtype):
        vacuum_wave = output_array.astype(str)
        output_array = np.core.defchararray.add(ion_array, '_')
        output_array = np.core.defchararray.add(output_array, vacuum_wave)
        output_array = np.core.defchararray.add(output_array, 'A')

    return output_array


def check_units_from_wave(line, str_ion, str_wave, lines_df):

    # Try to recover from dataframe
    if lines_df is not None:

        # Check literal unit
        wave = pd_get(lines_df, line, 'wavelength')
        units = pd_get(lines_df, line, 'units_wave')

        # Check core element
        if wave is None:
            core_element = f'{str_ion}_{str_wave}'
            wave = pd_get(lines_df, core_element, 'wavelength')
            units = pd_get(lines_df, core_element, 'units_wave')

        # Convert to units
        units = au.Unit(units) if units is not None else units

    else:
        wave, units = None, None

    # Otherwise decipher from label
    if (units is None) or (wave is None):

        # First check for Angstroms
        if str_wave[-1] == 'A':
            units_label = au.Unit('AA')
            wave_label = float(str_wave[:-1])

        else:
            au_unit = au.Unit(str_wave)
            units_label = au_unit.bases[0]
            wave_label = au_unit.scale

    else:
        units_label, wave_label = None, None

    # Give preferences to the tabel values
    wave = wave_label if wave is None else wave
    units = units_label if units is None else units

    return wave, units


def check_line_in_log(input_label, log=None, tol=1):

    # Guess the transition if only a wavelength provided
    if not isinstance(input_label, str):

        if log is not None:

            # Check the wavelength from the wave_obs column
            if 'wavelength' in log.columns:
                ref_waves = log.wavelength.to_numpy()


            # Get it from the indexes
            else:
                ion_array, ref_waves = zip(*log.index.str.split('_').to_numpy())
                _wave0, units_wave = check_units_from_wave(ref_waves[0])
                ref_waves = np.char.strip(ref_waves, units_wave).astype(float)

            # Check if table rows are not sorted
            if not all(np.diff(ref_waves) >= 0): # TODO we might need to use something else than searchsorted
                _logger.warning(f'The lines log rows are not sorted from lower to higher wavelengths. This can cause '
                                f'issues to identify the lines using the transition wavelength. Try to use the string '
                                f'line label')

            # Locate the best candidate
            idx_closest = np.argmin(np.abs(ref_waves - input_label))
            line_wave = ref_waves[idx_closest]

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


def latex_from_label(label, particle=None, wave=None, units_wave=None, kinem=None, transition_comp=None, scalar_output=False,
                     decimals=None):

    # Use input values if provided else compute from label
    if (particle is None) or (wave is None) or (wave is None) or (kinem is None) or (transition_comp is None):
        particle, wave, units_wave, kinem, profile_comp, transition_comp = label_composition(label)

    # Reshape to 1d array
    else:
        particle = np.array(particle, ndmin=1)
        wave = np.array(wave, ndmin=1)
        units_wave = np.array(units_wave, ndmin=1)
        kinem = np.array(kinem, ndmin=1)
        transition_comp = np.array(transition_comp, ndmin=1)

    n_items = wave.size
    latex_array = np.empty(n_items).astype('object')

    # Significant figures
    if decimals is None:
        wave = np.round(wave, 0).astype(int)
    else:
        wave = np.round(wave, decimals)

    for i in np.arange(n_items):

        # Particle label
        part_label = particle_notation(particle[i], transition_comp[i])

        # Wavelength and units label
        units_latex = f'{units_wave[i]:latex}'[9:-2]

        # Kinematic label
        kinetic_comp = '' if kinem[i] == 0 else f'-k_{kinem[i]}'

        # Combine items
        latex_array[i] = f'{part_label}{wave[i]}{units_latex}{kinetic_comp}$'

    # Scalar output if requested and 1 length array
    if scalar_output:
        if latex_array.size == 1:
            latex_array = latex_array[0]

    return latex_array


def label_composition(line_list, ref_df=None, default_profile=None):

    # Empty containers for the label componentes
    n_comps = len(line_list)
    particle = [None] * n_comps #empty(n_comps).astype(str)
    wavelength = np.empty(n_comps)
    units_wave = [None] * n_comps
    kinem = np.zeros(n_comps, dtype=int)
    profile_comp = [None] * n_comps
    transition_comp = [None] * n_comps

    # If there isn't an input profile use LiMe default
    default_profile = _DEFAULT_PROFILE if default_profile is None else default_profile

    # Loop through the components and get the components
    for i, line in enumerate(line_list):

        line_items = line.split('_')

        # Check the line has the items
        if len(line_items) < 2:
            raise LiMe_Error(f'The blend/merged component "{line}" in the transition list "{line_list}" does not have a'
                             f' recognised format. Please use a "Particle_WavelengthUnits" format in the configuration'
                             f'file')

        # Particle properties
        particle[i] = Particle.from_label(line_items[0])

        # Wavelength properties
        wavelength[i], units_wave[i] = check_units_from_wave(line, line_items[0], line_items[1], ref_df)

        # Split the optional components: "H1_1216A_t-rec_k-0_p-g" -> {'t': 'rec', 'k': '0', 'p': 'g'} # TODO better do that with optional_comps
        comp_conf = {optC[0]: optC[2:] for optC in line_items[2:]}

        # Check there are no accepted optional components
        for opt_key in comp_conf.keys():
            if opt_key != 'm' and opt_key != 'b':
                if opt_key not in ['k', 'p', 't']:
                    _logger.warning(f'Optional component "{opt_key}" is not recognised. Please use "_k-", "_p-", "_t-".')

        # Kinematic component
        kinem[i] = int(comp_conf.get('k', 0))

        # Profile component
        profile_comp[i] = comp_conf.get('p', None)
        if profile_comp[i] is None:
            profile_comp[i] = default_profile

        # Transition component
        trans = comp_conf.get('t', None)

        # If none is provided check from the table
        if (trans is None) and (ref_df is not None):
            trans = pd_get(ref_df, line, 'transition')

        # Else assume default
        if trans is None:
            trans = recover_transition(particle[i])

        transition_comp[i] = trans

    return particle, wavelength, units_wave, kinem, profile_comp, transition_comp


def label_decomposition(lines_list, bands=None, fit_conf=None, params_list=('particle', 'wavelength', 'latex_label'),
                        scalar_output=False):

    """
    This function takes a ``lines_list`` and returns several arrays with the requested parameters.

    If the user provides a `bands dataframe <https://lime-stable.readthedocs.io/en/latest/inputs/n_inputs3_line_bands.html>`_
    (``bands`` argument) dataframe and a `fitting documentation <https://lime-stable.readthedocs.io/en/latest/inputs/n_inputs4_fit_configuration.html>`_.
    (``fit_conf`` argument) the function will use this information to compute the requested outputs. Otherwise, only the
    `line label <https://lime-stable.readthedocs.io/en/latest/inputs/n_inputs2_line_labels.html>`_ will be used to derive
    the information.

    The ``params_list`` argument establishes the output parameters arrays. The options available are: "particle",
    "wavelength", "latex_label", "kinem", "profile_comp" and "transition_comp".

    If the ``lines_list`` argument only has one element the user can request an scalar output with ``scalar_output=True``.

    :param lines_list: Array of lines in LiMe notation.
    :type lines_list: list

    :param bands: Bands dataframe (or file address to the dataframe).
    :type bands: pandas.Dataframe, str, path.Pathlib, optional

    :param fit_conf: Fitting configuration.
    :type fit_conf: dict, optional

    :param params_list: List of output parameters. The default value is ('particle', 'wavelength', 'latex_label')
    :type params_list: tuple, optional

    :param scalar_output: Set to True for a Scalar output.
    :type scalar_output: bool

    """

    headers = ['particle', 'wavelength', 'latex_label', 'kinem', 'profile_comp', 'transition_comp']
    lines_df = pd.DataFrame(index=np.array(lines_list, ndmin=1), columns=headers)

    # Loop through the lines and derive their properties:
    for label in lines_df.index:
        line = Line(label, bands, fit_conf)

        lines_df.loc[label, :] = (line.particle[0].label, line.wavelength[0], line.latex_label[0], line.kinem[0],
                                  line.profile_comp[0], line.transition_comp[0])

    # Adjust column types
    lines_df['wavelength'] = pd.to_numeric(lines_df['wavelength'])

    # Recover the columns requested by the user
    output = []
    for i, param in enumerate(params_list):
        output.append(lines_df[param].to_numpy(copy=True))

    # If requested and single line, return the input as a scalar
    if scalar_output and (output[0].shape[0] == 1):
        output = tuple(item[0] for item in output)
    else:
        output = tuple(output)

    return output


def label_profiling(profile_comp, default_type='emi'):

    _p_type_list, _p_shape_list = [], []
    for prof_comp in profile_comp:
        _line_p_items = prof_comp.split('-')

        # Type of profile
        _p_shape = _line_p_items[0]

        # In case the user does not provide emi or abs
        _p_type = 'emi' if len(_line_p_items) == 1 else _line_p_items[1]

        # Emission or absorption
        _p_type = True if _p_type == default_type else False

        # Append to list
        _p_shape_list.append(_p_shape)
        _p_type_list.append(_p_type)

    _p_shape_list, _p_type_list = np.array(_p_shape_list), np.array(_p_type_list)

    return _p_shape_list, _p_type_list


def label_mask_assigning(line_label, input_band, blend_check, merged_check, core_comp):

    if isinstance(input_band, DataFrame):

        # Query for input label
        if pd_get(input_band, line_label, 'w1') is not None:
            ref_label = line_label

        # Blended or merged line
        elif pd_get(input_band, line_label[:-2], 'w1') is not None:
            ref_label = line_label[:-2]

        # Case where we introduce a line with a different profile (H1_4861A_l)
        elif pd_get(input_band, core_comp, 'w1') is not None:
            ref_label = core_comp

        # Not found
        else:
            ref_label = None

        # Recover the mask
        if ref_label is not None:
            mask = np.array([input_band.at[ref_label, 'w1'], input_band.at[ref_label, 'w2'],
                             input_band.at[ref_label, 'w3'], input_band.at[ref_label, 'w4'],
                             input_band.at[ref_label, 'w5'], input_band.at[ref_label, 'w6']])
        else:
            mask = None

    # No band
    elif input_band is None:
        mask = None

    # Convert input to numpy array
    else:
        mask = np.atleast_1d(input_band)

    return mask


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


class Particle:

    def __init__(self, label: str = None, symbol: str = None, ionization: int = str):

        self.label = label
        self.symbol = symbol
        self.ionization = ionization

        return

    @classmethod
    def from_label(cls, label):

        symbol, ionization = recover_ionization(label)

        return cls(label, symbol, ionization)

    def __str__(self):

        return self.label

    def __repr__(self):

        return self.label

    def __eq__(self, other):
        """Overrides the default implementation"""

        _equality = False
        if isinstance(other, Particle):
            if (other.label == self.label) and (other.symbol == self.symbol) and (other.ionization == self.ionization):
                _equality = True

        return _equality

    def __ne__(self, other):

        _inequality = True
        if isinstance(other, Particle):
            if (other.label == self.label) and (other.symbol == self.symbol) and (other.ionization == self.ionization):
                _inequality = False

        return _inequality


class Line:

    def __init__(self, label, band=None, fit_conf=None, profile=None, cont_from_bands=True, z_line=None,
                 interpret=True):

        # Label attributes
        self.label = label
        self.mask = None
        self.latex_label = None,
        self.group_label, self.list_comps = None, None

        self.particle = None
        self.wavelength, self.units_wave = None, None
        self.blended_check, self.merged_check = False, False

        self.kinem = None
        self.profile_comp = profile
        self.transition_comp = None

        self._p_type = None
        self._p_shape = None

        # Measurements attributes
        self.intg_flux, self.intg_flux_err = None, None
        self.peak_wave, self.peak_flux = None, None
        self.eqw, self.eqw_err = None, None
        self.eqw_intg, self.eqw_intg_err = None, None
        self.profile_flux, self.profile_flux_err = None, None
        self.cont, self.cont_err = None, None
        self.m_cont, self.n_cont = None, None
        self.m_cont_err, self.n_cont_err = None, None
        self.m_cont_err_intg, self.n_cont_err_intg = None, None
        self.amp, self.center, self.sigma, self.gamma = None, None, None, None
        self.amp_err, self.center_err, self.sigma_err, self.gamma_err = None, None, None, None
        self.alpha, self.beta = None, None
        self.alpha_err, self.beta_err = None, None
        self.frac, self.frac_err = None, None
        self.z_line = z_line
        self.v_r, self.v_r_err = None, None
        self.pixel_vel = None
        self.sigma_vel, self.sigma_vel_err = None, None
        self.sigma_thermal, self.sigma_instr = None, None
        self.snr_line, self.snr_cont = None, None
        self.observations, self.comments = 'no', 'no'
        self.pixel_mask = 'no'
        self.n_pixels = None
        self.FWHM_i, self.FWHM_p, self.FWZI = None, None, None
        self.w_i, self.w_f = None, None
        self.v_med, self.v_50 = None, None
        self.v_5, self.v_10 = None, None
        self.v_90, self.v_95 = None, None
        self.v_1, self.v_99 = None, None
        self.chisqr, self.redchi = None, None
        self.aic, self.bic = None, None
        self.pixelWidth = None

        # Extra checks
        self._cont_from_adjacent = cont_from_bands
        self._decimal_wave = False
        self._narrow_check = False

        # Interpret the line from the user reference
        if interpret:
            self._from_label(label, band, fit_conf)

        return

    def _from_label(self, label, band=None, fit_conf=None):

        # If band is not provided use default database
        if band is None:
            band = _PARENT_BANDS
        band = check_file_dataframe(band, DataFrame, copy_input=False)

        # Infer label from log if input line is a wavelength
        self.label = check_line_in_log(label, band)

        # Get label components
        comps_list = self.label.split('_')
        n_comps = len(comps_list)
        if n_comps < 2:
            raise LiMe_Error(f'The {self.label} the line label format is not recognized. '
                             f'Please use a "Particle_WavelengthUnits" format.')

        # Check the modularity      (only 2 comps)    (case b-6)
        modularity_comp = comps_list[-1] if (comps_list[-1] == 'b') or (comps_list[-1] == 'm') else None
        self._modularity_component(modularity_comp, fit_conf, band)

        # Review the components of the line
        ref_bands_df = band if isinstance(band, DataFrame) else None
        items = label_composition(self.list_comps, ref_df=ref_bands_df, default_profile=self.profile_comp)
        self.particle, self.wavelength, self.units_wave, self.kinem, self.profile_comp, self.transition_comp = items

        # Quick elements for the line profile
        self._p_shape, self._p_type = label_profiling(self.profile_comp)

        # Provide a bands from the log if possible
        core_comp = f'{comps_list[0]}_{comps_list[1]}'
        self.mask = label_mask_assigning(self.label, band, self.blended_check, self.merged_check, core_comp)

        # Check if there are masked pixels in the line
        self.pixel_mask = 'no' if fit_conf is None else fit_conf.get(f'{self.label}_mask', 'no')

        # Check if the wavelength has decimal transition
        self._decimal_wave = True if '.' in self.label else False

        # Compute latex entry if necessary
        self._review_latex_label(band)

        return

    @classmethod
    def from_log(cls, label, log=None, norm_flux=1):

        # Confirm we are not introducing a blended line which has been blended

        if label in log.index:
            measured_check = True
        elif (label[-2:] == '_b') and (label[:-2] in log.index):
            _logger.warning(f'Blended line {label} not found in log, reading {label[:-2]}')
            label = label[:-2]
            measured_check = True
        else:
            _logger.warning(f'Input line {label} not found in log')
            measured_check = False

        # Reload the line
        if measured_check:

            # Create the line object
            inline = cls(label, band=log)

            # Recover "simple" attributes
            for i, param in enumerate(_LOG_EXPORT):

                param_value = log.at[label, param]

                # Normalize
                if _LOG_COLUMNS[param][0]:
                    param_value = param_value / norm_flux

                inline.__setattr__(param, param_value)

        else:

            inline = None

        return inline

    def _modularity_component(self, modularity_label, fit_conf=None, bands_log=None):

        # Not a single line
        if modularity_label is not None:

            if modularity_label == 'b':
                self.blended_check = True

            elif modularity_label == 'm':
                self.merged_check = True

            else:
                raise LiMe_Error(f'The modularity component {modularity_label} in the input line "{self.label}" is not'
                                 f' recognized')

        # Restore the group label if provided (Configuration file over rules log)
        group_label_cfg = None if fit_conf is None else fit_conf.get(self.label, None)
        group_label_frame = None if not isinstance(bands_log, pd.DataFrame) else pd_get(bands_log, self.label,
                                                                                        'group_label', transform='none')

        self.group_label = group_label_cfg if group_label_cfg is not None else group_label_frame

        # Confirm blended or merged
        self.blended_check = True if (self.group_label is not None) and (self.merged_check is False) else False

        # Recover the profile components
        if self.merged_check or self.blended_check:

            # Reset and warned the line has a suffix but there are no components provided
            if self.group_label is None:
                self.merged_check, self.blended_check = False, False

                # Warning incase there is an input configuration but the line is not there
                if fit_conf is not None:
                    _logger.warning(f'The {self.label} line has a "_{modularity_label}" suffix but not components were '
                                    f' on the lines frames or the input configuration the configuration:\n{fit_conf}')

        # List of components only for blended
        if (self.merged_check or self.blended_check) and (self.group_label is not None):
            self.list_comps = self.group_label.split('+') if self.blended_check else [self.label]

            # Check if there are repeated elements
            if len(self.list_comps) > 1:
                uniq, count = np.unique(self.list_comps, return_counts=True)
                if any(count > 1):
                    _logger.warning(f'There are repeated entries in the line label: {self.label} = {self.list_comps}')

        else:
            self.list_comps = [self.label]

        return

    def _review_latex_label(self, bands_df):

        # Check if there is a latex label on the database
        latex_exists = False
        if isinstance(bands_df, DataFrame):
            if 'latex_label' in bands_df.columns:
                if np.sum(bands_df.index.isin(self.list_comps)) == len(self.list_comps):
                    if not np.all(pd.isnull(bands_df.loc[self.list_comps, 'latex_label'])):
                        latex_exists = True

        # Merged
        if self.merged_check:
            if latex_exists:
                latex_list = list(bands_df.loc[self.list_comps, 'latex_label'].to_numpy())
            else:
                latex_list = list(latex_from_label(self.group_label.split('+')))
            self.latex_label = np.array('+'.join(latex_list), ndmin=1)

        # Blended and single
        else:
            if latex_exists:
                self.latex_label = bands_df.loc[self.list_comps, 'latex_label'].to_numpy()
            else:
                self.latex_label = latex_from_label(None, self.particle, self.wavelength, self.units_wave, self.kinem,
                                                    self.transition_comp)

        return

    def __str__(self):

        return self.label

    def __repr__(self):

        return self.label

    def index_bands(self, wavelength_array, redshift, merge_continua=True, just_band_edges=False):

        if self.mask is None:
            raise KeyError(f'The line {self.label} does not have bands. Please select another line or update the database.')

        # Make sure it is a matrix
        masks_array = np.atleast_2d(self.mask) * (1 + redshift)

        if np.any(masks_array[:, 0] < wavelength_array[0]) or np.any(masks_array[:, 5] > wavelength_array[-1]):
            _logger.warning(f'The {self.label} bands do not match the spectrum wavelength range (observed):')
            _logger.warning(
                f'-- The spectrum wavelength range is: ({wavelength_array[0]:0.2f}, {wavelength_array[-1]:0.2f}) (observed frame)')
            _logger.warning(f'-- The {self.label} bands are: {masks_array} (rest frame * (1 + z))')

        # Check if it is a masked array
        if np.ma.isMaskedArray(wavelength_array):
            wave_arr = wavelength_array.data
        else:
            wave_arr = wavelength_array

        # Remove masked pixels from this function wavelength array
        if self.pixel_mask != 'no':

            # Convert cfg mask string to limits
            line_mask_limits = format_line_mask_option(self.pixel_mask, wave_arr)

            # Get masked indeces
            idcsMask = (wave_arr[:, None] >= line_mask_limits[:, 0]) & (wave_arr[:, None] <= line_mask_limits[:, 1])
            idcsValid = ~idcsMask.sum(axis=1).astype(bool)[:, None]

        else:
            idcsValid = np.ones(wave_arr.size).astype(bool)[:, None]

        # Find indeces for six points in spectrum
        idcsW = np.searchsorted(wave_arr, masks_array)

        # Return just the edges of the bands
        if just_band_edges:
            outputs = idcsW[0]

        # Return the indeces of all the pixels within the bands
        else:

            # Emission region
            idcsLineRegion = ((wave_arr[idcsW[:, 2]] <= wave_arr[:, None]) & (
                        wave_arr[:, None] <= wave_arr[idcsW[:, 3]]) & idcsValid).squeeze()

            # Return left and right continua merged in one array
            if merge_continua:
                idcsContRegion = (((wave_arr[idcsW[:, 0]] <= wave_arr[:, None]) &
                                   (wave_arr[:, None] <= wave_arr[idcsW[:, 1]])) |
                                  ((wave_arr[idcsW[:, 4]] <= wave_arr[:, None]) & (
                                          wave_arr[:, None] <= wave_arr[idcsW[:, 5]])) & idcsValid).squeeze()

                outputs = idcsLineRegion, idcsContRegion

            # Return left and right continua in separated arrays
            else:
                idcsContLeft = ((wave_arr[idcsW[:, 0]] <= wave_arr[:, None]) & (
                            wave_arr[:, None] <= wave_arr[idcsW[:, 1]]) & idcsValid).squeeze()
                idcsContRight = ((wave_arr[idcsW[:, 4]] <= wave_arr[:, None]) & (
                            wave_arr[:, None] <= wave_arr[idcsW[:, 5]]) & idcsValid).squeeze()

                outputs = idcsLineRegion, idcsContLeft, idcsContRight

        return outputs
