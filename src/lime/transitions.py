import logging

import numpy as np
import pandas as pd
from pandas import DataFrame
from numpy.core.fromnumeric import argmin

from lime.io import _LOG_EXPORT, _LOG_COLUMNS, check_file_dataframe, LiMe_Error, _RANGE_ATTRIBUTES_FIT, _ATTRIBUTES_FIT, load_frame, _LIME_FOLDER
from lime.tools import pd_get, au, unit_conversion
from lime import rsrc_manager

_DEFAULT_PROFILE = 'g'
_DEFAULT_SHAPE = 'emi'

_logger = logging.getLogger('LiMe')

_COMPs_KEYS = {'k': 'kinem',
               'p': 'profile',
               's': 'shape',
               't': 'trans'}

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


# Reading file with the format and export status for the measurements
_LIME_DATABASE_FILE = rf'{_LIME_FOLDER}/resources/lines_database_v2.0.0.txt'


class LinesDatabase:

    def __init__(self, frame_address=None, default_shape=None, default_profile=None):

        # Default_lines database
        self.frame_address = _LIME_DATABASE_FILE if frame_address is None else frame_address

        # Function attributes
        self.vacuum_check = False

        self.frame = load_frame(self.frame_address)

        self._shape = 'emi' if default_shape is None else default_shape
        self._profile = 'g' if default_profile is None else default_profile

        return

    def set_database(self, wave_intvl=None, line_list=None, particle_list=None, redshift=None, units_wave='Angstrom',
                     decimals=None, vacuum_waves=False, ref_bands=None, update_labels=False, update_latex=False,
                     vacuum_label=False, default_shape=None, default_profile=None):


        # Reload the database at each modification to avoid contamination
        ref_bands = load_frame(self.frame_address) if ref_bands is None else ref_bands

        self.frame = lines_frame(wave_intvl, line_list, particle_list, redshift, units_wave, decimals,
                                 vacuum_waves, ref_bands, update_labels, update_latex)

        self.vacuum_check = vacuum_waves

        if default_shape:
            self.set_shape(default_shape)

        if default_profile:
            self.set_profile(default_profile)

        return

    def reset_database(self):

        self.vacuum_check = False
        self.frame = load_frame(_LIME_DATABASE_FILE)

        return

    def copy(self):

        return self.frame.copy()

    def set_shape(self, value):
        self._shape = value
        return

    def set_profile(self, value):
        self._profile = value
        return

    def get_shape(self):
        return self._shape

    def get_profile(self):
        return self._profile


rsrc_manager.lineDB = LinesDatabase(_LIME_DATABASE_FILE)


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


def recover_transition(particle, input_trans=None):

    # Check for PyNeb particle notation
    if input_trans is None:

        element = ELEMENTS_DICT.get(particle.symbol)
        if (element is not None) and (particle.ionization is not None):

            if particle.label in ('H1', 'He1', 'He2'):
                transition = 'rec'
            else:
                transition = 'col'
        else:
            transition = None

        return transition

    else:
        return input_trans


def particle_notation(particle, transition):

    # Pre-assigned a transition time if particle and ionization
    if transition is None:
        transition = recover_transition(particle)

    if transition is not None:

        try:

            # Get ionization numeral
            particle = particle if not isinstance(particle, str) else Particle.from_label(particle)
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


def air_to_vacuum_function(wave_array, units_wave='AA'):

    r"""
    Converts air wavelengths to vacuum wavelengths using the method of `Greisen et al. (2006, A&A 446, 747) <https://ui.adsabs.harvard.edu/abs/2006A%26A...446..747G/abstract>`_.

    This function applies a wavelength correction to account for the refractive index of air. The input wavelengths are assumed to be measured in air, and
    the output values correspond to vacuum wavelengths. The conversion follows the formula:

    .. math::

        \lambda_{\text{vac}} = \lambda_{\text{air}} \times \left( 1 + 10^{-6} \times
        \left( 287.6155 + 1.62887 \sigma^2 + 0.01360 \sigma^4 \right) \right)

    where:

    .. math::

        \sigma = \frac{1}{\lambda_{\text{air}}^2} \quad \text{(in microns)}.

    :param wave_array: Input array of wavelengths in air.
    :type wave_array: numpy.ndarray or astropy.units.Quantity

    :param units_wave: The wavelength unit (default: ``'AA'`` for Angstroms). If `wave_array` is an `astropy.Quantity`, this parameter is ignored.
    :type units_wave: str, optional

    :return: The converted vacuum wavelengths, in the same unit as the input.
    :rtype: numpy.ndarray or astropy.units.Quantity

    :raises ValueError: If the input wavelength array is not valid.

    **Example Usage:**

    Converting an array of air wavelengths to vacuum wavelengths:

    .. code-block:: python

        import numpy as np
        import astropy.units as u
        from your_module import air_to_vacuum_function

        air_waves = np.array([5000, 6000, 7000])  # Angstroms
        vac_waves = air_to_vacuum_function(air_waves)

        print(vac_waves)  # Output: Vacuum wavelengths in the same unit

    Using Astropy units:

    .. code-block:: python

        air_waves = np.array([5000, 6000, 7000]) * u.AA
        vac_waves = air_to_vacuum_function(air_waves)

        print(vac_waves.to(u.nm))  # Convert output to nanometers

    """

    wave_arr_um = wave_array * au.Unit(units_wave)
    sigma2 = 1/np.square(wave_arr_um.to(au.um).value)

    return wave_array / (1 + 1e-6 * (287.6155 + 1.62887 * sigma2 + 0.01360 * np.square(sigma2)))


def check_units_from_wave(line, str_ion, str_wave, bands):

    # First the input database
    if bands is not None:

        # Check literal unit
        wave = pd_get(bands, line, 'wavelength', nan_to_none=True)
        units = pd_get(bands, line, 'units_wave', nan_to_none=True)

        # Check core element
        if wave is None:
            core_element = f'{str_ion}_{str_wave}'
            wave = pd_get(bands, core_element, 'wavelength', nan_to_none=True)
            units = pd_get(bands, core_element, 'units_wave', nan_to_none=True)

        # Convert to units
        units = au.Unit(units) if units is not None else units

    else:
        wave, units = None, None

    # Second the reference database
    if (units is None) or (wave is None):
        wave = pd_get(rsrc_manager.lineDB.frame, line, 'wavelength', nan_to_none=True)
        units = pd_get(rsrc_manager.lineDB.frame, line, 'units_wave', nan_to_none=True)

        # Convert to units
        units = au.Unit(units) if units is not None else units

    # Third decipher from label
    if (units is None) or (wave is None):

        # First check for Angstroms
        if str_wave[-1] == 'A':
            units = au.Unit('AA')
            wave = float(str_wave[:-1])

        else:
            au_unit = au.Unit(str_wave)
            units = au_unit.bases[0]
            wave = au_unit.scale

    return wave, units


def check_line_in_log(input_wave, log=None, tol=1):

    # Guess the transition if only a wavelength provided
    if not isinstance(input_wave, str):
        if log is not None:

            # Check the wavelength from the wave_obs column or get it from the labels
            if 'wavelength' in log.columns:
                ref_waves = log.wavelength.to_numpy()
            else:
                ion_array, ref_waves = zip(*log.index.str.split('_').to_numpy())
                _wave0, units_wave = check_units_from_wave(ref_waves[0])
                ref_waves = np.char.strip(ref_waves, units_wave).astype(float)


            # Locate the best candidate
            idx_closest = np.argmin(np.abs(ref_waves - input_wave))
            label = log.iloc[idx_closest].name
            _logger.warning(f'The transition wavelength "{input_wave}" has been identified as "{label}"')

            # Check if table rows are not sorted
            if not all(np.diff(ref_waves) >= 0): # TODO we might need to use something else than searchsorted
                _logger.warning(f'\nThe lines log rows are not sorted from lower to higher wavelengths.\nThis can cause '
                                f'issues to identify the lines using the transition wavelength.\nTry to use the string '
                                f'line label')

            return label

        else:
            _logger.critical(f'The line "{input_wave}" could not be identified: A lines log was not provided')

    return input_wave


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
        units_latex = f'{units_wave[i]:latex}'[9:-2].replace(' ',r'\,')

        # Kinematic label
        kinetic_comp = '' if kinem[i] == 0 else f'-k_{kinem[i]}'

        # Combine items
        latex_array[i] = f'{part_label}{wave[i]}{units_latex}{kinetic_comp}$'

    # Scalar output if requested and 1 length array
    if scalar_output:
        if latex_array.size == 1:
            latex_array = latex_array[0]

    return latex_array


def label_composition(line_list, bands=None, default_profile=None):

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
        wavelength[i], units_wave[i] = check_units_from_wave(line, line_items[0], line_items[1], bands)

        # Split the optional components: "H1_1216A_t-rec_k-0_p-g" -> {'t': 'rec', 'k': '0', 'p': 'g'} # TODO better do that with optional_comps
        comp_conf = {optC[0]: optC[2:] for optC in line_items[2:]}

        # Check there are no accepted optional components
        for opt_key in comp_conf.keys():
            if opt_key != 'm' and opt_key != 'b':
                if opt_key not in ['k', 'p', 't', 's']:
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
        if (trans is None) and (bands is not None):
            trans = pd_get(bands, line, 'transition')

        # Else assume default
        if trans is None:
            trans = recover_transition(particle[i])

        transition_comp[i] = trans

    return particle, wavelength, units_wave, kinem, profile_comp, transition_comp


def label_decomposition(lines_list, bands=None, fit_conf=None, params_list=('particle', 'wavelength', 'latex_label'),
                        scalar_output=False, verbose=True):

    """
    This function takes a ``lines_list`` and returns several arrays with the requested parameters.

    If the user provides a `bands dataframe <https://lime-stable.readthedocs.io/en/latest/inputs/n_inputs3_line_bands.html>`_
    (``bands`` argument) dataframe and a `fitting documentation <https://lime-stable.readthedocs.io/en/latest/inputs/n_inputs4_fit_configuration.html>`_.
    (``fit_conf`` argument) the function will use this information to compute the requested 3_explanations. Otherwise, only the
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
        line = Line.from_transition(label, fit_conf, bands, verbose=verbose)

        lines_df.loc[label, :] = (line.particle.label, line.wavelength, line.latex_label, line.kinem,
                                  line.profile, line.trans)

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


def lines_frame(wave_intvl=None, line_list=None, particle_list=None, redshift=None, units_wave='Angstrom', sig_digits=4,
                vacuum_waves=False, ref_bands=None, update_labels=False, update_latex=False):
    """

    This function returns `LiMe bands database <https://lime-stable.readthedocs.io/en/latest/inputs/n_inputs3_line_bands.html>`_
    as a pandas dataframe.

    If the user provides a wavelength array (``wave_inter``) the output dataframe will be limited to the lines within
    this wavelength interval.

    Similarly, the user provides a ``lines_list`` or a ``particle_list`` the output bands will be limited to the these
    lists. These 2_guides must follow `LiMe notation style <https://lime-stable.readthedocs.io/en/latest/inputs/n_inputs2_line_labels.html>`_

    If the user provides a redshift value alongside the wavelength interval (``wave_intvl``) the output bands will be
    limited to the transitions at that observed range.

    The user can specify the desired wavelength units using the `astropy string format <https://docs.astropy.org/en/stable/units/ref_api.html>`_
    or introducing the `astropy unit variable  <https://docs.astropy.org/en/stable/units/index.html>`_. The default value
    unit is angstroms.

    The argument ``sig_digits`` determines the number of decimal figures for the line labels.

    The user can request the output line labels and bands wavelengths in vacuum setting ``vacuum=True``. This conversion
    is done using the relation from `Greisen et al. (2006) <https://www.aanda.org/articles/aa/abs/2006/05/aa3818-05/aa3818-05.html>`_.

    Instead of the default LiMe database, the user can provide a ``ref_bands`` dataframe (or the dataframe file address)
    to use as the reference database.

    :param wave_intvl: Wavelength interval for output line transitions.
    :type wave_intvl: list, numpy.array, lime.Spectrum, lime.Cube, optional

    :param line_list: Line list for output line bands.
    :type line_list: list, numpy.array, optional

    :param particle_list: Particle list for output line bands.
    :type particle_list: list, numpy.array, optional

    :param redshift: Redshift interval for output line bands.
    :type redshift: list, numpy.array, optional

    :param units_wave: Labels and bands wavelength units. The default value is "A".
    :type units_wave: str, optional

    :param sig_digits: Number of decimal figures for the line labels.
    :type sig_digits: int, optional

    :param vacuum_waves: Set to True for vacuum wavelength values. The default value is False.
    :type vacuum_waves: bool, optional

    :param ref_bands: Reference bands dataframe. The default value is None.
    :type ref_bands: pandas.Dataframe, str, pathlib.Path, optional

    :return:
    """

    # Use the default lime mask if none provided
    ref_bands = ref_bands if ref_bands is not None else rsrc_manager.lineDB.frame

    # Load the reference bands
    bands_df = check_file_dataframe(ref_bands)

    # Convert to requested units
    if units_wave != 'Angstrom':
        wave_columns = ['wave_vac', 'wavelength', 'w1', 'w2', 'w3', 'w4', 'w5', 'w6']
        conversion_factor = unit_conversion(in_units='Angstrom', out_units=units_wave, wave_array=1)
        bands_df.loc[:, wave_columns] = bands_df.loc[:, wave_columns] * conversion_factor
        bands_df['units_wave'] = units_wave
    else:
        conversion_factor = 1

    # First slice by wavelength and redshift
    idcs_rows = np.ones(bands_df.index.size).astype(bool)
    if wave_intvl is not None:
        w_min, w_max = wave_intvl[0], wave_intvl[-1]

        # Account for redshift
        redshift = redshift if redshift is not None else 0
        if 'wavelength' in bands_df.columns:
            wave_arr = bands_df['wavelength'] * (1 + redshift)
        else:
            wave_arr = label_decomposition(bands_df.index.to_numpy(), params_list=['wavelength'])[0] * conversion_factor

        # Compare with wavelength values
        idcs_rows = idcs_rows & (wave_arr >= w_min) & (wave_arr <= w_max)

    # Second slice by particle
    if particle_list is not None:
        idcs_rows = idcs_rows & bands_df.particle.isin(particle_list)

    # Finally slice by the name of the lines
    if line_list is not None:
        idcs_rows = idcs_rows & bands_df.index.isin(line_list)

    # Convert to vacuum wavelengths if requested but after renaming the labels to keep air if requested
    if vacuum_waves:
        bands_df['wavelength'] = bands_df['wave_vac']
        bands_lim_columns = ['w1', 'w2', 'w3', 'w4', 'w5', 'w6']
        bands_df[bands_lim_columns] = air_to_vacuum_function(bands_df[bands_lim_columns].to_numpy())

    # Final table
    bands_df = bands_df.loc[idcs_rows]

    # Update the labels if requested
    if update_labels or update_latex:
        n_lines = bands_df.index.size
        labels, latex_list =  ([None] * n_lines if update_labels else None, [None] * n_lines if update_latex else None)

        for i, band in enumerate(bands_df.index):
            line = Line.from_transition(band, data_frame=bands_df)
            line.update_labels(sig_digits=sig_digits)

            if update_labels:
                labels[i] = line.label
            if update_latex:
                latex_list[i] = line.latex_label

        if update_latex:
            bands_df["latex_label"] = latex_list

        if update_labels:
            bands_df.rename(index=dict(zip(bands_df.index, labels)), inplace=True)

    return bands_df


def bands_from_measurements(frame, sample_levels=['id', 'line'], sort=True, remove_empty_columns=False, index_dict=None,
                            bands_hdrs=None):

    # Load the frame if necessary
    frame = check_file_dataframe(frame, sample_levels=sample_levels)

    # Single frame
    if not isinstance(frame.index, pd.MultiIndex):

        bands_hdrs = ('wavelength', 'w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'group_label', 'units_wave',
                                        'particle', 'transition') if bands_hdrs is None else None

        # Make dataframe with single lines
        idcs_single = frame.group_label == 'none'
        bands = frame.reindex(index=frame.loc[idcs_single].index, columns=bands_hdrs)

        # Get the indexes of grouped lines
        group_labels_arr = frame.loc[~idcs_single].group_label
        unique_values, unique_indices = np.unique(group_labels_arr, return_index=True)
        group_labels_arr = group_labels_arr.iloc[np.sort(unique_indices)]

        # Make dataframe with grouped lines and add blended suffix
        bands_group = frame.reindex(index=group_labels_arr.index, columns=bands_hdrs)
        blended_arr = bands_group.loc[~bands_group.index.to_series().str.endswith('_m')].index.to_numpy()
        bands_group.rename(index={key: f"{key}_b" for key in blended_arr}, inplace=True)

        # Combine the dataframes
        bands = pd.concat([bands, bands_group], axis=0)

    # Multi-index
    else:

        line_list = frame.index.get_level_values(sample_levels).unique()

        idcs_single = frame.group_label == 'none'
        bands = frame.reindex(index=frame.loc[idcs_single].index, columns=bands_hdrs)


        # Loop through the lines
        for i, line_label in enumerate(line_list):

            # Exclude kinematic components:
            if '_k-' not in line_label:
                df_line = frame.xs(line_label, level=sample_levels, drop_level=False)
                group_list = df_line.group_label.unique()

                for j, group in enumerate(group_list):

                    # Get same group entries and compute mean bands
                    df_group = df_line.loc[df_line.group_label == group]
                    bands_limits = np.median(df_group.loc[:, 'w1':'w6'].to_numpy(), axis=0)

                    # Single
                    if group == 'none':
                        entry_label = line_label

                    # Merged and blended
                    else:

                        # Merged
                        if line_label.endswith('_m'):
                            entry_label = line_label

                        # Blended Compute name
                        else:
                            entry_label = f'{line_label}_b'
                            if entry_label in bands.index:
                                comps = line_label.split('_')
                                entry_label = f'{comps[0]}-{j}_{comps[1]}_b'

                        # Re-assign scalar wavelength


                    # Generate single line df with the line information
                    ref_df = df_group.iloc[0,:].copy()
                    ref_df.name = entry_label
                    ref_df['w1':'w6'] = bands_limits
                    ref_df = ref_df.to_frame().T

                    # Define LiMe line and add data to dataframe
                    line = Line.from_transition(entry_label, data_frame=ref_df)

                    # Assign the values:
                    bands.loc[line.label, 'wavelength'] = line.wavelength[line._ref_idx]
                    bands.loc[line.label, 'w1':'w6'] = line.mask
                    bands.loc[line.label, 'group_label'] = line.group_label
                    bands.loc[line.label, 'latex_label'] = line.latex_label
                    bands.loc[line.label, 'units_wave'] = line.units_wave[line._ref_idx]
                    bands.loc[line.label, 'particle'] = line.particle[line._ref_idx]

    if sort:
        bands.sort_values(by=['wavelength', 'group_label'], inplace=True)

    if remove_empty_columns:
        bands = bands.dropna(axis=1, how='all')

    if index_dict is not None:
        bands.rename(index=index_dict, inplace=True)

    return bands


def construct_classic_notation(line=None, line_params=None):

    # From line objects
    if line is not None:
        if isinstance(line, Line):
            for comp in line.list_comps:
                particle_str = particle_notation(comp.particle, comp.trans)
                wavelength_str = round_sig_clean(comp.wavelength)
                units_str = f'{au.Unit(comp.units_wave):latex}'[9:-2].replace(' ', r'\,')
                comp.latex_label = f'{particle_str}{wavelength_str}{units_str}$'
            line.latex_label = '+'.join(line.param_arr('latex_label'))

        else:
            raise LiMe_Error('To reconstruct the line notation please use introduce a "lime.Line" object or a dictionary '
                             'with its parameters.')

    # From dictionary with the items
    else:
        particle_str = particle_notation(Particle.from_label(line_params["particle"]), line_params.get("trans"))
        wavelength_str = round_sig_clean(line_params["wavelength"])
        units_str = f'{au.Unit(line_params["units_wave"]):latex}'[9:-2].replace(' ', r'\,')
        line_params['latex_label'] = f'{particle_str}{wavelength_str}{units_str}$'

    return


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

        # Compare with strings
        if isinstance(other, str):
            return self.label == other

        # Compare with other particles
        if isinstance(other, Particle):
            if (other.label == self.label) and (other.symbol == self.symbol) and (other.ionization == self.ionization):
               return True
            else:
                return False

    def __ne__(self, other):

        _inequality = True
        if isinstance(other, Particle):
            if (other.label == self.label) and (other.symbol == self.symbol) and (other.ionization == self.ionization):
                _inequality = False

        return _inequality


class LineMeasurements:

    def __init__(self):

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
        self.a, self.a_err = None, None
        self.b, self.b_err = None, None
        self.c, self.c_err = None, None
        self.z_line = None
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

        return

def check_measurements_table(df):

    if df is not None:
        try:
            df.at[df.index[0], 'profile_flux']
            return True
        except:
            return False
    else:
        return False


class Line:

    def __init__(self, label, particle=None, wavelength=None, units_wave=None, latex_label=None, core=None,
                 group_label=None, group=None, list_comps=None, mask=None, kinem=None, trans=None, profile=None,
                 shape=None, pixel_mask=None):

        self.label = label
        self.particle = Particle.from_label(particle)
        self.wavelength = wavelength
        self.units_wave = units_wave
        self.latex_label = latex_label
        self.core = core
        self.group_label = group_label
        self.group = group
        self.list_comps = None
        self.ref_idx = None

        self.mask = mask
        self.pixel_mask = 'no' if pixel_mask is None else pixel_mask

        self.kinem = kinem
        self.trans = recover_transition(self.particle, trans)
        self.profile = profile
        self.shape = shape

        # Compile the line components
        self.list_comps = [self] if list_comps is None else list_comps

        # Get the reference index:
        if self.wavelength is not None:
            if self.group != 'b':
                self.ref_idx = 0
            else:
                if self.core in self.list_comps:
                    self.ref_idx = list_comps.index(self.core)
                else:
                    self.ref_idx = 0

        # For grouped lines without reference wavelength
        else:
            self.ref_idx = 0 if self.ref_idx is None else self.ref_idx
            self.wavelength = self.list_comps[self.ref_idx].wavelength
            self.units_wave = self.list_comps[self.ref_idx].units_wave

        # Add measurements variable
        self.measurements = None

        return

    def __str__(self):
        return self.label

    def __repr__(self):
        return self.label

    def __eq__(self, var):
        if isinstance(var, str):
            return self.label == var

        elif isinstance(var, self.__class__):
            return self.label == var.label

        return False

    @classmethod
    def from_transition(cls, label, fit_cfg=None, data_frame=None, parent_group_label=None, norm_flux=None,
                        def_shape=None, def_profile=None, verbose=True):

        # Extract line parameters from the input containers
        line_params = parse_container_data(label,
                                           fit_cfg,
                                           data_frame,
                                           parent_group_label=parent_group_label,
                                           def_shape=rsrc_manager.lineDB.get_shape() if def_shape is None else def_shape,
                                           def_profile=rsrc_manager.lineDB.get_profile() if def_profile is None else def_profile,
                                           verbose=verbose)

        # Check if input dataframe has measurements
        measurements_check = check_measurements_table(data_frame)

        # Reconstruct de-blended parent:
        if measurements_check:
            if (line_params.get('group') is None) and line_params.get('group_label') and parent_group_label is None:
                line_params['group'] = 'b'
                line_params['list_comps'] = line_params['group_label'].split('+')
                line_params['label'] = line_params.get('group_label').split('+')[0] + '_b'

        # Get the line parameters for additional components
        if line_params.get('group'):
            for i, line_i in enumerate(line_params['list_comps']):
                line_params['list_comps'][i] = Line.from_transition(line_i,
                                                                    fit_cfg,
                                                                    data_frame,
                                                                    parent_group_label=line_params['group_label'],
                                                                    norm_flux=None,
                                                                    def_shape=def_shape,
                                                                    def_profile=def_profile,
                                                                    verbose=False)

            # Combine the latex label
            line_params['latex_label'] = '+'.join([line_i.latex_label for line_i in line_params['list_comps']])

        # Create the line and initiate the measurements container
        line = cls(**line_params)
        if parent_group_label is None:
            if measurements_check:
                line.load_measurements(data_frame, norm_flux=norm_flux)
            else:
                line.measurements = LineMeasurements()

        return line

    def param_arr(self, param):

        out_arr = [None] * len(self.list_comps)
        for i, comp in enumerate(self.list_comps):
            out_arr[i] = getattr(comp, param)

        return np.array(out_arr)

    def index_bands(self, wavelength_array, redshift, merge_continua=True, just_band_edges=False):

        # Check the line has associated bands
        if self.mask is None:
            raise LiMe_Error(f'The line {self.label} does include bands. Please select another line or update the database.')

        # Transform the bands into the observed frame
        bands_obs_arr = np.atleast_1d(self.mask) * (1 + redshift)

        # Get the wavelength array and mask
        wave_arr = wavelength_array.data

        # Find the indeces of the bands
        idcs_bands = np.searchsorted(wave_arr, bands_obs_arr)

        # Return the boolean arrays
        if not just_band_edges:
            idcs_line = np.zeros(wave_arr.size, dtype=bool)
            idcs_line[idcs_bands[2]:idcs_bands[3]] = True

            idcs_blue = np.zeros(wave_arr.size, dtype=bool)
            idcs_blue[idcs_bands[0]:idcs_bands[1]] = True

            idcs_red = np.zeros(wave_arr.size, dtype=bool)
            idcs_red[idcs_bands[4]:idcs_bands[5]] = True

            # Check for line pixel masking
            if self.pixel_mask != 'no':
                line_mask_limits = format_line_mask_option(self.pixel_mask, wave_arr)
                idcs_mask = (wave_arr[:, None] >= line_mask_limits[:, 0]) & (wave_arr[:, None] <= line_mask_limits[:, 1])
                idcs_valid = ~idcs_mask.sum(axis=1).astype(bool)#[:, None]

                idcs_line = idcs_line & idcs_valid
                idcs_blue = idcs_blue & idcs_valid
                idcs_red = idcs_red & idcs_valid

            # Output as merged continua bands
            if merge_continua:
                return idcs_line, idcs_blue | idcs_red

            # Output as independent continua bands
            else:
                return idcs_line, idcs_blue, idcs_red

        # Return just the edges of the bands
        else:
            return idcs_bands

    def load_measurements(self, data_frame, norm_flux=None):

        # Reset the measurements
        self.measurements = LineMeasurements()

        # Recover measurements from table
        list_comps = [self.label] if self.group != 'b' else self.param_arr('label')

        for j in _RANGE_ATTRIBUTES_FIT:
            param = _ATTRIBUTES_FIT[j]

            # Get component parameter
            param_value = data_frame.loc[list_comps, param].to_numpy()

            # Convert empty arrays to None
            if np.issubdtype(param_value.dtype, np.floating) and np.isnan(param_value).all():
                param_value = None

            # Get component parameter
            if param_value is not None and _LOG_COLUMNS[param][3] == 0:
                param_value = param_value[0]

            # Normalize
            if (norm_flux is not None) and _LOG_COLUMNS[param][0] and (param_value is not None):
                param_value = param_value / norm_flux

            # Store in line measurements
            self.measurements.__setattr__(param, param_value)

        return

    def update_labels(self, sig_digits=None):

        # Get core components wavelength
        new_label = (f'{self.list_comps[self.ref_idx].particle}'
                     f'_{round_sig_clean(self.wavelength, sig_digits)}'
                     f'{"A" if self.units_wave == "Angstrom" else self.units_wave}')

        # Group suffix
        group_str = "_b" if self.group == 'b' else "_m" if self.group == 'm' else ""

        # Get optional components
        if self.group != 'b':
            new_label += f'_k-{self.list_comps[self.ref_idx].kinem}' if self.list_comps[self.ref_idx].kinem != 0 else ''
            new_label += f'_t-{self.list_comps[self.ref_idx].trans}' if '_t-' in self.label else ''
            new_label += f'_p-{self.list_comps[self.ref_idx].profile}' if '_p-' in self.label else ''
            new_label += f'_s-{self.list_comps[self.ref_idx].shape}' if '_s-' in self.label else ''

        # New label
        self.label = f'{new_label}{group_str}'

        # New latex label
        construct_classic_notation(line=self)

        return

def parse_container_data(label, fit_cfg, data_frame, parent_group_label, def_shape, def_profile, verbose):

    # Review line notation and its group structure
    line_params = get_line_group(label, fit_cfg, data_frame, parent_group_label, verbose)

    # Check for the line data on the configuration file
    line_params = get_line_from_cfg(line_params, fit_cfg)

    # The configuration parameters overwrite those from the database
    line_params = get_line_from_df(line_params, data_frame)

    # Review input for missing/default parameter values
    review_input_params(line_params, def_shape, def_profile)

    return line_params


def get_line_group(label, fit_cfg, data_frame, parent_group_label=None, verbose=True):

    # Container for the parameters
    line_params = {}

    # Numerical label
    if not isinstance(label, str):
        label = check_line_in_log(label, rsrc_manager.lineDB.frame if data_frame is None else data_frame)

    # Review label components
    items = label.split('_')
    match len(items):

        # Single transition
        case 2:
            group_type, opt_items = None, None

        # Wrong format
        case 0 | 1:
            raise LiMe_Error(f'The line {label} format is not recognized. Please use a "Particle_WavelengthUnits" format.')

        # Complex transition (H1_6563A_b) or manual formating (Fe3_4658A_s-abs_p-v)
        case _:

            # Blended or merged
            if items[-1][0] in ['b', 'm']:
                group_type = items[-1][0]
                opt_items = None if len(items) == 3 else items[2:-1]
            else:
                group_type = None
                opt_items = items[2:]


    # Untangle the optional components and update the default values:
    if opt_items is not None:
        for opt_i in  opt_items:
            key, value = opt_i.split('-')
            if _COMPs_KEYS.get(key) is not None:
                line_params[_COMPs_KEYS[key]] = value

            else:
                _logger.warning(f'Line {label} has an unrecognized optional item: {opt_i}.'
                                f' - Please use "_k-", "_t-", "_p-", "_s-".')

    # Get line core label
    core = f'{items[0]}_{items[1]}'

    # Confirm grouped lines have specified their components otherwise set to single
    if group_type:
        group_label = None if fit_cfg is None else fit_cfg.get(label, None)

        # Dict 2_guides have preference over database
        if (group_label is None) and isinstance(data_frame, pd.DataFrame):
            group_label = pd_get(data_frame, label, column='group_label', transform='none')
    else:
        group_label = None

    list_comps = None if group_label is None else group_label.split('+')

    # Warn if the grouped line does not have its component label does not have an entry
    if group_type is not None and list_comps is None:

        if verbose:
            _logger.warning(f'The {label} line has a "_{group_type}" suffix but its group lines components were not found  '
                            f'on the input configuration file or lines database. It will be treated a single line')

        label = label[:-2]
        group_type = None

    # Assign parent group_label if child does not have one
    if group_label is None and parent_group_label is not None:
        group_label = parent_group_label

    # Assign essential keys:
    line_params['label'] = label
    line_params['core'] = core

    # Add group keys if they passed all checks
    if group_type:
        line_params['group'] = group_type

    if group_label:
        line_params['group_label'] = group_label

    if list_comps:
        line_params['list_comps'] = list_comps

    return line_params


def get_line_from_cfg(line_params, fit_cfg):

    # Get the line information from the input configuration file if present
    if fit_cfg:

        # Check for transtitions data
        if fit_cfg.get('transitions'):

            # Check for the full label
            if fit_cfg['transitions'].get(line_params['label']):
                line_cfg = fit_cfg['transitions'].get(line_params['label'])
                line_cfg.update(line_params)
                return line_cfg

            # Check for the core label
            elif fit_cfg['transitions'].get(line_params['core']):
                line_cfg = fit_cfg['transitions'].get(line_params['core'])
                line_cfg.update(line_params)
                return line_cfg

            else:
                return line_params

        # Check for line mask
        if fit_cfg.get(f"{line_params['label']}_pixel_mask"):
            line_params['pixel_mask'] = fit_cfg.get(f"{line_params['label']}_pixel_mask")

        return line_params

    else:
        return line_params


def get_line_from_df(line_params, input_df):

    # Get the line information from the dataframe if available
    if input_df is not None:
        if line_params['label'] in input_df.index:
            row_name = line_params['label']
        elif line_params['core'] in input_df.index:
            row_name = line_params['core']
        else:
            row_name = None
    else:
        row_name = None

    # Try the lines database
    if row_name is None and line_params['core'] in rsrc_manager.lineDB.frame.index:
        input_df = rsrc_manager.lineDB.frame
        row_name = line_params['core']

    # Assign the data
    if row_name is not None:
        db_params = dict(particle=pd_get(input_df, row_name, 'particle'),
                         wavelength=pd_get(input_df, row_name, 'wavelength'),
                         units_wave=pd_get(input_df, row_name, 'units_wave'),
                         latex_label=pd_get(input_df, row_name, 'latex_label'),
                         group_label=pd_get(input_df, row_name, 'group_label', transform='none'),

                         trans=pd_get(input_df, row_name, 'trans'),
                         kinem=pd_get(input_df, row_name, 'kinem'),
                         profile=pd_get(input_df, row_name, 'profile'),
                         shape=pd_get(input_df, row_name, 'shape'),
                         )

        if 'w1' in input_df.columns:
            db_params['mask'] = [pd_get(input_df, row_name, 'w1'),
                                 pd_get(input_df, row_name, 'w2'),
                                 pd_get(input_df, row_name, 'w3'),
                                 pd_get(input_df, row_name, 'w4'),
                                 pd_get(input_df, row_name, 'w5'),
                                 pd_get(input_df, row_name, 'w6')]

        # Configuration file overwrite dataframe
        db_params.update(line_params)

        return db_params

    else:
        return line_params


def review_input_params(line_params, def_shape, def_profile):

    # Check core components
    line_items = line_params['label'].split('_')

    # Particle
    if line_params.get('particle') is None:
        line_params['particle'] = line_items[0]

    # Wavelength (for reference blended lines we assign the reference index later)
    if (line_params.get('wavelength') is None) and (line_params.get('group') is None):
        if line_items[1][-1] == 'A':
            line_params['units_wave'] = au.Unit('AA')
            line_params['wavelength']  = float(line_items[1][:-1])
        else:
            au_unit = au.Unit(line_items[1])
            line_params['units_wave'] = au_unit.bases[0]
            line_params['wavelength']  = au_unit.scale

    # Units:
    if line_params.get('units_wave') is None:
        line_items = line_params['label'].split('_')
        line_params['units_wave'] = au.Unit('AA') if line_items[1][-1] else au.Unit(line_items[1]).bases[0]

    # Kinematic component
    if line_params.get('kinem') is None:
        line_params['kinem'] = 0
    else:
        line_params['kinem'] = int(line_params['kinem'])

    # Profile component
    if line_params.get('profile') is None:
        line_params['profile'] = def_profile

    # Line shape (emi, abs)
    if line_params.get('shape') is None:
        line_params['shape'] = def_shape

    # Review latex label
    if line_params.get('group') is None:

        if line_params.get('latex_label') is None:
            construct_classic_notation(line_params=line_params)

        if line_params.get('kinem') > 0 and ('_k-' not in line_params['latex_label']):
            line_params['latex_label'] = line_params['latex_label'][:-1] + f'_k-{line_params["kinem"]}$'

    return


def round_sig_clean(x, sig=4):
    if x == 0:
        return 0
    rounded = round(x, sig - int(np.floor(np.log10(abs(x)))) - 1)
    return int(rounded) if np.isclose(rounded % 1, 0) else rounded



