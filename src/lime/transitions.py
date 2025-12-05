import logging

import numpy as np
import pandas as pd
from pandas import DataFrame
from numpy.core.fromnumeric import argmin

from lime.io import (_LOG_COLUMNS, check_file_dataframe, LiMe_Error, _RANGE_ATTRIBUTES_FIT, _ATTRIBUTES_FIT, load_frame,
                     _LIME_FOLDER)
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
_DATABASE_FILE = rf'{_LIME_FOLDER}/resources/lines_database_v2.0.0.txt'

def check_lines_frame_units(frame):

    if 'units_wave' in frame.columns:
        return au.Unit(frame["units_wave"].iat[0])
    else:
        return au.Unit(Line.from_transition(frame.index[0], data_frame=frame).units_wave)


class LinesDatabase:

    def __init__(self, frame_address=None, default_shape=None, default_profile=None):

        # Default_lines database
        self.frame_address = _DATABASE_FILE if frame_address is None else frame_address
        self.frame = load_frame(self.frame_address)

        # Default values
        self._vacuum_check = False
        self._shape = 'emi' if default_shape is None else default_shape
        self._profile = 'g' if default_profile is None else default_profile
        self.set_units_wave()

        return

    def set_database(self, wave_intvl=None, line_list=None, particle_list=None, redshift=None, units_wave='Angstrom',
                     sig_digits=4, vacuum_waves=False, ref_bands=None, update_labels=False, update_latex=False,
                     exclude_lines=None, default_shape=None, default_profile=None):

        # Reload the database at each modification to avoid contamination
        ref_bands = load_frame(self.frame_address) if ref_bands is None else ref_bands

        self.frame = lines_frame(wave_intvl, line_list, particle_list, redshift, units_wave, sig_digits=sig_digits,
                                 vacuum_waves=vacuum_waves, ref_bands=ref_bands, update_labels=update_labels,
                                 update_latex=update_latex, rejected_lines=exclude_lines)

        self._vacuum_check = vacuum_waves

        self.set_units_wave()

        if default_shape:
            self.set_shape(default_shape)

        if default_profile:
            self.set_profile(default_profile)

        return

    def reset(self, frame_address=None, default_shape=None, default_profile=None):

        self.__init__(frame_address, default_shape, default_profile)

        return

    def copy(self):

        return self.frame.copy()

    def set_shape(self, value):
        self._shape = value
        return

    def set_profile(self, value):
        self._profile = value
        return

    def set_units_wave(self):

        self._units_wave = check_lines_frame_units(self.frame)

        return

    def get_shape(self):
        return self._shape

    def get_profile(self):
        return self._profile

    def get_units(self):
        return self._units_wave


rsrc_manager.lineDB = LinesDatabase(_DATABASE_FILE)


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
    Decompose LiMe line labels into requested physical/formatting parameters.

    Given a list of line labels in LiMe notation, this function returns one or more
    arrays (or scalars) with selected parameters such as the particle label, line
    wavelength, LaTeX label, kinematic setup, profile, and transition information.
    When a bands table and/or a fitting configuration are provided, they are used
    (via :meth:`Line.from_transition`) to enrich or override defaults; otherwise,
    values are derived from the label and LiMe's internal database.

    Parameters
    ----------
    lines_list : list of str
        Sequence of line labels in LiMe notation (e.g., ``["O3_5007A", "H1_4861A"]``).
    bands : pandas.DataFrame or str or pathlib.Path, optional
        Bands table, or a file path to a readable bands table. When provided, its
        information is used by :meth:`Line.from_transition` to refine wavelengths,
        components, labels, etc.
    fit_conf : dict, optional
        Fitting configuration dictionary (e.g., parsed from TOML) used to override
        defaults (wavelengths, blends, shapes/profiles).
    params_list : tuple of {"particle", "wavelength", "latex_label", "kinem", "profile_comp", "transition_comp"}, optional
        Ordered list of output parameters to return. Default is
        ``("particle", "wavelength", "latex_label")``.
    scalar_output : bool, optional
        If ``True`` **and** only a single input label is provided, return scalars
        instead of 1-D arrays. Default is ``False``.
    verbose : bool, optional
        If ``True``, propagate verbose behavior to :meth:`Line.from_transition`.
        Default is ``True``.

    Returns
    -------
    tuple
        A tuple with the same length and order as ``params_list``. Each element is:
        - For multiple input labels: a 1-D ``ndarray`` with one value per label.
        - For a single input label and ``scalar_output=True``: a scalar value.

        The possible entries are:
        - ``"particle"`` : array of str — particle labels (e.g., ``"O3"``).
        - ``"wavelength"`` : array of float — wavelengths.
        - ``"latex_label"`` : array of str — LaTeX-formatted labels.
        - ``"kinem"`` : array of objects — kinematic configuration per line.
        - ``"profile_comp"`` : array of objects — line profile identifiers.
        - ``"transition_comp"`` : array of objects — transition descriptors.

    Notes
    -----
    - Each line is resolved through :meth:`Line.from_transition(label, fit_conf, bands, verbose=verbose)`.
      Missing information is filled from LiMe’s default database where possible.
    - The function constructs an internal DataFrame with columns:
      ``["particle", "wavelength", "latex_label", "kinem", "profile_comp", "transition_comp"]``,
      then extracts the columns requested by ``params_list``.
    - Column ``"wavelength"`` is coerced to numeric.

    Examples
    --------
    Return particle, wavelength, and LaTeX label arrays for two lines:

    >>> particles, waves, latex = label_decomposition(
    ...     ["O3_5007A", "H1_4861A"],
    ...     params_list=("particle", "wavelength", "latex_label")
    ... )

    Use a fitting configuration and bands table; request only wavelengths:

    >>> (waves,) = label_decomposition(
    ...     ["O2_3726A", "O2_3729A"],
    ...     bands=bands_df,
    ...     fit_conf=fit_cfg,
    ...     params_list=("wavelength",)
    ... )

    Single label with scalar output:

    >>> particle, wl = label_decomposition(
    ...     ["O3_5007A"],
    ...     params_list=("particle", "wavelength"),
    ...     scalar_output=True
    ... )
    >>> isinstance(wl, float)
    True
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
                vacuum_waves=False, ref_bands=None, update_labels=False, update_latex=False, rejected_lines=None):
    
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

        for i, label_i in enumerate(bands_df.index):
            line = Line.from_transition(label_i, data_frame=bands_df)
            line.update_labels(sig_digits=sig_digits)

            if update_labels:
                labels[i] = line.label
            if update_latex:
                latex_list[i] = line.latex_label

        if update_latex:
            bands_df["latex_label"] = latex_list

        if update_labels:
            bands_df.rename(index=dict(zip(bands_df.index, labels)), inplace=True)

    # Exclude lines
    if rejected_lines is not None:
        bands_df = bands_df.loc[~bands_df.index.isin(rejected_lines)]

    return bands_df


def bands_from_measurements(frame, sample_levels=['id', 'line'], sort=True, remove_empty_columns=False, index_dict=None,
                            bands_hdrs=None):

    """
    Re-constructs the lines frames from an output measurements lines frame consolidating the merged and blended lines.

    Given a LiMe measurements table (or a path to one), this function produces a bands DataFrame with
    wavelength limits (``w1``–``w6``), representative wavelength, grouping labels, and basic metadata
    (units, particle, transition, LaTeX label). It supports both single-index and MultiIndex inputs and
    handles merged/blended groups, including renaming and de-duplication logic.

    Parameters
    ----------
    frame : pandas.DataFrame or str or pathlib.Path
        Measurements table or a file path that can be read into a measurements DataFrame.
        The table is validated/loaded via ``check_file_dataframe``.
    sample_levels : list of str, optional
        For MultiIndex inputs, the index levels that identify a line sample (e.g., ``["id", "line"]``).
        Default is ``["id", "line"]``.
    sort : bool, optional
        If ``True``, sort the output by ``["wavelength", "group_label"]``. Default is ``True``.
    remove_empty_columns : bool, optional
        If ``True``, drop columns that are entirely ``NaN`` in the result. Default is ``False``.
    index_dict : dict, optional
        Mapping to rename output line labels. Applied at the end of processing.
    bands_hdrs : sequence of str, optional
        Column set for the output table. When not provided the default is
        ``("wavelength","w1","w2","w3","w4","w5","w6","group_label","units_wave","particle","trans","latex_label")``.

    Returns
    -------
    pandas.DataFrame
        A bands table indexed by line labels (with ``_b`` suffix for blended groups and ``_m`` for merged
        labels when applicable).

    Notes
    -----
    - **Single-index frames:**

      - Single lines are selected where ``group_label == "none"`` and copied directly.
      - Grouped lines are consolidated by unique ``group_label``, and blended labels get a ``_b`` suffix
        if they do not already end with ``"_m"``.

    - **MultiIndex frames:**

      - Unique line labels are retrieved from the given ``sample_levels``.
      - For each line and each group in its rows, the band limits (``w1``–``w6``) are computed as the **median**
        across matching entries.
      - Labels are determined as:
        - Single (``group == "none"``): the original line label.
        - Merged: preserve the ``_m`` label.
        - Blended: add ``_b`` to the label; if already present in the index, a disambiguated label like ``"{species}-{j}_{rest}_b"`` is created.

      - A minimal single-row reference DataFrame is built and passed to :meth:`Line.from_transition`
        to recover canonical metadata (wavelength, units, particle, LaTeX label, mask).
      - The representative wavelength and units are taken from the line’s reference component; mask values
        populate ``w1``–``w6``.

    - **Blended lines with merged components:**

      - Rows sharing identical band limits (``w1``–``w6``) and with ``group_label != "none"`` are considered duplicates.
      - Duplicates are collapsed into a single blended entry. The surviving row’s ``group_label`` becomes the
        ``"+"``-joined list of merged components, and its index is renamed with a ``_b`` suffix
        (or from ``*_m`` → ``*_b``).
      - If this auto-generated label isn’t in ``index_dict``, a warning is logged suggesting you add a stable rename.

    - **Sorting and cleanup:** ``sort`` orders the result; ``remove_empty_columns`` prunes all-NaN columns;
      ``index_dict`` renames the final index.

    Examples
    --------
    From a single-index measurements table:

    >>> bands = bands_from_measurements(frame)

    With MultiIndex and custom sample levels, dropping empty columns:

    >>> bands = bands_from_measurements(
    ...     frame,
    ...     sample_levels=["object_id", "line"],
    ...     remove_empty_columns=True
    ... )

    Enforcing stable external IDs:

    >>> rename_map = {"H1_6563A_b": "H1-N2_6563A_b"}
    >>> bands = bands_from_measurements(frame, index_dict=rename_map)
    """

    # Load the frame if necessary
    frame = check_file_dataframe(frame, sample_levels=sample_levels)

    # Single frame
    if not isinstance(frame.index, pd.MultiIndex):

        bands_hdrs = ('wavelength', 'w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'group_label', 'units_wave',
                                        'particle', 'trans', 'latex_label') if bands_hdrs is None else None

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

    # Check for blended merged lines
    idcs_merged_blended = bands.duplicated(subset=['w1', 'w2', 'w3', 'w4', 'w5', 'w6'], keep=False) & (bands.group_label != 'none')
    duplicate_groups = (bands.loc[idcs_merged_blended].groupby(['w1', 'w2', 'w3', 'w4', 'w5', 'w6']).filter(lambda x: len(x) > 1)  # Filter for groups with more than one row (i.e., duplicates)
                        .groupby(['w1', 'w2', 'w3', 'w4', 'w5', 'w6']).groups)
    idcs_repeated_bands = [index.tolist() for index in duplicate_groups.values()]

    if len(idcs_repeated_bands) > 0:
        indexes_to_drop, rename_groups = [], {}
        for group_merged in idcs_repeated_bands:
            bands.loc[group_merged[0], 'group_label'] = '+'.join(group_merged)
            indexes_to_drop += group_merged[1:]
            rename_groups[group_merged[0]] = group_merged[0] + '_b' if not group_merged[0].endswith("_m") else group_merged[0][:-2] + '_b'
            if index_dict is None or (rename_groups[group_merged[0]] not in index_dict):
                _logger.warning(f'\nBlended line "{group_merged[0]}" with merged components has been renamed "{rename_groups[group_merged[0]]}={bands.loc[group_merged[0], "group_label"]}"'
                                f'\nIt is adviced rename via index_dict argument to match future measurements.')
            
        bands.drop(indexes_to_drop, inplace=True)
        bands.rename(rename_groups, inplace=True)

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


def check_continua_bands(idcs, wave_rest, min_width_pixel = 1, min_sep_cont=2, reset_w2_w5=False):

    if reset_w2_w5:
        if idcs[1] > idcs[2]:
            idcs[1] = idcs[2] - min_sep_cont
        if idcs[0] >= idcs[1]:
            idcs[0] = idcs[1] - (min_width_pixel + min_sep_cont)

        if idcs[4] < idcs[3]:
            idcs[4] = idcs[3] + min_sep_cont
        if idcs[5] <= idcs[4]:
            idcs[5] = idcs[4] + (min_width_pixel + min_sep_cont)

    # Continua bands beyond the spectral wavelength range
    if idcs[0] < 0:
        idcs[0] = 0

    if idcs[1] < 0:
        idcs[1] = idcs[0] + min_width_pixel

    if idcs[5] > wave_rest.size - 1:
        idcs[5] = wave_rest.size - min_width_pixel

    if idcs[4] > wave_rest.size - 1:
        idcs[4] = idcs[5] - min_width_pixel

    # One pixel bands width
    if idcs[0] == idcs[1]:
        idcs[1] = idcs[0] + min_width_pixel

    if idcs[2] == idcs[3]:
        idcs[3] = idcs[2] + min_width_pixel

    if idcs[4] == idcs[5]:
        idcs[4] = idcs[5] - min_width_pixel

    return idcs


class Particle:

    """
    Representation of an atomic or ionic species used in spectral line definitions.

    A :class:`Particle` object encodes the physical identity of a species through its
    label, atomic symbol, and ionization stage. It provides convenience methods for
    reconstructing these attributes from a shorthand label (e.g., ``"O3"`` → oxygen,
    doubly ionized).

    Parameters
    ----------
    label : str, optional
        Canonical particle label (e.g., ``"H1"``, ``"O3"``, ``"He2"``). This string
        uniquely identifies the species and ionization stage within LiMe.
    symbol : str, optional
        Chemical symbol of the species (e.g., ``"O"`` for oxygen, ``"H"`` for hydrogen).
    ionization : int, optional
        Ionization stage of the particle, typically an integer where
        ``1 = neutral``, ``2 = singly ionized``, etc.

    Attributes
    ----------
    label : str
        Canonical identifier for the species (e.g., ``"O3"``).
    symbol : str
        Atomic symbol.
    ionization : int
        Ionization stage (1 = neutral, 2 = singly ionized, ...).

    Methods
    -------
    from_label(label)
        Create a :class:`Particle` from a shorthand label string (e.g., ``"O3"``).
    __eq__(other)
        Return ``True`` if two particles (or a particle and a label string) are equivalent.
    __ne__(other)
        Return ``True`` if two particles are different.

    Examples
    --------
    Create a particle manually:

    >>> Particle(label="O3", symbol="O", ionization=3)
    O3

    Construct a particle automatically from a label string:

    >>> Particle.from_label("He2")
    He2

    Compare particles:

    >>> Particle.from_label("O3") == Particle.from_label("O3")
    True
    >>> Particle.from_label("O3") == "O2"
    False
    """


    def __init__(self, label: str = None, symbol: str = None, ionization: int = str):

        self.label = label
        self.symbol = symbol
        self.ionization = ionization

        return

    @classmethod
    def from_label(cls, label):

        """
        Create a :class:`Particle` instance from a shorthand label string.

        This class method parses the label into its elemental symbol and ionization
        stage using :func:`recover_ionization`.

        Parameters
        ----------
        label : str
            Species identifier string (e.g., ``"H1"``, ``"O3"``).

        Returns
        -------
        Particle
            A :class:`Particle` instance with the corresponding ``symbol`` and
            ``ionization`` attributes.

        Examples
        --------
        >>> Particle.from_label("O3")
        O3
        >>> Particle.from_label("He2").symbol
        'He'
        """

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

    """

    Spectral line container with metadata, grouping, and measurement hooks.

    A ``Line`` holds the identifying metadata for a spectral feature (e.g., label,
    particle/ion, rest wavelength), optional grouping information (for blends or
    multiplets), and references to kinematics/transition/profile details used by LiMe.
    For grouped lines, a *reference component* is tracked via ``ref_idx`` to define the
    line’s representative wavelength.

    Parameters
    ----------
    label : str
        Human-readable identifier for the line (e.g., ``"H1_4861A"`` or ``"O3_5007A"``).
    particle : str or Particle, optional
        Species identifier. Converted to a :class:`Particle` via
        ``Particle.from_label(particle)``.
    wavelength : float, optional
        Rest (or reference) wavelength of the line. Units given by ``units_wave``.
    units_wave : str, optional
        Wavelength units (e.g., ``"Angstrom"``).
    latex_label : str, optional
        LaTeX-formatted label for rendering in plots or tables.
    core : Line, optional
        Core component of a blended/grouped feature. If ``group == "b"`` (blend) and
        ``core`` is included in ``list_comps``, its index is used as the reference.
    group_label : str, optional
        Group identifier for collections of related components (e.g., blend name).
    group : str, optional
        Group type flag. When ``"b"``, the line is treated as part of a blend and
        the reference component is determined as described in **Notes**.
    list_comps : list of Line, optional
        Explicit list of component lines comprising a grouped feature. If ``None``,
        a single-component list ``[self]`` is used.
    mask : any, optional
        User-defined mask/flag for downstream processing.
    kinem : any, optional
        Kinematic information (e.g., velocity/dispersion constraints) used by fitters.
    trans : any, optional
        Transition descriptor passed through ``recover_transition(self.particle, trans)``.
    profile : any, optional
        Line profile model identifier (e.g., Gaussian, Voigt).
    shape : any, optional
        Optional shape constraints or metadata for modeling.
    pixel_mask : str or any, optional
        Pixel mask mode/flag. Defaults to ``"no"`` when not provided.

    Attributes
    ----------
    label : str
    particle : Particle
        Result of ``Particle.from_label(particle)``.
    wavelength : float
    units_wave : str
    latex_label : str or None
    core : Line or None
    group_label : str or None
    group : str or None
    list_comps : list of Line
        Component list; defaults to ``[self]`` when not provided.
    ref_idx : int
        Index of the reference component within ``list_comps``.
    mask : any
    pixel_mask : str or any
        Defaults to ``"no"`` if not given.
    kinem : any
    trans : any
        Result of ``recover_transition(self.particle, trans)``.
    profile : any
    shape : any
    measurements : any or None
        Placeholder for later measurement results (populated downstream).

    Examples
    --------
    Single, standalone line:

    >>> Hbeta = Line(label="H1_4861A", particle="H1", wavelength=4861.33, units_wave="Angstrom")

    Blended feature with explicit core component:

    >>> comp1 = Line(label="O2_3726A", particle="O2", wavelength=3726.03, units_wave="Angstrom")
    >>> comp2 = Line(label="O2_3729A", particle="O2", wavelength=3728.82, units_wave="Angstrom")
    >>> OII_blend = Line(label="O2_3726A", group="b", list_comps=[comp1, comp2], core=comp1)
    >>> OII_blend.ref_idx  # uses core component index

    """

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
                        def_shape=None, def_profile=None, verbose=True, warn_missing_db=False):

        """
        Construct a :class:`Line` instance from transition label and optional user input fit_cfg and lines frame.

        This class method compiles all relevant parameters for a spectral line following a defined hierarchy of input sources.
        At the lowest level, default values are retrieved from LiMe’s internal line database. These can be overridden by entries
        in the user-provided lines frame, which in turn are superseded by parameters in the fitting configuration (`fit_cfg`).
        Finally, label suffixes take the highest precedence.

        Grouped (blended or merged) lines are recursively reconstructed into the ``list_comps`` attribute.


        Parameters
        ----------
        label : str
            Identifier of the target spectral line (e.g., ``"O3_5007A"`` or ``"H1_4861A"``).
        fit_cfg : dictionary, optional
            Fitting configuration defining line metadata such as wavelength, particle,
            and/or group components. This dictionary can be generated from a TOML file,
            where each transition is defined by its label.
            Example TOML snippet:

        data_frame : pandas.DataFrame, optional
            Table containing line measurement data.
            The expected columns are:

            ``"wavelength", "wave_vac", "w1", "w2", "w3", "w4", "w5", "w6", "latex_label", "units_wave", "particle", "trans"``

            If any of these columns are missing, the function attempts to recover missing
            information from LiMe’s default database at ``lime.lineDB``.

        parent_group_label : str, optional
            Label for the parent group when constructing component lines of a blend or merged group.
            Used internally during recursive creation.

        norm_flux : float, optional
            Normalization factor for fluxes in the measurements table.

        def_shape : str, optional
            Default line shape to use if not provided in the configuration or database.
            Defaults to ``rsrc_manager.lineDB.get_shape()``.

        def_profile : str, optional
            Default line profile model to use if not provided in the configuration or database.
            Defaults to ``rsrc_manager.lineDB.get_profile()``.

        verbose : bool, optional
            If ``True``, print informative messages during line reconstruction.

        warn_missing_db : bool, optional
            If ``True``, issue warnings when the line label is not found in the LiMe database.

        Returns
        -------
        Line
            A fully constructed :class:`Line` instance. For grouped or blended transitions,
            this object includes a populated ``list_comps`` of component :class:`Line` objects
            and a combined ``latex_label``.

        Notes
        -----
        - The function calls :func:`parse_container_data` to merge information from the
          configuration, measurement table, and LiMe’s default database.
        - If ``data_frame`` corresponds to a valid measurements table (verified via
          :func:`check_measurements_table`), the resulting ``Line`` includes a
          :class:`LineMeasurements` instance loaded with the corresponding data.
        - Grouped transitions (blends) are recursively reconstructed via
          :meth:`Line.from_transition` for each component listed in
          ``line_params['list_comps']``.

        Examples
        --------
        Create a single emission line directly from the database:

        >>> Hbeta = Line.from_transition("H1_4861A")

        If reading the configuration from a TOML file such as the one below:

        .. code-block:: toml

           transitions.O2_3726A_m.wavelength = 3728.484
           transitions.O2_7325A_m.wavelength = 7325.000
           transitions.O2_7325A_b.wavelength = 7325.000

           O2_3726A_m = 'O2_3726A+O2_3729A'
           O2_3726A_b = 'O2_3726A+O2_3729A'

        The line can be generated as:

        >>> fit_cfg_dict = Line.load_cfg("conf.toml")
        >>> OII = Line.from_transition("O2_3726A_m", fit_cfg=fit_cfg_dict)

        Or including a lines frame:

        >>> line = Line.from_transition("O3_5007A", fit_cfg=fit_cfg_dict, data_frame="lines_table.txt")
        """

        # Extract line parameters from the input containers
        line_params = parse_container_data(label,
                                           fit_cfg,
                                           data_frame,
                                           parent_group_label=parent_group_label,
                                           def_shape=rsrc_manager.lineDB.get_shape() if def_shape is None else def_shape,
                                           def_profile=rsrc_manager.lineDB.get_profile() if def_profile is None else def_profile,
                                           verbose=verbose,
                                           warn_missing_db=warn_missing_db)

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
                                                                    verbose=False,
                                                                    warn_missing_db=warn_missing_db)

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


def parse_container_data(label, fit_cfg, data_frame, parent_group_label, def_shape, def_profile, verbose, warn_missing_db=False):

    # Review line notation and its group structure
    line_params = get_line_group(label, fit_cfg, data_frame, parent_group_label, verbose)

    # Check for the line data on the configuration file
    line_params = get_line_from_cfg(line_params, fit_cfg)

    # The configuration parameters overwrite those from the database
    line_params = get_line_from_df(line_params, data_frame, warn_missing_db)

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
                # line_cfg = {**fit_cfg['transitions'].get(line_params['label']), **line_params}
                # line_cfg.update(line_params)
                return {**fit_cfg['transitions'].get(line_params['label']), **line_params}

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


def get_line_from_df(line_params, input_df, warn_missing_db=False):

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

        # Print warning if requested for a line not present on the database
        if warn_missing_db and line_params.get('group') is None:
            _logger.warning(f'Line {line_params["label"]} was not found on the database')

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



