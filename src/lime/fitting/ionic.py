import logging
import re
import warnings
import numpy as np
from dataclasses import dataclass, field
from scipy.sparse import csr_matrix
from scipy.special import wofz
from lmfit import Parameters, Minimizer, fit_report

from lime.io import LiMe_Error, check_fit_conf, check_file_dataframe, deep_dict_update, lime_cfg
from lime.tools import au, LSF, check_lsf
from lime.transitions import Line
from lime.fitting.lines import c_KMpS, tiny, _VERBOSE_WARNINGS


_logger = logging.getLogger('LiMe')

# Physical constants in cgs units (VoigtFit voigt.Voigt)
c_CGS = 2.99792458e10   # Speed of light (cm/s)
m_e_CGS = 9.1094e-28    # Electron mass (g)
e_CGS = 4.8032e-10      # Electron charge (esu)

# Starting values for ions missing in the lime.toml [voigt_default_params]
_DEFAULT_B = 10.0
_DEFAULT_LOGN = 14.0

# Column density below which a fitted component is considered undetected (its logN runs away)
_LOGN_UNDETECTED = 6.0

# Default configuration for the ionic components parameters (VoigtFit bounds: only b > 0)
_Z_ION_PAR = dict(value=None, min=-np.inf, max=np.inf, vary=True, expr=None)
_B_ION_PAR = dict(value=None, min=0, max=np.inf, vary=True, expr=None)
_LOGN_ION_PAR = dict(value=None, min=-np.inf, max=np.inf, vary=True, expr=None)

# Entries recognized at the "kinem" configuration levels
_KINEM_KEYS = ('z', 'b', 'logN', 'var_z', 'var_b', 'var_N')
_TIE_KEY = re.compile(r'^k-(\d+)_tie$')


def voigt_tau(wave_arr, lambda_0, osc_str, gamma, z, b, logN):

    """
    Optical depth of an ionic transition following VoigtFit ``voigt.Voigt``.

    The optical depth is computed as :math:`\\tau = K N H(a, x)`, where the Voigt-Hjerting function
    :math:`H(a, x) = Re[w(x + ia)]` is computed from the Faddeeva function (VoigtFit uses the Tepper-Garcia
    approximation).

    Parameters
    ----------
    wave_arr : numpy.ndarray
        Observed wavelength array in Angstroms.
    lambda_0 : float
        Rest frame wavelength of the transition in Angstroms.
    osc_str : float
        Transition oscillator strength.
    gamma : float
        Transition damping constant in s^-1 (the ``trans_prob`` column in the LiMe database).
    z : float
        Redshift of the absorbing component.
    b : float
        Doppler parameter in km/s.
    logN : float
        Column density in log10(cm^-2).

    Returns
    -------
    numpy.ndarray
        Optical depth at the ``wave_arr`` wavelengths.
    """

    # Doppler parameter in cm/s
    b_cgs = max(tiny, b) * 1e5

    # Frequency offset and damping parameter in Doppler units
    x = (c_CGS / b_cgs) * (1.0 - lambda_0 * (1 + z) / wave_arr)
    a = lambda_0 * 1e-8 * gamma / (4 * np.pi * b_cgs)

    # Optical depth constant
    k_tau = np.sqrt(np.pi) * e_CGS * e_CGS * osc_str * lambda_0 * 1e-8 / (m_e_CGS * c_CGS * b_cgs)

    return k_tau * np.power(10.0, logN) * wofz(x + 1j * a).real


def origin_key(line):

    # Lines without an origin belong to the observation ("none" in the configuration)
    origin = line.origin
    if (origin is None) or (isinstance(origin, float) and np.isnan(origin)):
        origin = 'none'

    return origin


def kinem_param_conf(fit_cfg, ion, origin, kinem):

    """
    Configuration of an ionic component from the "kinem" entries of the fitting configuration.

    The entries are read with increasing priority: origin < kinematic component < ion < ion kinematic component:

    .. code-block:: toml

       kinem.MW.var_z = false          # All the MW components
       kinem.MW.k-1.z = 0.0001         # The kinematic component 1 of all the MW ions
       kinem.MW.SiII.logN = 13.5       # All the MW SiII components
       kinem.MW.SiII.k-1.logN = 13.0   # The kinematic component 1 of the MW SiII
    """

    # Recover the origin and ion entries
    origin_cfg = fit_cfg.get('kinem', {}).get(origin, {})
    ion_cfg = origin_cfg.get(ion, {})

    # Origin < kinematic component < ion < ion kinematic component
    comp_conf = {key: value for key, value in origin_cfg.items()
                 if not isinstance(value, dict) and _TIE_KEY.match(key) is None}
    comp_conf.update(origin_cfg.get(f'k-{kinem}', {}))
    comp_conf.update({key: value for key, value in ion_cfg.items() if not isinstance(value, dict)})
    comp_conf.update(ion_cfg.get(f'k-{kinem}', {}))

    # Warn about entries which are not recognized
    for key in comp_conf:
        if key not in _KINEM_KEYS:
            _logger.warning(f'The configuration entry "{key}" for the {ion} component (origin = {origin}, kinematic '
                            f'component = {kinem}) is not recognized. Please use: {", ".join(_KINEM_KEYS)}')

    return comp_conf


def band_pixel_error(wave_arr, flux_arr, idcs_cont, min_pixels=3):

    # Linear fit to the adjacent continua and standard deviation of its residuals (as LiMe err_from_bands)
    x_cont, y_cont = wave_arr[idcs_cont], flux_arr[idcs_cont]
    if x_cont.size < min_pixels:
        return None

    m_cont, n_cont = np.polyfit(x_cont, y_cont, 1)

    return np.std(y_cont - (m_cont * x_cont + n_cont), ddof=2)


def lsf_convolution_matrix(lsf, wave_arr, width_arr):

    """
    Sparse matrix with the LSF convolution and pixel integration of a model evaluated on a sub-pixel grid.

    The model is computed on ``lsf.nsub`` sub-pixels per data pixel, extended beyond the region limits by the kernel
    half-width. Each output sub-pixel is convolved with the LSF at its wavelength (VoigtFit per-pixel kernel, flipped so
    asymmetric kernels act as a true convolution) and the sub-pixels are averaged into the data pixels. Since the LSF does
    not change during the fitting the matrix is computed only once.

    Parameters
    ----------
    lsf : LSF
        Instrument line spread function.
    wave_arr : numpy.ndarray
        Wavelength of the data pixels in Angstroms.
    width_arr : numpy.ndarray
        Width of the data pixels in Angstroms.

    Returns
    -------
    sub_wave : numpy.ndarray
        Model sub-pixel grid in Angstroms.
    conv_matrix : scipy.sparse.csr_matrix
        Matrix with shape ``(wave_arr.size, sub_wave.size)`` such that ``model_pixels = conv_matrix @ model_sub``.
    """

    nsub, n_samples = lsf.nsub, lsf.kernel.shape[0]
    half_kernel = n_samples // 2

    # Extend the data grid by the kernel half-width (in data pixels)
    n_ext = int(np.ceil(half_kernel / nsub)) + 1
    ext_blue = wave_arr[0] - width_arr[0] * np.arange(n_ext, 0, -1)
    ext_red = wave_arr[-1] + width_arr[-1] * np.arange(1, n_ext + 1)
    pixel_wave = np.concatenate([ext_blue, wave_arr, ext_red])
    pixel_width = np.concatenate([np.full(n_ext, width_arr[0]), width_arr, np.full(n_ext, width_arr[-1])])

    # Sub-pixel centers
    sub_offsets = (np.arange(nsub) + 0.5) / nsub - 0.5
    sub_wave = (pixel_wave[:, None] + sub_offsets[None, :] * pixel_width[:, None]).ravel()

    # Kernels for the sub-pixels within the data pixels
    idcs_out = n_ext * nsub + np.arange(wave_arr.size * nsub)
    kernel_arr = lsf(sub_wave[idcs_out])

    # Output sub-pixel i collects the model at i - (j - half_kernel) with weight kernel_i[j] (true convolution)
    rows = np.repeat(np.arange(idcs_out.size) // nsub, n_samples)
    cols = (idcs_out[:, None] - (np.arange(n_samples)[None, :] - half_kernel)).ravel()
    conv_matrix = csr_matrix((kernel_arr.ravel() / nsub, (rows, cols)), shape=(wave_arr.size, sub_wave.size))

    return sub_wave, conv_matrix


def param_value_err(params, name):

    # Recover the parameter value and its uncertainty (nan if not available)
    param = params[name]
    param_err = np.nan if param.stderr is None else param.stderr

    # Check for negative -0.0 # FIXME -0.0 error
    if np.signbit(param_err):
        param_err = np.nan

    return param.value, param_err



@dataclass
class IonComponent:

    """Absorbing component of an ion: one per ion, origin and kinematic component"""

    ion: str
    origin: str
    kinem: int
    z_ref: float
    names: dict = field(default_factory=dict)
    conf: dict = field(default_factory=dict)


@dataclass
class IonTerm:

    """Absorption of a transition by one of its ion components (a row in the output lines frame)"""

    label: str
    core: str
    comp_key: tuple
    line: object
    band_line: object
    declared: bool
    regions: list = field(default_factory=list)


@dataclass
class FitRegion:

    """Contiguous spectrum interval with one or more (overlapping) line bands"""

    idx_0: int
    idx_f: int
    bands: list = field(default_factory=list)
    wave: np.ndarray = None
    flux: np.ndarray = None
    err: np.ndarray = None
    mask: np.ndarray = None
    sub_wave: np.ndarray = None
    matrix: object = None
    terms: list = field(default_factory=list)
    model: np.ndarray = None


class IonicModelCompiler:

    """
    Forward model of the ionic absorptions following VoigtFit: the optical depth of every transition and component is
    computed on a sub-pixel grid, the normalized flux exp(-tau) is convolved with the instrument LSF, integrated in the
    data pixels and fitted in all the regions simultaneously.
    """

    def __init__(self, spec, line_list, fit_cfg, bands=None, err_from_bands=False, min_pixels=3):

        # Containers
        self.lsf = spec.lsf
        self.z_ref = spec.redshift
        self.transitions = {}
        self.components = {}
        self.terms = []
        self.regions = []
        self.params = None
        self.output = None
        self._bands = []
        self._declared = {}

        # Observation in Angstroms and normalized flux units (continuum = 1)
        try:
            wave_factor = au.Unit(spec.units_wave).to(au.AA)
        except Exception:
            raise LiMe_Error(f'The ionic absorptions fitting requires wavelength units in the spectrum dispersion axis '
                             f'(current units: {spec.units_wave})')

        self._wave = spec.wave.data * wave_factor
        self._width = np.gradient(self._wave)
        self._flux = spec.flux.data * spec.norm_flux
        self._err = None if spec.err_flux is None else spec.err_flux.data * spec.norm_flux
        self._pixel_mask = np.ma.getmaskarray(spec.flux)

        # Lines bands, transitions and components
        self.declared_lines(spec, line_list, err_from_bands, min_pixels)

        # Regions with the merged bands and their LSF matrices
        self.compile_regions()

        # Components parameters configuration
        self.components_conf(fit_cfg, bands)

        # Transitions absorbed by each component and their regions
        self.compile_terms(fit_cfg, bands)

        # Compile the LmFit parameters
        self.compile_params()

        return

    def declared_lines(self, spec, line_list, err_from_bands, min_pixels):

        # Containers for the band uncertainty
        missing_data = []
        err_sum, err_count = np.zeros(self._wave.size), np.zeros(self._wave.size)
        limit_blue, limit_red = spec.wave.data[0], spec.wave.data[-1]

        for line in line_list:

            # Check if the line bands are provided
            if line.mask is None:
                _logger.warning(f'Line {line} was not found on the input bands database. It won\'t be measured')
                continue

            # The bands redshift is the one from the line (z_line or origin) or the observation
            z_band = self.z_ref if line.redshift is None else line.redshift

            # Check if the line is within the w3, w4 limits
            w3_obs, w4_obs = line.mask[2] * (1 + z_band), line.mask[3] * (1 + z_band)
            if (w3_obs < limit_blue) or (w4_obs > limit_red):
                _logger.warning(f'Line {line} bands are outside spectrum wavelength range: w3 < w_min ({w3_obs:0.2f} < '
                                f'{limit_blue:0.2f}) or w4 > w_max ({w4_obs:0.2f} > {limit_red:0.2f}). '
                                f'It won\'t be measured')
                continue

            # Line band pixels (with the line pixel mask) and adjacent continua
            idcs_line, idcs_cont = line.index_bands(spec.wave, z_band)
            idx_0, idx_f = line.index_bands(spec.wave, z_band, just_band_edges=True)[2:4]
            idcs_valid = idcs_line & ~self._pixel_mask & np.isfinite(self._flux)

            # Check the band has enough pixels
            if np.sum(idcs_valid) < min_pixels:
                _logger.warning(f'Line {line} has only {np.sum(idcs_valid)} pixels. It won\'t be measured')
                continue

            # Pixel uncertainty from the adjacent continua
            if err_from_bands:
                band_err = band_pixel_error(self._wave, self._flux, idcs_cont & ~self._pixel_mask, min_pixels)
                if band_err is None:
                    _logger.warning(f'Line {line} continuum bands have less than {min_pixels} pixels to compute the '
                                    f'uncertainty. It won\'t be measured')
                    continue
                err_sum[idx_0:idx_f] += band_err
                err_count[idx_0:idx_f] += 1

            # Store the band
            idx_band = len(self._bands)
            self._bands.append(dict(line=line, idx_0=idx_0, idx_f=idx_f, idcs_valid=idcs_valid))

            # Unpack the merged components inside blended groups
            comp_list = []
            for comp in line.list_comps:
                comp_list += comp.list_comps if (comp.group == 'm' and comp is not line) else [comp]

            # Loop through the transitions in the band
            for comp in comp_list:

                # Check the atomic data
                osc_str = None if comp.atom_data is None else comp.atom_data.osc_str
                gamma = None if comp.atom_data is None else comp.atom_data.trans_prob
                if (osc_str is None) or (gamma is None):
                    missing_data.append(comp.label)
                    continue

                # Transition (one per core label)
                ion = comp.classic_notation(just_particle=True)
                if comp.core not in self.transitions:
                    lambda_0 = comp.wavelength * au.Unit(comp.units_wave).to(au.AA)
                    self.transitions[comp.core] = dict(lambda_0=lambda_0, osc_str=osc_str, gamma=gamma, ion=ion)

                # Component (one per ion, origin and kinematic component)
                comp_key = (ion, origin_key(comp), comp.kinem)
                if comp_key not in self.components:
                    z_comp = self.z_ref if comp.redshift is None else comp.redshift
                    self.components[comp_key] = IonComponent(ion, comp_key[1], comp.kinem, z_comp)

                # Transition absorbed by this component in the input bands
                self._declared[(comp.core, comp_key)] = (comp, idx_band)

        # Stop for missing atomic data
        if len(missing_data) > 0:
            raise LiMe_Error(f'Missing atomic data (osc_str, trans_prob) for the transitions: {", ".join(missing_data)}. '
                             f'Please include them in the input bands')

        if len(self._bands) == 0:
            raise LiMe_Error(f'None of the input lines could be measured')

        # Pixel uncertainty
        if err_from_bands:
            with np.errstate(invalid='ignore', divide='ignore'):
                self._err = err_sum / err_count

        return

    def compile_regions(self):

        # Merge the overlapping bands into regions (the pixels enter the fitting only once)
        for idx_band in np.argsort([band['idx_0'] for band in self._bands]):
            band = self._bands[idx_band]
            if (len(self.regions) > 0) and (band['idx_0'] < self.regions[-1].idx_f):
                self.regions[-1].idx_f = max(self.regions[-1].idx_f, band['idx_f'])
                self.regions[-1].bands.append(idx_band)
            else:
                self.regions.append(FitRegion(band['idx_0'], band['idx_f'], [idx_band]))

        # Data, fitting mask and LSF convolution matrix for each region
        for region in self.regions:

            idcs = slice(region.idx_0, region.idx_f)
            region.wave, region.flux, region.err = self._wave[idcs], self._flux[idcs], self._err[idcs]

            # Pixels within the bands with valid data and uncertainty
            idcs_valid = np.zeros(self._wave.size, dtype=bool)
            for idx_band in region.bands:
                idcs_valid |= self._bands[idx_band]['idcs_valid']
            region.mask = idcs_valid[idcs] & np.isfinite(region.err) & (region.err > 0)

            # Check the region is covered by the LSF table
            if (region.wave[0] < self.lsf.wave[0]) or (region.wave[-1] > self.lsf.wave[-1]):
                _logger.warning(f'The region [{region.wave[0]:0.2f}, {region.wave[-1]:0.2f}] Angstrom is not fully '
                                f'covered by the LSF table [{self.lsf.wave[0]:0.2f}, {self.lsf.wave[-1]:0.2f}] Angstrom. '
                                f'The closest table kernel will be used')

            region.sub_wave, region.matrix = lsf_convolution_matrix(self.lsf, region.wave, self._width[idcs])

        return

    def components_conf(self, fit_cfg, bands):

        # Default starting values: lime.toml < configuration file "voigt_default_params"
        default_params = deep_dict_update(lime_cfg.get('voigt_default_params', {}),
                                          fit_cfg.get('voigt_default_params', {}))
        missing_defaults = []

        for comp_key, comp in self.components.items():

            ion, origin, kinem = comp_key
            comp_conf = kinem_param_conf(fit_cfg, ion, origin, kinem)

            # LmFit parameters labels
            comp_label = re.sub(r'\W', '_', f'{ion}_{origin}_k{kinem}')
            comp.names = {param: f'{comp_label}_{param}' for param in ('z', 'b', 'logN')}

            # Starting values: defaults < configuration
            b_0 = default_params.get('b', {}).get(ion)
            logN_0 = default_params.get('logN', {}).get(ion)
            if ((b_0 is None) and ('b' not in comp_conf)) or ((logN_0 is None) and ('logN' not in comp_conf)):
                missing_defaults.append(ion)

            b_0 = _DEFAULT_B if b_0 is None else b_0
            logN_0 = _DEFAULT_LOGN if logN_0 is None else logN_0

            comp.conf['z'] = {**_Z_ION_PAR, 'value': comp_conf.get('z', comp.z_ref), 'vary': comp_conf.get('var_z', True)}
            comp.conf['b'] = {**_B_ION_PAR, 'value': comp_conf.get('b', b_0), 'vary': comp_conf.get('var_b', True)}
            comp.conf['logN'] = {**_LOGN_ION_PAR, 'value': comp_conf.get('logN', logN_0), 'vary': comp_conf.get('var_N', True)}

        # Warn for ions without starting values
        if len(missing_defaults) > 0:
            _logger.warning(f'No starting b or logN for {", ".join(np.unique(missing_defaults))} in the lime.toml '
                            f'[voigt_default_params] nor the configuration. Using b = {_DEFAULT_B} km/s and '
                            f'logN = {_DEFAULT_LOGN}')

        # Tie the redshift and Doppler parameter of the kinematic components to a parent line ion
        for origin, origin_cfg in fit_cfg.get('kinem', {}).items():
            for key, parent_label in origin_cfg.items():

                # Check for the tie entries: kinem.MW.k-0_tie = "C2_1335A_o-MW"
                tie_match = _TIE_KEY.match(key)
                if tie_match is None:
                    continue
                kinem = int(tie_match.group(1))

                # Parent component
                parent = Line.from_transition(parent_label, data_frame=bands, verbose=False)
                parent_key = (parent.classic_notation(just_particle=True), origin_key(parent), parent.kinem)
                if parent_key not in self.components:
                    _logger.warning(f'The parent line {parent_label} for the "kinem.{origin}.{key}" entry is not in '
                                    f'the measured lines. The kinematics will not be tied')
                    continue
                parent_names = self.components[parent_key].names

                # Child components: same origin and kinematic component but different ion
                for comp_key, comp in self.components.items():
                    if (comp_key[1] == origin) and (comp_key[2] == kinem) and (comp_key[0] != parent_key[0]):
                        for param in ('z', 'b'):

                            # Fixed parameters (var_z = false, var_b = false) are not tied
                            if comp.conf[param]['vary']:
                                comp.conf[param]['expr'] = parent_names[param]

        # Warn if kinematic components of the same ion start with the same redshift
        for comp_key, comp in self.components.items():
            if comp_key[2] > 0 and comp.conf['z']['expr'] is None:
                twin = self.components.get((comp_key[0], comp_key[1], 0))
                if (twin is not None) and (twin.conf['z']['value'] == comp.conf['z']['value']):
                    _logger.warning(f'The {comp_key[0]} kinematic component {comp_key[2]} (origin = {comp_key[1]}) has '
                                    f'the same starting redshift as component 0. Please provide a different "z" in the '
                                    f'configuration (kinem.{comp_key[1]}.k-{comp_key[2]}.z)')

        return

    def compile_terms(self, fit_cfg, bands):

        # Starting redshift of the components (from the parent if tied)
        name_values = {comp.names['z']: comp.conf['z']['value'] for comp in self.components.values()}
        z_start = {comp_key: name_values.get(comp.conf['z']['expr'], comp.conf['z']['value'])
                   for comp_key, comp in self.components.items()}

        # Region of each declared band
        band_region = {}
        for idx_region, region in enumerate(self.regions):
            for idx_band in region.bands:
                band_region[idx_band] = idx_region

        # Every transition of an ion is absorbed by all the ion components (VoigtFit)
        for core, trans in self.transitions.items():
            for comp_key, comp in self.components.items():

                if comp_key[0] != trans['ion']:
                    continue

                # Line from the input bands or rebuilt from its label
                declared_entry = self._declared.get((core, comp_key))
                if declared_entry is not None:
                    line, idx_band = declared_entry
                    label, band_line = line.label, self._bands[idx_band]['line']
                else:
                    idx_band = None
                    label = (f'{core}{f"_k-{comp_key[2]}" if comp_key[2] > 0 else ""}'
                             f'{f"_o-{comp_key[1]}" if comp_key[1] != "none" else ""}')
                    line = Line.from_transition(label, fit_cfg, bands, def_shape='abs', def_profile='v', verbose=False)
                    band_line = None

                # Regions: the declared band, those including the line center and all for HI (damping wings, VoigtFit)
                lambda_obs = trans['lambda_0'] * (1 + z_start[comp_key])
                term_regions = []
                for idx_region, region in enumerate(self.regions):
                    if ((trans['ion'] == 'HI') or (region.sub_wave[0] <= lambda_obs <= region.sub_wave[-1])
                            or (band_region.get(idx_band) == idx_region)):
                        term_regions.append(idx_region)

                # Bands for implied transitions from the first region with the line
                if (band_line is None) and (len(term_regions) > 0):
                    band_line = self._bands[self.regions[term_regions[0]].bands[0]]['line']

                # Store the transition component
                idx_term = len(self.terms)
                self.terms.append(IonTerm(label, core, comp_key, line, band_line, declared_entry is not None,
                                          term_regions))
                for idx_region in term_regions:
                    self.regions[idx_region].terms.append(idx_term)

        return

    def compile_params(self):

        # Parameters values first and their links afterwards (as VoigtFit)
        self.params = Parameters()
        for comp in self.components.values():
            for param in ('z', 'b', 'logN'):
                param_conf = comp.conf[param]
                self.params.add(comp.names[param], value=param_conf['value'], vary=param_conf['vary'],
                                min=param_conf['min'], max=param_conf['max'])

        for comp in self.components.values():
            for param in ('z', 'b', 'logN'):
                if comp.conf[param]['expr'] is not None:
                    self.params[comp.names[param]].expr = comp.conf[param]['expr']

        # Check there are enough pixels for the free parameters
        n_pixels = np.sum([region.mask.sum() for region in self.regions])
        n_free = np.sum([param.vary and not param.expr for param in self.params.values()])
        if n_pixels <= n_free:
            raise LiMe_Error(f'The ionic absorptions fitting has {n_free} free parameters but only {n_pixels} pixels')

        return

    def region_model(self, region, values):

        # Optical depth from the transitions in the region
        tau_arr = np.zeros(region.sub_wave.size)
        for idx_term in region.terms:
            term = self.terms[idx_term]
            trans, names = self.transitions[term.core], self.components[term.comp_key].names
            tau_arr += voigt_tau(region.sub_wave, trans['lambda_0'], trans['osc_str'], trans['gamma'],
                                 values[names['z']], values[names['b']], values[names['logN']])

        # Normalized flux convolved with the LSF and integrated in the data pixels (continuum = 1)
        return region.matrix @ np.exp(-tau_arr)

    def residual(self, params):

        # Residuals of all the regions weighted by the pixel uncertainty
        values = params.valuesdict()
        residual_list = [None] * len(self.regions)
        for i, region in enumerate(self.regions):
            model_arr = self.region_model(region, values)
            residual_list[i] = (region.flux[region.mask] - model_arr[region.mask]) / region.err[region.mask]

        return np.concatenate(residual_list)

    def fit(self, method='leastsq', scale_covar=True):

        # Small initial steps for the saturated lines as VoigtFit (factor=1) but with the default tolerance
        min_kws = {'factor': 1.0} if method == 'leastsq' else {}

        # Run the fitting
        with warnings.catch_warnings():
            warnings.simplefilter(_VERBOSE_WARNINGS)
            minimizer = Minimizer(self.residual, self.params, nan_policy='omit', scale_covar=scale_covar)
            self.output = minimizer.minimize(method=method, **min_kws)

        # Best fit model in each region
        values = self.output.params.valuesdict()
        for region in self.regions:
            region.model = self.region_model(region, values)

        # Check for undetected components: their logN runs away and the parameters covariance becomes singular
        undetected = [f'{comp.ion} (origin = {comp.origin}, k-{comp.kinem}): logN = {values[comp.names["logN"]]:0.1f}'
                      for comp in self.components.values()
                      if self.output.params[comp.names['logN']].vary and values[comp.names['logN']] < _LOGN_UNDETECTED]
        if len(undetected) > 0:
            _logger.warning(f'Components not detected (logN < {_LOGN_UNDETECTED}): {"; ".join(undetected)}. Please fix '
                            f'or remove them: they make the uncertainty estimation of the fit fail')

        # Check fitting quality
        if not self.output.errorbars:
            _logger.warning(f'The ionic absorptions fit uncertainty estimation failed')

        return

    def results_to_log(self, log):

        # Fitting diagnostics
        observations = 'no' if self.output.success else 'failed_convergence'
        if not self.output.errorbars:
            observations = 'No_errorbars' if observations == 'no' else f'{observations}_No_errorbars'

        for term in self.terms:

            # Only the input transitions or those with data
            if not term.declared and len(term.regions) == 0:
                continue

            label, line, band_line = term.label, term.line, term.band_line
            names = self.components[term.comp_key].names

            # Identifiers
            log.at[label, 'particle'] = line.particle.label
            log.at[label, 'wavelength'] = line.wavelength
            log.at[label, 'latex_label'] = line.latex_label
            log.at[label, 'group_label'] = 'none' if line.group_label is None else line.group_label
            log.at[label, 'profile'] = 'v'
            log.at[label, 'shape'] = 'abs'

            # Add bands wavelengths
            for i, band_edge in enumerate(('w1', 'w2', 'w3', 'w4', 'w5', 'w6')):
                log.at[label, band_edge] = band_line.mask[i]
            log.at[label, 'pixel_mask'] = band_line.pixel_mask

            # Profile parameters
            z, z_err = param_value_err(self.output.params, names['z'])
            b, b_err = param_value_err(self.output.params, names['b'])
            logN, logN_err = param_value_err(self.output.params, names['logN'])

            log.at[label, 'z_line'], log.at[label, 'z_line_err'] = z, z_err
            log.at[label, 'v_r'] = c_KMpS * (z - self.z_ref) / (1 + self.z_ref)
            log.at[label, 'v_r_err'] = c_KMpS * z_err / (1 + self.z_ref)
            log.at[label, 'b'], log.at[label, 'b_err'] = b, b_err
            log.at[label, 'logN'], log.at[label, 'logN_err'] = logN, logN_err

            # Fitting diagnostics
            log.at[label, 'n_pixels'] = np.sum([self.regions[i].mask.sum() for i in term.regions])
            log.at[label, 'chisqr'], log.at[label, 'redchi'] = self.output.chisqr, self.output.redchi
            log.at[label, 'aic'], log.at[label, 'bic'] = self.output.aic, self.output.bic
            log.at[label, 'observations'] = observations if logN >= _LOGN_UNDETECTED else f'{observations}_undetected'

        return


class IonicFitting:

    """Class to fit ionic absorptions as Voigt profiles from their redshift, Doppler parameter and column density
    (z, b, logN) including the instrument line spread function (VoigtFit forward model)"""

    def __init__(self):

        self.ionic = None

        return

    def ionic_absorptions(self, bands, fit_cfg=None, lsf=None, lsf_nsub=None, norm_obs=False, line_list=None,
                          min_method='leastsq', err_from_bands=None, scale_covar=True, default_cfg_prefix='default',
                          obj_cfg_prefix=None, update_default=True):

        """
        Fit ionic absorption lines as Voigt profiles parametrized by the redshift, Doppler parameter and column
        density (z, b, logN) of each ion component, following the VoigtFit forward model.

        Every transition in the input ``bands`` is absorbed by all the components of its ion. The components are defined
        by the ion (VoigtFit notation, e.g. ``"SiII"``, ``"CIIa"`` for C II*), the line origin (``_o-`` suffix, ``"none"``
        by default) and the kinematic component (``_k-`` suffix, ``0`` by default). The optical depth is computed on a
        sub-pixel grid and the normalized flux, exp(-tau), is convolved with the instrument line spread function (LSF)
        and integrated in the data pixels. All the regions defined by the line bands (w3-w4) are fitted simultaneously.

        Parameters
        ----------
        bands : pandas.DataFrame, str, or pathlib.Path
            Bands table defining the line labels and bands limits (w1 to w6) for each line. The table may include the
            ``osc_str`` and ``trans_prob`` columns with the atomic data otherwise these are read from the LiMe database.
        fit_cfg : dict, str, or pathlib.Path, optional
            Fitting configuration dictionary or a path to a TOML configuration file with the line groups and the
            ``kinem`` entries for the components parameters (see **Notes**).
        lsf : str, pathlib.Path or lime.fitting.ionic.LSF, optional
            Instrument line spread function in the HST/COS table format. If provided, it is stored in the spectrum
            ``lsf`` attribute. Otherwise, the spectrum ``lsf`` is used.
        lsf_nsub : int, optional
            Number of LSF kernel samples per spectrum pixel (VoigtFit ``nsub``). Default is 1.
        norm_obs : bool, optional
            If ``False`` (default), the input spectrum is assumed normalized with the continuum at 1. The continuum
            normalization (``True``) is not available yet.
        line_list : list of str, optional
            Subset of line labels from the bands table to process. If ``None``, all entries in ``bands`` are measured.
        min_method : str, optional
            Minimization algorithm used by :mod:`lmfit`. Default is ``"leastsq"`` (as VoigtFit).
        err_from_bands : bool or None, optional
            If ``True``, estimate the pixel uncertainty from the continuum bands. If ``None``, use the spectrum's
            ``err_flux`` data or fall back to the continuum regions if not available. The default value is None.
        scale_covar : bool, optional
            Scale the parameters uncertainty by the reduced chi-square. Default is ``True``.
        default_cfg_prefix : str, optional
            Section key prefix for the default configuration in ``fit_cfg``. Default is ``"default"``.
        obj_cfg_prefix : str, optional
            Section key prefix for the object-specific configuration in ``fit_cfg``.
        update_default : bool, optional
            If ``True`` (default), merge the ``obj_cfg_prefix`` section into the ``default_cfg_prefix`` one (including
            the nested ``kinem`` entries). If ``False``, the object configuration is used falling back to the default
            if not available.

        Returns
        -------
        None
            The measurements are stored in the spectrum's ``frame`` attribute with one row per transition and component
            (``z_line``, ``z_line_err``, ``v_r``, ``v_r_err``, ``b``, ``b_err``, ``logN``, ``logN_err``). The fitting model
            is available in the ``Spectrum.fit.ionic`` attribute.

        Notes
        -----
        - The components parameters are configured in the ``kinem`` entries with increasing priority:
          origin < kinematic component < ion < ion kinematic component::

              kinem.none.HI.logN = 21              # starting value
              kinem.none.HI.var_z = false          # fixed parameter (var_z, var_b, var_N)
              kinem.MW.k-1.z = 0.0001              # kinematic component 1 of the MW ions
              kinem.MW.SiII.k-1.logN = 13.0        # kinematic component 1 of the MW SiII
              kinem.none.k-0_tie = "C2_1335A"      # tie z and b of the k-0 ions to the C II k-0 component

        - The tied parameters follow the parent line ion component, except those fixed in the configuration
          (``var_z = false``, ``var_b = false``).
        - The starting ``b`` and ``logN`` values are read from the lime.toml ``[voigt_default_params]`` tables unless
          specified in the configuration. The starting ``z`` is the line redshift (``z_line`` or origin) or the spectrum
          redshift.
        - The velocities ``v_r`` are measured with respect to the spectrum redshift.
        - The H I transitions are computed in all regions to include their damping wings (as VoigtFit).

        Examples
        --------
        Fit the ionic absorptions using the COS LSF for a spectrum rebinned by 6 pixels:

        >>> spec.fit.ionic_absorptions(bands_df, fit_cfg="ism_fit.toml", lsf="aa_LSFTable_G130M_1291_LP4_cn.dat",
        ...                            lsf_nsub=6, obj_cfg_prefix="obj1")

        Print the fitting report:

        >>> spec.fit.ionic_report()
        """

        # Local import to avoid a circular dependency with the workflow module
        from lime.workflow import check_compound_line_exclusion

        # Only normalized observations at the current time
        if norm_obs:
            raise NotImplementedError('The continuum normalization is not available for the ionic absorptions yet. '
                                      'Please provide a normalized spectrum (continuum = 1) and use "norm_obs=False"')

        # Load the LSF if provided otherwise use the observation one
        if lsf is not None:
            self._spec.lsf = check_lsf(lsf, lsf_nsub)

        if self._spec.lsf is None:
            raise LiMe_Error('The ionic absorptions fitting requires the instrument LSF. Please provide an HST/COS style '
                             'table via the "lsf" argument or the lime.Spectrum "lsf" attribute')

        # Check if the lines log is a dataframe or a file address
        bands = check_file_dataframe(bands)
        if bands is None:
            raise LiMe_Error('The ionic absorptions fitting requires an input bands dataframe')

        # Crop the analysis to the target lines
        if line_list is not None:
            bands = bands.loc[bands.index.isin(line_list)]

        # Load configuration with a deep update of the nested entries
        input_conf = check_fit_conf(fit_cfg, default_cfg_prefix, obj_cfg_prefix, update_default=update_default,
                                    deep_update=True)

        # User inputs override default behaviour for the pixel error
        err_from_bands = True if (err_from_bands is None) and (self._spec.err_flux is None) else err_from_bands
        if (not err_from_bands) and (self._spec.err_flux is None):
            raise LiMe_Error('The observation does not have an error spectrum. Please set "err_from_bands=True" to '
                             'compute the uncertainty from the continuum bands')

        # Interpret the input lines ignoring those which are part of a group
        line_objs = [Line.from_transition(label, input_conf, bands, def_shape='abs', def_profile='v')
                     for label in bands.index if check_compound_line_exclusion(label, bands)]

        # Compile the model
        self.ionic = IonicModelCompiler(self._spec, line_objs, input_conf, bands, err_from_bands)

        n_free = np.sum([param.vary and not param.expr for param in self.ionic.params.values()])
        _logger.info(f'Ionic absorptions: {len(self.ionic.regions)} regions, {len(self.ionic.transitions)} transitions, '
                     f'{len(self.ionic.components)} components ({n_free} free parameters)')

        # Fit the model
        self.ionic.fit(min_method, scale_covar)

        # Save the measurements
        self.ionic.results_to_log(self._spec.frame)

        return

    def ionic_report(self, return_text=False):

        """Print (or return) the LmFit report of the last ionic absorptions fitting"""

        if (self.ionic is None) or (self.ionic.output is None):
            raise LiMe_Error('No ionic absorptions fitting has been run. Please run "Spectrum.fit.ionic_absorptions" first')

        report = fit_report(self.ionic.output)

        # Print by default, otherwise return the report string
        if return_text:
            return report
        else:
            print(report)

        return
