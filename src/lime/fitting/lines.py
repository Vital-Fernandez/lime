import logging

import numpy as np
from lmfit.models import Model
from lmfit import fit_report
from scipy.interpolate import interp1d
from scipy.special import wofz
from scipy.optimize import curve_fit
from lime.io import LiMe_Error
from lime.tools import compute_FWHM0, mult_err_propagation
import warnings
import re

_logger = logging.getLogger('LiMe')
_VERBOSE_WARNINGS = 'ignore'

c_KMpS = 299792.458  # Speed of light in Km/s

k_GaussArea = np.sqrt(2 * np.pi)
sqrt2 = np.sqrt(2)

k_gFWHM = 2 * np.sqrt(2 * np.log(2))

k_eFWHM = 2 * np.log(2)

TARGET_PERCENTILES = np.array([0, 1, 5, 10, 50, 90, 95, 99, 100])

lime_rng = np.random.default_rng()

# Atomic mass constant
amu = 1.66053906660e-27 # Kg
tiny = 1.0e-15

# Boltzmann constant
k_Boltzmann = 1.380649e-23 # m^2 kg s^-2 K^-1

# Dictionary with atomic masses https://www.ciaaw.org/atomic-weights.htm
ATOMIC_MASS = {'H': (1.00784+1.00811)/2 * amu,
               'He': 4.002602 * amu,
               'C': (12.0096+12.0116)/2 * amu,
               'N': (14.00643 + 14.00728)/2 * amu,
               'O':  (15.99903 + 15.99977)/2 * amu,
               'Ne': 20.1797 * amu,
               'S': (32.059 + 32.076)/2 * amu,
               'Cl': (35.446 + 35.457)/2 * amu,
               'Ar': (39.792 + 39.963)/2 * amu,
               'Fe': 55.845 * amu}


_AMP_PAR = dict(value=None, min=0, max=np.inf, vary=True, expr=None)
_CENTER_PAR = dict(value=None, min=-np.inf, max=np.inf, vary=True, expr=None)
_SIG_PAR = dict(value=None, min=0, max=np.inf, vary=True, expr=None)
_FRAC_PAR = dict(value=None, min=0, max=1, vary=True, expr=None)
_AREA_PAR = dict(value=None, min=-np.inf, max=np.inf, vary=True, expr=None)
_ALPHA_PAR = dict(value=None, min=np.inf, max=0, vary=True, expr=None)

_A_PAR = dict(value=None, min=0, max=np.inf, vary=True, expr=None)
_B_PAR = dict(value=None, min=0, max=np.inf, vary=True, expr=None)
_C_PAR = dict(value=None, min=-np.inf, max=np.inf, vary=True, expr=None)

_AMP_ABS_PAR = dict(value=None, min=-np.inf, max=0, vary=True, expr=None)

_SLOPE_PAR = dict(value=None, min=-np.inf, max=np.inf, vary=False, expr=None)
_INTER_PAR = dict(value=None, min=-np.inf, max=np.inf, vary=False, expr=None)

_SLOPE_FIX_PAR = dict(value=None, min=-np.inf, max=np.inf, vary=False, expr=None)
_INTER_FIX_PAR = dict(value=None, min=-np.inf, max=np.inf, vary=False, expr=None)
_SLOPE_FREE_PAR = dict(value=None, min=-np.inf, max=np.inf, vary=True, expr=None)
_INTER_FREE_PAR = dict(value=None, min=-np.inf, max=np.inf, vary=True, expr=None)


def const_cont_model(cont_array, prefix='cont_', allow_scale=False):
    """
    Build a Model that returns the supplied continuum array.
    If allow_scale=True, include a 'scale' parameter (free by default).
    If allow_scale=False, include 'scale' but freeze it at 1.0 so it's constant.
    """
    cont_array = np.asarray(cont_array)

    def array_component(x, scale=1.0):
        # x is ignored for values, but we check shape for safety
        if np.shape(x) != np.shape(cont_array):
            raise ValueError("continuum array and x must have the same shape")
        return scale * cont_array

    m = Model(array_component, prefix=prefix)
    params = m.make_params(scale=1.0)

    if not allow_scale:
        params[f'{prefix}scale'].set(vary=False)
        # freeze at 1.0 -> truly constant
    return m, params

def linear_model(x, m_cont, n_cont):
    """Linear line formulation"""
    return m_cont * x + n_cont


def gaussian_model(x, amp, center, sigma):
    """1-d gaussian curve : gaussian(x, amp, cen, wid)"""
    return amp * np.exp(-0.5 * (((x-center)/sigma) * ((x-center)/sigma)))


def lorentz_model(x, amp, center, sigma):
    """1-d lorentzian profile : lorentz(x, amp, cen, sigma)"""
    # A * (gamma / 2) ** 2 / ((x - x0) ** 2 + (gamma / 2) ** 2)
    # return (amp / np.pi) * (np.square(sigma) / (np.square(x-center) - np.square(sigma)))
    return amp * ((sigma*sigma) / ((x - center)*(x - center) + sigma*sigma))


def voigt_model(x, amp, center, sigma, gamma):
    # z = ((x-center) + 1j*gamma) / (sigma*sqrt2)
    # return amp * np.real(wofz(z)) / (sigma * k_GaussArea)
    z = (x-center + 1j*gamma) / max(tiny, (sigma*sqrt2))
    return amp*np.real(wofz(z)) / max(tiny, (sigma*k_GaussArea))


def exponential_model(x, amp, center, alpha):

    return amp * np.exp(-alpha * np.abs(x - center))


def pseudo_voigt_model(x, amp, center, sigma, frac):
    return frac * gaussian_model(x, amp, center, sigma) + (1 - frac) * lorentz_model(x, amp, center, sigma)


def broken_powerlaw_model(x, a, b, c, alpha):

    # alpha_in = np.where((x > (center - wbreak)) & (x < (center + wbreak)), 0, alpha)

    # Compute symmetric power law
    wave_shift_abs = np.abs(x - c)
    y = a * np.power(wave_shift_abs, -alpha)

    # Set broken region to zero
    idcs_core = wave_shift_abs < b
    y[idcs_core] = 0.0

    return y


def broken_powerlaw_model_ratio(x, a, b, c, alpha):

    # alpha_in = np.where((x > (center - wbreak)) & (x < (center + wbreak)), 0, alpha)

    # Compute symmetric power law
    wave_shift_abs = np.abs(x - c)
    y = a * np.power(wave_shift_abs, -alpha)

    # Set broken region to zero
    idcs_core = wave_shift_abs < b
    y[idcs_core] = 0.0

    return y


def pseudo_power_model(x, amp, center, sigma, alpha, frac):

    return frac * gaussian_model(x, amp, center, sigma) + (1 - frac) * broken_powerlaw_model(x, amp, sigma, center, alpha)


def g_FWHM(line, idx):

    sigma = line.sigma[idx]

    return k_gFWHM * sigma


def l_FWHM(line, idx):

    sigma = line.sigma[idx]

    return 2 * sigma


def v_FWHM(line, idx):

    # Approximation Kielkopf
    FWHM_gauss = g_FWHM(line, idx)
    FWHM_lorentz = l_FWHM(line, idx)

    return 0.5346 * FWHM_lorentz + np.sqrt(0.2166 * FWHM_lorentz * FWHM_lorentz + FWHM_gauss * FWHM_gauss)


def p_FWHM(line, idx):

    FWHM_power = np.nan

    return FWHM_power


def e_FWHM(line, idx):

    alpha = line.alpha[idx]

    return k_eFWHM / alpha


def pp_FWHM(line, idx):

    FWHM_gauss = g_FWHM(line, idx)

    return FWHM_gauss


def gaussian_area(line, idx, n_steps):

    amp = lime_rng.normal(line.amp[idx], line.amp_err[idx], n_steps)
    sigma = lime_rng.normal(line.sigma[idx], line.sigma_err[idx], n_steps)

    # amp = np.random.normal(line.amp[idx], line.amp_err[idx], n_steps)
    # sigma = np.random.normal(line.sigma[idx], line.sigma_err[idx], n_steps)

    # area = 2.5066282746 * line.amp[idx] * line.amp[idx]
    # area_err = area * np.sqrt(np.square(line.amp_err[idx] / line.amp[idx]) +
    #                           np.square(line.sigma_err[idx] / line.sigma[idx]))

    return 2.5066282746 * amp * sigma


def lorentz_area(line, idx, n_steps):

    amp = np.random.normal(line.amp[idx], line.amp_err[idx], n_steps)
    sigma = np.random.normal(line.sigma[idx], line.sigma_err[idx], n_steps)

    return 3.14159265 * amp * sigma


def voigt_area(line, idx, n_steps):

    amp = np.random.normal(line.amp[idx], line.amp_err[idx], n_steps)
    sigma = np.random.normal(line.sigma[idx], line.sigma_err[idx], n_steps)
    gamma = np.random.normal(line.gamma[idx], line.gamma_err[idx], n_steps)

    return gaussian_area(amp, sigma) + lorentz_area(amp, gamma)


def pseudo_voigt_area(line, idx, n_steps):

    amp = np.random.normal(line.amp[idx], line.amp_err[idx], n_steps)
    sigma = np.random.normal(line.sigma[idx], line.sigma_err[idx], n_steps)
    frac = np.random.normal(line.frac[idx], line.frac_err[idx], n_steps)

    return frac * (2.5066282746 * amp * sigma) + (1 - frac) * (3.14159265 * amp * sigma)


def power_area(line, idx, n_steps):

    return np.full(n_steps, np.nan)


def exp_area(line, idx, n_steps):

    amp = np.random.normal(line.amp[idx], line.amp_err[idx], n_steps)
    alpha = np.random.normal(line.alpha[idx], line.alpha[idx], n_steps)

    return 2.5066282746 * amp * 1/alpha


def pseudo_power_area(line, idx, n_steps):

    amp = np.random.normal(line.amp[idx], line.amp_err[idx], n_steps)
    sigma = np.random.normal(line.sigma[idx], line.sigma_err[idx], n_steps)
    frac = np.random.normal(line.frac[idx], line.frac_err[idx], n_steps)

    return frac * (2.5066282746 * amp * sigma) + (1 - frac) * (3.14159265 * amp * sigma)


def velocity_to_wavelength_band(n_sigma, band_velocity_sigma, lambda_obs, delta_instr):

    return n_sigma * ((band_velocity_sigma / c_KMpS) * lambda_obs + delta_instr)


ALL_PARAMS = np.array(['m_cont', 'n_cont', 'amp', 'center', 'sigma', 'gamma', 'alpha', 'frac', 'a', 'b', 'c'])

PROFILE_PARAMS = dict(g = ['amp', 'center', 'sigma'],
                      l = ['amp', 'center', 'sigma'],
                      v = ['amp', 'center', 'sigma', 'gamma'],
                      pv = ['amp', 'center', 'sigma', 'frac'],
                      pp = ['amp', 'center', 'sigma', 'alpha', 'frac'],
                      p = ['a', 'b', 'c', 'alpha'],
                      e = ['amp', 'center', 'alpha'])

PROFILE_ABBREV = dict(g = 'Gaussian',
                      l = 'Lorentzian',
                      v = 'Voigt',
                      pv = 'Pseudo-Voigt',
                      pp = 'Pseudo-Power law',
                      p = 'Broken Power law',
                      e = 'Exponential')

PROFILE_FUNCTIONS =  {'g': gaussian_model, 'l': lorentz_model, 'v': voigt_model,
                      'pv': pseudo_voigt_model, 'pp': pseudo_power_model,
                      'p': broken_powerlaw_model, 'e': exponential_model}

AREA_FUNCTIONS = {'g': gaussian_area, 'l': lorentz_area, 'v': voigt_area,
                  'pv': pseudo_voigt_area, 'pp': pseudo_power_area,
                  'p': power_area, 'e': exp_area}

FWHM_FUNCTIONS =  {'g': g_FWHM, 'l': l_FWHM, 'v': v_FWHM, 'pv': v_FWHM, 'pp': pp_FWHM, 'p': p_FWHM, 'e': e_FWHM}


def show_profile_parameters(profile_params=PROFILE_PARAMS, profile_abbrev=PROFILE_ABBREV):

    """
    Display the available emission line profile models and their parameters.

    Parameters
    ----------
    profile_params : dict, optional
        Dictionary mapping profile identifier characters (e.g., ``"g"``, ``"l"``)
        to lists of parameter names (e.g., ``["amplitude", "center", "sigma"]``).
        Default is :data:`PROFILE_PARAMS`.
    profile_abbrev : dict, optional
        Dictionary mapping profile identifier characters to their descriptive names
        (e.g., ``{"g": "Gaussian", "l": "Lorentzian"}``).
        Default is :data:`PROFILE_ABBREV`.

    Examples
    --------
    Display all registered profiles:

    >>> show_profile_parameters()

    Example output:

    .. code-block:: text

       Available profiles (with their identifying character) and their parameters:
       - Gaussian "g": ['amp', 'center', 'sigma']
       - Lorentzian "l": ['amp', 'center', 'sigma']
       - Voigt "v": ['amp', 'center', 'sigma', 'gamma']
    """

    print("\nAvailable profiles (with their identifying character) and their parameters:")
    for id_character, param_list in profile_params.items():
        profile = profile_abbrev[id_character]
        print(f'- {profile} "{id_character}": {param_list}')

    return


def signal_to_noise_rola(amp, std_cont, n_pixels):

    snr = (k_GaussArea/6) * (amp/std_cont) * np.sqrt(n_pixels)

    return snr


def profiles_computation(line_list, log, z_corr, shape_list, x_array=None, interval=('w3', 'w4'), res_factor=100):

    # All lines are computed with the same wavelength interval: The maximum interval[1]-interval[0] in the log times 3
    # and starting at interval[0] values beyond interval[0] are masked

    if x_array is None:

        # Create x array
        wmin_array = log.loc[line_list, interval[0]].to_numpy() * z_corr
        wmax_array = log.loc[line_list, interval[1]].to_numpy() * z_corr
        w_mean = np.max(wmax_array - wmin_array)

        x_zero = np.linspace(0, w_mean, res_factor)
        x_array = np.add(np.c_[x_zero], wmin_array)

        # Get the y array depending of the function
        gaussian_array = np.zeros((res_factor, len(line_list)))

        # Compile the flux profile for the corresponding shape
        for i, line_label in enumerate(line_list):
            profile_function = PROFILE_FUNCTIONS[shape_list[i]]

            if shape_list[i] == "g" or shape_list[i] == "l":
                amp_array = log.loc[line_list, 'amp'].to_numpy()
                center_array = log.loc[line_list, 'center'].to_numpy()
                sigma_array = log.loc[line_list, 'sigma'].to_numpy()
                gaussian_array[:, i] = profile_function(x_array, amp_array[i], center_array[i], sigma_array[i])[:, 0]

            elif shape_list[i] == "v":
                amp_array = log.loc[line_list, 'amp'].to_numpy()
                center_array = log.loc[line_list, 'center'].to_numpy()
                sigma_array = log.loc[line_list, 'sigma'].to_numpy()
                gamma_array = log.loc[line_list, 'gamma'].to_numpy()
                gaussian_array[:, i] = profile_function(x_array, amp_array[i], center_array[i], sigma_array[i], gamma_array[i])[:, 0]

            elif shape_list[i] == "pv":
                amp_array = log.loc[line_list, 'amp'].to_numpy()
                center_array = log.loc[line_list, 'center'].to_numpy()
                sigma_array = log.loc[line_list, 'sigma'].to_numpy()
                frac_array = log.loc[line_list, 'frac'].to_numpy()
                gaussian_array[:, i] = profile_function(x_array, amp_array[i], center_array[i], sigma_array[i], frac_array[i])[:, 0]

            elif shape_list[i] == "e":
                amp_array = log.loc[line_list, 'amp'].to_numpy()
                center_array = log.loc[line_list, 'center'].to_numpy()
                alpha_array = log.loc[line_list, 'alpha'].to_numpy()
                gaussian_array[:, i] = profile_function(x_array, amp_array[i], center_array[i], alpha_array[i])[:, 0]

            elif shape_list[i] == "pp":
                amp_array = log.loc[line_list, 'amp'].to_numpy()
                center_array = log.loc[line_list, 'center'].to_numpy()
                sigma_array = log.loc[line_list, 'sigma'].to_numpy()
                frac_array = log.loc[line_list, 'frac'].to_numpy()
                alpha_array = log.loc[line_list, 'alpha'].to_numpy()
                gaussian_array[:, i] = profile_function(x_array, amp_array[i], center_array[i], sigma_array[i],
                                                        alpha_array[i], frac_array[i])[:, 0]

            elif shape_list[i] == "p":
                a_array = log.loc[line_list, 'a'].to_numpy()
                b_array = log.loc[line_list, 'b'].to_numpy()
                c_array = log.loc[line_list, 'c'].to_numpy()
                alpha_array = log.loc[line_list, 'alpha'].to_numpy()
                gaussian_array[:, i] = profile_function(x_array, a_array[i], b_array[i], c_array[i], alpha_array[i])[:, 0]

            else:
                raise LiMe_Error(f'Profile curve "{shape_list[i]}" for line {line_label} is not recognized. Please use '
                                 f'_p-g (gaussian), _p-l (Lorentz) or _p-v (Voigt)')

        # Esto que hace
        for i in range(x_array.shape[1]):
            idcs_nan = x_array[:, i] > wmax_array[i]
            x_array[idcs_nan, i] = np.nan
            gaussian_array[idcs_nan, i] = np.nan

        return x_array, gaussian_array

    # All lines are computed with the wavelength range provided by the user
    else:

        # Profile container
        gaussian_array = np.zeros((len(x_array), len(line_list)))

        # Compute the individual profiles
        for i, comp in enumerate(line_list):
            profile_function = PROFILE_FUNCTIONS[shape_list[i]]
            if shape_list[i] in ['g', 'l']:
                amp = log.loc[comp.label, 'amp']
                center = log.loc[comp.label, 'center']
                sigma = log.loc[comp.label, 'sigma']
                gaussian_array[:, i] = profile_function(x_array, amp, center, sigma)

        return gaussian_array


def linear_continuum_computation(line_list, log, z_corr, res_factor=100, interval=('w3', 'w4'), x_array=None):

    # All lines are computed with the same wavelength interval: The maximum interval[1]-interval[0] in the log times 3
    # and starting at interval[0] values beyond interval[0] are masked
    if x_array is None:

        idcs_lines = (log.index.isin(line_list))

        m_array = log.loc[idcs_lines, 'm_cont'].values
        n_array = log.loc[idcs_lines, 'n_cont'].values
        wmin_array = log.loc[idcs_lines, interval[0]].values * z_corr
        wmax_array = log.loc[idcs_lines, interval[1]].values * z_corr
        w_mean = np.max(wmax_array - wmin_array)

        x_zero = np.linspace(0, w_mean, res_factor)
        x_array = np.add(np.c_[x_zero], wmin_array)

        cont_array = m_array * x_array + n_array

        for i in range(x_array.shape[1]):
            idcs_nan = x_array[:, i] > wmax_array[i]
            x_array[idcs_nan, i] = np.nan
            cont_array[idcs_nan, i] = np.nan

        return x_array, cont_array

    # All lines are computed with the wavelength range provided by the user
    else:

        cont_array = np.zeros((len(x_array), len(line_list)))

        for i, comp in enumerate(line_list):
            m_cont = log.loc[comp.label, 'm_cont']
            n_cont = log.loc[comp.label, 'n_cont']
            cont_array[:, i] = m_cont * x_array + n_cont

        return cont_array


def is_digit(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


def sigma_corrections(line, idcs_line, wave_arr, R_arr, temperature):

    # Thermal correction
    line.measurements.sigma_thermal = np.full(line.measurements.sigma.size, np.nan)
    for i in range(line.measurements.sigma_thermal.size):
        atom_mass = ATOMIC_MASS.get(line.list_comps[i].particle.symbol, np.nan)
        line.measurements.sigma_thermal[i] = np.sqrt(k_Boltzmann * temperature / atom_mass) / 1000

    # Instrumental correction
    if R_arr is not None:
        if np.isscalar(R_arr):
            line.measurements.sigma_instr = np.mean(wave_arr.compressed() / (R_arr * k_gFWHM))
        else:
            mask_data = ~wave_arr.mask
            line.measurements.sigma_instr = np.mean(wave_arr[mask_data] / (R_arr[idcs_line][mask_data] * k_gFWHM))
            wave_arr[mask_data] / (R_arr[idcs_line][mask_data] * k_gFWHM)
    else:
        line.measurements.sigma_instr = None

    return

# TODO esto tirarlo?
def compute_inst_sigma_array(wave_arr, res_power=None):

    # Aproximation using the wavelength array
    if res_power is None:
        deltalamb_arr = np.diff(wave_arr)
        R_arr = wave_arr[1:] / deltalamb_arr
        FWHM_arr = wave_arr[1:] / R_arr

        sigma_arr = np.zeros(wave_arr.size)
        sigma_arr[:-1] = FWHM_arr / k_gFWHM
        sigma_arr[-1] = sigma_arr[-2]

    # Use the true R_arr
    else:
        if wave_arr.size != res_power.size:
            raise LiMe_Error(f'The size fo the spectrum array and the resolving power array must have the same size')
        FWHM_arr = wave_arr / res_power
        sigma_arr = FWHM_arr / k_gFWHM

    return sigma_arr


class ProfileModelCompiler:

    def __init__(self, line, redshift, user_conf, y, cont_arr=None, narrow_check=False):

        self.model = None
        self.params = None
        self.output = None
        self.n_comps = None
        self.ref_wave = None

        # Define Reference attributes according to single profile or multiple
        if not line.group == 'b':
            self.lcomps = [line]
            self.n_comps = 1
            self.ref_wave = np.array([line.wavelength]) * (1 + redshift)
        else:
            self.lcomps = line.list_comps
            self.n_comps = len(line.list_comps)
            self.ref_wave = line.param_arr('wavelength') * (1 + redshift)

        # Continuum model
        if cont_arr is None:
            self.model = Model(linear_model)
            self.model.prefix = 'line0_'

            # Fix or not the continuum
            self.define_param(0, line, 'm_cont', line.measurements.m_cont, _SLOPE_FIX_PAR, user_conf)
            self.define_param(0, line, 'n_cont', line.measurements.n_cont, _INTER_FIX_PAR, user_conf)

        else:
            self.model = const_cont_model(cont_arr, 'line0_')[0]

        # Add one gaussian models per component
        for idx, comp in enumerate(self.lcomps):

            # Gaussian comp
            self.model += Model(PROFILE_FUNCTIONS[line.profile], prefix=f'line{idx}_')

            # Amplitude configuration
            if comp.shape == 'emi':

                # Emission min, init, max values
                min_lim = 0
                peak_0 = line.measurements.peak_flux - line.measurements.cont
                if narrow_check:
                    max_lim = line.measurements.peak_flux - line.measurements.cont + line.measurements.cont_err
                else:
                    max_lim = line.measurements.peak_flux * 1.5

            else:
                # Absorption
                trough = np.min(y)# This is necessary in case there are maximum and minimum lines... we need multiple peak_wave, peak_flux options
                min_lim = trough - line.measurements.cont
                max_lim = 0
                peak_0 = line.measurements.peak_flux * 0.5 - line.measurements.cont

            # Add the parameters according to the profile
            match comp.profile:

                # Gaussian, lorentz and Voigt
                case 'g':
                    AMP_PAR = dict(value=None, min=min_lim, max=max_lim, vary=True, expr=None)
                    self.define_param(idx, line, 'amp', peak_0, AMP_PAR, user_conf)
                    self.define_param(idx, line, 'center', self.ref_wave[idx], _CENTER_PAR, user_conf, redshift)
                    self.define_param(idx, line, 'sigma', 2*line.measurements.pixelWidth, _SIG_PAR, user_conf)

                # Lorentz
                case 'l':
                    AMP_PAR = dict(value=None, min=min_lim, max=max_lim, vary=True, expr=None)
                    self.define_param(idx, line, 'amp', peak_0, AMP_PAR, user_conf)
                    self.define_param(idx, line, 'center', self.ref_wave[idx], _CENTER_PAR, user_conf, redshift)
                    self.define_param(idx, line, 'sigma', 2 * line.measurements.pixelWidth, _SIG_PAR, user_conf)

                # Gamma param for Voigt
                case 'v':
                    AMP_PAR = dict(value=None, min=min_lim, max=max_lim, vary=True, expr=None)
                    self.define_param(idx, line, 'amp', peak_0, AMP_PAR, user_conf)
                    self.define_param(idx, line, 'center', self.ref_wave[idx], _CENTER_PAR, user_conf, redshift)
                    self.define_param(idx, line, 'sigma', 2*line.measurements.pixelWidth, _SIG_PAR, user_conf)
                    self.define_param(idx, line, 'gamma', 2*line.measurements.pixelWidth, _SIG_PAR, user_conf)

                # Frac for Pseudo-Voigt
                case 'pv':
                    AMP_PAR = dict(value=None, min=min_lim, max=max_lim, vary=True, expr=None)
                    self.define_param(idx, line, 'amp', peak_0, AMP_PAR, user_conf)
                    self.define_param(idx, line, 'center', self.ref_wave[idx], _CENTER_PAR, user_conf, redshift)
                    self.define_param(idx, line, 'sigma', 2*line.measurements.pixelWidth, _SIG_PAR, user_conf)
                    self.define_param(idx, line, 'frac', 0.5, _FRAC_PAR, user_conf)

                # Exponential profile
                case 'e':
                    AMP_PAR = dict(value=None, min=min_lim, max=max_lim, vary=True, expr=None)
                    _CENTER_PAR_e = dict(value= self.ref_wave[idx], min=line.mask[0]*(1+redshift),
                                         max=line.mask[5]*(1+redshift),
                                         vary=True, expr=None)
                    self.define_param(idx, line, 'amp', peak_0, AMP_PAR, user_conf)
                    self.define_param(idx, line, 'center', self.ref_wave[idx], _CENTER_PAR_e, user_conf, redshift)
                    self.define_param(idx, line, 'alpha', 1.0/line.measurements.pixelWidth, _ALPHA_PAR, user_conf)

                # Frac for Pseudo-Voigt
                case 'pp':
                    AMP_PAR = dict(value=None, min=min_lim, max=max_lim, vary=True, expr=None)
                    self.define_param(idx, line, 'amp', peak_0, AMP_PAR, user_conf)
                    self.define_param(idx, line, 'center', self.ref_wave[idx], _CENTER_PAR, user_conf, redshift)
                    self.define_param(idx, line, 'sigma', 2*line.measurements.pixelWidth, _SIG_PAR, user_conf)
                    self.define_param(idx, line, 'frac', 0.5, _FRAC_PAR, user_conf)
                    self.define_param(idx, line, 'alpha', 2, _ALPHA_PAR, user_conf)

                # Power law
                case 'p':
                    A_PAR = dict(value=None, min=min_lim, max=max_lim, vary=True, expr=None)
                    self.define_param(idx, line, 'a', peak_0, A_PAR, user_conf)
                    self.define_param(idx, line, 'b', self.ref_wave[idx], _B_PAR, user_conf, redshift)
                    self.define_param(idx, line, 'c', 2 * line.measurements.pixelWidth, _C_PAR, user_conf)
                    self.define_param(idx, line, 'alpha', -2, _ALPHA_PAR, user_conf)

        return

    def fit(self, line, x, y, err, method):

        # Unpack the mask for LmFit analysis
        idcs_good = ~x.mask
        x_in = x.data[idcs_good]
        y_in = y.data[idcs_good]

        # Compute weights
        if err is None:
            weights_in = np.full(x[idcs_good].size, 1.0 / line.measurements.cont_err)
        else:
            weights_in = 1.0/err[idcs_good].data

        # Compute model params displaying an error message if failed
        try:
            self.params = self.model.make_params()
        except Exception as e:
            msg = "Error compiling the line parameters below. Please check the input fitting configuration.\n"
            for j, (name, conf) in enumerate(self.model.param_hints.items(), start=1):
                name_comps = name.split('_')
                msg +=  f"{line.list_comps[int(name_comps[0][-1])].label}_{'_'.join(name_comps[1:])}: {conf}\n"
            raise LiMe_Error(msg)

        # Run the fitting
        with warnings.catch_warnings():
            warnings.simplefilter(_VERBOSE_WARNINGS)
            self.output = self.model.fit(y_in, self.params, x=x_in, weights=weights_in, method=method, nan_policy='omit')

        # Check fitting quality
        self.review_fitting(line)

        return

    def measurements_calc(self, line, user_conf, redshift):

        # Assign continuum values (If not already available from the fit)
        if getattr(line.measurements, 'm_cont') is not None:
            for j, param in enumerate(['m_cont', 'n_cont']):
                param_fit = self.output.params.get(f'line0_{param}', None)
                if param_fit is not None:
                    param_value = np.nan if param_fit is None else param_fit.value
                    param_err = np.nan if param_fit is None else param_fit.stderr
                    setattr(line.measurements, param, param_value)
                    setattr(line.measurements, f'{param}_err', param_err)

        # Loop through the line components and assign profile params
        for i, comp_label in enumerate(self.lcomps):
            for j, param in enumerate(PROFILE_PARAMS[comp_label.profile]):

                # Initialize the parameters
                if i == 0:
                    setattr(line.measurements, param, np.full(self.n_comps, np.nan))
                    setattr(line.measurements, f'{param}_err', np.full(self.n_comps, np.nan))

                # Recover possible component from fit
                param_fit = self.output.params.get(f'line{i}_{param}', None)
                param_value = np.nan if param_fit is None else param_fit.value
                param_err = np.nan if param_fit is None else param_fit.stderr

                # Assign array curve parameters
                getattr(line.measurements, param)[i] = param_value
                getattr(line.measurements, f'{param}_err')[i] = param_err

            # Initialize array parameters
            if i == 0:
                line.measurements.profile_flux = np.full(self.n_comps, np.nan)
                line.measurements.profile_flux_err = np.full(self.n_comps, np.nan)
                line.measurements.eqw = np.full(self.n_comps, np.nan)
                line.measurements.eqw_err = np.full(self.n_comps, np.nan)
                line.measurements.FWHM_p = np.full(self.n_comps, np.nan)

            # Check for negative -0.0 # TODO this needs a better place # FIXME -0.0 error
            if np.signbit(line.measurements.sigma_err[i]):
                line.measurements.sigma_err[i] = np.nan
                _logger.warning(f'Negative scale value for amplitude at {comp_label}')

            if np.signbit(line.measurements.amp_err[i]):
                line.measurements.amp_err[i] = np.nan
                _logger.warning(f'Negative scale value for amplitude at {comp_label}')

            profile_flux_dist = AREA_FUNCTIONS[comp_label.profile](line.measurements, i, 1000)
            line.measurements.profile_flux[i] = np.mean(profile_flux_dist)
            line.measurements.profile_flux_err[i] = np.std(profile_flux_dist)

            # Compute profile flux and uncertainty
            # measurements.profile_flux[i], measurements.profile_flux_err[i] = AREA_FUNCTIONS[comp_label.profile](measurements, i, 1000)

            # Compute FWHM_p (Profile Full Width Half Maximum)
            line.measurements.FWHM_p[i] = FWHM_FUNCTIONS[comp_label.profile](line.measurements, i)

            # Check parameters error propagation
            self.review_err_propagation(line.measurements, i, comp_label.label, user_conf, self.output.errorbars,
                                        line.group)

        # Compute the equivalent widths
        line.measurements.eqw = line.measurements.profile_flux / line.measurements.cont
        line.measurements.eqw_err = np.abs(line.measurements.eqw) * np.sqrt(np.square(line.measurements.profile_flux_err/line.measurements.profile_flux) +
                                                  np.square(line.measurements.cont_err/line.measurements.cont))

        # Centroid redshifts
        line.measurements.z_line = line.measurements.center/(self.ref_wave/(1 + redshift)) - 1

        # Kinematics
        line.measurements.v_r = c_KMpS * (line.measurements.center - self.ref_wave) / self.ref_wave
        line.measurements.v_r_err = c_KMpS * line.measurements.center_err / self.ref_wave
        line.measurements.sigma_vel = c_KMpS * line.measurements.sigma / self.ref_wave
        line.measurements.sigma_vel_err = c_KMpS * line.measurements.sigma_err / self.ref_wave

        # Fitting diagnostics
        line.measurements.chisqr, line.measurements.redchi = self.output.chisqr, self.output.redchi
        line.measurements.aic, line.measurements.bic = self.output.aic, self.output.bic

        # Updated signal-to-noise based on a successful profile fitting
        if self.output.errorbars:
            line.measurements.snr_line = signal_to_noise_rola(line.measurements.amp,
                                                              line.measurements.cont_err,
                                                              line.measurements.n_pixels)

        return

    def define_param(self, idx, line, param_label, param_value, default_conf={}, user_conf={}, z_obj=0):

        comps = line.list_comps

        # Line name i.e. H1_6563A_w1, H1_6563A_w1_amp
        line_label = comps[idx]

        # LmFit reference line0, line0_amp
        user_ref = f'{line_label}_{param_label}'
        param_ref = f'line{idx}_{param_label}'

        # TODO if min max provided the value should be in the middle
        # Overwrite default by the one provided by the user
        if user_ref in user_conf:
            param_conf = {**default_conf, **user_conf[user_ref]}
        else:
            param_conf = default_conf.copy()

        # Convert from LiMe -> LmFit label configuration in the expr entries if necessary
        if param_conf['expr'] is not None:
            # Do not parse expressions for single components
            if line.group == 'b':
                expr = param_conf['expr']
                for i, comp in enumerate(comps):
                    for g_param in ALL_PARAMS:
                        expr = expr.replace(f'{comp}_{g_param}', f'line{i}_{g_param}')
                param_conf['expr'] = expr
            else:
                param_conf['expr'] = None

        # Set initial value estimation from spectrum if not provided by the user
        if user_ref not in user_conf:
            param_conf['value'] = param_value

        else:
            # Special case inequalities: H1_6563A_w1_sigma = '>1.5*H1_6563A_sigma'
            if param_conf['expr'] is not None:
                if ('<' in param_conf['expr']) or ('>' in param_conf['expr']):

                    # Create additional parameter
                    ineq_name = f'{param_ref}_ineq'
                    ineq_operation = '*'  # TODO add remaining operations

                    # Split number and ref
                    ineq_expr = param_conf['expr'].replace('<', '').replace('>', '')
                    ineq_items = ineq_expr.split(ineq_operation)
                    ineq_linkedParam = ineq_items[0] if not is_digit(ineq_items[0]) else ineq_items[1]
                    ineq_lim = float(ineq_items[0]) if is_digit(ineq_items[0]) else float(ineq_items[1])

                    # Stablish the inequality configuration:
                    ineq_conf = {}  # TODO need to check these limits
                    if '>' in param_conf['expr']:
                        ineq_conf['value'] = ineq_lim * 1.2
                        ineq_conf['min'] = ineq_lim
                    else:
                        ineq_conf['value'] = ineq_lim * 0.8
                        ineq_conf['max'] = ineq_lim

                    # Define new param
                    self.model.set_param_hint(name=ineq_name, **ineq_conf)

                    # Prepare definition of param:
                    new_expresion = f'{ineq_name}{ineq_operation}{ineq_linkedParam}'
                    # param_conf = dict(expr=new_expresion)
                    param_conf = {'value': ineq_conf['value'], 'expr': new_expresion}

            # Case default value is not provided
            else:
                if param_conf['value'] is None:
                    param_conf['value'] = param_value

        # Additional preparation for center parameter: Multiply value, min, max by redshift
        if '_center' in param_ref:
            if user_ref in user_conf:
                param_user_conf = user_conf[user_ref]
                for param_conf_entry in ('value', 'min', 'max'):
                    if param_conf_entry in param_user_conf:
                        param_conf[param_conf_entry] = param_conf[param_conf_entry] * (1 + z_obj)

        # In case of parameter is defined by an expresion, but it has no value
        if (param_conf['value'] is None) and (param_value is not None):
            param_conf['value'] = param_value

        # Assign the parameter configuration to the models
        self.model.set_param_hint(param_ref, **param_conf)

        return

    def review_fitting(self, line):

        if not self.output.errorbars:
            if line.measurements.observations == 'no':
                line.measurements.observations = 'No_errorbars'
            else:
                line.measurements.observations += 'No_errorbars'
            _logger.warning(f'Gaussian fit uncertainty estimation failed for {line.label}')

        return

    def review_err_propagation(self, data, idx_line, comp, user_conf, error_check, line_group):

        # Check gaussian flux error
        if (data.profile_flux_err[idx_line] == 0.0) and (data.amp_err[idx_line] != 0.0) and (data.sigma_err[idx_line] != 0.0):
            data.profile_flux_err[idx_line] = mult_err_propagation(np.array([data.amp[idx_line], data.sigma[idx_line]]),
                                                                   np.array([data.amp_err[idx_line],
                                                                             data.sigma_err[idx_line]]),
                                                                             data.profile_flux[idx_line])

        # Check equivalent width error
        if line_group == 'b':
            if (data.eqw_err[idx_line] == 0.0) and (data.cont_err != 0.0) and (data.profile_flux_err[idx_line] != 0.0):
                data.eqw_err[idx_line] = mult_err_propagation(np.array([data.cont, data.profile_flux[idx_line]]),
                                                              np.array([data.cont_err, data.profile_flux_err[idx_line]]),
                                                              data.eqw[idx_line])

        # Continuum error
        if (data.m_cont_err == 0.0) and (data.m_cont != 0.0) and (data.m_cont_err_intg != 0.0):
            data.m_cont_err = data.m_cont_err_intg

        if (data.n_cont_err == 0.0) and (data.n_cont != 0.0) and (data.n_cont_err_intg != 0.0):
            data.n_cont_err = data.n_cont_err_intg

        # Velocity and sigma error from line kinematics imported from an external line
        if data.center_err[idx_line] == 0:
            if user_conf.get(f'{comp}_center_err') is not None:
                data.center_err[idx_line] = user_conf.get(f'{comp}_center_err', np.nan)
                data.sigma_err[idx_line] = user_conf.get(f'{comp}_sigma_err', np.nan)

        return


class LineFitting:

    """Class to measure emission line fluxes and fit them as gaussian curves"""

    def __init__(self):

        self.line = None
        self.profile = None
        self._narrow_check = False

        return

    def integrated_properties(self, line, emis_wave, emis_flux, emis_err, cont_arr, n_steps=1000, min_array_dim=15):

        # Use default line shape
        match line.shape:
            case 'emi':
                emission_check = True
                peakIdx = np.argmax(emis_flux) if emission_check else np.argmin(emis_flux)

            case 'abs':
                emission_check = False
                peakIdx = np.argmin(emis_flux)

            case _:
                if emis_flux[emis_flux.size // 2] > line.measurements.cont:
                    peakIdx = np.argmax(emis_flux)
                    line.p_shape = True
                else:
                    peakIdx = np.argmin(emis_flux)
                    line.p_shape = False

        # Assign values peak/through properties
        line.measurements.n_pixels = emis_wave.compressed().size
        line.measurements.peak_wave = emis_wave[peakIdx]
        line.measurements.peak_flux = emis_flux[peakIdx]
        line.measurements.pixelWidth = np.diff(emis_wave).mean()
        # line.measurements.cont = line.measurements.peak_wave * line.measurements.m_cont + line.measurements.n_cont
        line.measurements.cont = cont_arr[peakIdx]
        line.measurements.cont_err = emis_err[peakIdx]

        # Warning if continuum above or below line peak/through
        if emission_check and (cont_arr[peakIdx] > emis_flux[peakIdx]):
            _logger.info(f'Line {line.label} introduced as an emission but the line peak is below the continuum level')

        if emission_check and (cont_arr[peakIdx] > emis_flux[peakIdx]):
            _logger.info(f'Line {line.label} introduced as an absorption but the line peak is below the continuum level')

        # Monte Carlo to measure line flux and uncertainty
        normalNoise = np.random.normal(0.0, emis_err, (n_steps, emis_flux.size))
        lineFluxMatrix = emis_flux + normalNoise
        areasArray = (lineFluxMatrix.sum(axis=1) - cont_arr.sum()) * line.measurements.pixelWidth

        # Assign integrated fluxes and uncertainty
        line.measurements.intg_flux = areasArray.mean()
        line.measurements.intg_flux_err = areasArray.std()

        # Compute SN_r
        line.measurements.snr_line = signal_to_noise_rola(line.measurements.peak_flux - line.measurements.cont,
                                                          line.measurements.cont_err, line.measurements.n_pixels)
        line.measurements.snr_cont = line.measurements.cont/line.measurements.cont_err

        # Logic for very small lines
        snr_array = signal_to_noise_rola(emis_flux - line.measurements.cont, line.measurements.cont_err, 1)
        if (np.sum(snr_array > 5) < 3) and (line.group != 'b'):
            self._narrow_check = True
        else:
            self._narrow_check = False

        # Velocity calculations
        if (line.measurements.n_pixels >= min_array_dim) and (self._narrow_check is False):
            self.velocity_profile_calc(line.measurements, peakIdx, emis_wave, emis_flux, cont_arr, emission_check,
                                       min_array_dim=min_array_dim)

        # Equivalent width computation (it must be a 1d array to avoid conflict in blended lines)
        lineContinuumMatrix = cont_arr + normalNoise
        eqwMatrix = areasArray / lineContinuumMatrix.mean(axis=1)

        line.measurements.eqw_intg = np.array(eqwMatrix.mean(), ndmin=1)
        line.measurements.eqw_intg_err = np.array(eqwMatrix.std(), ndmin=1)

        return

    def profile_fitting(self, line, x_arr, y_arr, err_arr, cont_arr, user_conf, fit_method='leastsq'):

        # Compile the Lmfit component models
        self.profile = ProfileModelCompiler(line, self._spec.redshift, user_conf, y_arr, cont_arr, self._narrow_check)

        # Fit the models
        self.profile.fit(line, x_arr, y_arr, err_arr, fit_method)

        # Store the results into the line attributes
        self.profile.measurements_calc(line, user_conf, self._spec.redshift)

        return

    def continuum_calculation(self, idcs_emis, idcs_cont, user_cont_source, err_from_bands):

        # Use the continuum bands for the calculation
        match user_cont_source:

            case 'adjacent':

                # Check for zero err
                err_cont = self._spec.err_flux[idcs_cont].compressed() if self._spec.err_flux is not None else None
                err_cont = err_cont if np.any(err_cont) else None

                # Fit the model, including uncertainties
                params, covariance = curve_fit(linear_model,
                                               xdata=self._spec.wave[idcs_cont].compressed(),
                                               ydata=self._spec.flux[idcs_cont].compressed(),
                                               sigma=err_cont,
                                               absolute_sigma=True, check_finite=False)

                self.line.measurements.m_cont, self.line.measurements.n_cont = params
                self.line.measurements.m_cont_err_intg, self.line.measurements.n_cont_err_intg = np.sqrt(np.diag(covariance))
                cont_arr = self._spec.wave * self.line.measurements.m_cont + self.line.measurements.n_cont

            case 'central':

                x, y = self._spec.wave[idcs_emis].compressed(), self._spec.flux[idcs_emis].compressed()
                err = self._spec.err_flux[idcs_emis].compressed() if self._spec.err_flux is not None else [0, 0]

                self.line.measurements.m_cont = (y[-1] - y[0]) / (x[-1] - x[0])
                self.line.measurements.n_cont =  y[0] - self.line.measurements.m_cont * x[0]

                self.line.measurements.m_cont_err_intg = np.sqrt((err[0] * (-1 / (x[-1] - x[0]))) ** 2 + (err[-1] * (1/(x[-1] - x[0]))) ** 2)
                self.line.measurements.n_cont_err_intg = np.sqrt(err[0] ** 2 + (self.line.measurements.m_cont_err_intg * (-x[0])) ** 2)
                cont_arr = self._spec.wave * self.line.measurements.m_cont + self.line.measurements.n_cont

            case 'fit':

                x, y = self._spec.wave[idcs_cont].compressed(), self._spec.cont[idcs_cont].compressed()
                err = self._spec.err_flux[idcs_cont].compressed() if self._spec.err_flux is not None else [0, 0]

                self.line.measurements.m_cont = (y[-1] - y[0]) / (x[-1] - x[0])
                self.line.measurements.n_cont =  y[0] - self.line.measurements.m_cont * x[0]

                self.line.measurements.m_cont_err_intg = np.sqrt((err[0] * (-1 / (x[-1] - x[0]))) ** 2 + (err[-1] * (1/(x[-1] - x[0]))) ** 2)
                self.line.measurements.n_cont_err_intg = np.sqrt(err[0] ** 2 + (self.line.measurements.m_cont_err_intg * (-x[0])) ** 2)
                cont_arr = self._spec.cont

            case _:
                raise LiMe_Error(f'Continuum source "{user_cont_source}" is not recognized. '
                                 f'Please use "central", "adjacent" and "fit".')

        # # Initial continuum level value
        # self.line.measurements.cont = (self.line.measurements.m_cont * self._spec.wave[idcs_emis].compressed().mean() +
        #                                self.line.measurements.n_cont)

        return cont_arr

    def pixel_error_calculation(self, idcs_continua, user_error_from_bands):

        # Constant pixel error array from adjacent bands
        if user_error_from_bands:
            return np.ma.array(np.full(self._spec.wave.shape, np.std(self._spec.flux[idcs_continua] - (self.line.measurements.m_cont * self._spec.wave[idcs_continua] + self.line.measurements.n_cont))), mask=np.zeros(self._spec.wave.shape, dtype=bool))

        # Pixel array
        else:
            return self._spec.err_flux


    def velocity_profile_calc(self, measurements, peakIdx, selection_wave, selection_flux, selection_cont, emission_check,
                              min_array_dim=15):

        # line, peakIdx, emis_flux, cont_arr, emission_check
        idx_0 = compute_FWHM0(peakIdx, selection_flux, -1, selection_cont, emission_check)
        idx_f = compute_FWHM0(peakIdx, selection_flux, 1, selection_cont, emission_check)

        # Velocity calculations
        vel_array = (c_KMpS * (selection_wave[idx_0:idx_f] - measurements.peak_wave) / measurements.peak_wave).compressed()

        # In case the vel_array has length zero:
        if vel_array.size > min_array_dim:

            line_flux = selection_flux[idx_0:idx_f].compressed()
            cont_flux = selection_cont[idx_0:idx_f].compressed()

            peakIdx = np.argmax(line_flux)
            percentFluxArray = np.cumsum(line_flux-cont_flux) * measurements.pixelWidth / measurements.intg_flux * 100

            if emission_check:
                blue_range = line_flux[:peakIdx] > measurements.peak_flux/2
                red_range = line_flux[peakIdx:] < measurements.peak_flux/2
            else:
                blue_range = line_flux[:peakIdx] < measurements.peak_flux/2
                red_range = line_flux[peakIdx:] > measurements.peak_flux/2

            # In case the peak is at the edge
            if (blue_range.size > 2) and (red_range.size > 2):

                # Integrated FWHM
                vel_FWHM_blue = vel_array[:peakIdx][np.argmax(blue_range)]
                vel_FWHM_red = vel_array[peakIdx:][np.argmax(red_range)]

                measurements.FWHM_i = vel_FWHM_red - vel_FWHM_blue

                # Interpolation for integrated kinematics
                percentInterp = interp1d(percentFluxArray, vel_array, kind='slinear', fill_value='extrapolate')
                velocPercent = percentInterp(TARGET_PERCENTILES)

                # Bug with masked array median operation
                if np.ma.isMaskedArray(vel_array): #TODO Lime2.0 removal
                    measurements.v_med = np.ma.median(vel_array)
                else:
                    measurements.v_med = np.median(vel_array)

                measurements.w_i = (velocPercent[0] * measurements.peak_wave / c_KMpS) + measurements.peak_wave
                measurements.v_1 = velocPercent[1]
                measurements.v_5 = velocPercent[2]
                measurements.v_10 = velocPercent[3]
                measurements.v_50 = velocPercent[4]
                measurements.v_90 = velocPercent[5]
                measurements.v_95 = velocPercent[6]
                measurements.v_99 = velocPercent[7]
                measurements.w_f = (velocPercent[8] * measurements.peak_wave / c_KMpS) + measurements.peak_wave

                # Full width zero intensity
                measurements.FWZI = velocPercent[8] - velocPercent[0]

                W_80 = measurements.v_90 - measurements.v_10
                W_90 = measurements.v_95 - measurements.v_5

                # # This are not saved... should they
                # A_factor = ((measurements.v_90 - measurements.v_med) - (measurements.v_med-measurements.v_10)) / W_80
                # K_factor = W_90 / (1.397 * measurements.FWHM_i)

        return

    def report(self, return_text=False):

        # Get the Lmfit report
        report = fit_report(self.profile.output)

        # Dictionary with the generic entries to line components mapping
        map_dict = dict(zip([f'line{i}' for i in range(len(self.line.list_comps))], self.line.param_arr('label')))

        # Replace the generic names
        pattern = re.compile(r'(' + '|'.join(map(re.escape, map_dict.keys())) + r')')
        report = pattern.sub(lambda match: map_dict[match.group()], report)

        # Print by default, otherwise return the report string
        if return_text:
            return report
        else:
            print(report)


