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

c_KMpS = 299792.458  # Speed of light in Km/s (https://en.wikipedia.org/wiki/Speed_of_light)

k_GaussArea = np.sqrt(2 * np.pi)
sqrt2 = np.sqrt(2)

k_gFWHM = 2 * np.sqrt(2 * np.log(2))

k_eFWHM = 2 * np.log(2)

TARGET_PERCENTILES = np.array([0, 1, 5, 10, 50, 90, 95, 99, 100])

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

    amp = np.random.normal(line.amp[idx], line.amp_err[idx], n_steps)
    sigma = np.random.normal(line.sigma[idx], line.sigma_err[idx], n_steps)

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


PROFILE_PARAMS = np.array(['m_cont', 'n_cont', 'amp', 'center', 'sigma', 'gamma', 'alpha', 'frac', 'a', 'b', 'c'])

PROFILE_FUNCTIONS =  {'g': gaussian_model, 'l': lorentz_model, 'v': voigt_model,
                      'pv': pseudo_voigt_model, 'pp': pseudo_power_model,
                      'p': broken_powerlaw_model, 'e': exponential_model}

AREA_FUNCTIONS = {'g': gaussian_area, 'l': lorentz_area, 'v': voigt_area,
                  'pv': pseudo_voigt_area, 'pp': pseudo_power_area,
                  'p': power_area, 'e': exp_area}

FWHM_FUNCTIONS =  {'g': g_FWHM, 'l': l_FWHM, 'v': v_FWHM, 'pv': v_FWHM, 'pp': pp_FWHM, 'p': p_FWHM, 'e': e_FWHM}


def signal_to_noise_rola(amp, std_cont, n_pixels):

    snr = (k_GaussArea/6) * (amp/std_cont) * np.sqrt(n_pixels)

    return snr


def profiles_computation(line_list, log, z_corr, shape_list, x_array=None, interval=('w3', 'w4'), res_factor=100):

    # All lines are computed with the same wavelength interval: The maximum interval[1]-interval[0] in the log times 3
    # and starting at interval[0] values beyond interval[0] are masked

    #TODO Resfactor should be a lime parameter
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

                amp = log.loc[comp, 'amp']
                center = log.loc[comp, 'center']
                sigma = log.loc[comp, 'sigma']

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
            m_cont = log.loc[comp, 'm_cont']
            n_cont = log.loc[comp, 'n_cont']
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
    line.sigma_thermal = np.full(len(line.list_comps), np.nan)
    for i, comp in enumerate(line.sigma_thermal):
        atom_mass = ATOMIC_MASS.get(line.particle[i].symbol, np.nan)
        line.sigma_thermal[i] = np.sqrt(k_Boltzmann * temperature / atom_mass) / 1000

    # Instrumental correction
    if R_arr is not None:
        if np.isscalar(R_arr):
            line.sigma_instr = np.mean(wave_arr.compressed() / (R_arr * k_gFWHM))
        else:
            mask_data = ~wave_arr.mask
            line.sigma_instr = np.mean(wave_arr[mask_data] / (R_arr[idcs_line][mask_data] * k_gFWHM))
            wave_arr[mask_data] / (R_arr[idcs_line][mask_data] * k_gFWHM)
    else:
        line.sigma_instr = np.nan

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

    def __init__(self, line, redshift, user_conf, y):

        self.n_comps, self.ref_wave = None, None
        self.model = None
        self.params = None
        self.output = None

        # Confirm the number of gaussian components
        self.n_comps = len(line.list_comps)

        # Decide the line reference wavelength
        # self.ref_wave = line.wavelength * (1 + redshift) if line.blended_check else np.atleast_1d(line.peak_wave)
        self.ref_wave = line.wavelength * (1 + redshift)

        # Continuum models
        self.model = Model(linear_model)
        self.model.prefix = f'line0_'

        # Fix or not the continuum
        # m_cont_conf = _SLOPE_FIX_PAR if line._cont_from_adjacent else _SLOPE_FREE_PAR
        # n_cont_conf = _INTER_FIX_PAR if line._cont_from_adjacent else _INTER_FREE_PAR
        self.define_param(0, line, 'm_cont', line.m_cont, _SLOPE_FIX_PAR, user_conf)
        self.define_param(0, line, 'n_cont', line.n_cont, _INTER_FIX_PAR, user_conf)

        # Add one gaussian models per component
        for idx, comp in enumerate(line.list_comps):

            # Gaussian comp
            self.model += Model(PROFILE_FUNCTIONS[line._p_shape[idx]], prefix=f'line{idx}_')

            # Amplitude configuration
            if line._p_type[idx]:
                # Emission
                min_lim = 0
                max_lim = (line.peak_flux - line.cont) + line.cont_err if line._narrow_check else line.peak_flux * 1.5
                peak_0 = line.peak_flux - line.cont

            else:
                # Absorption
                trough = np.min(y)# This is necessary in case there are maximum and minimum lines... we need multiple peak_wave, peak_flux options
                min_lim = trough - line.cont
                max_lim = 0
                peak_0 = line.peak_flux * 0.5 - line.cont

            # Gaussian, lorentz and Voigt
            if (line._p_shape[idx] == 'g') or (line._p_shape[idx] == 'l'):
                AMP_PAR = dict(value=None, min=min_lim, max=max_lim, vary=True, expr=None)
                self.define_param(idx, line, 'amp', peak_0, AMP_PAR, user_conf)
                self.define_param(idx, line, 'center', self.ref_wave[idx], _CENTER_PAR, user_conf, redshift)
                self.define_param(idx, line, 'sigma', 2*line.pixelWidth, _SIG_PAR, user_conf)

            # Frac for Pseudo-Voigt
            if line._p_shape[idx] == 'pv':
                AMP_PAR = dict(value=None, min=min_lim, max=max_lim, vary=True, expr=None)
                self.define_param(idx, line, 'amp', peak_0, AMP_PAR, user_conf)
                self.define_param(idx, line, 'center', self.ref_wave[idx], _CENTER_PAR, user_conf, redshift)
                self.define_param(idx, line, 'sigma', 2*line.pixelWidth, _SIG_PAR, user_conf)
                self.define_param(idx, line, 'frac', 0.5, _FRAC_PAR, user_conf)

            # Exponential profile
            if line._p_shape[idx] == 'e':
                AMP_PAR = dict(value=None, min=min_lim, max=max_lim, vary=True, expr=None)
                _CENTER_PAR_e = dict(value= self.ref_wave[idx], min=line.mask[0]*(1+redshift),
                                     max=line.mask[5]*(1+redshift)
                                   , vary=True, expr=None) # TODO clean this

                self.define_param(idx, line, 'amp', peak_0, AMP_PAR, user_conf)
                self.define_param(idx, line, 'center', self.ref_wave[idx], _CENTER_PAR_e, user_conf, redshift)
                self.define_param(idx, line, 'alpha', 1.0/line.pixelWidth, _ALPHA_PAR, user_conf)

            # Gamma param for Voigt
            if line._p_shape[idx] == 'v':
                self.define_param(idx, line, 'amp', peak_0, AMP_PAR, user_conf)
                self.define_param(idx, line, 'center', self.ref_wave[idx], _CENTER_PAR, user_conf, redshift)
                self.define_param(idx, line, 'sigma', 2*line.pixelWidth, _SIG_PAR, user_conf)
                self.define_param(idx, line, 'gamma', 2 * line.pixelWidth, _SIG_PAR, user_conf)

            # Frac for Pseudo-Voigt
            if line._p_shape[idx] == 'pp':
                AMP_PAR = dict(value=None, min=min_lim, max=max_lim, vary=True, expr=None)
                self.define_param(idx, line, 'amp', peak_0, AMP_PAR, user_conf)
                self.define_param(idx, line, 'center', self.ref_wave[idx], _CENTER_PAR, user_conf, redshift)
                self.define_param(idx, line, 'sigma', 2*line.pixelWidth, _SIG_PAR, user_conf)
                self.define_param(idx, line, 'frac', 0.5, _FRAC_PAR, user_conf)
                self.define_param(idx, line, 'alpha', 2, _ALPHA_PAR, user_conf)

            if line._p_shape[idx] == 'p':
                A_PAR = dict(value=None, min=min_lim, max=max_lim, vary=True, expr=None)
                AMP_PAR = dict(value=None, min=min_lim, max=max_lim, vary=True, expr=None)

                self.define_param(idx, line, 'a', peak_0, A_PAR, user_conf)
                self.define_param(idx, line, 'b', self.ref_wave[idx], _B_PAR, user_conf, redshift)
                self.define_param(idx, line, 'c', 2 * line.pixelWidth, _C_PAR, user_conf)
                self.define_param(idx, line, 'alpha', -2, _ALPHA_PAR, user_conf)

        return

    def fit(self, line, x, y, err, method):

        # Unpack the mask for LmFit analysis
        idcs_good = ~x.mask
        x_in = x.data[idcs_good]
        y_in = y.data[idcs_good]

        # Compute weights
        if err is None:
            weights = np.full(x.size, 1.0 / line.cont_err)
            weights_in = weights
        else:
            weights = 1.0/err
            # weights_in = weights.data[idcs_good]
            weights_in = weights[idcs_good]

        # Fit the line
        self.params = self.model.make_params()

        with warnings.catch_warnings():
            warnings.simplefilter(_VERBOSE_WARNINGS)
            self.output = self.model.fit(y_in, self.params, x=x_in, weights=weights_in, method=method, nan_policy='omit')

        # Check fitting quality
        self.review_fitting(line)

        return

    def measurements_calc(self, line, user_conf):

        # Loop through the line components
        for i, comp_label in enumerate(line.list_comps):

            # Recover the profile parameters
            for j, param in enumerate(PROFILE_PARAMS):

                # Recover possible component from fit
                param_fit = self.output.params.get(f'line{i}_{param}', None)
                param_value = np.nan if param_fit is None else param_fit.value
                param_err = np.nan if param_fit is None else param_fit.stderr

                # First component: Recover scalar continuum parameters and initialize profile array parameters
                if i == 0:
                    if j < 2:
                        setattr(line, param, param_value)
                        setattr(line, f'{param}_err', param_err)

                    else:
                        setattr(line, param, np.full(self.n_comps, np.nan))
                        setattr(line, f'{param}_err', np.full(self.n_comps, np.nan))

                # Assign array curve parameters
                if j >= 2:
                    value_array, err_array = getattr(line, param), getattr(line, f'{param}_err')
                    value_array[i], err_array[i] = param_value, param_err

            # Initialize array parameters
            if i == 0:
                line.profile_flux = np.full(self.n_comps, np.nan)
                line.profile_flux_err = np.full(self.n_comps, np.nan)
                line.eqw = np.full(self.n_comps, np.nan)
                line.eqw_err = np.full(self.n_comps, np.nan)
                line.FWHM_p = np.full(self.n_comps, np.nan)
                # line.sigma_thermal = np.full(self.n_comps, np.nan)

            # Check for negative -0.0 # TODO this needs a better place # FIXME -0.0 error
            if np.signbit(line.sigma_err[i]):
                line.sigma_err[i] = np.nan
                if self.output.errorbars:
                    _logger.warning(f'Negative value for profile sigma at {line.label}')

            # Compute the profile areas
            profile_flux_dist = AREA_FUNCTIONS[line._p_shape[i]](line, i, 1000)
            line.profile_flux[i] = np.mean(profile_flux_dist)
            line.profile_flux_err[i] = np.std(profile_flux_dist)

            # Compute FWHM_p (Profile Full Width Half Maximum)
            line.FWHM_p[i] = FWHM_FUNCTIONS[line._p_shape[i]](line, i)

            # Check parameters error propagation
            self.review_err_propagation(line, i, comp_label, user_conf, self.output.errorbars)

        # Compute the equivalent widths
        line.eqw = line.profile_flux / line.cont
        line.eqw_err = np.abs(line.eqw) * np.sqrt(np.square(line.profile_flux_err/line.profile_flux) +
                                                  np.square(line.cont_err/line.cont))

        # Centroid redshifts
        line.z_line = line.center/line.wavelength - 1

        # Kinematics
        line.v_r = c_KMpS * (line.center - self.ref_wave) / self.ref_wave
        line.v_r_err = c_KMpS * line.center_err / self.ref_wave
        line.sigma_vel = c_KMpS * line.sigma / self.ref_wave
        line.sigma_vel_err = c_KMpS * line.sigma_err / self.ref_wave

        # Fitting diagnostics
        line.chisqr, line.redchi = self.output.chisqr, self.output.redchi
        line.aic, line.bic = self.output.aic, self.output.bic

        # Updated signal-to-noise based on a successful profile fitting
        if self.output.errorbars:
            line.snr_line = signal_to_noise_rola(line.amp, line.cont_err, line.n_pixels)

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
            if line.blended_check:
                expr = param_conf['expr']
                for i, comp in enumerate(comps):
                    for g_param in PROFILE_PARAMS:
                        expr = expr.replace(f'{comp}_{g_param}', f'line{i}_{g_param}')
                param_conf['expr'] = expr
            else:
                param_conf['expr'] = None
                _logger.info('Excluding expression from single line')

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
                    param_conf = dict(expr=new_expresion)

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

        # Assign the parameter configuration to the models
        self.model.set_param_hint(param_ref, **param_conf)

        return

    def review_fitting(self, line):

        if not self.output.errorbars:
            if line.observations == 'no':
                line.observations = 'No_errorbars'
            else:
                line.observations += 'No_errorbars'
            _logger.warning(f'Gaussian fit uncertainty estimation failed for {line.label}')

        return

    def review_err_propagation(self, line, idx_line, comp, user_conf, fit_success):

        # Check gaussian flux error
        if (line.profile_flux_err[idx_line] == 0.0) and (line.amp_err[idx_line] != 0.0) and (line.sigma_err[idx_line] != 0.0):
            line.profile_flux_err[idx_line] = mult_err_propagation(np.array([line.amp[idx_line], line.sigma[idx_line]]),
                                                                   np.array([line.amp_err[idx_line],
                                                                           line.sigma_err[idx_line]]),
                                                                   line.profile_flux[idx_line])

        # Check equivalent width error
        if line.blended_check:
            if (line.eqw_err[idx_line] == 0.0) and (line.cont_err != 0.0) and (line.profile_flux_err[idx_line] != 0.0):
                line.eqw_err[idx_line] = mult_err_propagation(np.array([line.cont, line.profile_flux[idx_line]]),
                                                              np.array([line.cont_err, line.profile_flux_err[idx_line]]),
                                                              line.eqw[idx_line])

        # Continuum error
        if (line.m_cont_err == 0.0) and (line.m_cont != 0.0) and (line.m_cont_err_intg != 0.0):
            line.m_cont_err = line.m_cont_err_intg

        if (line.n_cont_err == 0.0) and (line.n_cont != 0.0) and (line.n_cont_err_intg != 0.0):
            line.n_cont_err = line.n_cont_err_intg

        # Velocity and sigma error from line kinematics imported from an external line
        if line.center_err[idx_line] == 0:
            if user_conf.get(f'{comp}_center_err') is not None:
                line.center_err[idx_line] = user_conf.get(f'{comp}_center_err', np.nan)
                line.sigma_err[idx_line] = user_conf.get(f'{comp}_sigma_err', np.nan)

        return


class LineFitting:

    """Class to measure emission line fluxes and fit them as gaussian curves"""

    def __init__(self):

        self.profile = None

        return

    def integrated_properties(self, line, emis_wave, emis_flux, emis_err, n_steps=1000, min_array_dim=15):

        # Confirm line type
        emission_check = False if np.all(~line._p_type) else True

        # Assign values peak/through properties
        peakIdx = np.argmax(emis_flux) if emission_check else np.argmin(emis_flux)
        line.n_pixels = emis_wave.compressed().size
        line.peak_wave = emis_wave[peakIdx]
        line.peak_flux = emis_flux[peakIdx]
        line.pixelWidth = np.diff(emis_wave).mean()
        line.cont = line.peak_wave * line.m_cont + line.n_cont
        line.cont_err = emis_err[peakIdx]

        # Continuum calculation for the line region
        cont_arr = emis_wave * line.m_cont + line.n_cont

        # Warning if continuum above or below line peak/through
        if emission_check and (cont_arr[peakIdx] > emis_flux[peakIdx]):
            _logger.info(f'Line {line.label} introduced as an emission but the line peak is below the continuum level')

        if emission_check and (cont_arr[peakIdx] > emis_flux[peakIdx]):
            _logger.info(f'Line {line.label} introduced as an absorption but the line peak is below the continuum level')

        # Monte Carlo to measure line flux and uncertainty
        normalNoise = np.random.normal(0.0, emis_err, (n_steps, emis_flux.size))
        lineFluxMatrix = emis_flux + normalNoise
        areasArray = (lineFluxMatrix.sum(axis=1) - cont_arr.sum()) * line.pixelWidth

        # Assign integrated fluxes and uncertainty
        line.intg_flux = areasArray.mean()
        line.intg_flux_err = areasArray.std()

        # # Compute the integrated signal to noise # TODO is this an issue for absorptions
        # amp_ref = line.peak_flux - line.cont
        # if emission_check:
        #     if amp_ref < 0:
        #         amp_ref = line.peak_flux

        # Compute SN_r
        line.snr_line = signal_to_noise_rola(line.peak_flux - line.cont, line.cont_err, line.n_pixels)
        line.snr_cont = line.cont/line.cont_err

        # Logic for very small lines
        snr_array = signal_to_noise_rola(emis_flux - line.cont, line.cont_err, 1)
        if (np.sum(snr_array > 5) < 3) and (line.blended_check is False):
            line._narrow_check = True
        else:
            line._narrow_check = False

        # # Line width to the pixel below the continuum (or mask size if not happening) # TODO Lime2.0 skip all this if narrow
        # idx_0 = compute_FWHM0(peakIdx, emis_flux, -1, cont_arr, emission_check)
        # idx_f = compute_FWHM0(peakIdx, emis_flux, 1, cont_arr, emission_check)
        #
        # # Velocity calculations
        # velocArray = c_KMpS * (emis_wave[idx_0:idx_f] - line.peak_wave) / line.peak_wave
        # self.velocity_profile_calc(line, velocArray, emis_flux[idx_0:idx_f], cont_arr[idx_0:idx_f], emission_check)
        if (line.n_pixels >= min_array_dim) and (line._narrow_check is False):
            self.velocity_profile_calc(line, peakIdx, emis_wave, emis_flux, cont_arr, emission_check, min_array_dim=min_array_dim)

        # Pixel velocity # TODO we are not using this one
        line.pixel_vel = c_KMpS * line.pixelWidth/line.peak_wave

        # Equivalent width computation (it must be an 1d array to avoid conflict in blended lines) # TODO Lime2.0 put all this on its function
        lineContinuumMatrix = cont_arr + normalNoise
        eqwMatrix = areasArray / lineContinuumMatrix.mean(axis=1)

        line.eqw_intg = np.array(eqwMatrix.mean(), ndmin=1)
        line.eqw_intg_err = np.array(eqwMatrix.std(), ndmin=1)

        return

    def profile_fitting(self, line, x_arr, y_arr, err_arr, user_conf, fit_method='leastsq'):

        # Compile the Lmfit component models
        self.profile = ProfileModelCompiler(line, self._spec.redshift, user_conf, y_arr)

        # Fit the models
        self.profile.fit(line, x_arr, y_arr, err_arr, fit_method)

        # Store the results into the line attributes
        self.profile.measurements_calc(line, user_conf)

        return

    def continuum_calculation(self, idcs_emis, idcs_cont, user_cont_from_bands):

        # Use the continuum bands for the calculation
        if user_cont_from_bands:

            # Fit the model, including uncertainties
            params, covariance = curve_fit(linear_model,
                                           xdata=self._spec.wave[idcs_cont].compressed(),
                                           ydata=self._spec.flux[idcs_cont].compressed(),
                                           sigma=self._spec.err_flux[idcs_cont].compressed() if self._spec.err_flux is not None else None,
                                           absolute_sigma=True, check_finite=False)

            self.line.m_cont, self.line.n_cont = params
            self.line.m_cont_err_intg, self.line.n_cont_err_intg = np.sqrt(np.diag(covariance))


        # Use just the side bands edges
        else:
            x, y = self._spec.wave[idcs_emis].compressed(), self._spec.flux[idcs_emis].compressed()
            err = self._spec.err_flux[idcs_emis].compressed() if self._spec.err_flux is not None else [0, 0]

            self.line.m_cont = (y[-1] - y[0]) / (x[-1] - x[0])
            self.line.n_cont =  y[0] - self.line.m_cont * x[0]

            self.line.m_cont_err_intg = np.sqrt((err[0] * (-1 / (x[-1] - x[0]))) ** 2 + (err[-1] * (1/(x[-1] - x[0]))) ** 2)
            self.line.n_cont_err_intg = np.sqrt(err[0] ** 2 + (self.line.m_cont_err_intg * (-x[0])) ** 2)

        return

    def pixel_error_calculation(self, idcs_continua, user_error_from_bands):

        # Constant pixel error array from adjacent bands
        if user_error_from_bands:
            return np.full(self._spec.wave.shape, np.std(self._spec.flux[idcs_continua] -
                                                         (self.line.m_cont * self._spec.wave[idcs_continua] + self.line.n_cont)))

        # Pixel array
        else:
            return self._spec.err_flux

    def velocity_profile_calc(self, line, peakIdx, selection_wave, selection_flux, selection_cont, emission_check,
                              min_array_dim=15):

        # line, peakIdx, emis_flux, cont_arr, emission_check
        idx_0 = compute_FWHM0(peakIdx, selection_flux, -1, selection_cont, emission_check)
        idx_f = compute_FWHM0(peakIdx, selection_flux, 1, selection_cont, emission_check)

        # Velocity calculations
        vel_array = (c_KMpS * (selection_wave[idx_0:idx_f] - line.peak_wave) / line.peak_wave).compressed()

        # In case the vel_array has length zero:
        if vel_array.size > min_array_dim:

            line_flux = selection_flux[idx_0:idx_f].compressed()
            cont_flux = selection_cont[idx_0:idx_f].compressed()

            peakIdx = np.argmax(line_flux)
            percentFluxArray = np.cumsum(line_flux-cont_flux) * line.pixelWidth / line.intg_flux * 100

            if emission_check:
                blue_range = line_flux[:peakIdx] > line.peak_flux/2
                red_range = line_flux[peakIdx:] < line.peak_flux/2
            else:
                blue_range = line_flux[:peakIdx] < line.peak_flux/2
                red_range = line_flux[peakIdx:] > line.peak_flux/2

            # In case the peak is at the edge
            if (blue_range.size > 2) and (red_range.size > 2):

                # Integrated FWHM
                vel_FWHM_blue = vel_array[:peakIdx][np.argmax(blue_range)]
                vel_FWHM_red = vel_array[peakIdx:][np.argmax(red_range)]

                line.FWHM_i = vel_FWHM_red - vel_FWHM_blue

                # Interpolation for integrated kinematics
                percentInterp = interp1d(percentFluxArray, vel_array, kind='slinear', fill_value='extrapolate')
                velocPercent = percentInterp(TARGET_PERCENTILES)

                # Bug with masked array median operation
                if np.ma.isMaskedArray(vel_array): #TODO Lime2.0 removal
                    line.v_med = np.ma.median(vel_array)
                else:
                    line.v_med = np.median(vel_array)

                line.w_i = (velocPercent[0] * line.peak_wave / c_KMpS) + line.peak_wave
                line.v_1 = velocPercent[1]
                line.v_5 = velocPercent[2]
                line.v_10 = velocPercent[3]
                line.v_50 = velocPercent[4]
                line.v_90 = velocPercent[5]
                line.v_95 = velocPercent[6]
                line.v_99 = velocPercent[7]
                line.w_f = (velocPercent[8] * line.peak_wave / c_KMpS) + line.peak_wave

                # Full width zero intensity
                line.FWZI = velocPercent[8] - velocPercent[0]

                W_80 = line.v_90 - line.v_10
                W_90 = line.v_95 - line.v_5

                # This are not saved... should they
                A_factor = ((line.v_90 - line.v_med) - (line.v_med-line.v_10)) / W_80
                K_factor = W_90 / (1.397 * line.FWHM_i)

        # else:
        #     _logger.warning(f'{line.label} failure to measure the non-parametric FWHM')

        return

    def report(self, return_text=False):

        # Get the Lmfit report
        report = fit_report(self.profile.output)

        # Dictionary with the generic entries to line components mapping
        map_dict = dict(zip([f'line{i}' for i in range(len(self.line.list_comps))], self.line.list_comps))

        # Replace the generic names
        pattern = re.compile(r'(' + '|'.join(map(re.escape, map_dict.keys())) + r')')
        report = pattern.sub(lambda match: map_dict[match.group()], report)

        # Print by default, otherwise return the report string
        if return_text:
            return report
        else:
            print(report)


