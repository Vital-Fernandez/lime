import logging

import numpy as np
from lmfit.models import Model
from lmfit import fit_report
from scipy import stats
from scipy.interpolate import interp1d
from scipy.special import wofz
from .tools import compute_FWHM0, mult_err_propagation
from .io import LiMe_Error, _ATTRIBUTES_PROFILE, _RANGE_PROFILE_FIT, _LOG_COLUMNS

_logger = logging.getLogger('LiMe')

c_KMpS = 299792.458  # Speed of light in Km/s (https://en.wikipedia.org/wiki/Speed_of_light)

k_GaussArea = np.sqrt(2 * np.pi)
sqrt2 = np.sqrt(2)

k_gFWHM = 2 * np.sqrt(2 * np.log(2))

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


def pseudo_voigt_model(x, amp, center, sigma, frac):
    return frac * gaussian_model(x, amp, center, sigma) + (1 - frac) * lorentz_model(x, amp, center, sigma)


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

    return 0.5346 * FWHM_lorentz + np.sqrt(0.2166 * FWHM_lorentz + FWHM_gauss)


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


PROFILE_PARAMS = np.array(['m_cont', 'n_cont', 'amp', 'center', 'sigma', 'gamma', 'frac'])
PROFILE_FUNCTIONS =  {'g': gaussian_model, 'l': lorentz_model, 'v': voigt_model, 'pv': pseudo_voigt_model}
AREA_FUNCTIONS = {'g': gaussian_area, 'l': lorentz_area, 'v': voigt_area, 'pv': pseudo_voigt_area}
FWHM_FUNCTIONS =  {'g': g_FWHM, 'l': l_FWHM, 'v': v_FWHM, 'pv': v_FWHM}


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
                gamma_array = log.loc[line_list, 'frac'].to_numpy()
                gaussian_array[:, i] = profile_function(x_array, amp_array[i], center_array[i], sigma_array[i], gamma_array[i])[:, 0]
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

# def gaussian_profiles_computation(line_list, log, z_corr, res_factor=100, interval=('w3', 'w4'), x_array=None):
#
#     # All lines are computed with the same wavelength interval: The maximum interval[1]-interval[0] in the log times 3
#     # and starting at interval[0] values beyond interval[0] are masked
#
#     #TODO Resfactor should be a lime parameter
#     if x_array is None:
#
#         amp_array = log.loc[line_list, 'amp'].to_numpy()
#         center_array = log.loc[line_list, 'center'].to_numpy()
#         sigma_array = log.loc[line_list, 'sigma'].to_numpy()
#
#         wmin_array = log.loc[line_list, interval[0]].to_numpy() * z_corr
#         wmax_array = log.loc[line_list, interval[1]].to_numpy() * z_corr
#         w_mean = np.max(wmax_array - wmin_array)
#
#         x_zero = np.linspace(0, w_mean, res_factor)
#         x_array = np.add(np.c_[x_zero], wmin_array)
#
#         gaussian_array = gaussian_model(x_array, amp_array, center_array, sigma_array)
#
#         for i in range(x_array.shape[1]):
#             idcs_nan = x_array[:, i] > wmax_array[i]
#             x_array[idcs_nan, i] = np.nan
#             gaussian_array[idcs_nan, i] = np.nan
#
#         return x_array, gaussian_array
#
#     # All lines are computed with the wavelength range provided by the user
#     else:
#
#         # Profile container
#         gaussian_array = np.zeros((len(x_array), len(line_list)))
#
#         # Compute the individual profiles
#         for i, comp in enumerate(line_list):
#             amp = log.loc[comp, 'amp']
#             center = log.loc[comp, 'center']
#             sigma = log.loc[comp, 'sigma']
#
#             # Gaussian components calculation
#             gaussian_array[:, i] = gaussian_model(x_array, amp, center, sigma)
#
#         return gaussian_array


# g_params = np.array(['amp', 'center', 'sigma', 'cont_slope', 'cont_intercept'])


class ProfileModelCompiler:

    def __init__(self, line, redshift, user_conf, y):

        self.n_comps, self.ref_wave = None, None
        self.model = None
        self.params = None
        self.output = None

        # Confirm the number of gaussian components
        self.n_comps = len(line.list_comps)

        # Decide the line reference wavelength
        self.ref_wave = line.wavelength * (1 + redshift) if line.blended_check else np.atleast_1d(line.peak_wave)

        # Continuum model
        self.model = Model(linear_model)
        self.model.prefix = f'line0_'

        # Fix or not the continuum
        m_cont_conf = _SLOPE_FIX_PAR if line._cont_from_adjacent else _SLOPE_FREE_PAR
        n_cont_conf = _INTER_FIX_PAR if line._cont_from_adjacent else _INTER_FREE_PAR
        self.define_param(0, line, 'm_cont', line.m_cont, m_cont_conf, user_conf)
        self.define_param(0, line, 'n_cont', line.n_cont, n_cont_conf, user_conf)

        # Add one gaussian model per component
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
            AMP_PAR = dict(value=None, min=min_lim, max=max_lim, vary=True, expr=None)
            self.define_param(idx, line, 'amp', peak_0, AMP_PAR, user_conf)
            self.define_param(idx, line, 'center', self.ref_wave[idx], _CENTER_PAR, user_conf, redshift)
            self.define_param(idx, line, 'sigma', 2*line.pixelWidth, _SIG_PAR, user_conf)
            # self.define_param(idx, line, 'area', None, _AREA_PAR, user_conf)

            # Gamma param for Voigt
            if line._p_shape[idx] == 'v':
                self.define_param(idx, line, 'gamma', 2 * line.pixelWidth, _SIG_PAR, user_conf)

            # Frac for Pseudo-Voigt
            if line._p_shape[idx] == 'pv':
                self.define_param(idx, line, 'frac', 0.5, _FRAC_PAR, user_conf)

        return

    def fit(self, line, x, y, err, method):

        # Compute weights
        if err is None:
            weights = np.full(x.size, 1.0 / line.cont_err)
        else:
            weights = 1.0/err

        # Unpack the mask for LmFit analysis
        if np.ma.is_masked(x):
            idcs_good = ~x.mask
            x_in = x.data[idcs_good]
            y_in = y.data[idcs_good]
            weights_in = weights[idcs_good]
        else:
            x_in, y_in, weights_in = x, y, weights

        # Fit the line
        self.params = self.model.make_params()
        self.output = self.model.fit(y_in, self.params, x=x_in, weights=weights_in, method=method, nan_policy='omit')

        # Check fitting quality
        self.review_fitting(line)

        return

    def measurements_calc(self, line, user_conf, inst_FWHM, temp):

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
                line.sigma_thermal = np.full(self.n_comps, np.nan)

            # Compute the profile areas
            profile_flux_dist = AREA_FUNCTIONS[line._p_shape[i]](line, i, 1000)
            line.profile_flux[i] = np.mean(profile_flux_dist)
            line.profile_flux_err[i] = np.std(profile_flux_dist)

            # Compute the equivalent widths
            if i == 0:
                cont_dist = np.random.normal(line.cont, line.cont_err, 1000)

            eqw_dist = profile_flux_dist/cont_dist
            line.eqw[i] = np.mean(eqw_dist)
            line.eqw_err[i] = np.std(eqw_dist)

            # Compute FWHM_p (Profile Full Width Half Maximum)
            line.FWHM_p[i] = FWHM_FUNCTIONS[line._p_shape[i]](line, i)

            # Compute thermal correction
            atom_mass = ATOMIC_MASS.get(line.particle[i].symbol, np.nan)
            line.sigma_thermal[i] = np.sqrt(k_Boltzmann * temp / atom_mass) / 1000

            # Check parameters error propagation
            self.review_err_propagation(line, i, comp_label)

        # Centroid redshifts
        if line.blended_check:
            line.z_line = line.center/line.wavelength - 1
        else:
            line.z_line = line.peak_wave/line.wavelength - 1

        # Kinematics
        line.v_r = c_KMpS * (line.center - self.ref_wave) / self.ref_wave
        line.v_r_err = c_KMpS * (line.center_err) / self.ref_wave
        line.sigma_vel = c_KMpS * line.sigma / self.ref_wave
        line.sigma_vel_err = c_KMpS * line.sigma_err / self.ref_wave

        # Compute instrumental correction
        line.sigma_instr = k_gFWHM / line.inst_FWHM if not np.isnan(inst_FWHM) else None

        # Fitting diagnostics
        line.chisqr, line.redchi = self.output.chisqr, self.output.redchi
        line.aic, line.bic = self.output.aic, self.output.bic

        return

    def save_measurements(self, line, user_conf, inst_FWHM, temp):

        # Recalculate the equivalent width with the gaussian fluxes if blended
        if line.blended_check:
            line.eqw = np.empty(self.n_comps)
            line.eqw_err = np.empty(self.n_comps)

        line.amp, line.amp_err = np.empty(self.n_comps), np.empty(self.n_comps)
        line.center, line.center_err = np.empty(self.n_comps), np.empty(self.n_comps)
        line.sigma, line.sigma_err = np.empty(self.n_comps), np.empty(self.n_comps)
        line.gamma, line.gamma_err = np.empty(self.n_comps), np.empty(self.n_comps)
        line.frac, line.frac_err = np.empty(self.n_comps), np.empty(self.n_comps)
        line.alpha, line.alpha_err = np.empty(self.n_comps), np.empty(self.n_comps)
        line.beta, line.beta_err = np.empty(self.n_comps), np.empty(self.n_comps)

        line.v_r, line.v_r_err = np.empty(self.n_comps), np.empty(self.n_comps)
        line.sigma_vel, line.sigma_vel_err = np.empty(self.n_comps), np.empty(self.n_comps)
        line.profile_flux, line.profile_flux_err = np.empty(self.n_comps), np.empty(self.n_comps)

        line.FWHM_p = np.empty(self.n_comps)
        line.sigma_thermal = np.empty(self.n_comps)

        # Fitting diagnostics
        line.chisqr, line.redchi = self.output.chisqr, self.output.redchi
        line.aic, line.bic = self.output.aic, self.output.bic

        # Instrumental sigma #TODO check this
        line.sigma_instr = k_gFWHM / line.inst_FWHM if not np.isnan(inst_FWHM) else None

        # Store lmfit measurements
        for i, user_ref in enumerate(line.list_comps):

            # Recover using the lmfit name
            comp = f'line{i}'

            # Profile parameters
            for j, param in enumerate(['amp', 'center', 'sigma', 'gamma', 'frac']):
                param_fit = self.output.params.get(f'{comp}_{param}', None)

                if param_fit is not None:
                    term_mag = getattr(line, param)
                    term_mag[i] = param_fit.value
                    term_err = getattr(line, f'{param}_err')
                    term_err[i] = param_fit.stderr

                # Case with error propagation from _kinem command
                if (term_err[i] == 0) and (f'{user_ref}_{param}_err' in user_conf):
                    term_err[i] = user_conf[f'{user_ref}_{param}_err'] # TODO do I need this one here, can I use the one below

            # Gaussian area
            line.profile_flux[i] = self.output.params[f'{comp}_area'].value
            line.profile_flux_err[i] = self.output.params[f'{comp}_area'].stderr

            # Equivalent with gaussian flux for blended components TODO compute self.cont from linear fit
            if line.blended_check:
                line.eqw[i], line.eqw_err[i] = line.profile_flux[i] / line.cont, line.profile_flux_err[i] / line.cont

            # Kinematics
            line.v_r[i] = c_KMpS * (line.center[i] - self.ref_wave[i])/self.ref_wave[i]
            line.v_r_err[i] = c_KMpS * (line.center_err[i])/self.ref_wave[i]
            line.sigma_vel[i] = c_KMpS * line.sigma[i]/self.ref_wave[i]
            line.sigma_vel_err[i] = c_KMpS * line.sigma_err[i]/self.ref_wave[i]
            line.FWHM_p[i] = k_gFWHM * line.sigma_vel[i]

            # Compute the thermal correction
            atom_mass = ATOMIC_MASS.get(line.particle[i].symbol, np.nan)
            line.sigma_thermal[i] = np.sqrt(k_Boltzmann * temp / atom_mass) / 1000

            # Check parameters error progragation from the lmfit parameter
            self.review_err_propagation(line, i, comp)

        # Calculate the line redshift
        if line.blended_check:
            line.z_line = line.center/line.wavelength - 1
        else:
            line.z_line = line.peak_wave/line.wavelength - 1

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

            # TODO this one could be faster
            # Do not parse expressions for single components
            if line.blended_check:
                expr = param_conf['expr']
                for i, comp in enumerate(comps):
                    # for g_param in g_params:
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

        # Additional preparation for area parameter
        if '_area' in param_ref:
            if (param_conf['expr'] is None) and (param_conf['value'] == param_value):
                param_conf['value'] = None
                if line._p_shape[idx] == 'g':
                    param_conf['expr'] = f'line{idx}_amp*2.5066282746*line{idx}_sigma'
                elif line._p_shape[idx] == 'l':
                    param_conf['expr'] = f'3.14159265*line{idx}_amp*line{idx}_sigma'
                elif line._p_shape[idx] == 'v':
                    param_conf['expr'] = f'3.14159265*line{idx}_amp*line{idx}_sigma'
                elif line._p_shape[idx] == 'pv':
                    param_conf['expr'] = f'line{idx}_frac*line{idx}_amp*2.5066282746*line{idx}_sigma+(1-line{idx}_frac)*3.14159265*line{idx}_amp*line{idx}_sigma'
                else:
                    raise LiMe_Error(f'Profile type "{line._p_shape[idx]}" for line {line} is not recognized')

        # Additional preparation for center parameter: Multiply value, min, max by redshift
        if '_center' in param_ref:
            if user_ref in user_conf:
                param_user_conf = user_conf[user_ref]
                for param_conf_entry in ('value', 'min', 'max'):
                    if param_conf_entry in param_user_conf:
                        param_conf[param_conf_entry] = param_conf[param_conf_entry] * (1 + z_obj)

        # Check the limits on the fitting # TODO some params (blended lines) do not have initial value amp wide
        param_value = param_conf.get('value')
        # if param_value is not None:
        #     if (param_value < param_conf['min']) or (param_value > param_conf['max']):
        #         _logger.warning(f'Initial value for {param_ref} is outside min-max boundaries: '
        #                         f'{param_conf["min"]} < {param_value} < {param_conf["max"]}')

        # Assign the parameter configuration to the model
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

    def review_err_propagation(self, line, idx_line, comp):

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

        return


class LineFitting:

    """Class to measure emission line fluxes and fit them as gaussian curves"""

    # _minimize_method = 'leastsq'

    # _atomic_mass_dict = ATOMIC_MASS

    # Switch for emission and absorption lines

    def __init__(self):

        # self.fit_params = {}
        # self.fit_output = None
        self.profile = None

        return

    def integrated_properties(self, line, emis_wave, emis_flux, emis_err, cont_wave, cont_flux, cont_err, emission_check,
                              n_steps=1000):

        # Gradient and interception of linear continuum using adjacent regions
        if line._cont_from_adjacent:
            if np.ma.isMaskedArray(cont_flux): # TODO check if this is == or is
                input_wave, input_flux = cont_wave.data[cont_wave.mask == False], cont_flux.data[cont_flux.mask == False]
            else:
                input_wave, input_flux = cont_wave, cont_flux

            # TODO include error pixel
            line.m_cont, line.n_cont, r_value, p_value, std_err = stats.linregress(input_wave, input_flux)

        # Using line first and last point
        else:
            w2, w3 = emis_wave[0], emis_wave[-1]
            f2, f3 = emis_flux[0], emis_flux[-1]
            line.m_cont = (f3 - f2) / (w3 - w2)
            line.n_cont = f3 - line.m_cont * w3

        # Compute continuum
        continuaFit = cont_wave * line.m_cont + line.n_cont
        lineLinearCont = emis_wave * line.m_cont + line.n_cont

        # Peak or through index
        peakIdx = np.argmax(emis_flux) if emission_check else np.argmin(emis_flux)

        # Assign values
        line.n_pixels = emis_wave.size
        line.peak_wave = emis_wave[peakIdx]
        line.peak_flux = emis_flux[peakIdx]
        line.pixelWidth = np.diff(emis_wave).mean()
        line.cont = line.peak_wave * line.m_cont + line.n_cont
        line.cont_err = np.std(cont_flux - continuaFit) if cont_err is None else np.mean(cont_err)

        # Warning if continuum above or below line peak/through
        if emission_check and (lineLinearCont[peakIdx] > emis_flux[peakIdx]):
            _logger.warning(f'Line {line.label} introduced as an emission but the line peak is below the continuum level')

        if emission_check and (lineLinearCont[peakIdx] > emis_flux[peakIdx]):
            _logger.warning(f'Line {line.label} introduced as an absorption but the line peak is below the continuum level')

        # Establish the pixel sigma error
        err_array = line.cont_err if emis_err is None else emis_err

        # Monte Carlo to measure line flux and uncertainty
        normalNoise = np.random.normal(0.0, err_array, (n_steps, emis_flux.size))
        lineFluxMatrix = emis_flux + normalNoise
        areasArray = (lineFluxMatrix.sum(axis=1) - lineLinearCont.sum()) * line.pixelWidth

        # Assign values
        line.intg_flux = areasArray.mean()
        line.intg_flux_err = areasArray.std()

        # Compute the integrated singal to noise # TODO is this an issue for absorptions
        amp_ref = line.peak_flux - line.cont
        if emission_check:
            if amp_ref < 0:
                amp_ref = line.peak_flux

        # Compute SN_r
        err_cont = line.cont_err if emis_err is None else np.mean(emis_err)
        line.snr_line = signal_to_noise_rola(amp_ref, err_cont, line.n_pixels)
        line.snr_cont = line.cont/line.cont_err

        # Logic for very small lines
        snr_array = signal_to_noise_rola(emis_flux - line.cont, line.cont_err, 1)
        if (np.sum(snr_array > 5) < 3) and (line.blended_check is False):
            line._narrow_check = True
        else:
            line._narrow_check = False

        # Line width to the pixel below the continuum (or mask size if not happening)
        idx_0 = compute_FWHM0(peakIdx, emis_flux, -1, lineLinearCont, emission_check)
        idx_f = compute_FWHM0(peakIdx, emis_flux, 1, lineLinearCont, emission_check)

        # Velocity calculations
        velocArray = c_KMpS * (emis_wave[idx_0:idx_f] - line.peak_wave) / line.peak_wave
        self.velocity_profile_calc(line, velocArray, emis_flux[idx_0:idx_f], lineLinearCont[idx_0:idx_f], emission_check)

        # Pixel velocity # TODO we are not using this one
        line.pixel_vel = c_KMpS * line.pixelWidth/line.peak_wave

        # Equivalent width computation (it must be an 1d array to avoid conflict in blended lines)
        lineContinuumMatrix = lineLinearCont + normalNoise
        eqwMatrix = areasArray / lineContinuumMatrix.mean(axis=1)

        line.eqw = np.array(eqwMatrix.mean(), ndmin=1)
        line.eqw_err = np.array(eqwMatrix.std(), ndmin=1)

        return

    def profile_fitting(self, line, x, y, err, z_obj, user_conf, fit_method='leastsq', temp=10000.0, inst_FWHM=np.nan):

        # Compile the Lmfit component models
        self.profile = ProfileModelCompiler(line, z_obj, user_conf, y)

        # Fit the model
        self.profile.fit(line, x, y, err, fit_method)

        # Store the results into the line attributes
        self.profile.measurements_calc(line, user_conf, inst_FWHM, temp)

        return

    def velocity_profile_calc(self, line, vel_array, line_flux, cont_flux, emission_check, min_array_dim=15):

        # In case the vel_array has length zero:
        if vel_array.size > 2:

            # Only compute the velocity percentiles for line bands with more than 15 pixels
            valid_pixels = vel_array.size if not np.ma.is_masked(vel_array) else np.sum(~vel_array.mask)

            if valid_pixels > min_array_dim:

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
                    if np.ma.isMaskedArray(vel_array):
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

    def report(self):

        print(fit_report(self.profile.output))

        return

