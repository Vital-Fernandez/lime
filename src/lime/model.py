import logging

import matplotlib.pyplot as plt
import numpy as np
from lmfit.models import Model
from lmfit import fit_report
from scipy import stats, optimize
from scipy.interpolate import interp1d
from .tools import compute_FWHM0
from .io import LiMe_Error

_logger = logging.getLogger('LiMe')

c_KMpS = 299792.458  # Speed of light in Km/s (https://en.wikipedia.org/wiki/Speed_of_light)

k_GaussArea = np.sqrt(2 * np.pi)

k_FWHM = 2 * np.sqrt(2 * np.log(2))

TARGET_PERCENTILES = np.array([0, 1, 5, 10, 50, 90, 95, 99, 100])

# Atomic mass constant
amu = 1.66053906660e-27 # Kg

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

g_params = np.array(['amp', 'center', 'sigma', 'cont_slope', 'cont_intercept'])


def wavelength_to_vel(delta_lambda, lambda_wave, light_speed=c_KMpS):
    return light_speed * (delta_lambda/lambda_wave)


def iraf_snr(input_y):
    avg = np.average(input_y)
    rms = np.sqrt(np.mean(np.square(input_y - avg)))
    snr = avg/rms
    return snr


def signal_to_noise(flux_line, sigma_noise, n_pixels):
    # TODO this formula should be the complete one?
    snr = flux_line / (sigma_noise * np.sqrt(n_pixels))

    return snr


def signal_to_noise_rola(amp, std_cont, n_pixels):

    snr = (k_GaussArea/6) * (amp/std_cont) * np.sqrt(n_pixels)

    return snr


def gaussian_model(x, amp, center, sigma):
    """1-d gaussian curve : gaussian(x, amp, cen, wid)"""
    return amp * np.exp(-0.5 * (((x-center)/sigma) * ((x-center)/sigma)))


def lorentz_model(x, amp, center, sigma):
    "1-d lorentzian profile : lorentz(x, amp, cen, sigma)"

    return amp / ( 1 + (((x-center)/sigma) * ((x-center)/sigma)) )


def linear_model(x, slope, intercept):
    """a line"""
    return slope * x + intercept

def gaussian_profiles_computation(line_list, log, z_corr, res_factor=100, interval=('w3', 'w4'), x_array=None):

    # All lines are computed with the same wavelength interval: The maximum interval[1]-interval[0] in the log times 3
    # and starting at interval[0] values beyond interval[0] are masked

    #TODO Resfactor should be a lime parameter
    if x_array is None:

        idcs_lines = (log.index.isin(line_list))

        # TODO .values does not create a copy, we should switch to .to_numpy(copy=True)
        amp_array = log.loc[idcs_lines, 'amp'].values
        center_array = log.loc[idcs_lines, 'center'].values
        sigma_array = log.loc[idcs_lines, 'sigma'].values

        wmin_array = log.loc[idcs_lines, interval[0]].values * z_corr
        wmax_array = log.loc[idcs_lines, interval[1]].values * z_corr
        w_mean = np.max(wmax_array - wmin_array)

        x_zero = np.linspace(0, w_mean, res_factor)
        x_array = np.add(np.c_[x_zero], wmin_array)

        gaussian_array = gaussian_model(x_array, amp_array, center_array, sigma_array)

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
            amp = log.loc[comp, 'amp']
            center = log.loc[comp, 'center']
            sigma = log.loc[comp, 'sigma']

            # Gaussian components calculation
            gaussian_array[:, i] = gaussian_model(x_array, amp, center, sigma)

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


def mult_err_propagation(nominal_array, err_array, result):

    err_result = result * np.sqrt(np.sum(np.power(err_array/nominal_array, 2)))

    return err_result


def review_err_propagation(line, idx_line, comp):

    # Check gaussian flux error
    if (line.gauss_flux_err[idx_line] == 0.0) and (line.amp_err[idx_line] != 0.0) and (line.sigma_err[idx_line] != 0.0):
        line.gauss_flux_err[idx_line] = mult_err_propagation(np.array([line.amp[idx_line], line.sigma[idx_line]]),
                                                             np.array([line.amp_err[idx_line], line.sigma_err[idx_line]]),
                                                             line.gauss_flux[idx_line])

    # Check equivalent width error
    if line.blended_check:
        if (line.eqw_err[idx_line] == 0.0) and (line.std_cont != 0.0) and (line.gauss_flux_err[idx_line] != 0.0):
            line.eqw_err[idx_line] = mult_err_propagation(np.array([line.cont, line.gauss_flux[idx_line]]),
                                                          np.array([line.std_cont, line.gauss_flux_err[idx_line]]),
                                                          line.eqw[idx_line])

    # Check the error from the _kinem command imports

    return


def review_fitting(line, fit_output):

    if not fit_output.errorbars:
        if line.observations == 'no':
            line.observations = 'No_errorbars'
        else:
            line.observations += 'No_errorbars'
        _logger.warning(f'Gaussian fit uncertainty estimation failed for {line.label}')

    return


def g_FWHM(sigma_line):

    return k_FWHM * sigma_line


def l_FWHM(sigma_line):

    return np.pi * sigma_line


PROFILE_DICT = {'g': gaussian_model, 'l': lorentz_model}

AREA_DICT = {'g': gaussian_model, 'l': lorentz_model}

FWHM_DICT = {'g': g_FWHM, 'l': l_FWHM}


class LineFitting:

    """Class to measure emission line fluxes and fit them as gaussian curves"""

    _AMP_PAR = dict(value=None, min=0, max=np.inf, vary=True, expr=None)
    _CENTER_PAR = dict(value=None, min=-np.inf, max=np.inf, vary=True, expr=None)
    _SIG_PAR = dict(value=None, min=0, max=np.inf, vary=True, expr=None)
    _AREA_PAR = dict(value=None, min=-np.inf, max=np.inf, vary=True, expr=None)

    _AMP_ABS_PAR = dict(value=None, min=-np.inf, max=0, vary=True, expr=None)

    _SLOPE_PAR = dict(value=None, min=-np.inf, max=np.inf, vary=False, expr=None)
    _INTER_PAR = dict(value=None, min=-np.inf, max=np.inf, vary=False, expr=None)

    _SLOPE_FIX_PAR = dict(value=None, min=-np.inf, max=np.inf, vary=False, expr=None)
    _INTER_FIX_PAR = dict(value=None, min=-np.inf, max=np.inf, vary=False, expr=None)

    _minimize_method = 'leastsq'

    _atomic_mass_dict = ATOMIC_MASS

    # Switch for emission and absorption lines

    def __init__(self):

        self.fit_params = {}
        self.fit_output = None

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
        line.std_cont = np.std(cont_flux - continuaFit) if cont_err is None else np.mean(cont_err)

        # Warning if continuum above or below line peak/through
        if emission_check and (lineLinearCont[peakIdx] > emis_flux[peakIdx]):
            _logger.warning(f'Line {line.label} introduced as an emission but the line peak is below the continuum level')

        if emission_check and (lineLinearCont[peakIdx] > emis_flux[peakIdx]):
            _logger.warning(f'Line {line.label} introduced as an absorption but the line peak is below the continuum level')

        # Establish the pixel sigma error
        err_array = line.std_cont if emis_err is None else emis_err

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
        err_cont = line.std_cont if emis_err is None else np.mean(emis_err)
        line.snr_line = signal_to_noise_rola(amp_ref, err_cont, line.n_pixels)
        line.snr_cont = line.cont/line.std_cont

        # Logic for very small lines
        snr_array = signal_to_noise_rola(emis_flux-line.cont, line.std_cont, 1)
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

        # Confirm the number of gaussian components
        n_comps = len(self.line.list_comps)

        # Compute the line redshift and reference wavelength
        if line.blended_check:
            ref_wave = line.wavelength * (1 + z_obj)
        else:
            ref_wave = np.array([line.peak_wave], ndmin=1)

        # Continuum model
        fit_model = Model(linear_model)
        fit_model.prefix = f'line0_'

        # Fix or not the continuum
        cont_vary = False if line._cont_from_adjacent else True
        SLOPE_PAR = dict(value=None, min=-np.inf, max=np.inf, vary=cont_vary, expr=None)
        INTER_PAR = dict(value=None, min=-np.inf, max=np.inf, vary=cont_vary, expr=None)
        self.define_param(0, line, fit_model, 'slope', line.m_cont, SLOPE_PAR, user_conf)
        self.define_param(0, line, fit_model, 'intercept', line.n_cont, INTER_PAR, user_conf)

        # Add one gaussian model per component
        for idx, comp in enumerate(line.list_comps):

            # Gaussian comp
            fit_model += Model(PROFILE_DICT[line._p_shape[idx]], prefix=f'line{idx}_')

            # Amplitude configuration
            profile_comp = line.profile_comp[idx].split('-')

            if 'emi' in profile_comp:
                min_lim = 0
                max_lim = (line.peak_flux - line.cont) + line.std_cont if line._narrow_check else line.peak_flux * 1.5
                peak_0 = line.peak_flux - line.cont
            elif 'abs' in profile_comp:
                through = np.min(y)
                min_lim = through - line.cont
                max_lim = 0
                peak_0 = through * 0.5 - line.cont
            else:
                min_lim, max_lim = -np.inf, np.inf
                _logger.warning(f'No profile component LOCO {profile_comp} for "{line.profile_comp}" provided for line {comp}')
                peak_0 = line.peak_flux - line.cont

            AMP_PAR = dict(value=None, min=min_lim, max=max_lim, vary=True, expr=None)
            self.define_param(idx, line, fit_model, 'amp', peak_0, AMP_PAR, user_conf)
            self.define_param(idx, line, fit_model, 'center', ref_wave[idx], self._CENTER_PAR, user_conf, z_obj)
            self.define_param(idx, line, fit_model, 'sigma', 2*line.pixelWidth, self._SIG_PAR, user_conf)
            self.define_param(idx, line, fit_model, 'area', None, self._AREA_PAR, user_conf)

        # Compute weights
        if err is None:
            weights = np.full(x.size, 1.0/line.std_cont)
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
        self.fit_params = fit_model.make_params()
        self.fit_output = fit_model.fit(y_in, self.fit_params, x=x_in, weights=weights_in, method=fit_method,
                                        nan_policy='omit')

        # Check output quality
        review_fitting(line, self.fit_output)

        # Recalculate the equivalent width with the gaussian fluxes if blended
        if line.blended_check:
            line.eqw = np.empty(n_comps)
            line.eqw_err = np.empty(n_comps)

        line.amp, line.amp_err = np.empty(n_comps), np.empty(n_comps)
        line.center, line.center_err = np.empty(n_comps), np.empty(n_comps)
        line.sigma, line.sigma_err = np.empty(n_comps), np.empty(n_comps)

        line.v_r, line.v_r_err = np.empty(n_comps), np.empty(n_comps)
        line.sigma_vel, line.sigma_vel_err = np.empty(n_comps), np.empty(n_comps)
        line.gauss_flux, line.gauss_flux_err = np.empty(n_comps), np.empty(n_comps)

        line.FWHM_g = np.empty(n_comps)
        line.sigma_thermal = np.empty(n_comps)

        # Fitting diagnostics
        line.chisqr, line.redchi = self.fit_output.chisqr, self.fit_output.redchi
        line.aic, line.bic = self.fit_output.aic, self.fit_output.bic

        # Instrumental sigma #TODO check this
        line.sigma_instr = k_FWHM/line.inst_FWHM if not np.isnan(inst_FWHM) else None

        # Store lmfit measurements
        for i, user_ref in enumerate(line.list_comps):

            # Recover using the lmfit name
            comp = f'line{i}'

            # Gaussian parameters
            for j, param in enumerate(['amp', 'center', 'sigma']):
                param_fit = self.fit_output.params[f'{comp}_{param}']
                term_mag = getattr(line, param)
                term_mag[i] = param_fit.value
                term_err = getattr(line, f'{param}_err')
                term_err[i] = param_fit.stderr

                # Case with error propagation from _kinem command
                if (term_err[i] == 0) and (f'{user_ref}_{param}_err' in user_conf):
                    term_err[i] = user_conf[f'{user_ref}_{param}_err'] # TODO do I need this one here, can I use the one below

            # Gaussian area
            line.gauss_flux[i] = self.fit_output.params[f'{comp}_area'].value
            line.gauss_flux_err[i] = self.fit_output.params[f'{comp}_area'].stderr

            # Equivalent with gaussian flux for blended components TODO compute self.cont from linear fit
            if line.blended_check:
                line.eqw[i], line.eqw_err[i] = line.gauss_flux[i] / line.cont, line.gauss_flux_err[i] / line.cont

            # Kinematics
            line.v_r[i] = c_KMpS * (line.center[i] - ref_wave[i])/ref_wave[i]
            line.v_r_err[i] = c_KMpS * (line.center_err[i])/ref_wave[i]
            line.sigma_vel[i] = c_KMpS * line.sigma[i]/ref_wave[i]
            line.sigma_vel_err[i] = c_KMpS * line.sigma_err[i]/ref_wave[i]
            line.FWHM_g[i] = k_FWHM * line.sigma_vel[i]

            # Compute the thermal correction
            atom_mass = self._atomic_mass_dict.get(line.particle[i].symbol, np.nan)
            line.sigma_thermal[i] = np.sqrt(k_Boltzmann * temp / atom_mass) / 1000

            # Check parameters error progragation from the lmfit parameter
            review_err_propagation(line, i, comp)

        # Calculate the line redshift
        if line.blended_check:
            line.z_line = line.center/line.wavelength - 1
        else:
            line.z_line = line.peak_wave/line.wavelength - 1



        return

    def define_param(self, idx, line, model_obj, param_label, param_value, default_conf={}, user_conf={}, z_obj=0):

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
                    for g_param in g_params:
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
                    ineq_operation = '*' # TODO add remaining operations

                    # Split number and ref
                    ineq_expr = param_conf['expr'].replace('<', '').replace('>', '')
                    ineq_items = ineq_expr.split(ineq_operation)
                    ineq_linkedParam = ineq_items[0] if not is_digit(ineq_items[0]) else ineq_items[1]
                    ineq_lim = float(ineq_items[0]) if is_digit(ineq_items[0]) else float(ineq_items[1])

                    # Stablish the inequality configuration:
                    ineq_conf = {} # TODO need to check these limits
                    if '>' in param_conf['expr']:
                        ineq_conf['value'] = ineq_lim * 1.2
                        ineq_conf['min'] = ineq_lim
                    else:
                        ineq_conf['value'] = ineq_lim * 0.8
                        ineq_conf['max'] = ineq_lim

                    # Define new param
                    model_obj.set_param_hint(name=ineq_name, **ineq_conf)

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
        model_obj.set_param_hint(param_ref, **param_conf)

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

                    line.FWHM_intg = vel_FWHM_red - vel_FWHM_blue

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
                    K_factor = W_90 / (1.397 * line.FWHM_intg)

        # else:
        #     _logger.warning(f'{line.label} failure to measure the non-parametric FWHM')

        return

    def report(self):

        print(fit_report(self.fit_output))

        return