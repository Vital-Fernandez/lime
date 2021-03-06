import numpy as np
from lmfit.models import Model
from scipy import stats, optimize
from scipy.interpolate import interp1d
from .tools import label_decomposition, compute_FWHM0

c_KMpS = 299792.458  # Speed of light in Km/s (https://en.wikipedia.org/wiki/Speed_of_light)

k_GaussArea = np.sqrt(2 * np.pi)

k_FWHM = 2 * np.sqrt(2 * np.log(2))

TARGET_PERCENTILES = np.array([2, 5, 10, 50, 90, 95, 98])

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


def gaussian_model(x, amp, center, sigma):
    """1-d gaussian curve : gaussian(x, amp, cen, wid)"""
    return amp * np.exp(-0.5 * (((x-center)/sigma) * ((x-center)/sigma)))


def gauss_func(ind_params, a, mu, sigma):
    """
    Gaussian function

    This function returns the gaussian curve as the user speciefies and array of x values, the continuum level and
    the parameters of the gaussian curve

    :param ind_params: 2D array (x, z) where x is the array of abscissa values and z is the continuum level
    :param float a: Amplitude of the gaussian
    :param float mu: Center value of the gaussian
    :param float sigma: Sigma of the gaussian
    :return: Gaussian curve y array of values
    :rtype: np.ndarray
    """

    x, z = ind_params
    return a * np.exp(-((x - mu) * (x - mu)) / (2 * (sigma * sigma))) + z


def gaussian_profiles_computation(line_list, log, z_corr, res_factor=100, interval=('w3', 'w4'), x_array=None):

    # All lines are computed with the same wavelength interval: The maximum interval[1]-interval[0] in the log times 3
    # and starting at interval[0] values beyond interval[0] are masked

    #TODO Resfactor should be a lime parameter
    if x_array is None:

        idcs_lines = (log.index.isin(line_list))

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


def linear_model(x, slope, intercept):
    """a line"""
    return slope * x + intercept


def is_digit(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


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


def mult_err_propagation(nominal_array, err_array, result):

    err_result = result * np.sqrt(np.sum(np.power(err_array/nominal_array, 2)))

    return err_result


class EmissionFitting:

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

        self.line, self.mask = '', np.array([np.nan] * 6)
        self.blended_check, self.profile_label = False, 'no'

        self.intg_flux, self.intg_err = None, None
        self.peak_wave, self.peak_flux = None, None
        self.eqw, self.eqw_err = None, None
        self.gauss_flux, self.gauss_err = None, None
        self.cont, self.std_cont =None, None
        self.m_cont, self.n_cont = None, None
        self.amp, self.center, self.sigma = None, None, None
        self.amp_err, self.center_err, self.sigma_err = None, None, None
        self.z_line = None
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

        self.fit_params, self.fit_output = {}, None
        self.pixelWidth = None
        self.temp_line = None
        self._emission_check = True
        self._cont_from_adjacent = True
        self._decimal_wave = False

        return

    def define_masks(self, wavelength_array, masks_array, merge_continua=True, line_mask_entry=None):

        # Make sure it is a matrix
        masks_array = np.array(masks_array, ndmin=2)

        # Check if it is a masked array
        if np.ma.is_masked(wavelength_array):
            wave_arr = wavelength_array.data
        else:
            wave_arr = wavelength_array

        # Remove masked pixels from this function wavelength array
        if line_mask_entry is not None:

            # Convert cfg mask string to limits
            line_mask_limits = format_line_mask_option(line_mask_entry, wave_arr)

            # Get masked indeces
            idcsMask = (wave_arr[:, None] >= line_mask_limits[:, 0]) & (wave_arr[:, None] <= line_mask_limits[:, 1])
            idcsValid = ~idcsMask.sum(axis=1).astype(bool)[:, None]

            # Store the line pixel mask in the log
            self.pixel_mask = line_mask_entry

        else:
            idcsValid = np.ones(wave_arr.size).astype(bool)[:, None]

        # Find indeces for six points in spectrum
        idcsW = np.searchsorted(wave_arr, masks_array)

        # Emission region
        idcsLineRegion = ((wave_arr[idcsW[:, 2]] <= wave_arr[:, None]) & (wave_arr[:, None] <= wave_arr[idcsW[:, 3]]) & idcsValid).squeeze()

        # Return left and right continua merged in one array
        if merge_continua:

            idcsContRegion = (((wave_arr[idcsW[:, 0]] <= wave_arr[:, None]) &
                              (wave_arr[:, None] <= wave_arr[idcsW[:, 1]])) |
                              ((wave_arr[idcsW[:, 4]] <= wave_arr[:, None]) & (
                               wave_arr[:, None] <= wave_arr[idcsW[:, 5]])) & idcsValid).squeeze()



            return idcsLineRegion, idcsContRegion

        # Return left and right continua in separated arrays
        else:

            idcsContLeft = ((wave_arr[idcsW[:, 0]] <= wave_arr[:, None]) & (wave_arr[:, None] <= wave_arr[idcsW[:, 1]]) & idcsValid).squeeze()
            idcsContRight = ((wave_arr[idcsW[:, 4]] <= wave_arr[:, None]) & (wave_arr[:, None] <= wave_arr[idcsW[:, 5]]) & idcsValid).squeeze()

            return idcsLineRegion, idcsContLeft, idcsContRight

    def line_properties(self, emisWave, emisFlux, contWave, contFlux, emisErr = None, bootstrap_size=500):

        # Gradient and interception of linear continuum using adjacent regions
        if self._cont_from_adjacent:
            self.m_cont, self.n_cont, r_value, p_value, std_err = stats.linregress(contWave, contFlux)

        # Using line first and last point
        else:
            w2, w3 = emisWave[0], emisWave[-1]
            f2, f3 = emisFlux[0], emisFlux[-1]
            self.m_cont = (f3 - f2) / (w3 - w2)
            self.n_cont = f3 - self.m_cont * w3

        # Compute continuum
        continuaFit = contWave * self.m_cont + self.n_cont
        lineLinearCont = emisWave * self.m_cont + self.n_cont

        # Line Characteristics
        peakIdx = np.argmax(emisFlux) if self._emission_check else np.argmin(emisFlux)
        self.peak_wave, self.peak_flux = emisWave[peakIdx], emisFlux[peakIdx]
        self.pixelWidth = np.diff(emisWave).mean()
        self.std_cont = np.std(contFlux - continuaFit)
        self.cont = self.peak_wave * self.m_cont + self.n_cont

        # Establish the pixel sigma error
        err_array = self.std_cont if emisErr is None else emisErr

        # Monte Carlo to measure line flux and uncertainty
        normalNoise = np.random.normal(0.0, err_array, (bootstrap_size, emisFlux.size))
        lineFluxMatrix = emisFlux + normalNoise
        areasArray = (lineFluxMatrix.sum(axis=1) - lineLinearCont.sum()) * self.pixelWidth
        self.intg_flux, self.intg_err = areasArray.mean(), areasArray.std()
        self.snr_line = signal_to_noise(self.intg_flux, self.std_cont, emisWave.size)
        self.snr_cont = self.cont/self.std_cont

        # Line width to the pixel below the continuum (or mask size if not happening)
        idx_0 = compute_FWHM0(peakIdx, emisFlux, -1, lineLinearCont, self._emission_check)
        idx_f = compute_FWHM0(peakIdx, emisFlux, 1, lineLinearCont, self._emission_check)
        self.w_i, self.w_f = emisWave[idx_0], emisWave[idx_f]

        # Velocity calculations
        velocArray = c_KMpS * (emisWave[idx_0:idx_f] - self.peak_wave) / self.peak_wave
        self.FWZI = velocArray[-1] - velocArray[0]
        self.velocity_percentiles_calculations(velocArray, emisFlux[idx_0:idx_f], lineLinearCont[idx_0:idx_f])

        # Equivalent width computation
        lineContinuumMatrix = lineLinearCont + normalNoise
        eqwMatrix = areasArray / lineContinuumMatrix.mean(axis=1)
        self.eqw, self.eqw_err = eqwMatrix.mean(), eqwMatrix.std()

        return

    def gauss_lmfit(self, line_label, x, y, weights, user_conf={}, lines_df=None, z_obj=0):

        # Check if line is in a blended group
        line_ref = self.profile_label if self.blended_check else line_label

        # Confirm the number of gaussian components
        mixtureComponents = np.array(line_ref.split('-'), ndmin=1)
        n_comps = mixtureComponents.size

        # TODO maybe we need this operation just once and create self.ion, self.trans_array, self.latex_label
        ion_arr, theoWave_arr, latexLabel_arr = label_decomposition(mixtureComponents, comp_dict=user_conf, units_wave=self.units_wave)

        # Pixel velocity
        self.pixel_vel = c_KMpS * self.pixelWidth/self.peak_wave

        # Compute the line redshift and reference wavelength
        if self.blended_check:
            ref_wave = theoWave_arr * (1 + z_obj)
        else:
            ref_wave = np.array([self.peak_wave], ndmin=1)

        # Line redshift
        # TODO calculate z_line using the label reference
        self.z_line = self.peak_wave/theoWave_arr[0] - 1

        # Kinematics corrections
        self.sigma_instr = k_FWHM/self.inst_FWHM if not np.isnan(self.inst_FWHM) else None

        # Define fitting params for each component
        fit_model = Model(linear_model)
        for idx_comp, comp in enumerate(mixtureComponents):

            # Security check for labels with decimal wavelengths
            comp = comp.replace('.', 'dot')

            # Linear
            if idx_comp == 0:
                fit_model.prefix = f'{comp}_cont_' # For a blended line the continuum conf is defined by first line # TODO this should be the component _b

                SLOPE_PAR = self._SLOPE_PAR if self._cont_from_adjacent else self._SLOPE_FIX_PAR
                INTER_PAR = self._INTER_PAR if self._cont_from_adjacent else self._INTER_FIX_PAR

                self.define_param(fit_model, comp, 'cont_slope', self.m_cont, SLOPE_PAR, user_conf)
                self.define_param(fit_model, comp, 'cont_intercept', self.n_cont, INTER_PAR, user_conf)

            # Gaussian
            fit_model += Model(gaussian_model, prefix=f'{comp}_')

            # Amplitude default configuration changes according and emission or absorption feature
            AMP_PAR = self._AMP_PAR if self._emission_check else self._AMP_ABS_PAR

            # Define the curve parameters # TODO include the normalization here
            self.define_param(fit_model, comp, 'amp', self.peak_flux - self.cont, AMP_PAR, user_conf)
            self.define_param(fit_model, comp, 'center', ref_wave[idx_comp], self._CENTER_PAR, user_conf, z_cor=(1+z_obj))
            self.define_param(fit_model, comp, 'sigma', 2*self.pixelWidth, self._SIG_PAR, user_conf)
            self.define_param(fit_model, comp, 'area', comp, self._AREA_PAR, user_conf)

        # Unpack the mask
        if np.ma.is_masked(x):
            idcs_good = ~x.mask
            x_in = x.data[idcs_good]
            y_in = y.data[idcs_good]
            weights_in = weights[idcs_good]
        else:
            x_in, y_in, weights_in = x, y, weights

        # Fit the line
        self.fit_params = fit_model.make_params()
        self.fit_output = fit_model.fit(y_in, self.fit_params, x=x_in, weights=weights_in, method=self._minimize_method)

        if not self.fit_output.errorbars:
            if self.observations == 'no':
                self.observations = 'No_errorbars'
            else:
                self.observations += 'No_errorbars'
            print(f'-- WARNING: Parameter(s) uncertainty could not be measured for line {line_label}')

        # Generate containers for the results
        eqw_g, eqwErr_g = np.empty(n_comps), np.empty(n_comps)

        self.amp, self.amp_err = np.empty(n_comps), np.empty(n_comps)
        self.center, self.center_err = np.empty(n_comps), np.empty(n_comps)
        self.sigma, self.sigma_err = np.empty(n_comps), np.empty(n_comps)

        self.v_r, self.v_r_err = np.empty(n_comps), np.empty(n_comps)
        self.sigma_vel, self.sigma_vel_err = np.empty(n_comps), np.empty(n_comps)
        self.gauss_flux, self.gauss_err = np.empty(n_comps), np.empty(n_comps)
        self.FWHM_g = np.empty(n_comps)
        self.sigma_thermal = np.empty(n_comps)

        # Fitting diagnostics
        self.chisqr, self.redchi = self.fit_output.chisqr, self.fit_output.redchi
        self.aic, self.bic = self.fit_output.aic, self.fit_output.bic

        # Store lmfit measurements
        for i, comp in enumerate(mixtureComponents):

            # Security check for labels with decimal wavelengths
            comp = comp.replace('.', 'dot')

            # Gaussian parameters
            for j, param in enumerate(['amp', 'center', 'sigma']):
                param_fit = self.fit_output.params[f'{comp}_{param}']
                term_mag = getattr(self, param)
                term_mag[i] = param_fit.value
                term_err = getattr(self, f'{param}_err')
                term_err[i] = param_fit.stderr

                # Case with error propagation from _kinem command
                if (term_err[i] == 0) and (f'{comp}_{param}_err' in user_conf):
                    term_err[i] = user_conf[f'{comp}_{param}_err'] # TODO do I need this one here, can I use the one below

            # Gaussian area
            self.gauss_flux[i] = self.fit_output.params[f'{comp}_area'].value
            self.gauss_err[i] = self.fit_output.params[f'{comp}_area'].stderr

            # Equivalent with gaussian flux for blended components TODO compute self.cont from linear fit
            if self.blended_check:
                eqw_g[i], eqwErr_g[i] = self.gauss_flux[i] / self.cont, self.gauss_err[i] / self.cont

            # Kinematics
            self.v_r[i] = c_KMpS * (self.center[i] - ref_wave[i])/ref_wave[i]
            self.v_r_err[i] = c_KMpS * (self.center_err[i])/ref_wave[i]
            self.sigma_vel[i] = c_KMpS * self.sigma[i]/ref_wave[i]
            self.sigma_vel_err[i] = c_KMpS * self.sigma_err[i]/ref_wave[i]
            self.FWHM_g[i] = k_FWHM * self.sigma_vel[i]
            self.sigma_thermal[i] = np.sqrt(k_Boltzmann * self.temp_line / self._atomic_mass_dict[ion_arr[i][:-1]]) / 1000

            # Check parameters error progragation from the lmfit parameter
            self.error_propagation_check(i, comp)

        if self.blended_check:
            self.eqw, self.eqw_err = eqw_g, eqwErr_g
        else:
            self.eqw, self.eqw_err = np.array(self.eqw, ndmin=1), np.array(self.eqw_err, ndmin=1)

        return

    def define_param(self, model_obj, line_label, param_label, param_value, default_conf={}, user_conf={}, z_cor=1):

        param_ref = f'{line_label}_{param_label}'

        # Overwrite default by the one provided by the user
        if param_ref in user_conf:
            param_conf = {**default_conf, **user_conf[param_ref]}
        else:
            param_conf = default_conf.copy()

        # Set initial value estimation from spectrum if not provided by the user
        if param_ref not in user_conf:
            param_conf['value'] = param_value

        else:
            # Special case inequalities: H1_6563A_w1_sigma = '>1.5*H1_6563A_sigma'
            if param_conf['expr'] is not None:
                if ('<' in param_conf['expr']) or ('>' in param_conf['expr']):

                    # Create additional parameter
                    ineq_name = f'{param_ref}_ineq'
                    ineq_operation = '*' # TODO add remaining operations
                    print('----------------------------------PASAMOS----------------------------')
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
                param_conf['expr'] = f'{param_value}_amp*2.5066282746*{param_value}_sigma'

        # Additional preparation for center parameter: Multiply value, min, max by redshift
        if '_center' in param_ref:
            if param_ref in user_conf:
                param_user_conf = user_conf[param_ref]
                for param_conf_entry in ('value', 'min', 'max'):
                    if param_conf_entry in param_user_conf:
                        param_conf[param_conf_entry] = param_conf[param_conf_entry] * z_cor

        # Assign the parameter configuration to the model
        model_obj.set_param_hint(param_ref, **param_conf)

        return

    def error_propagation_check(self, idx_line, line_label):

        # Check gaussian flux error
        if (self.gauss_err[idx_line] == 0.0) and (self.amp_err[idx_line] != 0.0) and (self.sigma_err[idx_line] != 0.0):
            self.gauss_err[idx_line] = mult_err_propagation(np.array([self.amp[idx_line], self.sigma[idx_line]]),
                                                            np.array([self.amp_err[idx_line], self.sigma_err[idx_line]]),
                                                            self.gauss_flux[idx_line])

        # Check equivalent width error
        # for param in


        # Check the error from the _kinem command imports

        return

    def velocity_percentiles_calculations(self, vel_array, line_flux, cont_flux, min_array_dim=15):

        # Only compute the velocity percentiles for line bands with more than 15 pixels
        valid_pixels = vel_array.size if not np.ma.is_masked(vel_array) else np.sum(~vel_array.mask)

        if valid_pixels > min_array_dim:

            peakIdx = np.argmax(line_flux)

            percentFluxArray = np.cumsum(line_flux-cont_flux) * self.pixelWidth / self.intg_flux * 100

            if self._emission_check:
                blue_range = line_flux[:peakIdx] > self.peak_flux/2
                red_range = line_flux[peakIdx:] < self.peak_flux/2
            else:
                blue_range = line_flux[:peakIdx] < self.peak_flux/2
                red_range = line_flux[peakIdx:] > self.peak_flux/2

            # In case the peak is at the edge
            if (blue_range.size > 2) and (red_range.size > 2):

                vel_FWHM_blue = vel_array[:peakIdx][np.argmax(blue_range)]
                vel_FWHM_red = vel_array[peakIdx:][np.argmax(red_range)]
                self.FWHM_intg = vel_FWHM_red - vel_FWHM_blue

                # Interpolation for integrated kinematics
                percentInterp = interp1d(percentFluxArray, vel_array, kind='slinear', fill_value='extrapolate')
                velocPercent = percentInterp(TARGET_PERCENTILES)

                self.v_med, self.v_50 = np.median(vel_array), velocPercent[3] # FIXME np.median ignores the mask
                self.v_5, self.v_10 = velocPercent[1], velocPercent[2]
                self.v_90, self.v_95 = velocPercent[4], velocPercent[5]

                W_80 = self.v_90 - self.v_10
                W_90 = self.v_95 - self.v_5
                A_factor = ((self.v_90 - self.v_med) - (self.v_med-self.v_10)) / W_80
                K_factor = W_90 / (1.397 * self.FWHM_intg)

        return



