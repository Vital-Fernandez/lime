import numpy as np
from lmfit.models import Model
from scipy import stats, optimize
from scipy.interpolate import interp1d
from .tools import label_decomposition

c_KMpS = 299792.458  # Speed of light in Km/s (https://en.wikipedia.org/wiki/Speed_of_light)

k_GaussArea = np.sqrt(2 * np.pi)

k_FWHM = 2 * np.sqrt(2 * np.log(2))

TARGET_PERCENTILES = np.array([2, 5, 10, 50, 90, 95, 98])


def wavelength_to_vel(delta_lambda, lambda_wave, light_speed=c_KMpS):
    return light_speed * (delta_lambda/lambda_wave)


def iraf_snr(input_y):
    avg = np.average(input_y)
    rms = np.sqrt(np.mean(np.square(input_y - avg)))
    snr = avg/rms
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


def linear_model(x, slope, intercept):
    """a line"""
    return slope * x + intercept


def is_digit(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


class EmissionFitting:

    """Class to to measure emission line fluxes and fit them as gaussian curves"""

    _AMP_PAR = dict(value=None, min=0, max=np.inf, vary=True, expr=None)
    _CENTER_PAR = dict(value=None, min=-np.inf, max=np.inf, vary=True, expr=None)
    _SIG_PAR = dict(value=None, min=0, max=np.inf, vary=True, expr=None)
    _AREA_PAR = dict(value=None, min=-np.inf, max=np.inf, vary=True, expr=None)
    # _AREA_PAR = dict(value=None, min=0, max=np.inf, vary=True, expr=None)

    _AMP_ABS_PAR = dict(value=None, min=-np.inf, max=0, vary=True, expr=None)

    _SLOPE_PAR = dict(value=None, min=-np.inf, max=np.inf, vary=True, expr=None)
    _INTER_PAR = dict(value=None, min=-np.inf, max=np.inf, vary=True, expr=None)

    _SLOPE_FIX_PAR = dict(value=None, min=-np.inf, max=np.inf, vary=False, expr=None)
    _INTER_FIX_PAR = dict(value=None, min=-np.inf, max=np.inf, vary=False, expr=None)

    _minimize_method = 'leastsq'

    # Switch for emission and absorption lines

    def __init__(self):

        self.line, self.lineWaves = '', np.array([np.nan] * 6)
        self.blended_check, self.blended_label = False, 'None'

        self.intg_flux, self.intg_err = None, None
        self.peak_wave, self.peak_flux = None, None
        self.eqw, self.eqw_err = None, None
        self.gauss_flux, self.gauss_err = None, None
        self.cont, self.std_cont =None, None
        self.m_cont, self.n_cont = None, None
        self.amp, self.center, self.sigma = None, None, None
        self.amp_err, self.center_err, self.sigma_err = None, None, None
        self.z_line = 0
        self.v_r, self.v_r_err = None, None
        self.sigma_vel, self.sigma_vel_err = None, None
        self.snr_line, self.snr_cont = None, None
        self.observations, self.comments = '', 'None'
        self.FWHM_int, self.FWHM_g = None, None
        self.v_med, self.v_50 = None, None
        self.v_5, self.v_10 = None, None
        self.v_90, self.v_95 = None, None
        self.chisqr, self.redchi = None, None
        self.aic, self.bic = None, None

        self.fit_params, self.fit_output = {}, None
        self.pixelWidth = None
        self._emission_check = True
        self._cont_from_adjacent = True

        return

    def define_masks(self, wave_arr, flux_arr, masks_array, merge_continua=True):

        # Make sure it is a matrix
        masks_array = np.array(masks_array, ndmin=2)

        # Find indeces for six points in spectrum
        idcsW = np.searchsorted(wave_arr, masks_array)

        # Emission region
        idcsLineRegion = ((wave_arr[idcsW[:, 2]] <= wave_arr[:, None]) &
                          (wave_arr[:, None] <= wave_arr[idcsW[:, 3]])).squeeze()


        # Return left and right continua merged in one array
        # TODO add check for wavelengths beyond wavelengh limits
        if merge_continua:

            idcsContRegion = (((wave_arr[idcsW[:, 0]] <= wave_arr[:, None]) &
                              (wave_arr[:, None] <= wave_arr[idcsW[:, 1]])) |
                              ((wave_arr[idcsW[:, 4]] <= wave_arr[:, None]) & (
                               wave_arr[:, None] <= wave_arr[idcsW[:, 5]]))).squeeze()

            return idcsLineRegion, idcsContRegion

        # Return left and right continua in separated arrays
        else:

            idcsContLeft = ((wave_arr[idcsW[:, 0]] <= wave_arr[:, None]) & (wave_arr[:, None] <= wave_arr[idcsW[:, 1]])).squeeze()
            idcsContRight = ((wave_arr[idcsW[:, 4]] <= wave_arr[:, None]) & (wave_arr[:, None] <= wave_arr[idcsW[:, 5]])).squeeze()

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
        self.snr_line, self.snr_cont = iraf_snr(emisFlux), iraf_snr(contFlux)

        # Establish the pixel sigma error
        err_array = self.std_cont if emisErr is None else emisErr

        # Monte Carlo to measure line flux and uncertainty
        normalNoise = np.random.normal(0.0, err_array, (bootstrap_size, emisFlux.size))
        lineFluxMatrix = emisFlux + normalNoise
        areasArray = (lineFluxMatrix.sum(axis=1) - lineLinearCont.sum()) * self.pixelWidth
        self.intg_flux, self.intg_err = areasArray.mean(), areasArray.std()

        # Velocity calculations
        if emisWave.size > 15:

            velocArray = c_KMpS * (emisWave - self.peak_wave)/self.peak_wave
            percentFluxArray = np.cumsum(emisFlux-lineLinearCont) * self.pixelWidth / self.intg_flux * 100

            if self._emission_check:
                blue_range = emisFlux[:peakIdx] > self.peak_flux/2
                red_range = emisFlux[peakIdx:] < self.peak_flux/2
            else:
                blue_range = emisFlux[:peakIdx] < self.peak_flux/2
                red_range = emisFlux[peakIdx:] > self.peak_flux/2

            # In case the peak is at the edge
            if (blue_range.size > 2) and (red_range.size > 2):

                vel_FWHM_blue = velocArray[:peakIdx][np.argmax(blue_range)]
                vel_FWHM_red = velocArray[peakIdx:][np.argmax(red_range)]
                self.FWHM_int = vel_FWHM_red - vel_FWHM_blue

                # Interpolation for integrated kinematics
                percentInterp = interp1d(percentFluxArray, velocArray, kind='slinear', fill_value='extrapolate')
                velocPercent = percentInterp(TARGET_PERCENTILES)

                self.v_med, self.v_50 = np.median(velocArray), velocPercent[3]
                self.v_5, self.v_10 = velocPercent[1], velocPercent[2]
                self.v_90, self.v_95 = velocPercent[4], velocPercent[5]

                W_80 = self.v_90 - self.v_10
                W_90 = self.v_95 - self.v_5
                A = ((self.v_90 - self.v_med) - (self.v_med-self.v_10)) / W_80
                K = W_90 / (1.397 * self.FWHM_int)

        # TODO apply the redshift correction
        # Equivalent width computation
        lineContinuumMatrix = lineLinearCont + normalNoise
        eqwMatrix = areasArray / lineContinuumMatrix.mean(axis=1)
        self.eqw, self.eqw_err = eqwMatrix.mean(), eqwMatrix.std()

        return

    def gauss_lmfit(self, line_label, x, y, weights, user_conf={}, lines_df=None, z_obj=0):

        # Check if line is in a blended group
        line_ref = self.blended_label if self.blended_check else line_label

        # Confirm the number of gaussian components
        mixtureComponents = np.array(line_ref.split('-'), ndmin=1)
        n_comps = mixtureComponents.size
        ion_arr, theoWave_arr, latexLabel_arr = label_decomposition(mixtureComponents, comp_dict=user_conf)

        # Compute the line redshift and reference wavelength
        if self.blended_check:
            ref_wave = theoWave_arr * (1 + z_obj)
        else:
            ref_wave = np.array([self.peak_wave], ndmin=1)

        # Define fitting params for each component
        fit_model = Model(linear_model)
        for idx_comp, comp in enumerate(mixtureComponents):

            # Linear
            if idx_comp == 0:
                fit_model.prefix = f'{comp}_cont_' # For a blended line the continuum conf is defined by first line

                SLOPE_PAR = self._SLOPE_PAR if self._cont_from_adjacent else self._SLOPE_FIX_PAR
                INTER_PAR = self._INTER_PAR if self._cont_from_adjacent else self._INTER_FIX_PAR

                self.define_param(fit_model, comp, 'cont_slope', self.m_cont, SLOPE_PAR, user_conf)
                self.define_param(fit_model, comp, 'cont_intercept', self.n_cont, INTER_PAR, user_conf)

            # Gaussian
            fit_model += Model(gaussian_model, prefix=f'{comp}_')

            # Amplitude default configuration changes according and emission or absorption feature
            AMP_PAR = self._AMP_PAR if self._emission_check else self._AMP_ABS_PAR

            # Define the curve parameters
            self.define_param(fit_model, comp, 'amp', self.peak_flux - self.cont, AMP_PAR, user_conf)
            self.define_param(fit_model, comp, 'center', ref_wave[idx_comp], self._CENTER_PAR, user_conf, z_cor=(1+z_obj))
            self.define_param(fit_model, comp, 'sigma', 1.0, self._SIG_PAR, user_conf)
            self.define_param(fit_model, comp, 'area', comp, self._AREA_PAR, user_conf)

        # Fit the line
        self.fit_params = fit_model.make_params()
        self.fit_output = fit_model.fit(y, self.fit_params, x=x, weights=weights, method=self._minimize_method)

        if not self.fit_output.errorbars:
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
        self.z_line = np.empty(n_comps)
        self.FWHM_g = np.empty(n_comps)

        # Fitting diagnostics
        self.chisqr, self.redchi = self.fit_output.chisqr, self.fit_output.redchi
        self.aic, self.bic = self.fit_output.aic, self.fit_output.bic

        # Store lmfit measurements
        for i, line in enumerate(mixtureComponents):

            # Gaussian parameters
            for j, param in enumerate(['amp', 'center', 'sigma']):
                param_fit = self.fit_output.params[f'{line}_{param}']
                term_mag = getattr(self, param)
                term_mag[i] = param_fit.value
                term_err = getattr(self, f'{param}_err')
                term_err[i] = param_fit.stderr

                # Case with error propagation
                if (term_err[i] == 0) and (f'{line}_{param}_err' in user_conf):
                    term_err[i] = user_conf[f'{line}_{param}_err']

            # Compute line location
            self.z_line[i] = self.center[i]/theoWave_arr[i] - 1

            # Gaussian area
            self.gauss_flux[i] = self.fit_output.params[f'{line}_area'].value
            self.gauss_err[i] = self.fit_output.params[f'{line}_area'].stderr

            # Equivalent with gaussian flux for blended components TODO compute self.cont from linear fit
            if self.blended_check:
                eqw_g[i], eqwErr_g[i] = self.gauss_flux[i] / self.cont, self.gauss_err[i] / self.cont

            # Kinematics
            self.v_r[i] = c_KMpS * (self.center[i] - ref_wave[i])/ref_wave[i] # wavelength_to_vel(self.center[i] - theoWave_arr[i], theoWave_arr[i])#self.v_r[i] =
            self.v_r_err[i] = c_KMpS * (self.center_err[i])/ref_wave[i] # np.abs(wavelength_to_vel(self.center_err[i], theoWave_arr[i]))
            self.sigma_vel[i] = c_KMpS * self.sigma[i]/ref_wave[i] # wavelength_to_vel(self.sigma[i], theoWave_arr[i])
            self.sigma_vel_err[i] = c_KMpS * self.sigma_err[i]/ref_wave[i] # wavelength_to_vel(self.sigma_err[i], theoWave_arr[i])
            self.FWHM_g[i] = k_FWHM * self.sigma_vel[i]

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

    def gauss_mcfit(self, wave, flux, idcs_line, bootstrap_size=1000):

        # Get regions data
        lineWave, lineFlux = wave[idcs_line], flux[idcs_line]

        # Linear continuum linear fit
        lineContFit = lineWave * self.m_cont + self.n_cont

        # Initial gaussian fit values
        p0_array = np.array([self.peak_flux, self.peak_wave, 1])

        # Monte Carlo to fit gaussian curve
        normalNoise = np.random.normal(0.0, self.std_cont, (bootstrap_size, lineWave.size))
        lineFluxMatrix = lineFlux + normalNoise

        # Run the fitting
        try:
            p1_matrix = np.empty((bootstrap_size, 3))

            for i in np.arange(bootstrap_size):
                p1_matrix[i], pcov = optimize.curve_fit(gauss_func,
                                                        (lineWave, lineContFit),
                                                        lineFluxMatrix[i],
                                                        p0=p0_array,
                                                        # ftol=0.5,
                                                        # xtol=0.5,
                                                        # bounds=paramBounds,
                                                        maxfev=1200)

            p1, p1_Err = p1_matrix.mean(axis=0), p1_matrix.std(axis=0)

            lineArea = np.sqrt(2 * np.pi * p1_matrix[:, 2] * p1_matrix[:, 2]) * p1_matrix[:, 0]
            y_gauss, y_gaussErr = lineArea.mean(), lineArea.std()

        except:
            p1, p1_Err = np.array([np.nan, np.nan, np.nan]), np.array([np.nan, np.nan, np.nan])
            y_gauss, y_gaussErr = np.nan, np.nan

        self.p1, self.p1_Err, self.gauss_flux, self.gauss_err = p1, p1_Err, y_gauss, y_gaussErr

        return


