import logging
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

from ..io import LiMe_Error
from lime.fitting.lines import compute_inst_sigma_array, gaussian_model
from ..plotting.plots import redshift_key_evaluation, redshift_permu_evaluation

try:
    import aspect
    aspect_check = True
except ImportError:
    aspect_check = False

_logger = logging.getLogger('LiMe')


def compute_gaussian_ridges(redshift, lines_lambda, wave_matrix, amp_arr, sigma_arr):

    # Compute the observed line wavelengths
    obs_lambda = lines_lambda * (1 + redshift)
    obs_lambda = obs_lambda[(obs_lambda > wave_matrix[0, 0]) & (obs_lambda < wave_matrix[0, -1])]

    if obs_lambda.size > 0:

        # Compute indexes ion array
        idcs_obs = np.searchsorted(wave_matrix[0, :], obs_lambda)

        # Compute lambda arrays:
        sigma_lines = sigma_arr[idcs_obs]
        mu_lines = wave_matrix[0, :][idcs_obs]

        # Compute the Gaussian bands
        x_matrix = wave_matrix[:idcs_obs.size, :]
        gauss_matrix = gaussian_model(x_matrix, amp_arr, mu_lines[:, None], sigma_lines[:, None])
        gauss_arr = gauss_matrix.sum(axis=0)

        # Set maximum to 1:
        idcs_one = gauss_arr > 1
        gauss_arr[idcs_one] = 1

    else:
        gauss_arr = None

    return gauss_arr

def redshift_xor_method(spec, bands, z_min, z_max, z_nsteps, pred_arr, components_number, res_power, sigma_factor,
                        sig_digits=2, plot_results=False):

    # Use the detection bands if provided
    if (pred_arr is not None) and (components_number is not None):
        idcs_lines = np.isin(pred_arr, components_number)
    else:
        idcs_lines = None

    # Continue with measurement
    if idcs_lines is not None:

        # Extract the data
        pixel_mask = spec.flux.mask
        wave_arr = spec.wave.data
        flux_arr = spec.flux.data

        # Loop throught the redshift steps
        if not np.all(pixel_mask):

            # Compute the resolution params
            sigma_arr = compute_inst_sigma_array(wave_arr, res_power)
            sigma_arr = sigma_arr if sigma_factor is None else sigma_arr * sigma_factor

            # Lines selection
            theo_lambda = bands.wavelength.to_numpy()

            # Parameters for the brute analysis
            z_arr = np.linspace(z_min, z_max, z_nsteps)
            wave_matrix = np.tile(wave_arr, (theo_lambda.size, 1))
            xor_sum = np.zeros(z_arr.size)

            # Invert the mask
            data_mask = ~pixel_mask

            # Revert the data
            for i, z_i in enumerate(z_arr):
                gauss_arr = compute_gaussian_ridges(z_i, theo_lambda, wave_matrix, 1, sigma_arr)
                xor_sum[i] = 0 if gauss_arr is None else np.sum(idcs_lines[data_mask] * gauss_arr[data_mask])

            z_infer = np.round(z_arr[np.argmax(xor_sum)], decimals=sig_digits)

        # No lines or all masked
        else:
            z_infer = None

        if plot_results and (z_infer is not None):
            gauss_arr_max = compute_gaussian_ridges(z_infer, theo_lambda, wave_matrix, 1, sigma_arr)
            redshift_key_evaluation(spec, z_infer, data_mask, gauss_arr_max, z_arr, xor_sum)

    # Do not attempt measurement
    else:
        z_infer = None

    return z_infer


def redshift_key_method(spec, bands, z_min, z_max, z_nsteps, pred_arr, components_number, res_power, sigma_factor,
                        sig_digits=2, detection_only=True, plot_results=False):

    # Use the detection bands if provided
    if (pred_arr is not None) and (components_number is not None):
        idcs_lines = np.isin(pred_arr, components_number)
    else:
        idcs_lines = None

    # Decide if proceed
    if detection_only is False:
        measure_check = True
        idcs_lines = np.ones(idcs_lines.shape).astype(bool)
    else:
        measure_check = True if np.any(idcs_lines) else False

    # Continue with measurement
    if measure_check:

        # Extract the data
        pixel_mask = spec.flux.mask
        wave_arr = spec.wave.data
        flux_arr = spec.flux.data
        data_mask = ~pixel_mask

        # Compute the resolution params
        sigma_arr = compute_inst_sigma_array(wave_arr, res_power)
        sigma_arr = sigma_arr if sigma_factor is None else sigma_arr * sigma_factor

        # Lines selection
        theo_lambda = bands.wavelength.to_numpy()

        # Parameters for the brute analysis
        z_arr = np.linspace(z_min, z_max, z_nsteps)
        wave_matrix = np.tile(wave_arr, (theo_lambda.size, 1))
        flux_sum = np.zeros(z_arr.size)

        # Combine line and pixel_mask
        mask = data_mask & idcs_lines

        # Loop throught the redshift steps
        if not np.all(~mask):
            for i, z_i in enumerate(z_arr):
                # Generate the redshift key
                gauss_arr = compute_gaussian_ridges(z_i, theo_lambda, wave_matrix, 1, sigma_arr)

                # Compute flux cumulative sum
                flux_sum[i] = 0 if gauss_arr is None else np.sum(flux_arr[mask] * gauss_arr[mask])

            z_infer = np.round(z_arr[np.argmax(flux_sum)], decimals=sig_digits)

        # No lines or all masked
        else:
            z_infer = None

        if plot_results and (z_infer is not None):
            gauss_arr_max = compute_gaussian_ridges(z_infer, theo_lambda, wave_matrix, 1, sigma_arr)
            redshift_key_evaluation(spec, z_infer, mask, gauss_arr_max, z_arr, flux_sum)

    # Do not attempt measurement
    else:
        z_infer = None

    return z_infer

def permutation_objective_function(redshift, obs_arr, theo_arr):

    adjusted_observed = obs_arr / (1 + redshift)
    cost_matrix = np.abs(adjusted_observed[:, None] - theo_arr[None, :])

    # Find the best matching subset using linear sum assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    residual = np.sum(cost_matrix[row_ind, col_ind])

def permutation_residual(redshift, obs_arr, theo_arr):

    return permutation_objective_function(redshift, obs_arr, theo_arr)


# Residual computation function
def compute_residual(Z, observed, theoretical):
    """
    Computes the residual for a given redshift Z.

    Parameters:
    - Z: Redshift value.
    - observed: Observed transitions (array-like).
    - theoretical: Theoretical transitions (array-like).

    Returns:
    - Residual value (float).
    """
    adjusted_observed = observed / (1 + Z)
    cost_matrix = np.abs(adjusted_observed[:, None] - theoretical[None, :])

    # Find the best matching subset using linear sum assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    residual = np.sum(cost_matrix[row_ind, col_ind])

    return residual

def redshift_permutation_method(spec, bands, z_min, z_max, pred_arr, components_number, plot_results):

    # Use the detection array
    if (pred_arr is not None) and (components_number is not None):
        idcs_lines = np.isin(pred_arr, components_number)
    else:
        idcs_lines = None

    # Decide if proceed
    measure_check = True if np.any(idcs_lines) else False
    if measure_check:

        # Identify where changes occur (edges of ones and zeros)
        edges = np.diff(np.concatenate(([0], idcs_lines, [0])))
        start_indices = np.where(edges == 1)[0]
        end_indices = np.where(edges == -1)[0] - 1

        # Calculate observed wavelengths
        central_indices = [(start + end) // 2 for start, end in zip(start_indices, end_indices)]
        pixel_mask = spec.flux.mask
        data_mask = ~pixel_mask
        wave_arr = spec.wave.data if pixel_mask is not None else spec.wave
        wave_obs = wave_arr[central_indices]

        # Calculate theo wavelengths
        wave_theo = bands.wavelength.to_numpy()

        # Run the permutation
        # result = minimize(permutation_residual, x0=5, bounds=[(z_min, z_max)])
        result = minimize(lambda Z: compute_residual(Z, wave_obs, wave_theo), x0=1, bounds=[(z_min, z_max)],
                          method='dogleg')
        z_infer = result.x[0]

        # Recompute cost matrix and find the best matching subset
        adjusted_observed = wave_obs / (1 + z_infer)
        cost_matrix = np.abs(adjusted_observed[:, None] - wave_theo[None, :])
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Extract the best matching subset of theoretical transitions
        best_matching_subset = wave_theo[col_ind]

        if plot_results:
            idcs_theo = (wave_theo * (1 + z_infer) >= wave_arr[0]) & (wave_theo * (1 + z_infer) <= wave_arr[-1])
            redshift_permu_evaluation(spec, z_infer, wave_obs, wave_theo[idcs_theo] * (1 + z_infer))


    else:
        z_infer = None

    return z_infer

class RedshiftFitting:

    def __init__(self):

        return

    def redshift(self, bands, z_min=0, z_max=10, z_nsteps=2000,  mode='key', res_power=None, detection_only=True,
                 components=None, sigma_factor=None, sig_digits=2, plot_results=False):

        '''
        bands, z_min, z_max, z_nsteps, idcs_lines, res_power, sigma_factor, sig_digits=2,
                                detection_only=True, plot_results=False
        '''

        # Limits
        z_min = 0 if z_min is None else z_min
        z_max = 12 if z_max is None else z_max

        # Check that ASPECT is available
        if not aspect_check:
            _logger.info("ASPECT has not been installed the redshift measurements won't be constrained to lines")

        # Get the features array
        pred_arr, conf_arr = None, None
        if aspect_check:
            if self._spec.infer.pred_arr is None:
                _logger.warning("The observation does not have a components detection array please run ASPECT")
            else:
                pred_arr, conf_arr = self._spec.infer.pred_arr, self._spec.infer.conf_arr

        # Resolving power # TODO this should be read at another point...
        res_power = self._spec.res_power if res_power is None else res_power

        # Set the type of fitting and the components to use
        if mode == 'key':
            components = components if components is not None else ['emission', 'doublet']
            components_number = np.array([aspect.cfg['shape_number'][comp] for comp in components])
            z_infer = redshift_key_method(self._spec, bands, z_min, z_max, z_nsteps, pred_arr, components_number, res_power, sigma_factor,
                                          sig_digits=sig_digits, detection_only=detection_only, plot_results=plot_results)

        elif mode == 'permute':
            components = components if components is not None else ['emission', 'doublet', 'absorption']
            components_number = np.array([aspect.cfg['shape_number'][comp] for comp in components])
            z_infer = redshift_permutation_method(self._spec, bands, z_min, z_max, pred_arr, components_number,
                                                  plot_results=plot_results)

        elif mode == 'xor':
            components = components if components is not None else ['emission', 'doublet', 'absorption']
            components_number = np.array([aspect.cfg['shape_number'][comp] for comp in components])
            z_infer = redshift_xor_method(self._spec, bands, z_min, z_max, z_nsteps, pred_arr, components_number, res_power, sigma_factor,
                                          sig_digits=sig_digits, plot_results=plot_results)

        else:
            _logger.critical(f'Input redshift technique "{mode}" is not recognized, please use: "key" or "permute"')
            raise LiMe_Error


        return z_infer