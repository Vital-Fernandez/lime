import logging

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

from lime.io import LiMe_Error
from lime.fitting.lines import compute_inst_sigma_array, gaussian_model
from lime.plotting.plots import redshift_key_evaluation, redshift_permu_evaluation
from lime.tools import res_power_approx

try:
    import aspect
    aspect_check = True
except ImportError:
    aspect_check = False

_logger = logging.getLogger('LiMe')

c_KMpS = 299792.458
k_gFWHM = 2 * np.sqrt(2 * np.log(2))


def comp_counter(arr_mask: np.ndarray) -> int:
    return np.sum(~arr_mask[:-1] & arr_mask[1:]) + arr_mask[0]


def compute_gaussian_ridges_orig(redshift, lines_lambda, wave_matrix, amp_arr, band_vsigma, resol_arr):

    # Compute the observed line wavelengths
    obs_lambda = lines_lambda * (1 + redshift)
    obs_lambda = obs_lambda[(obs_lambda > wave_matrix[0, 0]) & (obs_lambda < wave_matrix[0, -1])]

    if obs_lambda.size > 1:

        # Compute Gaussian centroids
        idcs_obs = np.searchsorted(wave_matrix[0, :], obs_lambda)
        mu_lines = wave_matrix[0, :][idcs_obs]

        # Compute Gaussian sigmas
        sigma_lines = mu_lines * (band_vsigma / c_KMpS) + mu_lines / (resol_arr[idcs_obs] * k_gFWHM)

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



def compute_gaussian_ridges(redshift, lines_lambda, wave_matrix, amp_arr, band_vsigma, resol_arr, Rmin_arr=None):

    # Compute the observed line wavelengths
    obs_lambda = lines_lambda * (1 + redshift)
    idcs_range = (obs_lambda > wave_matrix[0, 0]) & (obs_lambda < wave_matrix[0, -1])
    obs_lambda = obs_lambda[idcs_range]

    # Compute the line pixel locations
    idcs_obs = np.searchsorted(wave_matrix[0, :], obs_lambda)

    # Exclude lines whose observed resolving power is below their minimum requirement
    if Rmin_arr is not None:
        idcs_res = resol_arr[idcs_obs] >= Rmin_arr[idcs_range]
        idcs_obs = idcs_obs[idcs_res]

    if idcs_obs.size > 1:

        # Compute Gaussian centroids
        mu_lines = wave_matrix[0, :][idcs_obs]

        # Compute Gaussian sigmas
        sigma_lines = mu_lines * (band_vsigma / c_KMpS) + mu_lines / (resol_arr[idcs_obs] * k_gFWHM)

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

# def redshift_xor_method(spec, bands, z_min, z_max, z_nsteps, pred_arr, components_number, res_power, sigma_factor,
#                         sig_digits=2, plot_results=False):
#
#     # Use the detection bands if provided
#     if (pred_arr is not None) and (components_number is not None):
#         idcs_lines = np.isin(pred_arr, components_number)
#     else:
#         idcs_lines = None
#
#     # Continue with measurement
#     if idcs_lines is not None:
#
#         # Extract the data
#         pixel_mask = spec.flux.mask
#         wave_arr = spec.wave.data
#         flux_arr = spec.flux.data
#
#         # Loop throught the redshift steps
#         if not np.all(pixel_mask):
#
#             # Compute the resolution params
#             sigma_arr = compute_inst_sigma_array(wave_arr, res_power)
#             sigma_arr = sigma_arr if sigma_factor is None else sigma_arr * sigma_factor
#
#             # Lines selection
#             theo_lambda = bands.wavelength.to_numpy()
#
#             # Parameters for the brute analysis
#             z_arr = np.linspace(z_min, z_max, z_nsteps)
#             wave_matrix = np.tile(wave_arr, (theo_lambda.size, 1))
#             xor_sum = np.zeros(z_arr.size)
#
#             # Invert the mask
#             data_mask = ~pixel_mask
#
#             # Revert the data
#             for i, z_i in enumerate(z_arr):
#                 gauss_arr = compute_gaussian_ridges(z_i, theo_lambda, wave_matrix, 1, sigma_arr)
#                 xor_sum[i] = 0 if gauss_arr is None else np.sum(idcs_lines[data_mask] * gauss_arr[data_mask])
#
#             z_infer = np.round(z_arr[np.argmax(xor_sum)], decimals=sig_digits)
#
#         # No lines or all masked
#         else:
#             z_infer = None
#
#         if plot_results and (z_infer is not None):
#             gauss_arr_max = compute_gaussian_ridges(z_infer, theo_lambda, wave_matrix, 1, sigma_arr)
#             redshift_key_evaluation(spec, z_infer, data_mask, gauss_arr_max, z_arr, xor_sum)
#
#     # Do not attempt measurement
#     else:
#         z_infer = None
#
#     return z_infer

def compile_Rmin_arr(bands, map_bands_Rname=None):

    # No resolution constraints
    if map_bands_Rname is None:
        return None

    # Default entry (0) means no minimum resolving power requirement for that line
    Rmin_arr = np.zeros(bands.index.size)

    for R_min, name_list in map_bands_Rname.items():

        # Locate the named lines as row positions in the bands table
        idcs_lines = bands.index.get_indexer(name_list)

        # Check the constrained lines are present in the bands table
        idcs_missing = idcs_lines == -1
        if idcs_missing.any():
            missing = np.asarray(name_list)[idcs_missing]
            raise ValueError(f'map_bands_Rname lines {missing} not found in bands table')

        # Keep the strictest minimum if a line appears in several entries
        Rmin_arr[idcs_lines] = np.maximum(Rmin_arr[idcs_lines], R_min)

    return Rmin_arr


def redshift_key_method(spec, bands, z_min, z_max, delta_z, pred_arr, components_number, band_vsigma,
                        method, map_band_R=None, sig_digits=2, detection_only=True, plot_results=False):

    # Use the detection bands if provided
    if (pred_arr is not None) and (components_number is not None):
        idcs_lines = np.isin(pred_arr, components_number)
    else:
        idcs_lines = None
        _logger.warning('The input spectrum does not have a components array from a previuos detection')

    # Continue with measurement
    z_infer_flux = None
    z_infer_pixel = None
    if idcs_lines is not None:

        # If there is only one line return nan
        match comp_counter(idcs_lines):
            case 0:
                return None, None # No components
            case 1:
                return np.nan, np.nan # Only one line

        # Extract the data
        wave_arr = spec.wave.data
        flux_arr = spec.flux.data

        # Compute the resolving power if necessary
        res_power = spec.res_power if spec.res_power is not None else res_power_approx(wave_arr)

        # Lines selection
        theo_lambda = bands.wavelength.to_numpy()

        # Compute the redshift range
        if delta_z is None:
            delta_arr = np.diff(wave_arr)
            delta_z = np.median(delta_arr)/np.median(wave_arr)
        z_arr = np.arange(z_min, z_max + 0.5 * delta_z, delta_z)

        # Parameters for the brute analysis
        wave_matrix = np.tile(wave_arr, (theo_lambda.size, 1))
        flux_sum = np.zeros(z_arr.size)
        pixel_count = np.zeros(z_arr.size)

        # Combine line and pixel_mask
        mask = ~spec.flux.mask & idcs_lines

        # Minimim R dictionary
        map_index_R = compile_Rmin_arr(bands, map_band_R)

        # Loop through the redshift steps
        for i, z_i in enumerate(z_arr):

            # Generate the redshift key
            gauss_arr = compute_gaussian_ridges(z_i, theo_lambda, wave_matrix, 1, band_vsigma, res_power, Rmin_arr=map_index_R)

            # Null gauss case
            if gauss_arr is None:
                flux_sum[i] = 0
                pixel_count[i] = 0

            # Compute cumulative flux or pixel-number sum
            else:
                # Check more than one line
                if comp_counter((gauss_arr * mask) > 0.001) >= 2:
                    flux_sum[i] = np.sum(flux_arr[mask] * gauss_arr[mask])
                    pixel_count[i] = np.sum(idcs_lines[mask] * gauss_arr[mask])

        z_infer_flux = np.round(z_arr[np.argmax(flux_sum)], decimals=sig_digits)
        z_infer_pixel = np.round(z_arr[np.argmax(pixel_count)], decimals=sig_digits)

    if plot_results and (z_infer_flux is not None):
        gauss_arr_max = compute_gaussian_ridges(z_infer_pixel, theo_lambda, wave_matrix, 1, band_vsigma, res_power, Rmin_arr=map_index_R)
        redshift_key_evaluation(spec, method, z_infer_flux, mask, gauss_arr_max, z_arr, flux_sum, theo_lambda)

    return z_infer_flux, z_infer_pixel


def permutation_objective_function(redshift, obs_arr, theo_arr):

    adjusted_observed = obs_arr / (1 + redshift)
    cost_matrix = np.abs(adjusted_observed[:, None] - theo_arr[None, :])

    # Find the best matching subset using linear sum assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    residual = np.sum(cost_matrix[row_ind, col_ind])

    return


def permutation_residual(redshift, obs_arr, theo_arr):

    return permutation_objective_function(redshift, obs_arr, theo_arr)


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
        result = minimize(lambda Z: compute_residual(Z, wave_obs, wave_theo), x0=0.01, bounds=[(z_min, z_max)],
                          method='L-BFGS-B')
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

    def redshift(self, bands, z_min=0, z_max=12, delta_z=None,  mode='key', comps_list=['emission', 'doublet-em'],
                 res_power=None, detection_only=True, band_vsigma=70, map_min_R=None, sig_digits=2, plot_results=False):

        '''
        bands, z_min, z_max, z_nsteps, idcs_lines, res_power, sigma_factor, sig_digits=2,
                                detection_only=True, plot_results=False
        '''

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

        # Get the reference for the components
        components_number = np.empty(len(comps_list)).astype(int)
        for i, comp in enumerate(comps_list):
            components_number[i] = aspect.cfg['shape_number'][comp]

        # Set the type of fitting and the components to use
        match mode:
            case 'key':
                z_flux, z_xor = redshift_key_method(self._spec, bands, z_min, z_max, delta_z, pred_arr, components_number,
                                                    band_vsigma, mode, map_band_R=map_min_R, sig_digits=sig_digits,
                                                    detection_only=detection_only, plot_results=plot_results)
            case 'permute':
                z_flux = redshift_permutation_method(self._spec, bands, z_min, z_max, pred_arr, components_number,
                                                      plot_results=plot_results)
                z_xor = None
            case _:
                raise KeyError(f'Input redshift fitting technique "{mode}" is not recognized, please use: '
                                 f'"key" or "xor"')

        return z_flux, z_xor