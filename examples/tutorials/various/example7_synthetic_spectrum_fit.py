import numpy as np
import pandas as pd
import lime
from lime.model import gaussian_model

# The object spectrum and flux normalization
z_obj = 0.12345
flux_norm = 1e-17

# Wavelength range definition
hdr_dict = {'CRVAL1': 4500.0,
           'CD1_1': 0.2,
           'NAXIS1': 20000}

w_min = hdr_dict['CRVAL1']
dw = hdr_dict['CD1_1']
nPixels = hdr_dict['NAXIS1']
w_max = w_min + dw * nPixels
# FWZI
wave_rest = np.linspace(w_min, w_max, nPixels, endpoint=False)
wave_obs = (1 + z_obj) * wave_rest

# Linear continuum : slope and interception
cont_coeffs = np.array([-0.001, 20.345])

# Gaussian emission lines:
# [Amplitude (height (normalized flux)), center (angstroms) and sigma (angstroms)]
emission_lines_dict = {'H1_4861A': [75.25, 4861.0, 1.123],
                       'H1_4861A_w1': [7.525, 4861.0, 5.615],
                       'O3_4959A': [150.50, 4959.0, 2.456],
                       'O3_5007A': [451.50, 5007.0, 2.456],
                       'H1_6563A': [225.75, 6563.0, 2.456],
                       'H1_6563A_w1': [225.75, 6566.0, 5.615]}

# Adding continuum as a linear function
flux_obs = wave_obs * cont_coeffs[0] + cont_coeffs[1]

# Adding emission lines
for lineLabel, gauss_params in emission_lines_dict.items():
    amp, center, sigma = gauss_params[0], gauss_params[1] * (1 + z_obj), gauss_params[2]
    flux_obs += gaussian_model(wave_obs, amp, center, sigma)

# Adding (very little) noise
noise_sigma = 0.05
flux_obs = flux_obs + np.random.normal(0, noise_sigma, size=flux_obs.size)

# Let's remove the flux normalization to establish a spectrum in c.g.s units
flux_obs = flux_obs * flux_norm

# ------------------------------------- Perform the fit ----------------------------------------------------------------

# Define the line masks
index_labels = ['H1_4861A_b', 'O3_4959A', 'O3_5007A', 'H1_6563A_b']
column_labels = ['w1', 'w2', 'w3', 'w4', 'w5', 'w6']
table_data = np.array([[4809.8, 4836.1, 4840.6, 4878.6, 4883.1, 4908.4],
                       [4925.2, 4940.4, 4943.0, 4972.9, 4976.7, 4990.2],
                       [4972.7, 4987.0, 4992.0, 5024.7, 5031.5, 5043.9],
                       [6438.0, 6508.6, 6535.10, 6600.9, 6627.69, 6661.8]])
mask_df = pd.DataFrame(data=table_data, index=index_labels, columns=column_labels)

# Fit configuration for the blended lines
cfg_dict = {'H1_4861A_b': 'H1_4861A-H1_4861A_w1',
            'H1_6563A_b': 'H1_6563A-H1_6563A_w1',
            'H1_4861A_w1_sigma': {'expr': '>1*H1_4861A_sigma'},
            'H1_6563A_w1_sigma': {'expr': '>1*H1_6563A_sigma'}}

# Define line measuring object
synth_spec = lime.Spectrum(wave_obs, flux_obs, redshift=z_obj, norm_flux=flux_norm)
# synth_spec.plot.spectrum()

# Measure the emission lines
for lineLabel in mask_df.index.values:

    # Run the fit
    wave_regions = mask_df.loc[lineLabel, 'w1':'w6'].values
    # synth_spec.fit_from_wavelengths(lineLabel, wave_regions, user_cfg=cfg_dict)
    synth_spec.fit.bands(lineLabel, wave_regions, fit_conf=cfg_dict)


    # Display the results
    # synth_spec.plot.line()
    synth_spec.plot.velocity_profile()

    # Compare the measurements with the true values
    if '_b' in lineLabel:
        gaus_comps = cfg_dict[lineLabel].split('-')
    else:
        gaus_comps = [lineLabel]

    for i, comp in enumerate(gaus_comps):
        amp_true, center_true, sigma_true = emission_lines_dict[comp]
        amp_attr, center_attr, sigma_attr = synth_spec.fit.line.amp, synth_spec.fit.line.center / (1 + z_obj), synth_spec.fit.line.sigma
        amp_df, center_df, sigma_df = synth_spec.log.loc[comp, 'amp'] / flux_norm, synth_spec.log.loc[comp, 'center'] / (1 + z_obj), synth_spec.log.loc[comp, 'sigma']

        print(f'\n- {comp}')
        print(f'True amplitude: {amp_true:0.4f}, amplitude attribute {amp_attr[i]:0.4f}, amplitude dataframe {amp_df:0.4f}')
        print(f'True center: {center_true:0.4f}, center attribute {center_attr[i]:0.4f}, center log dataframe {center_df:0.4f}')
        print(f'True sigma: {sigma_true:0.4f}, sigma attribute {sigma_attr[i]:0.4f}, sigma dataframe {sigma_df:0.4f}')


