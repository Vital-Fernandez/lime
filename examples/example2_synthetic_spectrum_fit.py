
# Include the library into the python path
import sys
from pathlib import Path
example_file_path = Path(__file__).resolve()
lime_path = example_file_path.parent.parent/'src'
src_folder = str(lime_path)
sys.path.append(src_folder)

import lime
import numpy as np
import pandas as pd
from lime.model import gaussian_model

# ------------------------------------- Generate the synthetic spectrum ------------------------------------------------
z_true = 0.12345
norm_obj = 1e-17

# Generate wavelength range
hdr_dict = {'CRVAL1':4500.0,
           'CD1_1':0.2,
           'NAXIS1':20000}

w_min = hdr_dict['CRVAL1']
dw = hdr_dict['CD1_1']
nPixels = hdr_dict['NAXIS1']
w_max = w_min + dw * nPixels

wave_rest = np.linspace(w_min, w_max, nPixels, endpoint=False)
wave_obs = (1 + z_true) * wave_rest

# Linear continuum : slope and interception
continuum_lineal = np.array([-0.001, 20.345])

# Gaussian emission lines: Amplitude (height (norm flux)), center (angstroms) and sigma (angstroms)
emission_lines_dict = {'H1_4861A': [75.25, 4861.0, 1.123],
                       'H1_4861A_w1': [7.525, 4861.0, 5.615],
                       'O3_4959A': [150.50, 4959.0, 2.456],
                       'O3_5007A': [451.50, 5007.0, 2.456],
                       'H1_6563A': [225.75, 6563.0, 2.456],
                       'H1_6563A_w1': [225.75, 6566.0, 5.615]}

# Adding coninuum as a linear function
flux_obs = wave_obs * continuum_lineal[0] + continuum_lineal[1]

# Adding emission lines
for lineLabel, gaus_params in emission_lines_dict.items():
    flux_obs += gaussian_model(wave_obs, gaus_params[0], gaus_params[1] * (1 + z_true), gaus_params[2])

# Add noise
noise_sigma = 0.05
flux_obs = flux_obs + np.random.normal(0, noise_sigma, size=flux_obs.size)

# deNormalise
flux_obs = flux_obs * norm_obj

# ------------------------------------- Perform the fit ----------------------------------------------------------------

# Declare line masks
input_lines = ['H1_4861A_b', 'O3_4959A', 'O3_5007A', 'H1_6563A_b']
input_columns = ['w1', 'w2', 'w3', 'w4', 'w5', 'w6']
mask_df = pd.DataFrame(index=input_lines, columns=input_columns)
mask_df.loc['H1_4861A_b'] = np.array([4809.8, 4836.1, 4840.6, 4878.6, 4883.1, 4908.4])
mask_df.loc['O3_4959A'] = np.array([4925.2, 4940.4, 4943.0, 4972.9, 4976.7, 4990.2])
mask_df.loc['O3_5007A'] = np.array([4972.7, 4987.0, 4992.0, 5024.7, 5031.5, 5043.984899])
mask_df.loc['H1_6563A_b'] = np.array([6438.0, 6508.6, 6535.10, 6600.9, 6627.69, 6661.8])

# Declare fit configuration

conf_dict = dict(fit_conf={'H1_4861A_b': 'H1_4861A-H1_4861A_w1',
                           'H1_6563A_b': 'H1_6563A-H1_6563A_w1',
                           'H1_6563A_w1_sigma': {'expr': '>1*H1_6563A_sigma'}})

# Declare line measuring object
lm = lime.Spectrum(wave_obs, flux_obs, redshift=z_true, normFlux=norm_obj)
lm.plot_spectrum()

# Find lines
noise_region = (1 + z_true) * np.array([5400, 5500])
norm_spec = lime.continuum_remover(lm.wave_rest, lm.flux, noiseRegionLims=noise_region)
obsLinesTable = lime.line_finder(lm.wave_rest, norm_spec, noiseWaveLim=noise_region, intLineThreshold=3)
matchedDF = lime.match_lines(lm.wave_rest, lm.flux, obsLinesTable, mask_df)
lm.plot_spectrum(obsLinesTable=obsLinesTable, matchedLinesDF=matchedDF, specLabel=f'Emission line detection')

# Measure the emission lines
for i, lineLabel in enumerate(matchedDF.index.values):
    wave_regions = matchedDF.loc[lineLabel, 'w1':'w6'].values
    lm.fit_from_wavelengths(lineLabel, wave_regions, user_conf=conf_dict['fit_conf'])
    lm.display_results(show_fit_report=True, show_plot=True, log_scale=True, frame='obs')
    lm.plot_line_velocity()

# Save to txt file
lm.save_line_log(Path.home()/'synth_spec_linelog.txt', output_type='txt')
lm.save_line_log(Path.home()/'synth_spec_flux_table', output_type='flux_table')
lm.save_line_log(Path.home()/'synth_spec_linelog.fits', output_type='fits')


