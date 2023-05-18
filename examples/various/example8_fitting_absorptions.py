import numpy as np
import lime

spectrum_address = './sample_data/star_spectrum.txt'
mask_address = './sample_data/abs_line_mask.txt'

wave, flux = np.loadtxt(spectrum_address, unpack=True)
noise_region = np.array([5780, 5850])

mask_log = lime.load_log(mask_address)

star_spec = lime.Spectrum(wave, flux, norm_flux=1e-4, crop_waves=(4515, 9500))
star_spec.plot_spectrum()

# Locate the line fluxes
peaks_table, matched_DF = star_spec.match_line_mask(mask_log, noise_region, line_type='absorption')
star_spec.plot_spectrum(peaks_table=peaks_table, match_log=matched_DF)

# Index of emission lines
idcsObsLines = (matched_DF.observation == 'detected')

# Fit and check the regions
obsLines = matched_DF.loc[idcsObsLines].index.values
for j, lineLabel in enumerate(obsLines):
    wave_regions = matched_DF.loc[lineLabel, 'w1':'w6'].values
    star_spec.fit_from_wavelengths(lineLabel, wave_regions, emission=False, adjacent_cont=False)
    star_spec.display_results(fit_report=True)
    star_spec.plot_fit_components(star_spec.fit_output, lineLabel, frame='rest', log_scale=True)
