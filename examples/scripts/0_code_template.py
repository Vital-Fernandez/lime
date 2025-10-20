import lime

# Data location
data_folder = '../doc_notebooks/0_resources/'

# Load the configuration file
cfgFile = f'{data_folder}/long_slit.toml'
obs_cfg = lime.load_cfg(cfgFile)

# Load the spectrum file
fits_file = f'{data_folder}/spectra/gp121903_osiris.fits'
spec = lime.Spectrum.from_file(fits_file, instrument='osiris', redshift=obs_cfg['osiris']['gp121903']['z'])

# Generate the object lines table
lines_frame = spec.retrieve.lines_frame(band_vsigma=100, automatic_grouping=True,
                                        fit_cfg=obs_cfg, obj_cfg_prefix='gp121903_osiris')

# Confirm the presence of lines using intensity thresholding
match_lines = spec.infer.peaks_troughs(lines_frame, emission_shape=True)

# Measure the lines
spec.fit.frame(match_lines, obs_cfg, obj_cfg_prefix='gp121903_osiris')

# Measure the lines
spec.fit.frame(match_lines, obs_cfg, obj_cfg_prefix='gp121903_osiris')

# Plot the results
spec.plot.spectrum(log_scale=True, rest_frame=True)
spec.plot.grid()

# Save the results
spec.save_frame('./gp121903_lines_frame.txt')

