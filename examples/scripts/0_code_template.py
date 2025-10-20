import lime

# Data location
data_folder = '../doc_notebooks/0_resources/'
cfg_file = f'{data_folder}/long_slit.toml'
fits_file = f'{data_folder}/spectra/gp121903_osiris.fits'

# Load the configuration file
obs_cfg = lime.load_cfg(cfg_file)

# Load the spectroscopic observation
spec = lime.Spectrum.from_file(fits_file, instrument='osiris', redshift=obs_cfg['osiris']['gp121903']['z'])

# Generate the object lines table
lines_frame = spec.retrieve.lines_frame(band_vsigma=120, automatic_grouping=True,
                                        fit_cfg=obs_cfg, obj_cfg_prefix='gp121903_osiris')

# Fit the continuum and confirm the lines via intensity thresholding
spec.fit.continuum(degree_list=[3, 6, 6], emis_threshold=[3, 2, 1.5], log_scale=True, plot_steps=True,)
match_lines = spec.infer.peaks_troughs(lines_frame, emission_shape=True, sigma_threshold=3, plot_steps=True)

# Measure the lines
spec.fit.frame(match_lines, obs_cfg, obj_cfg_prefix='gp121903_osiris')

# Plot the results
spec.plot.spectrum(log_scale=True, rest_frame=True)
spec.plot.grid(fname='lines_grid.png')

# Save the results
spec.save_frame('./gp121903_lines_frame.txt')

