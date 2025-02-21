import lime

# State the data files
obsFitsFile = '../sample_data/spectra/gp121903_osiris.fits'
lineBandsFile = '../sample_data/osiris_bands.txt'
cfgFile = '../sample_data/osiris.toml'

# Load configuration
obs_cfg = lime.load_cfg(cfgFile)
z_obj = obs_cfg['sample_data']['z_array'][2]
norm_flux = obs_cfg['sample_data']['norm_flux']

# Declare LiMe spectrum
gp_spec = lime.Spectrum.from_file(obsFitsFile, instrument='osiris', redshift=z_obj, norm_flux=norm_flux)
gp_spec.plot.spectrum()

# Fit the continuum
gp_spec.fit.continuum(degree_list=[3, 6, 6], emis_threshold=[3, 2, 1.5], plot_steps=False)

# Find lines
match_bands = gp_spec.line_detection(lineBandsFile, sigma_threshold=3, plot_steps=False)
gp_spec.plot.spectrum(label='GP121903 matched lines', bands=match_bands, log_scale=True)

# Saving GP121903 bands
obj_bands_file = '../sample_data/gp121903_bands.txt'
lime.save_frame(obj_bands_file, match_bands)

# Measure the emission lines
gp_spec.fit.frame(obj_bands_file, obs_cfg, id_conf_prefix='gp121903', line_detection=True)

# Display the fits on the spectrum
gp_spec.plot.spectrum(rest_frame=True)

# Display a grid with the fits
gp_spec.plot.grid(rest_frame=True, y_scale='auto')

# Save the data
gp_spec.save_frame('../sample_data/example3_linelog.txt')
gp_spec.save_frame('../sample_data/example3_linelog.xlsx', page='GP121903b')
gp_spec.save_frame('../sample_data/example3_linelog.pdf')
lime.save_frame('../sample_data/example3_linelog.fits', gp_spec.frame, page='GP121903b')

