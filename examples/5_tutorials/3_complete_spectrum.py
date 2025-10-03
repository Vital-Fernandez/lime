import lime

# State the data files
obsFitsFile = '../0_resources/spectra/gp121903_osiris.fits'
lineBandsFile = '../0_resources/bands/gp121903_bands.txt'
cfgFile = '../0_resources/long_slit.toml'

# Load configuration
obs_cfg = lime.load_cfg(cfgFile)
z_obj = obs_cfg['osiris']['gp121903']['z']
norm_flux = obs_cfg['osiris']['norm_flux']

# Declare LiMe spectrum
gp_spec = lime.Spectrum.from_file(obsFitsFile, instrument='osiris', redshift=z_obj, norm_flux=norm_flux)

# Fit the continuum
gp_spec.fit.continuum(degree_list=[3, 6, 6], emis_threshold=[3, 2, 1.5], plot_steps=True, log_scale=True,
                      smooth_scale=10)
in_bands = lime.load_frame(lineBandsFile)
gp_spec.plot.spectrum(bands=in_bands, show_cont=True, log_scale=True, rest_frame=True)

# Confirm present lines
match_bands = gp_spec.infer.peaks_troughs(lineBandsFile, emission_shape=True, sigma_threshold=3, plot_steps=True,
                                          log_scale=True)






# # Measure the emission lines
# gp_spec.fit.frame(match_bands, obs_cfg, obj_cfg_prefix='gp121903_osiris', line_detection=True)
# gp_spec.bokeh.spectrum()

# # Display the fits on the spectrum
# gp_spec.plot.spectrum(rest_frame=True, log_scale=True)
#
# # Display a grid with the fits
# gp_spec.plot.grid(rest_frame=True, y_scale='auto')
#
# # Save the data
# gp_spec.save_frame('../0_resources/results/example3_linelog.txt')
# gp_spec.save_frame('../0_resources/results/example3_linelog.xlsx', page='GP121903b')
# gp_spec.save_frame('../0_resources/results/example3_linelog.pdf')
# lime.save_frame('../0_resources/results/example3_linelog.fits', gp_spec.frame, page='GP121903b')
