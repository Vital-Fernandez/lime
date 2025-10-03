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

# Interactive line bands adjustment
gp_spec.check.bands(lineBandsFile, show_continua=False, band_vsigma=70, n_sigma=4)