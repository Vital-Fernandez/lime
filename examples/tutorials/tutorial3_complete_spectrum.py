import numpy as np
from astropy.io import fits
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
gp_spec.plot.spectrum(label='GP121903', rest_frame=True)

# Find lines
match_bands = gp_spec.line_detection(lineBandsFile, cont_fit_degree=[3, 7, 7, 7], cont_int_thres=[5, 3, 2, 1.5])
gp_spec.plot.spectrum(label='GP121903 matched lines', line_bands=match_bands, log_scale=True)

# Saving GP121903 bands
obj_bands_file = '../sample_data/gp121903_bands.txt'
lime.save_log(match_bands, obj_bands_file)

# Measure the emission lines
gp_spec.fit.frame(obj_bands_file, obs_cfg, id_conf_prefix='gp121903')

# Display the fits on the spectrum
gp_spec.plot.spectrum(include_fits=True)

# Display a grid with the fits
gp_spec.plot.grid(rest_frame=True)

# Save the data
gp_spec.save_log('../sample_data/example3_linelog.txt')
gp_spec.save_log('../sample_data/example3_linelog.xlsx', page='GP121903b')
gp_spec.save_log('../sample_data/example3_linelog.pdf', param_list=['eqw', 'gauss_flux', 'gauss_flux_err'])
lime.save_log(gp_spec.log, '../sample_data/example3_linelog.fits', page='GP121903b')

