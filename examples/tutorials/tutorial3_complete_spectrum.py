import numpy as np
from astropy.io import fits
import lime


def import_osiris_fits(file_address, ext=0):

    # Open fits file
    with fits.open(file_address) as hdul:
        data, hdr = hdul[ext].data, hdul[ext].header

    w_min, dw, n_pix = hdr['CRVAL1'],  hdr['CD1_1'], hdr['NAXIS1']
    w_max = w_min + dw * n_pix
    wavelength = np.linspace(w_min, w_max, n_pix, endpoint=False)

    return wavelength, data, hdr


# State the data files
obsFitsFile = '../sample_data/gp121903_osiris.fits'
lineBandsFile = '../sample_data/osiris_bands.txt'
cfgFile = '../sample_data/osiris.toml'

# Load spectrum
wave, flux, header = import_osiris_fits(obsFitsFile)

# Load line bands
bands = lime.load_log(lineBandsFile)

# Load configuration
obs_cfg = lime.load_cfg(cfgFile)
z_obj = obs_cfg['sample_data']['z_array'][2]
norm_flux = obs_cfg['sample_data']['norm_flux']

# Declare LiMe spectrum
gp_spec = lime.Spectrum(wave, flux, redshift=z_obj, norm_flux=norm_flux)
gp_spec.plot.spectrum(label='GP121903', rest_frame=True)

# Find lines
match_bands = gp_spec.line_detection(bands, cont_fit_degree=[3, 7, 7, 7], cont_int_thres=[5, 3, 2, 1.5])
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
# gp_spec.save_log('../sample_data/example3_linelog.txt')
# gp_spec.save_log('../sample_data/example3_linelog.xlsx', page='GP121903b')
# gp_spec.save_log('../sample_data/example3_linelog.pdf', param_list=['eqw', 'gauss_flux', 'gauss_flux_err'])
# lime.save_log(gp_spec.log, '../sample_data/example3_linelog.fits', page='GP121903b')
#
#
# gp_spec.log[['intg_flux_err', 'gauss_flux_err']]


import pandas as pd
import matplotlib.pyplot as plt
data = gp_spec.log['intg_flux_err']/gp_spec.log['gauss_flux_err'] - 1

# Plotting a histogram from the Pandas Series
plt.figure(figsize=(8, 6))  # Adjust the figure size if needed
plt.hist(data, color='skyblue', edgecolor='black')  # Adjust the number of bins as required
plt.title('Histogram of a Pandas Series')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.5)
plt.show()