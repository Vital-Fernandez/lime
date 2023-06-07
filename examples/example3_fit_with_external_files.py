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
obsFitsFile = './sample_data/gp121903_osiris.fits'
lineMaskFile = './sample_data/osiris_bands.txt'
cfgFile = './sample_data/osiris.toml'

# Load spectrum
wave, flux, header = import_osiris_fits(obsFitsFile)

# Load line bands
bands = lime.load_log(lineMaskFile)

# Load configuration
obs_cfg = lime.load_cfg(cfgFile)
z_obj = obs_cfg['sample_data']['z_array'][2]
norm_flux = obs_cfg['sample_data']['norm_flux']

# Declare LiMe spectrum
gp_spec = lime.Spectrum(wave, flux, redshift=z_obj, norm_flux=norm_flux)
gp_spec.plot.spectrum(label='GP121903', rest_frame=True)

# Find lines
match_bands = gp_spec.line_detection(bands, poly_degree=[3, 7, 7, 7], emis_threshold=[5, 3, 2, 0.7])
gp_spec.plot.spectrum(label='GP121903 matched lines', line_bands=match_bands, log_scale=True)

# Saving GP121903 bands
obj_bands_file = './sample_data/gp121903_bands.txt'
lime.save_log(match_bands, obj_bands_file)

# Object line fitting configuration
fit_cfg = obs_cfg['gp121903_line_fitting']

# Measure the emission lines
gp_spec.fit.frame(obj_bands_file, fit_cfg, progress_output='counter')

# Display the fits on the spectrum
gp_spec.plot.spectrum(include_fits=True)

# Display a grid with the fits
gp_spec.plot.grid(gp_spec.log, rest_frame=True)

# Save the data
gp_spec.save_log('./sample_data/example3_linelog.fits', ext='GP121903a')
gp_spec.save_log('./sample_data/example3_linelog.txt')
lime.save_log(gp_spec.log, './sample_data/example3_linelog.fits', ext='GP121903b')
