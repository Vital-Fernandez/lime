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
obsFitsFile = './sample_data/gp121903_ISIS_spectrum.fits'
lineMaskFile = './sample_data/osiris_bands.txt'
cfgFile = './sample_data/config_file.cfg'

# Load spectrum
wave, flux, header = import_osiris_fits(obsFitsFile)

# Load configuration
obs_cfg = lime.load_cfg(cfgFile)
fit_cfg = obs_cfg['gp121903_line_fitting']

# Declare line measuring object
norm_flux = obs_cfg['sample_data']['norm_flux']
gp_spec = lime.Spectrum(wave, flux, norm_flux=norm_flux)

# Reference lines for the redshift calculation
band_df = lime.spectral_bands(lines_list=['H1_6563A', 'H1_4861A', 'O3_5007A', 'O3_4363A'])

# Providing a table where the redshift prediction is saved
redshift_table = './sample_data/reshift_table.txt'
gp_spec.check.redshift('gp121903', band_df.index.values, redshift_table)
