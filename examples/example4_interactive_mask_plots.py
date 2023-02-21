import numpy as np
from astropy.io import fits
import lime
import shutil


def import_osiris_fits(file_address, ext=0):

    # Open fits file
    with fits.open(file_address) as hdul:
        data, header = hdul[ext].data, hdul[ext].header

    w_min, dw, n_pix = header['CRVAL1'],  header['CD1_1'] , header['NAXIS1']
    w_max = w_min + dw * n_pix
    wavelength = np.linspace(w_min, w_max, n_pix, endpoint=False)

    return wavelength, data, header


# State the data files
obsFitsFile = './sample_data/gp121903_BR.fits'
instrMaskFile = './sample_data/osiris_bands.txt'
cfgFile = './sample_data/config_file.cfg'

# Load the spectrum
wave, flux, header = import_osiris_fits(obsFitsFile)

# Load configuration
sample_cfg = lime.load_cfg(cfgFile)

# Object properties
z_obj = sample_cfg['sample_data']['z_array'][2]
norm_flux = sample_cfg['sample_data']['norm_flux']

# Create a new mask file for the galaxy
objMaskFile = './sample_data/GP121903_mask.txt'
shutil.copy(instrMaskFile, objMaskFile)

# Review the mask
gp_spec = lime.Spectrum(wave, flux, redshift=z_obj, norm_flux=norm_flux)
delta_obs = np.diff(gp_spec.wave).mean()
delta_rest = np.diff(gp_spec.wave_rest).mean()

# print(delta_obs, delta_rest, )
# gp_spec.wave/(gp_spec.wave_rest + delta_rest) - 1
# interval = (gp_spec.wave/(gp_spec.wave_rest + np.diff(gp_spec.wave_rest).median()) - 1).mean()

inputMask = lime.load_log(objMaskFile)
objMaskFile = './sample_data/GP121903_mask_review.txt'
redshift_log = './sample_data/redshift_log.txt'
gp_spec.check.bands(inputMask, objMaskFile, maximize=True, redshift_log=redshift_log, object_ref='GP121903')

