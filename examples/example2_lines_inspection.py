import numpy as np
from astropy.io import fits
import lime
from pathlib import Path


def import_osiris_fits(file_address, ext=0):

    # Open fits file
    with fits.open(file_address) as hdul:
        data, header = hdul[ext].data, hdul[ext].header

    w_min, dw, n_pix = header['CRVAL1'],  header['CD1_1'] , header['NAXIS1']
    w_max = w_min + dw * n_pix
    wavelength = np.linspace(w_min, w_max, n_pix, endpoint=False)

    return wavelength, data, header


# State the data files
obsFitsFile = './sample_data/gp121903_ISIS_spectrum.fits'
instrMaskFile = './sample_data/osiris_bands.txt'

# Load the spectrum
wave, flux, header = import_osiris_fits(obsFitsFile)

# Create the Spectrum object
z_obj = 0.19531
norm_flux = 1e-18
gp_spec = lime.Spectrum(wave, flux, redshift=z_obj, norm_flux=norm_flux)

# Import the lines database:
bands_df = lime.spectral_bands(wave_inter=gp_spec)

# Save to a file (if it does not exist already)
bands_df_file = Path('./sample_data/GP121903_bands.txt')
if bands_df_file.is_file() is not True:
    lime.save_log(bands_df, bands_df_file)

# Review the bands file
gp_spec.check.bands(bands_df_file, maximize=True)

# Adding a redshift file address to store the variations in redshift
redshift_file = './sample_data/redshift_log.txt'
redshift_file_header, object_ref = 'redshift', 'GP121903'
gp_spec.check.bands(bands_df_file, maximize=True, redshift_log=redshift_file,
                    redshift_column=redshift_file_header, object_ref='object_ref')

