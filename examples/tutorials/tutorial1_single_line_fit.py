import numpy as np
from astropy.io import fits
import lime


def import_osiris_fits(file_address, ext=0):

    # Open the fits file
    with fits.open(file_address) as hdul:
        data, header = hdul[ext].data, hdul[ext].header

    # Reconstruct the wavelength array from the header data
    w_min, dw, n_pix = header['CRVAL1'],  header['CD1_1'], header['NAXIS1']
    w_max = w_min + dw * n_pix
    wavelength = np.linspace(w_min, w_max, n_pix, endpoint=False)

    return wavelength, data, header


# Address of the Green Pea galaxy spectrum
fits_file = '../sample_data/gp121903_osiris.fits'

# Load spectrum
wave, flux, hdr = import_osiris_fits(fits_file)

# Galaxy redshift and the flux normalization
z_obj = 0.19531
normFlux = 1e-18

# Line name and its location mask in the rest _frame
line = 'H1_6563A'
band_edges = np.array([6438.03, 6508.66, 6535.10, 6600.95, 6627.70, 6661.82])

# Define a spectrum object
gp_spec = lime.Spectrum(wave, flux, redshift=z_obj, norm_flux=normFlux)
gp_spec.plot.spectrum(label='GP121903')

# Run the fit
gp_spec.fit.band(line, band_edges)

# Plot the results from the last fitting
gp_spec.plot.band()

# Fit configuration
line = 'H1_6563A_b'
fit_conf = {'H1_6563A_b': 'H1_6563A+N2_6584A+N2_6548A',
            'N2_6548A_amp': {'expr': 'N2_6584A_amp/2.94'},
            'N2_6548A_kinem': 'N2_6584A'}

# Second attempt including the fit configuration
gp_spec.fit.band(line, band_edges, fit_conf)
gp_spec.plot.band(line)

# You can also save the fitting plot to a file
gp_spec.plot.band(output_address=f'../sample_data/{line}.png')

# Each fit is stored in the lines dataframe (log) attribute
print(gp_spec.log)

# It can be saved into different types of document using the function
gp_spec.save_log('../sample_data/example1_linelog.txt')
gp_spec.save_log('../sample_data/example1_linelog.pdf', param_list=['eqw', 'gauss_flux', 'gauss_err'])
gp_spec.save_log('../sample_data/example1_linelog.fits', ext='GP121903')
gp_spec.save_log('../sample_data/example1_linelog.xlsx', ext='GP121903')
gp_spec.save_log('../sample_data/example1_linelog.asdf', ext='GP121903')

# A lines log can also be saved/loaded using the lime functions:
log_address = '../sample_data/example1_linelog.fits'
lime.save_log(gp_spec.log, log_address, ext='GP121903')
log = lime.load_log(log_address, ext='GP121903')
print(f'Is the original log equal to the one saved and loaded from a fits file?\n{gp_spec.log.equals(log)}')
