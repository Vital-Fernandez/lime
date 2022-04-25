import numpy as np
import lime
from astropy.io import fits
import matplotlib.pyplot as plt
from lime.model import gaussian_model


def import_osiris_fits(file_address, ext=0):

    # Open fits file
    with fits.open(file_address) as hdul:
        data, header = hdul[ext].data, hdul[ext].header

    w_min, dw, n_pix = header['CRVAL1'],  header['CD1_1'], header['NAXIS1']
    w_max = w_min + dw * n_pix
    wavelength = np.linspace(w_min, w_max, n_pix, endpoint=False)

    return wavelength, data, header


def gauss_calcu(log, x, line, frame='observed', z_cor = 1):

    if frame == 'observed':
        a, mu, sigma = log.loc[line, 'amp'],  log.loc[line, 'center'], log.loc[line, 'sigma']
        y_gaussian = gaussian_model(x, a, mu, sigma)

    else:
        a, mu, sigma = log.loc[line, 'amp'],  log.loc[line, 'center'], log.loc[line, 'sigma']
        y_gaussian = gaussian_model(x * z_cor, a, mu, sigma)
        y_gaussian = y_gaussian * z_cor

    return y_gaussian


def cont_calcu(log, x, line, frame='observed', z_cor = 1):

    if frame == 'observed':
        m, n = log.loc[line, 'm_cont'], log.loc[line, 'n_cont']
        y_cont = m * x + n

    else:
        m, n = log.loc[line, 'm_cont'], log.loc[line, 'n_cont']
        # y_cont = m * x + n
        # y_cont = y_cont/z_cor
        y_cont = m * (x * z_cor) + n
        y_cont = y_cont * z_cor

    return y_cont


# Address of the Green Pea galaxy spectrum
gp_fits = './sample_data/gp121903_BR.fits'

# Load spectrum
wave, flux, hdr = import_osiris_fits(gp_fits)

# Galaxy redshift and the flux normalization
z_gp = 0.19531
normFlux_gp = 1e-14

# Line name and its location mask in the rest frame
line = 'H1_6563A'
mask = np.array([6438.03, 6508.66, 6535.10, 6600.95, 6627.70, 6661.82])

# Define a spectrum object
gp_spec = lime.Spectrum(wave, flux, redshift=z_gp, norm_flux=normFlux_gp)
#
#  Fit configuration
fit_conf = {'H1_6563A_b': 'H1_6563A-N2_6584A-N2_6548A',
            'N2_6548A_amp': {'expr': 'N2_6584A_amp / 2.94'},
            'N2_6548A_kinem': 'N2_6584A'}

gp_spec.fit_from_wavelengths('H1_6563A', mask, user_cfg=fit_conf)
# gp_spec.display_results()
# gp_spec.display_results(frame='rest')

# x_range = np.linspace(mask[0]*(1+z_gp), mask[-1]*(1+z_gp), 400)
# gauss_prof = gauss_calcu(gp_spec.log, x_range, 'H1_6563A')
# y_cont = cont_calcu(gp_spec.log, x_range, 'H1_6563A')
#
# fig, ax = plt.subplots(figsize=(12, 12))
# ax.step(wave, flux)
# ax.plot(x_range, gauss_prof+y_cont)
# ax.plot(x_range, y_cont)
# ax.set_xlim(mask[0]*(1+z_gp), mask[-1]*(1+z_gp))
# ax.set_yscale('log')
# plt.show()

# z_corr = 1 + z_gp
# x_range = np.linspace(mask[0], mask[-1], 400)
# gauss_prof = gauss_calcu(gp_spec.log, x_range, 'H1_6563A', frame='rest', z_cor=z_corr)
# y_cont = cont_calcu(gp_spec.log, x_range, 'H1_6563A', frame='rest', z_cor=z_corr)
#
# fig, ax = plt.subplots(figsize=(12, 12))
# ax.step(wave/z_corr, flux, label='No flux correction')
# ax.step(wave/z_corr, flux*z_corr, label='Flux correction')
#
# ax.plot(x_range, y_cont)
# ax.plot(x_range, gauss_prof+y_cont, ':')
# ax.set_xlim(mask[0], mask[-1])
# ax.set_yscale('log')
# ax.legend()
# plt.show()

x_range = np.linspace(mask[0]*(1+z_gp), mask[-1]*(1+z_gp), 400)
gauss_prof = gauss_calcu(gp_spec.log, x_range, 'H1_6563A')
y_cont = cont_calcu(gp_spec.log, x_range, 'H1_6563A')

fig, ax = plt.subplots(figsize=(12, 12))
ax.step(wave, flux)
ax.plot(x_range, gauss_prof+y_cont)
ax.plot(x_range, y_cont)
ax.set_xlim(mask[0]*(1+z_gp), mask[-1]*(1+z_gp))
ax.set_yscale('log')
plt.show()

z_corr = 1 + z_gp
fig, ax = plt.subplots(figsize=(12, 12))
ax.step(wave/z_corr, flux*z_corr)
ax.plot(x_range/z_corr, (gauss_prof+y_cont)*z_corr)
ax.plot(x_range/z_corr, y_cont*z_corr)
ax.set_xlim(mask[0]*(1+z_gp)/z_corr, mask[-1]*(1+z_gp)/z_corr)
ax.set_yscale('log')
plt.show()