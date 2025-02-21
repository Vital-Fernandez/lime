import numpy as np
from astropy.io import fits
import lime
import pandas as pd

# Load spectrum
df = pd.read_csv('/home/vital/PycharmProjects/vital_tests/astro/consultations/marta/spec_8144-12703.txt')
wave = np.array(df.wave)
flux = np.array(df.flux)

cont_from_bands, err_from_bands = True, True

# Galaxy redshift and the flux normalization
z_obj = 0.00044
gp_spec = lime.Spectrum(wave, flux, redshift=z_obj, units_flux='1e20*FLAM')

# Bands for Hbeta from the default database
Hbeta_bands = gp_spec.retrieve.line_bands(lines_list=['H1_4861A'], bands_kinematic_width=None)
gp_spec.fit.bands('H1_4861A', bands=Hbeta_bands, cont_from_bands=cont_from_bands, err_from_bands=err_from_bands)
gp_spec.plot.bands(rest_frame=True)

Hbeta_bands = [4809.8,4836.1, 4857.0, 4866.1, 4883.13,4908.4,]
gp_spec.fit.bands('H1_4861A', bands=Hbeta_bands, cont_from_bands=cont_from_bands, err_from_bands=err_from_bands)
gp_spec.plot.bands()

gp_spec.fit.bands('H1_4861A', bands=Hbeta_bands, cont_from_bands=False, err_from_bands=err_from_bands)
gp_spec.plot.bands()

# Adjust bands to selection


# Second attempt including the fit configuration
fit_conf = {'H1_4861A_b': 'H1_4861A+H1_4861A_p-g-abs'}
Hbeta_bands = gp_spec.retrieve.line_bands(lines_list=['H1_4861A'], bands_kinematic_width=None)
gp_spec.fit.bands('H1_4861A_b', bands=Hbeta_bands, fit_conf=fit_conf, cont_from_bands=cont_from_bands, err_from_bands=err_from_bands)
print(f'\ncont_from_bands = {cont_from_bands}; err_from_bands = {err_from_bands}; continuum level = {gp_spec.fit.line.cont:0.3f} +/- {gp_spec.fit.line.cont_err:0.3f}')
gp_spec.plot.bands()
