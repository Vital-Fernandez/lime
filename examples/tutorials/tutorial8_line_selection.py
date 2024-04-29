import numpy as np
import lime
from lime.model import exponential_model

# Address of the Green Pea galaxy spectrum
fits_file = '/home/vital/Desktop/gp121903_osiris.fits'
conf_file = '/home/vital/Desktop/cfg.toml'
z_obj, normFlux = 0.19531, 1e-18
bands = np.array([4971.796688, 4984.514249, 4989.5, 5027.303156, 5035.74326, 5043.797081])
gp_spec = lime.Spectrum.from_file(fits_file, instrument='osiris', redshift=z_obj, norm_flux=normFlux)
# gp_spec.plot.spectrum()

gp_spec.fit.bands('O3-pe_5007A_b', bands=bands, fit_conf=conf_file)
gp_spec.fit.report()
gp_spec.plot.bands(rest_frame=True)

