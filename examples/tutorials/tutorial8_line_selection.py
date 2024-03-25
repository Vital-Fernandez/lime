import numpy as np
from astropy.io import fits
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

# gp_spec.fit.report()
# gp_spec.plot.bands()

# gp_spec.fit.bands('O3_5007A_p-pp')
# gp_spec.fit.report()
# gp_spec.plot.bands('O3_5007A_p-pp', y_scale='linear')
# gp_spec.save_log('/home/vital/Desktop/profiles_tests.txt')


# import numpy as np
# import matplotlib.pyplot as plt
#
# # Generate x values
# x_values = np.linspace(0.1, 5, 100)  # Avoid x=0 for logarithmic function
#
# # Calculate corresponding y values for ln(x)
# y_values = np.log10(x_values)
#
# # Plot the logarithmic function
# plt.plot(x_values, y_values, label='y = ln(x)')
# plt.title('Logarithmic Function: y = ln(x)')
# plt.xlabel('x')
# plt.ylabel('ln(x)')
# plt.legend()
# plt.grid(True)
# plt.show()
#
# import numpy as np
# from matplotlib import pyplot as plt
# from lime.model import gaussian_model, broken_powerlaw_model, broken_powerlaw_model_ratio
#
# # Profile parameters
# amp = 125.0
# mu = 4862.55
# sigma = 1.55
# gamma = 2.55
# m_cont = 0
# n_cont = 10
# noise = 2
# frac = 0.105
# n_pixels = 5000
# off = -2
# exp = -0.5
# x_break = 1
#
# # Arrays with the independent variables
# x_array = np.linspace(mu - 75, mu + 75, n_pixels)
#
# gaussian_array = gaussian_model(x_array, amp, mu, sigma)
#
# broken_power = broken_powerlaw_model(x_array, amp, sigma, mu, exp)
# broken_power2 = broken_powerlaw_model(x_array, amp, 2*sigma, mu, exp)
# broken_power3 = broken_powerlaw_model(x_array, amp, 3*sigma, mu, exp)
#
# broken_power_ratio = broken_powerlaw_model_ratio(x_array, amp, sigma, mu, exp)
# broken_power_ratio2 = broken_powerlaw_model_ratio(x_array, amp, 2*sigma, mu, exp)
# broken_power_ratio3 = broken_powerlaw_model_ratio(x_array, amp, 3*sigma, mu, exp)
#
# fig, ax = plt.subplots(dpi=200)
# # ax.plot(x_array, gaussian_array, label='Gaussian')
# # ax.plot(x_array, broken_power, label=r'Power law, break = 1$\sigma$', color='tab:blue')
# ax.scatter(x_array, broken_power_ratio, color='tab:blue')
# # ax.plot(x_array, broken_power2, label=r'Power law, break = 2$\sigma$', color='tab:orange')
# ax.plot(x_array, broken_power_ratio2, linestyle=':', color='tab:orange')
# # ax.plot(x_array, broken_power3, label=r'Power law, break = 3$\sigma$', color='tab:green')
# ax.plot(x_array, broken_power_ratio3, linestyle=':', color='tab:green')
# ax.legend()
# ax.set_ylabel('Flux')
# ax.set_xlabel('Wavelength')
# # ax.set_yscale('log')
# plt.show()


import numpy as np
from matplotlib import pyplot as plt
from lime.model import gaussian_model, broken_powerlaw_model

# # Profile parameters
# amp = 125.0
# mu = 4862.55
# sigma = 1.55
# gamma = 2.55
# m_cont = 0
# n_cont = 10
# noise = 2
# frac = 0.105
# n_pixels = 500
# off = -2
# exp = -0.5
# x_break = 1
#
# # Arrays with the independent variables
# x_array = np.linspace(mu - 75, mu + 75, n_pixels)
# cont_array = m_cont * x_array + n_cont
# noise_array = np.random.normal(m_cont, noise, size=x_array.shape)
# gaussian_array = gaussian_model(x_array, amp, mu, sigma)
# broken_power = broken_powerlaw_model(x_array, amp, sigma, mu, exp)
# broken_power2 = broken_powerlaw_model(x_array, amp, 2*sigma, mu, exp)
# broken_power3 = broken_powerlaw_model(x_array, amp, 3*sigma, mu, exp)
#
# fig, ax = plt.subplots(dpi=200)
# ax.plot(x_array, gaussian_array + noise_array, label='Gaussian')
# ax.plot(x_array, broken_power, label=r'Power law, break = 1$\sigma$')
# ax.plot(x_array, broken_power2, label=r'Power law, break = 2$\sigma$')
# ax.plot(x_array, broken_power3, label=r'Power law, break = 3$\sigma$')
# ax.legend()
# ax.set_ylabel('Flux')
# ax.set_xlabel('Wavelength')
# plt.show()





#
# import matplotlib.pyplot as plt
# import numpy as np
# import lime
# from lmfit.models import gaussian, lorentzian, voigt, pvoigt
# from lime.model import AREA_FUNCTIONS, FWHM_FUNCTIONS, PROFILE_FUNCTIONS, gaussian_model, broken_powerlaw_model, pseudo_power_model
# from astropy.modeling.powerlaws import SmoothlyBrokenPowerLaw1D, BrokenPowerLaw1D
#
#
# def broken_power(x, amp, alpha, center, x_break):
#
#     alpha_in = np.where((x > (center - x_break)) & (x < (center + x_break)), 0, alpha)
#
#     return amp * np.power(np.abs(x-center), alpha_in)
#
#
# # Profile parameters
# amp = 125.0
# mu = 4862.55
# sigma = 1.55
# gamma = 2.55
# m_cont = 0.01
# n_cont = 10
# noise = 0.1
# frac = 0.105
# n_pixels = 5000
# off = -2
# exp = -2.5
# x_break = 0.5
#
# # Arrays with the independent variables
# x_array = np.linspace(mu - 75, mu + 75, n_pixels)
# cont_array = m_cont * x_array + n_cont
# noise_array = np.random.normal(0, noise, size=x_array.shape)
# gaussian_array = gaussian(x_array, amp * np.sqrt(2*np.pi) * sigma, mu, sigma)
# lorentzian_array = lorentzian(x_array, amp * np.pi * sigma, mu, sigma)
# pseudo_voigt_array = frac * gaussian_array + (1 - frac) * lorentzian_array
# pseudo_power_array = pseudo_power_model(x_array, amp, sigma, mu, exp, frac)

# fig, ax = plt.subplots()
# ax.step(x_array, gaussian_array)
# ax.scatter(x_array, pseudo_power_array)
# plt.show()



# import numpy as np
# from astropy.io import fits
# import lime
#
# # Address of the Green Pea galaxy spectrum
# fits_file = '/home/vital/Desktop/gp121903_osiris.fits'
# conf_file = '/home/vital/Desktop/cfg.toml'
# z_obj, normFlux = 0.19531, 1e-18
# gp_spec = lime.Spectrum.from_file(fits_file, instrument='osiris', redshift=z_obj, norm_flux=normFlux)
#
# # gp_spec.fit.bands('O3_5007A_b', fit_conf=conf_file)
# # gp_spec.fit.report()
# # gp_spec.plot.bands()
#
# gp_spec.fit.bands('O3_5007A_p-pp')
# gp_spec.fit.report()
# gp_spec.plot.bands('O3_5007A_p-pp', y_scale='linear')
# gp_spec.save_log('/home/vital/Desktop/profiles_tests.txt')


