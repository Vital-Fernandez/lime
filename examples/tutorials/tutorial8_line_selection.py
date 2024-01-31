
import numpy as np
from astropy.io import fits
import lime

# Address of the Green Pea galaxy spectrum
fits_file = '/home/vital/Desktop/gp121903_osiris.fits'
z_obj, normFlux = 0.19531, 1e-18
gp_spec = lime.Spectrum.from_file(fits_file, instrument='osiris', redshift=z_obj, norm_flux=normFlux)
# gp_spec.plot.spectrum(label='GP121903')
#
# gp_spec.fit.bands('H1_4861A')
# gp_spec.plot.bands('H1_4861A', y_scale='linear')
# gp_spec.fit.bands('H1_4861A_p-l')
# gp_spec.plot.bands('H1_4861A_p-l', y_scale='linear')
gp_spec.fit.bands('H1_4861A_p-pv')
gp_spec.plot.bands('H1_4861A_p-pv', y_scale='linear')
gp_spec.save_log('/home/vital/Desktop/profiles_tests.txt')

# import numpy as np
# from scipy.optimize import curve_fit
# from scipy.special import wofz
# import matplotlib.pyplot as plt
# from lime.model import voigt_model
#
# def voigt(x, center, amplitude, sigma, gamma):
#     """
#     Voigt profile function.
#     x : array-like
#         Independent variable.
#     center : float
#         Center of the line.
#     amplitude : float
#         Amplitude of the line.
#     sigma : float
#         Gaussian standard deviation.
#     gamma : float
#         Lorentzian half-width at half-maximum.
#     """
#     z = ((x-center) + 1j*gamma) / (sigma*np.sqrt(2))
#     return amplitude * np.real(wofz(z)) / (sigma * np.sqrt(2*np.pi))
#
# # Generating synthetic data
# np.random.seed(0)
# x = np.linspace(-5, 5, 100)
# true_params = [0, 1, 1, 1]
# y = voigt(x, *true_params)
# y_noise = y + 0.01 * np.random.normal(size=x.size)
#
# # Fitting the Voigt profile to the data
# popt, pcov = curve_fit(voigt, x, y_noise, p0=[0, 1, 1, 1])
# popt_mine, pcov_mine = curve_fit(voigt_model, x, y_noise, p0=[0, 1, 1, 1])
#
# # Plotting the results
# plt.figure()
# plt.plot(x, y_noise, 'b:', label='Noisy data')
# plt.plot(x, voigt(x, *popt), 'r-', label='Fit')
# plt.plot(x, voigt_model(x, *popt_mine), 'orange', label='My model')
# plt.plot(x, y, 'k--', label='True profile')
# plt.legend()
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from lmfit.models import VoigtModel
#
# # Generate synthetic data
# x = np.linspace(-10, 10, 200)
# true_amplitude = 1340.0
# true_center = 0.0
# true_sigma = 2.0  # For Gaussian
# true_gamma = 3.0  # For Lorentzian
# noise = np.random.normal(0, 0.1, x.size)
# y = true_amplitude * np.exp(-((x - true_center) ** 2) / (2 * true_sigma ** 2))
# y += true_amplitude / (1 + ((x - true_center) / true_gamma) ** 2)
# y += noise
#
# # Create a Voigt model
# model = VoigtModel()
#
# # Create a parameters object and initialize the parameters
# params = model.make_params(amplitude=1500, center=true_center, sigma=1, gamma=1)
#
# # Fit the model to the data
# result = model.fit(y, params, x=x)
#
# # Plot the data and the fit
# plt.plot(x, y, 'b', label='data')
# plt.plot(x, result.best_fit, 'r-', label='fit')
# plt.legend(loc='best')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Voigt Profile Fit')
# plt.show()
#
#
# # import numpy as np
# # from astropy.io import fits
# # import lime
# #
#
# Address of the Green Pea galaxy spectrum
# fits_file = '/home/vital/Desktop/gp121903_osiris.fits'
# z_obj, normFlux = 0.19531, 1e-18
# gp_spec = lime.Spectrum.from_file(fits_file, instrument='osiris', redshift=z_obj, norm_flux=normFlux)
# # gp_spec.plot.spectrum(label='GP121903')
#
# # gp_spec.fit.bands('H1_4861A')
# # gp_spec.plot.bands('H1_4861A')
# gp_spec.fit.bands('H1_4861A_p-v')
# gp_spec.plot.bands('H1_4861A_p-v', y_scale='linear')
# # gp_spec.save_log('/home/vital/Desktop/profiles_tests.txt')
# #
# gp_spec.fit.bands('H1_4861A')
# gp_spec.plot.bands('H1_4861A')
# gp_spec.save_log('profiles_tests.txt')

# # Run the fit
# gp_spec.fit.bands(line, band_edges)
#
# # Plot the results from the last fitting
# gp_spec.plot.bands()
#
# # Fit configuration
# line = 'H1_6563A_b'
# fit_conf = {'H1_6563A_b': 'H1_6563A+N2_6584A+N2_6548A',
#             'N2_6548A_amp': {'expr': 'N2_6584A_amp/2.94'},
#             'N2_6548A_kinem': 'N2_6584A'}
#
# # Second attempt including the fit configuration
# gp_spec.fit.bands(line, band_edges, fit_conf)
# gp_spec.plot.bands(line)
#
# # You can also save the fitting plot to a file
# gp_spec.plot.bands(output_address=f'../sample_data/{line}.png')
#
# # Each fit is stored in the lines dataframe (log) attribute
# print(gp_spec.log)
#
# # It can be saved into different types of document using the function
# gp_spec.save_log('../sample_data/example1_linelog.txt')
# gp_spec.save_log('../sample_data/example1_linelog.pdf', param_list=['eqw', 'gauss_flux', 'gauss_flux_err'])
# gp_spec.save_log('../sample_data/example1_linelog.fits', page='GP121903')
# gp_spec.save_log('../sample_data/example1_linelog.xlsx', page='GP121903')
# gp_spec.save_log('../sample_data/example1_linelog.asdf', page='GP121903')
#
# # A lines log can also be saved/loaded using the lime functions:
# log_address = '../sample_data/example1_linelog.fits'
# lime.save_log(gp_spec.log, log_address, page='GP121903')
# log = lime.load_log(log_address, page='GP121903')
