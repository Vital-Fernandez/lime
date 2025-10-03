import lime

lime.show_profile_parameters()

# # Address of the Green Pea galaxy spectrum
# fits_file = '../0_resources/spectra/gp121903_osiris.fits'
#
# # Galaxy redshift
# z_obj = 0.19531
#
# # Generate a spectrum variable from the observation file
# gp_spec = lime.Spectrum.from_file(fits_file, instrument='osiris', redshift=z_obj)
# # gp_spec.plot.spectrum()
# #
# # # Plot the spectrum
# # gp_spec.plot.bands('H1_6563A')
# #
# # # Fit the Halpha line
# # gp_spec.fit.bands('H1_6563A')
# #
# # # Plot the fitted line profile (since we do not specify a line, it will display the last measurement)
# # gp_spec.plot.bands()
# #
# # These result is not very good. Let's start by adjusting the width of the bands for this line
# obj_bands = gp_spec.retrieve.lines_frame(band_vsigma = 450) # km / s
# #
# # # Fit the line
# # gp_spec.fit.bands('H1_6563A', obj_bands)
# # gp_spec.plot.bands()
#
# # Fit configuration
# fit_conf = {'H1_6563A_b': 'H1_6563A+N2_6584A+N2_6548A',
#             'N2_6548A_amp': {'expr': 'N2_6584A_amp/2.94'},
#             'N2_6548A_kinem': 'N2_6584A'}
# gp_spec.fit.bands('H1_6563A_b', obj_bands, fit_cfg=fit_conf)
# gp_spec.plot.bands()

# # You can also save the fitting plot to a file
# gp_spec.plot.bands(output_address=f'../0_resources/results/gp121903_Halpha.png')
#
# # Each fit is stored in the lines dataframe (log) attribute
# print(gp_spec.frame.wavelength)
#
# # It can be saved into different types of document using the function
# gp_spec.save_frame('../0_resources/results/example1_linelog.txt')
# gp_spec.save_frame('../0_resources/results/example1_linelog.pdf', param_list=['eqw', 'profile_flux', 'profile_flux_err'])
# gp_spec.save_frame('../0_resources/results/example1_linelog.fits', page='GP121903')
# gp_spec.save_frame('../0_resources/results/example1_linelog.xlsx', page='GP121903')
#
# # A lines log can also be saved/loaded using the lime functions:
# log_address = '../0_resources/results/example1_linelog.fits'
# lime.save_frame(log_address, gp_spec.frame, page='GP121903')
# log = lime.load_frame(log_address, page='GP121903')
# print(log)