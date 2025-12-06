import lime
from pathlib import Path

print(f'Version data')
print(lime.__version__)
lime.show_instrument_cfg()
lime.show_profile_parameters()

# Address of the Green Pea galaxy spectrum
data_folder = Path('../doc_notebooks/0_resources/')
fits_file = data_folder/'spectra/gp121903_osiris.fits'

# Galaxy redshift
z_obj = 0.19531

# Generate a spectrum variable from the observation file
gp_spec = lime.Spectrum.from_file(fits_file, instrument='osiris', redshift=z_obj)
gp_spec.plot.spectrum()

# Plot the spectrum
gp_spec.plot.bands('H1_6563A')

# Fit the Halpha line
gp_spec.fit.bands('H1_6563A', cont_source='adjacent')

# Plot the fitted line profile (since we do not specify a line, it will display the last measurement)
gp_spec.plot.bands()

# # These result is not very good. Let's start by adjusting the width of the bands for this line
obj_bands = gp_spec.retrieve.lines_frame(band_vsigma = 450) # km / s

# Fit the line
gp_spec.fit.bands('H1_6563A', obj_bands)
gp_spec.plot.bands()

# Fit configuration
line = 'H1_6563A_b'
fit_conf = {'H1_6563A_b': 'H1_6563A+H1_6563A_k-1+N2_6584A+N2_6548A',        # Line components of the line
            'N2_6548A_amp': {'expr': 'N2_6584A_amp/2.94'},                  # [NII] amplitude constrained by the emissivity ratio
            'N2_6548A_kinem': 'N2_6584A',                                   # Tie the kinematics of the [NII] doublet
            'H1_6563A_k-1_center': {'value':6562, 'min': 6561, 'max':6563}, # Range for the wide Hα value
            'H1_6563A_k-1_sigma': {'expr':'>1.0*H1_6563A_sigma'}}           # Second Hα sigma must be higher than first sigma

# Second attempt including the fit configuration
gp_spec.fit.bands(line, obj_bands, fit_conf)
gp_spec.plot.bands()

# Velocity
gp_spec.plot.velocity_profile('H1_6563A')

# You can also save the fitting plot to a file
gp_spec.plot.bands(fname=data_folder/'results/gp121903_Halpha.png')

# Each fit is stored in the lines dataframe (log) attribute
print(gp_spec.frame.wavelength)

# It can be saved into different types of document using the function
gp_spec.save_frame(data_folder/'results/example1_linelog.txt')
gp_spec.save_frame(data_folder/'results/example1_linelog.pdf', param_list=['eqw', 'profile_flux', 'profile_flux_err'])
gp_spec.save_frame(data_folder/'results/example1_linelog.fits', page='GP121903')
gp_spec.save_frame(data_folder/'results/example1_linelog.xlsx', page='GP121903')

# A lines log can also be saved/loaded using the lime functions:
log_address = data_folder/'results/example1_linelog.fits'
lime.save_frame(log_address, gp_spec.frame, page='GP121903')
log = lime.load_frame(log_address, page='GP121903')
print(log)