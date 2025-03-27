from pathlib import Path
import lime

# State the data files
obsFitsFile = Path('../sample_data/spectra/gp121903_osiris.fits')
instrMaskFile = Path('../sample_data/osiris_bands.txt')
ref_bands_file = Path('../sample_data/bands/lines_star_forming_galaxies_optical.txt')
obj_bands_file = Path('../sample_data/bands/gp121903_osiris_bands.fits')

# Create the Spectrum object
z_obj = 0.19531
norm_flux = 1e-18
gp_spec = lime.Spectrum.from_file(obsFitsFile, instrument='osiris', redshift=z_obj, norm_flux=norm_flux)

# Generate a bands file taking into account the observation resolution and wavelength range
bands_df = gp_spec.retrieve.line_bands(ref_bands=ref_bands_file)
gp_spec.plot.spectrum(bands=bands_df, log_scale=True)

# Dictionary with grouped profiles
line_components = {'O2_3726A_b': 'O2_3726A+O2_3729A',
                    'H1_3889A_m': "H1_3889A+He1_3889A",
                    'Ne3_3968A_m': 'Ne3_3968A+H1_3970A',
                    'Ar4_4711A_m': 'Ar4_4711A+He1_4713A',
                    'He1_4922A_m': 'He1_4922A+Fe3_4925A',
                    'N1_5198A_m': 'N1_5198A+N1_5200A',
                    'H1_6563A_b': 'H1_6563A+N2_6583A+N2_6548A',
                    'S2_6716A_b': 'S2_6716A+S2_6731A',
                    'O2_7319A_m': 'O2_7319A+O2_7330A'}

bands_df = gp_spec.retrieve.line_bands(ref_bands=ref_bands_file, fit_conf=line_components)
gp_spec.plot.spectrum(bands=bands_df, log_scale=True)

if not obj_bands_file.is_file():
    lime.save_frame(obj_bands_file, bands_df)

# Further manual adjustment
gp_spec.check.bands(obj_bands_file, ref_bands=ref_bands_file, exclude_continua=True, maximize=False)

gp_spec.plot.spectrum(bands=obj_bands_file)

# # Adding a redshift file address to store the variations in redshift
# redshift_file = '../sample_data/redshift_log.txt'
# redshift_file_header, object_ref = 'redshift', 'GP121903'
# gp_spec.check.bands(bands_df_file, maximize=False, z_log_address=redshift_file, z_column=redshift_file_header,
#                     object_label='gp121903', exclude_continua=True)
#
