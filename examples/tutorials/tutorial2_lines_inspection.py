from pathlib import Path
import lime

# State the data files
obsFitsFile = '../sample_data/spectra/gp121903_osiris.fits'
instrMaskFile = '../sample_data/osiris_bands.txt'

# Create the Spectrum object
z_obj = 0.19531
norm_flux = 1e-18
gp_spec = lime.Spectrum.from_file(obsFitsFile, instrument='osiris', redshift=z_obj, norm_flux=norm_flux)

# Import the default lines database:
bands_df = gp_spec.retrieve.line_bands()

# Save to a file (if it does not exist already)
bands_df_file = Path('../sample_data/gp121903_bands.txt')

if bands_df_file.is_file() is not True:
    lime.save_frame(bands_df_file, bands_df)

# Review the bands file
gp_spec.check.bands(bands_df_file, maximize=False)

# Adding a redshift file address to store the variations in redshift
redshift_file = '../sample_data/redshift_log.txt'
redshift_file_header, object_ref = 'redshift', 'GP121903'
gp_spec.check.bands(bands_df_file, maximize=False, z_log_address=redshift_file, z_column=redshift_file_header,
                    object_label='gp121903', exclude_continua=True)

