from pathlib import Path
import lime

# State the data files
obsFitsFile = Path('../0_resources/spectra/gp121903_osiris.fits')
instrMaskFile = Path('../0_resources/osiris_bands.txt')
ref_bands_file = Path('../0_resources/bands/lines_star_forming_galaxies_optical.txt')
obj_bands_file = Path('../0_resources/bands/gp121903_osiris_bands.txt')
# obj_bands_file = Path('../0_resources/bands/gp121903_bands.txt')

# Create the Spectrum object
z_obj = 0.19531
norm_flux = 1e-18
gp_spec = lime.Spectrum.from_file(obsFitsFile, instrument='osiris', redshift=z_obj, norm_flux=norm_flux)

# # Generate a bands file taking into account the observation resolution and wavelength range
# bands_df = gp_spec.retrieve.line_bands(band_vsigma=70, n_sigma=4, instrumental_correction=True,
#                                        map_band_vsigma={'H1_4861A': 140, 'H1_6563A': 140,
#                                                         'O3_4959A': 240, 'O3_5007A': 240})
#
# # Plot the bands (manually cropping a region of interest)
# gp_spec.plot.spectrum(fname=False, bands=bands_df, log_scale=True, rest_frame=False)
# gp_spec.plot.ax.set_xlim((4600 * (1 + gp_spec.redshift), 5050 * (1 + gp_spec.redshift)))
# gp_spec.plot.fig.tight_layout()
# gp_spec.plot.show()

# Dictionary with grouped profiles
line_components = {'O2_3726A_b': 'O2_3726A+O2_3729A',
                   'H1_3889A_m': "H1_3889A+He1_3889A",
                   'Ne3_3968A_m': 'Ne3_3968A+H1_3970A',
                   'Ar4_4711A_m': 'Ar4_4711A+He1_4713A',
                   'He1_4922A_m': 'He1_4922A+Fe3_4925A',
                   'O3_4959A_b': 'O3_4959A+O3_4959A_k-1',
                   'O3_5007A_b': 'O3_5007A+O3_5007A_k-1+He1_5016A',
                   'N1_5198A_m': 'N1_5198A+N1_5200A',
                   'H1_6563A_b': 'H1_6563A+N2_6583A+N2_6548A',
                   'S2_6716A_b': 'S2_6716A+S2_6731A'}

# Generate a bands file taking into account the observation resolution and wavelength range
bands_df = gp_spec.retrieve.line_bands(band_vsigma=70, n_sigma=4, instrumental_correction=True,
                                       map_band_vsigma={'H1_4861A': 140, 'H1_6563A': 140,
                                                        'O3_4959A': 240, 'O3_5007A': 240},
                                       fit_cfg=line_components)

# Plot the bands (manually cropping a region of interest)
gp_spec.plot.spectrum(fname=False, bands=bands_df, log_scale=True, rest_frame=False)
gp_spec.plot.ax.set_xlim((4600 * (1 + gp_spec.redshift), 5050 * (1 + gp_spec.redshift)))
gp_spec.plot.fig.tight_layout()
gp_spec.plot.show()

line_components.update({'O2_7319A_b' : "O2_7319A_m+O2_7330A_m",
                        'O2_7319A_m' : "O2_7319A+O2_7320A",
                        'O2_7330A_m' : "O2_7330A+O2_7331A"})
bands_df = gp_spec.retrieve.line_bands(band_vsigma=70, n_sigma=4, instrumental_correction=True,
                                       map_band_vsigma={'H1_4861A': 140, 'H1_6563A': 140,
                                                        'O3_4959A': 240, 'O3_5007A': 240},
                                       fit_cfg=line_components, automatic_grouping=True)

# Plot the bands (manually cropping a region of interest)
gp_spec.plot.spectrum(fname=False, bands=bands_df, log_scale=True, rest_frame=False)
gp_spec.plot.ax.set_xlim((7200 * (1 + gp_spec.redshift), 7400 * (1 + gp_spec.redshift)))
gp_spec.plot.fig.tight_layout()
gp_spec.plot.show()

# # Plot the bands (manually cropping a region of interest)
# gp_spec.plot.spectrum(fname=False, bands=bands_df, log_scale=True, rest_frame=False)
# gp_spec.plot.ax.set_xlim((4600 * (1 + gp_spec.redshift), 5050 * (1 + gp_spec.redshift)))
# gp_spec.plot.fig.tight_layout()
# gp_spec.plot.show()

# bands_df = gp_spec.retrieve.line_bands(fit_cfg=line_components)
# gp_spec.plot.spectrum(bands=bands_df, log_scale=True)

# if not obj_bands_file.is_file():
#     lime.save_frame(obj_bands_file, bands_df)

# Further manual adjustment
gp_spec.check.bands('../0_resources/bands/coso_nuevo3.txt', band_vsigma=70, n_sigma=4, instrumental_correction=True,
                    map_band_vsigma={'H1_4861A': 140, 'H1_6563A': 140, 'O3_4959A': 240, 'O3_5007A': 240},
                    fit_cfg=line_components, automatic_grouping=True, maximize=True)

# gp_spec.plot.spectrum(bands=obj_bands_file)

# # Adding a redshift file address to store the variations in redshift
# redshift_file = '../0_resources/redshift_log.txt'
# redshift_file_header, object_ref = 'redshift', 'GP121903'
# gp_spec.check.bands(bands_df_file, maximize=False, z_log_address=redshift_file, z_column=redshift_file_header,
#                     object_label='gp121903', exclude_continua=True)
#