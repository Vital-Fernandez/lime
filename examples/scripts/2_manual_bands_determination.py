import lime

# State the files location
obsFitsFile = '../0_resources/spectra/gp121903_osiris.fits'
lineBandsFile = '../0_resources/bands/gp121903_bands_v3.txt'
cfgFile = '../0_resources/long_slit.toml'
osiris_gp_df_path = '../0_resources/bands/osiris_green_peas_linesDF.txt'

# Load configuration
obs_cfg = lime.load_cfg(cfgFile)
z_obj = obs_cfg['osiris']['gp121903']['z']
norm_flux = obs_cfg['osiris']['norm_flux']

# Declare LiMe spectrum
gp_spec = lime.Spectrum.from_file(obsFitsFile, instrument='osiris', redshift=z_obj, norm_flux=norm_flux)

gp_spec.check.bands(lineBandsFile, band_vsigma=100, n_sigma=4, instrumental_correction=True,
                    map_band_vsigma={'H1_4861A': 200, 'H1_6563A': 200, 'N2_6548A': 200, 'N2_6583A': 200,
                                     'O3_4959A': 250, 'O3_5007A': 250},
                    rejected_lines=['Ne5_3426A', 'N2_5755A', 'He1_5016A'],
                    fit_cfg=cfgFile, default_cfg_prefix='default', obj_cfg_prefix='gp121903_osiris',
                    grouped_lines = ['O2_3726A_m', 'O1_6300A_b', 'S2_6716A_b'],
                    automatic_grouping=True,
                    ref_bands=osiris_gp_df_path, maximize=True)



# # Interactive line bands
# gp_spec.check.bands(lineBandsFile, band_vsigma=100, n_sigma=4, instrumental_correction=True,
#                     map_band_vsigma={'H1_4861A': 200, 'H1_6563A': 200,
#                                      'N2_6548A': 200, 'N2_6583A': 200,
#                                      'O3_4959A': 250, 'O3_5007A': 250},
#                     fit_cfg=obs_cfg, ref_bands=osiris_gp_df_path, maximize=True)
#
# # Interactive line and continua bands
# gp_spec.check.bands(lineBandsFile, band_vsigma=100, n_sigma=4, instrumental_correction=True,
#                     map_band_vsigma={'H1_4861A': 200, 'H1_6563A': 200,
#                                      'N2_6548A': 200, 'N2_6583A': 200,
#                                      'O3_4959A': 250, 'O3_5007A': 250},
#                     fit_cfg=obs_cfg, ref_bands=osiris_gp_df_path, show_continua=True, maximize=True)
#


# # State the data files
# obsFitsFile = '../0_resources/spectra/gp121903_osiris.fits'
# lineBandsFile = '../0_resources/bands/gp121903_bands.txt'
# cfgFile = '../0_resources/long_slit.toml'
#
# # Load configuration
# obs_cfg = lime.load_cfg(cfgFile)
# z_obj = obs_cfg['osiris']['gp121903']['z']
# norm_flux = obs_cfg['osiris']['norm_flux']
#
# # Declare LiMe spectrum
# gp_spec = lime.Spectrum.from_file(obsFitsFile, instrument='osiris', redshift=z_obj, norm_flux=norm_flux)
#
# # Interactive line bands adjustment
# gp_spec.check.bands(lineBandsFile, show_continua=False, band_vsigma=70, n_sigma=4)