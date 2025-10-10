import lime

# State the data files
data_folder = '../0_resources/spectra'
obsFitsFile = f'{data_folder}/gp121903_osiris.fits'
lineBandsFile = '../0_resources/bands/gp121903_bands.txt'
cfgFile = '../0_resources/long_slit.toml'
osiris_gp_df_path = '../0_resources/bands/osiris_green_peas_linesDF.txt'

# Load configuration
obs_cfg = lime.load_cfg(cfgFile)
z_obj = obs_cfg['osiris']['gp121903']['z']
norm_flux = obs_cfg['osiris']['norm_flux']

# Declare LiMe spectrum
gp_spec = lime.Spectrum.from_file(obsFitsFile, instrument='osiris', redshift=z_obj, norm_flux=norm_flux)

# Fit the continuum
gp_spec.fit.continuum(degree_list=[3, 6, 6], emis_threshold=[3, 2, 1.5], smooth_scale=2, plot_steps=False)

# Confirm present lines
match_bands = gp_spec.infer.peaks_troughs(lineBandsFile, emission_shape=True, sigma_threshold=3, plot_steps=False)

# Fit the lines
gp_spec.fit.frame(match_bands, obs_cfg, obj_cfg_prefix='gp121903_osiris', update_default=True)
gp_spec.plot.spectrum()

# Instrument - file dictionary
files_dict = {'osiris': 'gp121903_osiris.fits', 'isis': 'IZW18_isis.fits',
              'nirspec':'hlsp_ceers_jwst_nirspec_nirspec10-001027_comb-mgrat_v0.7_x1d-masked.fits', 'sdss':'SHOC579_SDSS_dr18.fits'}

# Instrument - object dictionary
object_dict = {'osiris':'gp121903', 'nirspec':'ceers1027', 'isis':'Izw18', 'sdss':'SHOC579'}

# Loop through the observations
for i, items in enumerate(object_dict.items()):

    inst, obj = items
    file_path = f'{data_folder}/{files_dict[inst]}'
    redshift = obs_cfg[inst][obj]['z']
    print('\n', obj, inst, redshift)

    # Create the observation object
    spec = lime.Spectrum.from_file(file_path, inst, redshift=redshift)

    # Unit conversion for NIRSPEC object
    if spec.units_wave != 'AA':
        spec.unit_conversion('AA', 'FLAM')

    # Revised bands for every object
    bands_df = spec.retrieve.lines_frame(band_vsigma = 100, map_band_vsigma = {'O2_3726A': 200, 'O2_3729A': 200,
                                                                               'H1_4861A': 200, 'H1_6563A': 200,
                                                                               'N2_6548A': 200, 'N2_6583A': 200,
                                                                               'O3_4959A': 250, 'O3_5007A': 250},
                                           fit_cfg=obs_cfg, obj_cfg_prefix=f'{obj}_{inst}',
                                           automatic_grouping=True, ref_bands=osiris_gp_df_path)

    # Fit the lines and plot the measurements
    spec.fit.frame(bands_df, fit_cfg=obs_cfg, obj_cfg_prefix=f'{obj}_{inst}', line_detection=True, cont_from_bands=False)

    # Save the measurements
    spec.save_frame(f'../0_resources/results/{obj}_{inst}_line_frame.txt')

    # Plot the profiles.plot.grid()
    spec.plot.grid()

# # Measure the emission lines
# gp_spec.fit.frame(match_bands, obs_cfg, obj_cfg_prefix='gp121903_osiris', line_detection=True)
# gp_spec.bokeh.spectrum()

# # Display the fits on the spectrum
# gp_spec.plot.spectrum(rest_frame=True, log_scale=True)
#
# # Display a grid with the fits
# gp_spec.plot.grid(rest_frame=True, y_scale='auto')
#
# # Save the data
# gp_spec.save_frame('../0_resources/results/example3_linelog.txt')
# gp_spec.save_frame('../0_resources/results/example3_linelog.xlsx', page='GP121903b')
# gp_spec.save_frame('../0_resources/results/example3_linelog.pdf')
# lime.save_frame('../0_resources/results/example3_linelog.fits', gp_spec.frame, page='GP121903b')
