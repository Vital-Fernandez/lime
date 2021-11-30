import lime

# State the data files
obsFitsFile = './sample_data/gp121903_BR.fits'
lineMaskFile = './sample_data/gp121903_BR_mask.txt'
cfgFile = './sample_data/example_configuration_file.cfg'

# Selection of reference for the plots
plots_frame = 'obs'

# Load configuration
sample_cfg = lime.load_cfg(cfgFile, objList_check=True)

# Load mask
maskDF = lime.load_lines_log(lineMaskFile)

# Load spectrum
wave, flux, header = lime.load_fits(obsFitsFile, instrument='OSIRIS')

# Declare line measuring object
z_obj = sample_cfg['sample_data']['z_array'][2]
norm_flux = sample_cfg['sample_data']['norm_flux']
gp_spec = lime.Spectrum(wave, flux, redshift=z_obj, norm_flux=norm_flux)
gp_spec.plot_spectrum()

# Find lines
peaks_table, matched_masks_DF = gp_spec.match_line_mask(maskDF, sample_cfg['sample_data']['noiseRegion_array'])
gp_spec.plot_spectrum(obsLinesTable=peaks_table, matchedLinesDF=matched_masks_DF, specLabel=f'Emission line detection',
                      frame=plots_frame)

# Correct line region
corrected_mask_file = './sample_data/gp121903_BR_mask_corrected'

# Object line fitting configuration
fit_cfg = sample_cfg['gp121903_line_fitting']

# Measure the emission lines
for i, lineLabel in enumerate(matched_masks_DF.index.values):
    print(lineLabel)
    wave_regions = matched_masks_DF.loc[lineLabel, 'w1':'w6'].values
    gp_spec.fit_from_wavelengths(lineLabel, wave_regions, user_cfg=fit_cfg)
    # gp_spec.display_results(show_fit_report=True, show_plot=True, log_scale=True, frame='obs')

# Display fits in grid
gp_spec.plot_line_grid(gp_spec.linesDF, frame=plots_frame)

# Display fits in along the spectrum
gp_spec.plot_spectrum(frame=plots_frame, profile_fittings=True)

# # Save the results
lime.save_line_log(gp_spec.linesDF, 'gp121903_linelog', 'txt')
lime.save_line_log(gp_spec.linesDF, 'gp121903_flux_table', 'pdf')
lime.save_line_log(gp_spec.linesDF, 'gp121903_linelog', 'fits')
lime.save_line_log(gp_spec.linesDF, 'gp121903_linelog', 'xls')

