import lime

# State the data files
obsFitsFile = './sample_data/gp121903_BR.fits'
lineMaskFile = './sample_data/gp121903_BR_mask.txt'
cfgFile = './sample_data/example_configuration_file.cfg'

# Load configuration
sample_cfg = lime.load_cfg(cfgFile, objList_check=True)

# Load mask
maskDF = lime.load_lines_log(lineMaskFile)

# Load spectrum
wave, flux, header = lime.load_fits(obsFitsFile, instrument='OSIRIS')

# Declare line measuring object
z_obj = sample_cfg['sample_data']['z_array'][2]
norm_flux = sample_cfg['sample_data']['norm_flux']
gp_spec = lime.Spectrum(wave, flux, redshift=z_obj, normFlux=norm_flux)
gp_spec.plot_spectrum()

# Find lines
norm_spec = lime.continuum_remover(gp_spec.wave_rest, gp_spec.flux, noiseRegionLims=sample_cfg['sample_data']['noiseRegion_array'])
obsLinesTable = lime.line_finder(gp_spec.wave_rest, norm_spec, noiseWaveLim=sample_cfg['sample_data']['noiseRegion_array'], intLineThreshold=3)
matchedDF = lime.match_lines(gp_spec.wave_rest, gp_spec.flux, obsLinesTable, maskDF)
gp_spec.plot_spectrum(obsLinesTable=obsLinesTable, matchedLinesDF=matchedDF, specLabel=f'Emission line detection')

# Correct line region
corrected_mask_file = './sample_data/gp121903_BR_mask_corrected.txt'

# Object line fitting configuration
fit_cfg = sample_cfg['gp121903_line_fitting']

# Measure the emission lines
for i, lineLabel in enumerate(matchedDF.index.values):
    wave_regions = matchedDF.loc[lineLabel, 'w1':'w6'].values
    gp_spec.fit_from_wavelengths(lineLabel, wave_regions, user_cfg=fit_cfg)
    # gp_spec.display_results(show_fit_report=True, show_plot=True, log_scale=True, frame='obs')

# Display results
gp_spec.plot_line_grid(gp_spec.linesDF, frame='obs')

# # Save the results
lime.save_line_log(gp_spec.linesDF, 'gp121903_linelog', 'txt')
lime.save_line_log(gp_spec.linesDF, 'gp121903_flux_table', 'pdf')
lime.save_line_log(gp_spec.linesDF, 'gp121903_linelog', 'fits')
lime.save_line_log(gp_spec.linesDF, 'gp121903_linelog', 'xls')


