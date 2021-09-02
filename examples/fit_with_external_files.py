
# Include the library into the python path
import sys
library_folder = '/home/vital/PycharmProjects/lime/src'
sys.path.append(library_folder)

from pathlib import Path
import lime
import pandas as pd

print(pd.__version__)

# Declare the data location
example_path = Path(__file__).parent
obsFitsFile = example_path/'sample_data/gp121903_BR.fits'
lineMaskFile = example_path/'sample_data/gp121903_BR_mask.txt'
cfgFile = example_path/'sample_data/gtc_greenpeas_data.ini'

# Load the data
obsConf = lime.loadConfData(cfgFile, objList_check=True, group_variables=False)
maskDF = lime.lineslogFile_to_DF(lineMaskFile)
wave, flux, header = lime.import_fits_data(obsFitsFile, instrument='OSIRIS')
user_conf = obsConf['gp121903_line_fitting']

# Declare line measuring object
lm = lime.Spectrum(wave, flux, redshift=obsConf['sample_data']['z_array'][2], normFlux=obsConf['sample_data']['norm_flux'])
lm.plot_spectrum()

# Find lines
norm_spec = lime.continuum_remover(lm.wave_rest, lm.flux, noiseRegionLims=obsConf['sample_data']['noiseRegion_array'])
obsLinesTable = lime.line_finder(lm.wave_rest, norm_spec, noiseWaveLim=obsConf['sample_data']['noiseRegion_array'], intLineThreshold=3)
matchedDF = lime.match_lines(lm.wave_rest, lm.flux, obsLinesTable, maskDF)
lm.plot_spectrum(obsLinesTable=obsLinesTable, matchedLinesDF=matchedDF, specLabel=f'Emission line detection')

# Correct line region
corrected_mask_file = Path('./sample_data/gp121903_BR_mask_corrected.txt')

# Load corrected masks
objMaskDF = lime.lineslogFile_to_DF(corrected_mask_file)

# Measure the emission lines
objMaskDF = lime.lineslogFile_to_DF(corrected_mask_file)
for i, lineLabel in enumerate(objMaskDF.index.values):
    wave_regions = objMaskDF.loc[lineLabel, 'w1':'w6'].values
    lm.fit_from_wavelengths(lineLabel, wave_regions, user_conf=user_conf)
    # lm.display_results(show_fit_report=True, show_plot=True, log_scale=True, frame='obs')

# Display results
lm.plot_line_grid(lm.linesDF, frame='obs')
var = lm.flux
var2 = lm.wave

# # Save the results
home_folder = Path.home()
lm.save_lineslog(lm.linesDF, home_folder/'linesLog.txt')
lm.table_fluxes(lm.linesDF, home_folder/'linesTable')



