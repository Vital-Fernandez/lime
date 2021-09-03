# Include the library into the python path
import sys
from pathlib import Path
example_file_path = Path(__file__).resolve()
lime_path = example_file_path.parent.parent/'src'
src_folder = str(lime_path)
sys.path.append(src_folder)
import lime
import numpy as np

# Declare the data location
example_path = Path(__file__).parent
obsFitsFile = example_path/'sample_data/gp121903_BR.fits'

# Load spectrum
wave, flux, header = lime.import_fits_data(obsFitsFile, instrument='OSIRIS')
z_obj = 0.19531
norm_flux = 1e-14

# Line label and mask array (in rest frame)
lineLabel = 'H1_6563A_b'
lineWaves = np.array([6438.03, 6508.66, 6535.10, 6600.95, 6627.70, 6661.82])

# Fit configuration
fit_conf = {'H1_6563A_b': 'H1_6563A-N2_6584A-N2_6548A',
            'N2_6548A_amp': {'expr': 'N2_6584A_amp / 2.94'},
            'N2_6548A_kinem': 'N2_6584A'}

# Declare spectrum to analyse
spec1 = lime.Spectrum(wave, flux, redshift=z_obj, normFlux=norm_flux)
spec1.plot_spectrum()

# Run the fit
spec1.fit_from_wavelengths(lineLabel, lineWaves, fit_conf)

# Show the results
spec1.display_results(show_fit_report=True, show_plot=True)
