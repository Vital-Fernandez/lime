import lime
from time import time
import numpy as np
# State the data files
obsFitsFile = '../sample_data/spectra/sdss_dr18_0358-51818-0504.fits'
lineBandsFile = '../sample_data/osiris_bands.txt'
cfgFile = '../sample_data/osiris.toml'

# Load configuration
obs_cfg = lime.load_cfg(cfgFile)
# z_obj = obs_cfg['sample_data']['z_array'][2]
# norm_flux = obs_cfg['sample_data']['norm_flux']

# Declare LiMe spectrum
shoc579 = lime.Spectrum.from_file(obsFitsFile, instrument='sdss')
start_time = time()
shoc579.features.detection_loopless()
fit_time = np.round((time()-start_time), 3)
print(f'- completed ({fit_time} seconds)')
shoc579.plot.spectrum(show_categories=True)
