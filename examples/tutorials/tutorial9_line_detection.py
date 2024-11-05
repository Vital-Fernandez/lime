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
shoc579.plot.spectrum(rest_frame=True)

start_time = time()
shoc579.features.detection(show_steps=False)
fit_time = np.round((time()-start_time), 3)
print(f'- completed ({fit_time} seconds)')
shoc579.plot.spectrum(show_categories=True, rest_frame=True)

# redshift = 4.299
# spec_address = '/home/vital/PycharmProjects/ceers-data/data/spectra/CEERs_DR0.9/nirspecDDT/prism/hlsp_ceers_jwst_nirspec_nirspecDDT-001586_prism_dr0.9_x1d.fits'
# spec = lime.Spectrum.from_file(spec_address, instrument='nirspec', redshift=redshift, crop_waves=(0.75, 5.2))
# spec.unit_conversion('AA', 'FLAM')
#
# # spec.plot.spectrum()
#
# start_time = time()
# spec.features.detection(show_steps=False)
# fit_time = np.round((time()-start_time), 3)
# print(f'- completed ({fit_time} seconds)')
# spec.plot.spectrum(show_categories=True)
