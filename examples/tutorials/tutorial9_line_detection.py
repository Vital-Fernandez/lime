import lime
from time import time
from pathlib import Path

fname = '/home/vital/Astrodata/CAPERS/CAPERS_UDS_V0.1/P3/CAPERS_UDS_P3_s000004341_x1d_optext.fits'
z_obj = 2.01
ref_bands = '/home/vital/Dropbox/Astrophysics/Data/CAPERS/data/CAPERS_UDS_V2/CAPERs_prism_lines_v2.txt'
cfg_fname = '/home/vital/PycharmProjects/CAPERS/CAPERS_v2.toml'
cfg = lime.load_cfg(cfg_fname)
spec = lime.Spectrum.from_file(fname, redshift=z_obj, instrument='nirspec')
spec.unit_conversion('AA', 'FLAM')

spec.infer.components()
spec.plot.spectrum(show_categories=True, rest_frame=True)

# obj_bands = spec.retrieve.line_bands(fit_cfg=cfg, default_cfg_prefix='blended_prism',
#                                      band_vsigma=600, obj_cfg_prefix=None)
# spec.plot.spectrum(bands=obj_bands)



# import numpy as np
# lime.theme.set_style('dark')
# # State the data files
# obsFitsFile = '../sample_data/spectra/sdss_dr18_0358-51818-0504.fits'
# lineBandsFile = '../sample_data/osiris_bands.txt'
# cfgFile = '../sample_data/osiris.toml'
#
# # Load configuration
# obs_cfg = lime.load_cfg(cfgFile)
# # z_obj = obs_cfg['sample_data']['z_array'][2]
# # norm_flux = obs_cfg['sample_data']['norm_flux']
#
# # Declare LiMe spectrum
# # shoc579 = lime.Spectrum.from_file(obsFitsFile, instrument='sdss')#, crop_waves=[7600, 7700])
# # # shoc579.plot.spectrum(rest_frame=False)
# #
# # wave, flux, err, z = shoc579.wave, shoc579.flux, shoc579.err_flux, shoc579.redshift
# # pixel_mask = (shoc579.wave > 5300 * (1+z)) & (shoc579.wave < 5400 * (1+z))
#
# # shoc579 = lime.Spectrum(wave, flux, err, z, pixel_mask=pixel_mask)
# # # shoc579.plot.spectrum(rest_frame=False)
# #
# # start_time = time()
# # shoc579.features.detection(show_steps=True, exclude_continuum=False, rest_wl_lim=(7300, 7400))
# # fit_time = np.round((time()-start_time), 3)
# # print(f'- completed ({fit_time} seconds for 3839 segments ({3839/fit_time} lines per second))')
# # shoc579.plot.spectrum(show_categories=True, rest_frame=True)
#
# redshift = 4.299
# spec_address = '/home/vital/PycharmProjects/ceers-data/data/spectra/CEERs_DR0.9/nirspecDDT/prism/hlsp_ceers_jwst_nirspec_nirspecDDT-001586_prism_dr0.9_x1d.fits'
# capers_bands = '/home/vital/Dropbox/Astrophysics/Data/CAPERS/PRISM_ref_bands.txt'
# spec = lime.Spectrum.from_file(spec_address, instrument='nirspec', redshift=redshift, crop_waves=(0.75, 5.2))
# spec.unit_conversion('AA', 'FLAM')
#
# # # spec.plot.spectrum()
# # ax_cfg = {'title': f'Galaxy MSA1586, NIRSPEC PRISM, at z = {redshift}'}
# #
# # start_time = time()
# spec.features.detection(show_steps=False)
# # fit_time = np.round((time()-start_time), 3)
# # print(f'- completed ({fit_time} seconds)')
# obj_bands = spec.retrieve.line_bands(ref_bands=capers_bands, bands_kinematic_width=100)
# # spec.plot.spectrum(bands=obj_bands, show_categories=True, rest_frame=True)
# spec.plot.spectrum(show_categories=True, rest_frame=True, ax_cfg={'title': 'CEERs MSA1586, z=4.29'})
#
#
# # obj_bands = spec.retrieve.line_bands(ref_bands=capers_bands, components_detection=True)
# # spec.plot.spectrum(bands=obj_bands, show_categories=True, rest_frame=True)