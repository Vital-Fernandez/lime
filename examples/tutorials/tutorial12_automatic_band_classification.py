import lime

# fname = r'../sample_data/spectra/hlsp_ceers_jwst_nirspec_nirspec10-001027_comb-mgrat_v0.7_x1d-masked.fits'
# cfg_fname = r'/home/vital/PycharmProjects/CAPERS/CAPERS_v3.toml'
#
# spec = lime.Spectrum.from_file(fname, 'nirspec', redshift=7.8189)
# spec.unit_conversion('AA', 'FLAM')
# # spec.plot.spectrum()
#
# bands = spec.retrieve.line_bands(fit_cfg=cfg_fname, default_cfg_prefix='default_prism')

fname = r'../sample_data/spectra/gp121903_osiris.fits'
cfg_fname = r'../sample_data/long_slit.toml'
cfg = lime.load_cfg(cfg_fname)

spec = lime.Spectrum.from_file(fname, 'osiris', redshift=0.19531)

bands = spec.retrieve.line_bands(fit_cfg=cfg_fname, automatic_grouping=True)

line = lime.Line('O2_7319A_b', band=bands, fit_conf=cfg['default_line_fitting'])

bands_O2 = bands.loc['O2_7319A', 'w1':'w6'].to_numpy()
# print(bands_O2)
bands_O2[3] = 7348.8


# spec.plot.bands('O2_7319A', bands=bands_O2, rest_frame=True)
spec.fit.bands('O2_7319A_b', bands_O2, cfg)
print(spec.frame.group_label)
spec.plot.bands()



# spec.fit.bands('O2_7319A_b', )
# print(line.list_comps)
# spec.plot.spectrum(bands=bands, rest_frame=True)
