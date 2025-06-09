import lime

lime.theme.set_style('dark')

# Declare input files
fname = 'sdss_dr18_0358-51818-0504.fits'
cfg_fname = 'shoc579_cfg.toml'
bands_fname = 'shoc579_sdss_bands.txt'

# Load observation
spec = lime.Spectrum.from_file(fname, instrument='SDSS')
data = spec.retrieve.spectrum()

spec.plot.spectrum(rest_frame=True)

# # Generate bands
# bands = spec.retrieve.line_bands(fit_cfg=cfg_fname)
#
# # Manual Review bands
# spec.plot.spectrum(rest_frame=True, bands=bands_fname)
# spec.check.bands(bands_fname, fit_cfg=cfg_fname, exclude_continua=False)
#
#
# # Fit lines
# spec.fit.frame(bands_fname, cfg_fname, cont_from_bands=True)
# spec.plot.spectrum(log_scale=True)
#
# """
# CAPERs/CEERs survey
# Site:Specsy, 156
# 119334 spectrum
# Line bands model
# Fit lines
#
# SHOC579 load results
# Load bands
# Load fluxes
# Extinction
#
# Components detection
# Bayesian model
#
# LiMe new structure
# Mail list/group
#
# """