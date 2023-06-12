import numpy as np
from astropy.io import fits
import lime


def import_osiris_fits(file_address, ext=0):

    # Open fits file
    with fits.open(file_address) as hdul:
        data, hdr = hdul[ext].data, hdul[ext].header

    w_min, dw, n_pix = hdr['CRVAL1'],  hdr['CD1_1'], hdr['NAXIS1']
    w_max = w_min + dw * n_pix
    wavelength = np.linspace(w_min, w_max, n_pix, endpoint=False)

    return wavelength, data, hdr


# State the data files
obsFitsFile = './sample_data/gp121903_osiris.fits'
lineBandsFile = './sample_data/osiris_bands.txt'
cfgFile = './sample_data/osiris.toml'

# Load spectrum
wave, flux, header = import_osiris_fits(obsFitsFile)

# Load line bands
bands = lime.load_log(lineBandsFile)

# Load configuration
obs_cfg = lime.load_cfg(cfgFile)
z_obj = obs_cfg['sample_data']['z_array'][2]
norm_flux = obs_cfg['sample_data']['norm_flux']

# Declare LiMe spectrum
gp_spec = lime.Spectrum(wave, flux, redshift=z_obj, norm_flux=norm_flux)

# Section with the fitting information:
fit_cfg = obs_cfg['gp121903_line_fitting']

# # Some lines with the multiple components:
# print(fit_cfg['H1_3889A_m'], fit_cfg['S2_6716A_b'])
#
# # Fit and plot of a blended line with two components
# gp_spec.fit.band('S2_6716A_b', bands, fit_cfg)
# gp_spec.plot.band()
#
# # Fit and plot of a line with a parameter using the "expr" :
# print(fit_cfg['N2_6548A_amp'])
# gp_spec.fit.band('H1_6563A_b', bands, fit_cfg)
# N2_flux_ratio = gp_spec.log.loc["N2_6584A", "gauss_flux"]/gp_spec.log.loc["N2_6548A", "gauss_flux"]
# print(f'[NII] doublet gaussian flux ratio: {N2_flux_ratio}')
# gp_spec.plot.band()

# Fit and plot of a blended line with two components
# print(fit_cfg)
# gp_spec.fit.band('O3_5007A_b', bands.loc['O3_5007A', 'w1':'w6'], fit_cfg)
# gp_spec.plot.band("O3_5007A_b", rest_frame=True)

# # Same line fitting but using inequalities
# O3_ineq_cfg = {'O3_5007A_b'         : 'O3_5007A+O3_5007A_k-1',
#                'O3_5007A_k-1_amp'   : {'expr': '<100.0*O3_5007A_amp', 'min': 0.0},
#                'O3_5007A_k-1_sigma' : {'expr': '>2.0*O3_5007A_sigma'}}
# gp_spec.fit.band('O3_5007A_b', bands.loc['O3_5007A', 'w1':'w6'], O3_ineq_cfg)
# gp_spec.plot.band("O3_5007A_b", rest_frame=True)

# Fit and plot of a blended line with two components
# print(fit_cfg)
# gp_spec.fit.band('O3_5007A_b', bands.loc['O3_5007A', 'w1':'w6'], fit_cfg)
# gp_spec.plot.band("O3_5007A_b", rest_frame=True)

gp_spec.fit.band('H1_4861A', bands, fit_cfg)
gp_spec.plot.band()

# Fit of a line importing the kinematics from an external line
Halpha_cfg = {'H1_6563A_b'      : 'H1_6563A+N2_6584A+N2_6548A',
              'H1_6563A_kinem'  : "H1_4861A",
              'N2_6584A_kinem'  : "H1_4861A",
              'N2_6548A_kinem'  : "H1_4861A"}
gp_spec.fit.band('H1_6563A_b', bands, Halpha_cfg)
gp_spec.plot.band()
