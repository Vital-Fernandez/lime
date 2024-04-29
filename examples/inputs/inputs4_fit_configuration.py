import numpy as np
from astropy.io import fits
import lime

# State the data files
obsFitsFile = '../sample_data/spectra/gp121903_osiris.fits'
lineBandsFile = '../sample_data//osiris_bands.txt'
cfgFile = '../sample_data/osiris.toml'

# Load line bands
bands = lime.load_frame(lineBandsFile)

# Load configuration
obs_cfg = lime.load_cfg(cfgFile)
z_obj = obs_cfg['sample_data']['z_array'][2]
norm_flux = obs_cfg['sample_data']['norm_flux']

# Declare LiMe spectrum
gp_spec = lime.Spectrum.from_file(obsFitsFile, instrument='osiris', redshift=z_obj, norm_flux=norm_flux)

# # Some lines with the multiple components:
# print(obs_cfg['default_line_fitting']['H1_3889A_m'], obs_cfg['gp121903_line_fitting']['S2_6716A_b'])
#
# # Fit and plot of a blended line with two components
# gp_spec.fit.bands('S2_6716A_b', lineBandsFile, fit_cfg, id_conf_prefix='gp121903')
# gp_spec.plot.bands()
#
# # Fit and plot of a line with a parameter using the "expr" :
# print(fit_cfg['N2_6548A_amp'])
# gp_spec.fit.bands('H1_6563A_b', bands, fit_cfg)
# N2_flux_ratio = gp_spec.frame.loc["N2_6584A", "profile_flux"]/gp_spec.frame.loc["N2_6548A", "profile_flux"]
# print(f'[NII] doublet gaussian flux ratio: {N2_flux_ratio}')
# gp_spec.plot.bands()
#
# # Fit and plot of a blended line with two components
# print(fit_cfg)
# gp_spec.fit.bands('O3_5007A_b', bands.loc['O3_5007A', 'w1':'w6'], obs_cfg['gp121903_line_fitting'])
# gp_spec.plot.bands("O3_5007A_b", rest_frame=True)
#
# # Same line fitting but using inequalities
# O3_ineq_cfg = {'O3_5007A_b'         : 'O3_5007A+O3_5007A_k-1',
#                'O3_5007A_k-1_amp'   : {'expr': '<100.0*O3_5007A_amp', 'min': 0.0},
#                'O3_5007A_k-1_sigma' : {'expr': '>2.0*O3_5007A_sigma'}}
# gp_spec.fit.bands('O3_5007A_b', bands.loc['O3_5007A', 'w1':'w6'], O3_ineq_cfg)
# gp_spec.plot.bands("O3_5007A_b", rest_frame=True)
#
# # Fit and plot of a blended line with two components
# print(fit_cfg)
# gp_spec.fit.bands('O3_5007A_b', bands.loc['O3_5007A', 'w1':'w6'], fit_cfg)
# gp_spec.plot.bands("O3_5007A_b", rest_frame=True)

gp_spec.fit.bands('H1_4861A', lineBandsFile, cfgFile, id_conf_prefix='gp121903')
# gp_spec.plot.bands()

# Fit of a line importing the kinematics from an external line
Halpha_cfg = {'H1_6563A_b'      : 'H1_6563A+N2_6584A+N2_6548A',
              'H1_6563A_kinem'  : "H1_4861A",
              'N2_6584A_kinem'  : "H1_4861A",
              'N2_6548A_kinem'  : "H1_4861A"}
gp_spec.fit.bands('H1_6563A_b', bands, Halpha_cfg)
gp_spec.plot.bands()
