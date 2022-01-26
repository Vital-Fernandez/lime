import numpy as np
from astropy.io import fits
import lime

# Input files
obsFitsFile = './sample_data/gp121903_BR.fits'
instrMaskFile = './sample_data/gp121903_BR_mask.txt'
cfgFile = './sample_data/config_file.cfg'

# Load configuration
sample_cfg = lime.load_cfg(cfgFile, obj_section={'sample_data': 'object_list'})

# Load mask
maskDF = lime.load_lines_log(instrMaskFile)

# Load spectrum
ext = 0
with fits.open('./sample_data/gp121903_BR.fits') as hdul:
    flux, header = hdul[ext].data, hdul[ext].header
w_min, dw, n_pix = header['CRVAL1'], header['CD1_1'], header['NAXIS1']
w_max = w_min + dw * n_pix

wave = np.linspace(w_min, w_max, n_pix, endpoint=False)

# Object properties
z_obj = sample_cfg['sample_data']['z_array'][2]
norm_flux = sample_cfg['sample_data']['norm_flux']

# Run the interative plot
objMaskFile = './sample_data/gp121903_BR_mask_corrected.txt'
# lime.MaskInspector(lines_log_address=objMaskFile, log=maskDF,
#                    input_wave=wave, input_flux=flux, redshift=z_obj, norm_flux=norm_flux)

# To treat just a a few of the lines at a time given, we may slice the log attribute.
lines_log_section = maskDF[:5]
lime.MaskInspector(lines_log_address=objMaskFile, log=lines_log_section,
                   input_wave=wave, input_flux=flux, redshift=z_obj, norm_flux=norm_flux)
lines_log_section = maskDF[6:10]
lime.MaskInspector(lines_log_address=objMaskFile, log=lines_log_section,
                   input_wave=wave, input_flux=flux, redshift=z_obj, norm_flux=norm_flux)
