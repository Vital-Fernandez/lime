from pathlib import Path
import numpy as np
import lime


# State the data files
data_folder = Path('../doc_notebooks/0_resources/')
obsFitsFile = f'{data_folder}/spectra/gp121903_osiris.fits'
lineBandsFile = f'{data_folder}/bands/gp121903_bands.txt'

cfgFile = f'{data_folder}/long_slit.toml'
osiris_gp_df_path =  f'{data_folder}/bands/osiris_green_peas_linesDF.txt'

# Load configuration
obs_cfg = lime.load_cfg(cfgFile)
z_obj = obs_cfg['osiris']['gp121903']['z']
norm_flux = obs_cfg['osiris']['norm_flux']

# Declare LiMe spectrum
gp_spec = lime.Spectrum.from_file(obsFitsFile, instrument='osiris', redshift=z_obj, norm_flux=norm_flux)

fname = f'./stored_masks.txt'
# intervals = np.array([[5000, 5050], [5100, 5150]])
intervals = np.array([[4000, 4100], [3800, 3900]])
gp_spec.check.masks(fname, intervals, log_scale=True, rest_frame=False)