import numpy as np
import lime
from lime.io import _LOG_EXPORT_DICT
from pathlib import Path

# Outputs
file_address = 'manga_spaxel.txt'
cube_plot_address = 'cube_manga_plot.png'
spectrum_plot_address = 'spectrum_manga_spaxel.png'
line_plot_address = 'Fe3_4658A_manga_spaxel.png'
line_bands_file = f'manga_line_bands.txt'
lines_log_file = f'manga_lines_log.txt'
cfg_file = 'baseline/manga.toml'

wave_array, flux_array, err_array = np.loadtxt(file_address, unpack=True)
pixel_mask = np.isnan(err_array)

# Parameters
redshift = 0.0475
norm_flux = 1e-17

cfg = lime.load_cfg(cfg_file)
spec = lime.Spectrum(wave_array, flux_array, err_array, redshift=redshift, norm_flux=norm_flux, pixel_mask=pixel_mask)
# spec.plot.spectrum()
spec.fit.frame(line_bands_file, cfg, obj_conf_prefix='38-35')
spec.plot.spectrum(include_fits=True)