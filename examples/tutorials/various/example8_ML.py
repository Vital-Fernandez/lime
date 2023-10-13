import numpy as np
import lime
import joblib
from lime.io import _LOG_EXPORT_DICT
from pathlib import Path
from matplotlib import pyplot as plt, rc_context
from lime.recognition import MACHINE_PATH
from lime.plots import STANDARD_PLOT

# Outputs
file_address = '../../../tests/data_tests/manga_spaxel.txt'
line_bands_file = '../../../tests/data_tests/manga_line_bands.txt'
cfg_file = '../../../tests/data_tests/manga.toml'

wave_array, flux_array, err_array = np.loadtxt(file_address, unpack=True)
pixel_mask = np.isnan(err_array)

# Parameters
redshift = 0.0475
norm_flux = 1e-17

cfg = lime.load_cfg(cfg_file)
spec = lime.Spectrum(wave_array, flux_array, err_array, redshift=redshift, norm_flux=norm_flux, pixel_mask=pixel_mask)
# spec.fit.frame(line_bands_file, cfg, id_conf_prefix='38-35')
# spec.plot.spectrum(include_fits=True)

spec.fit.continuum(plot_steps=False)

norm_spec = np.log10((spec.flux/spec.cont - 1) + 10)

detect_mask = spec.ml_line_detection(norm_spec, 11)

# 'figure.figsize': (11, 6),
#                   'axes.titlesize': 14,
#                   'axes.labelsize': 14,
#                   'legend.fontsize': 12,
#                   'xtick.labelsize': 12,
#                   'ytick.labelsize': 12}

STANDARD_PLOT['axes.titlesize'] = 30
STANDARD_PLOT['axes.labelsize'] = 30
STANDARD_PLOT['xtick.labelsize'] = 25
STANDARD_PLOT['ytick.labelsize'] = 25
STANDARD_PLOT['legend.fontsize'] = 25

with rc_context(STANDARD_PLOT):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.step(spec.wave, norm_spec)
    ax.scatter(spec.wave[detect_mask], norm_spec[detect_mask], marker='o', color='palegreen', label='ML line detection')
    ax.set_ylabel('Normalized flux')
    ax.set_xlabel(r'Wavelength ($\AA$)')
    ax.legend()
    plt.show()
