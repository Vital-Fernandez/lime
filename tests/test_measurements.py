from pathlib import Path
from copy import deepcopy
import numpy as np
import math
import lime
from lime.fitting.lines import signal_to_noise_rola


# Data for the tests
baseline_folder = Path(__file__).parent / 'baseline'
file_address = baseline_folder/'SHOC579_MANGA38-35.txt'
conf_file_address = baseline_folder/'lime_tests.toml'
bands_file_address = baseline_folder/'SHOC579_MANGA38-35_bands.txt'
lines_log_address = baseline_folder/'SHOC579_MANGA38-35_log.txt'

data_folder = Path(__file__).parent.parent/'examples/doc_notebooks/0_resources'
outputs_folder = data_folder/'results'
spectra_folder = data_folder/'spectra'

redshift = 0.0475
norm_flux = 1e-17
cfg = lime.load_cfg(conf_file_address)
cfg_copy = deepcopy(cfg)
tolerance_rms = 5.5

wave_array, flux_array, err_array, pixel_mask = np.loadtxt(file_address, unpack=True)

results_df = lime.load_frame(lines_log_address)

# S/N line computation

def test_default_constant_value():
    result = signal_to_noise_rola(1, 1, 1)
    assert math.isclose(result, 0.4177713791051667, rel_tol=1e-10)

def test_measurements_log():
    amp_arr, cont_sigma, n_pixels = results_df.amp.to_numpy(), results_df.cont_err.to_numpy(), results_df.n_pixels.to_numpy()
    SN_old = results_df.snr_line.to_numpy()
    SN_new = signal_to_noise_rola(amp_arr, cont_sigma, n_pixels)
    np.all(np.isclose(SN_new, SN_old, atol=0.01))
    return

def test_known_case():
    result = signal_to_noise_rola(10, 2, 6)
    expected = (np.sqrt(2 * np.pi) / 6) * 5 * np.sqrt(6)
    assert np.isclose(result, expected, rtol=1e-10)
    assert np.isclose(result, 5.116633, rtol=1e-5)

