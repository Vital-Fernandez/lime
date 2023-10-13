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
cfg_file = 'manga.toml'

wave_array, flux_array, err_array = np.loadtxt(file_address, unpack=True)
pixel_mask = np.isnan(err_array)

# Parameters
redshift = 0.0475
norm_flux = 1e-17

cfg = lime.load_cfg(cfg_file)
spec = lime.Spectrum(wave_array, flux_array, err_array, redshift=redshift, norm_flux=norm_flux, pixel_mask=pixel_mask)
# spec.plot.spectrum()
spec.fit.frame(line_bands_file, cfg, id_conf_prefix='38-35')
spec.plot.spectrum(include_fits=True)


# log_orig = lime.load_log(lines_log_file)

# extension = 'xlsx'
# test_log_file = f'test_lines_log.{extension}'
# spec.fit.frame(line_bands_file, cfg, id_conf_prefix='38-35')
# spec.save_log(test_log_file, page='NewLinesLog')
#
# log_orig = lime.load_log(lines_log_file)
# log_test = lime.load_log(test_log_file)
#
# for line in spec.log.index:
#     for param in spec.log.columns:
#
#         # String
#         if _LOG_EXPORT_DICT[param].startswith('<U'):
#             if log_orig.loc[line, param] is np.nan:
#                 assert log_orig.loc[line, param] is log_test.loc[line, param]
#             else:
#                 assert log_orig.loc[line, param] == log_test.loc[line, param]
#
#         # Float
#         else:
#             if param not in ['eqw', 'eqw_err']:
#                 # print('\n', param, log_orig.loc[line, param], log_test.loc[line, param])
#                 assert np.allclose(log_orig.loc[line, param], log_test.loc[line, param], rtol=0.05,
#                                    equal_nan=True)
#             else:
#                 assert np.allclose(log_orig.loc[line, param], log_test.loc[line, param], rtol=0.15,
#                                    equal_nan=True)


# for ext in ['txt', 'fits', 'csv', 'xlsx', 'asdf']:
#
#     spec.save_log(f'test_frame.{ext}')
#     log_test = lime.load_log(f'test_frame.{ext}')
#
#     for line in log_orig.index:
#
#         for param in log_orig.columns:
#
#             # String
#             if _LOG_EXPORT_DICT[param].startswith('<U'):
#                 if log_orig.loc[line, param] is np.nan:
#                     check_A = log_orig.loc[line, param] is log_test.loc[line, param]
#                 else:
#                     check_A = log_orig.loc[line, param] == log_test.loc[line, param]
#
#             # Float
#             else:
#                 if param != 'eqw_err':
#                     check_A = np.allclose(log_orig.loc[line, param], log_test.loc[line, param], rtol=0.02, equal_nan=True)
#                 else:
#                     check_A = np.allclose(log_orig.loc[line, param], log_test.loc[line, param], rtol=0.15, equal_nan=True)
#
#             if check_A is False:
#                 print(line, param, ext)
#                 print(f'-- Fail): {log_orig.loc[line, param]} != {log_test.loc[line, param]}')
#                 # raise 'Coso'

