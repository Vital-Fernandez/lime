from pathlib import Path
import lime

# State the data files
data_folder = Path('../doc_notebooks/0_resources/')
obsFitsFile = f'{data_folder}/spectra/SHOC579_SDSS_dr18.fits'
lineBandsFile = f'{data_folder}/bands/lines_star_forming_galaxies_optical.txt'

cfgFile = f'{data_folder}/long_slit.toml'
osiris_gp_df_path =  f'{data_folder}/bands/osiris_green_peas_linesDF.txt'

# Load configuration
obs_cfg = lime.load_cfg(cfgFile)

# Declare LiMe spectrum
spec = lime.Spectrum.from_file(obsFitsFile, instrument='sdss')

# # Revised bands for every object
# bands_df = spec.retrieve.lines_frame(band_vsigma=100, map_band_vsigma={'O2_3726A': 200, 'O2_3729A': 200,
#                                                                        'H1_4861A': 200, 'H1_6563A': 200,
#                                                                        'N2_6548A': 200, 'N2_6583A': 200,
#                                                                        'O3_4959A': 200, 'O3_5007A': 180},
#                                      fit_cfg=obs_cfg, obj_cfg_prefix=f'SHOC579_sdss',
#                                      automatic_grouping=True)
#
# output_frame = f'./SHOC579_sdss.txt'
# lime.save_frame(output_frame, bands_df)

output_frame = f'./SHOC579_sdss.txt'
# spec.plot.spectrum(bands=output_frame)

# # Fit the continuum
# gp_spec.fit.continuum(degree_list=[3, 6, 6], emis_threshold=[3, 2, 1.5], smooth_scale=2, plot_steps=True)
#
# # Confirm present lines
# match_bands = gp_spec.infer.peaks_troughs(lineBandsFile, emission_shape=True, sigma_threshold=3, plot_steps=True)

# Fit the lines
spec.fit.frame(output_frame, obs_cfg, obj_cfg_prefix='SHOC579_sdss', update_default=True)
# spec.plot.spectrum(log_scale=True)
spec.save_frame(f'./SHOC579_sdss_measurements.txt')