from pathlib import Path
import lime

# Data for the tests
baseline_folder = Path(__file__).parent / 'baseline'
file_address = baseline_folder/'SHOC579_MANGA38-35.txt'
conf_file_address = baseline_folder/'lime_tests.toml'
bands_file_address = baseline_folder/'SHOC579_MANGA38-35_bands.txt'
lines_log_address = baseline_folder/'SHOC579_MANGA38-35_log.txt'
lines_tex_address = baseline_folder/'SHOC579_MANGA38-35_log.tex'


data_folder = Path(__file__).parent.parent/'doc_notebooks/0_resources'
outputs_folder = data_folder/'results'
spectra_folder = data_folder/'spectra'

# State the data files
sdss_fits_fname = f'{spectra_folder}/SHOC579_SDSS_dr18.fits'

# Load configuration
cfgFile = f'{data_folder}/long_slit.toml'
obs_cfg = lime.load_cfg(cfgFile)

# Declare LiMe spectrum
spec = lime.Spectrum.from_file(sdss_fits_fname, instrument='sdss')
bands = spec.retrieve.lines_frame(fit_cfg=obs_cfg, obj_cfg_prefix='SHOC579_sdss', band_vsigma=120)
lime.save_frame(data_folder/'bands'/f'SHOC579_dr18_bands.txt', bands)
spec.plot.spectrum(bands=bands, log_scale=True)

# # Measure the lines
# spec.fit.frame(bands_df, fit_cfg=obs_cfg, obj_cfg_prefix='SHOC579_sdss', line_list=['O3_4959A', 'Ar3_7136A'])
# lime_df = spec.frame.copy()
