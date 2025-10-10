import lime
from pathlib import Path

# State the data location
cfg_file = '../0_resources/ifu_manga.toml'
cube_address = Path('../0_resources/spectra/manga-8626-12704-LOGCUBE.fits.gz')
bands_file_0 = Path('../0_resources/bands/SHOC579_MASK0_bands.txt')
spatial_mask_file = Path('../0_resources/results/SHOC579_mask_SN_line.fits')
output_lines_log_file = Path('../0_resources/SHOC579_log.fits')

# Load the configuration file:
obs_cfg = lime.load_cfg(cfg_file)

# Observation properties

# Load the Cube
z_obj = obs_cfg['SHOC579']['redshift']
shoc579 = lime.Cube.from_file(cube_address, instrument='manga', redshift=z_obj)
shoc579.check.cube('H1_6563A', masks_file=spatial_mask_file, rest_frame=True)

# Fit the lines in one spaxel
spaxel = shoc579.get_spectrum(38, 35)
spaxel.fit.frame(bands_file_0, cfg_file, obj_cfg_prefix='MASK_0')
spaxel.plot.spectrum(log_scale=True)

# Load the spaxels mask coordinates
masks_dict = lime.load_spatial_mask(spatial_mask_file, return_coords=True)
for i, coords in enumerate(masks_dict['MASK_0']):
    idx_Y, idx_X = coords
    spaxel = shoc579.get_spectrum(idx_Y, idx_Y)
    print(f'Spaxel {idx_Y}, {idx_X}')
    spaxel.fit.frame(bands_file_0, obs_cfg, line_list=['H1_6563A_b'], obj_cfg_prefix='MASK_0', plot_fit=False)

# Fit the lines in all the masks spaxels
shoc579.fit.spatial_mask(spatial_mask_file, fit_cfg=cfg_file, line_detection=True, output_address=output_lines_log_file)

# Check the individual spaxel fitting configuration
spaxel = shoc579.get_spectrum(38, 35)
spaxel.load_frame(output_lines_log_file, page='38-35_LINELOG')
spaxel.plot.bands('He1_5016A')

# Review the fittings
shoc579.check.cube('H1_6563A', lines_file=output_lines_log_file, masks_file=spatial_mask_file)