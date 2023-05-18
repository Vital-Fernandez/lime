import lime
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS

# State the data location
cfg_file = './sample_data/manga.cfg'
cube_file = Path('./sample_data/manga-8626-12704-LOGCUBE.fits.gz')
bands_file_0 = Path('./sample_data/SHOC579_MASK0_bands.txt')
spatial_mask_file = Path('./sample_data/SHOC579_mask.fits')
output_lines_log_file = Path('./sample_data/SHOC579_log.fits')

# Load the configuration file:
obs_cfg = lime.load_cfg(cfg_file)

# Observation properties
z_obj = obs_cfg['SHOC579']['redshift']
norm_flux = obs_cfg['SHOC579']['norm_flux']

# Open the MANGA cube fits file
with fits.open(cube_file) as hdul:
    wave = hdul['WAVE'].data
    flux_cube = hdul['FLUX'].data * norm_flux
    hdr = hdul['FLUX'].header

# World coordinate system from the observation
wcs = WCS(hdr)

# Define a LiMe cube object
shoc579 = lime.Cube(wave, flux_cube, redshift=z_obj, norm_flux=norm_flux, wcs=wcs)

# Fit the lines in one spaxel
spaxel = shoc579.get_spaxel(38, 35)
spaxel.fit.frame(bands_file_0, obs_cfg, line_detection=True, obj_ref='MASK_0')
spaxel.plot.spectrum(rest_frame=True, include_fits=True)

# Load the spaxels mask coordinates
masks_dict = lime.load_spatial_mask(spatial_mask_file, return_coords=True)
for i, coords in enumerate(masks_dict['MASK_0']):
    print(f'Spaxel {i}) Coordinates {coords}')
    idx_Y, idx_X = coords
    spaxel = shoc579.get_spaxel(idx_Y, idx_Y)
    spaxel.fit.frame(bands_file_0, obs_cfg, line_list=['H1_6563A_b'], obj_ref='MASK_0', plot_fit=False, progress_output=None)

# Fit the lines in all the masks spaxels
shoc579.fit.spatial_mask(spatial_mask_file, fit_conf=obs_cfg, line_detection=True, output_log=output_lines_log_file)

# Check the individual spaxel fitting configuration
spaxel = shoc579.get_spaxel(38, 35)
spaxel.load_log(output_lines_log_file, ext='38-35_LINESLOG')
spaxel.plot.band('He1_5016A')

# # Fit the lines in one mask
# shoc579.fit.spatial_mask(spatial_mask_file, bands_file_0, obs_cfg, mask_name_list=['MASK_0'],
#                          line_detection=True,  output_log=output_lines_log_file, progress_output='bar',
#                          plot_fit=False)

# Review the fittings
shoc579.check.cube('H1_6563A', lines_log_address=output_lines_log_file, masks_file=spatial_mask_file)
