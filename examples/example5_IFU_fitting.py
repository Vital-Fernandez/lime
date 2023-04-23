import lime
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
import inspect

cont_fit_params = inspect.signature(lime.Spectrum.line_detection).parameters
keys, values = list(cont_fit_params.keys()),  list(cont_fit_params.values())

cont_fit_params = {}
for i, items in enumerate(inspect.signature(lime.Spectrum.line_detection).parameters.items()):
    if i > 0:
        cont_fit_params[items[0]] = items[1]


# State the data location
cfg_file = './sample_data/manga.cfg'
cube_file = Path('./sample_data/manga-8626-12704-LOGCUBE.fits.gz')
bands_file = Path('./sample_data/SHOC579_region1_maskLog.txt')
spatial_mask = './sample_data/SHOC579_mask.fits'

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

# Define a LiMe cube object
shoc579 = lime.Cube(wave, flux_cube, redshift=z_obj, norm_flux=norm_flux)

# Check the input cube and spatial masks previously calculated
# shoc579.check.cube('H1_6563A', wcs=WCS(hdr), masks_file=spatial_mask)

# Fit the lines on the stored masked
log_address = './sample_data/SHOC579_log.fits'
shoc579.fit.spatial_mask(spatial_mask, fit_conf=obs_cfg, line_detection=True,
                         output_log=log_address, progress_output=True)
shoc579.check.cube('H1_6563A', wcs=WCS(hdr), masks_file=spatial_mask, lines_log_address=log_address)


# SHOC579_cube_address = data_folder/'manga-8626-12704-LOGCUBE.fits'
# extract_gz_file(SHOC579_gz_address, SHOC579_cube_address)
#
# # State the data location
# spatial_mask = data_folder/'SHOC579_mask.fits'
# cfgFile = data_folder/'config_file.cfg'
# bands_frame_file = data_folder/'SHOC579_region1_maskLog.txt'
#
# # Get the galaxy data
# obs_cfg = lime.load_cfg(cfgFile)
# z_SHOC579 = obs_cfg['SHOC579_data']['redshift']
# norm_flux = obs_cfg['SHOC579_data']['norm_flux']
#
# # Open the cube fits file
# with fits.open(SHOC579_cube_address) as hdul:
#     wave = hdul['WAVE'].data
#     flux = hdul['FLUX'].data * norm_flux
#     hdr = hdul['FLUX'].header

# Output data declaration:
# hdul_log = fits.HDUList([fits.PrimaryHDU()])

# # WCS header data
# hdr_coords = {}
# for key in COORD_ENTRIES:
#     if key in hdr:
#         hdr_coords[key] = hdr[key]
# hdr_coords = fits.Header(hdr_coords)

# # Define a LiMe cube object
# shoc579 = lime.Cube(wave, flux, redshift=z_SHOC579, norm_flux=norm_flux)

