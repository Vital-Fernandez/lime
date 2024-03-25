import numpy as np
import lime
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS

# State the data location
cfg_file = '../sample_data/manga.toml'
cube_file = Path('../sample_data/spectra/manga-8626-12704-LOGCUBE.fits.gz')
bands_file_0 = Path('../sample_data/SHOC579_MASK0_bands.txt')
spatial_mask_file = Path('../sample_data/SHOC579_mask.fits')
output_lines_log_file = Path('../sample_data/SHOC579_log.fits')

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
mask_cube = np.isnan(flux_cube)
shoc579 = lime.Cube(wave, flux_cube, redshift=z_obj, norm_flux=norm_flux, wcs=wcs, pixel_mask=mask_cube)

# Fit the lines in one spaxel
spaxel = shoc579.get_spectrum(38, 35)
spaxel.fit.frame(bands_file_0, obs_cfg, line_detection=True, id_conf_prefix='MASK_0')
spaxel.plot.spectrum(rest_frame=True, include_fits=True)

# Load the spaxels mask coordinates
masks_dict = lime.load_spatial_mask(spatial_mask_file, return_coords=True)
for i, coords in enumerate(masks_dict['MASK_0']):
    idx_Y, idx_X = coords
    spaxel = shoc579.get_spectrum(idx_Y, idx_Y)
    spaxel.fit.frame(bands_file_0, obs_cfg, line_list=['H1_6563A_b'], id_conf_prefix='MASK_0', plot_fit=False)

# Fit the lines in all the masks spaxels
shoc579.fit.spatial_mask(spatial_mask_file, fit_conf=obs_cfg,
                         line_detection=True, output_address=output_lines_log_file)

# Check the individual spaxel fitting configuration
spaxel = shoc579.get_spectrum(38, 35)
spaxel.load_log(output_lines_log_file, page='38-35_LINELOG')
spaxel.plot.bands('He1_5016A')

# Review the fittings
shoc579.check.cube('H1_6563A', lines_log_file=output_lines_log_file, masks_file=spatial_mask_file)
