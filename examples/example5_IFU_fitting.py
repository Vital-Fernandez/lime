import lime
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS

# State the data location
cfg_file = './sample_data/manga.cfg'
cube_file = Path('./sample_data/manga-8626-12704-LOGCUBE.fits.gz')
bands_file_1 = Path('./sample_data/SHOC579_region1_maskLog.txt')
spatial_mask_file = './sample_data/SHOC579_mask.fits'
output_lines_log_file = './sample_data/SHOC579_log.fits'

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


# Fit the lines in one spaxel
# spaxel = shoc579.get_spaxel(39, 40)
# spaxel.fit.frame(bands_file_1, obs_cfg['MASK_1_line_fitting'])
# spaxel.plot.spectrum(include_fits=True, rest_frame=True)

# Fit the lines on all the masks
# shoc579.fit.spatial_mask(spatial_mask_file, fit_conf=obs_cfg, line_detection=True, output_log=output_lines_log_file,
#                          progress_output='bar')

# Fit the lines in one mask
# shoc579.fit.spatial_mask(spatial_mask_file, mask_name_list=['MASK_1'], fit_conf=obs_cfg,
#                          line_detection=True, bands_frame=bands_file_1, output_log=output_lines_log_file,
#                          progress_output='bar')

# Review the fittings
shoc579.check.cube('H1_6563A', wcs=WCS(hdr), masks_file=spatial_mask_file, lines_log_address=output_lines_log_file)

