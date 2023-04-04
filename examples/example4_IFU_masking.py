import lime
import urllib.request
from astropy.io import fits
from pathlib import Path
from astropy.wcs import WCS

# # Progress bar for the cube download
# def show_progress(block_num, block_size, total_size):
#     print(round(block_num * block_size / total_size *100, 2), end="\r")

# Web address and save file location (it may take some time)
cube_url = 'https://data.sdss.org/sas/dr17/manga/spectro/redux/v3_1_1/8626/stack/manga-8626-12704-LOGCUBE.fits.gz'
cube_address = Path('./sample_data/manga-8626-12704-LOGCUBE.fits.gz')

# Download the file (it may take some time) if it does not exist.
if not cube_address.is_file():
    urllib.request.urlretrieve(cube_url, cube_address)

# Open the MANGA cube fits file
with fits.open(cube_address) as hdul:
    wave = hdul['WAVE'].data
    flux_cube = hdul['FLUX'].data * 1e-17
    hdr = hdul['FLUX'].header

# Load the configuration file:
cfgFile = './sample_data/config_file.cfg'
obs_cfg = lime.load_cfg(cfgFile)
z_SHOC579 = obs_cfg['SHOC579_data']['redshift']
norm_flux = obs_cfg['SHOC579_data']['norm_flux']

# Define a LiMe cube object
shoc579 = lime.Cube(wave, flux_cube, redshift=z_SHOC579, norm_flux=norm_flux)
9
# Extracting the world coordinate system from the header to display on the plots
wcs = WCS(hdr)

# Check the spaxels interactively
shoc579.plot.cube(6563, line_fg=4363, wcs=wcs)

# Generate a spatial mask as a function of the signal to noise
spatial_mask = './sample_data/SHOC579_mask.fits'
shoc579.spatial_masker(4363, param='SN_line', percentiles=[93, 95, 98], header_dict=hdr, output_address=spatial_mask)
shoc579.plot.cube(6563, masks_file=spatial_mask, wcs=wcs, maximise=True)

# # Generate a spatial mask as a function of the signal to noise
# spatial_mask = './sample_data/SHOC579_mask.fits'
# shoc579.check.cube(6563, wcs=WCS(hdr), maximise=False, masks_file=spatial_mask)

