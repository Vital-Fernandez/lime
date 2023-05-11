import lime
import wget
import gzip
import shutil
import numpy as np
from astropy.io import fits
from pathlib import Path
from matplotlib import pyplot as plt, cm, colors, patches
from astropy.wcs import WCS


# Function to download the cube if not done already
def fetch_spec(save_address, cube_url):
    if not Path(save_address).is_file():
        wget.download(cube_url, save_address)
    return


# Function to extract the compressed cube if not done already
def extract_gz_file(input_file_address, output_file_address):
    print(output_file_address)
    if not Path(output_file_address).is_file():
        with gzip.open(input_file_address, 'rb') as f_in:
            with open(output_file_address, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)


# Web link and saving location
SHOC579_url = 'https://data.sdss.org/sas/dr17/manga/spectro/redux/v3_1_1/8626/stack/manga-8626-12704-LOGCUBE.fits.gz'
SHOC579_gz_address = './sample_data/manga-8626-12704-LOGCUBE.fits.gz'

# Download the data (it may take some time)
fetch_spec(SHOC579_gz_address, SHOC579_url)

# Extract the gz file
SHOC579_cube_address = './sample_data/manga-8626-12704-LOGCUBE.fits'
extract_gz_file(SHOC579_gz_address, SHOC579_cube_address)

# State the data location (including the spatial mask and lines log from the previous tutorials)
spatial_mask = './sample_data/SHOC579_mask.fits'
cfgFile = './sample_data/config_file.cfg'
log_file = './sample_data/SHOC579_log.fits'

# Get the galaxy data
obs_cfg = lime.load_cfg(cfgFile)
z_SHOC579 = obs_cfg['SHOC579_data']['redshift']
norm_flux = obs_cfg['SHOC579_data']['norm_flux']

# Open the cube fits file
with fits.open(SHOC579_cube_address) as hdul:
    wave = hdul['WAVE'].data
    flux = hdul['FLUX'].data * norm_flux
    hdr = hdul['FLUX'].header

# and the masks file:
mask_file = 'sample_data/SHOC579_MASK0_bands.txt'
mask_log = lime.load_log(mask_file)

# Establish the band image for the plot bacground using Halpha
Halpha_band = mask_log.loc['H1_6563A_b', 'w3':'w4'].values * (1 + z_SHOC579)
idcs_Halpha = np.searchsorted(wave, Halpha_band)
Halpha_image = flux[idcs_Halpha[0]:idcs_Halpha[1], :, :].sum(axis=0)

# Use SII lines as the foreground image contours
SII_band = mask_log.loc['S2_6716A', 'w3':'w4'].values * (1 + z_SHOC579)
idcs_SII = np.searchsorted(wave, SII_band)
SII_image = flux[idcs_SII[0]:idcs_SII[1], :, :].sum(axis=0)

# Establishing the contours intensity using percentiles
percentile_array = np.array([80, 90, 95, 99, 99.9])
SII_contourLevels = np.nanpercentile(SII_image, percentile_array)

# Labels for the axes
ax_conf = {'image': {'xlabel': r'RA', 'ylabel': r'DEC', 'title': f'MANGA SHOC579'}}

# Color normalization for the flux image:
min_flux = np.nanpercentile(Halpha_image, 60)
log_norm_bg = colors.SymLogNorm(linthresh=min_flux, vmin=min_flux, base=10)

# Create the cube object
shoc579 = lime.Cube(wave, flux, norm_flux=norm_flux, redshift=z_SHOC579)

# Plot the cube with the results from the fittings
shoc579.check.cube(6563, line_fg=6716, wcs=WCS(hdr), lines_log_address=log_file, rest_frame=True)