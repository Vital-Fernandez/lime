import lime
import wget
import gzip
import shutil
import numpy as np
from astropy.io import fits
from pathlib import Path
from matplotlib import pyplot as plt, cm, colors, patches
from astropy.wcs import WCS
from lime.tools import COORD_ENTRIES


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

# Open the MANGA cube fits file
with fits.open(SHOC579_cube_address) as hdul:
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

# Check the spaxels interactively
shoc579.plot.cube(6563, line_fg=4363, wcs=WCS(hdr), maximise=True)

# Generate a spatial mask as a function of the signal to noise
spatial_mask = './sample_data/SHOC579_mask.fits'
shoc579.spatial_masker(4363, param='SN_line', percentiles=[93, 95, 98], header_dict=hdr, output_address=spatial_mask)
# shoc579.plot.cube(6563, masks_file=spatial_mask, wcs=WCS(hdr), maximise=True)

# Generate a spatial mask as a function of the signal to noise
spatial_mask = './sample_data/SHOC579_mask.fits'
shoc579.check.cube(6563, wcs=WCS(hdr), maximise=False, masks_file=spatial_mask)