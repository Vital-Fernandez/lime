import lime
import wget
import gzip
import shutil
from astropy.io import fits
from astropy.wcs import WCS
from pathlib import Path
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


lime.spectral_bands()

# Web link and saving location
data_folder = Path('./sample_data/')
SHOC579_url = 'https://data.sdss.org/sas/dr17/manga/spectro/redux/v3_1_1/8626/stack/manga-8626-12704-LOGCUBE.fits.gz'
SHOC579_gz_address = data_folder/'manga-8626-12704-LOGCUBE.fits.gz'

# Download the data (it may take some time)
fetch_spec(SHOC579_gz_address, SHOC579_url)

# Extract the gz file
SHOC579_cube_address = data_folder/'manga-8626-12704-LOGCUBE.fits'
extract_gz_file(SHOC579_gz_address, SHOC579_cube_address)

# State the data location
spatial_mask = data_folder/'SHOC579_mask.fits'
cfgFile = data_folder/'config_file.cfg'
bands_frame_file = data_folder/'SHOC579_region1_maskLog.txt'

# Get the galaxy data
obs_cfg = lime.load_cfg(cfgFile)
z_SHOC579 = obs_cfg['SHOC579_data']['redshift']
norm_flux = obs_cfg['SHOC579_data']['norm_flux']

# Open the cube fits file
with fits.open(SHOC579_cube_address) as hdul:
    wave = hdul['WAVE'].data
    flux = hdul['FLUX'].data * norm_flux
    hdr = hdul['FLUX'].header

# Output data declaration:
hdul_log = fits.HDUList([fits.PrimaryHDU()])

# WCS header data
hdr_coords = {}
for key in COORD_ENTRIES:
    if key in hdr:
        hdr_coords[key] = hdr[key]
hdr_coords = fits.Header(hdr_coords)

# Define a LiMe cube object
shoc579 = lime.Cube(wave, flux, redshift=z_SHOC579, norm_flux=norm_flux)

# Check the input cube and spatial masks previously calculated
shoc579.check.cube(6563, wcs=WCS(hdr), masks_file=spatial_mask)

# Fit the lines on the stored masked
log_address = './sample_data/SHOC579_log.fits'
shoc579.fit.spatial_mask(spatial_mask, bands_frame=bands_frame_file, fit_conf=obs_cfg, line_detection=True,
                         output_log=log_address, progress_output=True)
shoc579.check.cube(6563, wcs=WCS(hdr), masks_file=spatial_mask, lines_log_address=log_address)