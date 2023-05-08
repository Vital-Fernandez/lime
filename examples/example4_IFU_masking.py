import lime
import urllib.request
from astropy.io import fits
from pathlib import Path
from astropy.wcs import WCS
from sys import stdout


# Function to display the download progress on the terminal
def progress_bar(count, block_size, total_size):
    percent = int(count * block_size * 100 / total_size)
    stdout.write("\rDownloading...%d%%" % percent)
    stdout.flush()


# MANGA cube web link and save file location
cube_url = 'https://data.sdss.org/sas/dr17/manga/spectro/redux/v3_1_1/8626/stack/manga-8626-12704-LOGCUBE.fits.gz'
cube_address = Path('./sample_data/manga-8626-12704-LOGCUBE.fits.gz')

# Download the cube file if not available (this may take some time)
if cube_address.is_file() is not True:
    urllib.request.urlretrieve(cube_url, cube_address, reporthook=progress_bar)
    print(' Download completed!')
else:
    print('Observation found in folder')

# Load the configuration file:
cfgFile = './sample_data/manga.cfg'
obs_cfg = lime.load_cfg(cfgFile)

# Observation properties
z_obj = obs_cfg['SHOC579']['redshift']
norm_flux = obs_cfg['SHOC579']['norm_flux']

# Open the MANGA cube fits file
with fits.open(cube_address) as hdul:
    wave = hdul['WAVE'].data
    flux_cube = hdul['FLUX'].data * norm_flux
    hdr = hdul['FLUX'].header

# Declaring the world coordinate system from the header to display on the plots
wcs = WCS(hdr)

# Define a LiMe cube object
shoc579 = lime.Cube(wave, flux_cube, redshift=z_obj, norm_flux=norm_flux)

# Check the spaxels interactively
shoc579.plot.cube(6563, line_fg=4363, wcs=wcs)

# Generate a spatial mask as a function of the signal to noise
spatial_mask = './sample_data/SHOC579_mask.fits'
shoc579.spatial_masker('O3_4363A', param='SN_line', percentiles=[93, 96, 99], output_address=spatial_mask,
                       header_dict=hdr)

# We can visualize this mask using the .plot.cube function
shoc579.plot.cube('H1_6563A', masks_file=spatial_mask, wcs=wcs)

# Manually add/remove spaxels to the spatial mask
shoc579.check.cube('H1_6563A', masks_file=spatial_mask, wcs=wcs)

