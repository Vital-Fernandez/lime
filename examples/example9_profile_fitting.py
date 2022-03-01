import lime
import wget
import gzip
import shutil
import numpy as np
from astropy.io import fits
from pathlib import Path


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

# State the data location
spatial_mask = './sample_data/SHOC579_mask.fits'
cfgFile = './sample_data/config_file.cfg'

# Get the galaxy data
obs_cfg = lime.load_cfg(cfgFile)
z_SHOC579 = obs_cfg['SHOC579_data']['redshift']
norm_flux = obs_cfg['SHOC579_data']['norm_flux']
noise_region = obs_cfg['SHOC579_data']['noise_interval_array']

# Open the cube fits file
with fits.open(SHOC579_cube_address) as hdul:
    wave = hdul['WAVE'].data
    flux = hdul['FLUX'].data * norm_flux
    hdr = hdul['FLUX'].header

# Output data declaration:
log_address = './sample_data/SHOC579_log.fits'
hdul_log = fits.HDUList([fits.PrimaryHDU()])

# WCS header data
hdr_coords = {}
for key in lime.COORD_ENTRIES:
    if key in hdr:
        hdr_coords[key] = hdr[key]
hdr_coords = fits.Header(hdr_coords)

# Boolean check to plot the steps
verbose = False

# Counting the number of voxels and lines
n_voxels, n_lines = 0, 0

# Loop through the masks:
for idx_region in [0, 1, 2]:

    # Load the region spatial mask:
    region_label = f'S2_6716A_B_MASK_{idx_region}'
    region_mask = fits.getdata(spatial_mask, region_label, ver=1)
    region_mask = region_mask.astype(bool)
    n_voxels += np.sum(region_mask)

    # Convert the mask into an array of spaxel coordinates (idxY, idxX)
    idcs_voxels = np.argwhere(region_mask)

    # Load the region spectral mask:
    mask_log_file = f'./sample_data/SHOC579_region{idx_region}_maskLog.txt'
    mask_log = lime.load_lines_log(mask_log_file)

    mask_log = mask_log.rename(index={'He2_4686A': 'He2_4686A_b'})
    mask_log.loc['He2_4686A_b', 'w1':'w6'] = np.array([4630.507567,  4645.238986, 4653.613936, 4748.658238, 4757.090000, 4776.100000])

    # Load the region fitting configuration
    region_fit_cfg = obs_cfg[f'tests_line_fitting']

    # Loop through the spaxels
    print(f'- Treating region {idx_region}')
    for idx_spaxel, coords_spaxel in enumerate(idcs_voxels):

        # Define a spectrum object for the current spaxel
        idxY, idxX = coords_spaxel
        spaxel_spec = lime.Spectrum(wave, flux[:, idxY, idxX], redshift=z_SHOC579, norm_flux=norm_flux)

        # Limit the line fittings to those detected
        peaks_table, matched_mask_log = spaxel_spec.match_line_mask(mask_log, noise_region)
        n_lines += len(matched_mask_log.index)

        # Loop through the detected lines
        print(f'-- Treating spaxel {idx_spaxel}')
        # for idx_line, line in enumerate(matched_mask_log.index):
        for line in ['H1_6563A_b', 'O2_3726A_b']:
            wave_regions = matched_mask_log.loc[line, 'w1':'w6'].values
            spaxel_spec.fit_from_wavelengths(line, wave_regions, fit_method='least_squares', user_cfg=region_fit_cfg)
            spaxel_spec.display_results(fit_report=True)
