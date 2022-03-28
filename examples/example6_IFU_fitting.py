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

    # Load the region fitting configuration
    region_fit_cfg = obs_cfg[f'SHOC579_region{idx_region}_line_fitting']

    # Loop through the spaxels
    print(f'- Treating region {idx_region}')
    for idx_spaxel, coords_spaxel in enumerate(idcs_voxels):

        # Define a spectrum object for the current spaxel
        idxY, idxX = coords_spaxel
        spaxel_spec = lime.Spectrum(wave, flux[:, idxY, idxX], redshift=z_SHOC579, norm_flux=norm_flux)

        if verbose:
            spaxel_spec.plot_spectrum(spec_label=f'SHOC579 spaxel {idxY}-{idxX}')

        # Limit the line fittings to those detected
        peaks_table, matched_mask_log = spaxel_spec.match_line_mask(mask_log, noise_region)
        n_lines += len(matched_mask_log.index)

        if verbose:
            spaxel_spec.plot_spectrum(peaks_table=peaks_table, match_log=matched_mask_log,
                                      spec_label=f'SHOC579 spaxel {idxY}-{idxX}')

        # Loop through the detected lines
        print(f'-- Treating spaxel {idx_spaxel}')
        for idx_line, line in enumerate(matched_mask_log.index):

            wave_regions = matched_mask_log.loc[line, 'w1':'w6'].values
            try:
                spaxel_spec.fit_from_wavelengths(line, wave_regions, fit_method='least_squares', user_cfg=region_fit_cfg)

                if verbose:
                        spaxel_spec.display_results(fit_report=True)

            except ValueError as e:
                print(f'--- Line measuring failure at {line} in spaxel {idxY}-{idxX}:\n{e}')

        if verbose:
            spaxel_spec.plot_line_grid(spaxel_spec.log)

        # Convert the measurements log into a HDU and append it to the HDU list unless it is empty
        linesHDU = lime.log_to_HDU(spaxel_spec.log, ext_name=f'{idxY}-{idxX}_LINESLOG', header_dict=hdr_coords)

        # Check the HDU is not empty (no lines measured)
        if linesHDU is not None:
            hdul_log.append(linesHDU)

    # After the regions voxels have been analysed save all the measurements to a .fits file
    hdul_log.writeto(log_address, overwrite=True, output_verify='fix')

print(f'SHOC579 analysis finished with {n_lines} in {n_voxels}')










