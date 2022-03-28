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

# Open the cube fits file
with fits.open(SHOC579_cube_address) as hdul:
    wave = hdul['WAVE'].data
    flux_cube = hdul['FLUX'].data * 1e-17
    hdr = hdul['FLUX'].header

# Load the configuration file:
cfgFile = './sample_data/config_file.cfg'
obs_cfg = lime.load_cfg(cfgFile)
z_SHOC579 = obs_cfg['SHOC579_data']['redshift']

# and the masks file:
mask_file = './sample_data/SHOC579_region0_maskLog.txt'
mask_log = lime.load_lines_log(mask_file)

# Establish the band image for the plot bacground using Halpha
w3_Halpha, w4_Halpha = mask_log.loc['H1_6563A_b', 'w3':'w4'].values * (1 + z_SHOC579)
idcs_Halpha = np.searchsorted(wave, (w3_Halpha, w4_Halpha))
Halpha_image = flux_cube[idcs_Halpha[0]:idcs_Halpha[1], :, :].sum(axis=0)

# Use SII lines as the foreground image contours
w3_SII = mask_log.loc['S2_6716A', 'w3'] * (1 + z_SHOC579)
w4_SII = mask_log.loc['S2_6731A', 'w4'] * (1 + z_SHOC579)
idcs_SII = np.searchsorted(wave, (w3_SII, w4_SII))
SII_image = flux_cube[idcs_SII[0]:idcs_SII[1], :, :].sum(axis=0)

# Establishing the contours intensity using percentiles
percentile_array = np.array([80, 90, 95, 99, 99.9])
SII_contourLevels = np.nanpercentile(SII_image, percentile_array)

# Labels for the axes
ax_conf = {'image': {'xlabel': r'RA', 'ylabel': r'DEC', 'title': f'MANGA SHOC579'}}

# Color normalization for the flux band:
min_flux = np.nanpercentile(Halpha_image, 60)
log_norm_bg = colors.SymLogNorm(linthresh=min_flux, vmin=min_flux, base=10)

# Interactive plotter for IFU data cubes
lime.CubeInspector(wave, flux_cube, Halpha_image, SII_image, SII_contourLevels,
                   fits_header=hdr, ax_cfg=ax_conf, color_norm=log_norm_bg)

# Create a dictionary with the coordinate entries for the header
hdr_coords = {}
for key in lime.COORD_ENTRIES:
    if key in hdr:
        hdr_coords[key] = hdr[key]

# Run the task
output_mask_file = './sample_data/SHOC579_mask.fits'
lime.spatial_mask_generator('flux', wave, flux_cube, percentile_array, signal_band=(w3_SII, w4_SII),
                            mask_ref='S2_6716A_b', output_address=output_mask_file,
                            fits_header=hdr_coords, show_plot=True)
#
# w1_SII, w2_SII = mask_log.loc['S2_6716A', 'w1':'w2'].values * (1 + z_SHOC579)
# output_mask_file = './sample_data/SHOC579_mask.fits'
# lime.spatial_mask_generator('SN_cont', wave, flux_cube, percentile_array, signal_band=(w1_SII, w2_SII),
#                             mask_ref='S2_6716A_b', output_address=output_mask_file,
#                             fits_header=hdr_coords, show_plot=True)
#
# w1_SII, w2_SII = mask_log.loc['S2_6716A', 'w1':'w2'].values * (1 + z_SHOC579)
# output_mask_file = './sample_data/SHOC579_mask.fits'
# lime.spatial_mask_generator('SN_line', wave, flux_cube, percentile_array, signal_band=(w3_SII, w4_SII), cont_band=(w1_SII, w2_SII),
#                             mask_ref='S2_6716A_b', output_address=output_mask_file,
#                             fits_header=hdr_coords, show_plot=True)

# Parameters for the new masks
coord_lower_limit = 22
mask_list = ['S2_6716A_B_MASK_1', 'S2_6716A_B_MASK_2', 'S2_6716A_B_MASK_3']

# New HDUs for the modified mask
hdul_new = fits.HDUList([fits.PrimaryHDU()])

# Open the original mask file, loop through the target masks and set voxels below row 22 outside the mask (False)
with fits.open(output_mask_file) as hdul:
    for i, mask_ext in enumerate(mask_list):
        mask_frame = hdul[mask_ext].data.astype('bool')
        mask_frame[:coord_lower_limit, :] = False
        hdul_new.append(fits.ImageHDU(name=f'S2_6716A_B_MASK_{i}', data=mask_frame.astype(int),
                                  ver=1, header=fits.Header(hdr_coords)))

# Save the modified mask
hdul_new.writeto(output_mask_file, overwrite=True)

# Load one of the masks headers to get the WCS for the plot
masks_hdr = fits.getheader(output_mask_file, extname='S2_6716A_B_MASK_0')

# Plot the Halpha image of the cube with the new masks
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(projection=WCS(masks_hdr), slices=('x', 'y'))
im = ax.imshow(Halpha_image, cmap=cm.gray, norm=log_norm_bg)

# Color map for the contours
cmap_contours = cm.get_cmap('viridis', len(mask_list))

legend_list = [None] * len(mask_list)

# Open the mask .fits file and plot the masks as numpy masked array
with fits.open(output_mask_file) as hdul:
    for i, HDU in enumerate(hdul):
        if i > 0:
            mask_name, mask_frame = HDU.name, HDU.data
            mask_frame = hdul[i].data.astype('bool')
            n_voxels = np.sum(mask_frame)

            inv_masked_array = np.ma.masked_where(~mask_frame, Halpha_image)

            cm_i = colors.ListedColormap([cmap_contours(i - 1)])
            ax.imshow(inv_masked_array, cmap=cm_i, vmin=0, vmax=1, alpha=0.3)
            legend_list[i - 1] = patches.Patch(color=cm_i(i - 1), label=f'{mask_name} ({n_voxels} spaxels)')

# Define the legend for the imshow plot
ax.legend(handles=legend_list, bbox_to_anchor=(1.025, 1), loc=2, borderaxespad=0.)
ax.update({'title': r'SHOC579 Mask regions', 'xlabel': r'RA', 'ylabel': r'DEC'})
plt.tight_layout()
plt.show()

# You can adjust the spaxels manually including the mask file in the CubeInspector class
lime.CubeInspector(wave, flux_cube, Halpha_image, SII_image, SII_contourLevels,
                   fits_header=hdr, ax_cfg=ax_conf, color_norm=log_norm_bg,
                   mask_file=output_mask_file)
