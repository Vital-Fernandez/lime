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
    flux = hdul['FLUX'].data
    hdr = hdul['FLUX'].header

# Load the configuration file:
cfgFile = './sample_data/config_file.cfg'
obs_cfg = lime.load_cfg(cfgFile)
z_SHOC579 = obs_cfg['SHOC579_data']['redshift']

# and the masks file:
mask_file = './sample_data/osiris_mask.txt'
mask_log = lime.load_lines_log(mask_file)

# Establish the band image for the plot bacground using Halpha
Halpha_band = mask_log.loc['H1_6563A_b', 'w3':'w4'].values * (1 + z_SHOC579)
idcs_Halpha = np.searchsorted(wave, Halpha_band)
Halpha_image = flux[idcs_Halpha[0]:idcs_Halpha[1], :, :].sum(axis=0)

# Use SII lines as the foreground image contours
SII_band = mask_log.loc['S2_6716A_b', 'w3':'w4'].values * (1 + z_SHOC579)
idcs_SII = np.searchsorted(wave, SII_band)
SII_image = flux[idcs_SII[0]:idcs_SII[1], :, :].sum(axis=0)

# Establishing the contours intensity using percentiles
percentile_array = np.array([80, 90, 95, 99, 99.9])
SII_contourLevels = np.nanpercentile(SII_image, percentile_array)

# Labels for the axes
ax_conf = {'image': {'xlabel': r'RA', 'ylabel': r'DEC', 'title': f'MANGA SHOC579'}}

# Color normalization for the flux band:
min_flux = np.nanpercentile(Halpha_image, 60)
log_norm_bg = colors.SymLogNorm(linthresh=min_flux, vmin=min_flux, base=10)

# Interactive plotter for IFU data cubes
lime.CubeFitsInspector(wave, flux, Halpha_image, SII_image, SII_contourLevels,
                       fits_header=hdr, axes_conf=ax_conf, color_norm=log_norm_bg)

# Plot the log results as maps
lime.save_param_maps(fitsLog_address, param_images, objFolder, maskFits_address, ext_mask=masks, ext_log='_LINELOG',
                     page_hdr=plot_dict)

# fits_file = Path(objFolder) / f'{param}.fits'
# with fits.open(fits_file):
#     for line in user_lines:
#         param_image = fits.getdata(fits_file, line)
#         param_hdr = fits.getheader(fits_file, line)
#
#         fig = plt.figure(figsize=(10, 10))
#         ax = fig.add_subplot(projection=WCS(fits.Header(param_hdr)), slices=('x', 'y'))
#         im = ax.imshow(param_image)
#         ax.update({'title': f'Galaxy {obj}: {param}-{line}', 'xlabel': r'RA', 'ylabel': r'DEC'})
#         plt.show()

# for i, obj in enumerate(objList):
#
#     # Data location
#     cube_address = fitsFolder/fileList[i]
#     objFolder = resultsFolder/obj
#     db_address = objFolder / f'{obj}_database.fits'
#     maskFits_address = objFolder/f'{obj}_masks.fits'
#
#     parameter_fits = objFolder/'intg_flux.fits'
#     O3_4959A = fits.getdata(parameter_fits, 'O3_4959A')
#     O3_5007A = fits.getdata(parameter_fits, 'O3_5007A')
#     hdr_plot = fits.getheader(parameter_fits, 'O3_5007A')
#
#     ion_array, wave_array, latex_array = lime.label_decomposition(['O3_4959A', 'O3_5007A'])
#
#     # coeff_im = O3_5007A/O3_4959A
#     #
#     # divnorm = colors.TwoSlopeNorm(vmin=2.0,
#     #                               vcenter=2.984,
#     #                               vmax=4.0)
#     # cbar_label = f'Line ratio, theoretical value ({2.984}) white'
#
#     coeff_im = (O3_5007A/O3_4959A - 2.984) * 100
#
#     divnorm = colors.TwoSlopeNorm(vmin=-75.0,
#                                   vcenter=0,
#                                   vmax=75.0)
#
#     cbar_label = f'Line ratio discrepancy %'
#
#     fig = plt.figure(figsize=(10, 10))
#     ax = fig.add_subplot(projection=WCS(hdr_plot), slices=('x', 'y'))
#     im = ax.imshow(coeff_im, cmap='RdBu', norm=divnorm)
#     cbar = fig.colorbar(im, ax=ax)
#     cbar.set_label(cbar_label, rotation=270, labelpad=50, fontsize=15)
#     ratio_label = r'$\frac{{{}}}{{{}}}$'.format(latex_array[1].replace('$', ''), latex_array[0].replace('$', ''))
#     ax.update({'title': r'Galaxy {}: {}'.format(obj, ratio_label), 'xlabel': r'RA', 'ylabel': r'DEC'})
#     ax.set_xlim(95, 205)
#     ax.set_ylim(75, 225)
#     # plt.show()
#     plt.savefig(objFolder/'line_ratios'/f'map_{obj}_OIII_ratio.png')