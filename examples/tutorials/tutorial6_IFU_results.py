import lime
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib import pyplot as plt

# State the data location
cfg_file = '../sample_data/manga.toml'
cube_file = Path('../sample_data/manga-8626-12704-LOGCUBE.fits.gz')
bands_file_0 = Path('../sample_data/SHOC579_MASK0_bands.txt')
spatial_mask_file = Path('../sample_data/SHOC579_mask.fits')
output_lines_log_file = Path('../sample_data/SHOC579_log.fits')

bands_df = lime.line_bands()

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

# WCS from the obsevation header
wcs = WCS(hdr)

# Define a LiMe cube object
shoc579 = lime.Cube(wave, flux_cube, redshift=z_obj, norm_flux=norm_flux, wcs=wcs)
shoc579.check.cube('H1_6563A', lines_log_file=output_lines_log_file, masks_file=spatial_mask_file)

# Check the individual spaxel fitting configuration
spaxel = shoc579.get_spectrum(38, 35)
spaxel.load_log(output_lines_log_file, page='38-35_LINELOG')
spaxel.plot.grid()

# Export the measurements log as maps:
param_list = ['intg_flux', 'intg_flux_err', 'gauss_flux', 'gauss_flux_err', 'v_r', 'v_r_err']
lines_list = ['H1_4861A', 'H1_6563A', 'O3_4363A', 'O3_4959A', 'O3_5007A', 'S3_6312A', 'S3_9069A', 'S3_9531A']
lime.save_parameter_maps(output_lines_log_file, '../sample_data/', param_list, lines_list,
                         mask_file=spatial_mask_file, output_file_prefix='SHOC579_', wcs=wcs)

# State line ratios for the plots
lines_ratio = {'H1': ['H1_6563A', 'H1_4861A'],
               'O3': ['O3_5007A', 'O3_4959A'],
               'S3': ['S3_9531A', 'S3_9069A']}

# State the parameter map file
fits_file = f'../sample_data/SHOC579_gauss_flux.fits'

# Loop through the line ratios
for ion, lines in lines_ratio.items():

    # Recover the parameter measurements
    latex_array = lime.label_decomposition(lines, params_list=['latex_label'])[0]
    ratio_map = fits.getdata(fits_file, lines[0]) / fits.getdata(fits_file, lines[1])

    # Get the astronomical coordinates from one of the headers of the lines log
    hdr = fits.getheader(fits_file, lines[0])
    wcs_maps = WCS(hdr)

    # Create the plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection=wcs_maps, slices=('x', 'y'))
    im = ax.imshow(ratio_map)
    cbar = fig.colorbar(im, ax=ax)
    ax.update({'title': f'SHOC579 flux ratio: {latex_array[0]} / {latex_array[1]}', 'xlabel': r'RA', 'ylabel': r'DEC'})
    plt.show()

