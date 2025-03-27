import numpy as np
import lime
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib import pyplot as plt

# State the data location
cfg_file = '../sample_data/manga.toml'
cube_file = Path('../sample_data/spectra/manga-8626-12704-LOGCUBE.fits.gz')
bands_file_0 = Path('../sample_data/bands/SHOC579_MASK0_bands.txt')
spatial_mask_file = Path('../sample_data/SHOC579_mask.fits')
output_lines_log_file = Path('../sample_data/SHOC579_log.fits')

# Load the configuration file:
obs_cfg = lime.load_cfg(cfg_file)

# Observation properties
z_obj = obs_cfg['SHOC579']['redshift']

# Load the Cube
shoc579 = lime.Cube.from_file(cube_file, instrument='manga', redshift=z_obj)
shoc579.check.cube('H1_6563A', lines_file=output_lines_log_file, rest_frame=True)

# Check the individual spaxel fitting configuration
spaxel = shoc579.get_spectrum(38, 35)
spaxel.load_frame(output_lines_log_file, page='38-35_LINELOG')
spaxel.plot.grid()

# Export the line measurements as spatial maps:
param_list = ['intg_flux', 'intg_flux_err', 'profile_flux', 'profile_flux_err', 'v_r', 'v_r_err']
lines_list = ['H1_4861A', 'H1_6563A', 'O3_4363A', 'O3_4959A', 'O3_5007A', 'S3_6312A', 'S3_9069A', 'S3_9531A']
lime.save_parameter_maps(output_lines_log_file, '../sample_data/', param_list, lines_list,
                         mask_file=spatial_mask_file, output_file_prefix='SHOC579_', wcs=shoc579.wcs)

# State line ratios for the plots
lines_ratio = {'H1': ['H1_6563A', 'H1_4861A'],
               'O3': ['O3_5007A', 'O3_4959A'],
               'S3': ['S3_9531A', 'S3_9069A']}

# State the parameter map file
fits_file = f'../sample_data/SHOC579_profile_flux.fits'

# Loop through the line ratios
for ion, lines in lines_ratio.items():

    # Recover the parameter measurements
    latex_array = lime.label_decomposition(lines, params_list=['latex_label'])[0]
    ratio_map = fits.getdata(fits_file, lines[0]) / fits.getdata(fits_file, lines[1])
    Halpha = fits.getdata(fits_file, lines[0])
    Hbeta = fits.getdata(fits_file, lines[1])

    # Get the astronomical coordinates from one of the headers of the lines log
    hdr = fits.getheader(fits_file, lines[0])
    wcs_maps = WCS(hdr)

    # Create the plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection=wcs_maps, slices=('x', 'y'))
    im = ax.imshow(ratio_map, vmin=np.nanpercentile(ratio_map, 16), vmax=np.nanpercentile(ratio_map, 84))
    cbar = fig.colorbar(im, ax=ax)
    ax.update({'title': f'SHOC579 flux ratio: {latex_array[0]} / {latex_array[1]}', 'xlabel': r'RA', 'ylabel': r'DEC'})
    plt.show()

