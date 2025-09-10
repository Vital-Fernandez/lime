import lime
import urllib.request
from pathlib import Path


# MANGA cube web link and save file location
cube_url = 'https://data.sdss.org/sas/dr17/manga/spectro/redux/v3_1_1/8626/stack/manga-8626-12704-LOGCUBE.fits.gz'

# Data location
cube_address = Path('../0_resources/spectra/manga-8626-12704-LOGCUBE.fits.gz')
cfgFile = '../0_resources/manga.toml'

# Download the cube file if not available (this may take some time)
if not cube_address.is_file():
    urllib.request.urlretrieve(cube_url, cube_address)
    print(' Download completed!')
else:
    print('Observation found in folder')

# Load the configuration file:
obs_cfg = lime.load_cfg(cfgFile)

# Observation properties
z_obj = obs_cfg['SHOC579']['redshift']
norm_flux = obs_cfg['SHOC579']['norm_flux']

# Define a LiMe cube object
shoc579 = lime.Cube.from_file(cube_address, instrument='manga', redshift=z_obj)
shoc579.plot.cube('H1_6563A', line_fg='O3_4363A')

# Check the spaxels interactively
shoc579.check.cube('H1_6563A', line_fg='H1_6563A', min_pctl_bg=70, cont_pctls_fg=[80, 90, 95, 99])

# Line continuum mask
spatial_mask_SN_cont = '../0_resources/results/SHOC579_mask_SN_cont.fits'
shoc579.spatial_masking('O3_4363A', param='SN_cont', contour_pctls=[93, 96, 99], output_address=spatial_mask_SN_cont)
shoc579.plot.cube('H1_6563A', masks_file=spatial_mask_SN_cont)

# Line emission mask
spatial_mask_SN_line = '../0_resources/results/SHOC579_mask_SN_line.fits'
shoc579.spatial_masking('O3_4363A', param='SN_line', contour_pctls=[93, 96, 99], output_address=spatial_mask_SN_line)
shoc579.plot.cube('H1_6563A', masks_file=spatial_mask_SN_line)

# Manually add/remove spaxels to the spatial mask
shoc579.check.cube('H1_6563A', masks_file=spatial_mask_SN_line, maintain_y_zoom=True)