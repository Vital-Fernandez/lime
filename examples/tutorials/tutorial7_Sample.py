import lime
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS

# State the data location
cfg_file = '../sample_data/manga.toml'
cube_file = Path('../sample_data/manga-8626-12704-LOGCUBE.fits.gz')
spatial_mask_file = Path('../sample_data/SHOC579_mask.fits')
output_sample_file = Path('../sample_data/SHOC579_masked_spectra.fits')

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

# World coordinate system from the observation
wcs = WCS(hdr)

# Define a LiMe cube object
shoc579 = lime.Cube(wave, flux_cube, redshift=z_obj, norm_flux=norm_flux, wcs=wcs)

shoc579.export_spaxels(output_sample_file, spatial_mask_file)

print(fits.info(output_sample_file))