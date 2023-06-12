import subprocess
import os

os.chdir(f'../../../')

def launch_jupyter_notebook():
    subprocess.run(["jupyter", "notebook"])

if __name__ == "__main__":
    launch_jupyter_notebook()

# import lime
# from pathlib import Path
#
# conf_file_toml = Path(f'../sample_data/manga.toml')
#
# cfg_ini = lime.load_cfg(conf_file_ini)
# cfg_toml = lime.load_cfg(conf_file_toml)
#
# print(cfg_ini)
# print(cfg_toml)


import numpy as np
from astropy.io import fits
from pathlib import Path
from astropy.wcs import WCS
import lime

# fits_path = Path(f'../sample_data/SHOC579_SDSS_dr18.fits')
#
# # Open the fits file
# extension = 2
# with fits.open(fits_path) as hdul:
#     data = hdul[extension].data
#     header = hdul[extension].header
#
# flux_array = data['flux'] * 1e-17
#
# ivar_array = data['ivar']
# pixel_mask = ivar_array == 0
#
# wave_vac_array = np.power(10, data['loglam'])
#
spectra_path = Path('../../sample_data')
fits_path = spectra_path/'manga-8626-12704-LOGCUBE.fits.gz'

# Open the MANGA cube fits file
with fits.open(fits_path) as hdul:

    # Wavelength 1D array
    wave = hdul['WAVE'].data

    # Flux 3D array
    flux_cube = hdul['FLUX'].data * 1e-17

    # Convert inverse variance cube to standard error masking 0-value pixels first
    ivar_cube = hdul['IVAR'].data
    pixel_mask_cube = ivar_cube == 0
    pixel_mask_cube = pixel_mask_cube.reshape(ivar_cube.shape)
    err_cube = np.sqrt(1/np.ma.masked_array(ivar_cube, pixel_mask_cube)) * 1e-17

    # Header
    hdr = hdul['FLUX'].header


wcs = WCS(hdr)

cube = lime.Cube(wave, flux_cube, err_cube, redshift=0.0475, norm_flux=1e-17, pixel_mask=pixel_mask_cube, wcs=wcs)
spaxel = cube.get_spaxel(38, 35)
spaxel.plot.spectrum(rest_frame=True)
spaxel.fit.band('S3_6312A')
spaxel.plot.band()
spaxel.con