import numpy as np
from astropy.io import fits
from pathlib import Path
from astropy.wcs import WCS
import lime
import astropy

# Check the instruments with direct file reading support
lime.show_instrument_cfg()

# Data location
data_folder = Path('../doc_notebooks/0_resources/spectra')
sloan_SHOC579 = data_folder/'sdss_dr18_0358-51818-0504.fits'

# Direct file reading
shoc579 = lime.Spectrum.from_file(sloan_SHOC579, instrument='sdss', redshift=0.0475)
shoc579.plot.spectrum(rest_frame=True)

# Loading file using astropy
extension = 1
with fits.open(sloan_SHOC579) as hdul:
    data = hdul[extension].data
    header = hdul[extension].header

flux_array = data['flux']
wave_vac_array = np.power(10, data['loglam'])

units_flux = '1e-17*FLAM'

ivar_array = data['ivar']
err_array = np.sqrt(1/ivar_array)
pixel_mask = ivar_array == 0

shoc579 = lime.Spectrum(wave_vac_array, flux_array, err_array, pixel_mask=pixel_mask, units_flux=units_flux, redshift=0.0475)
shoc579.plot.spectrum(rest_frame=True)