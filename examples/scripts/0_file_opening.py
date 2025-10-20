from astropy.io import fits
from pathlib import Path
import lime

fname = '/home/vital/Astrodata/J0823_p2806.fits'
fits.info(fname)

cube = lime.Cube.from_file(fname, instrument='kcwi', redshift=0.04711, norm_flux=1)

# Line continuum mask
spatial_mask_SN_line = './ryan_mask_SN_line.fits'
cube.spatial_masking('O3_5007A', param='SN_line', contour_pctls=[80, 90, 99], fname=spatial_mask_SN_line)
cube.plot.cube('O3_5007A', masks_file=spatial_mask_SN_line)

cube.check.cube('O3_5007A', masks_file=spatial_mask_SN_line, rest_frame=True)

# Specify the data location the observations
data_folder = Path('../doc_notebooks/0_resources/')
fits_file = data_folder/'spectra'/'MRK209_cos_x1dsum.fits'

spec = lime.Spectrum.from_file(fits_file, instrument='COS', z=0)
spec.plot.spectrum()

