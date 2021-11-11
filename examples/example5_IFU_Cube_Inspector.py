import lime
import numpy as np
from astropy.io import fits

# Input the data files
obsFitsFile = './sample_data/manga-8626-12704-LOGCUBE.fits'

# Open the cube fits file
with fits.open(obsFitsFile) as hdul:
    hdr = hdul['FLUX'].header
    wave = hdul['WAVE'].data
    flux = hdul['FLUX'].data

# Establish the line intervals for the cube flux image slices
line_regions = {'H1_6563A': np.array([6850, 6910]),
                'S2_6731A_b': np.array([7027.5, 7057.5]),
                'O3_4363A': np.array([4565.0, 4575.0]),
                'S3_6312A': np.array([6606.5, 6617.0]),
                'O3_5007A': np.array([5232.0, 5260.0]),
                'S3_9069A': np.array([9492.5, 9506.5]),
                'S3_9531A': np.array([9975.5, 9995.0])}

# Use Halpha for the background
idcs_Halpha = np.searchsorted(wave, line_regions['H1_6563A'])
Halpha_image = flux[idcs_Halpha[0]:idcs_Halpha[1], :, :].sum(axis=0)

# Use SII lines for the foreground contours
idcs_line = np.searchsorted(wave, line_regions['S2_6731A_b'])
SII_image = flux[idcs_line[0]:idcs_line[1], :, :].sum(axis=0)

# Establishing the countours by percentiles
percentil_array = np.array([70, 80, 90, 95, 99, 99.9])
SII_contourLevels = np.nanpercentile(SII_image, percentil_array)

# Labels for the axes
ax_conf = {'image': {'xlabel': r'RA', 'ylabel': r'DEC', 'title': f'MANGA SHOC579'}}

lime.CubeInspector(wave, flux, Halpha_image, SII_image, SII_contourLevels,
                  header=hdr, axes_conf=ax_conf)

